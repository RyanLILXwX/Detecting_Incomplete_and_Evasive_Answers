"""Train whole-question or joint-decomposition Qwen3 adapters on Apple Silicon."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from importlib.metadata import version
import time
import uuid

from common import (
    CODE_DIR, DEV_FILE, HUMAN_DIR, MODEL_DIR, PROMPT_VERSION, TEST_INPUT, TRAIN_FILE,
    check_model, check_overlap, encode_training, file_hash, input_hash, json_text, load_model,
    local_path, read_rows, relative, setup_cache, signatures, validate_labels, validate_references, write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("whole", "joint"), default="joint")
    parser.add_argument("--train", default=TRAIN_FILE)
    parser.add_argument("--dev", default=DEV_FILE)
    parser.add_argument("--llm-test-reserve", default=TEST_INPUT)
    parser.add_argument("--human-test-reserve", default=HUMAN_DIR)
    parser.add_argument("--train-count", type=int, default=2700)
    parser.add_argument("--dev-count", type=int, default=300)
    parser.add_argument("--llm-test-count", type=int, default=300)
    parser.add_argument("--human-test-count", type=int, default=120)
    parser.add_argument("--model", default=MODEL_DIR)
    parser.add_argument("--output", help="Default: results/qlora/qwen3_4b_<variant>")
    parser.add_argument("--iters", type=int, default=2700, help="Microbatches, not optimizer updates")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=900)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-layers", type=int, default=8, help="Last N transformer layers")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--scale", type=float, default=16.0, help="MLX LoRA scale, not PEFT alpha")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true", help="Validate data without importing MLX or training")
    parser.add_argument("--overwrite", action="store_true", help="Replace this run's generated files; do not resume")
    args = parser.parse_args(argv)
    positive = ("train_count", "dev_count", "llm_test_count", "human_test_count", "iters",
                "batch_size", "grad_accumulation", "save_every", "max_seq_length", "lora_layers", "rank")
    if any(getattr(args, key) <= 0 for key in positive):
        parser.error("Counts, sequence length, rank, layers and step arguments must be positive")
    if args.iters % args.grad_accumulation or args.save_every % args.grad_accumulation:
        parser.error("--iters and --save-every must be divisible by --grad-accumulation")
    if not args.learning_rate > 0 or not args.scale > 0 or not 0 <= args.dropout < 1:
        parser.error("Require positive learning rate/scale and 0 <= dropout < 1")
    if args.train_count % args.batch_size:
        parser.error("--train-count must be divisible by --batch-size to avoid dropping records")
    return args


def prepare_data(args):
    specifications = {
        "train": (args.train, args.train_count), "dev": (args.dev, args.dev_count),
        "llm_test": (args.llm_test_reserve, args.llm_test_count),
        "human_test": (args.human_test_reserve, args.human_test_count),
    }
    data, errors = {}, []
    for name, (path, count) in specifications.items():
        try:
            data[name] = read_rows(path, count)
            if name == "train":
                validate_labels(data[name], args.variant)
            elif name == "dev":
                validate_references(data[name])
        except (ValueError, OSError) as error:
            errors.append(f"{name}: {error}")
    fingerprints = {name: signatures(rows) for name, rows in data.items()}
    for left, right in combinations(fingerprints, 2):
        try:
            check_overlap(fingerprints[left], fingerprints[right], f"{left}/{right}")
        except ValueError as error:
            errors.append(str(error))
    if errors:
        raise ValueError("Data checks failed:\n" + "\n".join(errors))
    info = {
        name: {"path": relative(specifications[name][0]), "count": len(rows),
               "signatures": fingerprints[name], "inputs_sha256": input_hash(rows)}
        for name, rows in data.items()
    }
    for name in ("train", "dev"):
        info[name]["sha256"] = file_hash(specifications[name][0])
        info[name]["labels"] = dict(Counter(row["overall_label"] for row in data[name]))
    return data, info


def assistant_loss(model, batch, lengths):
    import mlx.core as mx
    import mlx.nn as nn

    logits = model(batch[:, :-1])
    positions = mx.arange(1, batch.shape[1])
    mask = (positions >= lengths[:, 0:1]) & (positions < lengths[:, 1:])
    losses = nn.losses.cross_entropy(logits, batch[:, 1:]).astype(mx.float32)
    count = mask.sum()
    return (losses * mask).sum() / count, count


def train(args):
    data, data_info = prepare_data(args)
    print(json_text({"status": "data checks passed", "variant": args.variant,
                     "counts": {k: v["count"] for k, v in data_info.items()},
                     "train_labels": data_info["train"]["labels"]}), flush=True)
    if args.check_only:
        return
    setup_cache()
    model_path = local_path(args.model)
    check_model(model_path)
    run = local_path(args.output or f"results/qlora/qwen3_4b_{args.variant}")
    results_root = local_path("results/qlora")
    if results_root not in run.parents:
        raise ValueError("--output must be a run subfolder inside code/results/qlora")
    if run.exists() and any(run.iterdir()):
        if not args.overwrite:
            raise ValueError(f"Run already exists: {run}. Use --overwrite for a fresh run, not a resume")
        if not (run / "training.json").is_file():
            raise ValueError("Refusing to overwrite an unrecognized folder without training.json")
    adapter_config = {"fine_tune_type": "lora", "num_layers": args.lora_layers,
                      "lora_parameters": {"rank": args.rank, "scale": args.scale, "dropout": args.dropout}}
    model, tokenizer = load_model(model_path, adapter_config, seed=args.seed)
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.trainer import TrainingArgs, train as mlx_train

    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    encoded, errors = [], []
    for row in data["train"]:
        try:
            encoded.append(encode_training(tokenizer, row, args.variant, args.max_seq_length))
        except ValueError as error:
            errors.append(str(error))
    if errors:
        raise ValueError(f"{len(errors)} tokenization errors; no rows were dropped:\n" + "\n".join(errors[:8]))
    run.mkdir(parents=True, exist_ok=True)
    for file in run.iterdir():
        local_path(file)
    if args.overwrite:
        for pattern in ("*_adapters.safetensors", "adapters.safetensors", "selection.json", "*_metrics.json", "*_predictions.jsonl"):
            for file in run.glob(pattern):
                local_path(file).unlink()
    write_json(run / "adapter_config.json", adapter_config)
    config = {
        "state": "running", "run_id": uuid.uuid4().hex,
        "started_at": datetime.now(timezone.utc).isoformat(), "variant": args.variant,
        "prompt_version": PROMPT_VERSION, "model": relative(model_path),
        "model_config_sha256": file_hash(model_path / "config.json"),
        "adapter_config": adapter_config, "data": data_info,
        "arguments": {key: (relative(value) if key in ("train", "dev", "llm_test_reserve", "human_test_reserve", "model", "output") and value else value)
                      for key, value in vars(args).items()},
        "versions": {name: version(name) for name in ("mlx", "mlx-lm", "transformers")},
        "trainable_parameters": sum(p.size for _, p in tree_flatten(model.trainable_parameters())),
        "optimizer_updates": args.iters // args.grad_accumulation,
        "parent_examples_processed": args.iters * args.batch_size,
        "parent_epochs": args.iters * args.batch_size / len(encoded),
        "tokens_per_dataset_pass": sum(len(tokens) for tokens, _ in encoded),
        "supervised_tokens_per_dataset_pass": sum(len(tokens) - offset for tokens, offset in encoded),
        "longest_sequence": max(len(tokens) for tokens, _ in encoded),
        "budget_note": "Match parents, microbatches and optimizer updates across variants; target tokens and wall time differ.",
        "checkpoint_selection": "Generate overall labels on dev; maximize macro-F1. No test labels or dev sub-labels used.",
        "loss_mask": "Assistant target positions only: prefix_length <= position < sequence_length; excludes all padding.",
    }
    write_json(run / "training.json", config)
    history = local_path(run / "train_history.jsonl")
    history.write_text("", encoding="utf-8")

    class Callback:
        def on_train_loss_report(self, values):
            with history.open("a", encoding="utf-8") as handle:
                handle.write(json_text(values) + "\n")

    started = time.monotonic()
    try:
        model.train()
        mlx_train(
            model, optim.AdamW(learning_rate=args.learning_rate), encoded, val_dataset=None,
            loss=assistant_loss,
            args=TrainingArgs(
                batch_size=args.batch_size, iters=args.iters, max_seq_length=args.max_seq_length,
                grad_checkpoint=True, grad_accumulation_steps=args.grad_accumulation,
                steps_per_report=min(10, args.iters), steps_per_save=args.save_every,
                adapter_file=str(local_path(run / "adapters.safetensors")),
            ), training_callback=Callback(),
        )
        steps = list(range(args.save_every, args.iters + 1, args.save_every))
        config["checkpoints"] = [{"step": step, "file": f"{step:07d}_adapters.safetensors"} for step in steps]
        if args.iters not in steps:
            config["checkpoints"].append({"step": args.iters, "file": "adapters.safetensors"})
        for checkpoint in config["checkpoints"]:
            checkpoint["sha256"] = file_hash(run / checkpoint["file"])
        config["state"] = "trained"
    except BaseException as error:
        config["state"], config["error"] = "failed", str(error)
        raise
    finally:
        config["training_seconds"] = time.monotonic() - started
        write_json(run / "training.json", config)
    print(f"Saved {run}. Next: evaluate.py --run {relative(run)} --split dev --select-best", flush=True)


if __name__ == "__main__":
    try:
        train(parse_args())
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error))
