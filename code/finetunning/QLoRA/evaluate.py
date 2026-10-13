"""Select adapters on dev, evaluate held-out sets, or predict unlabeled QA records."""

import argparse
import time

from common import (
    PROMPT_VERSION, TEST_FILE, check_overlap, file_hash, input_hash, json_text,
    load_checkpoint, load_model, local_path, parse_prediction, prompt_tokens,
    read_json, read_rows, relative, score, signatures, validate_references, write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run folder inside code/results/qlora")
    parser.add_argument("--split", choices=("dev", "llm_test", "human_test", "predict"), required=True)
    parser.add_argument("--select-best", action="store_true", help="Evaluate all checkpoints on dev only")
    parser.add_argument("--input", help="Optional labeled input override; required for predict")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--context-length", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.select_best and args.split != "dev":
        parser.error("Checkpoint selection is allowed only on dev")
    if args.split == "predict" and not args.input:
        parser.error("--split predict requires --input JSONL with uid/question/answer")
    if args.max_tokens < 1 or args.context_length <= args.max_tokens:
        parser.error("Require 0 < --max-tokens < --context-length")
    return args


def rank_metrics(item):
    metrics = item["metrics"]
    return (metrics["macro_f1"], metrics["accuracy"], -metrics["invalid_count"], -item["step"])


def generate_rows(model, tokenizer, rows, variant, max_tokens, context_length, output):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    model.eval()
    predictions = []
    output = local_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            prompt = prompt_tokens(tokenizer, row, variant)
            if len(prompt) >= context_length:
                raise ValueError(f"UID {row['uid']}: input needs {len(prompt)} tokens; increase --context-length. No truncation")
            budget = min(max_tokens, context_length - len(prompt))
            sample_start = time.monotonic()
            raw, finish_reason, generated_tokens = "", None, 0
            for response in stream_generate(
                model, tokenizer, prompt, max_tokens=budget, sampler=make_sampler(temp=0.0),
            ):
                raw += response.text
                finish_reason = response.finish_reason
                generated_tokens = response.generation_tokens
            value, error, schema_error = None, None, None
            try:
                value, schema_error = parse_prediction(raw, row["answer"], variant)
            except (ValueError, TypeError, KeyError) as problem:
                error = str(problem)
            prediction = {
                "uid": row["uid"], "question": row["question"], "answer": row["answer"],
                "gold_label": row.get("overall_label"),
                "predicted_label": value["overall_label"] if value is not None else None,
                "prediction": value, "raw_output": raw, "parse_error": error,
                "schema_error": schema_error, "finish_reason": finish_reason,
                "prompt_tokens": len(prompt), "generated_tokens": generated_tokens,
                "generation_budget": budget, "seconds": time.monotonic() - sample_start,
            }
            predictions.append(prediction)
            handle.write(json_text(prediction) + "\n")
            handle.flush()
            print(f"{index}/{len(rows)}: {row['uid']} -> {prediction['predicted_label']}", flush=True)
    elapsed = time.monotonic() - started
    return predictions, elapsed


def metrics_for(rows, predictions, elapsed):
    gold = [row["overall_label"] for row in rows]
    result = score(gold, [item["predicted_label"] for item in predictions])
    strict = [item["predicted_label"] if not item["schema_error"] else None for item in predictions]
    result.update({
        "strict_schema_accuracy": score(gold, strict)["accuracy"],
        "schema_valid_count": sum(p["prediction"] is not None and p["schema_error"] is None for p in predictions),
        "length_limited_count": sum(p["finish_reason"] == "length" for p in predictions),
        "generation_seconds": elapsed,
        "metric_policy": "All records count. Missing/unparseable overall labels count as errors. A valid overall label and reason can be scored even if sub-question schema is invalid; strict_schema_accuracy also requires the full output schema.",
    })
    return result


def evaluate(args):
    run = local_path(args.run)
    if local_path("results/qlora") not in run.parents:
        raise ValueError("--run must be inside code/results/qlora")
    config = read_json(run / "training.json")
    if config["state"] != "trained":
        raise ValueError("Training is not complete; cannot select/evaluate this run")
    if config["prompt_version"] != PROMPT_VERSION:
        raise ValueError("Prompt version differs from training; use the matching code")
    if file_hash(local_path(config["model"]) / "config.json") != config["model_config_sha256"]:
        raise ValueError("Base model configuration changed since training")
    if args.split == "predict":
        rows = read_rows(args.input)
    else:
        info = config["data"][args.split]
        input_path = args.input or (TEST_FILE if args.split == "llm_test" else info["path"])
        rows = read_rows(input_path, info["count"])
        validate_references(rows)
        if input_hash(rows) != info["inputs_sha256"]:
            raise ValueError("Evaluation UID/question/answer set differs from the split reserved before training")
        if args.split == "dev" and file_hash(input_path) != info["sha256"]:
            raise ValueError("Dev data changed since training; rerun the experiment with a fixed dev set")
        check_overlap(config["data"]["train"]["signatures"], signatures(rows), f"train/{args.split}")
        if args.split != "dev":
            check_overlap(config["data"]["dev"]["signatures"], signatures(rows), f"dev/{args.split}")
    generation = {"max_tokens": args.max_tokens, "context_length": args.context_length, "temperature": 0.0}
    if args.select_best:
        checkpoints = config["checkpoints"]
    else:
        selection = read_json(run / "selection.json")
        if selection["run_id"] != config["run_id"]:
            raise ValueError("Stale checkpoint selection from a different training run")
        if selection["generation"] != generation:
            raise ValueError("Use the same --max-tokens and --context-length as dev checkpoint selection")
        checkpoints = [selection["best"]]
    for checkpoint in checkpoints:
        path = local_path(run / checkpoint["file"])
        if path.parent != run or file_hash(path) != checkpoint["sha256"]:
            raise ValueError(f"Checkpoint missing, changed or outside its run: {path}")
    model, tokenizer = load_model(config["model"], config["adapter_config"], config["arguments"]["seed"])
    # Validate every prompt before replacing a prior result file.
    for row in rows:
        length = len(prompt_tokens(tokenizer, row, config["variant"]))
        if length >= args.context_length:
            raise ValueError(f"UID {row['uid']}: input has {length} tokens; increase --context-length")
    reports = []
    for checkpoint in checkpoints:
        load_checkpoint(model, run / checkpoint["file"])
        stem = f"dev_{checkpoint['step']:07d}" if args.select_best else args.split
        predictions, elapsed = generate_rows(
            model, tokenizer, rows, config["variant"], args.max_tokens,
            args.context_length, run / f"{stem}_predictions.jsonl",
        )
        if args.split == "predict":
            print(f"Saved {run / (stem + '_predictions.jsonl')}")
            return
        metrics = metrics_for(rows, predictions, elapsed)
        report = {**checkpoint, "run_id": config["run_id"], "variant": config["variant"],
                  "split": args.split, "generation": generation, "metrics": metrics,
                  "reference_path": relative(input_path),
                  "reference_sha256": file_hash(input_path) if local_path(input_path).is_file() else {
                      relative(p): file_hash(p) for p in sorted(local_path(input_path).glob("batch_*.jsonl"))},
                  "label_source": "human" if args.split == "human_test" else "LLM (human verification must be documented separately)"}
        write_json(run / f"{stem}_metrics.json", report)
        reports.append(report)
        print(f"{stem}: accuracy={metrics['accuracy']:.4f}, macro-F1={metrics['macro_f1']:.4f}, invalid={metrics['invalid_count']}", flush=True)
    if args.select_best:
        best = max(reports, key=rank_metrics)
        write_json(run / "selection.json", {
            "run_id": config["run_id"], "selection_split": "dev", "generation": generation,
            "criterion": "macro-F1, accuracy, fewer invalid outputs, earlier step",
            "best": {key: best[key] for key in ("step", "file", "sha256")},
            "dev_results": reports,
        })
        print(f"Selected step {best['step']} on dev; test labels were not used.")


if __name__ == "__main__":
    try:
        evaluate(parse_args())
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error))
