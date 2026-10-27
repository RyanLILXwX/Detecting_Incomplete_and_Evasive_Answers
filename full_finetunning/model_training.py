import json
import math
import os
import random
import shutil
import tempfile
from functools import partial
from pathlib import Path
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler

from data_tokenization import collate_batch
from evaluation import evaluate_dataset

def load_training_config():
    """Read the adjacent YAML and preserve the configured path locations."""
    config_path = Path(__file__).with_name("training_config.yaml")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "task_type", "model", "train_file", "validation_file", "test_file", 
        "human_test_file", "output_dir", "seed", "epochs", "micro_batch", 
        "eval_batch", "gradient_accumulation", "optim", "learning_rate", 
        "weight_decay", "lr_scheduler_type", "warmup_ratio", "max_grad_norm", 
        "max_length", "bf16", "gradient_checkpointing", "validation_rounds"
    }
    if (not isinstance(config, dict) or set(config) != required):
        raise ValueError("training_config.yaml must contain exactly the 22 configured fields.")
    if (config["task_type"] not in ("whole_qa", "request")):
        raise ValueError("task_type must be whole_qa or request.")
    if (config["optim"] != "adamw_torch" or config["lr_scheduler_type"] != "cosine"):
        raise ValueError("This pipeline uses optim: adamw_torch and lr_scheduler_type: cosine.")
    integer_fields = ("epochs", "micro_batch", "eval_batch", "gradient_accumulation", "max_length", "validation_rounds")
    for name in integer_fields:
        if (type(config[name]) is not int or config[name] < 1):
            raise ValueError(f"{name} must be a positive integer.")
    if (type(config["seed"]) is not int or not 0 <= config["seed"] < 2 ** 32):
        raise ValueError("seed must be an integer in [0, 2**32).")
    for name in ("learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm"):
        value = config[name]
        if (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f"{name} must be a finite number.")
    if (config["learning_rate"] <= 0 or config["max_grad_norm"] <= 0):
        raise ValueError("learning_rate and max_grad_norm must be positive.")
    if (config["weight_decay"] < 0 or not 0 <= config["warmup_ratio"] < 1):
        raise ValueError("Require weight_decay >= 0 and 0 <= warmup_ratio < 1.")
    for name in ("bf16", "gradient_checkpointing"):
        if (type(config[name]) is not bool):
            raise ValueError(f"{name} must be true or false.")
    path_fields = ("model", "train_file", "validation_file", "test_file", "human_test_file", "output_dir")
    paths = {name: Path(config[name]) for name in path_fields}
    return config, paths

def load_model_and_tokenizer(config: dict, paths: dict, best_checkpoint: bool = False):
    """Load a local Qwen3 classifier, preserving FP32 optimizer parameters."""
    if (not torch.cuda.is_available()):
        raise RuntimeError("This full-parameter training pipeline requires a CUDA GPU.")
    if (config["bf16"] and not torch.cuda.is_bf16_supported()):
        raise RuntimeError("bf16 is enabled but the selected CUDA device does not support it.")
    if (not best_checkpoint):
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
    model_path = paths["output_dir"] / "best_checkpoint" if (best_checkpoint) else paths["model"]
    labels = (("direct", "intermediate", "fully_evasive") if (config["task_type"] == "whole_qa") else ("FULL", "PARTIAL", "NONE"))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if (tokenizer.pad_token_id is None):
        if (tokenizer.eos_token_id is None):
            raise ValueError("The tokenizer needs an existing padding or EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model, loading_info = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=3, 
        id2label=dict(enumerate(labels)), 
        label2id={label: index for index, label in enumerate(labels)}, 
        problem_type="single_label_classification", 
        torch_dtype=torch.float32, 
        attn_implementation="sdpa", 
        local_files_only=True, 
        output_loading_info=True
    )
    if (model.config.model_type != "qwen3" or getattr(model.config, "quantization_config", None)):
        raise ValueError("Use an unquantized Qwen3 base model for full-parameter fine-tuning.")
    allowed_missing = set() if (best_checkpoint) else {"score.weight", "score.bias"}
    if (set(loading_info["missing_keys"]) - allowed_missing):
        raise ValueError("The checkpoint is missing model weights beyond the new classification head.")
    if (config["max_length"] > model.config.max_position_embeddings):
        raise ValueError("max_length exceeds the model's configured position limit.")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    # Gradient checkpointing is enabled only when configured and loading a model for training.
    if (config["gradient_checkpointing"] and not best_checkpoint):
        # The non-reentrant implementation reduces GPU memory use by recomputing intermediate activations during backpropagation.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(torch.device("cuda"))
    if (best_checkpoint):
        model.eval()
    return model, tokenizer

def build_optimizer_scheduler(model, config: dict, total_steps: int):
    """Use AdamW with unregularized biases and normalization parameters."""
    # This list stores references to parameters that receive weight decay.
    decay_parameters = []
    # This list stores references to parameters that do not receive weight decay.
    other_parameters = []
    # Each iteration provides a model parameter and its name.
    for name, parameter in model.named_parameters():
        # Scalar parameters, one-dimensional parameters, and parameters named as biases are excluded from weight decay.
        if (parameter.ndim < 2 or name.endswith(".bias")):
            # This group disables weight decay without changing whether the parameter requires gradients.
            other_parameters.append(parameter)
        else:
            # The remaining parameters are assigned to the weight decay group.
            decay_parameters.append(parameter)
    # The AdamW optimizer is created with separate weight decay settings for the two parameter groups.
    optimizer = torch.optim.AdamW(
        # The first group uses the configured weight decay, and the second group uses zero weight decay.
        [{"params": decay_parameters, "weight_decay": config["weight_decay"]}, {"params": other_parameters, "weight_decay": 0.0}], 
        # The configured learning rate is the base rate that the scheduler will adjust.
        lr=config["learning_rate"], 
        # The single-tensor implementation reduces peak memory overhead compared with the foreach implementation.
        foreach=False
    )
    # The scheduler controls how the optimizer's learning rate changes across update steps.
    scheduler = get_scheduler(
        # The configured scheduler type selects the learning rate schedule.
        config["lr_scheduler_type"], 
        # The scheduler is attached to the optimizer created above.
        optimizer=optimizer, 
        # The warmup step count is the configured fraction of total optimizer updates rounded up to an integer.
        num_warmup_steps=math.ceil(total_steps * config["warmup_ratio"]), 
        # The schedule spans the total number of optimizer updates after accounting for gradient accumulation.
        num_training_steps=total_steps
    )
    return optimizer, scheduler

def build_validation_steps(total_steps: int, validation_rounds: int):
    """Select distinct, evenly spaced optimizer steps including the final step."""
    if (total_steps < validation_rounds):
        raise ValueError("validation_rounds cannot exceed the total optimizer-step count.")
    return {(round_index * total_steps + validation_rounds - 1) // validation_rounds for round_index in range(1, validation_rounds + 1)}

def write_json(path: Path, value):
    """Replace a JSON file atomically, without a terminal newline."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if (temporary_path is not None and temporary_path.exists()):
            temporary_path.unlink()

def save_best_checkpoint(model, tokenizer, output_dir: Path, metrics: dict):
    """Replace the single best model and its metrics after both are staged."""
    best_path = output_dir / "best_checkpoint"
    selected_path = output_dir / "selected_checkpoint.json"
    temporary_dir = Path(tempfile.mkdtemp(prefix=".best_checkpoint-", dir=output_dir))
    preserve_recovery_files = False
    try:
        staged_path = temporary_dir / "new"
        backup_path = temporary_dir / "previous"
        metrics_path = temporary_dir / "selected_checkpoint.json"
        backup_metrics_path = temporary_dir / "previous_metrics.json"
        model.save_pretrained(staged_path, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(staged_path)
        write_json(metrics_path, metrics)
        had_best = best_path.exists()
        had_metrics = selected_path.exists()
        if (had_best != had_metrics):
            raise ValueError("The existing best checkpoint and its metrics must both be present or absent.")
        if (had_metrics):
            shutil.copy2(selected_path, backup_metrics_path)
        try:
            if (had_best):
                os.replace(best_path, backup_path)
            os.replace(staged_path, best_path)
            os.replace(metrics_path, selected_path)
        except BaseException:
            try:
                # Inspect the paths so an interruption between renames also rolls back.
                if (backup_path.exists()):
                    if (best_path.exists()):
                        shutil.rmtree(best_path)
                    os.replace(backup_path, best_path)
                elif (not had_best and best_path.exists()):
                    shutil.rmtree(best_path)
                if (had_metrics):
                    os.replace(backup_metrics_path, selected_path)
                else:
                    selected_path.unlink(missing_ok=True)
            except BaseException:
                # A failed rollback must not delete the only recoverable old model.
                preserve_recovery_files = True
                raise
            raise
    finally:
        # The temporary directory is removed unless a rollback failed and recovery files must be preserved.
        if (not preserve_recovery_files):
            shutil.rmtree(temporary_dir)

def train_model(model, tokenizer, datasets: dict, encoded_datasets: dict, config: dict, paths: dict):
    """Train either task with identical updates and parent-level validation ranking."""
    train_inputs = encoded_datasets["train"]
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_inputs, 
        batch_size=config["micro_batch"], 
        shuffle=True, 
        drop_last=False, 
        collate_fn=partial(collate_batch, tokenizer=tokenizer), 
        generator=generator
    )
    batches_per_epoch = len(train_loader)
    updates_per_epoch = math.ceil(batches_per_epoch / config["gradient_accumulation"])
    total_steps = updates_per_epoch * config["epochs"]
    validation_steps = build_validation_steps(total_steps, config["validation_rounds"])
    output_dir = paths["output_dir"]
    if (output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir()))):
        raise ValueError("output_dir must be absent or empty; existing runs are never overwritten.")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run.json", config)
    history = []
    write_json(output_dir / "evaluation_history.json", history)
    optimizer, scheduler = build_optimizer_scheduler(model, config, total_steps)
    device = next(model.parameters()).device
    best_rank = None
    # This counter tracks completed optimizer updates rather than individual batches.
    global_step = 0
    # This accumulator tracks the summed training loss for the current validation interval.
    interval_loss_sum = 0.0
    # This counter tracks the number of training inputs processed during the current validation interval.
    interval_input_count = 0
    # Existing gradients are cleared before the first accumulation group.
    optimizer.zero_grad(set_to_none=True)
    # Training mode enables training-specific behavior such as dropout.
    model.train()
    # Each epoch makes one complete pass through the training loader.
    for _ in range(config["epochs"]):
        # Each iteration provides a batch and its zero-based position within the epoch.
        for batch_index, batch in enumerate(train_loader):
            # Integer division identifies the first batch index in the current accumulation group.
            group_start = (batch_index // config["gradient_accumulation"]) * config["gradient_accumulation"]
            # This exclusive end boundary also accommodates a shorter final accumulation group.
            group_stop = min(group_start + config["gradient_accumulation"], batches_per_epoch)
            # This count includes all inputs from the epoch start through the current group's end.
            group_input_count = min(group_stop * config["micro_batch"], len(train_inputs))
            # Subtracting earlier inputs gives the actual number of inputs in this accumulation group.
            group_input_count -= group_start * config["micro_batch"]
            # Reference class IDs are moved to the same device as the model.
            labels = batch["labels"].to(device)
            # Automatic mixed precision uses BF16 for eligible operations when enabled in the configuration.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config["bf16"]):
                # The model computes raw class scores from token IDs and attention masks without receiving reference labels.
                logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
                # Cross-entropy sums the losses for all inputs in the current batch using FP32 logits.
                loss_sum = F.cross_entropy(logits.float(), labels, reduction="sum")
            if (not torch.isfinite(loss_sum).item()):
                raise FloatingPointError("Training loss is non-finite.")
            # Dividing each batch loss by the whole group's input count accumulates gradients for the group's mean loss.
            (loss_sum / group_input_count).backward()
            # The detached scalar loss contributes to reporting without retaining its computation graph.
            interval_loss_sum += loss_sum.detach().item()
            # The number of class labels equals the number of training inputs in the batch.
            interval_input_count += labels.numel()
            # Parameter updates wait until the final batch of the current accumulation group.
            if (batch_index + 1 != group_stop):
                continue
            # The accumulated gradient norm is clipped to the configured limit, and non-finite norms raise an error.
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
            # The optimizer updates model parameters once using the accumulated gradients.
            optimizer.step()
            # The scheduler advances the learning rate for the next optimizer update.
            scheduler.step()
            # The completed group's gradients are cleared before the next group starts.
            optimizer.zero_grad(set_to_none=True)
            # The global step counter advances after the optimizer update is complete.
            global_step += 1
            # Only scheduled optimizer steps trigger validation.
            if (global_step not in validation_steps):
                continue
            # The current model is evaluated on the full validation set with validation loss enabled.
            metrics = evaluate_dataset(model, tokenizer, datasets["validation"], encoded_datasets["validation"], config, calculate_loss=True)
            # The reported training loss is averaged over the training inputs processed during the current validation interval.
            metrics["training_loss"] = interval_loss_sum / interval_input_count
            # The current metrics are appended to the in-memory evaluation history.
            history.append(metrics)
            # All evaluation records are saved to the history file after each validation.
            write_json(output_dir / "evaluation_history.json", history)
            # Checkpoints are ranked first by macro F1 and then by accuracy when macro F1 is equal.
            metrics_rank = (metrics["macro_f1"], metrics["accuracy"])
            if (best_rank is None or metrics_rank > best_rank):
                # The model, tokenizer, and metrics are saved as the new best result.
                save_best_checkpoint(model, tokenizer, output_dir, metrics)
                # The best recorded ranking is updated to match the saved checkpoint.
                best_rank = metrics_rank
            # The loss accumulator is reset for the next validation interval.
            interval_loss_sum = 0.0
            # The input counter is reset for the next validation interval.
            interval_input_count = 0
    # Residual parameter gradients are cleared after all training epochs finish.
    model.zero_grad(set_to_none=True)
    # The final training model is moved to the CPU before the pipeline loads the best checkpoint.
    model.cpu()
    # Unused CUDA allocator cache is released while memory held by live tensors remains allocated.
    torch.cuda.empty_cache()