from functools import partial
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data_tokenization import collate_batch

def predict_labels(model, tokenizer, tokenized_inputs: list[dict], config: dict, calculate_loss: bool = False):
    """Predict task labels without passing reference labels to the model."""
    # The first model parameter identifies the device that must also hold the inputs.
    device = next(model.parameters()).device
    # The data loader groups the tokenized inputs into evaluation batches.
    data_loader = DataLoader(
        # The input list contains token IDs, attention masks, parent identifiers, and optional labels.
        tokenized_inputs, 
        # The configuration specifies the maximum number of inputs in each evaluation batch.
        batch_size=config["eval_batch"], 
        # Preserving the input order keeps predictions aligned with the original records.
        shuffle=False, 
        # Keeping the final incomplete batch ensures that every input receives a prediction.
        drop_last=False, 
        # The partial function supplies the tokenizer so each batch can be padded and converted to tensors.
        collate_fn=partial(collate_batch, tokenizer=tokenizer)
    )
    predictions = []
    loss_sum = 0.0
    was_training = model.training
    model.eval()
    try:
        # Inference mode disables gradient recording because evaluation does not update model parameters.
        with torch.inference_mode():
            # Each iteration processes one padded batch produced by the data loader.
            for batch in data_loader:
                # The token IDs are moved to the same device as the model.
                input_ids = batch["input_ids"].to(device)
                # The attention mask is moved to the model device so padding positions can be masked out.
                attention_mask = batch["attention_mask"].to(device)
                # Automatic mixed precision uses BF16 for eligible operations when the configuration enables it.
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config["bf16"]):
                    # The model receives only text inputs and returns raw class scores that are converted to FP32.
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
                # The output must contain exactly three class scores for each input in the batch.
                if (logits.shape != (input_ids.shape[0], 3)):
                    # An unexpected output shape stops evaluation with a descriptive error.
                    raise ValueError("Expected three classification logits per example.")
                # Every class score must be finite rather than NaN or infinity.
                if (not torch.isfinite(logits).all().item()):
                    # Invalid scores stop evaluation before they can become class predictions.
                    raise FloatingPointError("Evaluation produced non-finite logits.")
                # The highest-scoring class IDs are moved to the CPU, converted to a list, and appended in order.
                predictions.extend(logits.argmax(dim=-1).cpu().tolist())
                # Reference labels are used only when loss calculation is requested.
                if (calculate_loss):
                    # Computing a supervised loss requires reference labels for the current batch.
                    if (batch["labels"] is None):
                        # Missing reference labels prevent the requested loss from being calculated.
                        raise ValueError("Validation loss requires task reference labels.")
                    # Cross-entropy losses are summed across the batch using reference labels on the model device.
                    loss = F.cross_entropy(logits, batch["labels"].to(device), reduction="sum")
                    # The batch loss must be finite before it contributes to the total.
                    if (not torch.isfinite(loss).item()):
                        # An invalid batch loss stops evaluation with a numerical error.
                        raise FloatingPointError("Validation produced a non-finite loss.")
                    # The scalar batch loss is converted to a Python number and added to the total.
                    loss_sum += loss.item()
    # This block runs whether evaluation succeeds or raises an exception.
    finally:
        # The saved flag restores the model to its previous training or evaluation mode.
        model.train(was_training)
    mean_loss = None
    if (calculate_loss):
        mean_loss = loss_sum / len(tokenized_inputs)
    return predictions, mean_loss

def aggregate_request_predictions(rows: list[dict], predictions: list[int], tokenized_inputs: list[dict]):
    """Aggregate predicted request labels into one overall label per parent QA."""
    if (not rows or len(predictions) != len(tokenized_inputs)):
        raise ValueError("Request aggregation requires nonempty, aligned predictions.")
    grouped_predictions = {row["uid"]: [] for row in rows}
    for tokenized_input, prediction in zip(tokenized_inputs, predictions):
        parent_uid = tokenized_input["parent_uid"]
        grouped_predictions[parent_uid].append(prediction)
    overall_predictions = []
    for row in rows:
        request_predictions = grouped_predictions[row["uid"]]
        if (not request_predictions or len(request_predictions) != len(row["sub_questions"])):
            raise ValueError(f"Incomplete request predictions for UID {row['uid']}.")
        # FULL/PARTIAL/NONE and direct/intermediate/fully_evasive use IDs 0/1/2.
        if (all(prediction == 0 for prediction in request_predictions)):
            overall_predictions.append(0)
        elif (all(prediction == 2 for prediction in request_predictions)):
            overall_predictions.append(2)
        else:
            overall_predictions.append(1)
    return overall_predictions

def compute_metrics(reference_labels: list[int], predicted_labels: list[int]):
    """Compute overall-QA metrics using a fixed three-class label set."""
    if (not reference_labels or len(reference_labels) != len(predicted_labels)):
        raise ValueError("Metrics require nonempty, aligned reference and predicted labels.")
    label_names = ("direct", "intermediate", "fully_evasive")
    matrix = [[0] * 3 for _ in label_names]
    for reference, prediction in zip(reference_labels, predicted_labels):
        matrix[reference][prediction] += 1
    per_class = {}
    for index, label_name in enumerate(label_names):
        true_positives = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = true_positives / predicted_count if (predicted_count) else 0.0
        recall = true_positives / support if (support) else 0.0
        denominator = support + predicted_count
        f1 = 2 * true_positives / denominator if (denominator) else 0.0
        per_class[label_name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    sample_count = len(reference_labels)
    return {
        "accuracy": sum(matrix[index][index] for index in range(3)) / sample_count, 
        "macro_f1": sum(values["f1"] for values in per_class.values()) / 3, 
        "weighted_f1": sum(values["f1"] * values["support"] for values in per_class.values()) / sample_count, 
        "per_class": per_class, 
        "sample_count": sample_count
    }

def evaluate_dataset(model, tokenizer, rows: list[dict], tokenized_inputs: list[dict], config: dict, calculate_loss: bool = False):
    """Score original QA pairs; request validation loss remains request-level CE."""
    predictions, mean_loss = predict_labels(model, tokenizer, tokenized_inputs, config, calculate_loss=calculate_loss)
    if (config["task_type"] == "request"):
        overall_predictions = aggregate_request_predictions(rows, predictions, tokenized_inputs)
    else:
        expected_uids = [row["uid"] for row in rows]
        predicted_uids = [tokenized_input["parent_uid"] for tokenized_input in tokenized_inputs]
        if (predicted_uids != expected_uids or len(predictions) != len(rows)):
            raise ValueError("Whole-QA predictions must align with the original QA records.")
        overall_predictions = predictions
    label_names = ("direct", "intermediate", "fully_evasive")
    reference_labels = [label_names.index(row["overall_label"]) for row in rows]
    metrics = compute_metrics(reference_labels, overall_predictions)
    if (calculate_loss):
        metrics["validation_loss"] = mean_loss
    return metrics

def evaluate_test_sets(model, tokenizer, datasets: dict, encoded_datasets: dict, config: dict):
    """Evaluate the caller's reloaded best model without computing test losses."""
    return {split: evaluate_dataset(model, tokenizer, datasets[split], encoded_datasets[split], config) for split in ("test", "human_test")}