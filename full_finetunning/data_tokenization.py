import json
import torch

def load_datasets(paths: dict, task_type: str):
    """Read and validate the four datasets."""
    overall_labels = ("direct", "intermediate", "fully_evasive")
    request_labels = ("FULL", "PARTIAL", "NONE")
    datasets = {}
    seen_uids = {}
    seen_pairs = {}
    for split in ("train", "validation", "test", "human_test"):
        rows = []
        with paths[f"{split}_file"].open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if (not line.strip()):
                    continue
                location = f"{split}:{line_number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{location}: {error}") from error
                if (not isinstance(row, dict)):
                    raise ValueError(f"{location}: each record must be a JSON object")
                for field in ("uid", "question", "answer"):
                    if (not isinstance(row.get(field), str) or not row[field].strip()):
                        raise ValueError(f"{location}: {field} must be a nonempty string")
                if (task_type == "whole_qa" or split != "train"):
                    if (row.get("overall_label") not in overall_labels):
                        raise ValueError(f"{location}: a valid overall_label is required")
                if (task_type == "request"):
                    questions = row.get("sub_questions")
                    if (not isinstance(questions, list) or not questions or any(not isinstance(question, str) or not question.strip() for question in questions)):
                        raise ValueError(f"{location}: sub_questions must be a nonempty list of nonempty strings")
                    if (split in ("train", "validation")):
                        labels = row.get("sub_question_labels")
                        if (not isinstance(labels, list) or len(labels) != len(questions) or any(label not in request_labels for label in labels)):
                            raise ValueError(f"{location}: valid sub_question_labels must align with sub_questions")
                uid = row["uid"]
                pair = (" ".join(row["question"].split()), " ".join(row["answer"].split()))
                if (uid in seen_uids):
                    raise ValueError(f"{location}: duplicate uid also found at {seen_uids[uid]}")
                if (pair in seen_pairs):
                    raise ValueError(f"{location}: duplicate question-answer pair also found at {seen_pairs[pair]}")
                seen_uids[uid] = location
                seen_pairs[pair] = location
                rows.append(row)
        if (not rows):
            raise ValueError(f"{split}: dataset must not be empty")
        datasets[split] = rows
    return datasets

def build_classification_inputs(rows: list[dict], task_type: str, split: str):
    """Build whole-QA or request classification inputs."""
    overall_labels = ("direct", "intermediate", "fully_evasive")
    request_labels = ("FULL", "PARTIAL", "NONE")
    classification_inputs = []
    for row in rows:
        if (task_type == "whole_qa"):
            questions = [row["question"]]
        else:
            questions = row["sub_questions"]
        labels = [None] * len(questions)
        if (split in ("train", "validation")):
            if (task_type == "whole_qa"):
                labels = [overall_labels.index(row["overall_label"])]
            else:
                labels = [request_labels.index(label) for label in row["sub_question_labels"]]
        for question, label in zip(questions, labels):
            classification_inputs.append({
                "parent_uid": row["uid"], 
                "input_text": f"Question:\n{question}\n\nAnswer:\n{row['answer']}", 
                "label": label
            })
    return classification_inputs

def tokenize_classification_inputs(classification_inputs: list[dict], tokenizer, max_length: int):
    """Tokenize inputs without truncation."""
    tokenized_inputs = []
    for classification_input in classification_inputs:
        encoded = tokenizer(
            classification_input["input_text"], 
            add_special_tokens=True, 
            truncation=False, 
            padding=False, 
            return_attention_mask=True, 
            return_token_type_ids=False
        )
        length = len(encoded["input_ids"])
        if (length == 0 or length > max_length):
            raise ValueError(f"UID {classification_input['parent_uid']}: input has {length} tokens; require 1 <= length <= {max_length}. No truncation or sample removal is allowed.")
        tokenized_inputs.append({
            "input_ids": encoded["input_ids"], 
            "attention_mask": encoded["attention_mask"], 
            "parent_uid": classification_input["parent_uid"], 
            "label": classification_input["label"]
        })
    return tokenized_inputs

def collate_batch(tokenized_inputs: list[dict], tokenizer):
    """Pad tokens and collect reference labels."""
    features = [{"input_ids": tokenized_input["input_ids"], "attention_mask": tokenized_input["attention_mask"]} for tokenized_input in tokenized_inputs]
    batch = tokenizer.pad(features, padding=True, return_tensors="pt")
    labels = [tokenized_input["label"] for tokenized_input in tokenized_inputs]
    label_tensor = None
    if (labels[0] is not None):
        label_tensor = torch.tensor(labels, dtype=torch.long)
    return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"], "labels": label_tensor}