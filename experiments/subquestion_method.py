"""Classify sub-questions with the original full answer, then aggregate labels."""

import sys
import classical_baselines as b

SUB_LABELS = ("FULL", "PARTIAL", "NONE")
ANSWER_SOURCE = "original_answer"

if __name__ == "__main__":
    sys.modules["subquestion_method"] = sys.modules[__name__]


def sub_questions(row, require_labels=False):
    uid = row["uid"]
    if row.get("decomposition_issue"):
        raise ValueError(f"{uid}: resolve decomposition_issue before using sub-questions.")
    questions = row.get("sub_questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"{uid}: provide a nonempty sub_questions list first.")
    if any(not isinstance(q, str) or not q.strip() for q in questions):
        raise ValueError(f"{uid}: every sub-question must be a nonempty string.")
    if "num_sub_questions" in row:
        count = row["num_sub_questions"]
        if type(count) is not int or count != len(questions):
            raise ValueError(f"{uid}: num_sub_questions does not match sub_questions.")
    labels = []
    if require_labels:
        annotations = row.get("annotations")
        if not isinstance(annotations, list) or len(annotations) != len(questions):
            raise ValueError(f"{uid}: annotations must align with every sub-question.")
        for index, annotation in enumerate(annotations):
            if not isinstance(annotation, dict) or annotation.get("label") not in SUB_LABELS:
                raise ValueError(f"{uid}, sub-question {index + 1}: expected {SUB_LABELS}.")
            if annotation.get("needs_review", False) is not False:
                raise ValueError(f"{uid}, sub-question {index + 1}: resolve needs_review first.")
            labels.append(annotation["label"])
    return questions, labels


def evaluation_sub_labels(rows):
    """Return aligned reference labels, or None if the whole split is unlabeled."""
    labeled, unlabeled = [], []
    for row in rows:
        annotations = row.get("annotations")
        if annotations is not None and not isinstance(annotations, list):
            raise ValueError(f"{row['uid']}: annotations must be a list, null, or absent.")
        (labeled if annotations else unlabeled).append(row["uid"])
    if labeled and unlabeled:
        raise ValueError(
            "Mixed evaluation annotations: either all rows must contain sub-question labels or none should. "
            f"Found {len(labeled)} labeled rows (e.g. {labeled[:3]}) and "
            f"{len(unlabeled)} unlabeled rows (e.g. {unlabeled[:3]})."
        )
    if not labeled:
        return None
    return [sub_questions(row, require_labels=True)[1] for row in rows]


def sub_answers(row, questions):
    """Validate optional evidence for explicit analysis; normal classification ignores it."""
    uid = row["uid"]
    answers = row.get("sub_answers")
    if not isinstance(answers, list) or len(answers) != len(questions):
        raise ValueError(f"{uid}: sub_answers must be a list aligned with every sub-question.")
    original = row.get("answer")
    if not isinstance(original, str) or not original.strip():
        raise ValueError(f"{uid}: the original answer is required to validate excerpts.")
    for index, passages in enumerate(answers):
        prefix = f"{uid}: sub_answers[{index}]"
        if passages is None:
            raise ValueError(f"{prefix} is pending (null); provide exact excerpts, or [] if no passage is relevant.")
        if not isinstance(passages, list):
            raise ValueError(f"{prefix} must be a list of exact answer excerpts, not a string.")
        if any(not isinstance(p, str) or not p.strip() or p not in original for p in passages):
            raise ValueError(f"{prefix} must contain nonempty, exact contiguous quotes from the original answer.")
    return [list(passages) for passages in answers]


def aggregate_sub_labels(labels):
    """Map predicted coverage to an overall label using a fixed completeness rule."""
    labels = list(labels)
    if not labels or any(label not in SUB_LABELS for label in labels):
        raise ValueError("Aggregation requires at least one valid sub-question label.")
    if all(label == "FULL" for label in labels):
        return "direct"
    if all(label == "NONE" for label in labels):
        return "fully_evasive"
    return "intermediate"


class FixedAggregator:
    """A replaceable predictor over grouped sub-question predictions."""

    def predict(self, label_sequences):
        return [aggregate_sub_labels(labels) for labels in label_sequences]


if __name__ == "__main__":
    FixedAggregator.__module__ = "subquestion_method"


def aggregate_predictions(model, label_sequences):
    """A future fitted aggregator can implement the same predict interface."""
    aggregator = model.get("aggregator", FixedAggregator())
    predictions = list(aggregator.predict(label_sequences))
    if len(predictions) != len(label_sequences) or any(label not in b.OVERALL_LABELS for label in predictions):
        raise ValueError("Aggregator must return one valid overall label per parent question.")
    return predictions


def prepare_examples(rows, require_labels=False):
    b.validate_records(rows, require_overall=require_labels)
    pairs, labels, sizes = [], [], []
    for row in rows:
        questions, targets = sub_questions(row, require_labels)
        pairs.extend({"question": q, "answer": row["answer"]} for q in questions)
        labels.extend(targets)
        sizes.append(len(questions))
    return pairs, labels, sizes


def prepare_oracle_examples(rows, require_labels=False):
    """Build excerpt-based pairs only for explicitly requested oracle/evidence analysis."""
    pairs, labels, sizes = prepare_examples(rows, require_labels)
    offset = 0
    for row in rows:
        for passages in sub_answers(row, row["sub_questions"]):
            pairs[offset]["answer"] = "\n".join(passages)
            offset += 1
    return pairs, labels, sizes


def train_ours(rows, method, **parameters):
    pairs, labels, _ = prepare_examples(rows, require_labels=True)
    model = b.fitted_model(rows, pairs, labels, method, "subquestion", parameters)
    model["answer_source"] = ANSWER_SOURCE
    model["aggregator"] = FixedAggregator()
    return model


def predict_ours(model, rows):
    if model["variant"] != "subquestion":
        raise ValueError("Use classical_baselines.predict_baseline for a whole-question model.")
    if model.get("answer_source") != ANSWER_SOURCE:
        raise ValueError("This sub-question model is not marked as trained on original full answers; retrain it before predicting.")
    pairs, _, sizes = prepare_examples(rows)
    flat = model["pipeline"].predict(pairs).tolist()
    if len(flat) != sum(sizes) or any(label not in SUB_LABELS for label in flat):
        raise ValueError("Classifier must return one valid label per sub-question.")
    grouped, offset = [], 0
    for size in sizes:
        grouped.append(flat[offset:offset + size])
        offset += size
    overall = aggregate_predictions(model, grouped)
    predictions = []
    for row, labels, overall_label in zip(rows, grouped, overall):
        predictions.append({"uid": row["uid"],
                            "predicted_overall_label": overall_label,
                            "sub_questions": list(row["sub_questions"]),
                            "predicted_sub_labels": labels})
    return predictions


def evaluate_ours(model, rows):
    b.validate_records(rows, require_overall=True)
    reference_sub_labels = evaluation_sub_labels(rows)
    b.check_no_overlap(model["train_signatures"], rows)
    predictions = predict_ours(model, rows)
    truth = [row["overall_label"] for row in rows]
    metrics = {"overall": b.classification_metrics(
        truth, [row["predicted_overall_label"] for row in predictions], b.OVERALL_LABELS)}
    for result, label in zip(predictions, truth):
        result["overall_label"] = label
    if reference_sub_labels is not None:
        sub_truth = []
        for labels, result in zip(reference_sub_labels, predictions):
            sub_truth.extend(labels)
            result["sub_labels"] = labels
        sub_predictions = [label for result in predictions for label in result["predicted_sub_labels"]]
        metrics["subquestion"] = b.classification_metrics(sub_truth, sub_predictions, SUB_LABELS)
    return metrics, predictions


if __name__ == "__main__":
    from run_experiment import experiment_main
    experiment_main("subquestion")
