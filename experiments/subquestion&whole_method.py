"""Combine whole-QA TF-IDF with out-of-fold sub-question coverage predictions."""

from collections import Counter
import sys
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import classical_baselines as b
import subquestion_method as s

METHODS = ("tfidf_lr", "tfidf_svm")
COVERAGE_FEATURES = ("full_ratio", "partial_ratio", "none_ratio", "num_sub_questions")
COVERAGE_SOURCE = "grouped_out_of_fold_predictions"

if __name__ == "__main__":
    sys.modules["subquestion&whole_method"] = sys.modules[__name__]


def coverage_features(label_sequences):
    values = []
    for labels in label_sequences:
        if not labels or any(label not in s.SUB_LABELS for label in labels):
            raise ValueError("Coverage requires nonempty sequences of predicted FULL/PARTIAL/NONE labels.")
        counts = Counter(labels)
        size = len(labels)
        values.append([counts[label] / size for label in s.SUB_LABELS] + [size])
    return np.asarray(values, dtype=np.float64).reshape(-1, len(COVERAGE_FEATURES))


def prediction_inputs(rows):
    """Expose only original QA and decomposition to the sub-question predictor."""
    return [{"uid": row["uid"], "question": row["question"], "answer": row["answer"],
             "sub_questions": list(row["sub_questions"])} for row in rows]


def parent_groups(rows):
    """Keep parents sharing a normalized whole question or answer in the same fold."""
    roots = list(range(len(rows)))

    def find(index):
        while roots[index] != index:
            roots[index] = roots[roots[index]]
            index = roots[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        for field in ("question", "answer"):
            key = (field, b.normalize(row[field]))
            if key in seen:
                roots[find(index)] = find(seen[key])
            else:
                seen[key] = index
    return np.asarray([find(index) for index in range(len(rows))])


def training_folds(rows, method, oof_folds=5, seed=42):
    """Validate the whole training split and construct parent-grouped OOF folds."""
    if method not in METHODS:
        raise ValueError(f"Hybrid supports {METHODS}, using the same family for both classifiers.")
    _, labels, _ = s.prepare_examples(rows, require_labels=True)
    if len(set(labels)) < 2 or len({row["overall_label"] for row in rows}) < 2:
        raise ValueError("Hybrid training needs at least two sub-question and two overall classes.")
    groups = parent_groups(rows)
    group_count = len(set(groups))
    if type(oof_folds) is not int or not 2 <= oof_folds <= group_count:
        raise ValueError(f"--oof-folds must be between 2 and the number of independent parent groups ({group_count}).")
    splitter = StratifiedGroupKFold(n_splits=oof_folds, shuffle=True, random_state=seed)
    overall = [row["overall_label"] for row in rows]
    try:
        folds = list(splitter.split(np.zeros(len(rows)), overall, groups))
    except ValueError as exc:
        raise ValueError(f"Cannot construct OOF folds: {exc} Adjust --oof-folds for this training set.") from exc
    for number, (train_indices, held_indices) in enumerate(folds, 1):
        if not len(train_indices) or not len(held_indices):
            raise ValueError(f"OOF fold {number} is empty; use fewer folds or more independent parents.")
        train_rows = [rows[index] for index in train_indices]
        held_rows = [rows[index] for index in held_indices]
        b.check_no_overlap(b.record_signatures(train_rows), held_rows, f"OOF fold {number} training")
        fold_labels = {label for row in train_rows for label in s.sub_questions(row, True)[1]}
        if len(fold_labels) < 2:
            raise ValueError(f"OOF fold {number} has fewer than two training sub-question classes; revise the folds or training data.")
    return folds


def validate_options(method, coverage_weight, parameters):
    if method not in METHODS:
        raise ValueError(f"Hybrid supports {METHODS}.")
    if parameters.get("text_mode", "qa") != "qa":
        raise ValueError("Hybrid uses complete Q+A; --text-mode must be qa.")
    if not np.isfinite(coverage_weight) or coverage_weight <= 0:
        raise ValueError("--coverage-weight must be finite and positive.")
    b.build_pipeline(method, **parameters)


def train_hybrid(rows, method, oof_folds=5, coverage_weight=1.0, **parameters):
    validate_options(method, coverage_weight, parameters)
    folds = training_folds(rows, method, oof_folds, parameters.get("seed", 42))
    oof_predictions = [None] * len(rows)
    fold_details = []
    oof_started = perf_counter()
    for number, (train_indices, held_indices) in enumerate(folds, 1):
        started = perf_counter()
        train_rows = [rows[index] for index in train_indices]
        held_rows = [rows[index] for index in held_indices]
        fold_model = s.train_ours(train_rows, method, **parameters)
        predicted = s.predict_ours(fold_model, prediction_inputs(held_rows))
        for index, result in zip(held_indices, predicted):
            if result["uid"] != rows[index]["uid"] or oof_predictions[index] is not None:
                raise ValueError("OOF predictions must cover each training parent exactly once.")
            oof_predictions[index] = {"uid": result["uid"], "fold": number,
                                      "predicted_sub_labels": result["predicted_sub_labels"]}
        fold_details.append({"fold": number, "train_parent_count": len(train_rows),
                             "held_parent_count": len(held_rows),
                             "train_subquestion_count": fold_model["train_example_count"],
                             "train_subquestion_class_counts": fold_model["train_class_counts"],
                             "seconds": perf_counter() - started})
        del fold_model
    if any(result is None for result in oof_predictions):
        raise ValueError("Missing OOF predictions; the overall classifier cannot be fitted.")
    oof_seconds = perf_counter() - oof_started
    coverage = coverage_features([result["predicted_sub_labels"] for result in oof_predictions])
    for result, values in zip(oof_predictions, coverage):
        result["coverage_features"] = dict(zip(COVERAGE_FEATURES, values.tolist()))

    started = perf_counter()
    pairs, overall_labels = b.prepare_examples(rows, require_labels=True)
    pipeline = b.build_pipeline(method, **parameters)
    whole_features = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]
    text_matrix = whole_features.fit_transform(pairs)
    scaler = StandardScaler(with_mean=False)
    coverage_matrix = scaler.fit_transform(coverage) * coverage_weight
    matrix = hstack([text_matrix, csr_matrix(coverage_matrix)], format="csr")
    classifier.fit(matrix, overall_labels)
    overall_seconds = perf_counter() - started

    started = perf_counter()
    subquestion_model = s.train_ours(rows, method, **parameters)
    final_subquestion_seconds = perf_counter() - started
    return {
        "schema_version": b.SCHEMA_VERSION, "method": method, "variant": "hybrid",
        "feature_mode": "qa_tfidf+predicted_coverage", "answer_source": s.ANSWER_SOURCE,
        "whole_features": whole_features, "coverage_scaler": scaler, "classifier": classifier,
        "subquestion_model": subquestion_model, "coverage_feature_names": list(COVERAGE_FEATURES),
        "coverage_source": COVERAGE_SOURCE, "coverage_weight": coverage_weight,
        "train_parent_count": len(rows), "train_example_count": len(rows),
        "train_class_counts": dict(Counter(overall_labels)), "train_signatures": b.record_signatures(rows),
        "sklearn_version": b.sklearn.__version__, "oof_predictions": oof_predictions,
        "training_details": {
            "oof_folds": oof_folds, "group_count": len(set(parent_groups(rows))),
            "coverage_source": COVERAGE_SOURCE, "coverage_feature_names": list(COVERAGE_FEATURES),
            "coverage_weight": coverage_weight, "oof_seconds": oof_seconds,
            "overall_fit_seconds": overall_seconds, "final_subquestion_fit_seconds": final_subquestion_seconds,
            "final_subquestion_example_count": subquestion_model["train_example_count"],
            "oof_subquestion_examples_total": sum(fold["train_subquestion_count"] for fold in fold_details),
            "folds": fold_details,
        },
    }


def predict_hybrid(model, rows):
    if (model.get("variant") != "hybrid" or model.get("answer_source") != s.ANSWER_SOURCE
            or model.get("coverage_source") != COVERAGE_SOURCE
            or model.get("coverage_feature_names") != list(COVERAGE_FEATURES)):
        raise ValueError("Expected a compatible hybrid model trained with full answers and OOF coverage.")
    s.prepare_examples(rows)
    sub_predictions = s.predict_ours(model["subquestion_model"], prediction_inputs(rows))
    coverage = coverage_features([result["predicted_sub_labels"] for result in sub_predictions])
    pairs, _ = b.prepare_examples(rows)
    text_matrix = model["whole_features"].transform(pairs)
    coverage_matrix = model["coverage_scaler"].transform(coverage) * model["coverage_weight"]
    matrix = hstack([text_matrix, csr_matrix(coverage_matrix)], format="csr")
    overall = model["classifier"].predict(matrix).tolist()
    if len(overall) != len(rows) or any(label not in b.OVERALL_LABELS for label in overall):
        raise ValueError("Hybrid classifier must return one valid overall label per parent.")
    return [{"uid": result["uid"], "predicted_overall_label": label,
             "sub_questions": result["sub_questions"], "predicted_sub_labels": result["predicted_sub_labels"],
             "coverage_features": dict(zip(COVERAGE_FEATURES, values.tolist())),
             "fixed_aggregation_label": result["predicted_overall_label"]}
            for result, values, label in zip(sub_predictions, coverage, overall)]


def evaluate_hybrid(model, rows):
    b.validate_records(rows, require_overall=True)
    reference_sub_labels = s.evaluation_sub_labels(rows)
    b.check_no_overlap(model["train_signatures"], rows)
    predictions = predict_hybrid(model, rows)
    truth = [row["overall_label"] for row in rows]
    metrics = {"overall": b.classification_metrics(
        truth, [result["predicted_overall_label"] for result in predictions], b.OVERALL_LABELS)}
    for result, label in zip(predictions, truth):
        result["overall_label"] = label
    if reference_sub_labels is not None:
        for result, labels in zip(predictions, reference_sub_labels):
            result["sub_labels"] = labels
        metrics["subquestion"] = b.classification_metrics(
            [label for labels in reference_sub_labels for label in labels],
            [label for result in predictions for label in result["predicted_sub_labels"]], s.SUB_LABELS)
    return metrics, predictions


if __name__ == "__main__":
    from run_experiment import experiment_main
    experiment_main("hybrid")
