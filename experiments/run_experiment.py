"""Shared single-experiment execution and comparison reports; batch loops live in run_experiments.sh."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib
from itertools import combinations
import json
from pathlib import Path
import sys
from time import perf_counter
import uuid

import numpy as np
import sklearn

import classical_baselines as b
import subquestion_method

hybrid_method = importlib.import_module("subquestion&whole_method")

CODE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CODE_DIR / "results"
DEFAULT_HUMAN = CODE_DIR / "first_round_data/human_annotated_data/first_round"
DEFAULT_LLM = CODE_DIR / "first_round_data/LLM_annotations/llm_dev_test_300/test_300.jsonl"
TEST_NAMES = ("human_test", "llm_test")
VARIANT_FOLDERS = {"whole": "baseline", "subquestion": "ours", "hybrid": "hybrid"}


def result_directory(run_name):
    if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("--run-name must be a single folder name.")
    output = b.code_path(RESULTS_DIR / "baselines" / run_name)
    if RESULTS_DIR.resolve() not in output.parents:
        raise ValueError("Output must remain inside results/.")
    return output


def resolve_files(values):
    paths = []
    for value in values:
        path = b.code_path(value)
        candidates = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        if not candidates:
            raise ValueError(f"No JSONL files in {path}")
        for candidate in candidates:
            candidate = b.code_path(candidate)
            if not candidate.is_file() or candidate.suffix != ".jsonl":
                raise ValueError(f"Expected a JSONL file: {candidate}")
            if candidate in paths:
                raise ValueError(f"Input file supplied more than once: {candidate}")
            paths.append(candidate)
    return paths


def file_details(paths):
    details = []
    for path in paths:
        path = b.code_path(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        details.append({"path": str(path), "sha256": digest.hexdigest()})
    return details


def attach_subquestions(rows, paths, split):
    """Join sub-questions without importing external labels or answer excerpts."""
    by_uid = {}
    for path in paths:
        path = b.code_path(path)
        with path.open(encoding="utf-8-sig") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON.") from exc
                if not isinstance(item, dict) or not isinstance(item.get("uid"), str):
                    raise ValueError(f"{path}:{number}: missing string uid.")
                uid = item["uid"]
                if uid in by_uid:
                    raise ValueError(f"{split}: duplicate decomposition UID {uid}")
                if item.get("decomposition_issue"):
                    raise ValueError(f"{uid}: resolve decomposition_issue first.")
                subquestion_method.sub_questions(item)
                by_uid[uid] = item
    required = {row["uid"] for row in rows}
    missing, extra = required - by_uid.keys(), by_uid.keys() - required
    if missing or extra:
        raise ValueError(f"{split} decomposition UID mismatch: {len(missing)} missing, "
                         f"{len(extra)} extra. Supply exactly this split's samples.")
    result = []
    for row in rows:
        item = by_uid[row["uid"]]
        if not isinstance(item.get("question"), str) or b.normalize(item["question"]) != b.normalize(row["question"]):
            raise ValueError(f"{row['uid']}: decomposition question differs from the test question.")
        if "answer" in item and item["answer"] != row["answer"]:
            raise ValueError(f"{row['uid']}: decomposition answer differs from the test answer.")
        if row.get("sub_questions") and row["sub_questions"] != item["sub_questions"]:
            raise ValueError(f"{row['uid']}: conflicting existing and external sub-questions.")
        result.append({**row, "sub_questions": list(item["sub_questions"]),
                       "num_sub_questions": len(item["sub_questions"])})
    return result


def validate_split(rows, name, variant, expected_count=None, training=False):
    b.validate_records(rows)
    issues = []
    if expected_count is not None and len(rows) != expected_count:
        issues.append(f"expected {expected_count} parents, found {len(rows)}")
    invalid = [row["uid"] for row in rows if row.get("overall_label") not in b.OVERALL_LABELS]
    if invalid:
        issues.append(f"{len(invalid)} missing/invalid overall_label values (e.g. {invalid[:3]})")
    if issues:
        raise ValueError(f"{name}: " + "; ".join(issues) + ". No samples were dropped.")
    if variant == "whole":
        return
    try:
        subquestion_method.prepare_examples(rows, require_labels=training)
        if not training:
            subquestion_method.evaluation_sub_labels(rows)
    except ValueError as exc:
        hint = "Training needs aligned sub-question labels." if training else (
            "Supply sub_questions in the records or with --human-subquestions / --llm-subquestions. "
            "Reference sub-question labels are optional, but must be present for all rows or none."
        )
        raise ValueError(f"{name}: {exc} {hint}") from exc


def split_details(rows, variant):
    details = {
        "parent_count": len(rows),
        "subquestion_count": sum(len(row.get("sub_questions") or []) for row in rows),
        "overall_class_counts": dict(Counter(row["overall_label"] for row in rows)),
        "annotation_status_counts": dict(Counter(str(row.get("annotation_status", "unspecified")) for row in rows)),
    }
    reference_labels = subquestion_method.evaluation_sub_labels(rows) if variant != "whole" else None
    if reference_labels is not None:
        disagreements, all_labels = [], []
        for row, labels in zip(rows, reference_labels):
            all_labels.extend(labels)
            aggregated = subquestion_method.aggregate_sub_labels(labels)
            if aggregated != row["overall_label"]:
                disagreements.append({"uid": row["uid"], "overall_label": row["overall_label"],
                                      "aggregated_reference_label": aggregated})
        details["subquestion_class_counts"] = dict(Counter(all_labels))
        details["reference_aggregation_agreement"] = 1 - len(disagreements) / len(rows)
        details["reference_aggregation_disagreements"] = disagreements
    return details


def load_datasets(args, variant):
    files = {"train": resolve_files(args.train), "human_test": resolve_files(args.human_test),
             "llm_test": resolve_files(args.llm_test)}
    datasets = {name: b.read_jsonl(paths) for name, paths in files.items()}
    if variant != "whole":
        for name, supplied in (("human_test", args.human_subquestions), ("llm_test", args.llm_subquestions)):
            if supplied:
                paths = resolve_files(supplied)
                files[f"{name}_decomposition"] = paths
                datasets[name] = attach_subquestions(datasets[name], paths, name)
    validate_split(datasets["train"], "train", variant, training=True)
    validate_split(datasets["human_test"], "human_test", variant, args.expected_human_count)
    validate_split(datasets["llm_test"], "llm_test", variant, args.expected_llm_count)
    for left, right in combinations(datasets, 2):
        b.check_no_overlap(b.record_signatures(datasets[left]), datasets[right], left)
    return datasets, files


def data_fingerprint(rows, include_subquestions=False):
    selected = []
    for row in sorted(rows, key=lambda row: row["uid"]):
        item = {key: row[key] for key in ("uid", "question", "answer", "overall_label")}
        if include_subquestions:
            item["sub_questions"] = row["sub_questions"]
            item["sub_labels"] = row.get("sub_labels", [annotation["label"] for annotation in (row.get("annotations") or [])])
        selected.append(item)
    encoded = json.dumps(selected, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_single(args, variant):
    parameters = {"text_mode": args.text_mode, "seed": args.seed, "max_features": args.max_features,
                  "min_df": args.min_df, "c": args.c, "alpha": args.alpha,
                  "n_estimators": args.n_estimators,
                  "class_weight": None if args.class_weight == "none" else args.class_weight}
    b.build_pipeline(args.method, **parameters)
    hybrid_parameters = {}
    if variant == "hybrid":
        hybrid_parameters = {"oof_folds": args.oof_folds, "coverage_weight": args.coverage_weight}
        hybrid_method.validate_options(args.method, args.coverage_weight, parameters)
    if args.expected_human_count <= 0 or args.expected_llm_count <= 0:
        raise ValueError("Expected test sizes must be positive.")
    output = b.code_path(result_directory(args.run_name) / VARIANT_FOLDERS[variant] / args.method)
    datasets, files = load_datasets(args, variant)
    if variant == "whole":
        _, labels = b.prepare_examples(datasets["train"], require_labels=True)
        fit, evaluate = b.train_baseline, b.evaluate_baseline
    elif variant == "hybrid":
        hybrid_method.training_folds(datasets["train"], args.method, args.oof_folds, args.seed)
        _, labels = b.prepare_examples(datasets["train"], require_labels=True)
        fit, evaluate = hybrid_method.train_hybrid, hybrid_method.evaluate_hybrid
    else:
        _, labels, _ = subquestion_method.prepare_examples(datasets["train"], require_labels=True)
        fit, evaluate = subquestion_method.train_ours, subquestion_method.evaluate_ours
    if args.method != "majority" and len(set(labels)) < 2:
        raise ValueError(f"{variant}: training requires at least two observed classes.")
    if args.check_only:
        print(f"OK: {variant} / {args.method}; train={len(datasets['train'])}, "
              f"human={len(datasets['human_test'])}, LLM={len(datasets['llm_test'])}")
        return None
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 2, "status": "running", "run_id": uuid.uuid4().hex,
        "run_name": args.run_name, "method": args.method, "variant": variant,
        "answer_source": subquestion_method.ANSWER_SOURCE if variant == "subquestion" else "original_answer",
        "started_at": datetime.now(timezone.utc).isoformat(), "parameters": parameters,
        "versions": {"python": sys.version.split()[0], "sklearn": sklearn.__version__, "numpy": np.__version__},
        "inputs": {name: file_details(paths) for name, paths in files.items()},
        "datasets": {name: split_details(rows, variant) for name, rows in datasets.items()},
        "data_signatures": {name: data_fingerprint(rows) for name, rows in datasets.items()},
    }
    if variant != "whole":
        result["subquestion_signatures"] = {name: data_fingerprint(rows, True) for name, rows in datasets.items()}
    if variant == "hybrid":
        result["hybrid_parameters"] = hybrid_parameters
    b.write_json(output / "results.json", result)
    started = perf_counter()
    model = fit(datasets["train"], args.method, **parameters, **hybrid_parameters)
    fit_seconds = perf_counter() - started
    b.save_model(model, output / "model.joblib")
    model_details = {key: model[key] for key in (
        "method", "variant", "feature_mode", "answer_source", "train_parent_count", "train_example_count", "train_class_counts")}
    model_details.update({"fit_seconds": fit_seconds, "model_file": "model.joblib"})
    if variant == "hybrid":
        model_details["training_details"] = model["training_details"]
        b.write_jsonl(output / "oof_predictions.jsonl", model["oof_predictions"])
    result["model"] = model_details
    result["evaluations"] = []
    saved_predictions = []
    for name in TEST_NAMES:
        started = perf_counter()
        metrics, predictions = evaluate(model, datasets[name])
        result["evaluations"].append({**model_details, "dataset": name, "metrics": metrics,
                                      "evaluation_seconds": perf_counter() - started})
        for row, prediction in zip(datasets[name], predictions):
            saved_predictions.append({"run_id": result["run_id"], "dataset": name,
                                      "method": args.method, "variant": variant,
                                      "question": row["question"], "answer": row["answer"],
                                      "source_row": row.get("source_row"),
                                      "overall_rationale": row.get("overall_rationale", ""), **prediction})
        score = metrics["overall"]
        print(f"{args.method} / {variant} | {name} | accuracy={score['accuracy']:.2%} "
              f"macro-F1={score['macro_f1']:.2%}", flush=True)
    b.write_jsonl(output / "predictions.jsonl", saved_predictions)
    result.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat()})
    b.write_json(output / "results.json", result)
    write_summary_csv(output / "summary.csv", result["evaluations"])
    print(f"Saved: {output} (same names overwritten)", flush=True)
    return output


def paired_predictions(method, dataset, rows, baseline, ours):
    result = []
    for row, before, after in zip(rows, baseline, ours):
        if row["uid"] != before["uid"] or row["uid"] != after["uid"]:
            raise ValueError("Prediction order or parent UID changed.")
        truth = row["overall_label"]
        baseline_correct = before["predicted_overall_label"] == truth
        ours_correct = after["predicted_overall_label"] == truth
        outcome = {(True, True): "both_correct", (False, True): "improved",
                   (True, False): "regressed", (False, False): "both_wrong"}
        item = {
            "dataset": dataset, "method": method, "uid": row["uid"],
            "source_row": row.get("source_row"),
            "question": row["question"], "answer": row["answer"],
            "overall_label": truth, "overall_rationale": row.get("overall_rationale", ""),
            "baseline_prediction": before["predicted_overall_label"],
            "our_prediction": after["predicted_overall_label"],
            "baseline_correct": baseline_correct, "our_correct": ours_correct,
            "outcome": outcome[(baseline_correct, ours_correct)],
            "sub_questions": after["sub_questions"],
            "predicted_sub_labels": after["predicted_sub_labels"],
        }
        if "sub_labels" in after:
            item["reference_sub_labels"] = after["sub_labels"]
        result.append(item)
    return result


def write_summary_csv(path, evaluations):
    path = b.code_path(path)
    rows = []
    for item in evaluations:
        metric = item["metrics"]["overall"]
        row = {key: item[key] for key in (
            "dataset", "method", "variant", "train_parent_count", "train_example_count",
            "fit_seconds", "evaluation_seconds",
        )}
        row.update({key: metric[key] for key in ("count", "accuracy", "macro_f1", "weighted_f1")})
        for label in b.OVERALL_LABELS:
            for key in ("precision", "recall", "f1-score", "support"):
                row[f"{label}_{key}"] = metric["classification_report"][label][key]
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(result, path):
    path = b.code_path(path)
    lines = ["# Classical baseline comparison", "", f"Run: `{result['run_name']}`", "",
             "Baseline predicts the overall label from the original question and answer. "
             "Our method pairs each supplied sub-question with the original full answer, "
             "predicts FULL/PARTIAL/NONE, and aggregates the labels. "
             "Optional sub_answers evidence is not used as classifier input. "
             "The fixed rule maps all FULL to direct, all NONE to fully_evasive, and everything else to intermediate.", "",
             "Variants use the same training parents, classifier family, and base hyperparameters. "
             "The sub-question variant has more training instances; equal compute budgets are not enforced. "
             "No test labels are used for fitting or model selection. Decomposition time is excluded.", "",
             "When included, Hybrid combines whole-QA TF-IDF with predicted FULL/PARTIAL/NONE ratios and request count. "
             "Its overall classifier is trained on grouped out-of-fold coverage predictions; a final sub-question "
             "model fitted on all training parents supplies coverage at evaluation. Extra OOF fits are included in fit time.", "",
             "Overall accuracy and F1 always use the same reference overall labels. "
             "LLM-test scores measure agreement with LLM labels and are not human-gold accuracy. "
             "Sub-question metrics, when available, are reported separately in results.json.", "",
             "## Data", "", "| Split | Parents | Sub-questions | direct | intermediate | fully_evasive |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, item in result["datasets"].items():
        counts = item["overall_class_counts"]
        lines.append(f"| {name} | {item['parent_count']} | {item['subquestion_count']} | "
                     + " | ".join(str(counts.get(label, 0)) for label in b.OVERALL_LABELS) + " |")
    for name, item in result["datasets"].items():
        if "reference_aggregation_agreement" in item:
            lines.extend(["", f"{name}: reference sub-label aggregation agrees with overall labels in "
                          f"{item['reference_aggregation_agreement']:.2%} of records. "
                          "This is an annotation consistency check, not model accuracy. References are unchanged."])
    lines.extend(["", "## Overall comparison", "",
                  "All scores below are percentages; deltas are percentage points.", "",
                  "| Test set | Method | Baseline accuracy | Our accuracy | Δ accuracy | Baseline macro-F1 | Our macro-F1 | Δ macro-F1 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for item in result["comparisons"]:
        lines.append(f"| {item['dataset']} | {item['method']} | "
                     f"{item['baseline_accuracy'] * 100:.2f} | {item['our_accuracy'] * 100:.2f} | "
                     f"{item['accuracy_delta_pp']:+.2f} | {item['baseline_macro_f1'] * 100:.2f} | "
                     f"{item['our_macro_f1'] * 100:.2f} | {item['macro_f1_delta_pp']:+.2f} |")
    if result.get("hybrid_comparisons"):
        lines.extend(["", "## Three-way comparison", "",
                      "| Test set | Method | Direct accuracy | Sub-question accuracy | Hybrid accuracy | Direct macro-F1 | Sub-question macro-F1 | Hybrid macro-F1 |",
                      "|---|---|---:|---:|---:|---:|---:|---:|"])
        for item in result["hybrid_comparisons"]:
            keys = ("baseline_accuracy", "subquestion_accuracy", "hybrid_accuracy",
                    "baseline_macro_f1", "subquestion_macro_f1", "hybrid_macro_f1")
            lines.append(f"| {item['dataset']} | {item['method']} | " + " | ".join(
                f"{item[key] * 100:.2f}" for key in keys) + " |")
    lines.extend(["", "## Paired outcomes", "",
                  "Improved: baseline wrong, our method correct. Regressed: baseline correct, our method wrong.", "",
                  "| Test set | Method | Both correct | Improved | Regressed | Both wrong |",
                  "|---|---|---:|---:|---:|---:|"])
    for item in result["comparisons"]:
        counts = item["paired_outcomes"]
        lines.append(f"| {item['dataset']} | {item['method']} | " + " | ".join(
            str(counts[key]) for key in ("both_correct", "improved", "regressed", "both_wrong")) + " |")
    lines.extend(["", "## Training", "",
                  "| Method | Variant | Parent samples | Training instances | Fit time (seconds) |",
                  "|---|---|---:|---:|---:|"])
    for item in result["models"]:
        lines.append(f"| {item['method']} | {item['variant']} | {item['train_parent_count']} | "
                     f"{item['train_example_count']} | {item['fit_seconds']:.3f} |")
    lines.extend(["", "## Parameters", "", "```json",
                  json.dumps(result["parameters"], ensure_ascii=False, indent=2), "```", "",
                  "## Files", "",
                  "- `results.json`: full metrics, per-class reports, confusion matrices, data checks, inputs and timings.",
                  "- `summary.csv`: one row per test set / classifier / variant.",
                  "- `predictions.jsonl`: paired predictions, reference labels, original QA, sub-questions and predicted sub-labels.",
                  "- `baseline/<method>/`, `ours/<method>/`, and optional `hybrid/<method>/`: separate models, metrics, and predictions.",
                  "- `hybrid/<method>/oof_predictions.jsonl`: training-parent fold IDs, predicted sub-labels, and coverage features.",
                  "- `figures/`: produced only when plot_results.py is run separately; same names are overwritten.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def read_completed_result(output, method, variant):
    folder = b.code_path(output / VARIANT_FOLDERS[variant] / method)
    path = b.code_path(folder / "results.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("schema_version") != 2 or result.get("status") != "complete"
            or result.get("method") != method or result.get("variant") != variant):
        raise ValueError(f"Incomplete or incompatible result: {path}")
    if variant != "whole" and result.get("answer_source") != subquestion_method.ANSWER_SOURCE:
        raise ValueError(f"Sub-question results were not generated using original full answers; rerun this experiment: {path}")
    predictions = b.read_jsonl(folder / "predictions.jsonl")
    for row in predictions:
        if row.get("run_id") != result["run_id"] or row.get("variant") != variant or row.get("method") != method:
            raise ValueError(f"Predictions and metrics belong to different runs: {folder}")
    for name in TEST_NAMES:
        rows = [row for row in predictions if row["dataset"] == name]
        if len(rows) != result["datasets"][name]["parent_count"]:
            raise ValueError(f"Incomplete {name} predictions: {folder}")
        if data_fingerprint(rows) != result["data_signatures"][name]:
            raise ValueError(f"Reference data changed in predictions: {folder}")
        if variant != "whole" and data_fingerprint(rows, True) != result["subquestion_signatures"][name]:
            raise ValueError(f"Sub-questions or reference sub-labels changed in predictions: {folder}")
    if len(predictions) != sum(result["datasets"][name]["parent_count"] for name in TEST_NAMES):
        raise ValueError(f"Unexpected predictions in {folder}")
    return result, predictions


def summarize(run_name, methods, include_hybrid=False):
    output = result_directory(run_name)
    selected = list(dict.fromkeys(methods))
    if include_hybrid and not any(method in hybrid_method.METHODS for method in selected):
        raise ValueError("Hybrid comparison needs at least one of tfidf_lr or tfidf_svm.")
    result = {"schema_version": 2, "status": "complete", "run_name": run_name,
              "methods": selected, "parameters": {}, "models": [], "evaluations": [],
              "comparisons": [], "hybrid_comparisons": [], "source_results": [], "inputs": {},
              "protocol": {"same_training_parents": True, "same_hyperparameters_within_each_pair": True,
                           "equal_compute_budget": False, "test_used_for_selection": False,
                           "aggregation": "all FULL -> direct; all NONE -> fully_evasive; otherwise intermediate",
                           "llm_test_scores": "agreement with LLM labels, not human-gold accuracy",
                           "subquestion_answer_source": subquestion_method.ANSWER_SOURCE,
                           "decomposition_time_included": False, "sub_answers_used_as_features": False,
                           "hybrid_included": include_hybrid}}
    all_pairs = []
    reference_signatures = None
    reference_sub_signatures = None
    for method in selected:
        baseline, baseline_predictions = read_completed_result(output, method, "whole")
        proposed, our_predictions = read_completed_result(output, method, "subquestion")
        hybrid, hybrid_predictions = None, []
        if include_hybrid and method in hybrid_method.METHODS:
            hybrid, hybrid_predictions = read_completed_result(output, method, "hybrid")
            if (hybrid["parameters"] != baseline["parameters"]
                    or hybrid["data_signatures"] != baseline["data_signatures"]
                    or hybrid["subquestion_signatures"] != proposed["subquestion_signatures"]):
                raise ValueError(f"{method}: hybrid must use the same data, decompositions, labels and base hyperparameters.")
        if baseline["parameters"] != proposed["parameters"]:
            raise ValueError(f"{method}: baseline and ours used different hyperparameters; rerun the pair.")
        if baseline["data_signatures"] != proposed["data_signatures"]:
            raise ValueError(f"{method}: baseline and ours used different training/test data or overall labels.")
        if reference_signatures is not None and reference_signatures != baseline["data_signatures"]:
            raise ValueError("Selected methods used different datasets; rerun before combining them.")
        if reference_sub_signatures is not None and reference_sub_signatures != proposed["subquestion_signatures"]:
            raise ValueError("Selected methods used different sub-question inputs or reference sub-labels.")
        reference_signatures = baseline["data_signatures"]
        reference_sub_signatures = proposed["subquestion_signatures"]
        result["datasets"] = proposed["datasets"]
        result["parameters"][method] = baseline["parameters"]
        for unit in ([baseline, proposed, hybrid] if hybrid else [baseline, proposed]):
            variant = unit["variant"]
            folder = Path(VARIANT_FOLDERS[variant]) / method
            result["inputs"][str(folder)] = unit["inputs"]
            result["models"].append({**unit["model"], "model_file": str(folder / "model.joblib")})
            result["evaluations"].extend(unit["evaluations"])
            filenames = ("results.json", "predictions.jsonl", "oof_predictions.jsonl") if variant == "hybrid" else ("results.json", "predictions.jsonl")
            for filename in filenames:
                relative = folder / filename
                result["source_results"].append({"file": str(relative), "sha256":
                    hashlib.sha256(b.code_path(output / relative).read_bytes()).hexdigest()})
        for name in TEST_NAMES:
            before = [row for row in baseline_predictions if row["dataset"] == name]
            by_uid = {row["uid"]: row for row in our_predictions if row["dataset"] == name}
            if set(by_uid) != {row["uid"] for row in before}:
                raise ValueError(f"{method}/{name}: mismatched prediction UIDs.")
            after = [by_uid[row["uid"]] for row in before]
            paired = paired_predictions(method, name, before, before, after)
            all_pairs.extend(paired)
            first = next(item["metrics"]["overall"] for item in baseline["evaluations"] if item["dataset"] == name)
            second = next(item["metrics"]["overall"] for item in proposed["evaluations"] if item["dataset"] == name)
            counts = Counter(row["outcome"] for row in paired)
            result["comparisons"].append({
                "method": method, "dataset": name, "count": len(paired),
                "baseline_accuracy": first["accuracy"], "our_accuracy": second["accuracy"],
                "accuracy_delta_pp": (second["accuracy"] - first["accuracy"]) * 100,
                "baseline_macro_f1": first["macro_f1"], "our_macro_f1": second["macro_f1"],
                "macro_f1_delta_pp": (second["macro_f1"] - first["macro_f1"]) * 100,
                "paired_outcomes": {key: counts[key] for key in ("both_correct", "improved", "regressed", "both_wrong")},
            })
            if hybrid:
                hybrid_by_uid = {row["uid"]: row for row in hybrid_predictions if row["dataset"] == name}
                if set(hybrid_by_uid) != {row["uid"] for row in before}:
                    raise ValueError(f"{method}/{name}: mismatched hybrid prediction UIDs.")
                third = next(item["metrics"]["overall"] for item in hybrid["evaluations"] if item["dataset"] == name)
                result["hybrid_comparisons"].append({
                    "method": method, "dataset": name, "count": len(paired),
                    "baseline_accuracy": first["accuracy"], "subquestion_accuracy": second["accuracy"],
                    "hybrid_accuracy": third["accuracy"], "baseline_macro_f1": first["macro_f1"],
                    "subquestion_macro_f1": second["macro_f1"], "hybrid_macro_f1": third["macro_f1"],
                    "accuracy_delta_vs_baseline_pp": (third["accuracy"] - first["accuracy"]) * 100,
                    "accuracy_delta_vs_subquestion_pp": (third["accuracy"] - second["accuracy"]) * 100,
                    "macro_f1_delta_vs_baseline_pp": (third["macro_f1"] - first["macro_f1"]) * 100,
                    "macro_f1_delta_vs_subquestion_pp": (third["macro_f1"] - second["macro_f1"]) * 100,
                })
                for item in paired:
                    prediction = hybrid_by_uid[item["uid"]]
                    item.update({"hybrid_prediction": prediction["predicted_overall_label"],
                                 "hybrid_correct": prediction["predicted_overall_label"] == item["overall_label"],
                                 "hybrid_predicted_sub_labels": prediction["predicted_sub_labels"],
                                 "hybrid_coverage_features": prediction["coverage_features"]})
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    b.write_jsonl(output / "predictions.jsonl", all_pairs)
    write_summary_csv(output / "summary.csv", result["evaluations"])
    write_report(result, output / "report.md")
    b.write_json(output / "results.json", result)
    print(f"Comparison saved to {output}. Plot separately with experiments/plot_results.py --run-name {run_name}.")
    return output


def experiment_parser(variant):
    parser = argparse.ArgumentParser(description=f"Train one {VARIANT_FOLDERS[variant]} classifier and evaluate both test sets.")
    parser.add_argument("--method", required=True, choices=hybrid_method.METHODS if variant == "hybrid" else b.METHODS)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--human-test", nargs="+", default=[str(DEFAULT_HUMAN)])
    parser.add_argument("--llm-test", nargs="+", default=[str(DEFAULT_LLM)])
    parser.add_argument("--human-subquestions", nargs="+", help="Sub_questions for the human test; external labels and answer excerpts are ignored.")
    parser.add_argument("--llm-subquestions", nargs="+", help="Sub_questions for the LLM test; external labels and answer excerpts are ignored.")
    parser.add_argument("--expected-human-count", type=int, default=120)
    parser.add_argument("--expected-llm-count", type=int, default=300)
    parser.add_argument("--run-name", default="comparison_v1", help="Same run/method/variant overwrites its existing files.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--text-mode", choices=("qa", "question", "answer"), default="qa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="balanced")
    if variant == "hybrid":
        parser.add_argument("--oof-folds", type=int, default=5, help="Parent-grouped folds for training coverage features.")
        parser.add_argument("--coverage-weight", type=float, default=1.0, help="Multiplier on scaled predicted coverage features.")
    return parser


def experiment_main(variant, argv=None):
    parser = experiment_parser(variant)
    args = parser.parse_args(argv)
    try:
        run_single(args, variant)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summarize", help="Combine already completed runs without training or drawing.")
    summary.add_argument("--run-name", default="comparison_v1")
    summary.add_argument("--methods", nargs="+", choices=b.METHODS, default=list(b.METHODS))
    summary.add_argument("--include-hybrid", action="store_true", help="Also compare completed LR/SVM hybrid runs.")
    args = parser.parse_args(argv)
    try:
        summarize(args.run_name, args.methods, args.include_hybrid)
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
