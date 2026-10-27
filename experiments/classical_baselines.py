"""Whole-question baselines and shared text classifiers; one model per invocation."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted


CODE_DIR = Path(__file__).resolve().parents[1]
OVERALL_LABELS = ("direct", "intermediate", "fully_evasive")
METHODS = ("majority", "tfidf_lr", "tfidf_svm", "tfidf_nb", "lexical_rf")
SCHEMA_VERSION = 2

if __name__ == "__main__":
    sys.modules["classical_baselines"] = sys.modules[__name__]


def code_path(value):
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else CODE_DIR / path).resolve()
    if path != CODE_DIR and CODE_DIR not in path.parents:
        raise ValueError(f"Paths must stay inside {CODE_DIR}: {path}")
    return path


def read_jsonl(paths):
    """Read explicitly selected files without merging duplicate parent UIDs."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rows = []
    for path in paths:
        path = code_path(path)
        with path.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object.")
                rows.append(row)
    validate_records(rows)
    return rows


def validate_records(rows, require_overall=False):
    if not rows:
        raise ValueError("The dataset is empty.")
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index}: expected a record object.")
        for field in ("uid", "question", "answer"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Row {index}: {field} must be a nonempty string.")
        uid = row["uid"]
        if uid in seen:
            raise ValueError(f"Duplicate parent UID: {uid}")
        seen.add(uid)
        if require_overall and row.get("overall_label") not in OVERALL_LABELS:
            raise ValueError(f"{uid}: overall_label must be one of {OVERALL_LABELS}.")

###########################################################################
# Make sure there is no overlap data. Make sure the dataset quality
def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def record_signatures(rows):
    """Collect UID and normalized text fingerprints for every question and answer."""
    signatures = {"uid": set(), "question": set(), "answer": set()}
    for row in rows:
        signatures["uid"].add(row["uid"])
        question, answer = normalize(row["question"]), normalize(row["answer"])
        signatures["question"].add(hashlib.sha256(question.encode("utf-8")).hexdigest())
        signatures["answer"].add(hashlib.sha256(answer.encode("utf-8")).hexdigest())
    return {key: sorted(values) for key, values in signatures.items()}


def check_no_overlap(reference_signatures, rows, reference_name="training"):
    current = record_signatures(rows)
    overlaps = {
        key: len(set(values) & set(reference_signatures[key]))
        for key, values in current.items()
    }
    if any(overlaps.values()):
        raise ValueError(f"Evaluation data overlaps {reference_name} data: {overlaps}")
    
###########################################################################

def lexical_features(pairs):
    values = []
    for pair in pairs:
        q, a = pair["question"], pair["answer"]
        qt, at = re.findall(r"\w+", normalize(q)), re.findall(r"\w+", normalize(a))
        qs, ans = set(qt) - ENGLISH_STOP_WORDS, set(at) - ENGLISH_STOP_WORDS
        qn = set(re.findall(r"\d+(?:\.\d+)?", q))
        an = set(re.findall(r"\d+(?:\.\d+)?", a))
        values.append([
            np.log1p(len(qt)), np.log1p(len(at)), len(at) / max(1, len(qt)),
            len(qs & ans) / max(1, len(qs)), len(qs & ans) / max(1, len(qs | ans)),
            q.count("?"), len(re.findall(r"\b(?:what|why|how|when|where|whether)\b", q.lower())),
            len(qn), len(an), len(qn & an) / max(1, len(qn)),
            len(re.findall(r"[.!?]", a)),
            len(re.findall(r"\b(?:no|not|cannot|can't|won't|unable)\b", a.lower())),
            len(re.findall(r"\b(?:expect|may|might|could|uncertain|probably)\b", a.lower())),
            len(re.findall(r"\b(?:guidance|disclose|comment|later|offline)\b", a.lower())),
        ])
    return np.asarray(values, dtype=np.float64).reshape(-1, 14)


class PairFeatures(TransformerMixin, BaseEstimator):
    """Independent TF-IDF blocks for Q/A, fixed lexical features, or a constant."""

    def __init__(self, mode="qa", max_features=30000, min_df=1):
        self.mode = mode
        self.max_features = max_features
        self.min_df = min_df

    def fit(self, X, y=None):
        if self.mode not in ("qa", "question", "answer", "lexical", "constant"):
            raise ValueError(f"Unknown feature mode: {self.mode}")
        self.vectorizers_ = {}
        if self.mode in ("lexical", "constant"):
            self.fields_ = []
        else:
            self.fields_ = ["question", "answer"] if self.mode == "qa" else [self.mode]
            for field in self.fields_:
                vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2), min_df=self.min_df,
                    max_features=self.max_features, sublinear_tf=True,
                    strip_accents="unicode", dtype=np.float64,
                )
                texts = [pair[field] for pair in X]
                analyzer = vectorizer.build_analyzer()
                if not any(analyzer(text) for text in texts):
                    self.vectorizers_[field] = None
                    continue
                vectorizer.fit(texts)
                self.vectorizers_[field] = vectorizer
        return self

    def transform(self, X):
        check_is_fitted(self, "vectorizers_")
        if self.mode == "constant":
            return np.zeros((len(X), 1))
        if self.mode == "lexical":
            return lexical_features(X)
        return hstack([
            self.vectorizers_[field].transform([pair[field] for pair in X])
            if self.vectorizers_[field] is not None else csr_matrix((len(X), 1), dtype=np.float64)
            for field in self.fields_
        ], format="csr")


def build_pipeline(method, text_mode="qa", seed=42, max_features=30000,
                   min_df=1, c=1.0, alpha=1.0, n_estimators=200,
                   class_weight="balanced"):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}; choose from {METHODS}.")
    if text_mode not in ("qa", "question", "answer"):
        raise ValueError("text_mode must be qa, question, or answer.")
    if class_weight not in (None, "balanced"):
        raise ValueError("class_weight must be None or balanced.")
    if min(max_features, min_df, c, alpha, n_estimators) <= 0:
        raise ValueError("Feature limits and model hyperparameters must be positive.")
    mode = text_mode
    if method == "majority":
        mode = "constant"
        classifier = DummyClassifier(strategy="most_frequent")
    elif method == "tfidf_lr":
        classifier = LogisticRegression(
            C=c, class_weight=class_weight, max_iter=2000, random_state=seed,
        )
    elif method == "tfidf_svm":
        classifier = LinearSVC(
            C=c, class_weight=class_weight, dual="auto", max_iter=10000, random_state=seed,
        )
    elif method == "tfidf_nb":
        classifier = MultinomialNB(alpha=alpha)
    else:
        if text_mode != "qa":
            raise ValueError("lexical_rf uses both question and answer; text_mode must be qa.")
        mode = "lexical"
        classifier = RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=2,
            class_weight=class_weight, random_state=seed, n_jobs=1,
        )
    return Pipeline([
        ("features", PairFeatures(mode, max_features=max_features, min_df=min_df)),
        ("classifier", classifier),
    ])


def classification_metrics(truth, predictions, labels):
    return {
        "count": len(truth),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=list(labels),
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predictions, labels=list(labels),
                                      average="weighted", zero_division=0)),
        "label_order": list(labels),
        "class_counts": {label: truth.count(label) for label in labels},
        "confusion_matrix": confusion_matrix(truth, predictions, labels=list(labels)).tolist(),
        "classification_report": classification_report(
            truth, predictions, labels=list(labels), output_dict=True, zero_division=0,
        ),
    }


if __name__ == "__main__":
    PairFeatures.__module__ = "classical_baselines"


def prepare_examples(rows, require_labels=False):
    validate_records(rows, require_overall=require_labels)
    pairs = [{"question": row["question"], "answer": row["answer"]} for row in rows]
    labels = [row["overall_label"] for row in rows] if require_labels else []
    return pairs, labels


def fitted_model(rows, pairs, labels, method, variant, parameters):
    if method != "majority" and len(set(labels)) < 2:
        raise ValueError("Training requires at least two observed classes.")
    pipeline = build_pipeline(method, **parameters)
    pipeline.fit(pairs, labels)
    return {
        "schema_version": SCHEMA_VERSION, "method": method, "variant": variant,
        "feature_mode": pipeline.named_steps["features"].mode, "pipeline": pipeline,
        "train_parent_count": len(rows), "train_example_count": len(pairs),
        "train_class_counts": dict(Counter(labels)), "train_signatures": record_signatures(rows),
        "sklearn_version": sklearn.__version__,
    }


def train_baseline(rows, method, **parameters):
    pairs, labels = prepare_examples(rows, require_labels=True)
    model = fitted_model(rows, pairs, labels, method, "whole", parameters)
    model["answer_source"] = "original_answer"
    return model


def predict_baseline(model, rows):
    if model["variant"] != "whole":
        raise ValueError("Use subquestion_method.predict_ours for a sub-question model.")
    pairs, _ = prepare_examples(rows)
    predicted = model["pipeline"].predict(pairs).tolist()
    return [{"uid": row["uid"], "predicted_overall_label": label}
            for row, label in zip(rows, predicted)]


def evaluate_baseline(model, rows):
    validate_records(rows, require_overall=True)
    check_no_overlap(model["train_signatures"], rows)
    predictions = predict_baseline(model, rows)
    truth = [row["overall_label"] for row in rows]
    metrics = {"overall": classification_metrics(
        truth, [row["predicted_overall_label"] for row in predictions], OVERALL_LABELS)}
    for result, label in zip(predictions, truth):
        result["overall_label"] = label
    return metrics, predictions


def save_model(model, path):
    path = code_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def load_model(path):
    """Load a trusted local joblib artifact."""
    # Support artifacts saved before the module was renamed.
    sys.modules.setdefault("baselines", sys.modules[__name__])
    model = joblib.load(code_path(path))
    if not isinstance(model, dict) or model.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported baseline model format.")
    if model.get("method") not in METHODS or model.get("variant") not in ("whole", "subquestion", "hybrid"):
        raise ValueError("Unknown method or variant in model artifact.")
    return model


def write_json(path, value):
    path = code_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl(path, rows):
    path = code_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    from run_experiment import experiment_main
    experiment_main("whole")
