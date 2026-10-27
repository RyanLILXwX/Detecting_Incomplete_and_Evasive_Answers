"""Draw saved comparison results without training any model."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CODE_DIR / "results"
TEST_NAMES = ("human_test", "llm_test")
VARIANTS = ("whole", "subquestion", "hybrid")
OVERALL_LABELS = ("direct", "intermediate", "fully_evasive")
DISPLAY_METHODS = {
    "majority": "Majority", "tfidf_lr": "TF-IDF + LR", "tfidf_svm": "TF-IDF + SVM",
    "tfidf_nb": "TF-IDF + NB", "lexical_rf": "Lexical + RF",
}
DISPLAY_VARIANTS = {"whole": "Baseline (whole question)", "subquestion": "Sub-questions (fixed rule)",
                    "hybrid": "Hybrid (whole QA + coverage)"}


def code_path(value):
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else CODE_DIR / path).resolve()
    if path != CODE_DIR and CODE_DIR not in path.parents:
        raise ValueError(f"Paths must stay inside {CODE_DIR}: {path}")
    return path


def plot_comparisons(result, output):
    """Save static comparison charts and confusion matrices without a GUI."""
    output = code_path(output)
    with tempfile.TemporaryDirectory(prefix=".plot_cache_", dir=output) as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import PercentFormatter
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous
        methods = result["methods"]
        lookup = {(item["dataset"], item["method"], item["variant"]): item["metrics"]["overall"]
                  for item in result["evaluations"]}
        variants = [variant for variant in VARIANTS if any(item["variant"] == variant for item in result["evaluations"])]
        colors = {"whole": "#64748b", "subquestion": "#137c8b", "hybrid": "#b45309"}
        x = np.arange(len(methods))
        chart_dir = code_path(output / "figures")
        chart_dir.mkdir(exist_ok=True)
        artifacts = []

        def save(figure, name):
            for extension in ("png", "pdf"):
                path = code_path(chart_dir / f"{name}.{extension}")
                figure.savefig(path, dpi=180, bbox_inches="tight")
                artifacts.append(str(path.relative_to(output)))
            plt.close(figure)

        def bars(axis, name, metric=None, label=None):
            width = 0.72 / len(variants)
            for index, variant in enumerate(variants):
                values = []
                for method in methods:
                    score = lookup.get((name, method, variant))
                    values.append(np.nan if score is None else score[metric] if metric else
                                  score["classification_report"][label]["f1-score"])
                rectangles = axis.bar(x + (index - (len(variants) - 1) / 2) * width, values, width,
                                      color=colors[variant], label=DISPLAY_VARIANTS[variant])
                axis.bar_label(rectangles, labels=[f"{v:.1%}" if np.isfinite(v) else "" for v in values],
                               padding=3, fontsize=8)
            axis.set_xticks(x, [DISPLAY_METHODS[m] for m in methods], rotation=15, ha="right")
            axis.set_ylim(0, 1.14)
            axis.set_yticks(np.linspace(0, 1, 6))
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)

        with plt.rc_context({"font.size": 10, "axes.titlesize": 12}):
            fig, axes = plt.subplots(2, 2, figsize=(max(12, len(methods) * 2.8), 9),
                                     squeeze=False, layout="constrained")
            for row, name in enumerate(TEST_NAMES):
                count = result["datasets"][name]["parent_count"]
                display = "Human test" if name == "human_test" else "LLM-labeled test"
                for col, metric in enumerate(("accuracy", "macro_f1")):
                    bars(axes[row, col], name, metric=metric)
                    axes[row, col].set_title(f"{display} (n={count}) — {metric.replace('_', ' ').title()}")
            fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
                       bbox_to_anchor=(0.5, 1.055), ncol=len(variants), frameon=False)
            save(fig, "overall_comparison")

            fig, axes = plt.subplots(2, 3, figsize=(max(15, len(methods) * 3.4), 9),
                                     squeeze=False, layout="constrained")
            for row, name in enumerate(TEST_NAMES):
                for col, label in enumerate(OVERALL_LABELS):
                    bars(axes[row, col], name, label=label)
                    support = result["datasets"][name]["overall_class_counts"].get(label, 0)
                    axes[row, col].set_title(f"{name}: {label}\nF1; support={support}")
            fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
                       bbox_to_anchor=(0.5, 1.055), ncol=len(variants), frameon=False)
            save(fig, "per_class_f1")

            for name in TEST_NAMES:
                fig, axes = plt.subplots(len(methods), len(variants), figsize=(5 * len(variants), 3.8 * len(methods)),
                                         squeeze=False, layout="constrained")
                maximum = max(max(max(row) for row in lookup[(name, m, v)]["confusion_matrix"])
                              for m in methods for v in variants if (name, m, v) in lookup)
                for row, method in enumerate(methods):
                    for col, variant in enumerate(variants):
                        axis = axes[row, col]
                        if (name, method, variant) not in lookup:
                            axis.set_axis_off()
                            continue
                        matrix = np.asarray(lookup[(name, method, variant)]["confusion_matrix"])
                        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, maximum))
                        for i in range(3):
                            for j in range(3):
                                axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                                          color="white" if matrix[i, j] > maximum * 0.55 else "black")
                        axis.set_xticks(range(3), OVERALL_LABELS, rotation=20, ha="right")
                        axis.set_yticks(range(3), OVERALL_LABELS)
                        axis.set_xlabel("Predicted label")
                        axis.set_ylabel("Reference label")
                        axis.set_title(f"{DISPLAY_METHODS[method]} — {variant}")
                fig.suptitle(f"{name}: confusion matrices (counts)")
                save(fig, f"confusion_{name}")
        return artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="comparison_v1")
    args = parser.parse_args(argv)
    try:
        if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
            raise ValueError("--run-name must be a single folder name.")
        output = code_path(RESULTS_DIR / "baselines" / args.run_name)
        if RESULTS_DIR.resolve() not in output.parents:
            raise ValueError("Results must remain inside results/.")
        result = json.loads(code_path(output / "results.json").read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            raise ValueError("The result summary is incomplete. Run experiments/run_experiment.py summarize first.")
        for source in result.get("source_results", []):
            path = code_path(output / source["file"])
            if output not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError("Saved results have changed. Run experiments/run_experiment.py summarize again before drawing.")
        artifacts = plot_comparisons(result, output)
        print(f"Saved {len(artifacts)} figures to {output / 'figures'} (same names overwritten).")
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
