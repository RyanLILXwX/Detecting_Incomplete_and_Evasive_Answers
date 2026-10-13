#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LOCAL_PYTHON="${CODE_DIR}/.venv/bin/python"
if [[ -x "$LOCAL_PYTHON" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-$LOCAL_PYTHON}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
export PYTHONDONTWRITEBYTECODE=1

methods=(all)
run_name="comparison_v1"
check_only=false
include_hybrid=false
hybrid_args=()
common_args=()

usage() {
    cat <<'HELP'
Usage: bash experiments/run_experiments.sh --train TRAIN_JSONL [options]

  --methods all                         Run all five classifiers (default).
  --methods tfidf_lr tfidf_svm           Run selected classifiers.
  --method tfidf_lr                      Run one classifier.
  --run-name comparison_v1               Overwrite matching files in this run.
  --python python3                       Choose the Python interpreter.
  --human-test PATH [PATH ...]           Human test JSONL files or directory.
  --llm-test PATH [PATH ...]             LLM test JSONL files or directory.
  --human-subquestions PATH [PATH ...]   Human test sub_questions.
  --llm-subquestions PATH [PATH ...]     LLM test sub_questions.
  --check-only                          Validate all jobs without training.
  --include-hybrid                      Add Hybrid for selected LR/SVM methods.
  --oof-folds 5                         Hybrid training folds (default: 5).
  --coverage-weight 1.0                 Hybrid coverage feature weight.

Other training options are forwarded unchanged to each selected variant.
Hybrid uses grouped out-of-fold predictions, not reference sub-labels, as coverage features.
This script does not generate figures.
The subquestion variant pairs sub_questions with full answers; external labels and excerpts are ignored.
Data paths are relative to code/; absolute paths must also remain inside code/.
From code/, draw separately with:
  python experiments/plot_results.py --run-name comparison_v1
HELP
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --methods|--method)
            shift
            methods=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                methods+=("$1")
                shift
            done
            if [[ ${#methods[@]} -eq 0 ]]; then
                echo "Error: --methods needs at least one classifier." >&2
                exit 2
            fi
            ;;
        --run-name|--python)
            option="$1"
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: $option needs a value." >&2
                exit 2
            fi
            if [[ "$option" == --run-name ]]; then
                run_name="$2"
            else
                PYTHON_BIN="$2"
            fi
            shift 2
            ;;
        --check-only)
            check_only=true
            shift
            ;;
        --include-hybrid)
            include_hybrid=true
            shift
            ;;
        --oof-folds|--coverage-weight)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: $1 needs a value." >&2
                exit 2
            fi
            hybrid_args+=("$1" "$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            common_args+=("$1")
            shift
            ;;
    esac
done

if [[ "$include_hybrid" == false && ${#hybrid_args[@]} -gt 0 ]]; then
    echo "Error: Hybrid options require --include-hybrid." >&2
    exit 2
fi
if [[ ${#methods[@]} -eq 1 && "${methods[0]}" == all ]]; then
    methods=(majority tfidf_lr tfidf_svm tfidf_nb lexical_rf)
fi
selected=()
for method in "${methods[@]}"; do
    case "$method" in
        majority|tfidf_lr|tfidf_svm|tfidf_nb|lexical_rf) ;;
        *) echo "Error: unknown classifier '$method'; use all alone or individual names." >&2; exit 2 ;;
    esac
    duplicate=false
    for previous in ${selected[@]+"${selected[@]}"}; do
        if [[ "$previous" == "$method" ]]; then duplicate=true; fi
    done
    if [[ "$duplicate" == false ]]; then selected+=("$method"); fi
done

hybrid_selected=false
for method in "${selected[@]}"; do
    if [[ "$method" == tfidf_lr || "$method" == tfidf_svm ]]; then hybrid_selected=true; fi
done
if [[ "$include_hybrid" == true && "$hybrid_selected" == false ]]; then
    echo "Error: --include-hybrid requires tfidf_lr or tfidf_svm among selected methods." >&2
    exit 2
fi

for method in "${selected[@]}"; do
    for script in classical_baselines.py subquestion_method.py; do
        "$PYTHON_BIN" "$SCRIPT_DIR/$script" --method "$method" --run-name "$run_name" \
            ${common_args[@]+"${common_args[@]}"} --check-only
    done
    if [[ "$include_hybrid" == true && ( "$method" == tfidf_lr || "$method" == tfidf_svm ) ]]; then
        "$PYTHON_BIN" "$SCRIPT_DIR/subquestion&whole_method.py" --method "$method" --run-name "$run_name" \
            ${common_args[@]+"${common_args[@]}"} ${hybrid_args[@]+"${hybrid_args[@]}"} --check-only
    fi
done
if [[ "$check_only" == true ]]; then exit 0; fi

for method in "${selected[@]}"; do
    for script in classical_baselines.py subquestion_method.py; do
        "$PYTHON_BIN" "$SCRIPT_DIR/$script" --method "$method" --run-name "$run_name" \
            ${common_args[@]+"${common_args[@]}"}
    done
    if [[ "$include_hybrid" == true && ( "$method" == tfidf_lr || "$method" == tfidf_svm ) ]]; then
        "$PYTHON_BIN" "$SCRIPT_DIR/subquestion&whole_method.py" --method "$method" --run-name "$run_name" \
            ${common_args[@]+"${common_args[@]}"} ${hybrid_args[@]+"${hybrid_args[@]}"}
    fi
done
summary_args=()
if [[ "$include_hybrid" == true ]]; then summary_args+=(--include-hybrid); fi
"$PYTHON_BIN" "$SCRIPT_DIR/run_experiment.py" summarize --run-name "$run_name" --methods "${selected[@]}" \
    ${summary_args[@]+"${summary_args[@]}"}
