#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
PYTHON="$CODE_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi

case "${1:-help}" in
    help|-h|--help)
        cat <<'USAGE'
Run from code/; all project paths are relative to code/.

Setup (Python 3.11+ on Apple Silicon):
  python3.11 -m venv .venv
  PIP_CACHE_DIR="$PWD/.cache/qlora/pip" .venv/bin/python -m pip install -r finetunning/QLoRA/requirements.txt

Download once (several GB; only this command downloads weights):
  bash finetunning/QLoRA/run.sh download

Check completed annotations and split overlap without MLX:
  bash finetunning/QLoRA/run.sh check --variant whole
  bash finetunning/QLoRA/run.sh check --variant joint

Train with matched parents, steps and optimizer updates:
  bash finetunning/QLoRA/run.sh train --variant whole
  bash finetunning/QLoRA/run.sh train --variant joint

Select on all 300 dev examples, then evaluate (repeat for joint):
  bash finetunning/QLoRA/run.sh evaluate --run results/qlora/qwen3_4b_whole --split dev --select-best
  bash finetunning/QLoRA/run.sh evaluate --run results/qlora/qwen3_4b_whole --split human_test
  bash finetunning/QLoRA/run.sh evaluate --run results/qlora/qwen3_4b_whole --split llm_test

Predict without reference labels (input requires uid, question, answer):
  bash finetunning/QLoRA/run.sh evaluate --run results/qlora/qwen3_4b_joint --split predict --input data/my_questions.jsonl

Training uses completed *_annotated data, never the unannotated templates.
Each training row must be complete; joint also requires exact sub-answer quotes.
Human tests require all 120 overall labels. Tests need no reference sub-questions.
Strict overlap checks match baselines: UID, normalized question and every answer.
Rerunning evaluation replaces same-name results. Training requires --overwrite
to replace an existing run; it starts fresh and does not resume optimizer state.

Defaults: 4-bit base, last 8 layers, rank 8, MLX scale 16, batch 1, accumulation 4,
2700 microbatches (= 675 optimizer updates), LR 1e-5, sequence length 4096.
Loss covers assistant output only; no truncation or automatic sample dropping.
Whole/joint supervision token counts and training time differ and are recorded.
MLX quantized LoRA is used; this is not a CUDA/bitsandbytes NF4 reproduction.
Use '--help' after any command for options. No model training runs automatically.
USAGE
        ;;
    download)
        shift
        exec "$PYTHON" "$SCRIPT_DIR/download_model.py" "$@"
        ;;
    check)
        shift
        exec "$PYTHON" "$SCRIPT_DIR/train.py" --check-only "$@"
        ;;
    train)
        shift
        exec "$PYTHON" "$SCRIPT_DIR/train.py" "$@"
        ;;
    evaluate)
        shift
        exec "$PYTHON" "$SCRIPT_DIR/evaluate.py" "$@"
        ;;
    *)
        echo "Unknown command: $1. Use: bash finetunning/QLoRA/run.sh help" >&2
        exit 2
        ;;
esac
