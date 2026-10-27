## Description

| File | Description |
|---|---|
| `data_split.py` | Splits all 16,726 raw records into four datasets while preserving existing split memberships and annotations. |
| `llm_as_annotator.py` | Completes missing overall and sub-question annotations with `Qwen/Qwen3.6-35B-A3B` while preserving existing annotations and `annotation_by`. |
| `train.jsonl` | Contains 12,000 question-answer pairs for training. |
| `validation.jsonl` | Contains 2,000 question-answer pairs for validation and model selection. |
| `test.jsonl` | Contains 2,126 question-answer pairs for testing. |
| `human_test.jsonl` | Contains 600 test records reserved for annotation by human reviewers. |
| `train_labeled.jsonl` | Stores training records with their overall and sub-question annotations. |
| `validation_labeled.jsonl` | Stores annotated validation records for model selection. |
| `test_labeled.jsonl` | Stores annotated test records for evaluating model predictions. |
| `human_test_labeled.jsonl` | Stores the human test set for human annotators to complete and verify all labels. |
| `failed.jsonl` | Records only `uid` and `reason` for samples that remain incomplete after three unsuccessful requests. |

## Packages

Set up a conda python environment and then install the required packages.

```bash
conda create -n your_env_name python=3.12 pip -y
conda activate your_env_name
conda install -c conda-forge pandas pyarrow -y
pip install vllm

# Or use pip install
pip install --upgrade pip
pip install pandas pyarrow vllm
```

## Run

Generate the four processed datasets:

```bash
cd full_data_processing
python data_split.py
```

In another terminal on the same machine, enter the same folder, activate the environment selected above, and keep this vLLM service running:

```bash
CUDA_VISIBLE_DEVICES=X vllm serve Qwen/Qwen3.6-35B-A3B --host 127.0.0.1 --port 8000 --served-model-name Qwen/Qwen3.6-35B-A3B --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.90 --language-model-only --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": true}'
```

Once the service is ready, return to the first terminal and annotate the datasets:

```bash
python llm_as_annotator.py
```

The client connects to `http://localhost:8000/v1` and saves only the final JSON annotations.