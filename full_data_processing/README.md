## Description

| File | Description |
|---|---|
| `data_split.py` | Splits 16,726 raw records from `first_round_data/` into four datasets, preserving existing split memberships and annotations. |
| `llm_as_annotator.py` | Completes missing annotations for train, validation, and test with `Qwen/Qwen3.6-35B-A3B`, preserving existing nonempty annotations and `annotation_by`. |
| `data_clean.py` | Reads failed UIDs and removes the first matching record in train, validation, then test. Preserves other records and keeps the failure log. |
| `train.jsonl` | Contains 12,000 training records before annotation and cleanup. |
| `validation.jsonl` | Contains 2,000 validation records before annotation and cleanup. |
| `test.jsonl` | Contains 2,126 test records before annotation and cleanup. |
| `human_test.jsonl` | Contains 600 records in the separately reserved test split before cleanup. |
| `train_labeled.jsonl` | Contains 11,920 annotated training records after review and cleanup. |
| `validation_labeled.jsonl` | Contains 1,994 annotated validation records for model selection. |
| `test_labeled.jsonl` | Contains 2,118 annotated test records for evaluating predictions. |
| `human_test_labeled.jsonl` | Contains 597 annotated records maintained separately from the current annotation script; 567 carry `human` and 30 carry `llm` in `annotation_by`. |
| `failed.jsonl` | Retains 5 historical failures with `uid` and `reason`; none of these UIDs remains in the labeled datasets. |


We removed duplicate records, records with failed annotations, and entries without a reliably identifiable question to improve data quality.

| Dataset | Current records |
|---|---:|
| `train` | 11,920 |
| `validation` | 1,994 |
| `test` | 2,118 |
| `human_test` | 597 |
| **Total** | **16,629** |

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

Use the existing reviewed labeled files for training. The commands below rebuild the initial processing outputs and do not reproduce the later deduplication and semantic corrections. Rerunning `data_split.py` overwrites the initial splits; rerunning `llm_as_annotator.py` overwrites the reviewed train, validation, and test outputs and `failed.jsonl`.

Generate the four initial datasets:

```bash
cd full_data_processing
python data_split.py
```

In another terminal on the same machine, enter the same folder, activate the environment selected above, and keep this vLLM service running:

```bash
CUDA_VISIBLE_DEVICES=X vllm serve Qwen/Qwen3.6-35B-A3B --host 127.0.0.1 --port 8000 --served-model-name Qwen/Qwen3.6-35B-A3B --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.90 --language-model-only --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": true}'
```

Once the service is ready, return to the first terminal and annotate train, validation, and test:

```bash
python llm_as_annotator.py
```

The client connects to `http://localhost:8000/v1` and saves only the final JSON annotations. After annotation finishes, remove failed records:

```bash
python data_clean.py
```

The cleanup scans only train, validation, and test, preserving the remaining records and their order. It does not clear `failed.jsonl`. Neither the annotation script nor the cleanup script writes `human_test_labeled.jsonl`.