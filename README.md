# Detecting Incomplete and Evasive Answers

Classify answer completeness in question–answer pairs and compare direct classification of the full question with classification based on decomposed subquestions.

## 1. Project Structure

```text
code/
├── annotation/
│   ├── 0_select_data.ipynb
│   └── 1_LLM_annotate.py
├── experiments/
│   ├── classical_baselines.py
│   ├── subquestion_method.py
│   ├── subquestion&whole_method.py
│   ├── run_experiment.py
│   ├── plot_results.py
│   └── run_experiments.sh
├── finetunning/
│   └── QLoRA/
│       ├── common.py
│       ├── download_model.py
│       ├── train.py
│       ├── evaluate.py
│       ├── requirements.txt
│       └── run.sh
├── data/
│   ├── raw/
│   │   └── evasionbench_17k_eva4b_labels_dedup.parquet
│   ├── human_annotations/
│   │   ├── gold_test_120total/
│   │   ├── gold_standard_12example/
│   │   ├── verified_dev/
│   │   └── verified_test/
│   └── LLM_annotations/
│       ├── llm_dev_test_300/
│       ├── llm_dev_test_300_annotated/
│       ├── llm_train_2700/
│       └── llm_train_2700_annotated/
├── model/
│   ├── gpt-oss-20b/
│   └── llama.cpp/
└── results/
    ├── baselines/
    └── qlora/

```

The QLoRA download command also creates `code/model/Qwen3-4B-4bit/`. The directory name `finetunning` follows the existing project spelling. All script paths for data, models, caches, and results must remain inside `code/`.

## 2. File Descriptions

| File | Purpose |
|---|---|
| `annotation/0_select_data.ipynb` | Selects 120 human test samples, 300 LLM dev samples, 300 LLM test samples, 12 standard examples, and 2,700 training samples. Checks duplicates and split overlap, and reuses existing outputs without overwriting them. |
| `annotation/1_LLM_annotate.py` | Annotates data with a local model using the 12 consensus examples. Dev/test receive overall labels and rationales; training data also receive decomposition and subquestion annotations. Invalid output can trigger up to 3 retries, or 4 calls in total. |
| `experiments/classical_baselines.py` | Extracts features from the original question and answer to predict an overall label. Also provides shared classifiers, data I/O, and metrics. |
| `experiments/subquestion_method.py` | Pairs each subquestion with the original full answer for training and prediction, then aggregates predicted labels. Provides separate helpers for optional excerpt analysis. |
| `experiments/subquestion&whole_method.py` | Trains a Hybrid LR/SVM classifier using whole-QA TF-IDF and predicted request coverage, with grouped out-of-fold training features. |
| `experiments/run_experiment.py` | Provides shared logic for individual experiments, data checks, and result saving. Its `summarize` command combines completed experiments. |
| `experiments/plot_results.py` | Reads combined results and saves accuracy/F1 comparisons and confusion matrices as PNG/PDF files. |
| `experiments/run_experiments.sh` | Runs direct and subquestion variants for selected or all classifiers. `--include-hybrid` adds Hybrid for LR/SVM. Combines results; plotting runs separately. |
| `finetunning/QLoRA/common.py` | Shared data validation, prompts, label schemas, token encoding, metrics, and model loading for fine-tuning. |
| `finetunning/QLoRA/download_model.py` | Downloads a pinned revision of the MLX Qwen3-4B 4-bit weights into `code/model/`. |
| `finetunning/QLoRA/train.py` | Trains a `whole` or `joint` LoRA adapter, computes loss only on assistant output, and saves checkpoints and training records. |
| `finetunning/QLoRA/evaluate.py` | Selects a checkpoint by dev Macro-F1, evaluates both test sets, and supports prediction on unlabeled question–answer pairs. |
| `finetunning/QLoRA/requirements.txt` | Dependencies for MLX fine-tuning. |
| `finetunning/QLoRA/run.sh` | Unified entry point for model download, data checks, training, and evaluation. |


Overall labels are `direct` (complete answer), `intermediate` (partial answer), and `fully_evasive` (no substantive requested information). Subquestion labels are `FULL`, `PARTIAL`, and `NONE`. The classical subquestion method uses a fixed aggregation rule: all `FULL` → `direct`; all `NONE` → `fully_evasive`; any other combination → `intermediate`.

| Data Directory/File | Purpose |
|---|---|
| `gold_test_120total/batch_01.jsonl` through `batch_06.jsonl` | Human test set, with 20 samples per batch. Reviewers provide overall labels and rationales. |
| `gold_standard_12example/annotation_template.jsonl` | Team annotation template covering overall labels, subquestions, answer excerpts, and subquestion annotations. |
| `gold_standard_12example/consensus.jsonl` | Consensus annotations agreed through team discussion, used as LLM examples and included in training. |
| `llm_dev_test_300/dev_300.jsonl`, `test_300.jsonl` | Selected dev/test samples awaiting annotation. |
| `llm_train_2700/train_2700.jsonl` | The 2,700 selected training samples, including the 12 standard examples. |
| `llm_train_2700/llm_input_2688.jsonl` | The remaining training samples requiring LLM annotation after excluding the 12 standard examples. |
| Both `*_annotated/` directories | LLM annotation outputs. Each split is exported as `dev_300.jsonl`, `test_300.jsonl`, or `train_2700.jsonl` only after all its records succeed. |
| `verified_dev/`, `verified_test/` | Reserved for annotations verified by humans. Scripts do not use these directories automatically; specify their input files explicitly. |

The 120 human test samples and standard examples are sampled evenly across original labels. Dev/test and training samples follow the original class proportions among eligible candidates. The historical exclusion list has been removed; new sampling still checks duplicates and overlap between current splits. Original labels guide sampling only; experiments use the completed annotations as supervision.

## 3. Usage

Start from the project root, enter `code/`, and run all subsequent commands there. Complete the required annotations before training or evaluation; sampling templates with missing labels cannot be used directly in experiments.

### 3.1 Environment and Data Selection

Use Python 3.11 or newer:

```bash
cd code
python3.11 -m venv .venv
source .venv/bin/activate
export PIP_CACHE_DIR="$PWD/.cache/pip"
python -m pip install numpy pandas pyarrow scikit-learn scipy joblib matplotlib jupyterlab ipykernel

export JUPYTER_CONFIG_DIR="$PWD/.cache/jupyter/config"
export JUPYTER_DATA_DIR="$PWD/.cache/jupyter/data"
export JUPYTER_RUNTIME_DIR="$PWD/.cache/jupyter/runtime"
python -m jupyter lab annotation/0_select_data.ipynb
```

Run the notebook cells in order. Existing data directories reuse their saved files; changing the requested sample count does not automatically overwrite existing annotations.

### 3.2 LLM Annotation

Complete `consensus.jsonl` first. For training annotations, each `sub_answers` entry must be a list of verbatim excerpts from the original answer. A value of `null` means the entry is unfinished; use `[]` when there is no relevant passage.

Prepare the default model file at `model/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf` and the server executable at `model/llama.cpp/llama-server` before running annotation.

```bash
python annotation/1_LLM_annotate.py --dataset dev_test
python annotation/1_LLM_annotate.py --dataset train
```

Training annotation reuses the 12 consensus annotations directly and calls the model for the remaining 2,688 samples. Rerunning with the same configuration skips both recorded successes and failures; failure details are saved in `failures.jsonl`. Use a new `--output-dir` when changing inputs, examples, or model settings. Review LLM outputs according to the experiment's human verification plan.

### 3.3 Classical Machine Learning Experiments

Available methods: `majority`, `tfidf_lr`, `tfidf_svm`, `tfidf_nb`, and `lexical_rf`.

Set the data paths for the current terminal session:

```bash
TRAIN_FILE=data/LLM_annotations/llm_train_2700_annotated/train_2700.jsonl
HUMAN_TEST_DIR=data/human_annotations/gold_test_120total
LLM_TEST_FILE=data/LLM_annotations/llm_dev_test_300_annotated/test_300.jsonl
```

Run the whole-question variant with logistic regression. Add `--check-only` to validate data without training:

```bash
python experiments/classical_baselines.py \
  --method tfidf_lr --train "$TRAIN_FILE" \
  --human-test "$HUMAN_TEST_DIR" --llm-test "$LLM_TEST_FILE" \
  --run-name comparison_v1
```

The subquestion variant requires precomputed test subquestions; **human labels for test subquestions and answer excerpts are not required**. Load subquestions by UID using `--human-subquestions` and `--llm-subquestions`. Each external record must contain at least `uid`, the original `question`, and a nonempty `sub_questions` list. If an `answer` field is included, it must match the original answer exactly. Records must cover exactly the corresponding test set. Derive the requests from the question without using test labels. External annotation labels and `sub_answers` are ignored; the overall reference labels remain unchanged.

The main classifier receives `subquestion + original full answer -> FULL / PARTIAL / NONE`. Every subquestion from a parent is paired with the same complete answer. It never uses `sub_answers`, reference labels, or rationales as input features. Missing, unfinished, or changed excerpts therefore do not affect normal training or prediction. The default `--text-mode qa` uses both texts; `question` and `answer` are optional feature ablations.

`sub_answers()` remains an explicit evidence validator, and `prepare_oracle_examples()` can build excerpt-based pairs for separate oracle or evidence-supervision experiments. Only these opt-in helpers require aligned lists of exact, contiguous quotes: `[]` means no relevant passage and `null` means unfinished. They are not called by the main training/evaluation APIs; no oracle model is trained automatically.

Evaluation accepts subquestion reference labels for either every row or no rows. Missing `annotations`, `null`, and `[]` count as unlabeled. Mixing labeled and unlabeled rows raises an error before prediction; nonempty annotations must contain valid, aligned labels. Overall metrics always use the original overall reference labels, with optional subquestion metrics reported separately.

`FixedAggregator.predict(label_sequences)` applies the fixed completeness rule. The subquestion-only model stores it under `model["aggregator"]`, providing an extension point for a future fitted aggregator with the same interface. Aggregation receives grouped predicted subquestion labels, not reference annotations. The separate Hybrid experiment below learns an overall classifier from whole-QA text and predicted coverage; it does not replace the fixed-rule experiment.

For example, an external test-input record can have this structure:

```json
{
  "uid": "example_uid",
  "question": "How much did revenue grow, and how did costs change?",
  "answer": "Revenue grew by 10%. We have no other updates.",
  "sub_questions": ["How much did revenue grow?", "How did costs change?"]
}
```

The following two `data/inference/` paths are examples of files you must prepare; the current scripts do not generate them automatically:

```bash
HUMAN_SUBQUESTIONS=data/inference/human_test_subquestions.jsonl
LLM_SUBQUESTIONS=data/inference/llm_test_subquestions.jsonl

python experiments/subquestion_method.py \
  --method tfidf_lr --train "$TRAIN_FILE" \
  --human-test "$HUMAN_TEST_DIR" --llm-test "$LLM_TEST_FILE" \
  --human-subquestions "$HUMAN_SUBQUESTIONS" --llm-subquestions "$LLM_SUBQUESTIONS" \
  --run-name comparison_v1
```

Run both variants in batch and combine their results:

```bash
bash experiments/run_experiments.sh \
  --methods all --train "$TRAIN_FILE" \
  --human-test "$HUMAN_TEST_DIR" --llm-test "$LLM_TEST_FILE" \
  --human-subquestions "$HUMAN_SUBQUESTIONS" --llm-subquestions "$LLM_SUBQUESTIONS" \
  --run-name comparison_v1

python experiments/plot_results.py --run-name comparison_v1
```

To run selected classifiers, replace `--methods all` with, for example, `--methods tfidf_lr tfidf_svm`. If the two variants were run separately, combine the completed model pairs manually:

```bash
python experiments/run_experiment.py summarize --run-name comparison_v1 --methods tfidf_lr
```

Results are saved under `results/baselines/comparison_v1/`, including models, `results.json`, `summary.csv`, `predictions.jsonl`, and `report.md`. Saved predictions include the original full QA, subquestions, and predicted subquestion labels; model/result metadata records `answer_source: "original_answer"`. Previously trained excerpt-based subquestion models and their results must be regenerated. Models without an explicit answer-source tag are also rejected. Optional `sub_answers` do not contribute to main-input fingerprints or comparison reports. Plots are saved separately in `figures/`. Rerunning an experiment overwrites matching result files. The current classical experiment scripts use the supplied hyperparameters and do not tune them automatically on dev.

#### Hybrid: Whole QA and Predicted Request Coverage

The Hybrid experiment supports `tfidf_lr` and `tfidf_svm`, using the same classifier family and base hyperparameters for its subquestion classifier and overall classifier. It requires `--text-mode qa`. Its overall input concatenates independent question/answer TF-IDF blocks with four numeric features: `full_ratio`, `partial_ratio`, `none_ratio`, and `num_sub_questions`. For predictions `[FULL, FULL, NONE]`, these are `[2/3, 0, 1/3, 3]`. Coverage features use a scaler fitted on training data only, without centering, and an optional positive `--coverage-weight` multiplier (default `1.0`).

Training coverage is generated with parent-grouped out-of-fold (OOF) predictions, with five folds by default. All subquestions from one parent stay together. Parents sharing a normalized original question or answer are linked into the same group, including transitive matches. Each fold fits its own subquestion TF-IDF and classifier using only its training parents. The held-out parents' predicted sublabels produce their coverage features. Reference sublabels supervise the subquestion models but are never directly used to construct the overall classifier's coverage features.

The overall classifier learns from full-QA TF-IDF plus these OOF features, using the original overall labels. A final subquestion model is then fitted on all training parents and supplies predicted coverage at evaluation. Test annotations and `sub_answers` never enter the prediction features. Evaluation still requires precomputed subquestions; optional reference sublabels must be available for all rows or none.

Run Hybrid alone using the data-path variables defined above. **Quote the script path because its filename contains `&`.**

```bash
python 'experiments/subquestion&whole_method.py' \
  --method tfidf_lr --train "$TRAIN_FILE" \
  --human-test "$HUMAN_TEST_DIR" --llm-test "$LLM_TEST_FILE" \
  --human-subquestions "$HUMAN_SUBQUESTIONS" --llm-subquestions "$LLM_SUBQUESTIONS" \
  --oof-folds 5 --coverage-weight 1.0 --run-name hybrid_v1
```

Or run and compare all three variants for both LR and SVM:

```bash
bash experiments/run_experiments.sh \
  --methods tfidf_lr tfidf_svm --include-hybrid \
  --train "$TRAIN_FILE" --human-test "$HUMAN_TEST_DIR" --llm-test "$LLM_TEST_FILE" \
  --human-subquestions "$HUMAN_SUBQUESTIONS" --llm-subquestions "$LLM_SUBQUESTIONS" \
  --oof-folds 5 --run-name hybrid_v1

python experiments/plot_results.py --run-name hybrid_v1
```

Add `--check-only` to either training command to validate inputs and the fold plan without fitting models. `--methods all --include-hybrid` keeps all five direct/subquestion comparisons and adds Hybrid for LR/SVM only. To combine separately completed runs, use `python experiments/run_experiment.py summarize --run-name hybrid_v1 --methods tfidf_lr tfidf_svm --include-hybrid`.

Hybrid outputs go to `results/baselines/hybrid_v1/hybrid/<method>/`. Along with `model.joblib`, `results.json`, `summary.csv`, and test `predictions.jsonl`, the directory contains `oof_predictions.jsonl` with each training UID, fold ID, predicted sublabels, and coverage features. Test predictions also retain the fixed-rule aggregation as a diagnostic alongside the learned overall prediction. The combined report includes three-way accuracy/Macro-F1 tables, and the plotting script includes a third series.

Training time includes every OOF fit, the final subquestion fit, and the overall fit. Hybrid `train_example_count` refers to the overall classifier's parent examples; `training_details` separately records fold counts, subquestion training instances, and stage timings. These comparisons use matched data and base hyperparameters, not equal compute budgets. Select settings on dev, not the test sets. This experiment does not implement the separate 4B multitask fine-tuning proposal.

### 3.4 Qwen3-4B Quantized LoRA Fine-Tuning

The current implementation targets Apple Silicon Macs and uses MLX quantized LoRA. The `whole` variant learns to map the original question and answer to an overall label and rationale. The `joint` variant learns to generate subquestions, verbatim answer excerpts, subquestion annotations, and an overall judgment. Both variants receive only the original question and answer at prediction time.

```bash
python -m pip install -r finetunning/QLoRA/requirements.txt
bash finetunning/QLoRA/run.sh download

bash finetunning/QLoRA/run.sh check --variant whole
bash finetunning/QLoRA/run.sh check --variant joint

bash finetunning/QLoRA/run.sh train --variant whole
bash finetunning/QLoRA/run.sh train --variant joint
```

By default, training and dev data are read from the two `*_annotated/` directories. Test inputs are used only for split isolation checks before training. Defaults are batch size 1, gradient accumulation 4, 2,700 microbatches (675 optimizer updates), learning rate `1e-5`, maximum sequence length 4,096, and LoRA rank 8. Both variants use the same sample count and number of updates; actual supervised token counts and training times are recorded separately.

After training, select a checkpoint on dev before evaluating the test sets. The following example uses `whole`; replace the final `whole` in the run path with `joint` to evaluate the other variant:

```bash
bash finetunning/QLoRA/run.sh evaluate \
  --run results/qlora/qwen3_4b_whole --split dev --select-best

bash finetunning/QLoRA/run.sh evaluate \
  --run results/qlora/qwen3_4b_whole --split human_test

bash finetunning/QLoRA/run.sh evaluate \
  --run results/qlora/qwen3_4b_whole --split llm_test
```

Test sets require only overall labels, with no reference subquestions. Metrics and per-record predictions are saved under the corresponding `results/qlora/qwen3_4b_whole/` or `qwen3_4b_joint/` directory. Reports include accuracy, Macro-F1, per-class metrics, and confusion matrices. LLM test scores measure agreement with its reference annotations; human test scores are reported separately.

Evaluation overwrites matching output files. To retrain into an existing run directory, add `--overwrite`; this starts a fresh training run rather than resuming a checkpoint. Inputs that exceed the sequence limit or have incomplete labels cause an error; training samples are not automatically truncated or dropped.

To predict new unlabeled question–answer pairs, prepare a JSONL file containing `uid`, `question`, and `answer`, then run:

```bash
bash finetunning/QLoRA/run.sh evaluate \
  --run results/qlora/qwen3_4b_joint --split predict --input data/my_questions.jsonl
```
