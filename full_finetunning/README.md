## Description

| File | Description |
|---|---|
| `data_tokenization.py` | Reads the four labeled datasets, builds task-specific inputs, and tokenizes them without truncation. |
| `model_training.py` | Loads a local Qwen3 model with a three-class classification head, updates all parameters, and saves the best validation checkpoint. |
| `evaluation.py` | Aggregates request predictions and evaluates both tasks at the original question-answer level. |
| `model_training_pipeline.py` | Runs data preparation, training, best-checkpoint loading, and final evaluation on both test sets. |
| `training_config.yaml` | Configures the task, model and data paths, output directory, and training parameters. |
| `run.json` | Stores a snapshot of the training configuration. |
| `evaluation_history.json` | Stores classification metrics, training loss, and validation loss for every validation round. |
| `selected_checkpoint.json` | Stores the metrics and losses from the best validation round. |
| `best_checkpoint/` | Stores the best model weights, model configuration, and tokenizer. |
| `test_metric.json` | Stores separate classification metrics for `test` and `human_test`, without losses. |

All training outputs are saved under `output_dir`. Classification metrics are `accuracy`, `macro_f1`, `weighted_f1`, `per_class` (precision, recall, F1, and support), and `sample_count`.

## Packages

Set up a conda Python environment and install the required packages:

```bash
conda create -n your_env_name python=3.12 pip -y
conda activate your_env_name
pip install torch "transformers>=4.51.0" pyyaml huggingface_hub
```

Use a CUDA-enabled PyTorch installation compatible with the server. The default `bf16: true` setting requires a GPU with BF16 support.

## Run

From the project root, enter this folder and download the unquantized model with the official Hugging Face CLI:

```bash
cd full_finetunning
# Download Qwen3-4B, or use the alternative command below for Qwen3-1.7B.
hf download Qwen/Qwen3-4B
# Alternative: download Qwen3-1.7B.
hf download Qwen/Qwen3-1.7B

# Copy the local model directory printed by the chosen download command. Set this directory as the model value in training_config.yaml.
```

Edit `training_config.yaml` to choose one task:

- `whole_qa`: classify the original question and complete answer as `direct`, `intermediate`, or `fully_evasive`.
- `request`: classify each sub-question with the complete answer as `FULL`, `PARTIAL`, or `NONE`; aggregate all `FULL` predictions to `direct`, all `NONE` predictions to `fully_evasive`, and every other combination to `intermediate`.

Run the tasks separately, using a new or empty `output_dir` for each experiment. 

```bash
cd full_finetunning
CUDA_VISIBLE_DEVICES=0 python model_training_pipeline.py
```