from data_tokenization import load_datasets, build_classification_inputs, tokenize_classification_inputs
from evaluation import evaluate_test_sets
from model_training import load_training_config, load_model_and_tokenizer, train_model, write_json

if __name__ == "__main__":
    config, paths = load_training_config()
    datasets = load_datasets(paths, config["task_type"])
    model, tokenizer = load_model_and_tokenizer(config, paths)
    classification_inputs_by_split = {split: build_classification_inputs(rows, config["task_type"], split) for split, rows in datasets.items()}
    encoded_datasets = {split: tokenize_classification_inputs(classification_inputs, tokenizer, config["max_length"]) for split, classification_inputs in classification_inputs_by_split.items()}
    train_model(model, tokenizer, datasets, encoded_datasets, config, paths)
    model, tokenizer = load_model_and_tokenizer(config, paths, best_checkpoint=True)
    test_metrics = evaluate_test_sets(model, tokenizer, datasets, encoded_datasets, config)
    write_json(paths["output_dir"] / "test_metric.json", test_metrics)