from datasets import load_dataset
from app.training.prompt_templates import build_prompt


def load_and_prepare_dataset(config: dict, tokenizer):
    dataset_mode = config.get("dataset_mode")
    dataset_split = config.get("dataset_split", "train")
    dataset_sample_size = config.get("dataset_sample_size")
    dataset_config = config.get("dataset_config")
    source_columns = config.get("source_columns", {})
    prompt_template = config.get("prompt_template", "alpaca")
    train_test_split = config.get("train_test_split", 0.2)
    filter_max_tokens = config.get("filter_max_tokens")

    if dataset_mode == "hf_dataset":
        dataset_id = config["dataset_id"]
        if dataset_config:
            dataset = load_dataset(dataset_id, dataset_config, split=dataset_split)
        else:
            dataset = load_dataset(dataset_id, split=dataset_split)

    elif dataset_mode == "local_jsonl":
        dataset = load_dataset("json", data_files=config["dataset_path"], split="train")

    else:
        raise ValueError(f"Unsupported dataset_mode: {dataset_mode}")

    if dataset_sample_size:
        dataset = dataset.shuffle(seed=config.get("training", {}).get("seed", 42))
        dataset = dataset.select(range(min(dataset_sample_size, len(dataset))))

    if source_columns:
        rename_map = {
            source_columns.get("instruction"): "instruction",
            source_columns.get("input"): "input",
            source_columns.get("output"): "output",
        }
        rename_map = {k: v for k, v in rename_map.items() if k and k in dataset.column_names}
        dataset = dataset.rename_columns(rename_map)

    eos_token = tokenizer.eos_token or ""

    def format_examples(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            row = {
                "instruction": examples["instruction"][i],
                "input": examples["input"][i] if "input" in examples else "",
                "output": examples["output"][i],
            }
            texts.append(build_prompt(row, prompt_template, eos_token=eos_token))
        return {"text": texts}

    dataset = dataset.map(format_examples, batched=True)

    if filter_max_tokens:
        dataset = dataset.filter(
            lambda x: len(tokenizer(x["text"])["input_ids"]) <= int(filter_max_tokens)
        )

    split_dataset = dataset.train_test_split(test_size=train_test_split, seed=config.get("training", {}).get("seed", 42))

    return split_dataset
