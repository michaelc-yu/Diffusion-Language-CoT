import json
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
# from training.corruptions import apply_corruptions
import yaml
from transformers import AutoTokenizer
import torch

# Load and preprocess CoT datasets (GSM8K, SVAMP, StrategyQA)

class CoTDataset(Dataset):
    def __init__(self, data_path, tokenizer: PreTrainedTokenizer, apply_corruption=False, corruption_config=None, max_length=64):
        with open(data_path, 'r') as f:
            self.examples = [json.loads(line) for line in f]
        
        print(f"{data_path} first 3 examples:")
        print(self.examples[:3])

        self.tokenizer = tokenizer
        self.apply_corruption = apply_corruption
        self.corruption_config = corruption_config or {}
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        input_text = ex["input"]
        target_text = ex["target"]

        # Tokenize
        tokenized = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Flatten tensors (from shape [1, T] → [T])
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "clean_target_text": target_text,
        }



def get_dataloaders(config, tokenizer):
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "clean_target_text": [b["clean_target_text"] for b in batch]
        }

    """
    Config example:
    {
        "dataset_name": "gsm8k",
        "train_path": "data/gsm8k/train.jsonl",
        "val_path": "data/gsm8k/val.jsonl",
        "batch_size": 8,
        "corruptions": {
            "masking": 0.2,
            "shuffle": 0.1
        }
    }
    """
    print(f"get_dataloaders config: {config}")

    train_dataset = CoTDataset(
        data_path=config["train_path"],
        tokenizer=tokenizer,
        apply_corruption=False, # set this to true later
        corruption_config=config.get("corruptions")
    )

    val_dataset = CoTDataset(
        data_path=config["val_path"],
        tokenizer=tokenizer,
        apply_corruption=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader



def main():
    with open("configs/data.yaml", "r") as f:
        data_config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader = get_dataloaders(data_config, tokenizer)

    print("Peek inside train loader DataLoader object")
    for batch in train_loader:
        print("Input ids:", batch["input_ids"][0])
        print("Attention mask:", batch["attention_mask"][0])
        print("Clean target:", batch["clean_target_text"][0])
        break


if __name__ == "__main__":
    main()
