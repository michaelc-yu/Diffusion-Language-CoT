# run_training.py

import yaml
from transformers import AutoTokenizer
from data.loaders import get_dataloaders
from training.train import create_trainer


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path="configs/train_base.yaml"):

    config = load_config(config_path)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])
    tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader = get_dataloaders(config, tokenizer)

    trainer = create_trainer(
        model_type='base',
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    trainer.train()


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/train_base.yaml"
    main(config_path)

