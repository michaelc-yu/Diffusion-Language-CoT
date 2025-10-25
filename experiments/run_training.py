# run_training.py

import yaml
from transformers import AutoTokenizer
from data.loaders import get_dataloaders
from training.train import create_trainer  # or wherever your `create_trainer()` is defined


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path="configs/train_base.yaml"):
    # Load config
    config = load_config(config_path)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    train_loader, val_loader = get_dataloaders(config, tokenizer)

    # Create trainer
    trainer = create_trainer(
        model_type='base',
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/train_base.yaml"
    main(config_path)

