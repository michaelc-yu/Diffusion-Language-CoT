import argparse
import json
import yaml
import os
from tqdm import tqdm
from transformers import AutoTokenizer
import torch

from models.base_diffusion_adapter import DiffusionTransformer
from data.loaders import CoTDataset  # already written
from eval.llm_judge import evaluate_with_llm  # optional, described below
from data.loaders import get_dataloaders


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_trained_model(config, checkpoint_path, device):
    model = DiffusionTransformer(
        backbone=config['model']['backbone'],
        hidden_dim=config['model']['hidden_dim'],
        vocab_size=config['model']['vocab_size'],
        max_seq_length=config['model']['max_seq_length'],
        parameterization=config['model'].get('parameterization', 'x0')
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def run_inference(model, tokenizer, val_loader, device):
    predictions = []
    references = []

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model.sample(input_ids=input_ids, attention_mask=attention_mask)
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
        predictions.extend(decoded_output)
        references.extend(batch["clean_target_text"])

    return predictions, references


def main(config_path="configs/eval_base.yaml"):
    # config_path = "configs/eval_base.yaml"
    checkpoint_path = "results/checkpoints/test/model_epoch_50.pt"  # example path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path)

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    tokenizer.pad_token = tokenizer.eos_token

    _, val_loader = get_dataloaders(config, tokenizer)
    model = load_trained_model(config, checkpoint_path, device)

    print("Running inference...")
    predictions, references = run_inference(model, tokenizer, val_loader, device)

    # LLM-based evaluation (using llm as a judge)
    if config.get("use_llm_judge", False):
        results = evaluate_with_llm(predictions, references)
        print("LLM Judge Results:")
        print(results)
    else:
        # Fallback
        exact_matches = sum([pred.strip() == ref.strip() for pred, ref in zip(predictions, references)])
        acc = exact_matches / len(predictions)
        print(f"Exact Match Accuracy: {acc:.2%}")

if __name__ == "__main__":
    main()

