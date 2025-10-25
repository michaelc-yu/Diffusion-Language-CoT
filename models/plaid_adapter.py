# models/plaid_adapter.py

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple


class PLAIDDiffusion(nn.Module):
    """
    Wrapper for loading and running inference with the PLAID 1B model.
    Expects weights in: models/pretrained_plaid/plaid1b_weights/
    """
    def __init__(
        self,
        pretrained_dir: str = "models/pretrained_plaid/plaid1b_weights",
        max_seq_length: int = 512,
    ):
        super().__init__()

        self.pretrained_dir = Path(pretrained_dir)
        self.max_seq_length = max_seq_length

        # Load components
        self.embedding_matrix = torch.load(self.pretrained_dir / "embedding_matrix.pt", map_location=torch.device("cpu"))  # [V, D]
        self.model = torch.load(self.pretrained_dir / "model.pt", map_location=torch.device("cpu"))  # transformer backbone
        self.gamma_bounds = torch.load(self.pretrained_dir / "gamma_bounds.pt", map_location=torch.device("cpu"))  # (gamma_0, gamma_T)
        self.noise_schedule = torch.load(self.pretrained_dir / "noise_schedule.pt", map_location=torch.device("cpu"))  # [T]

        # Inspect data types of embedding_matrix, gamma_bounds, and noise_schedule

        # self.embedding_matrix: dict_keys(['matrix'])
        self.embedding_matrix = self.embedding_matrix['matrix']

        # self.gamma_bounds: dict_keys(['gamma_0', 'gamma_1'])
        self.gamma_bounds = (self.gamma_bounds["gamma_0"], self.gamma_bounds["gamma_1"])

        # self.noise_schedule: dict_keys(['W1', 'b1', 'W2'])

        # Important dimensions
        self.vocab_size = self.embedding_matrix.shape[0]
        self.hidden_dim = self.embedding_matrix.shape[1]

        # Wrap embedding layer
        self.token_embeddings = nn.Embedding.from_pretrained(self.embedding_matrix, freeze=False)

    def get_noise_schedule(self) -> torch.Tensor:
        """Returns the noise schedule as tensor."""
        return self.noise_schedule

    def get_gamma_bounds(self) -> Tuple[float, float]:
        """Returns gamma_0, gamma_T."""
        return self.gamma_bounds

    def forward(
        self,
        noisy_embeddings: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer backbone.
        
        Args:
            noisy_embeddings: [B, L, D] noisy input tokens
            t: [B] diffusion timestep indices
            attention_mask: [B, L] binary mask

        Returns:
            denoised output: [B, L, D]
        """
        # Some models may expect timestep t as part of input (e.g., added to embeddings or input tokens)
        # You can modify this to inject t if required by PLAID

        # Currently assumes PLAID model is callable directly with inputs
        output = self.model(
            x=noisy_embeddings,
            t=t,
            mask=attention_mask
        )

        return output


if __name__ == "__main__":
    model = PLAIDDiffusion()
    print("PLAID 1B model loaded")
    print(f"Vocab size: {model.vocab_size}")
    print(f"Hidden dim: {model.hidden_dim}")
    print(f"Gamma bounds: {model.get_gamma_bounds()}")
    print(f"Noise schedule: {model.noise_schedule}")


    # test inference:

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "What is the capital of France?"

    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # shape [1, seq_len]

    # Convert to embeddings
    input_embeddings = model.token_embeddings(input_ids)

    timestep = torch.tensor([500], dtype=torch.long)

    with torch.no_grad():
        output_embeddings = model(input_embeddings, timestep)
    
    logits = model.unembedding(output_embeddings)  # shape [1, seq_len, vocab]

    # Greedy decode
    decoded_ids = torch.argmax(logits, dim=-1)
    decoded_text = tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)

    print(decoded_text)


