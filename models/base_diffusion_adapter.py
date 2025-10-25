# models/base_diffusion_adapter.py

import torch
import torch.nn as nn
from einops import rearrange, repeat
from transformers import AutoModel

class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        backbone: str = "gpt2",
        hidden_dim: int = 768,
        vocab_size: int = 50257,
        max_seq_length: int = 512,
        parameterization: str = "x0",
    ):
        super().__init__()
        self.parameterization = parameterization
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.hidden_dim = hidden_dim

        # Load pretrained transformer backbone (e.g., GPT2)
        self.model = AutoModel.from_pretrained(backbone)
        self.backbone_dim = self.model.config.hidden_size

        # Project to match the diffusion hidden size if needed
        self.input_proj = (
            nn.Identity()
            if self.backbone_dim == hidden_dim
            else nn.Linear(self.backbone_dim, hidden_dim)
        )

        # Optional output projection
        self.output_proj = nn.Linear(hidden_dim, self.backbone_dim)

        # Final decoder head to logits
        self.final = nn.Linear(self.backbone_dim, vocab_size)

    def token_embeddings(self, token_ids: torch.LongTensor) -> torch.FloatTensor:
        """Return the input embeddings of token ids."""
        return self.model.get_input_embeddings()(token_ids)

    def forward(self, z, gamma, embedding_matrix=None, bias_scale=1.0, x_selfcond=None):
        """
        Args:
            z: [batch, seq_len, hidden_dim] - noised input
            gamma: [batch] or [batch, 1] - diffusion step embedding (can be scalar or broadcasted)
            embedding_matrix: [vocab_size, embed_dim] (optional for output logits)
            bias_scale: Scalar multiplier for residual
            x_selfcond: Self-conditioning (optional)

        Returns:
            logits: [batch, seq_len, vocab_size]
            denoised: [batch, seq_len, hidden_dim]
        """
        # Project z if needed
        x = self.input_proj(z)

        # Inject time embedding (simple method: add to input — you can improve later)
        # Expand gamma to match z shape and add
        if gamma.dim() == 1:
            gamma = gamma[:, None, None]
        elif gamma.dim() == 2:
            gamma = gamma[:, :, None]
        x = x + gamma

        # Pass through transformer
        transformer_outputs = self.model(inputs_embeds=x)
        hidden_states = transformer_outputs.last_hidden_state  # [B, T, H]

        # Project back if needed
        if hasattr(self, 'output_proj'):
            hidden_states = self.output_proj(hidden_states)

        logits = self.final(hidden_states)

        return logits, hidden_states




batch_size = 2
seq_len = 16
hidden_dim = 768
vocab_size = 50257

# === Load model ===
model = DiffusionTransformer(
    backbone='gpt2',
    hidden_dim=hidden_dim,
    vocab_size=vocab_size,
    max_seq_length=seq_len
)

model.eval()

# random inputs
token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
x_t = model.token_embeddings(token_ids)  # (B, T, D)
t = torch.randint(0, 1000, (batch_size,))  # timesteps

print("x_t shape:", x_t.shape)  # should be (B, T, D)
print("t shape:", t.shape)      # should be (B,)

# forward
with torch.no_grad():
    logits, denoised = model(x_t, t)

print("Logits shape:", logits.shape)
print("Denoised shape:", denoised.shape)
print("base_diffusion.py passes sanity check.")


