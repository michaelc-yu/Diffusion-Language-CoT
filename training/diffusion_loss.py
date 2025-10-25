import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional


class NoiseSchedule:
    def __init__(self, num_timesteps: int = 1000, schedule_type: str = 'cosine'):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type

        if schedule_type == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        elif schedule_type == 'linear':
            self.betas = self._linear_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def _linear_beta_schedule(self, timesteps):
        return torch.linspace(1e-4, 0.02, timesteps)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def get_alpha_bar(self, t: torch.Tensor):
        return self.alpha_bars[t]  # shape: [batch]

    def get_variance(self, t: torch.Tensor):
        return self.betas[t]

    def device(self, device):
        """Move schedule to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self




class VLBLoss(nn.Module):
    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        vocab_size: int,
        parameterization: str = 'x0',   # 'x0' or 'eps'
        loss_type: str = 'mse'          # 'mse' or 'kl'
    ):
        super().__init__()
        self.noise_schedule = noise_schedule
        self.parameterization = parameterization
        self.loss_type = loss_type
        self.vocab_size = vocab_size

    def forward(
        self,
        model,
        x_start: torch.Tensor,        # token IDs [B, L]
        embeddings: torch.Tensor,     # corrupted embeddings x_t [B, L, D]
    ):
        device = embeddings.device
        B, L, D = embeddings.shape
        T = self.noise_schedule.num_timesteps

        # Sample random timestep for each sample in batch
        t = torch.randint(0, T, (B,), device=device)
        t_broadcast = t.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

        # Get original x_0 from input IDs
        x0_emb = model.token_embeddings(x_start)  # [B, L, D]

        # Get alpha_bar_t and noise for timestep t
        alpha_bar = self.noise_schedule.alpha_bars[t].view(B, 1, 1)  # [B,1,1]
        noise = torch.randn_like(x0_emb)

        # Forward: construct x_t from x0
        xt = torch.sqrt(alpha_bar) * x0_emb + torch.sqrt(1 - alpha_bar) * noise

        # Predict using model
        pred = model(
            xt, t  # model must accept t as timestep input
        )  # shape [B, L, D]

        # Loss computation
        if self.parameterization == 'x0':
            target = x0_emb
        elif self.parameterization == 'eps':
            target = noise
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")

        if self.loss_type == 'mse':
            # print("pred type:", type(pred))
            # print("target type:", type(target))
            if isinstance(pred, tuple): # TODO: check this
                pred = pred[0]
            loss = F.mse_loss(pred, target, reduction='mean')
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not supported")

        # Log stats
        loss_per_t = F.mse_loss(pred, target, reduction='none').view(B, -1).mean(dim=1)

        metrics = {
            "loss_t_mean": loss_per_t.mean().item(),
            "loss_t_std": loss_per_t.std().item(),
        }

        return loss, metrics


