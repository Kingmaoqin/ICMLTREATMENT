from __future__ import annotations

from typing import Tuple

import math
import torch
from torch import nn

from .encoder import MLP


def make_beta_schedule(schedule: str, timesteps: int) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, timesteps)
    if schedule == "cosine":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 1e-4, 0.999)
    raise ValueError(f"Unknown schedule: {schedule}")


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class DeltaNet(nn.Module):
    def __init__(self, z_dim: int, t_dim: int, hidden_dims: list[int], dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(z_dim + 2 * t_dim, hidden_dims, z_dim, dropout=dropout)

    def forward(self, z: torch.Tensor, t: torch.Tensor, t_prime: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([z, t, t_prime], dim=-1))


class Denoiser(nn.Module):
    def __init__(self, z_dim: int, t_dim: int, hidden_dims: list[int], time_embed_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.mlp = MLP(z_dim + 2 * t_dim + time_embed_dim + z_dim, hidden_dims, z_dim, dropout=dropout)

    def forward(self, delta: torch.Tensor, t_idx: torch.Tensor, z: torch.Tensor, t: torch.Tensor, t_prime: torch.Tensor) -> torch.Tensor:
        t_embed = sinusoidal_time_embedding(t_idx, self.time_embed_dim)
        inp = torch.cat([delta, z, t, t_prime, t_embed], dim=-1)
        return self.mlp(inp)


class DiffusionProcess(nn.Module):
    def __init__(self, timesteps: int, beta_schedule: str):
        super().__init__()
        betas = make_beta_schedule(beta_schedule, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.timesteps = timesteps

    def q_sample(self, delta0: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alphas_cumprod = self.alphas_cumprod[t_idx].unsqueeze(-1)
        return torch.sqrt(alphas_cumprod) * delta0 + torch.sqrt(1 - alphas_cumprod) * noise

    def predict_start_from_noise(self, delta_t: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alphas_cumprod = self.alphas_cumprod[t_idx].unsqueeze(-1)
        return (delta_t - torch.sqrt(1 - alphas_cumprod) * noise) / torch.sqrt(alphas_cumprod)

    def p_sample_ddim(self, denoiser: nn.Module, z: torch.Tensor, t: torch.Tensor, t_prime: torch.Tensor,
                      steps: int = 50) -> torch.Tensor:
        device = z.device
        step_ids = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        delta = torch.randn_like(z)
        for i, t_idx in enumerate(step_ids):
            t_idx_batch = torch.full((z.shape[0],), int(t_idx.item()), device=device, dtype=torch.long)
            eps = denoiser(delta, t_idx_batch, z, t, t_prime)
            alpha_bar = self.alphas_cumprod[t_idx]
            if i < len(step_ids) - 1:
                alpha_bar_prev = self.alphas_cumprod[step_ids[i + 1]]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)
            delta0 = (delta - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
            delta = torch.sqrt(alpha_bar_prev) * delta0 + torch.sqrt(1 - alpha_bar_prev) * eps
        return delta
