from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .encoder import MLP


class PiNet(nn.Module):
    def __init__(self, x_dim: int, t_dim: int, treatment_type: str, hidden_dims: list[int], dropout: float = 0.0):
        super().__init__()
        self.treatment_type = treatment_type
        self.t_dim = t_dim
        self.mlp = MLP(x_dim, hidden_dims, t_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(x)
        if self.treatment_type == "discrete":
            return torch.softmax(logits, dim=-1)
        if self.treatment_type == "multi_binary":
            return torch.sigmoid(logits)
        return logits


class TauNet(nn.Module):
    def __init__(self, input_dim: int, t_dim: int, hidden_dims: list[int], dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_dims, t_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def encode_treatment(t: torch.Tensor, treatment_type: str, t_dim: int) -> torch.Tensor:
    if treatment_type == "discrete":
        if t.ndim == 2 and t.shape[1] == 1:
            t = t.squeeze(-1)
        t_long = t.long()
        onehot = torch.zeros(t_long.shape[0], t_dim, device=t.device)
        onehot.scatter_(1, t_long.unsqueeze(1), 1.0)
        return onehot
    if treatment_type in {"multi_binary", "continuous"}:
        return t
    raise ValueError(f"Unknown treatment_type: {treatment_type}")


def treatment_dot(tau: torch.Tensor, t_vec: torch.Tensor) -> torch.Tensor:
    if tau.ndim == 1:
        return tau * t_vec.squeeze(-1)
    return torch.sum(tau * t_vec, dim=-1)


def sample_counterfactual_treatment(t: torch.Tensor, treatment_type: str, t_dim: int) -> torch.Tensor:
    if treatment_type == "discrete":
        t_long = t.squeeze(-1).long()
        rand = torch.randint(low=0, high=t_dim, size=t_long.shape, device=t.device)
        rand = torch.where(rand == t_long, (rand + 1) % t_dim, rand)
        return rand.unsqueeze(-1).float()
    if treatment_type == "multi_binary":
        flip = torch.randint(low=0, high=2, size=t.shape, device=t.device).float()
        return torch.where(flip > 0, 1.0 - t, t)
    if treatment_type == "continuous":
        noise = torch.randn_like(t) * 0.5
        return t + noise
    raise ValueError(f"Unknown treatment_type: {treatment_type}")


def treatment_difference_mask(t: torch.Tensor, t_prime: torch.Tensor, treatment_type: str, eps: float) -> torch.Tensor:
    if treatment_type == "discrete":
        return (t_prime.squeeze(-1).long() != t.squeeze(-1).long()).float()
    diff = torch.abs(t_prime - t)
    if diff.ndim > 1:
        diff = torch.max(diff, dim=1).values
    return (diff > eps).float()
