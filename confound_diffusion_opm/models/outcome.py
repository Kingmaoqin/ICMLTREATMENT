from __future__ import annotations

import torch
from torch import nn

from .encoder import MLP


class OutcomeNet(nn.Module):
    def __init__(self, z_dim: int, t_dim: int, hidden_dims: list[int], dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(z_dim + t_dim, hidden_dims, 1, dropout=dropout)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z, t], dim=-1)
        return self.mlp(inp).squeeze(-1)


class MHatNet(nn.Module):
    def __init__(self, x_dim: int, hidden_dims: list[int], dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(x_dim, hidden_dims, 1, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)
