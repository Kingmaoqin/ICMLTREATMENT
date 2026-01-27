from __future__ import annotations

import torch
from torch import nn

from .encoder import MLP


class NuisanceProjector(nn.Module):
    def __init__(self, z_dim: int, k_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.k_dim = k_dim
        self.mlp = MLP(z_dim, hidden_dims, z_dim * k_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        b = self.mlp(z).view(batch, z.shape[1], self.k_dim)
        q, _ = torch.linalg.qr(b, mode="reduced")
        projector = torch.bmm(q, q.transpose(1, 2))
        return projector
