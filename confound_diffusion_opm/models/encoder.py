from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

from utils import clip_by_value


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureMask(nn.Module):
    def __init__(self, mask: Optional[np.ndarray], clip_min: float, clip_max: float, power: float):
        super().__init__()
        if mask is None:
            self.register_buffer("mask", None)
        else:
            mask_tensor = torch.tensor(mask, dtype=torch.float32)
            mask_tensor = clip_by_value(mask_tensor, clip_min, clip_max) ** power
            self.register_buffer("mask", mask_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            return x
        if x.shape[-1] != self.mask.shape[0]:
            raise ValueError("Feature mask dimension mismatch")
        return x * self.mask


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int], dropout: float = 0.0,
                 mask: Optional[FeatureMask] = None):
        super().__init__()
        self.mask = mask
        self.mlp = MLP(input_dim, hidden_dims, latent_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask is not None:
            x = self.mask(x)
        return self.mlp(x)


class TreatmentEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int], dropout: float = 0.0,
                 mask: Optional[FeatureMask] = None):
        super().__init__()
        self.mask = mask
        self.mlp = MLP(input_dim, hidden_dims, latent_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask is not None:
            x = self.mask(x)
        return self.mlp(x)
