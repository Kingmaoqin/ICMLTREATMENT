from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .encoder import MLP


class ProxyNet(nn.Module):
    def __init__(self, z_dim: int, out_dim: int, hidden_dims: list[int], dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(z_dim, hidden_dims, out_dim, dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


class BridgeNet(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, hidden_dims: list[int], dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_dims, out_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class BasisFeatures(nn.Module):
    def __init__(self, basis_type: str = "poly", degree: int = 2, spline_knots: int = 3, max_features: int = 8):
        super().__init__()
        self.basis_type = basis_type
        self.degree = degree
        self.spline_knots = spline_knots
        self.max_features = max_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 0:
            return torch.zeros(x.shape[0], 0, device=x.device)
        x_in = x[:, : self.max_features]
        feats = [x_in]
        if self.basis_type == "poly":
            for d in range(2, self.degree + 1):
                feats.append(x_in ** d)
        elif self.basis_type == "spline":
            # Truncated power basis as a lightweight spline approximation.
            for d in range(2, self.degree + 1):
                feats.append(x_in ** d)
            if self.spline_knots > 0:
                min_x = x_in.min(dim=0).values
                max_x = x_in.max(dim=0).values
                knots = [min_x + (i + 1) / (self.spline_knots + 1) * (max_x - min_x)
                         for i in range(self.spline_knots)]
                for knot in knots:
                    feats.append(torch.relu(x_in - knot) ** 3)
        else:
            raise ValueError(f"Unknown basis_type: {self.basis_type}")
        return torch.cat(feats, dim=-1)
