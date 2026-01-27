from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from utils import sigmoid_range


class DirichletWeighter(nn.Module):
    def __init__(
        self,
        num_losses: int,
        beta_low: List[float],
        beta_high: List[float],
        kappa_min: float,
        kappa_max: float,
        ema_decay: float,
        include_ssl: bool = False,
    ) -> None:
        super().__init__()
        if num_losses < 2:
            raise ValueError("DirichletWeighter requires at least 2 losses")
        if len(beta_low) != num_losses or len(beta_high) != num_losses:
            raise ValueError("beta_low/beta_high must match num_losses")
        self.num_losses = num_losses
        self.include_ssl = include_ssl
        self.beta_low = nn.Parameter(torch.tensor(beta_low, dtype=torch.float32), requires_grad=False)
        self.beta_high = nn.Parameter(torch.tensor(beta_high, dtype=torch.float32), requires_grad=False)
        self.v = nn.Parameter(torch.zeros(num_losses))
        self.v0 = nn.Parameter(torch.tensor(0.0))
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.ema_decay = ema_decay
        self.register_buffer("ema", torch.ones(num_losses))

    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(losses) != self.num_losses:
            raise ValueError("Loss count mismatch")
        loss_vec = torch.stack(losses)
        with torch.no_grad():
            self.ema = self.ema * self.ema_decay + (1 - self.ema_decay) * loss_vec.detach().clamp(min=1e-8)
        normed = loss_vec / (self.ema + 1e-8)

        beta_hat = sigmoid_range(self.v, self.beta_low, self.beta_high)
        beta = beta_hat / beta_hat.sum()
        kappa0 = self.kappa_min + (self.kappa_max - self.kappa_min) * torch.sigmoid(self.v0)
        alpha = kappa0 * beta
        dist = torch.distributions.Dirichlet(alpha)
        w = dist.sample()
        w_st = beta + (w - beta).detach()

        if not self.include_ssl and self.num_losses == 3:
            w_st = w_st.clone()
            w_st[0] = 0.0
            w_st = w_st / w_st.sum()

        total = torch.sum(w_st * normed)
        return total, w_st
