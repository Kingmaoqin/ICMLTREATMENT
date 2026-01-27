from __future__ import annotations

import torch

from utils import batch_mean


def cf_orth_loss(
    y_cf: torch.Tensor,
    m_hat: torch.Tensor,
    tau_hat: torch.Tensor,
    t_prime_vec: torch.Tensor,
    pi_hat: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    residual = y_cf - m_hat - torch.sum(tau_hat * t_prime_vec, dim=-1)
    psi = residual.unsqueeze(-1) * (t_prime_vec - pi_hat)
    if mask is not None:
        psi = psi * mask.unsqueeze(-1)
    moment = batch_mean(psi)
    return torch.sum(moment ** 2)
