from __future__ import annotations

import torch

from utils import batch_mean


def orthogonal_score(y: torch.Tensor, m_hat: torch.Tensor, tau_hat: torch.Tensor,
                     t_vec: torch.Tensor, pi_hat: torch.Tensor) -> torch.Tensor:
    residual = y - m_hat - torch.sum(tau_hat * (t_vec - pi_hat), dim=-1)
    return residual.unsqueeze(-1) * (t_vec - pi_hat)


def bridge_y_moments(residual_y: torch.Tensor, basis_y: torch.Tensor) -> torch.Tensor:
    return batch_mean(basis_y * residual_y.unsqueeze(-1))


def bridge_t_moments(residual_t: torch.Tensor, basis_t: torch.Tensor) -> torch.Tensor:
    # basis_t: (n, L), residual_t: (n, t_dim)
    moments = (basis_t.unsqueeze(-1) * residual_t.unsqueeze(1)).mean(dim=0)
    return moments


def _to_float(name: str, value: object) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar tensor")
        return float(value.item())
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{name} must be a scalar")
        return _to_float(name, value[0])
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError(f"{name} must be a scalar ndarray")
            return float(value.reshape(-1)[0])
    except Exception:
        pass
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be convertible to float") from exc


def opm_loss(
    psi_orth: torch.Tensor,
    bridge_y: torch.Tensor,
    bridge_t: torch.Tensor,
    lambda_y: float,
    lambda_t: float,
    tau_l2: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    lambda_y = _to_float("lambda_y", lambda_y)
    lambda_t = _to_float("lambda_t", lambda_t)
    rho = _to_float("rho", rho)
    orth_term = torch.sum(batch_mean(psi_orth) ** 2)
    bridge_y_term = torch.sum(bridge_y ** 2)
    bridge_t_term = torch.sum(bridge_t ** 2)
    return orth_term + lambda_y * bridge_y_term + lambda_t * bridge_t_term + rho * tau_l2
