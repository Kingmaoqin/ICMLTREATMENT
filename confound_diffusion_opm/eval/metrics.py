from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from utils import cosine_similarity, l2_norm, median_knn_distance


def compute_ate_discrete(y_hat_by_treatment: Dict[int, torch.Tensor]) -> Dict[str, float]:
    keys = sorted(y_hat_by_treatment.keys())
    ate = {}
    for i, ti in enumerate(keys):
        for tj in keys[i + 1:]:
            diff = (y_hat_by_treatment[ti] - y_hat_by_treatment[tj]).mean().item()
            ate[f"ate_{ti}_vs_{tj}"] = diff
    return ate


def compute_ate_continuous(y_hat_t: torch.Tensor, y_hat_ref: torch.Tensor) -> float:
    return float((y_hat_t - y_hat_ref).mean().item())


def compute_pehe(y_hat_by_treatment: Dict[int, torch.Tensor], gt: np.ndarray) -> float:
    keys = sorted(y_hat_by_treatment.keys())
    if len(keys) < 2:
        return float("nan")
    pred = (y_hat_by_treatment[keys[1]] - y_hat_by_treatment[keys[0]]).cpu().numpy()
    true = gt[:, keys[1]] - gt[:, keys[0]]
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def orthogonality_score(grad: torch.Tensor, projector: torch.Tensor) -> float:
    proj_grad = torch.bmm(projector, grad.unsqueeze(-1)).squeeze(-1)
    score = (l2_norm(proj_grad, dim=-1) / (l2_norm(grad, dim=-1) + 1e-8)).mean().item()
    return float(score)


def diffusion_plausibility(z: torch.Tensor, delta: torch.Tensor, knn_k: int) -> Dict[str, float]:
    sid = cosine_similarity(z, z + delta).mean().item()
    median_dist = median_knn_distance(z, k=knn_k)
    sstep = (l2_norm(delta, dim=-1).mean().item()) / (median_dist + 1e-8)
    return {"sid": float(sid), "sstep": float(sstep)}


def mse_rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    mse = torch.mean((y_pred - y_true) ** 2).item()
    rmse = float(mse ** 0.5)
    return {"test_mse": float(mse), "test_rmse": rmse}


def average_ate(ate_map: Dict[str, float]) -> float:
    if not ate_map:
        return float("nan")
    values = list(ate_map.values())
    return float(sum(values) / len(values))
