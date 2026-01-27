from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from models.treatment import encode_treatment


def compute_rmse_mae_mse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    return rmse, mae, mse


def calibration_summary(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"ece": np.nan, "slope": np.nan}
    bins = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    idx = np.digitize(y_pred, bins[1:-1])
    ece = 0.0
    total = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        pred_mean = float(np.mean(y_pred[mask]))
        true_mean = float(np.mean(y_true[mask]))
        ece += (mask.sum() / total) * abs(pred_mean - true_mean)
    slope = float(np.polyfit(y_pred, y_true, 1)[0]) if np.unique(y_pred).size > 1 else np.nan
    return {"ece": float(ece), "slope": slope}


def compute_delta_ocp(y_true: np.ndarray, t_idx: np.ndarray, num_treatments: int) -> np.ndarray:
    means = np.full(num_treatments, np.nan, dtype=np.float64)
    for k in range(num_treatments):
        mask = t_idx == k
        if mask.any():
            means[k] = float(np.mean(y_true[mask]))
    delta = np.full((num_treatments, num_treatments), np.nan, dtype=np.float64)
    for a in range(num_treatments):
        for b in range(num_treatments):
            if a == b:
                continue
            if np.isfinite(means[a]) and np.isfinite(means[b]):
                delta[a, b] = means[a] - means[b]
    return delta


def compute_ate_hat_matrix(po: np.ndarray) -> np.ndarray:
    num_treatments = po.shape[1]
    ate_hat = np.zeros((num_treatments, num_treatments), dtype=np.float64)
    for a in range(num_treatments):
        for b in range(num_treatments):
            if a == b:
                continue
            ate_hat[a, b] = float(np.mean(po[:, a] - po[:, b]))
    return ate_hat


def compute_ate_calibration_error(ate_hat: np.ndarray, delta_ocp: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    errors = []
    pairs = []
    num_treatments = ate_hat.shape[0]
    for a in range(num_treatments):
        for b in range(a + 1, num_treatments):
            if np.isfinite(delta_ocp[a, b]):
                errors.append(abs(ate_hat[a, b] - delta_ocp[a, b]))
                pairs.append((a, b))
    if not errors:
        return np.nan, pairs
    return float(np.mean(errors)), pairs


def compute_itt_ate(
    po: np.ndarray,
    y_true: np.ndarray,
    t_idx: np.ndarray,
    baseline_idx: int,
) -> Dict[int, Dict[str, float]]:
    num_treatments = po.shape[1]
    out: Dict[int, Dict[str, float]] = {}
    base_mask = t_idx == baseline_idx
    base_mean = float(np.mean(y_true[base_mask])) if base_mask.any() else np.nan
    for k in range(num_treatments):
        if k == baseline_idx:
            continue
        treat_mask = t_idx == k
        treat_mean = float(np.mean(y_true[treat_mask])) if treat_mask.any() else np.nan
        delta_rct = treat_mean - base_mean if np.isfinite(treat_mean) and np.isfinite(base_mean) else np.nan
        ate = float(np.mean(po[:, k] - po[:, baseline_idx]))
        out[k] = {
            "ate_itt": ate,
            "delta_rct_itt": delta_rct,
            "abs_diff": abs(ate - delta_rct) if np.isfinite(delta_rct) else np.nan,
        }
    return out


def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"min": np.nan, "p01": np.nan, "p05": np.nan, "mean": np.nan, "p95": np.nan, "p99": np.nan, "max": np.nan}
    return {
        "min": float(values.min()),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def compute_propensity_diagnostics(prop: np.ndarray, t_idx: np.ndarray) -> Dict[str, Dict[str, float]]:
    e_obs = prop[np.arange(len(t_idx)), t_idx]
    e_obs = np.clip(e_obs, 1e-12, 1.0)
    weights = 1.0 / e_obs
    ess = float((weights.sum() ** 2) / np.sum(weights ** 2)) if weights.size else np.nan
    return {
        "propensity_obs": _summary_stats(e_obs),
        "weight_stats": {
            "mean": float(weights.mean()) if weights.size else np.nan,
            "p95": float(np.percentile(weights, 95)) if weights.size else np.nan,
            "p99": float(np.percentile(weights, 99)) if weights.size else np.nan,
            "max": float(weights.max()) if weights.size else np.nan,
            "ess": ess,
            "frac_gt_10": float(np.mean(weights > 10.0)) if weights.size else np.nan,
            "frac_gt_20": float(np.mean(weights > 20.0)) if weights.size else np.nan,
        },
    }


def compute_policy_value_ipw(
    po: np.ndarray,
    y_true: np.ndarray,
    t_idx: np.ndarray,
    prop: np.ndarray,
    weight_cap: float | None = None,
    prop_floor: float | None = None,
) -> float:
    policy = np.argmin(po, axis=1)
    p_obs = prop[np.arange(len(t_idx)), t_idx]
    p_obs = np.clip(p_obs, 1e-6, 1.0)
    if prop_floor is not None:
        p_obs = np.clip(p_obs, prop_floor, 1.0)
    weights = (t_idx == policy).astype(float) / p_obs
    if weight_cap is not None:
        weights = np.minimum(weights, weight_cap)
    return float(np.mean(weights * y_true))


class SwitchedHAMDEvaluator:
    def __init__(self, model: torch.nn.Module, device: torch.device, treatment_type: str):
        self.model = model.to(device)
        self.device = device
        self.treatment_type = treatment_type

    @torch.no_grad()
    def evaluate(
        self,
        loader: torch.utils.data.DataLoader,
        train_loader: torch.utils.data.DataLoader,
        treatment_dim: int,
        treatment_names: Dict[int, str],
        ece_bins: int = 10,
    ) -> Dict[str, object]:
        self.model.eval()
        y_true_all: List[np.ndarray] = []
        y_pred_all: List[np.ndarray] = []
        t_idx_all: List[np.ndarray] = []
        po_all: List[np.ndarray] = []
        prop_all: List[np.ndarray] = []

        for batch in loader:
            x = batch["x"].to(self.device)
            t = batch["t"].to(self.device)
            y = batch["y"].to(self.device)

            z = self.model.encoder(x)
            t_vec = encode_treatment(t, self.treatment_type, treatment_dim)
            y_pred = self.model.outcome(z, t_vec)

            preds = []
            for k in range(treatment_dim):
                t_prime = torch.full_like(t, float(k))
                t_prime_vec = encode_treatment(t_prime, self.treatment_type, treatment_dim)
                delta0 = self.model.delta_net(z, t_vec, t_prime_vec)
                y_hat = self.model.outcome(z + delta0, t_prime_vec)
                preds.append(y_hat.cpu().numpy())
            po_all.append(np.stack(preds, axis=1))

            pi_hat = self.model.pi_hat(x).cpu().numpy()
            prop_all.append(pi_hat)

            y_true_all.append(y.cpu().numpy().reshape(-1))
            y_pred_all.append(y_pred.cpu().numpy().reshape(-1))
            t_idx_all.append(t.squeeze(-1).long().cpu().numpy())

        y_true = np.concatenate(y_true_all, axis=0)
        y_pred = np.concatenate(y_pred_all, axis=0)
        t_idx = np.concatenate(t_idx_all, axis=0)
        po_pred = np.concatenate(po_all, axis=0)
        prop = np.concatenate(prop_all, axis=0)

        rmse, mae, mse = compute_rmse_mae_mse(y_true, y_pred)
        calib = calibration_summary(y_true, y_pred, n_bins=ece_bins)

        delta_ocp = compute_delta_ocp(y_true, t_idx, treatment_dim)
        ate_hat = compute_ate_hat_matrix(po_pred)
        ate_calibration_error, valid_pairs = compute_ate_calibration_error(ate_hat, delta_ocp)

        # Baseline treatment: most frequent in training set
        train_t = []
        for batch in train_loader:
            train_t.append(batch["t"].squeeze(-1).long().cpu().numpy())
        train_t_idx = np.concatenate(train_t, axis=0)
        unique, counts = np.unique(train_t_idx, return_counts=True)
        baseline_idx = int(unique[np.argmax(counts)])
        baseline_name = treatment_names.get(baseline_idx, str(baseline_idx))

        itt_ate = compute_itt_ate(po_pred, y_true, t_idx, baseline_idx)
        propensity_diag = compute_propensity_diagnostics(prop, t_idx)
        policy_value = compute_policy_value_ipw(po_pred, y_true, t_idx, prop)
        policy_sensitivity = {
            "cap_10": compute_policy_value_ipw(po_pred, y_true, t_idx, prop, weight_cap=10.0),
            "cap_20": compute_policy_value_ipw(po_pred, y_true, t_idx, prop, weight_cap=20.0),
            "pmin_0p01": compute_policy_value_ipw(po_pred, y_true, t_idx, prop, prop_floor=0.01),
            "pmin_0p05": compute_policy_value_ipw(po_pred, y_true, t_idx, prop, prop_floor=0.05),
        }

        ate_hat_pairs = {f"{a},{b}": float(ate_hat[a, b]) for a, b in valid_pairs}
        delta_ocp_pairs = {f"{a},{b}": float(delta_ocp[a, b]) for a, b in valid_pairs}

        return {
            "rmse": rmse,
            "mae": mae,
            "mse": mse,
            "calibration_ece": calib["ece"],
            "calibration_slope": calib["slope"],
            "ate_calibration_error": ate_calibration_error,
            "policy_value_ipw": policy_value,
            "baseline_treatment_idx": baseline_idx,
            "baseline_treatment_name": baseline_name,
            "ate_hat_pairs": ate_hat_pairs,
            "delta_ocp_pairs": delta_ocp_pairs,
            "itt_ate": itt_ate,
            "num_treatments": treatment_dim,
            "treatment_names": treatment_names,
            "propensity_diagnostics": propensity_diag,
            "policy_value_ipw_sensitivity": policy_sensitivity,
            "propensity_e_p01": propensity_diag["propensity_obs"]["p01"],
            "propensity_e_p99": propensity_diag["propensity_obs"]["p99"],
            "propensity_weight_p99": propensity_diag["weight_stats"]["p99"],
            "propensity_weight_max": propensity_diag["weight_stats"]["max"],
            "propensity_ess": propensity_diag["weight_stats"]["ess"],
            "propensity_weight_frac_gt_10": propensity_diag["weight_stats"]["frac_gt_10"],
            "propensity_weight_frac_gt_20": propensity_diag["weight_stats"]["frac_gt_20"],
            "policy_value_ipw_cap10": policy_sensitivity["cap_10"],
            "policy_value_ipw_cap20": policy_sensitivity["cap_20"],
            "policy_value_ipw_pmin01": policy_sensitivity["pmin_0p01"],
            "policy_value_ipw_pmin05": policy_sensitivity["pmin_0p05"],
        }
