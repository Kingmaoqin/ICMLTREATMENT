from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import ConfoundDiffusionOPM
from models.treatment import encode_treatment
from utils import to_device
from .metrics import (
    average_ate,
    compute_ate_continuous,
    compute_ate_discrete,
    compute_pehe,
    diffusion_plausibility,
    mse_rmse,
    orthogonality_score,
)


class Evaluator:
    def __init__(self, model: ConfoundDiffusionOPM, device: torch.device):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        treatment_type: str,
        contrast_treatments: Optional[list[int]] = None,
        continuous_contrast: Optional[list[float]] = None,
        gt_counterfactual: Optional[np.ndarray] = None,
        knn_k: int = 5,
    ) -> Dict[str, float]:
        self.model.eval()
        outputs = []
        for batch in loader:
            batch = to_device(batch, self.device)
            outputs.append(batch)
        batch_all = {k: torch.cat([b[k] for b in outputs], dim=0) for k in outputs[0]}

        z = self.model.encoder(batch_all["x"])
        t = batch_all["t"]
        t_vec = encode_treatment(t, treatment_type, self.model.t_dim)

        results: Dict[str, float] = {}
        y_pred = self.model.outcome(z, t_vec)
        results.update(mse_rmse(y_pred, batch_all["y"]))
        if treatment_type == "discrete":
            treatments = contrast_treatments or list(range(self.model.t_dim))
            preds = {}
            for t_val in treatments:
                t_prime = torch.full_like(t, float(t_val))
                t_prime_vec = encode_treatment(t_prime, treatment_type, self.model.t_dim)
                delta0 = self.model.delta_net(z, t_vec, t_prime_vec)
                y_hat = self.model.outcome(z + delta0, t_prime_vec)
                preds[t_val] = y_hat
            ate_map = compute_ate_discrete(preds)
            results.update(ate_map)
            results["ate"] = average_ate(ate_map)
            if gt_counterfactual is not None:
                results["pehe"] = compute_pehe(preds, gt_counterfactual)
        elif treatment_type == "continuous":
            if not continuous_contrast or len(continuous_contrast) != 2:
                raise ValueError("continuous_contrast must be [t, t_ref]")
            t_val, t_ref = continuous_contrast
            t_prime = torch.full_like(t, float(t_val))
            t_ref_tensor = torch.full_like(t, float(t_ref))
            t_prime_vec = encode_treatment(t_prime, treatment_type, self.model.t_dim)
            t_ref_vec = encode_treatment(t_ref_tensor, treatment_type, self.model.t_dim)
            delta0 = self.model.delta_net(z, t_vec, t_prime_vec)
            delta_ref = self.model.delta_net(z, t_vec, t_ref_vec)
            y_hat = self.model.outcome(z + delta0, t_prime_vec)
            y_ref = self.model.outcome(z + delta_ref, t_ref_vec)
            results["ate"] = compute_ate_continuous(y_hat, y_ref)

        delta0 = self.model.delta_net(z, t_vec, t_vec)
        results.update(diffusion_plausibility(z, delta0, knn_k))
        if self.model.projector is not None:
            with torch.enable_grad():
                z = z.detach().requires_grad_(True)
                projector = self.model.projector(z)
                y_hat = self.model.outcome(z, t_vec)
                grad = torch.autograd.grad(y_hat.sum(), z, create_graph=False)[0]
                results["rho_perp"] = orthogonality_score(grad, projector)

        output = self.model.forward(batch_all, t_prime=t)
        opm = self.model.opm_losses(batch_all, output, lambda_y=1.0, lambda_t=1.0, rho=0.0)
        results["opm_orth_residual"] = float(torch.norm(opm["psi_orth"].mean(dim=0)).item())
        bridge_y_norm = torch.sum(opm["moments_y"] ** 2).sqrt().item()
        bridge_t_norm = torch.sum(opm["moments_t"] ** 2).sqrt().item()
        results["opm_bridge_residual"] = float(bridge_y_norm + bridge_t_norm)
        return results
