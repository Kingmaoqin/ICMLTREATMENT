from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from .bridge import BasisFeatures, BridgeNet, ProxyNet
from .diffusion import DeltaNet, Denoiser, DiffusionProcess
from .encoder import Encoder, FeatureMask, TreatmentEncoder
from .nuisance import NuisanceProjector
from .outcome import MHatNet, OutcomeNet
from .treatment import PiNet, TauNet, encode_treatment, sample_counterfactual_treatment, treatment_difference_mask
from losses import bridge_t_moments, bridge_y_moments, cf_orth_loss, opm_loss, orthogonal_score
from utils import l2_norm


@dataclass
class ForwardOutput:
    z: torch.Tensor
    t_vec: torch.Tensor
    t_prime: torch.Tensor
    t_prime_vec: torch.Tensor
    delta0: torch.Tensor
    y_hat: torch.Tensor
    y_cf_hat: torch.Tensor
    m_hat: torch.Tensor
    pi_hat: torch.Tensor
    tau_hat: torch.Tensor
    w_use: Optional[torch.Tensor]
    v_use: Optional[torch.Tensor]


class ConfoundDiffusionOPM(nn.Module):
    def __init__(
        self,
        x_dim: int,
        t_dim: int,
        treatment_type: str,
        latent_dim: int,
        nuisance_dim: int,
        encoder_hidden: list[int],
        t_encoder_hidden: list[int],
        t_encoder_dim: int,
        outcome_hidden: list[int],
        m_hat_hidden: list[int],
        pi_hidden: list[int],
        tau_hidden: list[int],
        delta_hidden: list[int],
        denoiser_hidden: list[int],
        proxy_hidden: list[int],
        bridge_hidden: list[int],
        proxy_w_dim: int,
        proxy_v_dim: int,
        basis_type: str,
        basis_degree: int,
        spline_knots: int,
        max_basis_features: int,
        mask: Optional[FeatureMask],
        t_mask: Optional[FeatureMask],
        dropout: float,
        tau_input: str,
        use_projector: bool,
        diffusion_steps: int,
        beta_schedule: str,
    ) -> None:
        super().__init__()
        # LSSL is enabled for confounder separation when configured.
        self.treatment_type = treatment_type
        self.t_dim = t_dim
        self.latent_dim = latent_dim
        self.tau_input = tau_input
        self.use_projector = use_projector

        self.encoder = Encoder(x_dim, latent_dim, encoder_hidden, dropout=dropout, mask=mask)
        self.treatment_encoder = TreatmentEncoder(x_dim, t_encoder_dim, t_encoder_hidden, dropout=dropout, mask=t_mask)
        self.outcome = OutcomeNet(latent_dim, t_dim, outcome_hidden, dropout=dropout)
        self.m_hat = MHatNet(x_dim, m_hat_hidden, dropout=dropout)
        self.pi_hat = PiNet(x_dim, t_dim, treatment_type, pi_hidden, dropout=dropout)
        self.tau_hat = TauNet(latent_dim if tau_input == "Z" else x_dim, t_dim, tau_hidden, dropout=dropout)

        self.delta_net = DeltaNet(latent_dim, t_dim, delta_hidden, dropout=dropout)
        self.denoiser = Denoiser(latent_dim, t_dim, denoiser_hidden, dropout=dropout)
        self.diffusion = DiffusionProcess(diffusion_steps, beta_schedule)

        self.proxy_w = ProxyNet(latent_dim, proxy_w_dim, proxy_hidden, dropout=dropout)
        self.proxy_v = ProxyNet(latent_dim, proxy_v_dim, proxy_hidden, dropout=dropout)
        self.bridge_y = BridgeNet(x_dim + proxy_v_dim, 1, bridge_hidden, dropout=dropout)
        self.bridge_t = BridgeNet(x_dim + proxy_w_dim, t_dim, bridge_hidden, dropout=dropout)

        self.basis_y = BasisFeatures(basis_type, basis_degree, spline_knots, max_basis_features)
        self.basis_t = BasisFeatures(basis_type, basis_degree, spline_knots, max_basis_features)

        if use_projector:
            self.projector = NuisanceProjector(latent_dim, nuisance_dim, [latent_dim])
        else:
            self.projector = None

    def sample_t_prime(self, t: torch.Tensor, self_prob: float) -> torch.Tensor:
        t_prime = sample_counterfactual_treatment(t, self.treatment_type, self.t_dim)
        if self_prob <= 0:
            return t_prime
        keep = torch.rand(t.shape[0], device=t.device) < self_prob
        t_prime = torch.where(keep.unsqueeze(-1), t, t_prime)
        return t_prime

    def forward(self, batch: Dict[str, torch.Tensor], t_prime: Optional[torch.Tensor] = None) -> ForwardOutput:
        x = batch["x"]
        t = batch["t"]
        z = self.encoder(x)
        t_vec = encode_treatment(t, self.treatment_type, self.t_dim)
        t_prime = t_prime if t_prime is not None else t
        t_prime_vec = encode_treatment(t_prime, self.treatment_type, self.t_dim)

        delta0 = self.delta_net(z, t_vec, t_prime_vec)
        y_hat = self.outcome(z, t_vec)
        y_cf_hat = self.outcome(z + delta0, t_prime_vec)

        m_hat = self.m_hat(x)
        pi_hat = self.pi_hat(x)
        tau_input = z if self.tau_input == "Z" else x
        tau_hat = self.tau_hat(tau_input)

        w_use = batch.get("w")
        v_use = batch.get("v")
        if w_use is None:
            w_use = self.proxy_w(z)
        if v_use is None:
            v_use = self.proxy_v(z)

        return ForwardOutput(
            z=z,
            t_vec=t_vec,
            t_prime=t_prime,
            t_prime_vec=t_prime_vec,
            delta0=delta0,
            y_hat=y_hat,
            y_cf_hat=y_cf_hat,
            m_hat=m_hat,
            pi_hat=pi_hat,
            tau_hat=tau_hat,
            w_use=w_use,
            v_use=v_use,
        )

    def diffusion_losses(
        self,
        batch: Dict[str, torch.Tensor],
        output: ForwardOutput,
        self_prob: float,
        eps: float,
        lambda_self: float,
        lambda_cf: float,
        lambda_perp: float,
    ) -> Dict[str, torch.Tensor]:
        t_prime = output.t_prime
        z = output.z

        timesteps = self.diffusion.timesteps
        t_idx = torch.randint(0, timesteps, (z.shape[0],), device=z.device)
        noise = torch.randn_like(output.delta0)
        delta_t = self.diffusion.q_sample(output.delta0, t_idx, noise)
        eps_pred = self.denoiser(delta_t, t_idx, z, output.t_vec, output.t_prime_vec)
        l_diff = torch.mean((eps_pred - noise) ** 2)

        mask_self = 1.0 - treatment_difference_mask(batch["t"], t_prime, self.treatment_type, eps)
        if mask_self.sum() > 0:
            l_self = torch.mean(((output.y_cf_hat - batch["y"]) ** 2) * mask_self)
        else:
            l_self = torch.tensor(0.0, device=z.device)

        mask_cf = treatment_difference_mask(batch["t"], t_prime, self.treatment_type, eps)
        l_cf = cf_orth_loss(output.y_cf_hat, output.m_hat, output.tau_hat, output.t_prime_vec, output.pi_hat, mask_cf)

        r_perp = torch.tensor(0.0, device=z.device)
        if self.projector is not None and torch.is_grad_enabled():
            projector = self.projector(z)
            delta_unit = output.delta0 / (l2_norm(output.delta0, dim=-1, eps=eps).unsqueeze(-1) + eps)
            grad = torch.autograd.grad(output.y_cf_hat.sum(), z, create_graph=True)[0]
            proj_delta = torch.bmm(projector, delta_unit.unsqueeze(-1)).squeeze(-1)
            r_perp = torch.mean((torch.sum(grad * proj_delta, dim=1)) ** 2)

        l_total = l_diff + lambda_self * l_self + lambda_cf * l_cf + lambda_perp * r_perp
        return {
            "l_diff": l_diff,
            "l_self": l_self,
            "l_cf": l_cf,
            "r_perp": r_perp,
            "l_total": l_total,
        }

    def opm_losses(
        self,
        batch: Dict[str, torch.Tensor],
        output: ForwardOutput,
        lambda_y: float,
        lambda_t: float,
        rho: float,
    ) -> Dict[str, torch.Tensor]:
        psi_orth = orthogonal_score(batch["y"], output.m_hat, output.tau_hat, output.t_vec, output.pi_hat)

        bridge_y_in = torch.cat([batch["x"], output.v_use], dim=-1)
        h_y = self.bridge_y(bridge_y_in).squeeze(-1)
        residual_y = batch["y"] - h_y
        basis_y = self.basis_y(bridge_y_in)
        moments_y = bridge_y_moments(residual_y, basis_y)

        bridge_t_in = torch.cat([batch["x"], output.w_use], dim=-1)
        h_t = self.bridge_t(bridge_t_in)
        residual_t = output.t_vec - h_t
        basis_t = self.basis_t(bridge_t_in)
        moments_t = bridge_t_moments(residual_t, basis_t)

        tau_l2 = sum(torch.sum(p ** 2) for p in self.tau_hat.parameters())
        l_total = opm_loss(psi_orth, moments_y, moments_t, lambda_y, lambda_t, tau_l2, rho)
        return {
            "psi_orth": psi_orth,
            "moments_y": moments_y,
            "moments_t": moments_t,
            "l_total": l_total,
        }

    def ssl_losses(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        temperature: float,
        lambda_pres: float,
        lambda_div: float,
        eps: float,
    ) -> Dict[str, torch.Tensor]:
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        g1 = self.treatment_encoder(x1)
        g2 = self.treatment_encoder(x2)

        temp = max(float(temperature), 1e-4)
        z1_norm = torch.nn.functional.normalize(z1, dim=-1, eps=1e-8)
        z2_norm = torch.nn.functional.normalize(z2, dim=-1, eps=1e-8)
        logits = torch.matmul(z1_norm, z2_norm.t()) / temp
        logits = torch.clamp(logits, -50.0, 50.0)
        labels = torch.arange(z1.shape[0], device=z1.device)
        l_contrast = torch.nn.functional.cross_entropy(logits, labels)

        sim_g = torch.nn.functional.cosine_similarity(g1, g2, dim=-1, eps=1e-8)
        diff = torch.mean((z1 - z2) ** 2, dim=-1)
        l_pres = -torch.mean(diff * sim_g)

        z_centered = z1 - z1.mean(dim=0, keepdim=True)
        cov = torch.matmul(z_centered.t(), z_centered) / (z1.shape[0] - 1 + eps)
        cov = cov + torch.eye(cov.shape[0], device=cov.device) * eps
        var = torch.diag(cov)
        std = torch.sqrt(torch.clamp(var, min=eps))
        corr = cov / (std.unsqueeze(0) * std.unsqueeze(1) + eps)
        corr = torch.clamp(corr, -1.0, 1.0)
        eigvals = torch.linalg.eigvalsh(corr)
        eigvals = torch.clamp(eigvals, min=eps)
        logdet = torch.sum(torch.log(eigvals))
        logdet = torch.minimum(logdet, torch.tensor(0.0, device=logdet.device))
        l_div = -logdet

        l_contrast = torch.nan_to_num(l_contrast, nan=0.0, posinf=1e6, neginf=-1e6)
        l_pres = torch.nan_to_num(l_pres, nan=0.0, posinf=1e6, neginf=-1e6)
        l_div = torch.nan_to_num(l_div, nan=0.0, posinf=1e6, neginf=-1e6)
        l_ssl = l_contrast + float(lambda_pres) * l_pres + float(lambda_div) * l_div
        l_ssl = torch.nan_to_num(l_ssl, nan=0.0, posinf=1e6, neginf=-1e6)
        return {
            "l_contrast": l_contrast,
            "l_preserve": l_pres,
            "l_diverse": l_div,
            "l_total": l_ssl,
        }
