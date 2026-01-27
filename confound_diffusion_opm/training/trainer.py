from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from configs.config import Config
from losses import DirichletWeighter
from models import ConfoundDiffusionOPM
from utils import augment_x, ensure_finite, maybe_make_dirs, setup_logging, to_device


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: ConfoundDiffusionOPM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        checkpoint_extra: Optional[Dict[str, object]] = None,
    ) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_extra = checkpoint_extra

        num_losses = 3 if cfg.ssl.enabled else 2
        self.dirichlet = DirichletWeighter(
            num_losses=num_losses,
            beta_low=cfg.dirichlet.beta_low,
            beta_high=cfg.dirichlet.beta_high,
            kappa_min=cfg.dirichlet.kappa_min,
            kappa_max=cfg.dirichlet.kappa_max,
            ema_decay=cfg.dirichlet.ema_decay,
            include_ssl=cfg.dirichlet.include_ssl,
        ).to(device)

        params = list(model.parameters()) + list(self.dirichlet.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler(device_type, enabled=cfg.training.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision)
        self.logger = setup_logging(cfg.training.log_dir)
        maybe_make_dirs(cfg.training.checkpoint_dir)

        self.best_val = float("inf")
        self.bad_epochs = 0

    def _aux_t_loss(self, pi_hat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.cfg.data.treatment_type == "discrete":
            if t.ndim == 2 and t.shape[1] > 1:
                target = torch.argmax(t, dim=1).long()
            else:
                target = t.squeeze(-1).long()
            log_probs = torch.log(pi_hat + 1e-8)
            return torch.nn.functional.nll_loss(log_probs, target)
        return torch.mean((pi_hat - t) ** 2)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        for step, batch in enumerate(self.train_loader):
            batch = to_device(batch, self.device)
            t_prime = self.model.sample_t_prime(batch["t"], self.cfg.diffusion.self_prob)
            output = self.model.forward(batch, t_prime=t_prime)

            diff_losses = self.model.diffusion_losses(
                batch,
                output,
                self.cfg.diffusion.self_prob,
                self.cfg.diffusion.eps,
                self.cfg.loss.lambda_self,
                self.cfg.loss.lambda_cf,
                self.cfg.loss.lambda_perp,
            )
            opm_losses = self.model.opm_losses(
                batch,
                output,
                self.cfg.loss.lambda_Y,
                self.cfg.loss.lambda_T,
                self.cfg.loss.rho,
            )
            if self.cfg.ssl.enabled:
                x1 = augment_x(batch["x"], self.cfg.ssl.noise_std, self.cfg.ssl.drop_prob)
                x2 = augment_x(batch["x"], self.cfg.ssl.noise_std, self.cfg.ssl.drop_prob)
                ssl_losses = self.model.ssl_losses(
                    x1,
                    x2,
                    self.cfg.ssl.temperature,
                    self.cfg.ssl.lambda_pres,
                    self.cfg.ssl.lambda_div,
                    self.cfg.ssl.eps,
                )
                weighted_loss, weights = self.dirichlet(
                    [ssl_losses["l_total"], diff_losses["l_total"], opm_losses["l_total"]]
                )
            else:
                weighted_loss, weights = self.dirichlet([diff_losses["l_total"], opm_losses["l_total"]])
            aux_loss = torch.tensor(0.0, device=self.device)
            if self.cfg.aux.use_y_aux:
                aux_loss = aux_loss + torch.mean((output.y_hat - batch["y"]) ** 2)
            if self.cfg.aux.use_t_aux:
                aux_loss = aux_loss + self._aux_t_loss(output.pi_hat, output.t_vec)
            loss = weighted_loss + aux_loss

            ensure_finite("loss", loss)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.cfg.training.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_batches += 1

            if step % self.cfg.training.log_every == 0:
                if self.cfg.ssl.enabled:
                    self.logger.info(
                        "Epoch %d Step %d Loss %.4f w=%.3f/%.3f/%.3f",
                        epoch,
                        step,
                        loss.item(),
                        weights[0].item(),
                        weights[1].item(),
                        weights[2].item(),
                    )
                else:
                    self.logger.info(
                        "Epoch %d Step %d Loss %.4f w=%.3f/%.3f",
                        epoch,
                        step,
                        loss.item(),
                        weights[0].item(),
                        weights[1].item(),
                    )
        return {"train_loss": total_loss / max(total_batches, 1)}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        for batch in self.val_loader:
            batch = to_device(batch, self.device)
            t_prime = self.model.sample_t_prime(batch["t"], self.cfg.diffusion.self_prob)
            output = self.model.forward(batch, t_prime=t_prime)
            diff_losses = self.model.diffusion_losses(
                batch,
                output,
                self.cfg.diffusion.self_prob,
                self.cfg.diffusion.eps,
                self.cfg.loss.lambda_self,
                self.cfg.loss.lambda_cf,
                self.cfg.loss.lambda_perp,
            )
            opm_losses = self.model.opm_losses(
                batch,
                output,
                self.cfg.loss.lambda_Y,
                self.cfg.loss.lambda_T,
                self.cfg.loss.rho,
            )
            if self.cfg.ssl.enabled:
                x1 = augment_x(batch["x"], self.cfg.ssl.noise_std, self.cfg.ssl.drop_prob)
                x2 = augment_x(batch["x"], self.cfg.ssl.noise_std, self.cfg.ssl.drop_prob)
                ssl_losses = self.model.ssl_losses(
                    x1,
                    x2,
                    self.cfg.ssl.temperature,
                    self.cfg.ssl.lambda_pres,
                    self.cfg.ssl.lambda_div,
                    self.cfg.ssl.eps,
                )
                weighted_loss, _ = self.dirichlet(
                    [ssl_losses["l_total"], diff_losses["l_total"], opm_losses["l_total"]]
                )
            else:
                weighted_loss, _ = self.dirichlet([diff_losses["l_total"], opm_losses["l_total"]])
            aux_loss = torch.tensor(0.0, device=self.device)
            if self.cfg.aux.use_y_aux:
                aux_loss = aux_loss + torch.mean((output.y_hat - batch["y"]) ** 2)
            if self.cfg.aux.use_t_aux:
                aux_loss = aux_loss + self._aux_t_loss(output.pi_hat, output.t_vec)
            loss = weighted_loss + aux_loss
            total_loss += loss.item()
            total_batches += 1
        return {"val_loss": total_loss / max(total_batches, 1)}

    def fit(self) -> None:
        for epoch in range(1, self.cfg.training.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()
            self.logger.info("Epoch %d Train %.4f Val %.4f", epoch, train_metrics["train_loss"], val_metrics["val_loss"])

            if val_metrics["val_loss"] < self.best_val:
                self.best_val = val_metrics["val_loss"]
                self.bad_epochs = 0
                self.save_checkpoint(os.path.join(self.cfg.training.checkpoint_dir, "best.ckpt"))
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.cfg.training.patience:
                    self.logger.info("Early stopping triggered")
                    break

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, object]] = None) -> None:
        payload = {
            "model": self.model.state_dict(),
            "dirichlet": self.dirichlet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.cfg),
        }
        if self.checkpoint_extra:
            payload.update(self.checkpoint_extra)
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        self.logger.info("Saved checkpoint: %s", path)
