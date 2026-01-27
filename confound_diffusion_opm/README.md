# Confounding-Adjusted Counterfactual Diffusion + OPM

This repository implements **"Confounding Adjusted Multi-Treatment Effect Estimation via Counterfactual Diffusion + Orthogonal Proximal Moments (OPM)"** in PyTorch. It is end-to-end trainable with a modular design and a CLI.

**Important constraints (explicitly verified):**
- **No LLMs are used** anywhere in this codebase.
- **Self-supervised LSSL is implemented** (contrastive, preservation, diversity/logdet) for confounder separation.
- The representation encoder is trained **end-to-end from `L_SSL + L_DIFF + L_OPM`** plus optional supervised auxiliaries.

## Installation

Minimal dependencies:
- Python 3.9+
- PyTorch
- numpy, pandas, pyyaml

Example (CPU):
```bash
pip install torch numpy pandas pyyaml
```

## Dataset format

Input CSV is assumed to be tabular with columns for:
- Covariates `X`
- Treatment `T` (discrete, multi-binary, or continuous)
- Outcome `Y`
- Optional proxies `W` and `V`

For the provided depression dataset, the default config:
- uses `THERCODE` as **discrete treatment**
- uses sum of `HAMD01..HAMD17` as **outcome**

Update `configs/default.yaml` if your schema differs.

## Quickstart

From this directory:
```bash
python -m scripts.train --config configs/default.yaml
python -m scripts.eval  --config configs/default.yaml --ckpt checkpoints/best.ckpt
```

## Configuration fields (high level)

- `data.*`: data path, treatment type, column lists, split ratios
- `model.*`: network sizes, latent dimension, projector dimension
- `diffusion.*`: timesteps, schedule, DDIM sampling steps
- `loss.*`: lambda_self, lambda_cf, lambda_perp, lambda_Y, lambda_T, rho
- `dirichlet.*`: dynamic loss weighting ranges
- `basis.*`: polynomial/spline basis choices for bridge moments
- `aux.*`: optional supervised auxiliaries
- `ssl.*`: self-supervised losses and augmentations

## Outputs

Training outputs:
- `checkpoints/best.ckpt` (best validation loss)
- `checkpoints/final.ckpt`
- `logs/run.log`

Evaluation outputs:
- Test MSE / RMSE
- ATE estimates
- diagnostics: `rho_perp`, `sid`, `sstep`

## Notes on expert priors

You may pass precomputed `m_conf` and `m_out` arrays to `mask.m_conf_path` / `mask.m_out_path`. These arrays must match the **preprocessed feature dimension**. The mask only scales inputs; no LLM is used.

## Tests

Run unit tests:
```bash
python -m unittest discover -s tests
```

## Method summary (implemented modules)

- **Representation**: `Phi_omega(X)` as MLP encoder (optional feature mask).
- **Self-supervision**: contrastive + preservation + diversity losses on augmented views.
- **Outcome predictor**: `f_theta(Z, T)`.
- **Nuisance models**: `m_hat(X)` and `pi_hat(X)`.
- **Diffusion in latent space**: denoising objective for `delta_phi`.
- **Counterfactual loss**: orthogonal moment penalty for generated counterfactuals.
- **Geometric control**: nuisance projector penalty `R_perp`.
- **OPM calibration**: Neyman orthogonal score + proximal bridge moments.
- **Dynamic weighting**: Dirichlet module over `(L_DIFF, L_OPM)`.

This codebase strictly excludes any LLM usage and includes the self-supervised LSSL block.
