# Confounding-Adjusted Counterfactual Diffusion + OPM


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
