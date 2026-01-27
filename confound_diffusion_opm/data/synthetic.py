from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class SyntheticConfig:
    n: int = 2000
    p: int = 8
    treatment_type: str = "discrete"
    seed: int = 123


def generate_synthetic_data(cfg: SyntheticConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    x = rng.normal(size=(cfg.n, cfg.p))
    u = rng.normal(size=(cfg.n, 1))
    w = 0.5 * u + rng.normal(scale=0.5, size=(cfg.n, 1))
    v = 0.5 * u + rng.normal(scale=0.5, size=(cfg.n, 1))

    logits = 0.5 * x[:, 0] - 0.2 * x[:, 1] + 0.8 * u[:, 0]
    if cfg.treatment_type == "discrete":
        probs = 1 / (1 + np.exp(-logits))
        t = rng.binomial(1, probs, size=(cfg.n, 1))
    elif cfg.treatment_type == "multi_binary":
        probs = 1 / (1 + np.exp(-logits))
        t = rng.binomial(1, probs, size=(cfg.n, 3))
    else:
        t = logits.reshape(-1, 1) + rng.normal(scale=0.2, size=(cfg.n, 1))

    y = 1.5 * x[:, 0] - x[:, 1] + 2.0 * u[:, 0] + 1.2 * t[:, 0] + rng.normal(scale=0.5, size=cfg.n)
    df = pd.DataFrame(x, columns=[f"X{i}" for i in range(cfg.p)])
    df["W"] = w
    df["V"] = v
    if cfg.treatment_type == "multi_binary":
        for j in range(t.shape[1]):
            df[f"T{j}"] = t[:, j]
    else:
        df["T"] = t[:, 0]
    df["Y"] = y

    # Ground-truth response for PEHE evaluation (example for binary).
    y0 = 1.5 * x[:, 0] - x[:, 1] + 2.0 * u[:, 0] + rng.normal(scale=0.5, size=cfg.n)
    y1 = y0 + 1.2
    gt = np.stack([y0, y1], axis=1)
    return df, gt
