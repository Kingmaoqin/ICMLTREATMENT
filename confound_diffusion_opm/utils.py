import json
import logging
import math
import os
import random
from dataclasses import asdict
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


LOGGER_NAME = "confound_diffusion_opm"


def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "run.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_json_dump(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def l2_norm(t: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    if isinstance(eps, str):
        eps = float(eps)
    return torch.sqrt(torch.sum(t * t, dim=dim) + eps)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    numerator = torch.sum(a * b, dim=-1)
    denom = l2_norm(a, eps=eps) * l2_norm(b, eps=eps)
    return numerator / denom


def kludge_named_params(module: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    # Keep a dedicated hook in case we need to exclude params later.
    return module.parameters()


def config_to_dict(config: Any) -> Dict[str, Any]:
    if hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    raise TypeError("Expected dataclass config")


def median_knn_distance(z: torch.Tensor, k: int = 5, max_samples: int = 2048) -> float:
    if z.shape[0] > max_samples:
        idx = torch.randperm(z.shape[0])[:max_samples]
        z = z[idx]
    with torch.no_grad():
        dist = torch.cdist(z, z, p=2)
        dist += torch.eye(dist.shape[0], device=dist.device) * 1e6
        knn, _ = torch.topk(dist, k, largest=False)
        return float(torch.median(knn[:, -1]).item())


def ensure_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"Non-finite detected in {name}")


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def clip_by_value(t: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    return torch.clamp(t, min_value, max_value)


def sigmoid_range(param: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(param)


def maybe_make_dirs(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def format_float_dict(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in metrics.items()}


def batch_mean(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sum(t, dim=0) / (t.shape[0] + eps)


def augment_x(x: torch.Tensor, noise_std: float, drop_prob: float) -> torch.Tensor:
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    if drop_prob > 0:
        mask = (torch.rand_like(x) > drop_prob).float()
        x = x * mask
    return x
