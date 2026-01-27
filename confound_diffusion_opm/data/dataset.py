from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocess import TabularPreprocessor, infer_column_types


@dataclass
class TreatmentEncoding:
    treatment_type: str
    treatment_dim: int
    label_mapping: Optional[Dict[float, int]] = None


@dataclass
class DataSplit:
    x: np.ndarray
    t: np.ndarray
    y: np.ndarray
    w: Optional[np.ndarray]
    v: Optional[np.ndarray]


class TreatmentDataset(Dataset):
    def __init__(self, split: DataSplit):
        self.split = split

    def __len__(self) -> int:
        return len(self.split.y)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.tensor(self.split.x[idx], dtype=torch.float32)
        t = torch.tensor(self.split.t[idx], dtype=torch.float32)
        y = torch.tensor(self.split.y[idx], dtype=torch.float32)
        batch = {"x": x, "t": t, "y": y}
        if self.split.w is not None:
            batch["w"] = torch.tensor(self.split.w[idx], dtype=torch.float32)
        if self.split.v is not None:
            batch["v"] = torch.tensor(self.split.v[idx], dtype=torch.float32)
        return batch


def _encode_discrete_treatment(series: pd.Series) -> Tuple[np.ndarray, TreatmentEncoding]:
    series = pd.to_numeric(series, errors="coerce")
    unique_vals = sorted(series.dropna().unique().tolist())
    mapping = {float(val): i for i, val in enumerate(unique_vals)}
    labels = series.map(mapping).fillna(0).astype(int).to_numpy()
    return labels.reshape(-1, 1), TreatmentEncoding("discrete", len(unique_vals), mapping)


def _encode_multi_binary(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, TreatmentEncoding]:
    values = df[cols].fillna(0.0).astype(float).to_numpy()
    return values, TreatmentEncoding("multi_binary", len(cols), None)


def _encode_continuous(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, TreatmentEncoding]:
    values = df[cols].fillna(0.0).astype(float).to_numpy()
    return values, TreatmentEncoding("continuous", values.shape[1], None)


def prepare_splits(
    df: pd.DataFrame,
    treatment_type: str,
    treatment_col: Optional[str],
    treatment_cols: Optional[List[str]],
    outcome_col: Optional[str],
    outcome_sum_cols: Optional[List[str]],
    covariate_cols: Optional[List[str]],
    categorical_cols: Optional[List[str]],
    proxy_w_cols: Optional[List[str]],
    proxy_v_cols: Optional[List[str]],
    drop_cols: Optional[List[str]],
    val_size: float,
    test_size: float,
    seed: int,
    standardize: bool,
    ensure_train_all_treatments: bool = False,
    return_indices: bool = False,
) -> Tuple[Dict[str, DataSplit], TabularPreprocessor, TreatmentEncoding, List[str]]:
    rng = np.random.default_rng(seed)
    df = df.copy()
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if outcome_col:
        y = df[outcome_col].astype(float).to_numpy()
    elif outcome_sum_cols:
        y = df[outcome_sum_cols].astype(float).sum(axis=1).to_numpy()
    else:
        raise ValueError("Either outcome_col or outcome_sum_cols must be provided")

    if treatment_type == "discrete":
        if treatment_col is None:
            raise ValueError("treatment_col must be set for discrete treatment")
        t_encoded, t_info = _encode_discrete_treatment(df[treatment_col])
    elif treatment_type == "multi_binary":
        if not treatment_cols:
            raise ValueError("treatment_cols must be set for multi_binary treatment")
        t_encoded, t_info = _encode_multi_binary(df, treatment_cols)
    elif treatment_type == "continuous":
        if treatment_cols:
            t_encoded, t_info = _encode_continuous(df, treatment_cols)
        elif treatment_col:
            t_encoded, t_info = _encode_continuous(df, [treatment_col])
        else:
            raise ValueError("treatment_col or treatment_cols must be set for continuous treatment")
    else:
        raise ValueError(f"Unknown treatment_type: {treatment_type}")

    exclude_cols = set()
    if treatment_col:
        exclude_cols.add(treatment_col)
    if treatment_cols:
        exclude_cols.update(treatment_cols)
    if outcome_col:
        exclude_cols.add(outcome_col)
    if outcome_sum_cols:
        exclude_cols.update(outcome_sum_cols)
    if proxy_w_cols:
        exclude_cols.update(proxy_w_cols)
    if proxy_v_cols:
        exclude_cols.update(proxy_v_cols)

    if covariate_cols is None:
        covariate_cols = [c for c in df.columns if c not in exclude_cols]

    if categorical_cols is None:
        inferred_numeric, inferred_cat = infer_column_types(df, exclude_cols)
        categorical_cols = inferred_cat
        if covariate_cols is not None:
            categorical_cols = [c for c in categorical_cols if c in covariate_cols]

    numeric_cols = [c for c in covariate_cols if c not in categorical_cols]

    preprocessor = TabularPreprocessor(standardize=standardize)
    preprocessor.fit(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    x = preprocessor.transform(df, covariate_cols)

    w = preprocessor.transform(df, proxy_w_cols) if proxy_w_cols else None
    v = preprocessor.transform(df, proxy_v_cols) if proxy_v_cols else None

    indices = np.arange(len(df))
    rng.shuffle(indices)

    test_count = int(len(indices) * test_size)
    val_count = int(len(indices) * val_size)

    if treatment_type == "discrete" and ensure_train_all_treatments:
        labels = t_encoded.reshape(-1)
        train_idx = []
        remaining = []
        for lab in np.unique(labels):
            lab_idx = np.where(labels == lab)[0]
            rng.shuffle(lab_idx)
            if lab_idx.size == 0:
                continue
            train_idx.append(lab_idx[0])
            remaining.extend(lab_idx[1:])
        remaining = np.array(remaining, dtype=int)
        rng.shuffle(remaining)
        if remaining.size < test_count + val_count:
            test_count = min(test_count, remaining.size)
            val_count = min(val_count, max(0, remaining.size - test_count))
        test_idx = remaining[:test_count]
        val_idx = remaining[test_count:test_count + val_count]
        train_idx = np.array(train_idx + remaining[test_count + val_count:].tolist(), dtype=int)
    else:
        test_idx = indices[:test_count]
        val_idx = indices[test_count:test_count + val_count]
        train_idx = indices[test_count + val_count:]

    def split(arr: Optional[np.ndarray], idx: np.ndarray) -> Optional[np.ndarray]:
        if arr is None:
            return None
        return arr[idx]

    splits = {
        "train": DataSplit(x[train_idx], t_encoded[train_idx], y[train_idx], split(w, train_idx), split(v, train_idx)),
        "val": DataSplit(x[val_idx], t_encoded[val_idx], y[val_idx], split(w, val_idx), split(v, val_idx)),
        "test": DataSplit(x[test_idx], t_encoded[test_idx], y[test_idx], split(w, test_idx), split(v, test_idx)),
    }
    if return_indices:
        split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
        return splits, preprocessor, t_info, covariate_cols, split_indices
    return splits, preprocessor, t_info, covariate_cols
