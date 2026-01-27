from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PreprocessorState:
    numeric_cols: List[str]
    categorical_cols: List[str]
    means: Dict[str, float]
    stds: Dict[str, float]
    categories: Dict[str, List[str]]


class TabularPreprocessor:
    def __init__(self, standardize: bool = True):
        self.standardize = standardize
        self.state: Optional[PreprocessorState] = None

    def fit(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> None:
        means = {}
        stds = {}
        categories = {}
        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            means[col] = float(series.mean()) if series.notna().any() else 0.0
            std = float(series.std()) if series.notna().any() else 1.0
            stds[col] = std if std > 1e-6 else 1.0
        for col in categorical_cols:
            series = df[col].fillna("missing").astype(str)
            cats = sorted(series.unique().tolist())
            if "missing" not in cats:
                cats.append("missing")
            categories[col] = cats
        self.state = PreprocessorState(numeric_cols, categorical_cols, means, stds, categories)

    def transform(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        if self.state is None:
            raise ValueError("Preprocessor is not fitted")
        numeric_cols = [c for c in cols if c in self.state.numeric_cols]
        categorical_cols = [c for c in cols if c in self.state.categorical_cols]

        numeric_features = []
        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce").fillna(self.state.means[col]).to_numpy()
            if self.standardize:
                series = (series - self.state.means[col]) / self.state.stds[col]
            numeric_features.append(series.reshape(-1, 1))

        categorical_features = []
        for col in categorical_cols:
            cats = self.state.categories[col]
            series = df[col].fillna("missing").astype(str)
            one_hot = np.zeros((len(series), len(cats)), dtype=np.float32)
            cat_to_idx = {c: i for i, c in enumerate(cats)}
            for i, val in enumerate(series):
                idx = cat_to_idx.get(val, cat_to_idx.get("missing"))
                one_hot[i, idx] = 1.0
            categorical_features.append(one_hot)

        if numeric_features and categorical_features:
            return np.concatenate(numeric_features + categorical_features, axis=1)
        if numeric_features:
            return np.concatenate(numeric_features, axis=1)
        if categorical_features:
            return np.concatenate(categorical_features, axis=1)
        return np.zeros((len(df), 0), dtype=np.float32)

    def get_feature_dim(self, cols: List[str]) -> int:
        if self.state is None:
            raise ValueError("Preprocessor is not fitted")
        dim = 0
        for col in cols:
            if col in self.state.numeric_cols:
                dim += 1
            elif col in self.state.categorical_cols:
                dim += len(self.state.categories[col])
        return dim

    def to_dict(self) -> Dict[str, object]:
        if self.state is None:
            raise ValueError("Preprocessor is not fitted")
        return {
            "standardize": self.standardize,
            "numeric_cols": self.state.numeric_cols,
            "categorical_cols": self.state.categorical_cols,
            "means": self.state.means,
            "stds": self.state.stds,
            "categories": self.state.categories,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "TabularPreprocessor":
        pre = cls(standardize=bool(data["standardize"]))
        pre.state = PreprocessorState(
            numeric_cols=list(data["numeric_cols"]),
            categorical_cols=list(data["categorical_cols"]),
            means=dict(data["means"]),
            stds=dict(data["stds"]),
            categories={k: list(v) for k, v in data["categories"].items()},
        )
        return pre


def infer_column_types(df: pd.DataFrame, exclude: Iterable[str]) -> Tuple[List[str], List[str]]:
    numeric_cols = []
    categorical_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols
