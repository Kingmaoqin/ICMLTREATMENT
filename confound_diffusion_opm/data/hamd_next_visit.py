from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .dataset import DataSplit, TreatmentEncoding


REQUIRED_COLUMNS = [
    "RAW_ID", "UNIQUEID", "PROTOCOL", "VISIT",
    "THERAPY_STATUS", "THERAPY", "THERCODE",
    "AGE", "ORIGIN", "GENDER", "GEOCODE",
    "HAMD01", "HAMD02", "HAMD03", "HAMD04", "HAMD05", "HAMD06", "HAMD07",
    "HAMD08", "HAMD09", "HAMD10", "HAMD11", "HAMD12", "HAMD13", "HAMD14",
    "HAMD15", "HAMD16", "HAMD17",
]

HAMD_COLS = [f"HAMD{i:02d}" for i in range(1, 18)]


@dataclass
class PreprocessingConfig:
    numeric_mean: Dict[str, float]
    numeric_std: Dict[str, float]
    numeric_features: List[str]
    cat_vocabularies: Dict[str, Dict[str, int]]
    cat_features: List[str]
    treatment_vocab: Dict[str, int]
    treatment_inverse_vocab: Dict[int, str]
    num_treatments: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "numeric_mean": self.numeric_mean,
            "numeric_std": self.numeric_std,
            "numeric_features": self.numeric_features,
            "cat_vocabularies": self.cat_vocabularies,
            "cat_features": self.cat_features,
            "treatment_vocab": self.treatment_vocab,
            "treatment_inverse_vocab": self.treatment_inverse_vocab,
            "num_treatments": self.num_treatments,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PreprocessingConfig":
        return cls(
            numeric_mean=dict(data["numeric_mean"]),
            numeric_std=dict(data["numeric_std"]),
            numeric_features=list(data["numeric_features"]),
            cat_vocabularies={k: dict(v) for k, v in data["cat_vocabularies"].items()},
            cat_features=list(data["cat_features"]),
            treatment_vocab=dict(data["treatment_vocab"]),
            treatment_inverse_vocab={int(k): v for k, v in data["treatment_inverse_vocab"].items()},
            num_treatments=int(data["num_treatments"]),
        )


def load_and_clean_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    available_cols = [col for col in REQUIRED_COLUMNS if col in df.columns]
    df = df[available_cols]

    if "THERAPY" in df.columns:
        df["THERAPY"] = df["THERAPY"].astype(str).str.strip().str.upper()
    if "THERAPY_STATUS" in df.columns:
        df["THERAPY_STATUS"] = df["THERAPY_STATUS"].astype(str).str.strip().str.upper()

    dtype_map = {
        "RAW_ID": "int64",
        "VISIT": "int64",
        "THERCODE": "int64",
        "AGE": "float32",
        "GEOCODE": "str",
        "PROTOCOL": "str",
        "UNIQUEID": "str",
        "ORIGIN": "str",
        "GENDER": "str",
        "THERAPY_STATUS": "str",
        "THERAPY": "str",
    }

    for col in HAMD_COLS:
        if col in df.columns:
            dtype_map[col] = "float32"

    for col, dtype in dtype_map.items():
        if col not in df.columns:
            continue
        try:
            if dtype == "str":
                df[col] = df[col].astype(str)
            elif dtype in ["int64", "float32", "float64"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if dtype == "int64":
                    df = df.dropna(subset=[col])
                    df[col] = df[col].astype(dtype)
                else:
                    df[col] = df[col].astype(dtype)
        except Exception as exc:
            print(f"Warning: Could not convert {col} to {dtype}: {exc}")

    df = df.dropna(subset=["UNIQUEID", "VISIT"])
    df = df.dropna(subset=["THERAPY", "THERAPY_STATUS"])

    valid_hamd = df[HAMD_COLS].notna().all(axis=1)
    df = df[valid_hamd].copy()

    df = df.sort_values(["UNIQUEID", "VISIT"])
    df = df.drop_duplicates(subset=["UNIQUEID", "VISIT"], keep="first")

    df["HAMD_TOTAL"] = df[HAMD_COLS].sum(axis=1)
    return df


def create_next_visit_samples(df: pd.DataFrame) -> pd.DataFrame:
    samples = []
    for _, group in df.groupby("UNIQUEID"):
        group = group.sort_values("VISIT").reset_index(drop=True)
        for i in range(len(group) - 1):
            current_visit = group.iloc[i]
            next_visit = group.iloc[i + 1]
            sample = current_visit.copy()
            sample["NEXT_HAMD_TOTAL"] = next_visit["HAMD_TOTAL"]
            samples.append(sample)
    return pd.DataFrame(samples).reset_index(drop=True)


def patient_level_split(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    unique_patients = df["UNIQUEID"].unique()
    train_patients, temp_patients = train_test_split(
        unique_patients,
        train_size=train_size,
        random_state=seed,
    )
    val_ratio = val_size / (val_size + test_size)
    val_patients, test_patients = train_test_split(
        temp_patients,
        train_size=val_ratio,
        random_state=seed,
    )

    train_df = df[df["UNIQUEID"].isin(train_patients)].copy()
    val_df = df[df["UNIQUEID"].isin(val_patients)].copy()
    test_df = df[df["UNIQUEID"].isin(test_patients)].copy()
    return train_df, val_df, test_df


def fit_preprocessing(train_df: pd.DataFrame, treatment_col: str) -> PreprocessingConfig:
    numeric_features = ["AGE", "VISIT"] + HAMD_COLS + ["HAMD_TOTAL"]
    cat_features = ["PROTOCOL", "ORIGIN", "GENDER", "GEOCODE", "THERAPY_STATUS"]

    numeric_mean: Dict[str, float] = {}
    numeric_std: Dict[str, float] = {}
    for col in numeric_features:
        if col not in train_df.columns:
            continue
        series = pd.to_numeric(train_df[col], errors="coerce")
        valid = series.dropna()
        if len(valid) > 0:
            numeric_mean[col] = float(valid.mean())
            numeric_std[col] = float(valid.std() + 1e-8)
        else:
            numeric_mean[col] = 0.0
            numeric_std[col] = 1.0

    cat_vocabularies: Dict[str, Dict[str, int]] = {}
    for col in cat_features:
        if col in train_df.columns:
            unique_vals = train_df[col].dropna().astype(str).unique()
            cat_vocabularies[col] = {val: idx for idx, val in enumerate(unique_vals)}

    treatments = train_df[treatment_col].dropna().astype(str).unique()
    treatment_vocab = {val: idx for idx, val in enumerate(treatments)}
    treatment_inverse_vocab = {idx: val for val, idx in treatment_vocab.items()}
    num_treatments = len(treatment_vocab)

    return PreprocessingConfig(
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
        numeric_features=numeric_features,
        cat_vocabularies=cat_vocabularies,
        cat_features=cat_features,
        treatment_vocab=treatment_vocab,
        treatment_inverse_vocab=treatment_inverse_vocab,
        num_treatments=num_treatments,
    )


def apply_preprocessing(df: pd.DataFrame, config: PreprocessingConfig, treatment_col: str) -> pd.DataFrame:
    df = df.copy()
    for col in config.numeric_features:
        if col in df.columns:
            df[col] = (df[col] - config.numeric_mean[col]) / config.numeric_std[col]
    df[config.numeric_features] = df[config.numeric_features].fillna(0.0)

    for col in config.cat_features:
        if col in df.columns:
            df[col + "_encoded"] = df[col].map(config.cat_vocabularies[col]).fillna(-1).astype(int)

    df["THERAPY_encoded"] = df[treatment_col].astype(str).map(config.treatment_vocab).fillna(-1).astype(int)
    return df


def build_covariates(df: pd.DataFrame, config: PreprocessingConfig) -> np.ndarray:
    numeric = df[config.numeric_features].to_numpy(dtype=np.float32)
    cat_cols = [col + "_encoded" for col in config.cat_features]
    cat = df[cat_cols].to_numpy(dtype=np.float32)
    return np.concatenate([numeric, cat], axis=1)


def _filter_unknown(df: pd.DataFrame) -> pd.DataFrame:
    if "THERAPY_encoded" not in df.columns:
        return df
    return df[df["THERAPY_encoded"] >= 0].copy()


def prepare_hamd_next_visit_splits(
    csv_path: str,
    treatment_col: Optional[str],
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[Dict[str, DataSplit], PreprocessingConfig, TreatmentEncoding, List[str]]:
    treatment_col = treatment_col or "THERAPY"
    df = load_and_clean_csv(csv_path)
    df_samples = create_next_visit_samples(df)
    train_df, val_df, test_df = patient_level_split(df_samples, train_size, val_size, test_size, seed)

    preprocessor = fit_preprocessing(train_df, treatment_col)
    train_df = apply_preprocessing(train_df, preprocessor, treatment_col)
    val_df = apply_preprocessing(val_df, preprocessor, treatment_col)
    test_df = apply_preprocessing(test_df, preprocessor, treatment_col)

    train_df = _filter_unknown(train_df)
    val_df = _filter_unknown(val_df)
    test_df = _filter_unknown(test_df)

    def build_split(split_df: pd.DataFrame) -> DataSplit:
        x = build_covariates(split_df, preprocessor)
        t_idx = split_df["THERAPY_encoded"].to_numpy(dtype=np.int64).reshape(-1, 1).astype(np.float32)
        y = split_df["NEXT_HAMD_TOTAL"].to_numpy(dtype=np.float32)
        return DataSplit(x=x, t=t_idx, y=y, w=None, v=None)

    splits = {
        "train": build_split(train_df),
        "val": build_split(val_df),
        "test": build_split(test_df),
    }

    t_info = TreatmentEncoding(
        treatment_type="discrete",
        treatment_dim=preprocessor.num_treatments,
        label_mapping=preprocessor.treatment_vocab,
    )
    covariate_cols = preprocessor.numeric_features + preprocessor.cat_features
    return splits, preprocessor, t_info, covariate_cols
