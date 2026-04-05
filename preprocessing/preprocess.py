from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from preprocessing.feature_engineering import ensure_basic_features, select_feature_columns


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def detect_label_column(df: pd.DataFrame, label_col: str | None = None) -> str:
    df.columns = [c.strip() for c in df.columns]
    if label_col and label_col in df.columns:
        return label_col
    lowered = {c.lower(): c for c in df.columns}
    for candidate in ("label", "attack_cat", "class"):
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError("No label column found. Provide --label-col.")


def normalize_label(df: pd.DataFrame, label_col: str, mode: str) -> pd.DataFrame:
    if mode == "binary":
        if label_col.lower() == "label":
            df["label"] = df[label_col].astype(str).str.lower()
            df["label"] = df["label"].apply(lambda v: "benign" if v in {"0", "benign"} else "attack")
        else:
            df["label"] = df[label_col].astype(str).str.lower()
            df["label"] = df["label"].apply(lambda v: "benign" if v in {"benign", "normal"} else "attack")
    else:
        df["label"] = df[label_col].astype(str)
    return df


def load_cicids_csvs(archive_dir: str | Path) -> pd.DataFrame:
    archive_dir = Path(archive_dir)
    files = sorted(archive_dir.glob("*pcap_ISCX.csv"))
    if not files:
        candidate = archive_dir / "CICIDS2017"
        if candidate.exists():
            archive_dir = candidate
            files = sorted(archive_dir.glob("*pcap_ISCX.csv"))
    if not files:
        raise FileNotFoundError("No CIC-IDS-2017 CSV files found in archive.")

    dfs = []
    for file in files:
        df = pd.read_csv(file, low_memory=False)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_unsw_parquet(archive_dir: str | Path) -> pd.DataFrame:
    archive_dir = Path(archive_dir)
    files = sorted(archive_dir.glob("UNSW_NB15_*set.parquet"))
    if not files:
        candidate = archive_dir / "UNSW-NB15"
        if candidate.exists():
            archive_dir = candidate
            files = sorted(archive_dir.glob("UNSW_NB15_*set.parquet"))
    if not files:
        raise FileNotFoundError("No UNSW_NB15 parquet files found in archive.")
    dfs = []
    for file in files:
        dfs.append(pd.read_parquet(file))
    return pd.concat(dfs, ignore_index=True)


def load_unsw_csvs(archive_dir: str | Path) -> pd.DataFrame:
    archive_dir = Path(archive_dir)
    files = sorted(archive_dir.glob("UNSW*_NB15*.csv")) + sorted(
        archive_dir.glob("UNSW-NB15_*.csv")
    )
    if not files:
        candidate = archive_dir / "UNSW-NB15"
        if candidate.exists():
            archive_dir = candidate
            files = sorted(archive_dir.glob("UNSW*_NB15*.csv")) + sorted(
                archive_dir.glob("UNSW-NB15_*.csv")
            )
    files = [
        f
        for f in files
        if "features" not in f.name.lower() and "list_events" not in f.name.lower()
    ]
    if not files:
        raise FileNotFoundError("No UNSW_NB15 CSV files found in archive.")
    dfs = []
    for file in files:
        dfs.append(pd.read_csv(file, low_memory=False))
    return pd.concat(dfs, ignore_index=True)


def load_unsw_data(archive_dir: str | Path) -> pd.DataFrame:
    try:
        return load_unsw_parquet(archive_dir)
    except FileNotFoundError:
        return load_unsw_csvs(archive_dir)


def load_features_config(path: str | Path | None) -> list[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    return cfg.get("features", [])


def preprocess_dataframe(df: pd.DataFrame, label_col: str, label_mode: str, features: list[str]):
    df = normalize_columns(df)
    df = normalize_label(df, label_col, label_mode)
    df = ensure_basic_features(df)

    label = df["label"].copy()

    if features:
        df = select_feature_columns(df, features)
        if "label" not in df.columns:
            df["label"] = label

    if "label" not in df.columns:
        raise ValueError("Label column missing after preprocessing.")
    feature_df = df.drop(columns=["label"])
    feature_df = feature_df.select_dtypes(include=["float64", "int64", "int32", "float32"])
    feature_df = feature_df.replace([float("inf"), float("-inf")], 0).dropna()
    label = label.loc[feature_df.index]
    feature_df["label"] = label.values

    return feature_df


def preprocess_dataset(path):
    df = pd.read_csv(path)
    label_col = detect_label_column(df)
    df = preprocess_dataframe(df, label_col=label_col, label_mode="multiclass", features=[])

    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop("label", axis=1))
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
