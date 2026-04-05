from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from preprocessing.class_balancing import balance_dataset
from preprocessing.preprocess import (
    detect_label_column,
    load_cicids_csvs,
    load_features_config,
    load_unsw_data,
    preprocess_dataframe,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare balanced datasets from archive.")
    parser.add_argument("--archive", default="datasets", help="Path to archive folder")
    parser.add_argument(
        "--dataset",
        choices=["cicids2017", "unsw"],
        required=True,
        help="Which dataset family to process",
    )
    parser.add_argument(
        "--label-mode",
        choices=["binary", "multiclass"],
        default="binary",
        help="Label mode for output",
    )
    parser.add_argument("--label-col", default=None, help="Override label column name")
    parser.add_argument(
        "--features",
        default="configs/features.yml",
        help="Features config (YAML) or empty to keep all",
    )
    parser.add_argument(
        "--balance",
        choices=["none", "smote", "smote_tomek", "undersample", "oversample"],
        default="smote_tomek",
        help="Balancing method (applied to train split only)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", default="data/processed")

    args = parser.parse_args()

    if args.dataset == "cicids2017":
        df = load_cicids_csvs(args.archive)
    else:
        df = load_unsw_data(args.archive)

    label_col = detect_label_column(df, args.label_col)
    features = load_features_config(args.features) if args.features else []

    df = preprocess_dataframe(df, label_col=label_col, label_mode=args.label_mode, features=features)

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    if args.balance != "none":
        X_train, y_train = balance_dataset(X_train, y_train, method=args.balance, random_state=args.random_state)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df["label"] = y_train
    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df["label"] = y_test

    train_path = out_dir / f"{args.dataset}_train.csv"
    test_path = out_dir / f"{args.dataset}_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()
