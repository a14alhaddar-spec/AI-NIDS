import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(path):
    df = pd.read_csv(path)
    return df


def preprocess(df, label_col="label"):
    y = df[label_col]
    X = df.drop(columns=[label_col])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


def train_rf(X, y):
    sampler = SMOTETomek()
    X_res, y_res = sampler.fit_resample(X, y)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )
    model.fit(X_res, y_res)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--label", default="label", help="Label column")
    parser.add_argument("--out", default="../../models/model.joblib")
    parser.add_argument("--scaler", default="../../models/scaler.joblib")
    args = parser.parse_args()

    df = load_dataset(args.data)
    X, y, scaler = preprocess(df, label_col=args.label)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_rf(X_train, y_train)
    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    joblib.dump(scaler, args.scaler)


if __name__ == "__main__":
    main()
