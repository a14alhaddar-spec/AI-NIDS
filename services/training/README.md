# Training Pipeline

This folder contains the offline training workflow for CICIDS2017 and UNSW-NB15.

## Expected inputs
- CICIDS2017 CSVs or preprocessed parquet
- UNSW-NB15 CSVs or preprocessed parquet

## Outputs
- Saved models in ../models
- Metrics report in ./reports

## Quick start (from repo root)
Prepare a train/test split from the datasets folder:

```bash
python -m preprocessing.prepare_dataset --dataset cicids2017 --archive datasets
python -m preprocessing.prepare_dataset --dataset unsw --archive datasets
```

Train a model from the prepared CSV:

```bash
python services/training/train.py --data data/processed/cicids2017_train.csv
```
