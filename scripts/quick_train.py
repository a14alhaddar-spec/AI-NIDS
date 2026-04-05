"""
Quick Model Training Script
Train a Random Forest model on CICIDS2017 dataset
"""

import os
import sys
from pathlib import Path

def train_model():
    print("=" * 60)
    print("Training Random Forest Model on CICIDS2017 Dataset")
    print("=" * 60)
    print()
    
    # Check if dataset exists
    dataset_dir = Path("datasets/CICIDS2017")
    if not dataset_dir.exists():
        print("✗ Dataset not found at:", dataset_dir)
        print("  Please download CICIDS2017 dataset first")
        return False
    
    # Step 1: Prepare dataset
    print("Step 1: Preparing dataset...")
    print("-" * 60)
    
    os.system(
        "python -m preprocessing.prepare_dataset "
        "--dataset cicids2017 "
        "--label-mode multiclass "
        "--balance smote_tomek "
        "--out-dir data/processed"
    )
    
    # Check if processed data was created
    train_file = Path("data/processed/train.csv")
    if not train_file.exists():
        print("\n✗ Dataset preparation failed")
        return False
    
    print("\n✓ Dataset prepared successfully\n")
    
    # Step 2: Train model
    print("Step 2: Training Random Forest model...")
    print("-" * 60)
    
    os.system(
        "python services/training/train.py "
        "--data data/processed/train.csv "
        "--label label "
        "--out models/model.joblib "
        "--scaler models/scaler.joblib"
    )
    
    # Check if model was created
    model_file = Path("models/model.joblib")
    if not model_file.exists():
        print("\n✗ Model training failed")
        return False
    
    print("\n✓ Model trained and saved successfully\n")
    
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {model_file}")
    print(f"Scaler saved to: models/scaler.joblib")
    print()
    print("Next steps:")
    print("  1. Run: python scripts/verify_models.py")
    print("  2. Or start the system: docker-compose up")
    
    return True


if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
