"""
Simple Quick Training Script - Train on CICIDS2017 data directly
"""
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

def simplify_labels(label):
    """Map CICIDS labels to our 6 classes"""
    label = str(label).lower()
    
    if 'benign' in label:
        return 'benign'
    elif 'ddos' in label or 'dos' in label:
        return 'dos_ddos'
    elif 'portscan' in label or 'port' in label:
        return 'port_scan'
    elif 'brute' in label or 'ftp' in label or 'ssh' in label:
        return 'brute_force'
    elif 'bot' in label or 'infiltration' in label:
        return 'malware_c2'
    elif 'web' in label or 'sql' in label or 'xss' in label:
        return 'data_exfil'
    else:
        return 'benign'

def main():
    print("=" * 70)
    print("Quick Model Training - CICIDS2017")
    print("=" * 70)
    print()
    
    # Load dataset
    print("Step 1: Loading CICIDS2017 dataset...")
    data_path = Path("datasets/CICIDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv")
    
    if not data_path.exists():
        print(f"✗ Dataset not found: {data_path}")
        return False
    
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} samples")
    print(f"  Columns: {len(df.columns)}")
    
    # Find label column
    label_col = None
    for col in df.columns:
        if 'label' in col.lower():
            label_col = col
            break
    
    if not label_col:
        print("✗ No label column found")
        return False
    
    print(f"  Label column: {label_col}")
    print(f"  Original labels: {df[label_col].unique()}")
    
    # Simplify labels
    df['label_simple'] = df[label_col].apply(simplify_labels)
    print(f"  Simplified labels: {df['label_simple'].unique()}")
    print(f"  Label distribution:")
    for label, count in df['label_simple'].value_counts().items():
        print(f"    {label}: {count:,}")
    
    #Select numeric features only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove label columns and problematic columns
    feature_cols = [c for c in numeric_cols if c not in [label_col, 'label_simple']]
    
    # Handle infinite values and NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)
    
    print(f"\n  Using {len(feature_cols)} numeric features")
    print(f"  After cleaning: {len(df):,} samples")
    
    # Prepare data - take only first 7 features for simplicity
    feature_names = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow Bytes/s', 'Flow Packets/s', 
        'Average Packet Size', 'Fwd Packet Length Mean'
    ]
    
    # Find matching columns (case-insensitive)
    selected_features = []
    for feat in feature_names:
        matches = [c for c in feature_cols if feat.lower() in c.lower()]
        if matches:
            selected_features.append(matches[0])
    
    if len(selected_features) < 5:
        # Fallback: use first 7 numeric columns
        selected_features = feature_cols[:7]
    
    print(f"  Selected features: {selected_features}")
    
    X = df[selected_features].values
    y = df['label_simple'].values
    
    print(f"\nStep 2: Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    
    # Scale features
    print("\nStep 3: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\nStep 4: Training Random Forest model...")
    print("  Parameters: 100 trees, balanced class weights")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        n_jobs=-1,
        class_weight='balanced',
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\nStep 5: Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    
    print("\nClassification Report:")
    print("=" * 70)
    print(classification_report(y_test, y_pred))
   
    # Save model
    print("\nStep 6: Saving model and scaler...")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "model.joblib"
    scaler_path = model_dir / "scaler.joblib"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"  ✓ Model saved: {model_path}")
    print(f"  ✓ Scaler saved: {scaler_path}")
    
    print("\n" + "=" * 70)
    print("✓ Training Complete!")
    print("=" * 70)
    print(f"Feature importances:")
    for feat, imp in zip(selected_features, model.feature_importances_):
        print(f"  {feat}: {imp:.4f}")
    
    print("\nNow run: python scripts/verify_models.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
