"""
Train CNN Model for Network Intrusion Detection
Uses CICIDS2017 dataset to train a standalone CNN model
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import time
from models.cnn import build_cnn

# Paths
DATA_PATH = 'datasets/CICIDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv'
MODEL_PATH = 'models/cnn_model.h5'
SCALER_PATH = 'models/cnn_scaler.joblib'
ENCODER_PATH = 'models/cnn_encoder.joblib'

def simplify_labels(label):
    """Map CICIDS labels to simplified categories"""
    label = str(label).strip().lower()
    if 'benign' in label:
        return 'Benign'
    elif 'ddos' in label:
        return 'DDoS'
    elif 'portscan' in label or 'port scan' in label:
        return 'Port Scan'
    elif 'bot' in label:
        return 'Botnet'
    elif 'infiltration' in label or 'infilteration' in label:
        return 'Infiltration'
    else:
        return 'Attack'

def load_and_prepare_data():
    """Load and prepare CICIDS2017 dataset"""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    print(f"Original shape: {df.shape}")
    print(f"Original labels: {df[' Label'].value_counts()}")
    
    # Simplify labels
    df['Label_Simple'] = df[' Label'].apply(simplify_labels)
    print(f"\nSimplified labels:\n{df['Label_Simple'].value_counts()}")
    
    # Select important features (same as RF model for consistency)
    feature_cols = [
        ' Destination Port',
        ' Flow Duration',
        ' Total Fwd Packets',
        ' Total Backward Packets',
        'Total Length of Fwd Packets',
        ' Total Length of Bwd Packets',
        'Flow Bytes/s'
    ]
    
    X = df[feature_cols].copy()
    y = df['Label_Simple'].copy()
    
    # Handle infinite and missing values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Selected features: {feature_cols}")
    
    return X, y

def main():
    start_time = time.time()
    
    print("=" * 60)
    print("CNN Model Training")
    print("=" * 60)
    
    # Load data
    X, y = load_and_prepare_data()
    
    # Encode labels
    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build CNN model
    print("\nBuilding CNN model...")
    input_shape = X_train_scaled.shape[1]
    model = build_cnn(input_shape, num_classes, learning_rate=1e-3)
    
    print(f"\nModel architecture:")
    model.summary()
    
    # Train model
    print("\nTraining CNN model...")
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=15,
        batch_size=128,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save model and artifacts
    print(f"\nSaving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    
    print(f"Saving scaler to {SCALER_PATH}...")
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"Saving encoder to {ENCODER_PATH}...")
    joblib.dump(label_encoder, ENCODER_PATH)
    
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training completed in {elapsed_time:.2f} seconds!")
    print("=" * 60)
    print(f"\nModel files saved:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {SCALER_PATH}")
    print(f"  - {ENCODER_PATH}")
    print(f"\nFinal accuracy: {test_acc*100:.2f}%")
    
if __name__ == '__main__':
    main()
