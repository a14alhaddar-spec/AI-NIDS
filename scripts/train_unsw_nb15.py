"""
Train ALL models on UNSW-NB15 Dataset
Uses official training/testing split
Creates production-ready models: RF, CNN, LSTM, CNN-LSTM
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import time
from models.cnn import build_cnn
from models.lstm import build_lstm
from models.cnn_lstm import build_cnn_lstm

# UNSW-NB15 paths
TRAIN_PATH = 'datasets/UNSW-NB15/UNSW_NB15_training-set.csv'
TEST_PATH = 'datasets/UNSW-NB15/UNSW_NB15_testing-set.csv'

# Model save paths
OUTPUT_DIR = 'models/unsw_nb15'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_unsw_nb15():
    """Load UNSW-NB15 dataset"""
    print("=" * 70)
    print("Loading UNSW-NB15 Dataset")
    print("=" * 70)
    
    print(f"\nLoading training set from {TRAIN_PATH}...")
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"✓ Training set: {len(train_df):,} rows, {len(train_df.columns)} columns")
    
    print(f"\nLoading testing set from {TEST_PATH}...")
    test_df = pd.read_csv(TEST_PATH)
    print(f"✓ Testing set: {len(test_df):,} rows, {len(test_df.columns)} columns")
    
    # Show label distribution
    if 'label' in train_df.columns:
        print(f"\nTraining set labels:")
        print(f"  Normal (0): {(train_df['label'] == 0).sum():,}")
        print(f"  Attack (1): {(train_df['label'] == 1).sum():,}")
    
    if 'attack_cat' in train_df.columns:
        print(f"\nAttack categories in training set:")
        for cat, count in train_df['attack_cat'].value_counts().items():
            print(f"  {cat}: {count:,}")
    
    return train_df, test_df

def prepare_unsw_features(train_df, test_df):
    """Prepare features from UNSW-NB15 dataset"""
    print("\n" + "=" * 70)
    print("Preparing features...")
    print("=" * 70)
    
    # Select numerical features (excluding categorical and target columns)
    exclude_cols = ['id', 'attack_cat', 'label', 'proto', 'service', 'state']
    
    # Get all numerical columns
    numerical_cols = []
    for col in train_df.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[col]):
            numerical_cols.append(col)
    
    # Limit to most important features for consistency
    # Use similar number of features as CICIDS (7-10 features)
    important_features = [
        'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 
        'rate', 'sttl', 'dttl', 'sload', 'dload'
    ]
    
    # Use only features that exist
    feature_cols = [col for col in important_features if col in train_df.columns]
    
    print(f"Selected {len(feature_cols)} features:")
    for col in feature_cols:
        print(f"  - {col}")
    
    # Extract features
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    # Use attack categories as labels (more detailed than binary)
    if 'attack_cat' in train_df.columns:
        y_train = train_df['attack_cat'].copy()
        y_test = test_df['attack_cat'].copy()
        print(f"\nUsing attack categories as labels")
    else:
        y_train = train_df['label'].apply(lambda x: 'Attack' if x == 1 else 'Normal')
        y_test = test_df['label'].apply(lambda x: 'Attack' if x == 1 else 'Normal')
        print(f"\nUsing binary labels")
    
    # Handle missing values
    X_train.replace([np.inf, -np.inf], np.nan,  inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)
    
    print(f"\nTraining features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest model"""
    print("\n" + "=" * 70)
    print("Training Random Forest")
    print("=" * 70)
    
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=25,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("Training...")
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"\nTrain Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Save
    model_path = f'{OUTPUT_DIR}/rf_model.joblib'
    joblib.dump(model, model_path)
    print(f"✓ Saved to {model_path}")
    
    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.2f}s")
    
    return test_acc

def train_cnn(X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, num_classes):
    """Train CNN model"""
    print("\n" + "=" * 70)
    print("Training CNN")
    print("=" * 70)
    
    start_time = time.time()
    
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    model = build_cnn(X_train_scaled.shape[1], num_classes, learning_rate=1e-3)
    print("\nModel architecture:")
    model.summary()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
    print("\nTraining...")
    history = model.fit(
        X_train_scaled, y_train_encoded,
        validation_split=0.2,
        epochs=20,
        batch_size=256,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save
    model_path = f'{OUTPUT_DIR}/cnn_model.h5'
    model.save(model_path)
    print(f"✓ Saved to {model_path}")
    
    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.2f}s")
    
    return test_acc

def train_lstm(X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, num_classes):
    """Train LSTM model"""
    print("\n" + "=" * 70)
    print("Training LSTM")
    print("=" * 70)
    
    start_time = time.time()
    
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    model = build_lstm(X_train_scaled.shape[1], num_classes, learning_rate=1e-3)
    print("\nModel architecture:")
    model.summary()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
    print("\nTraining...")
    history = model.fit(
        X_train_scaled, y_train_encoded,
        validation_split=0.2,
        epochs=20,
        batch_size=256,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save
    model_path = f'{OUTPUT_DIR}/lstm_model.h5'
    model.save(model_path)
    print(f"✓ Saved to {model_path}")
    
    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.2f}s")
    
    return test_acc

def train_cnn_lstm(X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, num_classes):
    """Train CNN-LSTM hybrid model"""
    print("\n" + "=" * 70)
    print("Training CNN-LSTM Hybrid")
    print("=" * 70)
    
    start_time = time.time()
    
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    model = build_cnn_lstm(X_train_scaled.shape[1], num_classes, learning_rate=1e-3)
    print("\nModel architecture:")
    model.summary()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
    print("\nTraining...")
    history = model.fit(
        X_train_scaled, y_train_encoded,
        validation_split=0.2,
        epochs=20,
        batch_size=256,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save
    model_path = f'{OUTPUT_DIR}/cnn_lstm_model.h5'
    model.save(model_path)
    print(f"✓ Saved to {model_path}")
    
    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.2f}s")
    
    return test_acc

def main():
    global_start = time.time()
    
    print("\n" + "=" * 70)
    print("UNSW-NB15 TRAINING - ALL 4 MODELS")
    print("=" * 70)
    
    # Load dataset
    train_df, test_df = load_unsw_nb15()
    
    # Prepare features
    X_train, X_test, y_train, y_test = prepare_unsw_features(train_df, test_df)
    
    # Encode labels
    print("\n" + "=" * 70)
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler and encoder
    joblib.dump(scaler, f'{OUTPUT_DIR}/scaler.joblib')
    joblib.dump(label_encoder, f'{OUTPUT_DIR}/label_encoder.joblib')
    print(f"✓ Saved scaler and encoder to {OUTPUT_DIR}/")
    
    # Train all models
    results = {}
    
    # 1. Random Forest
    results['Random Forest'] = train_random_forest(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 2. CNN
    results['CNN'] = train_cnn(X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, num_classes)
    
    # 3. LSTM
    results['LSTM'] = train_lstm(X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, num_classes)
    
    # 4. CNN-LSTM
    results['CNN-LSTM'] = train_cnn_lstm(X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, num_classes)
    
    # Final summary
    elapsed_total = time.time() - global_start
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nTotal training time: {elapsed_total/60:.2f} minutes")
    print(f"\nFinal Test Accuracies:")
    for model_name, acc in results.items():
        print(f"  {model_name}: {acc*100:.2f}%")
    
    print(f"\nAll models saved to: {OUTPUT_DIR}/")
    print("=" * 70)

if __name__ == '__main__':
    main()
