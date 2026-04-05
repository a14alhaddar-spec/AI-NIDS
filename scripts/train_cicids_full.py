"""
Train ALL models on FULL CICIDS2017 Dataset
Combines all 8 CSV files for comprehensive training
Creates production-ready models: RF, CNN, LSTM, CNN-LSTM
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import time
from models.cnn import build_cnn
from models.lstm import build_lstm
from models.cnn_lstm import build_cnn_lstm

# CICIDS2017 files
CICIDS_FILES = [
    'datasets/CICIDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'datasets/CICIDS2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'datasets/CICIDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'datasets/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv',
    'datasets/CICIDS2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'datasets/CICIDS2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'datasets/CICIDS2017/Tuesday-WorkingHours.pcap_ISCX.csv',
    'datasets/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv'
]

# Model save paths
OUTPUT_DIR = 'models/cicids_full'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simplify_labels(label):
    """Map CICIDS labels to attack categories"""
    label = str(label).strip().lower()
    if 'benign' in label:
        return 'Benign'
    elif 'ddos' in label or 'dos' in label:
        return 'DoS_DDoS'
    elif 'portscan' in label or 'port scan' in label:
        return 'PortScan'
    elif 'bot' in label:
        return 'Botnet'
    elif 'infiltration' in label or 'infilteration' in label:
        return 'Infiltration'
    elif 'web' in label or 'xss' in label or 'sql' in label:
        return 'WebAttack'
    elif 'brute' in label or 'patator' in label:
        return 'BruteForce'
    elif 'heartbleed' in label:
        return 'Heartbleed'
    else:
        return 'Attack'

def load_cicids_full():
    """Load and combine all CICIDS2017 files"""
    print("=" * 70)
    print("Loading FULL CICIDS2017 Dataset")
    print("=" * 70)
    
    all_dfs = []
    total_rows = 0
    
    for i, file_path in enumerate(CICIDS_FILES, 1):
        print(f"\n[{i}/{len(CICIDS_FILES)}] Loading {file_path.split('/')[-1]}...")
        try:
            df = pd.read_csv(file_path)
            rows = len(df)
            total_rows += rows
            print(f"  ✓ Loaded {rows:,} rows")
            
            if ' Label' in df.columns:
                labels = df[' Label'].value_counts()
                print(f"  Labels: {dict(labels)}")
            
            all_dfs.append(df)
        except Exception as e:
            print(f"  ✗ Error loading file: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"Combining {len(all_dfs)} files...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"✓ Total dataset size: {len(combined_df):,} rows")
    print(f"{'='*70}")
    
    # Simplify labels
    combined_df['Label_Simple'] = combined_df[' Label'].apply(simplify_labels)
    print(f"\nLabel distribution:")
    for label, count in combined_df['Label_Simple'].value_counts().items():
        print(f"  {label}: {count:,}")
    
    return combined_df

def prepare_features(df):
    """Extract and prepare features"""
    # Select important features
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
    print("FULL CICIDS2017 TRAINING - ALL 4 MODELS")
    print("=" * 70)
    
    # Load full dataset
    df = load_cicids_full()
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Encode labels
    print("\n" + "=" * 70)
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Split data
    print("\nSplitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded = train_test_split(
        X, y, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Train size: {X_train.shape[0]:,}, Test size: {X_test.shape[0]:,}")
    
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
    print("TRAINING COMPLETE !")
    print("=" * 70)
    print(f"\nTotal training time: {elapsed_total/60:.2f} minutes")
    print(f"\nFinal Test Accuracies:")
    for model_name, acc in results.items():
        print(f"  {model_name}: {acc*100:.2f}%")
    
    print(f"\nAll models saved to: {OUTPUT_DIR}/")
    print("=" * 70)

if __name__ == '__main__':
    main()
