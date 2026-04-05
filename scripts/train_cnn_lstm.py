"""
Train CNN-LSTM Model for Network Intrusion Detection
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.utils import to_categorical
except ImportError:
    print("✗ TensorFlow not installed. Install with: pip install tensorflow")
    sys.exit(1)


def build_cnn_lstm(input_shape, num_classes, learning_rate=1e-3):
    """Build CNN-LSTM hybrid model"""
    inputs = layers.Input(shape=(input_shape,))
    
    # Reshape for CNN
    x = layers.Reshape((input_shape, 1))(inputs)
    
    # CNN layers
    x = layers.Conv1D(64, 3, activation="relu", padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation="relu", padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    
    # LSTM layers
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)
    
    # Dense layers
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


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
    print("CNN-LSTM Model Training - CICIDS2017")
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
    
    # Simplify labels
    df['label_simple'] = df[label_col].apply(simplify_labels)
    print(f"  Classes: {df['label_simple'].unique()}")
    print(f"  Label distribution:")
    for label, count in df['label_simple'].value_counts().items():
        print(f"    {label}: {count:,}")
    
    # Select numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in [label_col, 'label_simple']]
    
    # Handle infinite values and NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)
    
    print(f"\n  Using {len(feature_cols)} numeric features")
    print(f"  After cleaning: {len(df):,} samples")
    
    # Select features (use first 7 for consistency with RF)
    feature_names = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow Bytes/s', 'Flow Packets/s', 
        'Average Packet Size', 'Fwd Packet Length Mean'
    ]
    
    selected_features = []
    for feat in feature_names:
        matches = [c for c in feature_cols if feat.lower() in c.lower()]
        if matches:
            selected_features.append(matches[0])
    
    if len(selected_features) < 5:
        selected_features = feature_cols[:7]
    
    print(f"  Selected features: {selected_features}")
    
    X = df[selected_features].values
    y = df['label_simple'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    print(f"\n  Classes ({num_classes}): {list(label_encoder.classes_)}")
    
    # Split data
    print("\nStep 2: Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    
    # Scale features
    print("\nStep 3: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    # Build model
    print("\nStep 4: Building CNN-LSTM model...")
    input_shape = X_train_scaled.shape[1]
    model = build_cnn_lstm(input_shape, num_classes)
    
    print(f"  Input shape: {input_shape}")
    print(f"  Output classes: {num_classes}")
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    print("\nStep 5: Training model...")
    print("  Epochs: 10, Batch size: 128")
    
    history = model.fit(
        X_train_scaled, y_train_cat,
        validation_split=0.2,
        epochs=10,
        batch_size=128,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )
    
    # Evaluate
    print("\nStep 6: Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Test loss: {test_loss:.4f}")
    
    # Save model
    print("\nStep 7: Saving model and artifacts...")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "cnn_lstm_model.h5"
    scaler_path = model_dir / "cnn_lstm_scaler.joblib"
    encoder_path = model_dir / "cnn_lstm_encoder.joblib"
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)
    
    print(f"  ✓ Model saved: {model_path}")
    print(f"  ✓ Scaler saved: {scaler_path}")
    print(f"  ✓ Encoder saved: {encoder_path}")
    
    print("\n" + "=" * 70)
    print("✓ CNN-LSTM Training Complete!")
    print("=" * 70)
    
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
