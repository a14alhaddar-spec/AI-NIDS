#!/usr/bin/env python3
"""
Convert legacy Keras .h5 models to modern .keras format.

This script is intended to be executed in a TensorFlow/Keras 2.13 environment
that can still deserialize older .h5 files, then outputs .keras artifacts that
load cleanly in newer Keras runtimes.

Usage:
    python scripts/convert_models_to_savedmodel.py
"""
import os
import sys
import json
import h5py
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CICIDS_DIR = os.path.join(MODELS_DIR, "cicids_full")

# Models to convert
MODELS_TO_CONVERT = [
    ("CNN-LSTM", os.path.join(CICIDS_DIR, "cnn_lstm_model.h5"), os.path.join(CICIDS_DIR, "cnn_lstm_model.keras")),
    ("CNN", os.path.join(CICIDS_DIR, "cnn_model.h5"), os.path.join(CICIDS_DIR, "cnn_model.keras")),
    ("LSTM", os.path.join(CICIDS_DIR, "lstm_model.h5"), os.path.join(CICIDS_DIR, "lstm_model.keras")),
]

def detect_keras_version():
    """Return keras version string safely across TF/Keras variants."""
    keras_version = getattr(tf.keras, "__version__", None)
    if keras_version:
        return keras_version
    try:
        import keras

        return getattr(keras, "__version__", "unknown")
    except Exception:
        return "unknown"


def _sanitize_keras_config(obj):
    """Recursively remove keys that are unsupported across Keras versions."""
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            if key in {"quantization_config", "optional"}:
                continue
            cleaned[key] = _sanitize_keras_config(value)
        return cleaned
    if isinstance(obj, list):
        return [_sanitize_keras_config(item) for item in obj]
    return obj


def load_keras_model_compat(model_path):
    """Load model with a fallback path for incompatible legacy H5 metadata."""
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as primary_exc:
        if not model_path.endswith(".h5"):
            raise primary_exc

        with h5py.File(model_path, "r") as h5_file:
            raw_config = h5_file.attrs.get("model_config")
            if raw_config is None:
                raise RuntimeError(f"No model_config found in {model_path}") from primary_exc
            if isinstance(raw_config, bytes):
                raw_config = raw_config.decode("utf-8")
            model_config = _sanitize_keras_config(json.loads(raw_config))

        model = tf.keras.models.model_from_json(json.dumps(model_config))
        model.load_weights(model_path)
        return model


def convert_model(model_name, h5_path, output_file):
    """Convert a single .h5 model to .keras format."""
    print(f"\n{'='*60}")
    print(f"Converting {model_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(h5_path):
        print(f"[ERROR] Model not found: {h5_path}")
        return False
    
    try:
        print(f"[INFO] Loading .h5 model from: {h5_path}")
        
        # Load in legacy-compatible runtime with config sanitization fallback.
        model = load_keras_model_compat(h5_path)
        
        print(f"[OK] Model loaded successfully")
        print(f"[INFO] Model structure:")
        model.summary()
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print(f"[INFO] Saving as .keras format to: {output_file}")
        
        # Save in modern, Keras-native format.
        model.save(output_file)
        
        print(f"[OK] {model_name} converted successfully!")
        print(f"[INFO] File saved: {output_file}")
        
        # Verify the saved model can be loaded
        print(f"[INFO] Verifying conversion...")
        loaded_model = tf.keras.models.load_model(output_file, compile=False)
        print(f"[OK] Verification successful - model can be loaded!")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to convert {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Convert all models."""
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"Keras version: {detect_keras_version()}")
    print(f"Models directory: {CICIDS_DIR}")
    
    if not os.path.exists(CICIDS_DIR):
        print(f"[ERROR] Models directory not found: {CICIDS_DIR}")
        sys.exit(1)
    
    successful = []
    failed = []
    
    for model_name, h5_path, output_file in MODELS_TO_CONVERT:
        if convert_model(model_name, h5_path, output_file):
            successful.append(model_name)
        else:
            failed.append(model_name)
    
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"[OK] Successful: {', '.join(successful) if successful else 'None'}")
    print(f"[FAIL] Failed: {', '.join(failed) if failed else 'None'}")
    
    if failed:
        print("\n[INFO] Ensure run_test.py MODEL_PATHS includes .keras paths:")
        print("""
    'CNN-LSTM': first_existing_path(
        os.path.join(BASE_DIR, "models", "cicids_full", "cnn_lstm_model.keras"),
        os.path.join(BASE_DIR, "models", "cnn_lstm_model.keras"),
    ),
""")
        sys.exit(1)
    
    print("\n[SUCCESS] All models converted to .keras format.")
    sys.exit(0)

if __name__ == "__main__":
    main()
