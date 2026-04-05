#!/usr/bin/env python3
"""
Convert Keras .h5 models to TensorFlow SavedModel format
for compatibility with Keras 3.12.1 and TensorFlow 2.21.0

Usage:
    python scripts/convert_models_to_savedmodel.py
"""
import os
import sys
import tensorflow as tf
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CICIDS_DIR = os.path.join(MODELS_DIR, "cicids_full")

# Models to convert
MODELS_TO_CONVERT = [
    ("CNN-LSTM", os.path.join(CICIDS_DIR, "cnn_lstm_model.h5"), os.path.join(CICIDS_DIR, "cnn_lstm_model")),
    ("CNN", os.path.join(CICIDS_DIR, "cnn_model.h5"), os.path.join(CICIDS_DIR, "cnn_model")),
    ("LSTM", os.path.join(CICIDS_DIR, "lstm_model.h5"), os.path.join(CICIDS_DIR, "lstm_model")),
]

def convert_model(model_name, h5_path, output_dir):
    """Convert a single .h5 model to SavedModel format."""
    print(f"\n{'='*60}")
    print(f"Converting {model_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(h5_path):
        print(f"[ERROR] Model not found: {h5_path}")
        return False
    
    try:
        print(f"[INFO] Loading .h5 model from: {h5_path}")
        
        # Load the model with custom_objects to handle quantization_config
        model = tf.keras.models.load_model(
            h5_path,
            custom_objects=None,
            compile=False,  # Don't require compile during load
        )
        
        print(f"[OK] Model loaded successfully")
        print(f"[INFO] Model structure:")
        model.summary()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[INFO] Saving as SavedModel format to: {output_dir}")
        
        # Save in SavedModel format (TensorFlow native format)
        tf.saved_model.save(model, output_dir)
        
        print(f"[OK] {model_name} converted successfully!")
        print(f"[INFO] Files saved in: {output_dir}")
        
        # Verify the saved model can be loaded
        print(f"[INFO] Verifying conversion...")
        loaded_model = tf.keras.models.load_model(output_dir)
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
    print(f"Keras version: {tf.keras.__version__}")
    print(f"Models directory: {CICIDS_DIR}")
    
    if not os.path.exists(CICIDS_DIR):
        print(f"[ERROR] Models directory not found: {CICIDS_DIR}")
        sys.exit(1)
    
    successful = []
    failed = []
    
    for model_name, h5_path, output_dir in MODELS_TO_CONVERT:
        if convert_model(model_name, h5_path, output_dir):
            successful.append(model_name)
        else:
            failed.append(model_name)
    
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"[OK] Successful: {', '.join(successful) if successful else 'None'}")
    print(f"[FAIL] Failed: {', '.join(failed) if failed else 'None'}")
    
    if failed:
        print("\n[INFO] Update run_test.py MODEL_PATHS to use SavedModel format:")
        print("""
    'CNN-LSTM': first_existing_path(
        os.path.join(BASE_DIR, "models", "cicids_full", "cnn_lstm_model"),  # SavedModel dir
        os.path.join(BASE_DIR, "models", "cnn_lstm_model"),
    ),
""")
        sys.exit(1)
    
    print("\n[SUCCESS] All models converted! Update run_test.py and retrain if needed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
