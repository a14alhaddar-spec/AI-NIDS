"""
CNN Inference Service for CICIDS2017
Loads the full CICIDS CNN model
"""

from flask import Flask, jsonify, request
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import os
import sys

app = Flask(__name__)

# Shared metrics setup
try:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if base_dir not in sys.path:
        sys.path.append(base_dir)
    from shared_metrics import add_prediction
    METRICS_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Metrics unavailable: {e}")
    METRICS_AVAILABLE = False

# Load CNN model
MODEL_DIR = "models/cicids_full"
print("\n" + "="*60)
print("Loading CNN Model...")
print("="*60)

try:
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "cnn_model.h5"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    
    print(f"✅ Model loaded: {model.input_shape} input shape")
    print(f"✅ Classes: {label_encoder.classes_}")
    print(f"✅ Scaler ready")
    
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Error loading model: {e}")
    MODEL_LOADED = False
    model = None
    scaler = None
    label_encoder = None

# Expected features
FEATURE_COLUMNS = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Total Length of Fwd Packets',
    'Total Length of Bwd Packets', 'Flow Bytes/s'
]

@app.route('/health', methods=['GET'])
def health():
    if MODEL_LOADED:
        return jsonify({"status": "ok", "model": "loaded"})
    else:
        return jsonify({"status": "error", "model": "not loaded"}), 503

@app.route('/infer', methods=['POST'])
def infer():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        features_dict = request.get_json()
        
        # Extract features
        feature_values = []
        for col in FEATURE_COLUMNS:
            val = features_dict.get(col, features_dict.get(col.strip(), 0.0))
            if pd.isna(val) or val == float('inf') or val == float('-inf'):
                val = 0.0
            feature_values.append(float(val))
        
        # Prepare for CNN (needs shape: samples, features, 1)
        X = np.array([feature_values])
        X_scaled = scaler.transform(X)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        
        # Predict
        probabilities = model.predict(X_reshaped, verbose=0)[0]
        prediction_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction_idx])
        
        prediction_label = label_encoder.classes_[prediction_idx]
        threat_detected = prediction_label.lower() not in ['benign', 'normal']
        
        # Track metrics
        if METRICS_AVAILABLE:
            try:
                add_prediction(prediction_label, confidence, threat_detected)
            except Exception as e:
                print(f"[WARN] Metrics write failed: {e}")
        
        return jsonify({
            "prediction": prediction_label,
            "confidence": confidence,
            "threat_detected": threat_detected,
            "model": "CNN-CICIDS"
        })
        
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CNN Inference Service - CICIDS Full")
    print("="*60)
    print("Listening on port 5004...")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5004, debug=False)
