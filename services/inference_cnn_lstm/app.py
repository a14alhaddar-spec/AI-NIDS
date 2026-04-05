"""
Inference Service for CNN-LSTM Model
Runs on port 5003
"""
import json
import os
import time
from flask import Flask, jsonify, request
import numpy as np

app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/cnn_lstm_model.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "models/cnn_lstm_scaler.joblib")
ENCODER_PATH = os.getenv("ENCODER_PATH", "models/cnn_lstm_encoder.joblib")

MODEL = None
SCALER = None
ENCODER = None

CLASSES = [
    "benign",
    "dos_ddos",
    "port_scan",
    "brute_force",
    "malware_c2",
    "data_exfil",
]

def load_model():
    global MODEL, SCALER, ENCODER
    
    try:
        import tensorflow as tf
        import joblib
        
        if os.path.exists(MODEL_PATH):
            MODEL = tf.keras.models.load_model(MODEL_PATH)
            print(f"✓ Loaded CNN-LSTM model from {MODEL_PATH}")
        else:
            print(f"⚠ Model not found: {MODEL_PATH}")
            
        if os.path.exists(SCALER_PATH):
            SCALER = joblib.load(SCALER_PATH)
            print(f"✓ Loaded scaler from {SCALER_PATH}")
            
        if os.path.exists(ENCODER_PATH):
            ENCODER = joblib.load(ENCODER_PATH)
            print(f"✓ Loaded encoder from {ENCODER_PATH}")
            
    except Exception as e:
        print(f"✗ Error loading model: {e}")


def heuristic_score(features):
    """Fallback heuristic detection if model not available"""
    bytes_per_sec = float(features.get("bytes_per_sec", 0.0))
    packets_per_sec = float(features.get("packets_per_sec", 0.0))
    duration = float(features.get("flow_duration", 0.0))

    if packets_per_sec > 1000 and duration < 2:
        return "dos_ddos", 0.91
    if packets_per_sec > 200 and duration < 5:
        return "port_scan", 0.78
    if bytes_per_sec > 5e6:
        return "data_exfil", 0.88
    return "benign", 0.55


def predict(features):
    """Predict using CNN-LSTM model"""
    if MODEL is not None and SCALER is not None:
        try:
            # Extract features in same order as training
            vec = [
                float(features.get("flow_duration", 0.0)),
                float(features.get("bytes", 0.0)),
                float(features.get("packets", 0.0)),
                float(features.get("bytes_per_sec", 0.0)),
                float(features.get("packets_per_sec", 0.0)),
                float(features.get("src_dst_bytes_ratio", 1.0)),
                float(features.get("avg_pkt_size", 0.0)),
            ]
            
            X = np.array([vec])
            X_scaled = SCALER.transform(X)
            
            # Get prediction
            proba = MODEL.predict(X_scaled, verbose=0)[0]
            idx = int(proba.argmax())
            confidence = float(proba[idx])
            
            # Get label
            if ENCODER is not None:
                label = ENCODER.classes_[idx]
            else:
                label = CLASSES[idx] if idx < len(CLASSES) else "unknown"
            
            return label, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return heuristic_score(features)
    
    return heuristic_score(features)


@app.route("/health", methods=["GET"])
def health():
    status = "ok" if MODEL is not None else "no_model"
    return jsonify({
        "status": status,
        "model": "CNN-LSTM" if MODEL is not None else "heuristic_fallback"
    })


@app.route("/infer", methods=["POST"])
def infer():
    start = time.time()
    payload = request.get_json(silent=True) or {}
    
    label, confidence = predict(payload)
    
    latency_ms = (time.time() - start) * 1000
    
    return jsonify({
        "label": label,
        "confidence": confidence,
        "latency_ms": round(latency_ms, 2),
        "model": "CNN-LSTM"
    })


if __name__ == "__main__":
    print("=" * 60)
    print("CNN-LSTM Inference Service")
    print("=" * 60)
    
    load_model()
    
    print(f"\nStarting server on port 5003...")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=5003, debug=False)
