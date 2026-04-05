"""
Simple Random Forest Inference Service for CICIDS2017
Loads the full CICIDS model and handles all 78 features
"""

from flask import Flask, jsonify, request
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

# Load CICIDS full model
MODEL_DIR = "models/cicids_full"
print("\n" + "="*60)
print("Loading CICIDS Random Forest Model...")
print("="*60)

try:
    model = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    
    print(f"✅ Model loaded: {model.n_features_in_} features")
    print(f"✅ Classes: {label_encoder.classes_}")
    print(f"✅ Scaler ready")
    
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Error loading model: {e}")
    MODEL_LOADED = False
    model = None
    scaler = None
    label_encoder = None

# Expected feature columns (from CICIDS training)
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
        # Get features from request
        features_dict = request.get_json()
        
        print(f"\n[DEBUG] Received features: {list(features_dict.keys())}")
        
        # Extract the 7 features we use (with proper column names)
        feature_values = []
        for col in FEATURE_COLUMNS:
            # Try both with and without spaces
            val = features_dict.get(col, features_dict.get(col.strip(), 0.0))
            
            # Handle NaN and infinity
            if pd.isna(val) or val == float('inf') or val == float('-inf'):
                val = 0.0
            
            feature_values.append(float(val))
            print(f"[DEBUG] {col}: {val}")
        
        # Create feature array
        X = np.array([feature_values])
        print(f"[DEBUG] Feature array shape: {X.shape}")
        
        # Scale features
        X_scaled = scaler.transform(X)
        print(f"[DEBUG] Scaled shape: {X_scaled.shape}")
        
        # Predict
        prediction_label = model.predict(X_scaled)[0]  # Returns label string directly!
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Find the index of the predicted class to get confidence
        predicted_classes = model.classes_
        prediction_idx = list(predicted_classes).index(prediction_label)
        confidence = float(probabilities[prediction_idx])
        
        print(f"[DEBUG] Prediction: {prediction_label} ({confidence:.2%})")
        
        # Determine if threat
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
            "model": "RandomForest-CICIDS"
        })
        
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Random Forest Inference Service - CICIDS Full")
    print("="*60)
    print("Listening on port 5002...")
    print("Ready for predictions!")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5002, debug=False)
