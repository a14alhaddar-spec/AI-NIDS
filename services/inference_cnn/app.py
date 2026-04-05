"""
CNN Inference Service
Provides real-time threat detection using trained CNN model
"""
from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
import time

app = Flask(__name__)

# Paths
MODEL_PATH = 'models/cnn_model.h5'
SCALER_PATH = 'models/cnn_scaler.joblib'
ENCODER_PATH = 'models/cnn_encoder.joblib'

# Global model variables
MODEL = None
SCALER = None
ENCODER = None

def load_model():
    """Load the CNN model and preprocessing artifacts"""
    global MODEL, SCALER, ENCODER
    try:
        if os.path.exists(MODEL_PATH):
            import tensorflow as tf
            MODEL = tf.keras.models.load_model(MODEL_PATH)
            print(f"✓ Loaded CNN model from {MODEL_PATH}")
        else:
            print(f"⚠  Model file not found: {MODEL_PATH}")
            print("   Using heuristic fallback")
            
        if os.path.exists(SCALER_PATH):
            SCALER = joblib.load(SCALER_PATH)
            print(f"✓ Loaded scaler from {SCALER_PATH}")
            
        if os.path.exists(ENCODER_PATH):
            ENCODER = joblib.load(ENCODER_PATH)
            print(f"✓ Loaded encoder from {ENCODER_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using heuristic fallback")

def heuristic_score(flow_data):
    """Simple heuristic scoring when model unavailable"""
    score = 0.0
    
    # Check destination port (common attack ports)
    dst_port = flow_data.get('Destination Port', 0)
    if dst_port in [22, 23, 3389, 445, 135, 139]:  # SSH, Telnet, RDP, SMB
        score += 0.3
    
    # Check packet counts (DDoS indicators)
    total_packets = flow_data.get('Total Fwd Packets', 0) + flow_data.get('Total Backward Packets', 0)
    if total_packets > 1000:
        score += 0.4
    
    # Check flow duration (very short or very long could be suspicious)
    duration = flow_data.get('Flow Duration', 0)
    if duration < 1000 or duration > 120000:
        score += 0.2
    
    # Check bytes/second ratio
    bytes_sec = flow_data.get('Flow Bytes/s', 0)
    if bytes_sec > 100000:  # High throughput
        score += 0.3
    
    return min(score, 1.0)

def predict(flow_data):
    """Make prediction using CNN model or heuristic"""
    start_time = time.time()
    
    # Extract features in correct order
    features = np.array([[
        flow_data.get('Destination Port', 0),
        flow_data.get('Flow Duration', 0),
        flow_data.get('Total Fwd Packets', 0),
        flow_data.get('Total Backward Packets', 0),
        flow_data.get('Total Length of Fwd Packets', 0),
        flow_data.get('Total Length of Bwd Packets', 0),
        flow_data.get('Flow Bytes/s', 0)
    ]])
    
    # Use CNN model if available
    if MODEL is not None and SCALER is not None and ENCODER is not None:
        try:
            features_scaled = SCALER.transform(features)
            prediction_probs = MODEL.predict(features_scaled, verbose=0)[0]
            predicted_class_idx = np.argmax(prediction_probs)
            predicted_label = ENCODER.inverse_transform([predicted_class_idx])[0]
            confidence = float(prediction_probs[predicted_class_idx])
            
            inference_time = (time.time() - start_time) * 1000
            
            return {
                'threat_detected': predicted_label != 'Benign',
                'threat_type': predicted_label,
                'confidence': confidence,
                'inference_time_ms': round(inference_time, 2),
                'model': 'CNN',
                'source': 'ml_model'
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            pass
    
    # Fallback to heuristic
    score = heuristic_score(flow_data)
    inference_time = (time.time() - start_time) * 1000
    
    return {
        'threat_detected': score > 0.5,
        'threat_type': 'Suspicious' if score > 0.5 else 'Benign',
        'confidence': score,
        'inference_time_ms': round(inference_time, 2),
        'model': 'CNN (Heuristic)',
        'source': 'heuristic'
    }

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'scaler_loaded': SCALER is not None,
        'encoder_loaded': ENCODER is not None,
        'model_type': 'CNN'
    })

@app.route('/infer', methods=['POST'])
def infer():
    """Inference endpoint"""
    try:
        data = request.get_json()
        result = predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("CNN Inference Service")
    print("=" * 60)
    load_model()
    print("\nStarting server on http://0.0.0.0:5004")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5004, debug=False)
