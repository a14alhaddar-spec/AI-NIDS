#!/usr/bin/env python3
"""
Direct test of inference services with real CICIDS data
Primary Model: CNN-LSTM (Hybrid CNN-LSTM Deep Learning Model)

Performs DIRECT inference - no HTTP services required!

Usage:
    python run_test.py                      # Interactive menu mode
    python run_test.py --attack DDoS        # Test specific attack
    python run_test.py --attack DDoS --attack PortScan  # Test multiple attacks
    python run_test.py --all-attacks        # Test all attack types
    python run_test.py --benign             # Test benign traffic only
    python run_test.py --list               # List available attack types
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime
from threading import RLock

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "CICIDS2017")

# Import shared_metrics for updating dashboard metrics
sys.path.insert(0, BASE_DIR)
from shared_metrics import add_prediction

# Primary model is CNN-LSTM
PRIMARY_MODEL = "CNN-LSTM"
METRICS_MODE = os.getenv("NIDS_METRICS_MODE", "offline").strip().lower() or "offline"

def first_existing_path(*candidate_paths):
    for candidate_path in candidate_paths:
        if candidate_path and os.path.exists(candidate_path):
            return candidate_path
    return None


# Model paths for CICIDS2017 models
# Check for SavedModel format (directory) first, then .h5 files
MODEL_PATHS = {
    'CNN-LSTM': first_existing_path(
        os.path.join(BASE_DIR, "models", "cicids_full", "cnn_lstm_model"),  # SavedModel directory
        os.path.join(BASE_DIR, "models", "cicids_full", "cnn_lstm_model.h5"),  # .h5 file
        os.path.join(BASE_DIR, "models", "cnn_lstm_model"),
        os.path.join(BASE_DIR, "models", "cnn_lstm_model.h5"),
    ),
    'CNN': first_existing_path(
        os.path.join(BASE_DIR, "models", "cicids_full", "cnn_model"),
        os.path.join(BASE_DIR, "models", "cicids_full", "cnn_model.h5"),
        os.path.join(BASE_DIR, "models", "cnn_model"),
        os.path.join(BASE_DIR, "models", "cnn_model.h5"),
    ),
    'LSTM': first_existing_path(
        os.path.join(BASE_DIR, "models", "cicids_full", "lstm_model"),
        os.path.join(BASE_DIR, "models", "cicids_full", "lstm_model.h5"),
        os.path.join(BASE_DIR, "models", "lstm_model"),
        os.path.join(BASE_DIR, "models", "lstm_model.h5"),
    ),
    'Random Forest': first_existing_path(
        os.path.join(BASE_DIR, "models", "cicids_full", "rf_model.joblib"),
        os.path.join(BASE_DIR, "models", "model.joblib"),
    ),
}

SCALER_PATH = first_existing_path(
    os.path.join(BASE_DIR, "models", "cicids_full", "scaler.joblib"),
    os.path.join(BASE_DIR, "models", "scaler.joblib"),
)
ENCODER_PATH = first_existing_path(
    os.path.join(BASE_DIR, "models", "cicids_full", "label_encoder.joblib"),
    os.path.join(BASE_DIR, "models", "label_encoder.joblib"),
)

# Loaded models
MODELS = {}
SCALER = None
ENCODER = None

# Feature names as expected by the scaler (exact names from training)
FEATURE_NAMES = [
    ' Destination Port',
    ' Flow Duration',
    ' Total Fwd Packets',
    ' Total Backward Packets',
    'Total Length of Fwd Packets',
    ' Total Length of Bwd Packets',
    'Flow Bytes/s',
]

# Mapping from run_test.py feature names to dataset column names
FEATURE_MAP = {
    "Destination Port": " Destination Port",
    "Flow Duration": " Flow Duration",
    "Total Fwd Packets": " Total Fwd Packets",
    "Total Backward Packets": " Total Backward Packets",
    "Total Length of Fwd Packets": "Total Length of Fwd Packets",
    "Total Length of Bwd Packets": " Total Length of Bwd Packets",
    "Flow Bytes/s": "Flow Bytes/s",
}

# Model-specific results for run_test.py
MODEL_RESULTS_DIR = os.path.join(BASE_DIR, "models", "results", "run_test")
MODEL_RESULTS_PATHS = {
    'CNN-LSTM': os.path.join(MODEL_RESULTS_DIR, "cnn_lstm_results.json"),
    'CNN': os.path.join(MODEL_RESULTS_DIR, "cnn_results.json"),
    'LSTM': os.path.join(MODEL_RESULTS_DIR, "lstm_results.json"),
    'Random Forest': os.path.join(MODEL_RESULTS_DIR, "random_forest_results.json"),
}

MODEL_LOCKS = {
    'CNN-LSTM': RLock(),
    'CNN': RLock(),
    'LSTM': RLock(),
    'Random Forest': RLock(),
}

def load_models():
    """Load all models and preprocessing artifacts."""
    global MODELS, SCALER, ENCODER

    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)

    # Load scaler and encoder when available; keep going if they are missing.
    if SCALER_PATH:
        try:
            SCALER = joblib.load(SCALER_PATH)
            print(f"[OK] Scaler loaded: {SCALER_PATH}")
        except Exception as exc:
            print(f"[WARN] Error loading scaler: {exc}")
            SCALER = None
    else:
        print("[WARN] No scaler file found. Heuristic mode will be used when needed.")

    if ENCODER_PATH:
        try:
            ENCODER = joblib.load(ENCODER_PATH)
            print(f"[OK] Label encoder loaded: {ENCODER_PATH}")
            print(f"  Classes: {list(ENCODER.classes_)}")
        except Exception as exc:
            print(f"[WARN] Error loading encoder: {exc}")
            ENCODER = None
    else:
        print("[WARN] No label encoder file found. Heuristic mode will be used when needed.")

    # Load each model when present.
    for model_name, model_path in MODEL_PATHS.items():
        try:
            if not model_path:
                raise FileNotFoundError(f"No artifact found for {model_name}")

            if model_path.endswith('.joblib'):
                # Load scikit-learn model
                MODELS[model_name] = joblib.load(model_path)
            elif os.path.isdir(model_path):
                # Load SavedModel format (TensorFlow directory)
                import tensorflow as tf
                MODELS[model_name] = tf.keras.models.load_model(model_path)
            else:
                # Load .h5 file (Keras format)
                import tensorflow as tf
                MODELS[model_name] = tf.keras.models.load_model(model_path)

            print(f"[OK] {model_name} loaded: {model_path}")
            reset_model_metrics(model_name)
            print(f"[OK] Reset metrics for {model_name}")
        except Exception as exc:
            print(f"[WARN] {model_name} unavailable: {exc}")
            MODELS[model_name] = None

    if not any(model is not None for model in MODELS.values()):
        print("[WARN] No trained models are available. Heuristic predictions will be used.")

    print("=" * 60)
    return True


def heuristic_score(features):
    """
    Fallback rule-based threat detection when models unavailable.
    Returns: (prediction_label, confidence)
    """
    # Extract features for heuristic analysis
    flow_bytes_sec = features.get('Flow Bytes/s', 0.0)
    total_fwd_packets = features.get(' Total Fwd Packets', 0.0)
    total_bwd_packets = features.get(' Total Backward Packets', 0.0)
    flow_duration = features.get(' Flow Duration', 0.0)
    total_fwd_len = features.get('Total Length of Fwd Packets', 0.0)
    total_bwd_len = features.get(' Total Length of Bwd Packets', 0.0)
    dst_port = features.get(' Destination Port', 0.0)
    
    # Simple anomaly scoring
    threat_score = 0.0
    threat_reasons = []
    
    # High data rate (DDoS/DoS indicator)
    if flow_bytes_sec > 100000:
        threat_score += 0.3
        threat_reasons.append("High data rate")
    
    # High packet count (potential attack)
    total_packets = total_fwd_packets + total_bwd_packets
    if total_packets > 1000:
        threat_score += 0.25
        threat_reasons.append("High packet count")
    
    # Imbalanced packet flow (port scan or specific attacks)
    if total_packets > 0:
        packet_ratio = total_fwd_packets / total_packets if total_packets > 0 else 0
        if packet_ratio > 0.9 or packet_ratio < 0.1:
            threat_score += 0.2
            threat_reasons.append("Imbalanced packet flow")
    
    # Very short duration with data (scan-like behavior)
    if flow_duration < 1000 and total_packets > 10:
        threat_score += 0.15
        threat_reasons.append("Short duration with activity")
    
    # Port-based indicators
    if dst_port in [22, 21, 23]:  # SSH, FTP, Telnet
        threat_score += 0.1
        threat_reasons.append("Sensitive port access")
    
    # Determine label and confidence
    if threat_score >= 0.6:
        # More likely to be a threat
        label = 'DoS_DDoS'  # Default threat type
        confidence = min(threat_score, 1.0)
    elif threat_score >= 0.3:
        label = 'PortScan'  # Suspicious activity
        confidence = threat_score
    else:
        label = 'Benign'
        confidence = 1.0 - threat_score
    
    return label, confidence


def predict_with_model(model_name, features):
    """
    Make prediction using a specific model
    features: dict with keys matching FEATURE_NAMES
    Returns: (prediction_label, confidence)
    """
    global MODELS, SCALER, ENCODER

    model = MODELS.get(model_name)
    if model is None or SCALER is None or ENCODER is None:
        return heuristic_score(features)

    try:
        # Prepare feature vector in exact order expected by scaler
        feature_vector = np.array([[features.get(fn, 0.0) for fn in FEATURE_NAMES]])

        # Scale the features
        feature_scaled = SCALER.transform(feature_vector)

        # Get prediction based on model type
        if model_name == 'Random Forest':
            proba = model.predict_proba(feature_scaled)[0]
            idx = int(proba.argmax())
            confidence = float(proba[idx])
            label = ENCODER.classes_[idx]
        else:
            proba = model.predict(feature_scaled, verbose=0)[0]
            idx = int(proba.argmax())
            confidence = float(proba[idx])
            label = ENCODER.classes_[idx]

        return label, confidence
    except Exception as exc:
        print(f"  Prediction error for {model_name}: {exc}")
        return heuristic_score(features)

def reset_model_metrics(model_name):
    """Reset metrics for specific model"""
    path = MODEL_RESULTS_PATHS[model_name]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    initial_metrics = {
        "total_predictions": 0,
        "threats_detected": 0,
        "benign_count": 0,
        "threat_types": {},
        "recent_predictions": [],
        "last_update": datetime.now().isoformat()
    }
    save_model_metrics(initial_metrics, model_name)

def load_model_metrics(model_name):
    """Load metrics for specific model"""
    path = MODEL_RESULTS_PATHS[model_name]
    try:
        with open(path, 'r') as f:
            metrics = json.load(f)
            metrics["recent_predictions"] = metrics.get("recent_predictions", [])
            return metrics
    except:
        return {"total_predictions": 0, "threats_detected": 0, "benign_count": 0, "threat_types": {}, "recent_predictions": [], "last_update": ""}

def save_model_metrics(metrics, model_name):
    """Save metrics for specific model"""
    path = MODEL_RESULTS_PATHS[model_name]
    with MODEL_LOCKS[model_name]:
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)

def add_model_prediction(model_name, prediction, confidence, threat_detected):
    """Add prediction to model-specific metrics"""
    try:
        metrics = load_model_metrics(model_name)
        
        metrics["total_predictions"] += 1
        if threat_detected:
            metrics["threats_detected"] += 1
            threat_type = prediction
            metrics["threat_types"][threat_type] = metrics["threat_types"].get(threat_type, 0) + 1
        else:
            metrics["benign_count"] += 1
        
        # Add to recent predictions (keep last 50)
        now = datetime.now()
        recent = {
            "timestamp": int(now.timestamp()),
            "label": prediction,
            "confidence": confidence,
            "is_threat": threat_detected
        }
        metrics["recent_predictions"].insert(0, recent)
        metrics["recent_predictions"] = metrics["recent_predictions"][:50]
        metrics["last_update"] = now.isoformat()
        
        save_model_metrics(metrics, model_name)
    except Exception as e:
        print(f"Error updating model metrics for {model_name}: {e}")


# Attack type to CSV file mapping
ATTACK_MAPPING = {
    "BENIGN": {
        "files": ["Monday-WorkingHours.pcap_ISCX.csv"],
        "label": "BENIGN",
        "count": 3
    },
    "PortScan": {
        "files": ["Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"],
        "label": "PortScan",
        "count": 5
    },
    "DDoS": {
        "files": ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"],
        "label": "DDoS",
        "count": 5
    },
    "Bot": {
        "files": ["Friday-WorkingHours-Morning.pcap_ISCX.csv"],
        "label": "Bot",
        "count": 5
    },
    "DoS GoldenEye": {
        "files": ["Wednesday-workingHours.pcap_ISCX.csv"],
        "label": "DoS GoldenEye",
        "count": 5
    },
    "DoS Hulk": {
        "files": ["Wednesday-workingHours.pcap_ISCX.csv"],
        "label": "DoS Hulk",
        "count": 5
    },
    "DoS Slowhttptest": {
        "files": ["Wednesday-workingHours.pcap_ISCX.csv"],
        "label": "DoS Slowhttptest",
        "count": 5
    },
    "DoS slowloris": {
        "files": ["Wednesday-workingHours.pcap_ISCX.csv"],
        "label": "DoS slowloris",
        "count": 5
    },
    "FTP-Patator": {
        "files": ["Tuesday-WorkingHours.pcap_ISCX.csv"],
        "label": "FTP-Patator",
        "count": 5
    },
    "Heartbleed": {
        "files": ["Wednesday-workingHours.pcap_ISCX.csv"],
        "label": "Heartbleed",
        "count": 5
    },
    "Infiltration": {
        "files": ["Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"],
        "label": "Infiltration",
        "count": 5
    },
    "SSH-Patator": {
        "files": ["Tuesday-WorkingHours.pcap_ISCX.csv"],
        "label": "SSH-Patator",
        "count": 5
    },
    "Web Attack - Brute Force": {
        "files": ["Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"],
        "label": "Web Attack � Brute Force",
        "count": 5
    },
    "Web Attack - Sql Injection": {
        "files": ["Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"],
        "label": "Web Attack � Sql Injection",
        "count": 5
    },
    "Web Attack - XSS": {
        "files": ["Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"],
        "label": "Web Attack � XSS",
        "count": 5
    },
}

# Display names for attack types
ATTACK_DISPLAY_NAMES = {
    "BENIGN": "BENIGN (Normal Traffic)",
    "PortScan": "PortScan",
    "DDoS": "DDoS",
    "Bot": "Bot",
    "DoS GoldenEye": "DoS GoldenEye",
    "DoS Hulk": "DoS Hulk",
    "DoS Slowhttptest": "DoS Slowhttptest",
    "DoS slowloris": "DoS slowloris",
    "FTP-Patator": "FTP-Patator",
    "Heartbleed": "Heartbleed",
    "Infiltration": "Infiltration",
    "SSH-Patator": "SSH-Patator",
    "Web Attack - Brute Force": "Web Attack - Brute Force",
    "Web Attack - Sql Injection": "Web Attack - Sql Injection",
    "Web Attack - XSS": "Web Attack - XSS",
}

def to_number(value):
    try:
        number = float(value)
        if pd.isna(number):
            return 0.0
        return number
    except:
        return 0.0

def load_sample(file_path, label, count=1):
    """Load samples from CICIDS file"""
    try:
        df = pd.read_csv(file_path)
        filtered = df[df[" Label"].astype(str).str.strip() == label]
        return filtered.head(count)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def load_attack_samples(attack_type, count=None):
    """Load samples for a specific attack type"""
    if attack_type not in ATTACK_MAPPING:
        print(f"Unknown attack type: {attack_type}")
        return pd.DataFrame()
    
    config = ATTACK_MAPPING[attack_type]
    label = config["label"]
    sample_count = count if count else config["count"]
    
    all_samples = pd.DataFrame()
    for filename in config["files"]:
        file_path = os.path.join(DATASET_DIR, filename)
        samples = load_sample(file_path, label, sample_count)
        all_samples = pd.concat([all_samples, samples], ignore_index=True)
    
    return all_samples.head(sample_count)

def test_sample(features, label):
    """Test sample with all loaded models."""
    print(f"\n{'='*60}")
    print(f"Testing sample - Actual: {label}")
    print(f"{'='*60}")

    # Prepare features in the exact order expected by the scaler
    feature_dict = {}
    for target_key, source_key in FEATURE_MAP.items():
        feature_dict[source_key] = to_number(features.get(source_key, 0))

    print("Features:", {k: round(v, 2) for k, v in feature_dict.items()})

    # Test each model
    for model_name in ['CNN-LSTM', 'Random Forest', 'CNN', 'LSTM']:
        if model_name not in MODELS or MODELS[model_name] is None:
            print(f"  {model_name:15} NOT LOADED - using heuristics")

        pred, conf = predict_with_model(model_name, feature_dict)

        # Determine if threat
        is_threat = pred.lower() not in ['benign', 'error', 'unknown']
        icon = "DETECTED" if is_threat else "BENIGN"

        # Highlight primary model
        if model_name == PRIMARY_MODEL:
            print(f"  >>> {model_name:15} {icon:10} {pred:15} {conf*100:5.1f}% <<<")
        else:
            print(f"  {model_name:15} {icon:10} {pred:15} {conf*100:5.1f}%")

        # Update model-specific metrics (primary)
        add_model_prediction(model_name, pred, conf, is_threat)
        # Keep shared metrics for dashboard (backward compat)
        add_prediction(pred, conf, is_threat, model_name, {"mode": METRICS_MODE})

def test_attack_type(attack_type, sample_count=None):
    """Test a specific attack type"""
    print(f"\n{'='*60}")
    print(f"TESTING: {attack_type}")
    print(f"{'='*60}")
    
    samples = load_attack_samples(attack_type, sample_count)
    
    if samples.empty:
        print(f"No samples found for {attack_type}")
        return
    
    for idx, row in samples.iterrows():
        test_sample(row, attack_type)
        time.sleep(0.5)

def list_attack_types():
    """List all available attack types"""
    print("\nAvailable attack types:")
    print("=" * 50)
    for i, (attack_type, display_name) in enumerate(ATTACK_DISPLAY_NAMES.items(), 1):
        print(f"  {i:2}. {display_name}")
    print("=" * 50)
    print(f"Total: {len(ATTACK_DISPLAY_NAMES)} attack types + BENIGN")

def interactive_menu():
    """Show interactive menu for selecting attack types"""
    print("\n" + "=" * 60)
    print("       AI-NIDS-SOAR Attack Testing Menu")
    print("=" * 60)
    print("\nAvailable options:")
    print("  1. Test a specific attack type")
    print("  2. Test multiple attack types")
    print("  3. Test ALL attack types")
    print("  4. Test BENIGN traffic only")
    print("  5. List all available attack types")
    print("  6. Exit")
    print("\n" + "=" * 60)
    
    choice = input("Enter your choice (1-6): ").strip()
    
    if choice == "1":
        list_attack_types()
        attack = input("\nEnter attack type name: ").strip()
        if attack in ATTACK_MAPPING:
            test_attack_type(attack)
        else:
            print(f"Invalid attack type: {attack}")
    elif choice == "2":
        list_attack_types()
        attacks_input = input("\nEnter attack types (comma-separated): ").strip()
        attacks = [a.strip() for a in attacks_input.split(",")]
        for attack in attacks:
            if attack in ATTACK_MAPPING:
                test_attack_type(attack)
            else:
                print(f"Skipping invalid attack type: {attack}")
    elif choice == "3":
        print("\nTesting ALL attack types...")
        for attack_type in ATTACK_MAPPING.keys():
            test_attack_type(attack_type)
    elif choice == "4":
        test_attack_type("BENIGN")
    elif choice == "5":
        list_attack_types()
    elif choice == "6":
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice!")

def main():
    parser = argparse.ArgumentParser(
        description="Test inference services with CICIDS2017 attack data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_test.py                      # Interactive menu
  python run_test.py --attack DDoS         # Test DDoS
  python run_test.py --attack PortScan --attack DDoS  # Test multiple
  python run_test.py --all-attacks        # Test all attacks
  python run_test.py --benign             # Test benign only
  python run_test.py --list               # List attack types
  python run_test.py --attack DDoS --samples 10  # Custom sample count
        """,
    )

    parser.add_argument(
        "--attack", "-a",
        action="append",
        dest="attacks",
        help="Attack type to test (can be specified multiple times)",
    )
    parser.add_argument(
        "--all-attacks",
        action="store_true",
        help="Test all attack types",
    )
    parser.add_argument(
        "--benign", "-b",
        action="store_true",
        help="Test BENIGN traffic only",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available attack types and exit",
    )
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=None,
        help="Number of samples to test per attack type",
    )

    args = parser.parse_args()

    if args.list:
        list_attack_types()
        return

    load_models()

    if not (args.attacks or args.all_attacks or args.benign):
        interactive_menu()
        return

    if args.attacks:
        for attack in args.attacks:
            if attack in ATTACK_MAPPING:
                test_attack_type(attack, args.samples)
            else:
                print(f"Unknown attack type: {attack}")
                print("Use --list to see available attack types")

    if args.all_attacks:
        print("\n*** Testing ALL attack types ***")
        for attack_type in ATTACK_MAPPING.keys():
            test_attack_type(attack_type, args.samples)

    if args.benign:
        test_attack_type("BENIGN", args.samples)

    print("\n" + "=" * 60)
    print("TESTING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

