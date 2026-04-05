import json
import os
import threading
import time

from flask import Flask, jsonify, request
from prometheus_client import Counter, Histogram, generate_latest
import redis
import yaml

try:
    import joblib
    import numpy as np
except Exception:
    joblib = None
    np = None

app = Flask(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
FEATURE_STREAM = os.getenv("FEATURE_STREAM", "features")
ALERT_STREAM = os.getenv("ALERT_STREAM", "alerts")
MODEL_PATH = os.getenv("MODEL_PATH", "")
SCALER_PATH = os.getenv("SCALER_PATH", "")
THRESHOLDS_PATH = os.getenv("THRESHOLDS_PATH", "")

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

ALERTS = Counter("nids_alerts_total", "Total alerts emitted")
FAILS = Counter("inference_failures_total", "Total inference failures")
LAT = Histogram("nids_inference_latency_ms", "Inference latency ms")

CLASSES = [
    "benign",
    "dos_ddos",
    "port_scan",
    "brute_force",
    "malware_c2",
    "data_exfil",
]


def load_thresholds():
    if not THRESHOLDS_PATH or not os.path.exists(THRESHOLDS_PATH):
        return {"thresholds": {"alert": 0.65, "auto": 0.85}, "class_overrides": {}}
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


THRESHOLDS = load_thresholds()

MODEL = None
SCALER = None

if MODEL_PATH and joblib and os.path.exists(MODEL_PATH):
    MODEL = joblib.load(MODEL_PATH)
if SCALER_PATH and joblib and os.path.exists(SCALER_PATH):
    SCALER = joblib.load(SCALER_PATH)


def heuristic_score(features):
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
    if MODEL and np is not None:
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
        if SCALER is not None:
            X = SCALER.transform(X)
        proba = MODEL.predict_proba(X)[0]
        idx = int(proba.argmax())
        return CLASSES[idx], float(proba[idx])

    return heuristic_score(features)


def should_alert(label, confidence):
    if label == "benign":
        return False
    return confidence >= THRESHOLDS["thresholds"]["alert"]


def process_feature(features):
    start = time.time()
    label, confidence = predict(features)
    if should_alert(label, confidence):
        alert = {
            "label": label,
            "confidence": confidence,
            "src_ip": features.get("src_ip"),
            "dst_ip": features.get("dst_ip"),
            "ts": features.get("ts"),
        }
        r.xadd(ALERT_STREAM, {"alert": json.dumps(alert)})
        ALERTS.inc()
    LAT.observe((time.time() - start) * 1000)


def worker():
    last_id = "0-0"
    while True:
        try:
            items = r.xread({FEATURE_STREAM: last_id}, block=1000, count=100)
            if not items:
                continue
            for _, messages in items:
                for msg_id, fields in messages:
                    features = json.loads(fields.get("features", "{}"))
                    process_feature(features)
                    last_id = msg_id
        except Exception:
            FAILS.inc()
            time.sleep(1)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {"Content-Type": "text/plain"}


@app.route("/infer", methods=["POST"])
def infer():
    payload = request.get_json(silent=True) or {}
    label, confidence = predict(payload)
    
    # Determine if threat
    threat_detected = label.lower() not in ['benign', 'normal']
    
    # Track metrics
    try:
        import sys
        sys.path.append('..')
        from shared_metrics import add_prediction
        add_prediction(label, confidence, threat_detected, "Random Forest", {"mode": "online"})
    except:
        pass  # Metrics not critical
    
    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "threat_detected": threat_detected
    })


if __name__ == "__main__":
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5002)
