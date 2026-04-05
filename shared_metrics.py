"""Shared metrics storage for inference services and dashboards

Model-specific results support for isolated dashboard displays.
"""

import json
import os
from datetime import datetime
from threading import RLock

# Model-specific results directory
MODEL_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "results", "run_test")

def get_model_results_path(model_name):
    """Get path to model-specific results JSON"""
    return os.path.join(MODEL_RESULTS_DIR, f"{model_name.lower().replace(' ', '_')}_results.json")

def get_model_metrics(model_name):
    """Load metrics for specific model from its dedicated JSON file"""
    path = get_model_results_path(model_name)
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                metrics = json.load(f)
                # Normalize recent predictions timestamps if needed
                if "recent_predictions" in metrics:
                    metrics["recent_predictions"] = _normalize_recent(metrics["recent_predictions"])
                return metrics
        else:
            print(f"Warning: Model results file not found: {path}")
            return {
                "total_predictions": 0,
                "threats_detected": 0,
                "benign_count": 0,
                "threat_types": {},
                "recent_predictions": [],
                "last_update": ""
            }
    except Exception as e:
        print(f"Error loading model metrics for {model_name}: {e}")
        return {"error": str(e), "total_predictions": 0}
import os
from datetime import datetime
from threading import RLock

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_FILE = os.path.join(BASE_DIR, "metrics_data.json")
lock = RLock()
VALID_MODES = {"offline", "online"}

try:
    from reponse_engine.soar_controller import process_detection_event
    SOAR_AVAILABLE = True
except Exception:
    process_detection_event = None
    SOAR_AVAILABLE = False

def init_metrics():
    """Initialize metrics file"""
    if not os.path.exists(METRICS_FILE):
        save_metrics(_default_metrics_payload())


def _empty_mode_metrics():
    return {
        "total_predictions": 0,
        "threats_detected": 0,
        "benign_count": 0,
        "threat_types": {},
        "recent_predictions": [],
        "model_predictions": {},
        "last_update": datetime.now().isoformat(),
    }


def _default_metrics_payload(active_mode="offline"):
    mode = str(active_mode).lower()
    if mode not in VALID_MODES:
        mode = "offline"

    return {
        "active_mode": mode,
        "modes": {
            "offline": _empty_mode_metrics(),
            "online": _empty_mode_metrics(),
        },
    }


def _normalize_mode(mode):
    value = str(mode or "").lower().strip()
    return value if value in VALID_MODES else "offline"


def _normalize_payload(raw_metrics):
    """Normalize old/new metrics payloads to mode-scoped schema."""
    if not isinstance(raw_metrics, dict):
        return _default_metrics_payload()

    # New format already persisted.
    if isinstance(raw_metrics.get("modes"), dict):
        payload = {
            "active_mode": _normalize_mode(raw_metrics.get("active_mode")),
            "modes": {},
        }
        for mode_name in ("offline", "online"):
            section = raw_metrics["modes"].get(mode_name, {})
            if not isinstance(section, dict):
                section = {}
            payload["modes"][mode_name] = {
                "total_predictions": int(section.get("total_predictions", 0)),
                "threats_detected": int(section.get("threats_detected", 0)),
                "benign_count": int(section.get("benign_count", 0)),
                "threat_types": dict(section.get("threat_types", {})),
                "recent_predictions": _normalize_recent(section.get("recent_predictions", [])),
                "model_predictions": dict(section.get("model_predictions", {})),
                "last_update": section.get("last_update", datetime.now().isoformat()),
            }
        return payload

    # Backward compatibility: migrate old flat metrics into offline bucket.
    payload = _default_metrics_payload(active_mode="offline")
    payload["modes"]["offline"] = {
        "total_predictions": int(raw_metrics.get("total_predictions", 0)),
        "threats_detected": int(raw_metrics.get("threats_detected", 0)),
        "benign_count": int(raw_metrics.get("benign_count", 0)),
        "threat_types": dict(raw_metrics.get("threat_types", {})),
        "recent_predictions": _normalize_recent(raw_metrics.get("recent_predictions", [])),
        "model_predictions": dict(raw_metrics.get("model_predictions", {})),
        "last_update": raw_metrics.get("last_update", datetime.now().isoformat()),
    }
    return payload

def load_metrics():
    """Load metrics from file"""
    try:
        with open(METRICS_FILE, 'r') as f:
            raw_metrics = json.load(f)
            metrics = _normalize_payload(raw_metrics)
            return metrics
    except:
        init_metrics()
        return load_metrics()

def save_metrics(metrics):
    """Save metrics to file"""
    with lock:
        normalized = _normalize_payload(metrics)
        with open(METRICS_FILE, 'w') as f:
            json.dump(normalized, f, indent=2)

def _normalize_recent(recent):
    normalized = []
    for pred in recent:
        if not isinstance(pred, dict):
            continue

        ts = pred.get("timestamp")
        if isinstance(ts, str):
            try:
                ts_epoch = int(datetime.fromisoformat(ts).timestamp())
            except Exception:
                ts_epoch = int(datetime.now().timestamp())
        elif isinstance(ts, (int, float)):
            ts_epoch = int(ts)
        else:
            ts_epoch = int(datetime.now().timestamp())

        label = pred.get("label") or pred.get("prediction") or "unknown"
        confidence = float(pred.get("confidence", 0.0))
        is_threat = pred.get("is_threat")
        if is_threat is None:
            is_threat = pred.get("threat", False)

        message = pred.get("message") or f"{label} detected"

        normalized.append({
            "timestamp": ts_epoch,
            "label": label,
            "confidence": confidence,
            "is_threat": bool(is_threat),
            "message": message,
            "prediction": label,
            "threat": bool(is_threat)
        })

    return normalized

def get_active_mode():
    with lock:
        metrics = load_metrics()
        return _normalize_mode(metrics.get("active_mode"))


def set_active_mode(mode):
    with lock:
        metrics = load_metrics()
        metrics["active_mode"] = _normalize_mode(mode)
        save_metrics(metrics)


def add_prediction(prediction, confidence, threat_detected, model_name="CNN-LSTM", metadata=None):
    """Add a new prediction to metrics
    Args:
        prediction: The predicted label
        confidence: Confidence score (0-1)
        threat_detected: Whether a threat was detected
        model_name: Name of the model making the prediction (default: CNN-LSTM)
    """
    metadata = metadata or {}
    mode = _normalize_mode(metadata.get("mode") or get_active_mode())

    with lock:
        metrics = load_metrics()
        mode_metrics = metrics["modes"][mode]
        
        # Initialize model tracking if not exists
        if "model_predictions" not in mode_metrics:
            mode_metrics["model_predictions"] = {}
        
        # Update model-specific counters
        if model_name not in mode_metrics["model_predictions"]:
            mode_metrics["model_predictions"][model_name] = {
                "total": 0,
                "threats": 0,
                "benign": 0
            }
        
        # Update counters
        mode_metrics["total_predictions"] += 1
        mode_metrics["model_predictions"][model_name]["total"] += 1
        
        if threat_detected:
            mode_metrics["threats_detected"] += 1
            mode_metrics["model_predictions"][model_name]["threats"] += 1
            threat_type = prediction
            mode_metrics["threat_types"][threat_type] = mode_metrics["threat_types"].get(threat_type, 0) + 1
        else:
            mode_metrics["benign_count"] += 1
            mode_metrics["model_predictions"][model_name]["benign"] += 1
        
        # Add to recent predictions (keep last 50)
        now = datetime.now()
        recent = {
            "timestamp": int(now.timestamp()),
            "label": prediction,
            "prediction": prediction,
            "confidence": confidence,
            "is_threat": threat_detected,
            "threat": threat_detected,
            "message": f"{prediction} detected",
            "model": model_name
        }
        mode_metrics["recent_predictions"].insert(0, recent)
        mode_metrics["recent_predictions"] = mode_metrics["recent_predictions"][:50]
        
        mode_metrics["last_update"] = now.isoformat()
        
        save_metrics(metrics)

    if SOAR_AVAILABLE:
        try:
            soar_metadata = dict(metadata)
            soar_metadata.setdefault("model", model_name)
            process_detection_event(
                label=prediction,
                confidence=confidence,
                threat_detected=threat_detected,
                metadata=soar_metadata,
            )
        except Exception as e:
            print(f"Warning: SOAR processing failed: {e}")

def get_metrics(mode=None):
    """Get metrics scoped to requested mode (or active mode)."""
    with lock:
        metrics = load_metrics()
        selected_mode = _normalize_mode(mode or metrics.get("active_mode"))
        section = dict(metrics["modes"].get(selected_mode, _empty_mode_metrics()))
        section["mode"] = selected_mode
        section["active_mode"] = _normalize_mode(metrics.get("active_mode"))
        return section

def reset_metrics(mode=None):
    """Reset metrics for one mode, or all modes when mode='all'."""
    with lock:
        metrics = load_metrics()
        normalized_mode = str(mode or "").lower().strip()

        if normalized_mode == "all":
            metrics = _default_metrics_payload(active_mode=metrics.get("active_mode", "offline"))
        else:
            selected_mode = _normalize_mode(mode or metrics.get("active_mode"))
            metrics["modes"][selected_mode] = _empty_mode_metrics()

        save_metrics(metrics)
