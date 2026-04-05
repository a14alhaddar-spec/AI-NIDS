import json
import os
import time

from flask import Flask, jsonify
from prometheus_client import generate_latest
import redis

app = Flask(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
ALERT_STREAM = os.getenv("ALERT_STREAM", "alerts")
ACTION_STREAM = os.getenv("ACTION_STREAM", "actions")

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def read_stream(stream, count=20):
    items = r.xrevrange(stream, "+", "-", count=count)
    result = []
    for _, fields in items:
        key = "alert" if stream == ALERT_STREAM else "action"
        data = json.loads(fields.get(key, "{}"))
        result.append(data)
    return result


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_summary(alerts, actions, now_ts):
    window_sec = 300
    active = [a for a in alerts if now_ts - to_float(a.get("ts"), now_ts) <= window_sec]

    categories = {}
    confidences = []
    detection_lat = []
    for alert in alerts:
        label = alert.get("label", "unknown")
        categories[label] = categories.get(label, 0) + 1
        confidence = to_float(alert.get("confidence"))
        if confidence:
            confidences.append(confidence)
        ts = to_float(alert.get("ts"))
        if ts:
            detection_lat.append(max(0.0, (now_ts - ts) * 1000))

    action_counts = {}
    response_lat = []
    auto_actions = 0
    for action in actions:
        action_name = action.get("action", "unknown")
        action_counts[action_name] = action_counts.get(action_name, 0) + 1
        if action_name not in ("escalate", "unknown"):
            auto_actions += 1
        ts = to_float(action.get("ts"))
        if ts:
            response_lat.append(max(0.0, (now_ts - ts) * 1000))

    def avg(values):
        return sum(values) / len(values) if values else 0.0

    def max_val(values):
        return max(values) if values else 0.0

    def percentile(values, pct):
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(round((len(sorted_vals) - 1) * pct))
        return sorted_vals[idx]

    model_summary = {
        "alerts_total": len(alerts),
        "actions_total": len(actions),
        "auto_action_rate": (auto_actions / len(actions)) if actions else 0.0,
        "last_alert_age_ms": max_val(detection_lat),
    }

    return {
        "active_threats": len(active),
        "alert_window_sec": window_sec,
        "categories": categories,
        "confidence": {
            "avg": avg(confidences),
            "max": max_val(confidences),
            "min": min(confidences) if confidences else 0.0,
        },
        "detection_latency_ms": {
            "avg": avg(detection_lat),
            "p95": percentile(detection_lat, 0.95),
            "max": max_val(detection_lat),
        },
        "response_latency_ms": {
            "avg": avg(response_lat),
            "p95": percentile(response_lat, 0.95),
            "max": max_val(response_lat),
        },
        "actions": action_counts,
        "model": model_summary,
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {"Content-Type": "text/plain"}


@app.route("/summary", methods=["GET"])
def summary():
    alerts = read_stream(ALERT_STREAM, count=50)
    actions = read_stream(ACTION_STREAM, count=50)
    now_ts = time.time()
    return jsonify(
        {
            "status": "ok",
            "updated_at": now_ts,
            "alerts": alerts,
            "actions": actions,
            "overview": build_summary(alerts, actions, now_ts),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
