import json
import os
import threading
import time

from flask import Flask, jsonify
from prometheus_client import Counter, Histogram, generate_latest
import redis
import yaml

app = Flask(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
FLOW_STREAM = os.getenv("FLOW_STREAM", "flows")
FEATURE_STREAM = os.getenv("FEATURE_STREAM", "features")

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

PROCESSED = Counter("feature_processed_total", "Total flows processed")
FAILS = Counter("feature_failures_total", "Total feature failures")
LAT = Histogram("feature_latency_ms", "Feature extraction latency ms")


def extract_features(flow):
    duration = max(float(flow.get("duration", 0.0)), 1e-6)
    bytes_total = float(flow.get("bytes", 0.0))
    packets = float(flow.get("packets", 0.0))

    features = {
        "flow_duration": duration,
        "bytes": bytes_total,
        "packets": packets,
        "bytes_per_sec": bytes_total / duration,
        "packets_per_sec": packets / duration,
        "src_dst_bytes_ratio": float(flow.get("src_bytes", bytes_total))
        / max(float(flow.get("dst_bytes", bytes_total)), 1.0),
        "avg_pkt_size": bytes_total / max(packets, 1.0),
        "src_ip": flow.get("src_ip"),
        "dst_ip": flow.get("dst_ip"),
        "ts": flow.get("ts"),
    }
    return features


def worker():
    last_id = "0-0"
    while True:
        try:
            items = r.xread({FLOW_STREAM: last_id}, block=1000, count=100)
            if not items:
                continue
            for _, messages in items:
                for msg_id, fields in messages:
                    start = time.time()
                    flow = json.loads(fields.get("flow", "{}"))
                    features = extract_features(flow)
                    r.xadd(FEATURE_STREAM, {"features": json.dumps(features)})
                    PROCESSED.inc()
                    LAT.observe((time.time() - start) * 1000)
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


if __name__ == "__main__":
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000)
