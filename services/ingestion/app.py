import json
import os
import time

from flask import Flask, jsonify, request
from prometheus_client import Counter, Histogram, generate_latest
import redis

app = Flask(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
FLOW_STREAM = os.getenv("FLOW_STREAM", "flows")

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

REQUESTS = Counter("ingestion_requests_total", "Total ingestion requests")
FAILS = Counter("ingestion_failures_total", "Total ingestion failures")
LAT = Histogram("ingestion_latency_ms", "Ingestion latency ms")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {"Content-Type": "text/plain"}


@app.route("/ingest", methods=["POST"])
def ingest():
    start = time.time()
    REQUESTS.inc()

    payload = request.get_json(silent=True) or {}
    required = ["src_ip", "dst_ip", "bytes", "packets", "duration"]
    if not all(k in payload for k in required):
        FAILS.inc()
        return jsonify({"error": "missing required fields"}), 400

    payload["ts"] = payload.get("ts", time.time())
    r.xadd(FLOW_STREAM, {"flow": json.dumps(payload)})

    LAT.observe((time.time() - start) * 1000)
    return jsonify({"status": "queued"}), 202


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
