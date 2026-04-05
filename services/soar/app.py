import json
import os
import subprocess
import threading
import time

from flask import Flask, jsonify
from prometheus_client import Counter, Histogram, generate_latest
import redis
import yaml

app = Flask(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
ALERT_STREAM = os.getenv("ALERT_STREAM", "alerts")
ACTION_STREAM = os.getenv("ACTION_STREAM", "actions")
THRESHOLDS_PATH = os.getenv("THRESHOLDS_PATH", "")
ENABLE_IPTABLES = os.getenv("ENABLE_IPTABLES", "false").lower() == "true"

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

ACTIONS = Counter("nids_actions_total", "Total actions executed")
FAILS = Counter("soar_failures_total", "Total SOAR failures")
LAT = Histogram("soar_latency_ms", "SOAR latency ms")


def load_thresholds():
    if not THRESHOLDS_PATH or not os.path.exists(THRESHOLDS_PATH):
        return {"thresholds": {"alert": 0.65, "auto": 0.85}, "class_overrides": {}}
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


THRESHOLDS = load_thresholds()


def should_auto(label, confidence):
    override = THRESHOLDS.get("class_overrides", {}).get(label, {})
    auto = override.get("auto", THRESHOLDS["thresholds"]["auto"])
    return confidence >= auto


def iptables_block(ip):
    if not ENABLE_IPTABLES or not ip:
        return "simulated"
    cmd = ["/sbin/iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"]
    subprocess.run(cmd, check=False)
    return "executed"


def respond(alert):
    start = time.time()
    label = alert.get("label")
    confidence = float(alert.get("confidence", 0.0))
    src_ip = alert.get("src_ip")

    action = "escalate"
    if should_auto(label, confidence):
        if label in ("dos_ddos", "port_scan", "brute_force"):
            action = "block_ip"
        elif label == "malware_c2":
            action = "isolate_host"
        elif label == "data_exfil":
            action = "terminate_connection"

    result = None
    if action == "block_ip":
        result = iptables_block(src_ip)

    record = {
        "label": label,
        "confidence": confidence,
        "src_ip": src_ip,
        "action": action,
        "result": result or "queued",
        "ts": alert.get("ts"),
    }
    r.xadd(ACTION_STREAM, {"action": json.dumps(record)})
    ACTIONS.inc()
    LAT.observe((time.time() - start) * 1000)


def worker():
    last_id = "0-0"
    while True:
        try:
            items = r.xread({ALERT_STREAM: last_id}, block=1000, count=100)
            if not items:
                continue
            for _, messages in items:
                for msg_id, fields in messages:
                    alert = json.loads(fields.get("alert", "{}"))
                    respond(alert)
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
