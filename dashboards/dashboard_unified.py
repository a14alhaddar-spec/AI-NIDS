"""
Unified AI-NIDS Dashboard
Primary Model: CNN-LSTM (Hybrid CNN-LSTM Deep Learning Model)
Displays metrics from all models during testing
Runs on port 8080
"""
# Suppress verbose TensorFlow and sklearn warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('sklearn').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

from flask import Flask, render_template_string, jsonify, request, make_response
import subprocess
import sys
import time
import json
from collections import Counter
from threading import Lock, Thread

app = Flask(__name__)

# Primary model configuration
PRIMARY_MODEL = "CNN-LSTM"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_TEST_LOCK = Lock()
RUN_TEST_STATE_LOCK = Lock()
RUN_TEST_PROCESS = None
RUN_TEST_STATE = {
  "running": False,
  "status": "idle",
  "message": "",
  "command": "",
  "output": [],
  "returncode": None,
  "started_at": None,
  "finished_at": None,
}
SYSTEM_MODE_LOCK = Lock()
SYSTEM_MODE_STATE = {
  "mode": "offline",
  "monitoring": False,
  "message": "Offline mode active.",
  "warning": "",
  "command": "",
  "started_at": None,
  "last_changed_at": int(time.time() * 1000),
}
MONITOR_PROCESS = None
RUN_TEST_OPTIONS = [
  {"value": "BENIGN", "label": "Benign"},
  {"value": "PortScan", "label": "PortScan"},
  {"value": "DDoS", "label": "DDoS"},
  {"value": "Bot", "label": "Bot"},
  {"value": "DoS GoldenEye", "label": "DoS GoldenEye"},
  {"value": "DoS Hulk", "label": "DoS Hulk"},
  {"value": "DoS Slowhttptest", "label": "DoS Slowhttptest"},
  {"value": "DoS slowloris", "label": "DoS slowloris"},
  {"value": "FTP-Patator", "label": "FTP-Patator"},
  {"value": "Heartbleed", "label": "Heartbleed"},
  {"value": "Infiltration", "label": "Infiltration"},
  {"value": "SSH-Patator", "label": "SSH-Patator"},
  {"value": "Web Attack - Brute Force", "label": "Web Attack - Brute Force"},
  {"value": "Web Attack - Sql Injection", "label": "Web Attack - Sql Injection"},
  {"value": "Web Attack - XSS", "label": "Web Attack - XSS"},
]
RUN_TEST_FEATURE_NAMES = [
  ' Destination Port',
  ' Flow Duration',
  ' Total Fwd Packets',
  ' Total Backward Packets',
  'Total Length of Fwd Packets',
  ' Total Length of Bwd Packets',
  'Flow Bytes/s',
]
MODEL_DIR = os.path.join(BASE_DIR, "models", "cicids_full")
PRIMARY_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_lstm_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
LIVE_MONITOR_SCRIPT_PATH = os.path.join(BASE_DIR, "scripts", "live_monitor.py")
LIVE_MONITOR_LOG_PATH = os.path.join(BASE_DIR, "data", "forensics", "live_monitor.log")
FORENSICS_RESPONSE_ACTIONS_PATH = os.path.join(BASE_DIR, "data", "forensics", "response_actions.jsonl")
FORENSICS_SYSTEM_EVENTS_PATH = os.path.join(BASE_DIR, "data", "forensics", "system_events.jsonl")
FORENSICS_ATTACK_METADATA_PATH = os.path.join(BASE_DIR, "data", "forensics", "attack_metadata.jsonl")
FORENSICS_ALERT_ANALYSTS_PATH = os.path.join(BASE_DIR, "data", "forensics", "alerts_security_analysts.jsonl")
FORENSICS_ALERT_SIEM_PATH = os.path.join(BASE_DIR, "data", "forensics", "alerts_siem.jsonl")
FORENSICS_ALERT_IRT_PATH = os.path.join(BASE_DIR, "data", "forensics", "alerts_incident_response_team.jsonl")

_UPLOAD_MODEL_CACHE = {
  "model": None,
  "scaler": None,
  "encoder": None,
}
_UPLOAD_MODEL_LOCK = Lock()

# Import shared metrics
try:
  base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
  if base_dir not in sys.path:
    sys.path.append(base_dir)
  from shared_metrics import get_metrics, add_prediction, reset_metrics, set_active_mode
  METRICS_AVAILABLE = True
except Exception as e:
  print(f"Warning: Could not import shared_metrics: {e}")
  METRICS_AVAILABLE = False

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>AI-NIDS Unified Dashboard</title>
    <link
      rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Urbanist:wght@400;600;700&display=swap"
    />
    <style>
:root {
  --bg: #08131f;
  --accent: #f5b74e;
  --accent-2: #40d3a3;
  --danger: #ff6767;
  --card: #101f33;
  --card-strong: #14263f;
  --text: #e9f0f9;
  --muted: #9fb3c8;
  --border: #203754;
  --shadow: rgba(7, 12, 20, 0.4);
  --card-height: 350px;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: "Space Grotesk", "Urbanist", "Segoe UI", sans-serif;
  color: var(--text);
  background: radial-gradient(circle at top, #0f2944, #050b14 55%);
  min-height: 100vh;
  overflow-x: hidden;
}

.app {
  min-height: 100vh;
  width: min(100%, 1600px);
  margin: 0 auto;
  padding: clamp(1rem, 2.3vw, 2.5rem) clamp(0.9rem, 2.8vw, 2.75rem) 4rem;
  position: relative;
  overflow-x: clip;
}

.app::before {
  content: "";
  position: absolute;
  top: -120px;
  right: -180px;
  width: 420px;
  height: 420px;
  background: radial-gradient(circle, rgba(64, 211, 163, 0.25), transparent 65%);
  filter: blur(10px);
  z-index: 0;
}

.app::after {
  content: "";
  position: absolute;
  bottom: -200px;
  left: -120px;
  width: 520px;
  height: 520px;
  background: radial-gradient(circle, rgba(245, 183, 78, 0.2), transparent 70%);
  filter: blur(18px);
  z-index: 0;
}

.hero {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  margin-bottom: 2rem;
  position: relative;
  z-index: 1;
}

.hero-title h1 {
  font-size: clamp(2.5rem, 4vw, 3.75rem);
  margin: 0;
  letter-spacing: 0.02em;
}

.hero-title p {
  margin: 0.25rem 0 0;
  color: var(--muted);
}

.eyebrow {
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.3em;
  color: var(--accent);
}

.subtitle {
  max-width: 640px;
}

.hero-status {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 1rem;
}

.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-left: auto;
}

.modal-backdrop {
  position: fixed;
  inset: 0;
  display: flex;
  background: linear-gradient(180deg, rgba(3, 8, 14, 0.62), rgba(3, 8, 14, 0.8));
  backdrop-filter: blur(5px);
  opacity: 0;
  visibility: hidden;
  pointer-events: none;
  align-items: center;
  justify-content: center;
  padding: 1.25rem;
  z-index: 1000;
  transition: opacity 0.2s ease, visibility 0.2s ease;
}

.modal-backdrop.open {
  opacity: 1;
  visibility: visible;
  pointer-events: auto;
}

.modal {
  width: min(920px, 100%);
  background: linear-gradient(180deg, #152d49 0%, #0e1f34 56%, #0b1727 100%);
  border: 1px solid rgba(159, 179, 200, 0.3);
  border-radius: 24px;
  box-shadow: 0 36px 90px rgba(0, 0, 0, 0.48), inset 0 1px 0 rgba(255, 255, 255, 0.04);
  padding: 1.4rem;
  position: relative;
  height: min(88vh, 780px);
  max-height: min(88vh, 780px);
  overflow: auto;
  transform: translateY(8px) scale(0.985);
  transition: transform 0.2s ease;
}

.modal::before {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: inherit;
  pointer-events: none;
  background: radial-gradient(circle at top right, rgba(245, 183, 78, 0.14), transparent 40%),
    radial-gradient(circle at bottom left, rgba(64, 211, 163, 0.12), transparent 42%);
}

.modal > * {
  position: relative;
  z-index: 1;
}

.modal-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  padding-bottom: 0.95rem;
  margin-bottom: 0.95rem;
  border-bottom: 1px solid rgba(159, 179, 200, 0.16);
}

.modal-kicker {
  margin: 0 0 0.45rem;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  font-size: 0.72rem;
  color: var(--accent);
  font-weight: 700;
}

.modal-backdrop.open .modal {
  transform: translateY(0) scale(1);
}

.modal h2 {
  margin: 0;
  font-size: 1.5rem;
  letter-spacing: 0.01em;
}

.modal label {
  color: var(--muted);
  font-size: 0.86rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
}

.modal p {
  margin: 0.5rem 0 1rem;
  color: var(--muted);
  line-height: 1.45;
}

.modal-intro {
  margin-top: 0;
  margin-bottom: 1.05rem;
}

.modal-intro strong {
  color: #f8d08a;
}

.modal-close {
  width: 2.15rem;
  height: 2.15rem;
  border-radius: 10px;
  border: 1px solid rgba(159, 179, 200, 0.32);
  background: rgba(255, 255, 255, 0.04);
  color: var(--text);
  font-size: 1.25rem;
  line-height: 1;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: background 0.18s ease, border-color 0.18s ease, transform 0.18s ease;
}

.modal-close:hover {
  background: rgba(255, 255, 255, 0.09);
  border-color: rgba(159, 179, 200, 0.58);
  transform: translateY(-1px);
}

.modal-close:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px rgba(64, 211, 163, 0.23);
}

.modal-grid {
  display: grid;
  gap: 0.7rem;
  padding: 0.9rem;
  border: 1px solid rgba(159, 179, 200, 0.16);
  border-radius: 14px;
  background: rgba(6, 17, 29, 0.42);
}

.mode-switch {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.6rem;
  margin-bottom: 0.9rem;
}

.mode-option {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  padding: 0.68rem 0.82rem;
  border: 1px solid rgba(159, 179, 200, 0.25);
  border-radius: 12px;
  background: rgba(7, 17, 29, 0.62);
  color: var(--text);
  font-size: 0.9rem;
}

.mode-option input[type="radio"] {
  accent-color: var(--accent-2);
}

.modal-section {
  margin-bottom: 0.8rem;
}

.modal-section[hidden] {
  display: none;
}

.attack-select {
  width: 100%;
  border: 1px solid rgba(59, 96, 131, 0.9);
  border-radius: 14px;
  background: rgba(6, 15, 26, 0.9);
  color: var(--text);
  padding: 0.85rem 2.55rem 0.85rem 0.95rem;
  font: inherit;
  outline: none;
  appearance: none;
  background-image: linear-gradient(45deg, transparent 50%, #9fb3c8 50%), linear-gradient(135deg, #9fb3c8 50%, transparent 50%);
  background-position: calc(100% - 1.1rem) calc(50% - 2px), calc(100% - 0.78rem) calc(50% - 2px);
  background-size: 7px 7px, 7px 7px;
  background-repeat: no-repeat;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.attack-select option {
  background: #0d1f33;
  color: #e9f0f9;
}

.attack-select:focus {
  border-color: rgba(64, 211, 163, 0.65);
  box-shadow: 0 0 0 3px rgba(64, 211, 163, 0.22);
}

.chip-mode {
  display: inline-flex;
  gap: 0.45rem;
  flex-wrap: wrap;
}

.chip-mode .mode-option {
  margin: 0;
  padding: 0.5rem 0.72rem;
}

.chip-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
}

.chip-button {
  border: 1px solid rgba(159, 179, 200, 0.26);
  border-radius: 999px;
  padding: 0.44rem 0.72rem;
  background: rgba(7, 17, 29, 0.62);
  color: var(--text);
  font: inherit;
  font-size: 0.83rem;
  cursor: pointer;
  transition: border-color 0.18s ease, background 0.18s ease, transform 0.18s ease;
}

.chip-button:hover {
  border-color: rgba(159, 179, 200, 0.55);
  transform: translateY(-1px);
}

.chip-button.selected {
  border-color: rgba(64, 211, 163, 0.7);
  background: linear-gradient(135deg, rgba(64, 211, 163, 0.24), rgba(245, 183, 78, 0.14));
  color: #f0fff9;
}

.chip-button:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px rgba(64, 211, 163, 0.24);
}

.modal-help {
  margin: 0;
  font-size: 0.82rem;
  color: var(--muted);
}

.csv-input {
  width: 100%;
  border: 1px dashed rgba(159, 179, 200, 0.4);
  border-radius: 12px;
  background: rgba(7, 17, 29, 0.84);
  color: var(--text);
  padding: 0.7rem 0.8rem;
  font: inherit;
}

.csv-input:hover {
  border-color: rgba(159, 179, 200, 0.62);
}

.csv-input:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px rgba(64, 211, 163, 0.23);
  border-color: rgba(64, 211, 163, 0.65);
}

.modal-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 0.65rem;
  margin-top: 1rem;
  padding-top: 0.95rem;
  border-top: 1px solid rgba(159, 179, 200, 0.16);
}

.modal-actions .action-button {
  min-width: 9.5rem;
}

.reset-modal {
  width: min(620px, 100%);
  height: auto;
  max-height: calc(100vh - 2rem);
  overflow: auto;
}

.reset-modal-body {
  display: grid;
  gap: 0.75rem;
}

.reset-warning {
  margin: 0;
  color: var(--muted);
  line-height: 1.5;
}

.reset-warning strong {
  color: #f8d08a;
}

.console-wrap {
  margin-top: 1rem;
}

.console-title {
  margin: 0 0 0.45rem;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--muted);
  font-weight: 700;
}

.console {
  margin-top: 0;
  border: 1px solid rgba(159, 179, 200, 0.22);
  border-radius: 16px;
  background: rgba(3, 10, 18, 0.9);
  padding: 0.9rem;
  min-height: 190px;
  max-height: 320px;
  overflow: auto;
  white-space: pre-wrap;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: 0.82rem;
  line-height: 1.45;
  color: #dbe8f7;
}

#run-test-modal .modal {
  scrollbar-width: thin;
  scrollbar-color: rgba(64, 211, 163, 0.55) rgba(255, 255, 255, 0.08);
}

#run-test-modal .modal::-webkit-scrollbar {
  width: 9px;
}

#run-test-modal .modal::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.06);
  border-radius: 999px;
}

#run-test-modal .modal::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(245, 183, 78, 0.78), rgba(64, 211, 163, 0.78));
  border-radius: 999px;
  border: 1px solid rgba(9, 20, 34, 0.35);
}

#run-test-modal .modal::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, rgba(245, 183, 78, 0.92), rgba(64, 211, 163, 0.9));
}

#run-test-modal .console {
  scrollbar-width: thin;
  scrollbar-color: rgba(64, 211, 163, 0.55) rgba(255, 255, 255, 0.08);
}

#run-test-modal .console::-webkit-scrollbar {
  width: 9px;
}

#run-test-modal .console::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.06);
  border-radius: 999px;
}

#run-test-modal .console::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(245, 183, 78, 0.78), rgba(64, 211, 163, 0.78));
  border-radius: 999px;
  border: 1px solid rgba(9, 20, 34, 0.35);
}

#run-test-modal .console::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, rgba(245, 183, 78, 0.92), rgba(64, 211, 163, 0.9));
}

.console-line {
  margin: 0;
}

.console-line.dim {
  color: var(--muted);
}

.secondary-button {
  border: 1px solid rgba(159, 179, 200, 0.28);
  background: rgba(255, 255, 255, 0.03);
  color: var(--text);
  box-shadow: none;
}

.secondary-button:hover {
  border-color: rgba(159, 179, 200, 0.5);
  background: rgba(255, 255, 255, 0.08);
}

.primary-launch {
  box-shadow: 0 14px 30px rgba(64, 211, 163, 0.2);
}

.status {
  appearance: none;
  cursor: pointer;
  font: inherit;
  color: inherit;
  padding: 0.5rem 1rem;
  background: var(--card);
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  border: 1px solid var(--border);
  font-weight: 600;
  box-shadow: 0 10px 20px var(--shadow);
}

.status:hover {
  border-color: rgba(64, 211, 163, 0.45);
  transform: translateY(-1px);
}

.status:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px rgba(64, 211, 163, 0.24), 0 10px 20px var(--shadow);
}

.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--danger);
  box-shadow: 0 0 0 0 rgba(255, 103, 103, 0.6);
}

.dot.live {
  background: var(--accent-2);
  animation: pulse 1.8s infinite;
}

.timestamp {
  color: var(--muted);
  font-size: 0.9rem;
}

.monitor-warning-banner {
  margin: 0.2rem 0 1.25rem;
  padding: 0.7rem 0.95rem;
  border-radius: 12px;
  border: 1px solid rgba(255, 103, 103, 0.4);
  background: linear-gradient(135deg, rgba(255, 103, 103, 0.16), rgba(245, 183, 78, 0.1));
  color: #ffd9d9;
  font-weight: 600;
  letter-spacing: 0.01em;
  position: relative;
  z-index: 1;
}

.action-button {
  appearance: none;
  border: 1px solid rgba(245, 183, 78, 0.35);
  background: linear-gradient(135deg, rgba(245, 183, 78, 0.18), rgba(64, 211, 163, 0.12));
  color: var(--text);
  border-radius: 999px;
  padding: 0.8rem 1.2rem;
  font: inherit;
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 12px 28px var(--shadow);
  transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
}

.action-button:hover {
  transform: translateY(-1px);
  border-color: rgba(245, 183, 78, 0.65);
  background: linear-gradient(135deg, rgba(245, 183, 78, 0.25), rgba(64, 211, 163, 0.16));
}

.action-button.primary-launch {
  border-color: rgba(64, 211, 163, 0.35);
}

.action-button:disabled {
  cursor: wait;
  opacity: 0.65;
  transform: none;
}

.action-status {
  color: var(--muted);
  font-size: 0.9rem;
  min-height: 1.2em;
}

.reset-button {
  border-color: rgba(255, 103, 103, 0.35);
  background: linear-gradient(135deg, rgba(255, 103, 103, 0.2), rgba(245, 183, 78, 0.12));
}

.reset-button:hover {
  border-color: rgba(255, 103, 103, 0.6);
  background: linear-gradient(135deg, rgba(255, 103, 103, 0.28), rgba(245, 183, 78, 0.18));
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1.25rem;
  margin-bottom: 2rem;
  position: relative;
  z-index: 1;
}

.kpi {
  background: linear-gradient(135deg, var(--card-strong), var(--card));
  border-radius: 16px;
  padding: 1.25rem;
  border: 1px solid var(--border);
  box-shadow: 0 18px 40px var(--shadow);
}

.kpi p {
  margin: 0 0 0.5rem;
  color: var(--muted);
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.2em;
}

.kpi h2 {
  margin: 0;
  font-size: 2rem;
}

.kpi span {
  color: var(--muted);
  font-size: 0.9rem;
}

.kpi.danger h2 {
  color: var(--danger);
}

.kpi.success h2 {
  color: var(--accent-2);
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  position: relative;
  z-index: 1;
}

.card {
  background: var(--card);
  border-radius: 16px;
  padding: 1.5rem;
  border: 1px solid var(--border);
  height: var(--card-height);
  box-shadow: 0 24px 50px var(--shadow);
  display: flex;
  flex-direction: column;
  gap: 1rem;
  overflow: hidden;
  animation: floatIn 0.6s ease;
}

.card h2 {
  margin-top: 0;
  margin-bottom: 0;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.card-header-main {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.card-header-actions {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
}

.expandable-card {
  cursor: pointer;
  transition: border-color 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
}

.expandable-card:hover {
  border-color: rgba(159, 179, 200, 0.5);
  transform: translateY(-1px);
}

.expandable-card:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px rgba(64, 211, 163, 0.24), 0 24px 50px var(--shadow);
}

.card-hint {
  color: var(--muted);
  font-size: 0.76rem;
  letter-spacing: 0.03em;
  text-transform: uppercase;
}

.view-all-btn {
  border: 1px solid rgba(159, 179, 200, 0.33);
  background: rgba(255, 255, 255, 0.04);
  color: var(--text);
  border-radius: 10px;
  width: 2rem;
  height: 2rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: border-color 0.18s ease, background 0.18s ease;
}

.view-all-btn:hover {
  border-color: rgba(159, 179, 200, 0.58);
  background: rgba(255, 255, 255, 0.11);
}

.view-all-btn:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px rgba(64, 211, 163, 0.22);
}

.view-all-btn svg {
  width: 1rem;
  height: 1rem;
  stroke: currentColor;
}

.pill {
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  background: rgba(64, 211, 163, 0.15);
  color: var(--accent-2);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.pill.danger {
  background: rgba(255, 103, 103, 0.15);
  color: var(--danger);
}

.card ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.scroll-pane {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding-right: 0.35rem;
  scrollbar-width: thin;
  scrollbar-color: rgba(64, 211, 163, 0.5) rgba(255, 255, 255, 0.08);
}

.scroll-pane::-webkit-scrollbar {
  width: 9px;
}

.scroll-pane::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.06);
  border-radius: 999px;
}

.scroll-pane::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(245, 183, 78, 0.7), rgba(64, 211, 163, 0.7));
  border-radius: 999px;
  border: 1px solid rgba(9, 20, 34, 0.35);
}

.scroll-pane::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, rgba(245, 183, 78, 0.9), rgba(64, 211, 163, 0.88));
}

.card li {
  padding: 0.5rem 0;
  border-bottom: 1px dashed #254b72;
  color: var(--muted);
  display: flex;
  justify-content: space-between;
  gap: 0.75rem;
}

.card li strong {
  color: var(--text);
}

.card li.threat {
  color: var(--danger);
}

.card li.threat strong {
  color: var(--danger);
}

.bar-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.pie-card-body {
  display: grid;
  grid-template-columns: minmax(160px, 1fr);
  gap: 1rem;
  align-items: center;
  min-height: 0;
}

.pie-chart-shell {
  position: relative;
  display: grid;
  place-items: center;
  min-height: 220px;
}

.pie-chart {
  width: min(210px, 100%);
  aspect-ratio: 1;
  display: block;
}

.pie-chart-ring {
  fill: none;
  stroke: rgba(255, 255, 255, 0.08);
  stroke-width: 26;
}

.pie-chart-slice {
  fill: none;
  stroke-width: 26;
  stroke-linecap: butt;
  transform: rotate(-90deg);
  transform-origin: 50% 50%;
  cursor: pointer;
  transition: stroke-width 0.22s cubic-bezier(0.22, 1, 0.36, 1), filter 0.22s cubic-bezier(0.22, 1, 0.36, 1), opacity 0.22s ease;
  will-change: stroke-width, filter;
}

.pie-chart-slice.active {
  stroke-width: 31;
  filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.32));
  opacity: 1;
}

.pie-chart-center {
  position: absolute;
  inset: 50% auto auto 50%;
  transform: translate(-50%, -50%);
  display: grid;
  place-items: center;
  text-align: center;
  pointer-events: none;
}

.pie-chart-center strong {
  font-size: 1.35rem;
}

.pie-chart-center span {
  color: var(--muted);
  font-size: 0.74rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  max-width: 160px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.pie-legend {
  display: grid;
  gap: 0.45rem;
}

.pie-legend-item,
.pie-detail-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  color: var(--muted);
  font-size: 0.86rem;
}

.pie-legend-key {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  min-width: 0;
}

.pie-swatch {
  width: 0.8rem;
  height: 0.8rem;
  border-radius: 999px;
  flex: 0 0 auto;
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.05);
}

.pie-detail-stack {
  display: grid;
  gap: 0.75rem;
}

.pie-detail-chart {
  width: min(320px, 100%);
}

.pie-detail-list {
  display: grid;
  gap: 0.55rem;
}

.category-response-item {
  border-left-width: 5px;
  border-left-style: solid;
}

.category-response-item .response-actions {
  color: var(--muted);
}

.pie-detail-row .bar-track {
  flex: 1;
  margin-left: 0.75rem;
}

.pie-detail-row .bar-value {
  min-width: 3rem;
}

.bar-row {
  display: grid;
  grid-template-columns: 120px 1fr 50px;
  gap: 0.75rem;
  align-items: center;
}

.bar-label {
  color: var(--text);
  font-size: 0.9rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.bar-track {
  position: relative;
  height: 8px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
}

.bar-fill {
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
}

.bar-fill.danger {
  background: linear-gradient(90deg, var(--danger), #ff8a8a);
}

.bar-value {
  text-align: right;
  color: var(--muted);
  font-size: 0.85rem;
}

.empty {
  color: var(--muted);
  font-size: 0.9rem;
  padding: 1rem 0;
}

.metrics li {
  display: flex;
  justify-content: space-between;
  color: var(--muted);
}

.metrics li strong {
  color: var(--text);
}

.system-status-wrap {
  display: grid;
  gap: 0.9rem;
}

.system-health {
  display: grid;
  grid-template-columns: 92px 1fr;
  gap: 0.85rem;
  align-items: center;
}

.system-health-ring {
  --health: 0;
  width: 92px;
  height: 92px;
  border-radius: 50%;
  background: conic-gradient(#40d3a3 calc(var(--health) * 1%), rgba(255, 255, 255, 0.1) 0);
  display: grid;
  place-items: center;
  position: relative;
}

.system-health-ring::before {
  content: "";
  position: absolute;
  inset: 10px;
  border-radius: 50%;
  background: rgba(8, 18, 31, 0.95);
  border: 1px solid rgba(159, 179, 200, 0.2);
}

.system-health-value {
  position: relative;
  z-index: 1;
  font-weight: 800;
  font-size: 1rem;
}

.system-health-meta {
  display: grid;
  gap: 0.22rem;
}

.system-health-title {
  color: var(--text);
  font-weight: 700;
}

.system-health-subtitle {
  color: var(--muted);
  font-size: 0.82rem;
}

.system-health-bars {
  display: grid;
  gap: 0.55rem;
}

.system-bar-row {
  display: grid;
  grid-template-columns: 92px 1fr 42px;
  align-items: center;
  gap: 0.65rem;
  font-size: 0.8rem;
}

.system-bar-label {
  color: var(--muted);
}

.system-bar-track {
  height: 7px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.1);
  overflow: hidden;
  position: relative;
}

.system-bar-fill {
  position: absolute;
  inset: 0 auto 0 0;
  width: 0;
  border-radius: inherit;
  background: linear-gradient(90deg, #40d3a3, #f5b74e);
}

.system-bar-value {
  color: var(--text);
  text-align: right;
  font-weight: 700;
}

.response-list li {
  align-items: flex-start;
  flex-direction: column;
  gap: 0.25rem;
  border-left: 3px solid rgba(159, 179, 200, 0.35);
  padding-left: 0.6rem;
}

.response-meta {
  color: var(--muted);
  font-size: 0.82rem;
}

.response-actions {
  color: var(--text);
  font-size: 0.86rem;
}

.prediction-list,
.response-list,
.details-list {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.prediction-item,
.response-item {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  border: 1px solid rgba(159, 179, 200, 0.14);
  border-radius: 11px;
  background: rgba(6, 15, 26, 0.48);
  padding: 0.65rem 0.7rem;
}

.response-item {
  border-left-width: 3px;
}

.prediction-item-header,
.response-item-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.7rem;
}

.prediction-label,
.response-label {
  color: var(--text);
  font-weight: 700;
}

.prediction-meta,
.response-meta {
  color: var(--muted);
  font-size: 0.82rem;
}

.prediction-item.threat {
  border-color: rgba(255, 103, 103, 0.42);
  background: rgba(53, 16, 20, 0.34);
}

.prediction-item .confidence-badge {
  border-radius: 999px;
  padding: 0.16rem 0.52rem;
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  border: 1px solid rgba(159, 179, 200, 0.34);
  color: var(--text);
  background: rgba(255, 255, 255, 0.06);
}

.prediction-item.threat .confidence-badge {
  border-color: rgba(255, 103, 103, 0.5);
  background: rgba(255, 103, 103, 0.16);
  color: #ffd8d8;
}

.preview-list {
  flex: 1;
  min-height: 0;
  overflow: hidden;
  position: relative;
}

.preview-list::after {
  content: "";
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 34px;
  pointer-events: none;
  background: linear-gradient(180deg, rgba(16, 31, 51, 0), rgba(16, 31, 51, 0.95));
}

.details-modal {
  width: min(920px, 100%);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.details-modal-body {
  margin-top: 0.5rem;
  flex: 1;
  min-height: 0;
  max-height: min(68vh, 560px);
}

.details-list {
  margin: 0;
}

.details-chart-wrap {
  display: grid;
  gap: 1rem;
}

.details-chart-panel {
  display: grid;
  grid-template-columns: minmax(220px, 320px) minmax(220px, 1fr);
  gap: 1rem 1.25rem;
  align-items: center;
}

@media (max-width: 760px) {
  .details-chart-panel {
    grid-template-columns: 1fr;
  }
}

.category-detail-item,
.forensics-detail-item {
  display: grid;
  gap: 0.45rem;
  border: 1px solid rgba(159, 179, 200, 0.14);
  border-radius: 12px;
  background: rgba(6, 15, 26, 0.5);
  padding: 0.75rem 0.8rem;
}

.forensics-detail-item {
  gap: 0.3rem;
}

.details-filters {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.65rem;
}

.details-filter-btn {
  border: 1px solid rgba(159, 179, 200, 0.28);
  background: rgba(255, 255, 255, 0.04);
  color: var(--text);
  border-radius: 999px;
  padding: 0.38rem 0.8rem;
  font: inherit;
  font-size: 0.8rem;
  font-weight: 700;
  cursor: pointer;
  transition: border-color 0.18s ease, background 0.18s ease, transform 0.18s ease;
}

.details-filter-btn:hover {
  border-color: rgba(159, 179, 200, 0.54);
  transform: translateY(-1px);
}

.details-filter-btn.active {
  border-color: rgba(64, 211, 163, 0.72);
  background: linear-gradient(135deg, rgba(64, 211, 163, 0.2), rgba(245, 183, 78, 0.12));
  color: #f4fff8;
}

.details-filter-btn:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px rgba(64, 211, 163, 0.22);
}

.soar-legend.details-legend-wrap {
  margin-top: 0.35rem;
  display: none;
}

.soar-legend.details-legend-wrap.show {
  display: inline-flex;
}

.recent-card,
.categories-card,
.summary-card,
.system-card,
.soar-card {
  min-width: 0;
}

.response-list {
  max-height: none;
}

.soar-header-group {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-wrap: wrap;
}

.soar-legend {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.soar-legend-item {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  color: var(--muted);
  font-size: 0.73rem;
  letter-spacing: 0.02em;
}

.soar-legend-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
}

.soar-legend-dot.auto {
  background: rgba(64, 211, 163, 0.95);
}

.soar-legend-dot.escalate {
  background: rgba(245, 183, 78, 0.95);
}

.soar-legend-dot.benign {
  background: rgba(96, 160, 255, 0.9);
}

.soar-legend-dot.monitor {
  background: rgba(159, 179, 200, 0.9);
}

.response-item.decision-auto_response {
  border-left-color: rgba(64, 211, 163, 0.95);
}

.response-item.decision-escalate_only {
  border-left-color: rgba(245, 183, 78, 0.95);
}

.response-item.decision-monitor_only {
  border-left-color: rgba(159, 179, 200, 0.9);
}

.response-item.decision-benign {
  border-left-color: rgba(96, 160, 255, 0.8);
}

.response-item.decision-none {
  border-left-color: rgba(159, 179, 200, 0.45);
}

.integrations {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin-top: 2rem;
  position: relative;
  z-index: 1;
}

@media (min-width: 1200px) {
  .grid {
    grid-template-columns: repeat(6, minmax(0, 1fr));
  }

  .recent-card {
    grid-column: 1 / span 3;
    border-color: rgba(109, 176, 255, 0.24);
    background: linear-gradient(180deg, rgba(109, 176, 255, 0.1), var(--card));
  }

  .soar-card {
    grid-column: 4 / span 3;
    border-color: rgba(64, 211, 163, 0.24);
    background: linear-gradient(180deg, rgba(64, 211, 163, 0.1), var(--card));
  }

  .categories-card {
    grid-column: 1 / span 2;
  }

  .summary-card {
    grid-column: 3 / span 2;
  }

  .forensics-card {
    grid-column: 5 / span 2;
    border-color: rgba(245, 183, 78, 0.24);
    background: linear-gradient(180deg, rgba(245, 183, 78, 0.08), var(--card));
  }

  .system-card {
    grid-column: 1 / -1;
  }
}

@keyframes pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(64, 211, 163, 0.6);
  }
  70% {
    box-shadow: 0 0 0 12px rgba(64, 211, 163, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(64, 211, 163, 0);
  }
}

@keyframes floatIn {
  from {
    opacity: 0;
    transform: translateY(12px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 600px) {
  :root {
    --card-height: 320px;
  }

  .app {
    width: 100%;
    padding: 1.5rem;
  }

  .modal {
    border-radius: 18px;
    padding: 1.1rem;
    width: 100%;
    height: calc(100vh - 2rem);
    max-height: calc(100vh - 2rem);
  }

  .modal-header {
    margin-bottom: 0.8rem;
    padding-bottom: 0.8rem;
  }

  .modal-grid {
    padding: 0.8rem;
  }

  .mode-switch {
    grid-template-columns: 1fr;
  }

  .modal-actions {
    flex-direction: column;
    align-items: stretch;
  }

  .modal-actions .action-button {
    width: 100%;
    justify-content: center;
  }

  .bar-row {
    grid-template-columns: 1fr;
    gap: 0.4rem;
  }

  .bar-value {
    text-align: left;
  }
}

/* Detection Summary Visualization */
.detection-summary-viz {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.detection-flow-container {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.detection-flow-label {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.detection-flow-bar {
  position: relative;
  height: 44px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.06);
  overflow: hidden;
  display: flex;
  align-items: center;
  border: 1px solid rgba(159, 179, 200, 0.12);
}

.detection-flow-segment {
  position: relative;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.6s cubic-bezier(0.22, 1, 0.36, 1);
  font-weight: 700;
  font-size: 0.85rem;
  color: white;
  cursor: pointer;
  overflow: hidden;
}

.detection-flow-segment::before {
  content: '';
  position: absolute;
  inset: 0;
  background: inherit;
  filter: brightness(1.2);
  opacity: 0;
  transition: opacity 0.3s ease;
  z-index: -1;
}

.detection-flow-segment:hover::before {
  opacity: 1;
}

.detection-flow-segment.threats {
  background: linear-gradient(135deg, #ff4a5c 0%, #ff1744 100%);
}

.detection-flow-segment.threats::after {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at top left, rgba(255, 255, 255, 0.3) 0%, transparent 70%);
  opacity: 0.6;
}

.detection-flow-segment.benign {
  background: linear-gradient(135deg, #40d3a3 0%, #1dd1a1 100%);
}

.detection-flow-segment.benign::after {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at top right, rgba(255, 255, 255, 0.2) 0%, transparent 70%);
  opacity: 0.6;
}

.detection-flow-value {
  position: relative;
  z-index: 2;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.2rem;
}

.detection-flow-count {
  font-size: 0.75rem;
  opacity: 0.9;
}

.detection-flow-percentage {
  font-size: 1rem;
  font-weight: 800;
}

.detection-flow-divider {
  width: 2px;
  height: 28px;
  background: rgba(255, 255, 255, 0.2);
  margin: 0 0;
}

.detection-metrics-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.6rem;
}

.detection-metric-badge {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 0.7rem;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(159, 179, 200, 0.15);
  transition: all 0.3s cubic-bezier(0.22, 1, 0.36, 1);
  cursor: default;
}

.detection-metric-badge:hover {
  background: rgba(255, 255, 255, 0.12);
  border-color: rgba(159, 179, 200, 0.25);
  transform: translateY(-2px);
}

.detection-metric-value {
  font-size: 1.4rem;
  font-weight: 800;
  background: linear-gradient(135deg, #40d3a3, #f5b74e);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: pulse-glow 2s ease-in-out infinite;
}

.detection-metric-label {
  font-size: 0.7rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.3px;
  margin-top: 0.3rem;
  font-weight: 600;
}

@keyframes pulse-glow {
  0%, 100% {
    opacity: 1;
    filter: drop-shadow(0 0 0px rgba(64, 211, 163, 0));
  }
  50% {
    opacity: 1;
    filter: drop-shadow(0 0 6px rgba(64, 211, 163, 0.4));
  }
}

.detection-threat-rate {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem;
  border-radius: 8px;
  background: linear-gradient(135deg, rgba(255, 26, 26, 0.1) 0%, rgba(255, 74, 92, 0.05) 100%);
  border: 1px solid rgba(255, 74, 92, 0.3);
}

.threat-rate-label {
  font-size: 0.8rem;
  color: var(--text);
  font-weight: 600;
}

.threat-rate-value {
  font-size: 1.2rem;
  font-weight: 800;
  color: #ff1744;
  text-shadow: 0 0 8px rgba(255, 26, 26, 0.4);
}

@media (min-width: 1800px) {
  .app {
    width: min(100%, 1760px);
  }
}
    </style>
  </head>
  <body>
    <main class="app">
      <header class="hero">
        <div class="hero-title">
          <p class="eyebrow">AI-NIDS Monitoring</p>
          <h1>Unified Detection Dashboard</h1>
          <p class="subtitle">Real-Time Intrusion Detection.</p>
        </div>
        <div class="hero-status">
          <button class="status" id="status-toggle" type="button" title="Toggle online and offline monitoring modes">
            <span class="dot" id="status-dot"></span>
            <span id="status">Loading...</span>
          </button>
          <div class="timestamp" id="updated"></div>
          <div class="hero-actions">
            <button class="action-button" id="run-test-button" type="button" onclick="openRunTestModal()">Run Test</button>
            <button class="action-button reset-button" id="reset-metrics-button" type="button" onclick="resetDashboardMetrics()">Reset Metrics</button>
          </div>
        </div>
        <div class="monitor-warning-banner" id="monitor-warning-banner" role="alert" hidden></div>
      </header>

      <div class="modal-backdrop" id="run-test-modal" aria-hidden="true">
        <div class="modal" role="dialog" aria-modal="true" aria-labelledby="run-test-modal-title">
          <div class="modal-header">
            <div>
              <p class="modal-kicker">Run Test</p>
              <h2 id="run-test-modal-title">Choose a test target</h2>
            </div>
            <button class="modal-close" id="close-run-test" type="button" aria-label="Close dialog">&times;</button>
          </div>
          <div class="mode-switch" role="radiogroup" aria-label="Test mode">
            <label class="mode-option" for="mode-known">
              <input id="mode-known" type="radio" name="test-mode" value="known" checked />
              <span>Known attacks</span>
            </label>
            <label class="mode-option" for="mode-unknown">
              <input id="mode-unknown" type="radio" name="test-mode" value="unknown" />
              <span>Unknown traffic (upload)</span>
            </label>
          </div>

          <div class="modal-section" id="known-attack-section">
            <div class="modal-grid">
              <label>Attack type selection</label>
              <div class="chip-mode" role="radiogroup" aria-label="Attack selection mode">
                <label class="mode-option" for="attack-selection-single">
                  <input id="attack-selection-single" type="radio" name="attack-selection-mode" value="single" checked />
                  <span>Single</span>
                </label>
                <label class="mode-option" for="attack-selection-multi">
                  <input id="attack-selection-multi" type="radio" name="attack-selection-mode" value="multi" />
                  <span>Multiple</span>
                </label>
              </div>
              <div id="attack-chip-list" class="chip-list" aria-label="Attack chips"></div>
              <p class="modal-help" id="attack-selection-help">Select one attack chip to run. Switch to Multiple to choose several attacks.</p>
            </div>
          </div>

          <div class="modal-section" id="unknown-upload-section" hidden>
            <div class="modal-grid">
              <label for="upload-csv">Upload CSV file</label>
              <input id="upload-csv" class="csv-input" type="file" accept=".csv,text/csv" />
              <p class="modal-help">Upload a CSV captured from your environment to classify likely attack types.</p>
            </div>
          </div>

          <div class="modal-actions">
            <button class="action-button secondary-button" id="cancel-run-test" type="button" onclick="closeRunTestModal()">Cancel</button>
            <button class="action-button secondary-button" id="analyze-upload-csv" type="button" hidden onclick="analyzeUploadedCsv()">Analyze uploaded CSV</button>
            <button class="action-button primary-launch" id="launch-run-test" type="button" onclick="runSelectedAttack()">Run selected attack</button>
          </div>
          <div class="console-wrap">
            <p class="console-title">Live Output</p>
            <div class="console" id="run-test-console" aria-live="polite">
              <div class="console-line dim">Select an attack and start the test to see live output here.</div>
            </div>
          </div>
        </div>
      </div>

      <div class="modal-backdrop" id="details-modal" aria-hidden="true">
        <div class="modal details-modal" role="dialog" aria-modal="true" aria-labelledby="details-modal-title">
          <div class="modal-header">
            <div>
              <h2 id="details-modal-title">All events</h2>
              <div class="soar-legend details-legend-wrap" id="details-soar-legend" aria-label="SOAR response legend">
                <span class="soar-legend-item"><span class="soar-legend-dot auto"></span>auto-response</span>
                <span class="soar-legend-item"><span class="soar-legend-dot escalate"></span>escalate-only</span>
                <span class="soar-legend-item"><span class="soar-legend-dot benign"></span>benign</span>
                <span class="soar-legend-item"><span class="soar-legend-dot monitor"></span>monitor</span>
              </div>
              <div class="details-filters" id="details-filters" aria-label="Prediction and response filters">
                <button class="details-filter-btn active" type="button" data-filter="all">All</button>
                <button class="details-filter-btn" type="button" data-filter="threats">Threats</button>
                <button class="details-filter-btn" type="button" data-filter="benign">Benign</button>
              </div>
            </div>
            <button class="modal-close" id="close-details-modal" type="button" aria-label="Close details dialog">&times;</button>
          </div>
          <div class="details-modal-body scroll-pane">
            <ul class="metrics details-list" id="details-modal-list"></ul>
          </div>
        </div>
      </div>

      <div class="modal-backdrop" id="reset-modal" aria-hidden="true">
        <div class="modal reset-modal" role="dialog" aria-modal="true" aria-labelledby="reset-modal-title">
          <div class="modal-header">
            <div>
              <p class="modal-kicker">Confirm reset</p>
              <h2 id="reset-modal-title">Reset dashboard metrics?</h2>
            </div>
            <button class="modal-close" id="close-reset-modal" type="button" aria-label="Close reset dialog">&times;</button>
          </div>
          <div class="reset-modal-body">
            <p class="reset-warning">This will clear <strong>all dashboard metrics, recent predictions, and SOAR response logs</strong>.</p>
            <p class="modal-help">This action cannot be undone.</p>
          </div>
          <div class="modal-actions">
            <button class="action-button secondary-button" id="cancel-reset-modal" type="button" onclick="closeResetModal()">Cancel</button>
            <button class="action-button reset-button" id="confirm-reset-modal" type="button" onclick="confirmResetDashboardMetrics()">Reset Metrics</button>
          </div>
        </div>
      </div>

      <section class="kpi-grid">
        <div class="kpi danger">
          <p>Total Predictions</p>
          <h2 id="total-predictions">0</h2>
          <span>All time</span>
        </div>
        <div class="kpi danger">
          <p>Threats Detected</p>
          <h2 id="threats-detected">0</h2>
          <span id="threat-rate">0% of total</span>
        </div>
        <div class="kpi success">
          <p>Benign Traffic</p>
          <h2 id="benign-count">0</h2>
          <span>Normal traffic</span>
        </div>
        <div class="kpi">
          <p>Avg Confidence</p>
          <h2 id="avg-confidence">0.0%</h2>
          <span>Detection certainty</span>
        </div>
      </section>

      <section class="grid">
        <div class="card recent-card expandable-card" id="recent-card" role="button" tabindex="0" aria-label="Open all recent predictions">
          <div class="card-header">
            <div class="card-header-main">
              <h2>Recent Predictions</h2>
            </div>
            <div class="card-header-actions">
              <button class="view-all-btn" id="view-all-recent" type="button" aria-label="View all recent predictions">
                <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <path d="M8 8h8v8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path>
                  <path d="M16 8l-9 9" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path>
                </svg>
              </button>
              <span class="pill" id="recent-count">0</span>
            </div>
          </div>
          <ul id="alerts" class="preview-list prediction-list"></ul>
        </div>
        <div class="card soar-card expandable-card" id="soar-card" role="button" tabindex="0" aria-label="Open all SOAR responses">
          <div class="card-header">
            <div class="soar-header-group">
              <div class="card-header-main">
                <h2>SOAR Responses</h2>
              </div>
            </div>
            <div class="card-header-actions">
              <button class="view-all-btn" id="view-all-responses" type="button" aria-label="View all SOAR responses">
                <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <path d="M8 8h8v8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path>
                  <path d="M16 8l-9 9" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path>
                </svg>
              </button>
              <span class="pill" id="response-count">0</span>
            </div>
          </div>
          <ul class="metrics response-list preview-list" id="response-actions"></ul>
        </div>
        <div class="card categories-card expandable-card" id="categories-card" role="button" tabindex="0" aria-label="Open all attack categories">
          <div class="card-header">
            <div class="card-header-main">
              <h2>Attack Categories</h2>
            </div>
            <div class="card-header-actions">
              <button class="view-all-btn" id="view-all-categories" type="button" aria-label="View all attack categories">
                <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <path d="M8 8h8v8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path>
                  <path d="M16 8l-9 9" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path>
                </svg>
              </button>
              <span class="pill danger">Top Threats</span>
            </div>
          </div>
          <div id="categories" class="pie-card-body scroll-pane"></div>
        </div>
        <div class="card summary-card">
          <div class="card-header">
            <h2>Detection Summary</h2>
            <span class="pill" id="summary-pill">Live</span>
          </div>
          <ul class="metrics scroll-pane" id="summary-metrics"></ul>
        </div>
        <div class="card forensics-card expandable-card" id="forensics-card" role="button" tabindex="0" aria-label="Open forensic evidence">
          <div class="card-header">
            <div class="card-header-main">
              <h2>Forensics</h2>
            </div>
            <div class="card-header-actions">
              <button class="view-all-btn" id="view-all-forensics" type="button" aria-label="View all forensic evidence">
                <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <path d="M8 8h8v8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path>
                  <path d="M16 8l-9 9" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path>
                </svg>
              </button>
              <span class="pill">Evidence</span>
            </div>
          </div>
          <ul class="metrics scroll-pane" id="forensics-metrics"></ul>
        </div>
        <div class="card system-card">
          <div class="card-header">
            <h2>System Status</h2>
            <span class="pill" id="system-pill">OK</span>
          </div>
          <div class="metrics scroll-pane system-status-wrap" id="system-metrics"></div>
        </div>
      </section>
    </main>
    <script>
function formatPct(value) {
  if (value === undefined || value === null || isNaN(value)) return "0.0%";
  return (value * 100).toFixed(1) + "%";
}

function renderBars(container, items, isThreat = true) {
  container.innerHTML = "";
  if (!items || !items.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No data yet - Run tests to see metrics";
    container.appendChild(empty);
    return;
  }
  const max = Math.max(...items.map((item) => item.value), 1);
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const label = document.createElement("span");
    label.className = "bar-label";
    label.textContent = item.label;
    label.title = item.label;

    const bar = document.createElement("span");
    bar.className = "bar-fill" + (isThreat ? " danger" : "");

    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = item.value;

    const track = document.createElement("span");
    track.className = "bar-track";
    track.appendChild(bar);

    bar.style.width = ((item.value / max) * 100) + "%";

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    container.appendChild(row);
  });
}

function buildPieSegments(entries) {
  const total = entries.reduce((sum, item) => sum + Math.max(Number(item.value) || 0, 0), 0);
  if (!total) {
    return { total: 0, segments: [] };
  }

  const palette = ["#40d3a3", "#f5b74e", "#ff6767", "#6fa8ff", "#8b9dff", "#f28bb5", "#67d7ff", "#d6b56d"];
  let offset = 0;
  const segments = entries.map((entry, index) => {
    const value = Math.max(Number(entry.value) || 0, 0);
    const fraction = value / total;
    const segment = {
      label: entry.label,
      value,
      fraction,
      start: offset,
      end: offset + fraction,
      color: palette[index % palette.length],
    };
    offset += fraction;
    return segment;
  });

  return { total, segments };
}

function renderPieChart(container, entries, options = {}) {
  if (!container) return;
  const compact = options.compact !== false;
  const title = options.title || "Total";
  const showLegend = options.showLegend !== false;
  const maxLegend = options.maxLegend || (compact ? 4 : entries.length);
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const { total, segments } = buildPieSegments(entries || []);

  container.innerHTML = "";

  if (!segments.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No category data yet - run tests to populate the chart.";
    container.appendChild(empty);
    return;
  }

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 120 120");
  svg.setAttribute("class", compact ? "pie-chart" : "pie-chart pie-detail-chart");
  svg.setAttribute("aria-label", "Attack categories pie chart");
  svg.setAttribute("role", "img");

  const ring = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  ring.setAttribute("class", "pie-chart-ring");
  ring.setAttribute("cx", "60");
  ring.setAttribute("cy", "60");
  ring.setAttribute("r", String(radius));
  svg.appendChild(ring);

  segments.forEach((segment) => {
    const slice = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    slice.setAttribute("class", "pie-chart-slice");
    slice.setAttribute("cx", "60");
    slice.setAttribute("cy", "60");
    slice.setAttribute("r", String(radius));
    slice.setAttribute("stroke", segment.color);
    slice.setAttribute("stroke-dasharray", `${segment.fraction * circumference} ${circumference}`);
    slice.setAttribute("stroke-dashoffset", String(-segment.start * circumference));
    svg.appendChild(slice);
  });

  const center = document.createElement("div");
  center.className = "pie-chart-center";
  center.innerHTML = `<strong>${total}</strong><span>${title}</span>`;

  const chartWrap = document.createElement("div");
  chartWrap.className = "pie-chart-shell";
  chartWrap.appendChild(svg);
  chartWrap.appendChild(center);
  container.appendChild(chartWrap);

  const centerValueEl = center.querySelector("strong");
  const centerLabelEl = center.querySelector("span");
  const resetCenter = () => {
    if (centerValueEl) centerValueEl.textContent = String(total);
    if (centerLabelEl) centerLabelEl.textContent = title;
    svg.querySelectorAll(".pie-chart-slice.active").forEach((activeSlice) => {
      activeSlice.classList.remove("active");
    });
  };

  svg.querySelectorAll(".pie-chart-slice").forEach((slice, index) => {
    const segment = segments[index];
    if (!segment) return;

    const tooltip = document.createElementNS("http://www.w3.org/2000/svg", "title");
    tooltip.textContent = `${segment.label}: ${(segment.fraction * 100).toFixed(1)}%`;
    slice.appendChild(tooltip);

    const showSegment = () => {
      svg.querySelectorAll(".pie-chart-slice.active").forEach((activeSlice) => {
        activeSlice.classList.remove("active");
      });
      slice.classList.add("active");
      if (centerValueEl) centerValueEl.textContent = `${(segment.fraction * 100).toFixed(1)}%`;
      if (centerLabelEl) centerLabelEl.textContent = segment.label;
    };

    slice.addEventListener("mouseenter", showSegment);
  });

  svg.addEventListener("mouseleave", resetCenter);
  chartWrap.addEventListener("mouseleave", resetCenter);

  if (!showLegend) {
    return;
  }

  const legend = document.createElement("div");
  legend.className = compact ? "pie-legend" : "pie-detail-list";

  segments.slice(0, maxLegend).forEach((segment) => {
    const row = document.createElement("div");
    row.className = compact ? "pie-legend-item" : "pie-detail-row";

    const key = document.createElement("span");
    key.className = "pie-legend-key";

    const swatch = document.createElement("span");
    swatch.className = "pie-swatch";
    swatch.style.background = segment.color;

    const label = document.createElement("span");
    label.textContent = segment.label;

    key.appendChild(swatch);
    key.appendChild(label);

    const value = document.createElement("strong");
    const pct = total ? ((segment.value / total) * 100).toFixed(1) : "0.0";
    value.textContent = `${segment.value} (${pct}%)`;

    row.appendChild(key);
    row.appendChild(value);
    legend.appendChild(row);
  });

  if (!compact && segments.length > maxLegend) {
    const overflow = document.createElement("div");
    overflow.className = "empty";
    overflow.textContent = `${segments.length - maxLegend} more categories are hidden in the summary view.`;
    legend.appendChild(overflow);
  }

  container.appendChild(legend);
}

let attackSelectionMode = "single";
let selectedAttacks = new Set(["BENIGN"]);
let detailsViewType = "recent";
let detailsFilter = "all";
let latestThreatTypes = {};
let latestForensicsSummary = [];
let latestRecentForensics = [];
let currentSystemMode = "offline";
let systemModeBusy = false;
let dashboardApiLive = false;

function updateMonitorWarningBanner(modeState) {
  const banner = document.getElementById("monitor-warning-banner");
  if (!banner) {
    return;
  }

  const state = modeState || {};
  const mode = state.mode === "online" ? "online" : "offline";
  const monitoring = Boolean(state.monitoring);
  const warningText = (state.warning || state.message || "").trim();
  const shouldShow = mode === "online" && !monitoring;

  if (!shouldShow) {
    banner.hidden = true;
    banner.textContent = "";
    return;
  }

  banner.hidden = false;
  banner.textContent = warningText || "Online mode selected, but external monitor is not running.";
}

function updateTopStatus(isLive, modeState) {
  const status = document.getElementById("status");
  const statusDot = document.getElementById("status-dot");
  const statusToggle = document.getElementById("status-toggle");
  const state = modeState || {};
  const mode = state.mode === "online" ? "online" : "offline";
  const apiLabel = isLive ? "Live" : "Offline";
  const modeLabel = mode === "online" ? "online mode" : "offline mode";

  if (status) {
    status.textContent = `${apiLabel} | ${modeLabel}`;
  }
  if (statusDot) {
    statusDot.classList.toggle("live", isLive);
  }
  if (statusToggle) {
    statusToggle.setAttribute("aria-label", `Current mode: ${modeLabel}. Click to toggle mode.`);
    statusToggle.setAttribute("aria-pressed", mode === "online" ? "true" : "false");
    statusToggle.disabled = systemModeBusy;
  }

  updateMonitorWarningBanner(state);
}

function applySystemModeUI(modeState) {
  const state = modeState || {};
  const mode = state.mode === "online" ? "online" : "offline";
  currentSystemMode = mode;

  const systemPill = document.getElementById("system-pill");

  if (systemPill) {
    const monitoring = Boolean(state.monitoring);
    systemPill.textContent = monitoring ? "ONLINE" : "OFFLINE";
  }

  updateMonitorWarningBanner(state);
}

async function setSystemMode(mode) {
  const desiredMode = mode === "online" ? "online" : "offline";
  const previousMode = currentSystemMode;
  systemModeBusy = true;
  applySystemModeUI({ mode: previousMode, monitoring: previousMode === "online" });

  try {
    const res = await fetch('/api/system-mode', {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ mode: desiredMode })
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Unable to switch system mode.");
    }

    const nextState = data.system_mode || { mode: desiredMode, monitoring: desiredMode === "online" };
    applySystemModeUI(nextState);
    updateTopStatus(dashboardApiLive, nextState);
    await loadSummary();
  } catch (err) {
    console.error("Error switching system mode:", err);
    const fallbackState = {
      mode: previousMode,
      monitoring: previousMode === "online",
      message: err.message || "Unable to switch system mode.",
      warning: previousMode === "online" ? (err.message || "Online monitoring failed.") : "",
    };
    applySystemModeUI(fallbackState);
    updateTopStatus(dashboardApiLive, fallbackState);
  } finally {
    systemModeBusy = false;
    updateTopStatus(dashboardApiLive, { mode: currentSystemMode, monitoring: currentSystemMode === "online" });
  }
}

function handleSystemModeToggle() {
  if (systemModeBusy) {
    return;
  }
  const nextMode = currentSystemMode === "online" ? "offline" : "online";
  setSystemMode(nextMode);
}

function matchesDetailsFilter(type, filter, item) {
  if (filter === "all") {
    return true;
  }

  if (type === "recent") {
    const isThreat = item.is_threat || item.threat || false;
    return filter === "threats" ? isThreat : !isThreat;
  }

  const decision = String(item.decision || "").toLowerCase();
  const isThreat = decision !== "benign";
  return filter === "threats" ? isThreat : !isThreat;
}

function renderCategoryDetails(container, threatTypes) {
  if (!container) return;
  container.innerHTML = "";

  const entries = Object.entries(threatTypes || {})
    .map(([label, value]) => ({ label, value }))
    .sort((a, b) => b.value - a.value);

  if (!entries.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "No attack categories yet - run tests to populate the breakdown.";
    container.appendChild(empty);
    return;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "details-chart-wrap";

  const chartPanel = document.createElement("div");
  chartPanel.className = "details-chart-panel";

  const chartHolder = document.createElement("div");
  chartHolder.className = "pie-card-body";
  chartPanel.appendChild(chartHolder);

  const detailList = document.createElement("div");
  detailList.className = "pie-detail-list";

  chartPanel.appendChild(detailList);
  wrapper.appendChild(chartPanel);
  container.appendChild(wrapper);

  renderPieChart(chartHolder, entries, { compact: false, title: "Total categories", maxLegend: 8, showLegend: false });

  const chartSegments = buildPieSegments(entries).segments;
  chartSegments.forEach((segment) => {
    const row = document.createElement("div");
    row.className = "response-item category-response-item";
    row.style.borderLeftColor = segment.color;

    const header = document.createElement("div");
    header.className = "response-item-header";

    const label = document.createElement("span");
    label.className = "response-label";
    label.textContent = segment.label;

    const pct = document.createElement("span");
    pct.className = "confidence-badge";
    pct.textContent = `${(segment.fraction * 100).toFixed(1)}%`;

    header.appendChild(label);
    header.appendChild(pct);

    const meta = document.createElement("span");
    meta.className = "response-meta";
    meta.textContent = `Detected ${segment.value} events`;

    const actionText = document.createElement("span");
    actionText.className = "response-actions";
    actionText.textContent = `Category share of all attacks`;

    row.appendChild(header);
    row.appendChild(meta);
    row.appendChild(actionText);
    detailList.appendChild(row);
  });
}

function renderForensicsDetailList(container, summary, recentEntries) {
  if (!container) return;
  container.innerHTML = "";

  const rows = Array.isArray(recentEntries) ? recentEntries : [];
  const counts = Array.isArray(summary) ? summary : [];

  if (rows.length) {
    rows.forEach((entry) => {
      const item = document.createElement("li");
      item.className = "forensics-detail-item";

      const title = entry.title || "forensic record";
      const detail = entry.detail || "recorded";
      const source = entry.source || "artifact";
      const ts = entry.ts ? new Date(entry.ts).toLocaleTimeString() : "N/A";

      item.innerHTML = "<span class='prediction-label'>" + title + "</span><span class='response-meta'>" + source + " | " + detail + "</span><span class='response-actions'>" + ts + "</span>";
      container.appendChild(item);
    });
    return;
  }

  if (!counts.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "No forensic evidence yet - trigger detections to populate artifacts.";
    container.appendChild(empty);
    return;
  }

  counts.forEach((entry) => {
    const item = document.createElement("li");
    item.className = "forensics-detail-item";
    item.innerHTML = "<span class='prediction-label'>" + entry.label + "</span><span class='confidence-badge'>" + entry.value + "</span>";
    container.appendChild(item);
  });
}

function setDetailsFilter(filter) {
  detailsFilter = filter === "threats" || filter === "benign" ? filter : "all";

  document.querySelectorAll(".details-filter-btn").forEach((button) => {
    button.classList.toggle("active", button.dataset.filter === detailsFilter);
  });

  const modalList = document.getElementById("details-modal-list");
  if (detailsViewType === "responses") {
    renderResponseActionsList(modalList, latestRecentResponses, detailsFilter, true);
  } else {
    renderPredictionsList(modalList, latestRecentPredictions, detailsFilter, true);
  }
}

function renderAttackChips() {
  const container = document.getElementById("attack-chip-list");
  if (!container) {
    return;
  }

  container.innerHTML = "";
  RUN_TEST_OPTIONS.forEach((option) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chip-button" + (selectedAttacks.has(option.value) ? " selected" : "");
    button.textContent = option.label;
    button.setAttribute("aria-pressed", selectedAttacks.has(option.value) ? "true" : "false");
    button.addEventListener("click", () => toggleAttackChip(option.value));
    container.appendChild(button);
  });

  const helpEl = document.getElementById("attack-selection-help");
  if (helpEl) {
    helpEl.textContent = attackSelectionMode === "multi"
      ? "Select multiple attacks. Choosing Benign clears other selections."
      : "Select one attack to run.";
  }
}

function setAttackSelectionMode(mode) {
  attackSelectionMode = mode === "multi" ? "multi" : "single";
  if (!selectedAttacks.size) {
    selectedAttacks = new Set(["BENIGN"]);
  }

  if (attackSelectionMode === "single" && selectedAttacks.size > 1) {
    const first = selectedAttacks.values().next().value;
    selectedAttacks = new Set([first]);
  }

  renderAttackChips();
}

function toggleAttackChip(value) {
  if (attackSelectionMode === "single") {
    selectedAttacks = new Set([value]);
    renderAttackChips();
    return;
  }

  if (value === "BENIGN") {
    selectedAttacks = new Set(["BENIGN"]);
    renderAttackChips();
    return;
  }

  if (selectedAttacks.has("BENIGN")) {
    selectedAttacks.delete("BENIGN");
  }

  if (selectedAttacks.has(value)) {
    selectedAttacks.delete(value);
  } else {
    selectedAttacks.add(value);
  }

  if (!selectedAttacks.size) {
    selectedAttacks = new Set(["BENIGN"]);
  }

  renderAttackChips();
}

function renderPredictionsList(container, items, filter = "all") {
  if (!container) return;
  container.innerHTML = "";

  if (!items || !items.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "No predictions yet - Run tests to populate metrics";
    container.appendChild(empty);
    return;
  }

  const filteredItems = items.filter((item) => matchesDetailsFilter("recent", filter, item));

  if (!filteredItems.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "No predictions match this filter.";
    container.appendChild(empty);
    return;
  }

  filteredItems.forEach((pred) => {
    const item = document.createElement("li");
    const label = pred.label || pred.prediction || "unknown";
    const confidence = pred.confidence || 0;
    const isThreat = pred.is_threat || pred.threat || false;
    const ts = pred.timestamp;

    item.className = "prediction-item" + (isThreat ? " threat" : "");
    let tsText = "";
    if (ts) {
      const date = new Date(ts * 1000);
      tsText = date.toLocaleTimeString();
    }

    const header = document.createElement("div");
    header.className = "prediction-item-header";

    const labelEl = document.createElement("span");
    labelEl.className = "prediction-label";
    labelEl.textContent = label;

    const confidenceEl = document.createElement("span");
    confidenceEl.className = "confidence-badge";
    confidenceEl.textContent = formatPct(confidence);

    header.appendChild(labelEl);
    header.appendChild(confidenceEl);

    const meta = document.createElement("span");
    meta.className = "prediction-meta";
    meta.textContent = tsText ? ("Time: " + tsText) : "Time: N/A";

    item.appendChild(header);
    item.appendChild(meta);
    container.appendChild(item);
  });
}

function renderResponseActionsList(container, entries, filter = "all") {
  if (!container) return;
  container.innerHTML = "";

  if (!entries || !entries.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "No response actions yet - trigger a threat to see SOAR actions.";
    container.appendChild(empty);
    return;
  }

  const filteredEntries = entries.filter((entry) => matchesDetailsFilter("responses", filter, entry));

  if (!filteredEntries.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "No responses match this filter.";
    container.appendChild(empty);
    return;
  }

  filteredEntries.forEach((entry) => {
    const item = document.createElement("li");

    const label = entry.label || "unknown";
    const confidence = typeof entry.confidence === "number" ? formatPct(entry.confidence) : "0.0%";
    const decision = entry.decision || "unknown";
    const ts = entry.ts || 0;
    const tsText = ts ? new Date(ts).toLocaleTimeString() : "";
    const actions = (entry.actions || []).map((a) => a.action).filter(Boolean);
    const decisionClass = (decision || "none").replaceAll("-", "_");

    item.className = "response-item decision-" + decisionClass;

    const header = document.createElement("div");
    header.className = "response-item-header";

    const headline = document.createElement("span");
    headline.className = "response-label";
    headline.textContent = label;

    const decisionTag = document.createElement("span");
    decisionTag.className = "confidence-badge";
    decisionTag.textContent = decision;

    header.appendChild(headline);
    header.appendChild(decisionTag);

    const meta = document.createElement("span");
    meta.className = "response-meta";
    meta.textContent = confidence + (tsText ? " | " + tsText : " | N/A");

    const actionText = document.createElement("span");
    actionText.className = "response-actions";
    actionText.textContent = actions.length ? ("Actions: " + actions.join(", ")) : "Actions: none";

    item.appendChild(header);
    item.appendChild(meta);
    item.appendChild(actionText);
    container.appendChild(item);
  });
}

function renderForensicsList(container, entries) {
  if (!container) return;
  container.innerHTML = "";

  if (!entries || !entries.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "No forensic records yet - trigger detections to populate evidence.";
    container.appendChild(empty);
    return;
  }

  entries.forEach((entry) => {
    const item = document.createElement("li");
    item.innerHTML = "<span>" + entry.label + "</span><strong>" + entry.value + "</strong>";
    container.appendChild(item);
  });
}

function renderSystemStatusViz(container, payload) {
  if (!container) return;
  container.innerHTML = "";

  const connectivity = payload.connectivity;
  const freshness = payload.freshness;
  const confidence = payload.confidence;
  const threatPressure = payload.threatPressure;

  const healthScore = Math.max(0, Math.min(100, Math.round(
    (connectivity * 0.35) +
    (freshness * 0.25) +
    (confidence * 0.20) +
    ((100 - threatPressure) * 0.20)
  )));

  const statusText = healthScore >= 80 ? "Healthy" : (healthScore >= 55 ? "Degraded" : "At Risk");

  const summary = document.createElement("div");
  summary.className = "system-health";
  summary.innerHTML = ""
    + "<div class='system-health-ring' style='--health: " + healthScore + ";'>"
    + "<span class='system-health-value'>" + healthScore + "%</span>"
    + "</div>"
    + "<div class='system-health-meta'>"
    + "<span class='system-health-title'>" + statusText + "</span>"
    + "<span class='system-health-subtitle'>Model: " + payload.model + "</span>"
    + "<span class='system-health-subtitle'>Updated: " + payload.lastUpdateLabel + "</span>"
    + "</div>";

  const bars = document.createElement("div");
  bars.className = "system-health-bars";

  const metrics = [
    { label: "Connectivity", value: connectivity },
    { label: "Freshness", value: freshness },
    { label: "Confidence", value: confidence },
    { label: "Threat Load", value: threatPressure }
  ];

  metrics.forEach((metric) => {
    const clamped = Math.max(0, Math.min(100, Math.round(metric.value)));
    const row = document.createElement("div");
    row.className = "system-bar-row";
    row.innerHTML = ""
      + "<span class='system-bar-label'>" + metric.label + "</span>"
      + "<span class='system-bar-track'><span class='system-bar-fill' style='width: " + clamped + "%;'></span></span>"
      + "<span class='system-bar-value'>" + clamped + "%</span>";
    bars.appendChild(row);
  });

  container.appendChild(summary);
  container.appendChild(bars);
}

function renderDetectionSummaryViz(container, payload) {
  if (!container) return;
  container.innerHTML = "";

  const threatCount = payload.threatCount || 0;
  const benignCount = payload.benignCount || 0;
  const totalCount = threatCount + benignCount;
  const attackTypeCount = payload.attackTypeCount || 0;
  const recentEventsCount = payload.recentEventsCount || 0;
  const threatRate = totalCount > 0 ? Math.round((threatCount / totalCount) * 100) : 0;
  const benignRate = totalCount > 0 ? Math.round((benignCount / totalCount) * 100) : 0;

  const wrapper = document.createElement("div");
  wrapper.className = "detection-summary-viz";

  // Flow bar showing threat vs benign
  const flowContainer = document.createElement("div");
  flowContainer.className = "detection-flow-container";

  const flowLabel = document.createElement("div");
  flowLabel.className = "detection-flow-label";
  flowLabel.textContent = "Detection Breakdown";
  flowContainer.appendChild(flowLabel);

  const flowBar = document.createElement("div");
  flowBar.className = "detection-flow-bar";

  const threatSegment = document.createElement("div");
  threatSegment.className = "detection-flow-segment threats";
  threatSegment.style.width = threatRate + "%";
  threatSegment.innerHTML = "<div class='detection-flow-value'><span class='detection-flow-percentage'>" + threatRate + "%</span><span class='detection-flow-count'>" + threatCount.toLocaleString() + "</span></div>";
  flowBar.appendChild(threatSegment);

  if (threatRate > 0 && benignRate > 0) {
    const divider = document.createElement("div");
    divider.className = "detection-flow-divider";
    flowBar.appendChild(divider);
  }

  const benignSegment = document.createElement("div");
  benignSegment.className = "detection-flow-segment benign";
  benignSegment.style.width = benignRate + "%";
  benignSegment.innerHTML = "<div class='detection-flow-value'><span class='detection-flow-percentage'>" + benignRate + "%</span><span class='detection-flow-count'>" + benignCount.toLocaleString() + "</span></div>";
  flowBar.appendChild(benignSegment);

  flowContainer.appendChild(flowBar);
  wrapper.appendChild(flowContainer);

  // Metrics grid
  const metricsGrid = document.createElement("div");
  metricsGrid.className = "detection-metrics-grid";

  const attackTypesBadge = document.createElement("div");
  attackTypesBadge.className = "detection-metric-badge";
  attackTypesBadge.innerHTML = ""
    + "<div class='detection-metric-value'>" + attackTypeCount + "</div>"
    + "<div class='detection-metric-label'>Attack Types</div>";
  metricsGrid.appendChild(attackTypesBadge);

  const recentEventsBadge = document.createElement("div");
  recentEventsBadge.className = "detection-metric-badge";
  recentEventsBadge.innerHTML = ""
    + "<div class='detection-metric-value'>" + recentEventsCount + "</div>"
    + "<div class='detection-metric-label'>Recent Events</div>";
  metricsGrid.appendChild(recentEventsBadge);

  wrapper.appendChild(metricsGrid);

  // Threat rate indicator
  const threatRateBox = document.createElement("div");
  threatRateBox.className = "detection-threat-rate";
  threatRateBox.innerHTML = ""
    + "<div class='threat-rate-label'>THREAT RATE</div>"
    + "<div class='threat-rate-value'>" + threatRate + "%</div>";
  wrapper.appendChild(threatRateBox);

  container.appendChild(wrapper);
}

let latestRecentPredictions = [];
let latestRecentResponses = [];

function openDetailsModal(type) {
  const modal = document.getElementById("details-modal");
  const titleEl = document.getElementById("details-modal-title");
  const listEl = document.getElementById("details-modal-list");
  const legendEl = document.getElementById("details-soar-legend");
  const filtersEl = document.getElementById("details-filters");

  if (!modal || !titleEl || !listEl || !legendEl || !filtersEl) {
    return;
  }

  detailsViewType = ["responses", "categories", "forensics"].includes(type) ? type : "recent";
  detailsFilter = "all";

  filtersEl.style.display = (detailsViewType === "recent" || detailsViewType === "responses") ? "flex" : "none";

  if (detailsViewType === "responses") {
    titleEl.textContent = "All SOAR Response Actions";
    legendEl.classList.add("show");
    renderResponseActionsList(listEl, latestRecentResponses, detailsFilter);
  } else if (detailsViewType === "recent") {
    titleEl.textContent = "All Recent Predictions";
    legendEl.classList.remove("show");
    renderPredictionsList(listEl, latestRecentPredictions, detailsFilter);
  } else if (detailsViewType === "categories") {
    titleEl.textContent = "All Attack Categories";
    legendEl.classList.remove("show");
    renderCategoryDetails(listEl, latestThreatTypes);
  } else {
    titleEl.textContent = "Forensic Evidence";
    legendEl.classList.remove("show");
    renderForensicsDetailList(listEl, latestForensicsSummary, latestRecentForensics);
  }

  document.querySelectorAll(".details-filter-btn").forEach((button) => {
    button.classList.toggle("active", button.dataset.filter === "all");
  });

  modal.classList.add("open");
  modal.setAttribute("aria-hidden", "false");
}

function closeDetailsModal() {
  const modal = document.getElementById("details-modal");
  const legendEl = document.getElementById("details-soar-legend");
  const filtersEl = document.getElementById("details-filters");
  if (!modal) {
    return;
  }
  modal.classList.remove("open");
  modal.setAttribute("aria-hidden", "true");
  if (legendEl) {
    legendEl.classList.remove("show");
  }
  if (filtersEl) {
    filtersEl.style.display = "none";
  }
}

async function loadSummary() {
  const alertsEl = document.getElementById("alerts");
  const categoriesEl = document.getElementById("categories");
  const summaryMetricsEl = document.getElementById("summary-metrics");
  const forensicsMetricsEl = document.getElementById("forensics-metrics");
  const systemMetricsEl = document.getElementById("system-metrics");
  const responseActionsEl = document.getElementById("response-actions");
  const updatedEl = document.getElementById("updated");

  try {
    const res = await fetch('/api/metrics?ts=' + Date.now(), { cache: "no-store" });
    const data = await res.json();
    dashboardApiLive = true;

    const metrics = data.metrics || {};
    const recent = metrics.recent_predictions || [];
    const threatTypes = metrics.threat_types || {};
    const totalPredictions = metrics.total_predictions || 0;
    const threatsDetected = metrics.threats_detected || 0;
    const benignCount = metrics.benign_count || 0;
    const recentResponses = metrics.recent_response_actions || [];
    const forensicsSummary = metrics.forensics_summary || [];
    const recentForensics = metrics.recent_forensics || [];
    const systemMode = metrics.system_mode || {};
    latestThreatTypes = threatTypes;
    latestForensicsSummary = forensicsSummary;
    latestRecentForensics = recentForensics;
    latestRecentPredictions = recent;
    latestRecentResponses = recentResponses;
    applySystemModeUI(systemMode);
    updateTopStatus(true, systemMode);

    // Update KPIs
    document.getElementById("total-predictions").textContent = totalPredictions.toLocaleString();
    document.getElementById("threats-detected").textContent = threatsDetected.toLocaleString();
    document.getElementById("benign-count").textContent = benignCount.toLocaleString();

    const threatRate = totalPredictions > 0 ? threatsDetected / totalPredictions : 0;
    document.getElementById("threat-rate").textContent = formatPct(threatRate) + " of total";

    // Calculate average confidence from recent predictions
    let avgConf = 0;
    if (recent.length > 0) {
      const confSum = recent.reduce((sum, p) => sum + (p.confidence || 0), 0);
      avgConf = confSum / recent.length;
    }
    document.getElementById("avg-confidence").textContent = formatPct(avgConf);

    // Update recent predictions list
    alertsEl.innerHTML = "";
    const displayRecent = recent.slice(0, 10);
    document.getElementById("recent-count").textContent = displayRecent.length + " events";
    renderPredictionsList(alertsEl, displayRecent);

    // Update attack categories
    const categoryItems = Object.entries(threatTypes)
      .map(([label, value]) => ({ label, value }))
      .sort((a, b) => b.value - a.value);
    renderPieChart(categoriesEl, categoryItems, { compact: true, title: "Threats", maxLegend: 4, showLegend: false });

    // Update summary metrics with new visualization
    renderDetectionSummaryViz(summaryMetricsEl, {
      threatCount: threatsDetected,
      benignCount: benignCount,
      attackTypeCount: Object.keys(threatTypes).length,
      recentEventsCount: recent.length
    });

    // Update system status visualization
    const rawLastUpdate = metrics.last_update || "";
    const parsedLastUpdate = Date.parse(rawLastUpdate);
    const fallbackUpdated = (data.updated_at || 0) * (data.updated_at > 1000000000000 ? 1 : 1000);
    const effectiveLastUpdate = Number.isFinite(parsedLastUpdate) && parsedLastUpdate > 0 ? parsedLastUpdate : fallbackUpdated;
    const ageMs = effectiveLastUpdate > 0 ? Math.max(0, Date.now() - effectiveLastUpdate) : 300000;
    const freshness = Math.max(0, Math.min(100, 100 - Math.round(ageMs / 1000 / 3)));
    const confidencePct = Math.max(0, Math.min(100, Math.round(avgConf * 100)));
    const threatLoadPct = totalPredictions > 0 ? Math.round((threatsDetected / totalPredictions) * 100) : 0;
    renderSystemStatusViz(systemMetricsEl, {
      connectivity: METRICS_AVAILABLE ? 100 : 55,
      freshness,
      confidence: confidencePct,
      threatPressure: threatLoadPct,
      model: "CNN-LSTM",
      lastUpdateLabel: metrics.last_update || "N/A"
    });

    if (forensicsMetricsEl) {
      renderForensicsList(forensicsMetricsEl, forensicsSummary);
    }

    // Update SOAR response actions
    if (responseActionsEl) {
      responseActionsEl.innerHTML = "";
      document.getElementById("response-count").textContent = recentResponses.length + " events";
      renderResponseActionsList(responseActionsEl, recentResponses.slice(0, 8));
    }

    if (updatedEl) {
      const rawUpdated = data.updated_at || 0;
      const updatedMs = rawUpdated > 1000000000000 ? rawUpdated : rawUpdated * 1000;
      const dt = new Date(updatedMs);
      updatedEl.textContent = "Updated " + dt.toLocaleTimeString();
    }

  } catch (err) {
    console.error("Error loading metrics:", err);
    dashboardApiLive = false;
    updateTopStatus(false, {
      mode: currentSystemMode,
      monitoring: false,
      message: "Unable to load metrics.",
      warning: currentSystemMode === "online" ? "Online mode selected, but monitoring status is unavailable." : "",
    });
  }
}

function openRunTestModal() {
  const modal = document.getElementById("run-test-modal");
  const statusEl = document.getElementById("run-test-status");
  const knownMode = document.getElementById("mode-known");

  if (modal) {
    modal.classList.add("open");
    modal.setAttribute("aria-hidden", "false");
  }

  if (knownMode) {
    knownMode.checked = true;
  }

  const singleMode = document.getElementById("attack-selection-single");
  if (singleMode) {
    singleMode.checked = true;
  }
  selectedAttacks = new Set(["BENIGN"]);
  setAttackSelectionMode("single");

  syncRunTestMode();

  if (statusEl) {
    statusEl.textContent = "";
  }

  refreshRunTestStatus();
}

function closeRunTestModal() {
  const modal = document.getElementById("run-test-modal");

  if (modal) {
    modal.classList.remove("open");
    modal.setAttribute("aria-hidden", "true");
  }
}

function openResetModal() {
  const modal = document.getElementById("reset-modal");
  if (!modal) {
    return;
  }

  modal.classList.add("open");
  modal.setAttribute("aria-hidden", "false");
}

function closeResetModal() {
  const modal = document.getElementById("reset-modal");
  if (!modal) {
    return;
  }

  modal.classList.remove("open");
  modal.setAttribute("aria-hidden", "true");
}

async function confirmResetDashboardMetrics() {
  closeResetModal();

  const resetButton = document.getElementById("reset-metrics-button");
  const statusEl = document.getElementById("run-test-status");

  if (resetButton) {
    resetButton.disabled = true;
  }
  if (statusEl) {
    statusEl.textContent = "Resetting metrics...";
  }

  try {
    const res = await fetch('/api/metrics/reset', {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      }
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Failed to reset metrics");
    }

    if (statusEl) {
      statusEl.textContent = data.message || "Metrics reset complete.";
    }
    await loadSummary();
  } catch (err) {
    console.error("Error resetting metrics:", err);
    if (statusEl) {
      statusEl.textContent = err.message || "Unable to reset metrics.";
    }
  } finally {
    if (resetButton) {
      resetButton.disabled = false;
    }
  }
}

async function resetDashboardMetrics() {
  openResetModal();
}

function syncRunTestMode() {
  const selected = document.querySelector("input[name='test-mode']:checked");
  const mode = selected ? selected.value : "known";
  const knownSection = document.getElementById("known-attack-section");
  const unknownSection = document.getElementById("unknown-upload-section");
  const launchButton = document.getElementById("launch-run-test");
  const uploadButton = document.getElementById("analyze-upload-csv");

  const knownMode = mode === "known";

  if (knownSection) {
    knownSection.hidden = !knownMode;
  }
  if (unknownSection) {
    unknownSection.hidden = knownMode;
  }
  if (launchButton) {
    launchButton.hidden = !knownMode;
  }
  if (uploadButton) {
    uploadButton.hidden = knownMode;
  }
}

async function launchRunTest(payload) {
  const button = document.getElementById("run-test-button");
  const statusEl = document.getElementById("run-test-status");
  const consoleEl = document.getElementById("run-test-console");

  if (button) {
    button.disabled = true;
  }
  if (statusEl) {
    statusEl.textContent = "Starting run_test.py...";
  }
  if (consoleEl) {
    consoleEl.innerHTML = "<div class='console-line dim'>Launching test and waiting for output...</div>";
  }

  try {
    const res = await fetch('/api/run-test', {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Unable to start run_test.py");
    }

    if (statusEl) {
      statusEl.textContent = data.message || "run_test.py started.";
    }
    await refreshRunTestStatus();
    startRunTestPolling();
  } catch (err) {
    console.error("Error starting run_test.py:", err);
    if (statusEl) {
      statusEl.textContent = err.message || "Failed to start run_test.py.";
    }
  } finally {
    if (button) {
      button.disabled = false;
    }
  }
}

async function runSelectedAttack() {
  const attacks = Array.from(selectedAttacks);
  if (!attacks.length) {
    const statusEl = document.getElementById("run-test-status");
    if (statusEl) {
      statusEl.textContent = "Please select at least one attack chip.";
    }
    return;
  }

  if (attacks.length === 1 && attacks[0] === "BENIGN") {
    await launchRunTest({ mode: "benign" });
    return;
  }

  if (attacks.length === 1) {
    await launchRunTest({ mode: "attack", attack: attacks[0] });
    return;
  }

  await launchRunTest({ mode: "attack", attacks });
}

async function analyzeUploadedCsv() {
  const fileInput = document.getElementById("upload-csv");
  const statusEl = document.getElementById("run-test-status");
  const consoleEl = document.getElementById("run-test-console");

  if (!fileInput || !fileInput.files || !fileInput.files.length) {
    if (statusEl) {
      statusEl.textContent = "Please choose a CSV file first.";
    }
    return;
  }

  const file = fileInput.files[0];
  if (!file.name.toLowerCase().endsWith(".csv")) {
    if (statusEl) {
      statusEl.textContent = "Only CSV files are supported.";
    }
    return;
  }

  if (statusEl) {
    statusEl.textContent = "Uploading CSV for analysis...";
  }
  if (consoleEl) {
    consoleEl.innerHTML = "<div class='console-line dim'>Uploading file and initializing model analysis...</div>";
  }

  const payload = new FormData();
  payload.append("file", file);

  try {
    const res = await fetch('/api/run-test/upload-csv', {
      method: "POST",
      body: payload
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Unable to analyze uploaded CSV");
    }

    if (statusEl) {
      statusEl.textContent = data.message || "CSV analysis started.";
    }
    await refreshRunTestStatus();
    startRunTestPolling();
  } catch (err) {
    console.error("Error uploading CSV:", err);
    if (statusEl) {
      statusEl.textContent = err.message || "Failed to analyze CSV.";
    }
  }
}

let runTestPollTimer = null;

async function refreshRunTestStatus() {
  const statusEl = document.getElementById("run-test-status");
  const consoleEl = document.getElementById("run-test-console");
  const launchButton = document.getElementById("launch-run-test");
  const uploadButton = document.getElementById("analyze-upload-csv");

  if (!statusEl && !consoleEl) {
    return;
  }

  try {
    const res = await fetch('/api/run-test/status?ts=' + Date.now(), { cache: "no-store" });
    const data = await res.json();
    const output = data.output || [];

    if (output.length) {
      if (consoleEl) {
        consoleEl.innerHTML = output.map((line) => "<div class='console-line'>" + line.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;") + "</div>").join("");
        consoleEl.scrollTop = consoleEl.scrollHeight;
      }
    } else if (!data.running) {
      if (consoleEl) {
        consoleEl.innerHTML = "<div class='console-line dim'>Select an attack or upload a CSV to start analysis.</div>";
      }
    }

    if (data.running) {
      if (statusEl) {
        statusEl.textContent = data.message || "run_test.py is running...";
      }
      if (launchButton) launchButton.disabled = true;
      if (uploadButton) uploadButton.disabled = true;
      startRunTestPolling();
    } else {
      if (statusEl && !statusEl.textContent) {
        statusEl.textContent = data.message || "Ready to run.";
      }
      if (launchButton) launchButton.disabled = false;
      if (uploadButton) uploadButton.disabled = false;
    }

    return data;
  } catch (err) {
    console.error("Error fetching run_test status:", err);
    if (statusEl) {
      statusEl.textContent = "Unable to load run_test status.";
    }
    if (launchButton) launchButton.disabled = false;
    if (uploadButton) uploadButton.disabled = false;
    stopRunTestPolling();
    return {
      running: false,
      status: "error",
      message: "Unable to load run_test status."
    };
  }
}

function startRunTestPolling() {
  if (runTestPollTimer) {
    return;
  }

  runTestPollTimer = setInterval(async () => {
    const data = await refreshRunTestStatus();
    if (data && !data.running) {
      stopRunTestPolling();
    }
  }, 1000);
}

function stopRunTestPolling() {
  if (runTestPollTimer) {
    clearInterval(runTestPollTimer);
    runTestPollTimer = null;
  }
}

// Expose config values to JavaScript
const METRICS_AVAILABLE = """ + ("true" if METRICS_AVAILABLE else "false") + """;
const RUN_TEST_OPTIONS = """ + json.dumps(RUN_TEST_OPTIONS) + """;

function bindEventById(id, eventName, handler) {
  const el = document.getElementById(id);
  if (!el) {
    console.warn("Missing UI element:", id);
    return null;
  }
  el.addEventListener(eventName, handler);
  return el;
}

// Critical modal actions use inline onclick fallback handlers in HTML.
bindEventById("close-run-test", "click", closeRunTestModal);
bindEventById("close-reset-modal", "click", closeResetModal);
bindEventById("close-details-modal", "click", closeDetailsModal);
bindEventById("recent-card", "click", () => openDetailsModal("recent"));
bindEventById("soar-card", "click", () => openDetailsModal("responses"));
bindEventById("categories-card", "click", () => openDetailsModal("categories"));
bindEventById("forensics-card", "click", () => openDetailsModal("forensics"));
bindEventById("view-all-recent", "click", (event) => {
  event.stopPropagation();
  openDetailsModal("recent");
});
bindEventById("view-all-responses", "click", (event) => {
  event.stopPropagation();
  openDetailsModal("responses");
});
bindEventById("view-all-categories", "click", (event) => {
  event.stopPropagation();
  openDetailsModal("categories");
});
bindEventById("view-all-forensics", "click", (event) => {
  event.stopPropagation();
  openDetailsModal("forensics");
});
bindEventById("recent-card", "keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    openDetailsModal("recent");
  }
});
bindEventById("soar-card", "keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    openDetailsModal("responses");
  }
});
bindEventById("categories-card", "keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    openDetailsModal("categories");
  }
});
bindEventById("forensics-card", "keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    openDetailsModal("forensics");
  }
});
bindEventById("mode-known", "change", syncRunTestMode);
bindEventById("mode-unknown", "change", syncRunTestMode);
bindEventById("status-toggle", "click", handleSystemModeToggle);
bindEventById("attack-selection-single", "change", () => setAttackSelectionMode("single"));
bindEventById("attack-selection-multi", "change", () => setAttackSelectionMode("multi"));
bindEventById("run-test-modal", "click", (event) => {
  if (event.target && event.target.id === "run-test-modal") {
    closeRunTestModal();
  }
});
bindEventById("reset-modal", "click", (event) => {
  if (event.target && event.target.id === "reset-modal") {
    closeResetModal();
  }
});
bindEventById("details-modal", "click", (event) => {
  if (event.target && event.target.id === "details-modal") {
    closeDetailsModal();
  }
});
document.querySelectorAll(".details-filter-btn").forEach((button) => {
  button.addEventListener("click", () => {
    setDetailsFilter(button.dataset.filter || "all");
  });
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeRunTestModal();
    closeResetModal();
    closeDetailsModal();
  }
});

refreshRunTestStatus();
syncRunTestMode();
renderAttackChips();
setInterval(loadSummary, 2000);
loadSummary();
    </script>
  </body>
</html>
"""

@app.route('/')
def home():
  response = make_response(render_template_string(HTML_TEMPLATE))
  response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
  response.headers["Pragma"] = "no-cache"
  response.headers["Expires"] = "0"
  return response

@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "dashboard": "unified"})


def _run_test_command(mode):
  script_path = os.path.join(BASE_DIR, "run_test.py")
  command = [sys.executable, script_path]

  if mode == "all-attacks":
    command.append("--all-attacks")
  elif mode == "benign":
    command.append("--benign")

  return command


def _update_run_test_state(**updates):
  with RUN_TEST_STATE_LOCK:
    RUN_TEST_STATE.update(updates)


def _append_run_test_output(line):
  with RUN_TEST_STATE_LOCK:
    output = RUN_TEST_STATE.setdefault("output", [])
    output.append(line)
    RUN_TEST_STATE["output"] = output[-300:]


def _snapshot_system_mode_state():
  global MONITOR_PROCESS

  with SYSTEM_MODE_LOCK:
    is_running = MONITOR_PROCESS is not None and MONITOR_PROCESS.poll() is None
    if not is_running:
      MONITOR_PROCESS = None

    if SYSTEM_MODE_STATE.get("mode") == "online" and not is_running:
      existing_warning = str(SYSTEM_MODE_STATE.get("warning") or "").strip()
      existing_message = str(SYSTEM_MODE_STATE.get("message") or "").strip()
      warning_message = existing_warning or existing_message or "Online mode selected, but external monitor is not running."
      SYSTEM_MODE_STATE["monitoring"] = False
      SYSTEM_MODE_STATE["message"] = warning_message
      if SYSTEM_MODE_STATE.get("warning") != warning_message:
        SYSTEM_MODE_STATE["last_changed_at"] = int(time.time() * 1000)
      SYSTEM_MODE_STATE["warning"] = warning_message
    else:
      SYSTEM_MODE_STATE["monitoring"] = is_running
      if is_running or SYSTEM_MODE_STATE.get("mode") == "offline":
        SYSTEM_MODE_STATE["warning"] = ""

    return dict(SYSTEM_MODE_STATE)


def _set_system_mode_state(mode, monitoring, message, warning="", command="", started_at=None):
  with SYSTEM_MODE_LOCK:
    SYSTEM_MODE_STATE.update({
      "mode": mode,
      "monitoring": monitoring,
      "message": message,
      "warning": warning,
      "command": command,
      "started_at": started_at,
      "last_changed_at": int(time.time() * 1000),
    })


def _tail_file(path, max_lines=8):
  if not os.path.exists(path):
    return ""

  try:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
      lines = handle.readlines()
  except Exception:
    return ""

  tail = [line.strip() for line in lines[-max_lines:] if line.strip()]
  return " | ".join(tail)


def _stop_external_monitor_locked():
  global MONITOR_PROCESS

  process = MONITOR_PROCESS
  MONITOR_PROCESS = None

  if process is None:
    return

  try:
    if process.poll() is None:
      process.terminate()
      try:
        process.wait(timeout=4)
      except Exception:
        process.kill()
        process.wait(timeout=2)
  except Exception:
    pass


def _switch_system_mode(mode):
  global MONITOR_PROCESS

  requested_mode = "online" if str(mode).lower() == "online" else "offline"
  command = [sys.executable, LIVE_MONITOR_SCRIPT_PATH]

  with SYSTEM_MODE_LOCK:
    running = MONITOR_PROCESS is not None and MONITOR_PROCESS.poll() is None

    if requested_mode == "online":
      if running:
        SYSTEM_MODE_STATE.update({
          "mode": "online",
          "monitoring": True,
          "message": "Online monitoring is active.",
          "warning": "",
          "command": " ".join(command),
          "last_changed_at": int(time.time() * 1000),
        })
        return dict(SYSTEM_MODE_STATE)

      if not os.path.exists(LIVE_MONITOR_SCRIPT_PATH):
        raise FileNotFoundError(f"Missing monitor script: {LIVE_MONITOR_SCRIPT_PATH}")

      os.makedirs(os.path.dirname(LIVE_MONITOR_LOG_PATH), exist_ok=True)
      with open(LIVE_MONITOR_LOG_PATH, "a", encoding="utf-8", errors="replace") as monitor_log:
        monitor_log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting live monitor\n")
        process = subprocess.Popen(
          command,
          cwd=BASE_DIR,
          stdout=monitor_log,
          stderr=subprocess.STDOUT,
          creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
      time.sleep(1.0)
      returncode = process.poll()
      if returncode is not None:
        MONITOR_PROCESS = None
        log_tail = _tail_file(LIVE_MONITOR_LOG_PATH)
        details = f" Last log lines: {log_tail}" if log_tail else ""
        raise RuntimeError(
          f"External monitor exited immediately with code {returncode}."
          f" Check {LIVE_MONITOR_LOG_PATH}.{details}"
        )

      MONITOR_PROCESS = process

      SYSTEM_MODE_STATE.update({
        "mode": "online",
        "monitoring": True,
        "message": "Online monitoring started.",
        "warning": "",
        "command": " ".join(command),
        "started_at": int(time.time() * 1000),
        "last_changed_at": int(time.time() * 1000),
      })
      if METRICS_AVAILABLE:
        try:
          set_active_mode("online")
        except Exception:
          pass
      return dict(SYSTEM_MODE_STATE)

    _stop_external_monitor_locked()
    SYSTEM_MODE_STATE.update({
      "mode": "offline",
      "monitoring": False,
      "message": "Offline mode active.",
      "warning": "",
      "command": "",
      "started_at": None,
      "last_changed_at": int(time.time() * 1000),
    })
    if METRICS_AVAILABLE:
      try:
        set_active_mode("offline")
      except Exception:
        pass
    return dict(SYSTEM_MODE_STATE)


def _run_test_worker(command):
  global RUN_TEST_PROCESS
  metrics_mode = _snapshot_system_mode_state().get("mode", "offline")

  _update_run_test_state(
    running=True,
    status="running",
    message="run_test.py is running in a visible console below.",
    command=" ".join(command),
    output=[],
    returncode=None,
    started_at=int(time.time() * 1000),
    finished_at=None,
  )

  try:
    env = os.environ.copy()
    env["NIDS_METRICS_MODE"] = metrics_mode
    process = subprocess.Popen(
      command,
      cwd=BASE_DIR,
      env=env,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      bufsize=1,
      creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    RUN_TEST_PROCESS = process

    assert process.stdout is not None
    for raw_line in iter(process.stdout.readline, ""):
      if raw_line == "" and process.poll() is not None:
        break
      line = raw_line.rstrip("\r\n")
      if line:
        _append_run_test_output(line)

    return_code = process.wait()
    _append_run_test_output(f"[run_test.py exited with code {return_code}]")
    _update_run_test_state(
      running=False,
      status="finished" if return_code == 0 else "failed",
      message="run_test.py finished." if return_code == 0 else "run_test.py failed.",
      returncode=return_code,
      finished_at=int(time.time() * 1000),
    )
  except Exception as exc:
    _append_run_test_output(f"[error starting run_test.py: {exc}]")
    _update_run_test_state(
      running=False,
      status="failed",
      message=str(exc),
      returncode=-1,
      finished_at=int(time.time() * 1000),
    )
  finally:
    RUN_TEST_PROCESS = None


def _load_recent_response_actions(limit=20):
  if not os.path.exists(FORENSICS_RESPONSE_ACTIONS_PATH):
    return []

  records = []
  try:
    with open(FORENSICS_RESPONSE_ACTIONS_PATH, "r", encoding="utf-8") as f:
      for line in f:
        text = line.strip()
        if not text:
          continue
        try:
          payload = json.loads(text)
        except Exception:
          continue
        records.append(payload)
  except Exception:
    return []

  records.sort(key=lambda item: int(item.get("ts", 0)), reverse=True)
  return records[:limit]


def _load_forensics_snapshot(limit=8):
  sources = [
    ("Response Actions", FORENSICS_RESPONSE_ACTIONS_PATH),
    ("System Events", FORENSICS_SYSTEM_EVENTS_PATH),
    ("Attack Metadata", FORENSICS_ATTACK_METADATA_PATH),
    ("Analyst Alerts", FORENSICS_ALERT_ANALYSTS_PATH),
    ("SIEM Alerts", FORENSICS_ALERT_SIEM_PATH),
    ("IRT Alerts", FORENSICS_ALERT_IRT_PATH),
  ]

  summary = []
  recent_entries = []

  for label, path in sources:
    if not os.path.exists(path):
      summary.append({"label": label, "value": 0})
      continue

    count = 0
    latest = None

    try:
      with open(path, "r", encoding="utf-8") as f:
        for line in f:
          text = line.strip()
          if not text:
            continue
          count += 1
          try:
            payload = json.loads(text)
          except Exception:
            continue
          latest = payload
    except Exception:
      summary.append({"label": label, "value": 0})
      continue

    summary.append({"label": label, "value": count})
    if latest:
      recent_entries.append({
        "title": label,
        "detail": latest.get("decision") or latest.get("event") or latest.get("status") or "recorded",
        "source": latest.get("component") or latest.get("channel") or latest.get("label") or "artifact",
        "ts": latest.get("ts", 0),
      })

  recent_entries.sort(key=lambda item: int(item.get("ts", 0)), reverse=True)
  return {
    "summary": summary,
    "recent": recent_entries[:limit],
  }


def _load_upload_inference_artifacts():
  with _UPLOAD_MODEL_LOCK:
    if _UPLOAD_MODEL_CACHE["model"] is not None:
      return (
        _UPLOAD_MODEL_CACHE["model"],
        _UPLOAD_MODEL_CACHE["scaler"],
        _UPLOAD_MODEL_CACHE["encoder"],
      )

    import joblib
    import tensorflow as tf

    if not os.path.exists(PRIMARY_MODEL_PATH):
      raise FileNotFoundError(f"Missing model file: {PRIMARY_MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
      raise FileNotFoundError(f"Missing scaler file: {SCALER_PATH}")
    if not os.path.exists(ENCODER_PATH):
      raise FileNotFoundError(f"Missing label encoder file: {ENCODER_PATH}")

    _UPLOAD_MODEL_CACHE["scaler"] = joblib.load(SCALER_PATH)
    _UPLOAD_MODEL_CACHE["encoder"] = joblib.load(ENCODER_PATH)
    _UPLOAD_MODEL_CACHE["model"] = tf.keras.models.load_model(PRIMARY_MODEL_PATH)
    return (
      _UPLOAD_MODEL_CACHE["model"],
      _UPLOAD_MODEL_CACHE["scaler"],
      _UPLOAD_MODEL_CACHE["encoder"],
    )


def _run_upload_csv_worker(file_path, original_name):
  metrics_mode = _snapshot_system_mode_state().get("mode", "offline")

  _update_run_test_state(
    running=True,
    status="running",
    message="CSV analysis is running in the console below.",
    command=f"analyze-csv {original_name}",
    output=[],
    returncode=None,
    started_at=int(time.time() * 1000),
    finished_at=None,
  )

  try:
    import numpy as np
    import pandas as pd

    _append_run_test_output("=" * 60)
    _append_run_test_output("CSV UPLOAD ANALYSIS")
    _append_run_test_output("=" * 60)
    _append_run_test_output(f"File: {original_name}")

    model, scaler, encoder = _load_upload_inference_artifacts()
    _append_run_test_output("[OK] Loaded CNN-LSTM model, scaler, and label encoder")

    frame = pd.read_csv(file_path)
    if frame.empty:
      raise ValueError("Uploaded CSV is empty.")

    normalized_columns = {str(col).strip().lower(): col for col in frame.columns}
    selected_columns = []
    missing = []
    for feature_name in RUN_TEST_FEATURE_NAMES:
      key = feature_name.strip().lower()
      if key in normalized_columns:
        selected_columns.append(normalized_columns[key])
      else:
        missing.append(feature_name)

    if missing:
      missing_text = ", ".join(missing)
      raise ValueError(f"Missing required feature columns: {missing_text}")

    model_input = frame[selected_columns].apply(pd.to_numeric, errors="coerce")
    model_input = model_input.replace([np.inf, -np.inf], np.nan)
    finite_limit = np.finfo(np.float64).max / 10.0
    model_input = model_input.clip(lower=-finite_limit, upper=finite_limit)
    model_input = model_input.fillna(0.0)
    scaled = scaler.transform(model_input.values)

    probabilities = model.predict(scaled, verbose=0)
    if probabilities.ndim == 1:
      probabilities = np.expand_dims(probabilities, axis=0)

    predicted_idx = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    labels = [str(encoder.classes_[idx]) for idx in predicted_idx]

    counts = Counter(labels)
    total = len(labels)
    benign_count = sum(1 for label in labels if label.strip().lower() == "benign")
    threat_count = total - benign_count
    avg_conf = float(np.mean(confidences)) if total else 0.0

    if METRICS_AVAILABLE:
      for label, confidence in zip(labels, confidences):
        is_threat = label.strip().lower() != "benign"
        add_prediction(label, float(confidence), is_threat, "CNN-LSTM", {"mode": metrics_mode})
      _append_run_test_output(f"[OK] Dashboard metrics updated with {total} CSV predictions")

    _append_run_test_output(f"Rows processed: {total}")
    _append_run_test_output(f"Threat rows: {threat_count}")
    _append_run_test_output(f"Benign rows: {benign_count}")
    _append_run_test_output(f"Average confidence: {avg_conf * 100:.1f}%")
    _append_run_test_output("-" * 60)
    _append_run_test_output("Predicted attack distribution:")

    for label, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
      pct = (count / total) * 100 if total else 0
      _append_run_test_output(f"  {label:<24} {count:>6} rows ({pct:>5.1f}%)")

    _append_run_test_output("-" * 60)
    _append_run_test_output("Top sample predictions:")
    for idx in range(min(8, total)):
      _append_run_test_output(f"  Row {idx + 1}: {labels[idx]} ({confidences[idx] * 100:.1f}%)")

    _append_run_test_output("=" * 60)
    _append_run_test_output("CSV ANALYSIS COMPLETE")
    _append_run_test_output("=" * 60)
    _append_run_test_output("[CSV analysis exited with code 0]")

    _update_run_test_state(
      running=False,
      status="finished",
      message=f"CSV analysis finished for {original_name}.",
      returncode=0,
      finished_at=int(time.time() * 1000),
    )
  except Exception as exc:
    _append_run_test_output(f"[CSV analysis error: {exc}]")
    _append_run_test_output("[CSV analysis exited with code 1]")
    _update_run_test_state(
      running=False,
      status="failed",
      message=str(exc),
      returncode=1,
      finished_at=int(time.time() * 1000),
    )
  finally:
    try:
      if os.path.exists(file_path):
        os.remove(file_path)
    except Exception:
      pass


@app.route('/api/run-test', methods=['POST'])
def run_test():
  global RUN_TEST_PROCESS

  payload = request.get_json(silent=True) or {}
  mode = payload.get("mode", "all-attacks")
  attack = payload.get("attack")
  attacks = payload.get("attacks")
  valid_attacks = {item["value"] for item in RUN_TEST_OPTIONS}

  with RUN_TEST_LOCK:
    if (RUN_TEST_PROCESS and RUN_TEST_PROCESS.poll() is None) or RUN_TEST_STATE.get("running"):
      return jsonify({
        "status": "running",
        "message": "run_test.py is already running in the console below."
      }), 200

    try:
      command = _run_test_command(mode)
      if mode == "attack":
        selected_attacks = []
        if isinstance(attacks, list):
          selected_attacks = [str(value) for value in attacks if str(value).strip()]
        elif attack:
          selected_attacks = [str(attack)]

        if not selected_attacks:
          return jsonify({"status": "error", "error": "Missing attack selection."}), 400

        invalid = [value for value in selected_attacks if value not in valid_attacks]
        if invalid:
          return jsonify({"status": "error", "error": f"Unknown attack(s): {', '.join(invalid)}"}), 400

        if "BENIGN" in selected_attacks and len(selected_attacks) > 1:
          return jsonify({"status": "error", "error": "Benign cannot be combined with other attacks."}), 400

        if selected_attacks == ["BENIGN"]:
          mode = "benign"
          command = _run_test_command(mode)
        else:
          for value in selected_attacks:
            command.extend(["--attack", value])
      elif mode not in {"all-attacks", "benign"}:
        return jsonify({"status": "error", "error": f"Unknown mode: {mode}"}), 400

      _update_run_test_state(
        running=True,
        status="starting",
        message="Starting run_test.py in the console below...",
        command=" ".join(command),
        output=[],
        returncode=None,
        started_at=int(time.time() * 1000),
        finished_at=None,
      )

      worker = Thread(target=_run_test_worker, args=(command,), daemon=True)
      worker.start()
    except Exception as exc:
      RUN_TEST_PROCESS = None
      return jsonify({"status": "error", "error": str(exc)}), 500

  if mode == "attack":
    selected_message = []
    if isinstance(attacks, list) and attacks:
      selected_message = [str(value) for value in attacks if str(value).strip()]
    elif attack:
      selected_message = [str(attack)]
    message = f"Started run_test.py for attack selection: {', '.join(selected_message)}."
  elif mode == "benign":
    message = "Started run_test.py --benign."
  else:
    message = "Started run_test.py --all-attacks."

  return jsonify({
    "status": "started",
    "message": message
  }), 202


@app.route('/api/run-test/status')
def run_test_status():
    with RUN_TEST_STATE_LOCK:
        state = dict(RUN_TEST_STATE)
        state["output"] = list(RUN_TEST_STATE.get("output", []))

    return jsonify(state)


@app.route('/api/run-test/upload-csv', methods=['POST'])
def run_test_upload_csv():
  global RUN_TEST_PROCESS

  with RUN_TEST_LOCK:
    if (RUN_TEST_PROCESS and RUN_TEST_PROCESS.poll() is None) or RUN_TEST_STATE.get("running"):
      return jsonify({
        "status": "running",
        "message": "Another analysis is already running in the console below."
      }), 200

    upload_file = request.files.get("file")
    if upload_file is None or not upload_file.filename:
      return jsonify({"status": "error", "error": "No CSV file uploaded."}), 400

    if not upload_file.filename.lower().endswith(".csv"):
      return jsonify({"status": "error", "error": "Only .csv files are supported."}), 400

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = os.path.basename(upload_file.filename)
    timestamp = int(time.time() * 1000)
    saved_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{safe_name}")
    upload_file.save(saved_path)

    _update_run_test_state(
      running=True,
      status="starting",
      message="Starting CSV analysis in the console below...",
      command=f"analyze-csv {safe_name}",
      output=[],
      returncode=None,
      started_at=int(time.time() * 1000),
      finished_at=None,
    )

    worker = Thread(target=_run_upload_csv_worker, args=(saved_path, safe_name), daemon=True)
    worker.start()

  return jsonify({
    "status": "started",
    "message": f"Started CSV analysis for {safe_name}."
  }), 202


@app.route('/api/system-mode', methods=['GET'])
def get_system_mode():
  return jsonify({
    "status": "ok",
    "system_mode": _snapshot_system_mode_state(),
  })


@app.route('/api/system-mode', methods=['POST'])
def set_system_mode():
  payload = request.get_json(silent=True) or {}
  mode = payload.get("mode", "offline")

  if mode not in {"offline", "online"}:
    return jsonify({"status": "error", "error": "Mode must be either 'offline' or 'online'."}), 400

  try:
    state = _switch_system_mode(mode)
    return jsonify({
      "status": "ok",
      "message": state.get("message", "System mode updated."),
      "system_mode": state,
    }), 200
  except Exception as exc:
    _set_system_mode_state(
      "offline",
      False,
      f"Unable to start online monitoring: {exc}",
    )
    return jsonify({
      "status": "error",
      "error": str(exc),
      "system_mode": _snapshot_system_mode_state(),
    }), 500

@app.route('/api/metrics')
def metrics():
  try:
    recent_responses = _load_recent_response_actions(limit=1000)
    forensics_snapshot = _load_forensics_snapshot(limit=8)
    mode_state = _snapshot_system_mode_state()
    active_mode = mode_state.get('mode', 'offline')
    if METRICS_AVAILABLE:
      metrics_data = get_metrics(active_mode)

      # Calculate average confidence from recent predictions
      recent = metrics_data.get('recent_predictions', [])
      avg_confidence = 0
      if recent:
        avg_confidence = sum(p.get('confidence', 0) for p in recent) / len(recent)

      return jsonify({
        "metrics": {
          "total_predictions": metrics_data.get('total_predictions', 0),
          "threats_detected": metrics_data.get('threats_detected', 0),
          "benign_count": metrics_data.get('benign_count', 0),
          "threat_types": metrics_data.get('threat_types', {}),
          "recent_predictions": recent,
          "recent_response_actions": recent_responses,
          "forensics_summary": forensics_snapshot.get('summary', []),
          "recent_forensics": forensics_snapshot.get('recent', []),
          "system_mode": mode_state,
          "average_confidence": avg_confidence,
          "last_update": metrics_data.get('last_update', '')
        },
        "updated_at": int(time.time() * 1000)
      })

    # Fallback dummy data
    return jsonify({
      "metrics": {
        "total_predictions": 0,
        "threats_detected": 0,
        "benign_count": 0,
        "threat_types": {},
        "recent_predictions": [],
        "recent_response_actions": recent_responses,
        "forensics_summary": forensics_snapshot.get('summary', []),
        "recent_forensics": forensics_snapshot.get('recent', []),
        "system_mode": mode_state,
        "average_confidence": 0,
        "last_update": ""
      },
      "updated_at": int(time.time() * 1000)
    })
  except Exception as e:
    print(f"Error getting metrics: {e}")
    return jsonify({
      "error": str(e),
      "metrics": {
        "total_predictions": 0,
        "threats_detected": 0,
        "benign_count": 0,
        "threat_types": {},
        "recent_predictions": [],
        "recent_response_actions": _load_recent_response_actions(limit=20),
        "forensics_summary": _load_forensics_snapshot(limit=8).get('summary', []),
        "recent_forensics": _load_forensics_snapshot(limit=8).get('recent', []),
        "system_mode": _snapshot_system_mode_state(),
      }
    }), 500


@app.route('/api/metrics/reset', methods=['POST'])
def reset_dashboard_metrics():
  if not METRICS_AVAILABLE:
    return jsonify({"status": "error", "error": "Shared metrics backend is unavailable."}), 503

  try:
    current_mode = _snapshot_system_mode_state().get("mode", "offline")
    reset_metrics(current_mode)
    forensics_logs = [
      FORENSICS_RESPONSE_ACTIONS_PATH,
      FORENSICS_SYSTEM_EVENTS_PATH,
      FORENSICS_ATTACK_METADATA_PATH,
      FORENSICS_ALERT_ANALYSTS_PATH,
      FORENSICS_ALERT_SIEM_PATH,
      FORENSICS_ALERT_IRT_PATH,
    ]
    for path in forensics_logs:
      try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8'):
          pass
      except Exception:
        pass

    return jsonify({
      "status": "ok",
      "message": f"Dashboard and SOAR metrics reset for {current_mode} mode."
    }), 200
  except Exception as e:
    return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == "__main__":
    print("=" * 60)
    print("AI-NIDS Unified Dashboard")
    print("=" * 60)
    print(f"Dashboard URL: http://localhost:8080")
    print("Displays metrics from shared_metrics.json")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)

