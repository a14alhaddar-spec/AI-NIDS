import json
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime

try:
	import yaml
except Exception:
	yaml = None

from .response_executor import execute_playbook


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FORENSICS_DIR = os.path.join(BASE_DIR, "data", "forensics")
DEFAULT_POLICY_PATH = os.path.join(BASE_DIR, "configs", "soar_policy.yml")
DEFAULT_PLAYBOOK_DIR = os.path.join(os.path.dirname(__file__), "playbooks")

_IO_LOCK = threading.RLock()


def _now_iso():
	return datetime.utcnow().isoformat() + "Z"


def _safe_load_structured(path):
	if not os.path.exists(path):
		return {}

	with open(path, "r", encoding="utf-8") as f:
		raw = f.read().strip()

	if not raw:
		return {}

	if path.lower().endswith(".json"):
		return json.loads(raw)

	if yaml is not None:
		return yaml.safe_load(raw) or {}

	return json.loads(raw)


def _append_jsonl(path, payload):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with _IO_LOCK:
		with open(path, "a", encoding="utf-8") as f:
			f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _resolve_path(path_value):
	if not path_value:
		return path_value
	if os.path.isabs(path_value):
		return path_value
	return os.path.join(BASE_DIR, path_value)


class SoarController:
	def __init__(self, policy_path=None, playbook_dir=None):
		self.policy_path = policy_path or DEFAULT_POLICY_PATH
		self.playbook_dir = playbook_dir or DEFAULT_PLAYBOOK_DIR
		self.policy = self._load_policy()

	def _load_policy(self):
		defaults = {
			"thresholds": {
				"alert": 0.55,
				"auto_response": 0.80,
			},
			"escalation": {
				"security_analysts": True,
				"siem": True,
				"incident_response_team": True,
			},
			"playbooks": {
				"default": "terminate_connection.yaml",
				"mappings": {
					"DDoS": "block_ip.yaml",
					"DoS Hulk": "block_ip.yaml",
					"DoS GoldenEye": "block_ip.yaml",
					"DoS Slowhttptest": "block_ip.yaml",
					"DoS slowloris": "block_ip.yaml",
					"PortScan": "terminate_connection.yaml",
					"Bot": "isolate_host.yaml",
					"Infiltration": "isolate_host.yaml",
					"Web Attack - Brute Force": "block_ip.yaml",
					"Web Attack - Sql Injection": "terminate_connection.yaml",
					"Web Attack - XSS": "terminate_connection.yaml",
					"Heartbleed": "terminate_connection.yaml",
					"FTP-Patator": "block_ip.yaml",
					"SSH-Patator": "block_ip.yaml",
				},
			},
			"alerts": {
				"analyst_log": os.path.join(FORENSICS_DIR, "alerts_security_analysts.jsonl"),
				"siem_log": os.path.join(FORENSICS_DIR, "alerts_siem.jsonl"),
				"irt_log": os.path.join(FORENSICS_DIR, "alerts_incident_response_team.jsonl"),
				"siem_webhook": "",
				"analyst_webhook": "",
				"irt_webhook": "",
			},
			"forensics": {
				"attack_metadata": os.path.join(FORENSICS_DIR, "attack_metadata.jsonl"),
				"system_events": os.path.join(FORENSICS_DIR, "system_events.jsonl"),
				"response_actions": os.path.join(FORENSICS_DIR, "response_actions.jsonl"),
			},
		}

		try:
			loaded = _safe_load_structured(self.policy_path)
			if not loaded:
				return defaults

			merged = defaults
			for k, v in loaded.items():
				if isinstance(v, dict) and isinstance(merged.get(k), dict):
					merged[k].update(v)
				else:
					merged[k] = v
			self._resolve_policy_paths(merged)
			return merged
		except Exception:
			return defaults

	def _resolve_policy_paths(self, policy):
		alerts = policy.get("alerts", {})
		for key in ("analyst_log", "siem_log", "irt_log"):
			if key in alerts:
				alerts[key] = _resolve_path(alerts[key])

		forensics = policy.get("forensics", {})
		for key in ("attack_metadata", "system_events", "response_actions"):
			if key in forensics:
				forensics[key] = _resolve_path(forensics[key])

	def _playbook_for_label(self, label):
		mappings = self.policy.get("playbooks", {}).get("mappings", {})
		default_pb = self.policy.get("playbooks", {}).get("default", "terminate_connection.yaml")
		candidate = mappings.get(label, default_pb)
		return os.path.join(self.playbook_dir, candidate)

	def _log_forensics(self, attack_meta, system_event, action_result):
		forensics_cfg = self.policy.get("forensics", {})
		_append_jsonl(forensics_cfg["attack_metadata"], attack_meta)
		_append_jsonl(forensics_cfg["system_events"], system_event)
		_append_jsonl(forensics_cfg["response_actions"], action_result)

	def _send_webhook(self, url, payload):
		if not url:
			return {"status": "skipped", "reason": "no_webhook_configured"}

		data = json.dumps(payload).encode("utf-8")
		req = urllib.request.Request(
			url,
			data=data,
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		try:
			with urllib.request.urlopen(req, timeout=3) as resp:
				return {"status": "ok", "code": getattr(resp, "status", 200)}
		except urllib.error.URLError as exc:
			return {"status": "error", "error": str(exc)}

	def _escalate_alerts(self, alert_event):
		esc = self.policy.get("escalation", {})
		alert_cfg = self.policy.get("alerts", {})

		deliveries = {}

		if esc.get("security_analysts", True):
			_append_jsonl(alert_cfg["analyst_log"], alert_event)
			deliveries["security_analysts"] = self._send_webhook(
				alert_cfg.get("analyst_webhook", ""),
				{"channel": "security_analysts", "alert": alert_event},
			)

		if esc.get("siem", True):
			_append_jsonl(alert_cfg["siem_log"], alert_event)
			deliveries["siem"] = self._send_webhook(
				alert_cfg.get("siem_webhook", ""),
				{"channel": "siem", "alert": alert_event},
			)

		if esc.get("incident_response_team", True):
			_append_jsonl(alert_cfg["irt_log"], alert_event)
			deliveries["incident_response_team"] = self._send_webhook(
				alert_cfg.get("irt_webhook", ""),
				{"channel": "incident_response_team", "alert": alert_event},
			)

		return deliveries

	def process_detection(self, detection):
		label = str(detection.get("label", "unknown"))
		confidence = float(detection.get("confidence", 0.0) or 0.0)
		threat_detected = bool(detection.get("threat_detected", False))
		now_ms = int(time.time() * 1000)

		thresholds = self.policy.get("thresholds", {})
		alert_threshold = float(thresholds.get("alert", 0.55))
		auto_threshold = float(thresholds.get("auto_response", 0.80))

		attack_meta = {
			"ts": now_ms,
			"ts_iso": _now_iso(),
			"label": label,
			"confidence": confidence,
			"model": detection.get("model", "unknown"),
			"source_ip": detection.get("source_ip"),
			"destination_ip": detection.get("destination_ip"),
			"source_port": detection.get("source_port"),
			"destination_port": detection.get("destination_port"),
			"protocol": detection.get("protocol"),
			"host_id": detection.get("host_id"),
			"process_name": detection.get("process_name"),
			"pid": detection.get("pid"),
			"threat_detected": threat_detected,
		}

		action_result = {
			"ts": now_ms,
			"ts_iso": _now_iso(),
			"decision": "none",
			"playbook": None,
			"actions": [],
			"label": label,
			"confidence": confidence,
		}

		escalation = {}
		triggered = False

		if threat_detected and confidence >= alert_threshold:
			alert_event = {
				"ts": now_ms,
				"label": label,
				"confidence": confidence,
				"severity": "high" if confidence >= auto_threshold else "medium",
				"metadata": attack_meta,
			}
			escalation = self._escalate_alerts(alert_event)
			triggered = True

		if threat_detected and confidence >= auto_threshold:
			playbook_path = self._playbook_for_label(label)
			execution = execute_playbook(playbook_path, attack_meta)
			action_result["decision"] = "auto_response"
			action_result["playbook"] = playbook_path
			action_result["actions"] = execution.get("actions", [])
			action_result["status"] = execution.get("status", "ok")
			triggered = True
		elif threat_detected and confidence >= alert_threshold:
			action_result["decision"] = "escalate_only"
			action_result["status"] = "ok"
		elif threat_detected:
			action_result["decision"] = "monitor_only"
			action_result["status"] = "ok"
		else:
			action_result["decision"] = "benign"
			action_result["status"] = "ok"

		system_event = {
			"ts": now_ms,
			"ts_iso": _now_iso(),
			"component": "soar_controller",
			"event": "detection_processed",
			"label": label,
			"confidence": confidence,
			"threat_detected": threat_detected,
			"triggered": triggered,
			"decision": action_result["decision"],
		}

		self._log_forensics(attack_meta, system_event, action_result)

		return {
			"triggered": triggered,
			"decision": action_result["decision"],
			"escalation": escalation,
			"action_result": action_result,
		}


_SOAR = SoarController()


def process_detection_event(label, confidence, threat_detected, metadata=None):
	metadata = metadata or {}
	payload = {
		"label": label,
		"confidence": confidence,
		"threat_detected": threat_detected,
		"model": metadata.get("model", "unknown"),
		"source_ip": metadata.get("source_ip"),
		"destination_ip": metadata.get("destination_ip"),
		"source_port": metadata.get("source_port"),
		"destination_port": metadata.get("destination_port"),
		"protocol": metadata.get("protocol"),
		"host_id": metadata.get("host_id"),
		"process_name": metadata.get("process_name"),
		"pid": metadata.get("pid"),
	}
	return _SOAR.process_detection(payload)
