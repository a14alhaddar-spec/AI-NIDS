import json
import os

try:
    import yaml
except Exception:
    yaml = None

from .firewall_controller import (
    block_malicious_ip,
    block_suspicious_port,
    disable_network_interface,
    drop_malicious_connection,
    isolate_infected_machine,
    kill_malicious_process,
    modify_firewall_rules,
    restrict_protocol,
)


def _load_playbook(playbook_path):
    if not os.path.exists(playbook_path):
        raise FileNotFoundError(f"Playbook not found: {playbook_path}")

    with open(playbook_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return {"name": os.path.basename(playbook_path), "steps": []}

    if playbook_path.lower().endswith(".json"):
        return json.loads(raw)

    if yaml is not None:
        return yaml.safe_load(raw) or {}

    return json.loads(raw)


def _action_handlers():
    handlers = {
        "block_malicious_ip": lambda step, ctx: block_malicious_ip(
            ip=step.get("ip") or ctx.get("source_ip"),
            direction=step.get("direction", "ingress"),
        ),
        "modify_firewall_rules": lambda step, ctx: modify_firewall_rules(
            rule=step.get("rule", "deny all from suspicious source"),
            context=ctx,
        ),
        "drop_malicious_connection": lambda step, ctx: drop_malicious_connection(
            source_ip=step.get("source_ip") or ctx.get("source_ip"),
            destination_ip=step.get("destination_ip") or ctx.get("destination_ip"),
            source_port=step.get("source_port") or ctx.get("source_port"),
            destination_port=step.get("destination_port") or ctx.get("destination_port"),
            protocol=step.get("protocol") or ctx.get("protocol"),
        ),
        "isolate_infected_machine": lambda step, ctx: isolate_infected_machine(
            host_id=step.get("host_id") or ctx.get("host_id") or ctx.get("source_ip")
        ),
        "disable_network_interface": lambda step, ctx: disable_network_interface(
            interface_name=step.get("interface_name", "eth0"),
            host_id=step.get("host_id") or ctx.get("host_id"),
        ),
        "kill_malicious_process": lambda step, ctx: kill_malicious_process(
            pid=step.get("pid") or ctx.get("pid"),
            process_name=step.get("process_name") or ctx.get("process_name"),
            host_id=step.get("host_id") or ctx.get("host_id"),
        ),
        "block_suspicious_port": lambda step, ctx: block_suspicious_port(
            port=step.get("port") or ctx.get("destination_port"),
            protocol=step.get("protocol") or ctx.get("protocol") or "tcp",
        ),
        "restrict_protocol": lambda step, ctx: restrict_protocol(
            protocol=step.get("protocol") or ctx.get("protocol") or "tcp",
            policy=step.get("policy", "deny"),
        ),
    }

    handlers["block_ip"] = handlers["block_malicious_ip"]
    handlers["isolate_host"] = handlers["isolate_infected_machine"]
    handlers["terminate_connection"] = handlers["drop_malicious_connection"]
    return handlers


def execute_playbook(playbook_path, context):
    playbook = _load_playbook(playbook_path)
    handlers = _action_handlers()
    results = []

    for step in playbook.get("steps", []):
        action = step.get("action")
        if not action:
            continue

        handler = handlers.get(action)
        if handler is None:
            results.append({
                "action": action,
                "status": "unsupported",
                "error": f"Unsupported action: {action}",
            })
            continue

        try:
            result = handler(step, context)
            if not isinstance(result, dict):
                result = {"status": "ok", "result": result}
            result["action"] = action
            results.append(result)
        except Exception as exc:
            results.append({"action": action, "status": "error", "error": str(exc)})

    status = "ok"
    if any(item.get("status") in ("error", "unsupported") for item in results):
        status = "partial"

    return {
        "status": status,
        "playbook": playbook.get("name", os.path.basename(playbook_path)),
        "actions": results,
    }
