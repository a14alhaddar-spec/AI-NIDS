import os
import platform
import subprocess


EXECUTE_ACTIONS = os.getenv("SOAR_EXECUTE_ACTIONS", "false").lower() == "true"


def _run_command(cmd):
    if not EXECUTE_ACTIONS:
        return {
            "status": "simulated",
            "command": cmd,
            "message": "Action simulated (set SOAR_EXECUTE_ACTIONS=true to execute).",
        }

    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
    return {
        "status": "ok" if completed.returncode == 0 else "error",
        "command": cmd,
        "stdout": (completed.stdout or "").strip(),
        "stderr": (completed.stderr or "").strip(),
        "returncode": completed.returncode,
    }


def _os_name():
    return platform.system().lower()


def block_malicious_ip(ip, direction="ingress"):
    if not ip:
        return {"status": "skipped", "reason": "missing_ip"}

    if _os_name().startswith("win"):
        cmd = [
            "netsh",
            "advfirewall",
            "firewall",
            "add",
            "rule",
            f"name=AI-NIDS-Block-{ip}",
            "dir=in" if direction == "ingress" else "dir=out",
            "action=block",
            f"remoteip={ip}",
        ]
    else:
        cmd = ["iptables", "-A", "INPUT", "-s", str(ip), "-j", "DROP"]

    result = _run_command(cmd)
    result["target_ip"] = ip
    return result


def modify_firewall_rules(rule, context=None):
    context = context or {}
    if _os_name().startswith("win"):
        rule_name = str(rule).replace(" ", "-")[:50]
        cmd = [
            "netsh",
            "advfirewall",
            "firewall",
            "add",
            "rule",
            f"name=AI-NIDS-{rule_name}",
            "dir=in",
            "action=block",
            "protocol=any",
        ]
    else:
        cmd = ["iptables", "-A", "INPUT", "-j", "DROP"]

    result = _run_command(cmd)
    result["rule"] = rule
    result["context"] = context
    return result


def drop_malicious_connection(source_ip, destination_ip, source_port=None, destination_port=None, protocol=None):
    if _os_name().startswith("win"):
        cmd = [
            "netsh",
            "advfirewall",
            "firewall",
            "add",
            "rule",
            "name=AI-NIDS-Drop-Connection",
            "dir=in",
            "action=block",
            f"remoteip={source_ip or 'any'}",
        ]
    else:
        cmd = ["iptables", "-A", "INPUT", "-s", str(source_ip or "0.0.0.0/0"), "-j", "DROP"]

    result = _run_command(cmd)
    result.update(
        {
            "source_ip": source_ip,
            "destination_ip": destination_ip,
            "source_port": source_port,
            "destination_port": destination_port,
            "protocol": protocol,
        }
    )
    return result


def isolate_infected_machine(host_id):
    if not host_id:
        return {"status": "skipped", "reason": "missing_host_id"}

    return {
        "status": "simulated" if not EXECUTE_ACTIONS else "ok",
        "host_id": host_id,
        "message": "Host isolation requested",
    }


def disable_network_interface(interface_name, host_id=None):
    if _os_name().startswith("win"):
        cmd = ["netsh", "interface", "set", "interface", interface_name, "disable"]
    else:
        cmd = ["ip", "link", "set", interface_name, "down"]

    result = _run_command(cmd)
    result["interface_name"] = interface_name
    result["host_id"] = host_id
    return result


def kill_malicious_process(pid=None, process_name=None, host_id=None):
    if pid is None and not process_name:
        return {"status": "skipped", "reason": "missing_pid_or_process_name"}

    if _os_name().startswith("win"):
        if pid is not None:
            cmd = ["taskkill", "/PID", str(pid), "/F"]
        else:
            cmd = ["taskkill", "/IM", str(process_name), "/F"]
    else:
        if pid is not None:
            cmd = ["kill", "-9", str(pid)]
        else:
            cmd = ["pkill", "-f", str(process_name)]

    result = _run_command(cmd)
    result["pid"] = pid
    result["process_name"] = process_name
    result["host_id"] = host_id
    return result


def block_suspicious_port(port, protocol="tcp"):
    if port is None:
        return {"status": "skipped", "reason": "missing_port"}

    if _os_name().startswith("win"):
        cmd = [
            "netsh",
            "advfirewall",
            "firewall",
            "add",
            "rule",
            f"name=AI-NIDS-Block-Port-{port}",
            "dir=in",
            "action=block",
            f"protocol={protocol.upper()}",
            f"localport={port}",
        ]
    else:
        cmd = ["iptables", "-A", "INPUT", "-p", protocol.lower(), "--dport", str(port), "-j", "DROP"]

    result = _run_command(cmd)
    result["port"] = port
    result["protocol"] = protocol
    return result


def restrict_protocol(protocol="tcp", policy="deny"):
    protocol = str(protocol).lower()
    policy = str(policy).lower()

    if _os_name().startswith("win"):
        cmd = [
            "netsh",
            "advfirewall",
            "firewall",
            "add",
            "rule",
            f"name=AI-NIDS-Protocol-{protocol}",
            "dir=in",
            "action=block" if policy == "deny" else "allow",
            f"protocol={protocol.upper()}",
        ]
    else:
        target = "DROP" if policy == "deny" else "ACCEPT"
        cmd = ["iptables", "-A", "INPUT", "-p", protocol, "-j", target]

    result = _run_command(cmd)
    result["protocol"] = protocol
    result["policy"] = policy
    return result
