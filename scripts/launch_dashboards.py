"""
Launch Script - Start all inference services and unified dashboard
Services: Random Forest, CNN-LSTM, CNN, LSTM, Unified Dashboard
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

def resolve_python_executable(repo_root):
    """Prefer the project's virtual environment Python, fall back to current Python."""
    if os.name == "nt":
        candidate = repo_root / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = repo_root / ".venv" / "bin" / "python"

    if candidate.exists():
        return str(candidate.resolve())

    return sys.executable


def start_service(name, script_path, python_exe, wait_time=2):
    """Start a service process for the current platform."""
    repo_root = Path(__file__).resolve().parents[1]
    service_script = (repo_root / script_path).resolve()

    print(f"Starting {name}...")

    if os.name == "nt":
        cmd = [
            "powershell",
            "-NoExit",
            "-Command",
            (
                f"Set-Location '{repo_root}'; "
                f"Write-Host '=== {name} ===' -ForegroundColor Cyan; "
                f"& '{python_exe}' '{service_script}'"
            ),
        ]
        process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        # For Linux/systemd startup, keep all child logs in one place.
        log_dir = repo_root / "data" / "forensics" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / (service_script.stem + ".log")
        log_handle = open(log_path, "a", encoding="utf-8")
        process = subprocess.Popen(
            [python_exe, str(service_script)],
            cwd=str(repo_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        process.log_handle = log_handle

    time.sleep(wait_time)
    return process


def stop_services(processes):
    """Gracefully stop launched child processes."""
    for proc in processes:
        if proc.poll() is None:
            try:
                if os.name == "nt":
                    proc.terminate()
                else:
                    proc.send_signal(signal.SIGTERM)
            except Exception:
                pass

    for proc in processes:
        try:
            proc.wait(timeout=5)
        except Exception:
            pass

        log_handle = getattr(proc, "log_handle", None)
        if log_handle:
            try:
                log_handle.close()
            except Exception:
                pass


def main():
    repo_root = Path(__file__).resolve().parents[1]
    python_exe = resolve_python_executable(repo_root)

    print("=" * 70)
    print("AI-NIDS Unified Full Stack Launcher")
    print("Models: Random Forest, CNN-LSTM, CNN, LSTM")
    print("=" * 70)
    print()
    print(f"Python executable: {python_exe}")
    print()
    
    services = [
        ("Random Forest Inference (Port 5002)", "services/inference/app.py"),
        ("CNN-LSTM Inference (Port 5003)", "services/inference_cnn_lstm/app.py"),
        ("CNN Inference (Port 5004)", "services/inference_cnn/app.py"),
        ("LSTM Inference (Port 5005)", "services/inference_lstm/app.py"),
        ("Unified Dashboard (Port 8080)", "dashboards/dashboard_unified.py"),
    ]
    
    launched_processes = []
    for name, script in services:
        launched_processes.append((name, start_service(name, script, python_exe, wait_time=3)))
    
    print()
    print("=" * 70)
    print("All services started!")
    print("=" * 70)
    print()
    print("Dashboard URLs:")
    print("  Unified:       http://localhost:8080")
    print()
    print("Inference Services:")
    print("  Random Forest: http://localhost:5002")
    print("  CNN-LSTM:      http://localhost:5003")
    print("  CNN:           http://localhost:5004")
    print("  LSTM:          http://localhost:5005")
    if os.name != "nt":
        print()
        print("Service logs: data/forensics/logs/")
    print()
    print("Press Ctrl+C to exit")
    print("=" * 70)
    
    try:
        while True:
            dashboard_failed = False
            for name, proc in launched_processes:
                return_code = proc.poll()
                if return_code is None:
                    continue

                if name == "Unified Dashboard (Port 8080)":
                    print(f"\n[ERROR] {name} exited with code {return_code}. Stopping launcher.")
                    dashboard_failed = True
                    break

                print(f"\n[WARN] {name} exited with code {return_code}. Keeping dashboard online.")

            if dashboard_failed:
                return 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        if os.name != "nt":
            stop_services(launched_processes)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

