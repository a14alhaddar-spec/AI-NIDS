#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

RUN_AS_USER="${SUDO_USER:-${USER}}"
RUN_AS_GROUP="$(id -gn "${RUN_AS_USER}")"
RUN_AS_HOME="$(getent passwd "${RUN_AS_USER}" | cut -d: -f6)"

SERVICE_NAME="ai-nids-dashboard.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

echo "[1/5] Ensuring Python virtual environment exists..."
if [[ ! -d "${REPO_ROOT}/.venv" ]]; then
  python3 -m venv "${REPO_ROOT}/.venv"
fi

# shellcheck source=/dev/null
source "${REPO_ROOT}/.venv/bin/activate"


echo "[2/5] Installing/updating Python dependencies..."
pip install --upgrade pip wheel
if [[ -f "${REPO_ROOT}/services/inference/requirements.txt" ]]; then
  pip install -r "${REPO_ROOT}/services/inference/requirements.txt"
fi
if [[ -f "${REPO_ROOT}/services/dashboard_api/requirements.txt" ]]; then
  pip install -r "${REPO_ROOT}/services/dashboard_api/requirements.txt"
fi
# Additional runtime libraries used by launched services.
pip install tensorflow scikit-learn pandas scapy


echo "[2b/5] Granting packet-capture capability to the virtualenv Python..."
sudo setcap cap_net_raw,cap_net_admin=eip "${REPO_ROOT}/.venv/bin/python" || true


echo "[3/5] Creating systemd service..."
SERVICE_CONTENT="[Unit]
Description=AI-NIDS unified dashboard stack
After=network.target

[Service]
Type=simple
User=${RUN_AS_USER}
Group=${RUN_AS_GROUP}
WorkingDirectory=${REPO_ROOT}
Environment=HOME=${RUN_AS_HOME}
Environment=PYTHONUNBUFFERED=1
ExecStart=${REPO_ROOT}/.venv/bin/python ${REPO_ROOT}/scripts/launch_dashboards.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"

echo "${SERVICE_CONTENT}" | sudo tee "${SERVICE_PATH}" > /dev/null


echo "[4/5] Reloading and enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"


echo "[5/5] Starting service..."
sudo systemctl restart "${SERVICE_NAME}"


echo
echo "Setup complete."
echo "Check status:   sudo systemctl status ${SERVICE_NAME}"
echo "View logs:      sudo journalctl -u ${SERVICE_NAME} -f"
echo "Disable startup: sudo systemctl disable --now ${SERVICE_NAME}"
