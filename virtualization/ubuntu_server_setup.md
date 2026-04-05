# Ubuntu VM Setup and Auto-Start

This guide installs AI-NIDS-SOAR on Ubuntu and configures it to auto-start when the VM boots.

## 1) Install base packages

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```

## 2) Clone the repository

Replace `<REPO_URL>` with your repository URL.

```bash
cd ~
git clone <REPO_URL> AI-NIDS-SOAR
cd AI-NIDS-SOAR
```

## 3) Run auto-start setup script

```bash
chmod +x scripts/setup_ubuntu_autostart.sh
./scripts/setup_ubuntu_autostart.sh
```

The script will:
- Create `.venv` if missing
- Install required Python dependencies
- Create a systemd service: `ai-nids-dashboard.service`
- Enable startup at boot
- Start the service immediately

## 4) Verify service status

```bash
sudo systemctl status ai-nids-dashboard.service
sudo journalctl -u ai-nids-dashboard.service -f
```

## 5) Access dashboard

After startup, open:

```text
http://<UBUNTU_VM_IP>:8080
```

If using VirtualBox, make sure network mode/port forwarding allows access to port `8080`.

## Useful commands

```bash
# Stop service
sudo systemctl stop ai-nids-dashboard.service

# Start service
sudo systemctl start ai-nids-dashboard.service

# Disable auto-start
sudo systemctl disable --now ai-nids-dashboard.service

# Re-enable auto-start
sudo systemctl enable --now ai-nids-dashboard.service
```
