#!/usr/bin/env bash
#
# Install the reverse SSH tunnel (backend -> Azure VM) as a systemd service.
# The backend dials OUT to the VM and exposes its local model server (port 8001)
# on the VM's 127.0.0.1:8001, so the thin frontend container can reach it.
#
# Usage (on the backend, e.g. cyttic@192.168.3.56):
#     scp deploy/install-tunnel.sh cyttic@192.168.3.56:~
#     ssh cyttic@192.168.3.56
#     sudo VM_HOST=<AZURE_VM_PUBLIC_IP> bash install-tunnel.sh
#
# Optional overrides (shown with defaults):
#     VM_USER=azureuser
#     KEY=/home/<user>/vm-framework_key.pem
#     PORT=8001
#
set -euo pipefail

RUN_USER="${RUN_USER:-${SUDO_USER:-$(id -un)}}"
VM_USER="${VM_USER:-azureuser}"
VM_HOST="${VM_HOST:-}"
KEY="${KEY:-/home/${RUN_USER}/vm-framework_key.pem}"
PORT="${PORT:-8001}"
SERVICE=/etc/systemd/system/tunnel.service

[ -n "${VM_HOST}" ] || { echo "ERROR: set the Azure VM IP, e.g.  sudo VM_HOST=1.2.3.4 bash install-tunnel.sh"; exit 1; }

echo ">> backend user : ${RUN_USER}"
echo ">> VM target    : ${VM_USER}@${VM_HOST}"
echo ">> ssh key      : ${KEY}"
echo ">> forward      : VM:127.0.0.1:${PORT} -> backend:127.0.0.1:${PORT}"

[ -f "${KEY}" ] || { echo "ERROR: ssh key not found: ${KEY}  (override with KEY=/path/to/key)"; exit 1; }

# Prefer autossh; fall back to plain ssh (systemd's Restart=always reconnects on drop).
COMMON_OPTS="-o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
if command -v autossh >/dev/null; then
  echo ">> transport    : autossh"
  EXECSTART="/usr/bin/autossh -M 0 -N ${COMMON_OPTS} -i ${KEY} -R ${PORT}:127.0.0.1:${PORT} ${VM_USER}@${VM_HOST}"
  EXTRA_ENV="Environment=AUTOSSH_GATETIME=0"
else
  echo ">> transport    : plain ssh (autossh not installed; systemd will auto-restart)"
  EXECSTART="/usr/bin/ssh -N ${COMMON_OPTS} -i ${KEY} -R ${PORT}:127.0.0.1:${PORT} ${VM_USER}@${VM_HOST}"
  EXTRA_ENV=""
fi

# key must be private (ssh refuses world-readable keys)
chmod 600 "${KEY}" 2>/dev/null || true

# stop any manually-started tunnel holding the forward
echo ">> stopping any manual autossh/ssh -R tunnel..."
pkill -f "ssh .*-R ${PORT}:127.0.0.1:${PORT}" 2>/dev/null || true
pkill -f "autossh .*${PORT}:127.0.0.1:${PORT}" 2>/dev/null || true
sleep 1

echo ">> writing ${SERVICE}"
cat > "${SERVICE}" <<EOF
[Unit]
Description=Reverse SSH tunnel to Azure VM (exposes the OCR model server)
After=network-online.target model-server.service
Wants=network-online.target

[Service]
User=${RUN_USER}
${EXTRA_ENV}
ExecStart=${EXECSTART}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now tunnel
echo ">> waiting for the tunnel to establish..."
sleep 8

echo ">> tunnel status:"
systemctl is-active tunnel || true
pgrep -a -f "ssh .*-R ${PORT}:127.0.0.1:${PORT}" || echo "WARN: no ssh tunnel process — check: journalctl -u tunnel -n 30 --no-pager"

# verify the forward actually landed: ask the VM (over the same key) what it sees on its 8001
echo ">> verifying from the VM side (does the VM now reach the model server?)..."
if ssh -i "${KEY}" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "${VM_USER}@${VM_HOST}" \
     "python3 -c \"import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:${PORT}/models', timeout=10).read().decode())\"" 2>/dev/null; then
  echo
  echo "DONE. Tunnel is a systemd service and the VM can reach the model server. Chain is live."
else
  echo "WARN: could not confirm from the VM (it may lack python3/curl, or the tunnel needs a few more seconds)."
  echo "      Check on the VM:  docker ps  &&  docker logs --tail 30 web-hebrew-ocr"
fi
