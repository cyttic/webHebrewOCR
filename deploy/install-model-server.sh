#!/usr/bin/env bash
#
# Install the Hebrew OCR model server as a systemd service on the BACKEND notebook.
# Makes it auto-start on boot and restart on failure — same treatment the tunnel has.
#
# Usage (on the backend, e.g. cyttic@192.168.3.56):
#     scp deploy/install-model-server.sh cyttic@192.168.3.56:~
#     ssh cyttic@192.168.3.56
#     sudo bash install-model-server.sh
#
# Optional overrides:
#     sudo PROJECT_DIR=/path/to/webHebrewOCR PYTHON=/path/to/venv/bin/python bash install-model-server.sh
#
set -euo pipefail

# ── config (override via env vars if your paths differ) ───────────────
RUN_USER="${RUN_USER:-${SUDO_USER:-$(id -un)}}"
PROJECT_DIR="${PROJECT_DIR:-/home/${RUN_USER}/projects/webHebrewOCR}"
PORT="${PORT:-8001}"
SERVICE=/etc/systemd/system/model-server.service

echo ">> user        : ${RUN_USER}"
echo ">> project dir : ${PROJECT_DIR}"
echo ">> port        : ${PORT}"

[ -d "${PROJECT_DIR}" ] || { echo "ERROR: project dir not found: ${PROJECT_DIR}"; exit 1; }
[ -f "${PROJECT_DIR}/model_server.py" ] || { echo "ERROR: model_server.py not in ${PROJECT_DIR}"; exit 1; }

# ── locate the venv python (the one with torch/transformers) ──────────
if [ -n "${PYTHON:-}" ]; then
  PY="${PYTHON}"
else
  PY=""
  for cand in "${PROJECT_DIR}/env/bin/python" \
              "${PROJECT_DIR}/.venv/bin/python" \
              "${PROJECT_DIR}/venv/bin/python"; do
    [ -x "${cand}" ] && { PY="${cand}"; break; }
  done
fi
[ -n "${PY}" ] || { echo "ERROR: could not find a venv python under ${PROJECT_DIR}. Re-run with PYTHON=/full/path/to/python"; exit 1; }
echo ">> python      : ${PY}"

# sanity: make sure that interpreter actually has the deps
"${PY}" -c "import uvicorn, torch, transformers" 2>/dev/null \
  || { echo "ERROR: ${PY} is missing uvicorn/torch/transformers — wrong venv?"; exit 1; }

# ── stop any manually-started model server holding the port ───────────
echo ">> stopping any manual 'uvicorn model_server' process..."
pkill -f "uvicorn model_server:app" 2>/dev/null || true
sleep 1

# ── write the unit ────────────────────────────────────────────────────
echo ">> writing ${SERVICE}"
cat > "${SERVICE}" <<EOF
[Unit]
Description=Hebrew OCR model server (FastAPI)
After=network-online.target
Wants=network-online.target

[Service]
User=${RUN_USER}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PY} -m uvicorn model_server:app --host 127.0.0.1 --port ${PORT}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# ── enable + start ────────────────────────────────────────────────────
systemctl daemon-reload
systemctl enable --now model-server
echo ">> waiting for it to come up..."
sleep 6

# ── verify (backend has no curl, so use python) ───────────────────────
echo ">> systemctl status:"
systemctl is-active model-server || true
echo ">> /models response:"
"${PY}" -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:${PORT}/models', timeout=10).read().decode())" \
  || { echo "WARN: could not reach the server — check: journalctl -u model-server -n 30 --no-pager"; exit 1; }

echo
echo "DONE. Model server is a systemd service now — survives reboots & restarts on failure."
