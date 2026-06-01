# Remote-backend architecture

The model runs on the **notebook** (enough RAM/GPU); the **Azure VM** runs only a
thin web frontend that forwards images to the notebook through a **reverse SSH
tunnel**. The notebook dials *out* to Azure, so it works behind home NAT.

```
[user] → [Azure VM: thin web UI :80] → (reverse SSH tunnel) → [notebook: model server :8001] → text
```

Pieces:
- `model_server.py` — FastAPI inference server (notebook). Uses `ocr.py` + `block_processor.py`.
- `main.py` — thin frontend (Azure). Forwards to `MODEL_SERVER_URL` (default `http://127.0.0.1:8001`).
- `deploy/model-server.service`, `deploy/ocr-tunnel.service` — systemd units (notebook).

---

## 1. Notebook — one-time setup

```bash
sudo apt update && sudo apt install -y autossh

# deps for the model server (already present in this venv from training)
/mnt/ssd2/cyttic/projects/TrOCR_Hebrew/.venv/bin/pip install -r requirements-model.txt

# SSH key the tunnel will use (skip if you already have one trusted by the VM)
ssh-keygen -t ed25519 -f ~/.ssh/azure_deploy_key -N ""
ssh-copy-id -i ~/.ssh/azure_deploy_key.pub azureuser@<VM_PUBLIC_IP>
ssh -i ~/.ssh/azure_deploy_key azureuser@<VM_PUBLIC_IP>   # confirm it logs in
```

Edit `deploy/ocr-tunnel.service` and set `VM_PUBLIC_IP` (and the key path if different).

## 2. Notebook — install the services

```bash
sudo cp deploy/model-server.service deploy/ocr-tunnel.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now model-server ocr-tunnel

systemctl status model-server ocr-tunnel      # both should be "active (running)"
curl localhost:8001/health                    # {"status":"ok","models":["first_iteration"]}
```

They start on boot and auto-restart on crash / network drop.

## 3. Azure VM — verify the tunnel landed

```bash
curl localhost:8001/health        # same JSON, proxied from the notebook
```

## 4. Deploy the thin frontend

Push this branch → GitHub Actions builds the small image, pushes to Docker Hub,
and (on the VM) runs it with `--network host` + `MODEL_SERVER_URL=http://127.0.0.1:8001`.

Open `http://<VM_PUBLIC_IP>/` — the UI loads from Azure, OCR runs on the notebook.

---

## Notes
- The public site works only while the **notebook is on** and the tunnel is up.
- Frontend image is ~150 MB (no torch/model) → fast builds and no VM OOM.
- Logs: `journalctl -u model-server -f` / `journalctl -u ocr-tunnel -f` on the notebook.
- Local test without Azure: run `uvicorn model_server:app --port 8001` and
  `uvicorn main:app --port 8000`, then open `http://localhost:8000`.
