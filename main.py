# -*- coding: utf-8 -*-
"""
Thin web frontend (deployed to Azure). Serves the UI and forwards images to the
model server running on the notebook, reached via the reverse SSH tunnel.

No model / torch here — just static files + an HTTP forward, so the container
stays tiny and needs almost no RAM.

    MODEL_SERVER_URL  env var -> where the model server is (default the tunnel).

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

MODEL_SERVER = os.environ.get("MODEL_SERVER_URL", "http://127.0.0.1:8001").rstrip("/")
IMAGES_DIR = "images"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

os.makedirs(IMAGES_DIR, exist_ok=True)

app = FastAPI(title="Hebrew Handwriting OCR (frontend)")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/api/models")
def get_models():
    """Proxy the model list from the model server (empty if it's unreachable)."""
    try:
        r = httpx.get(f"{MODEL_SERVER}/models", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"models": []}


@app.get("/api/examples")
def get_examples():
    files = [f for f in sorted(os.listdir(IMAGES_DIR)) if f.lower().endswith(IMAGE_EXTS)]
    return {"examples": files}


@app.post("/api/ocr")
async def api_ocr(
    model: str = Form(...),
    beams: int = Form(4),
    file: UploadFile = File(None),
    example: str = Form(None),
):
    # grab the raw image bytes from an upload or a chosen example
    if file is not None:
        data = await file.read()
        fname = file.filename or "upload.png"
    elif example:
        path = os.path.join(IMAGES_DIR, os.path.basename(example))
        if not os.path.exists(path):
            raise HTTPException(404, f"example not found: {example}")
        with open(path, "rb") as f:
            data = f.read()
        fname = os.path.basename(example)
    else:
        raise HTTPException(400, "no image provided")

    # forward to the model server (notebook, via the tunnel)
    try:
        r = httpx.post(
            f"{MODEL_SERVER}/ocr",
            data={"model": model, "beams": str(beams)},
            files={"file": (fname, data, "application/octet-stream")},
            timeout=120,
        )
    except httpx.RequestError:
        raise HTTPException(503, "model server unreachable — is the notebook + tunnel up?")
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    return r.json()
