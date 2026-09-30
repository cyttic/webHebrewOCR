# -*- coding: utf-8 -*-
"""
Model inference server — runs on the notebook (where there is enough RAM/GPU).

Exposes a tiny API the Azure frontend calls (through the reverse SSH tunnel):
    GET  /health   -> {"status": "ok", "models": [...]}
    GET  /models   -> {"models": [...]}
    POST /ocr      -> form: model, beams, file(image)  ->  {"text": "...", "confidence": 0.0-1.0}
    POST /detect   -> form: file(image)  ->  {"boxes": [[x0,y0,x1,y1], ...]}  (CRAFT)

Run (from the webHebrewOCR directory, so models/ and block_processor.py resolve):
    uvicorn model_server:app --host 127.0.0.1 --port 8001
"""

import io
import os
import sys

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image, UnidentifiedImageError

import ocr   # the actual model loading + inference (loads torch/transformers)

# CRAFT word detection — code + weights live in the font-extractor project, so the
# font-extractor VM frontend can stay thin and call us for boxes over the same tunnel.
import cv2
import numpy as np
_FE_SRC = os.environ.get("FONT_EXTRACTOR_SRC", "/mnt/ssd2/cyttic/projects/font-extractor/src")
sys.path.insert(0, _FE_SRC)
from run_craft import load_net as _load_craft, score_maps as _craft_scores, boxes_from_scores as _craft_boxes

app = FastAPI(title="Hebrew OCR model server")

print("loading CRAFT ...")
_CRAFT = _load_craft()           # stock craft_mlt_25k.pth (clovaai general model)
# with an OCR_MODELS allow-list, load those models now so the first request is not slow
for _m in ocr.ALLOWED:
    if _m in ocr.exp32_backend.MODELS:
        print(f"preloading {_m} ...")
        ocr.exp32_backend._load(_m)
print("ready.", "models:", ocr.list_models())


@app.get("/health")
def health():
    return {"status": "ok", "models": ocr.list_models()}


@app.get("/models")
def models():
    return {"models": ocr.list_models()}


@app.post("/ocr")
async def do_ocr(
    model: str = Form(...),
    beams: int = Form(4),
    file: UploadFile = File(...),
):
    model = ocr.resolve_model(model)
    if model not in ocr.list_models():
        raise HTTPException(400, f"unknown model: {model}")
    try:
        image = Image.open(io.BytesIO(await file.read()))
    except UnidentifiedImageError:
        raise HTTPException(400, "uploaded file is not a valid image")
    text, confidence = ocr.run_ocr(image, model, beams)
    return {"text": text, "confidence": confidence}


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    text_threshold: float = Form(0.7),
    link_threshold: float = Form(0.4),
    low_text: float = Form(0.4),
):
    """CRAFT word detection -> axis-aligned boxes in the image's pixel coords."""
    arr = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(400, "could not decode image")
    rgb = cv2.cvtColor(cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
    st, sl, ratio = _craft_scores(_CRAFT, rgb)
    boxes = []
    for p in _craft_boxes(st, sl, ratio, text_threshold, link_threshold, low_text):
        p = np.array(p)
        boxes.append([int(p[:, 0].min()), int(p[:, 1].min()),
                      int(p[:, 0].max()), int(p[:, 1].max())])
    return {"boxes": boxes}
