# -*- coding: utf-8 -*-
"""
Model inference server — runs on the notebook (where there is enough RAM/GPU).

Exposes a tiny API the Azure frontend calls (through the reverse SSH tunnel):
    GET  /health   -> {"status": "ok", "models": [...]}
    GET  /models   -> {"models": [...]}
    POST /ocr      -> form: model, beams, file(image)  ->  {"text": "..."}

Run (from the webHebrewOCR directory, so models/ and block_processor.py resolve):
    uvicorn model_server:app --host 127.0.0.1 --port 8001
"""

import io

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image, UnidentifiedImageError

import ocr   # the actual model loading + inference (loads torch/transformers)

app = FastAPI(title="Hebrew OCR model server")


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
    if model not in ocr.list_models():
        raise HTTPException(400, f"unknown model: {model}")
    try:
        image = Image.open(io.BytesIO(await file.read()))
    except UnidentifiedImageError:
        raise HTTPException(400, "uploaded file is not a valid image")
    text = ocr.run_ocr(image, model, beams)
    return {"text": text}
