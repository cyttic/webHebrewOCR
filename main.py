# -*- coding: utf-8 -*-
"""
FastAPI app to visualize the Hebrew handwriting OCR model.

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
Then open http://localhost:8000
"""

import io
import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

import ocr

IMAGES_DIR = "images"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

os.makedirs("models", exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

app = FastAPI(title="Hebrew Handwriting OCR")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/api/models")
def get_models():
    return {"models": ocr.list_models()}


@app.get("/api/examples")
def get_examples():
    files = [f for f in sorted(os.listdir(IMAGES_DIR))
             if f.lower().endswith(IMAGE_EXTS)]
    return {"examples": files}


@app.post("/api/ocr")
async def api_ocr(
    model: str = Form(...),
    beams: int = Form(4),
    file: UploadFile = File(None),
    example: str = Form(None),
):
    if model not in ocr.list_models():
        raise HTTPException(400, f"unknown model: {model}")

    # image comes either from an upload or a chosen example
    if file is not None:
        try:
            image = Image.open(io.BytesIO(await file.read()))
        except UnidentifiedImageError:
            raise HTTPException(400, "uploaded file is not a valid image")
    elif example:
        path = os.path.join(IMAGES_DIR, os.path.basename(example))
        if not os.path.exists(path):
            raise HTTPException(404, f"example not found: {example}")
        image = Image.open(path)
    else:
        raise HTTPException(400, "no image provided")

    text = ocr.run_ocr(image, model, beams)
    return {"text": text}
