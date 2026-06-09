# -*- coding: utf-8 -*-
"""Model loading + inference for the Hebrew handwriting OCR web app."""

import os
import torch
import numpy as np
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer

from block_processor import HebrewBlockProcessor

MODELS_DIR = "models"

_processor = HebrewBlockProcessor()
_cache = {}          # model_name -> (model, tokenizer )
_device = "cuda" if torch.cuda.is_available() else "cpu"


def list_models():
    """Folder names under models/ that look like a saved HF model."""
    if not os.path.isdir(MODELS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(MODELS_DIR)):
        path = os.path.join(MODELS_DIR, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
            out.append(name)
    return out


def _load(name):
    if name in _cache:
        return _cache[name]
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(os.path.join(path, "config.json")):
        raise FileNotFoundError(f"no model at {path}")
    model = VisionEncoderDecoderModel.from_pretrained(path).to(_device).eval()
    tok_path = path if os.path.exists(os.path.join(path, "tokenizer_config.json")) \
        else os.path.join(MODELS_DIR, "trocr-hebrew-synthetic-cont")
    tok = AutoTokenizer.from_pretrained(tok_path)
    # make sure generation has its special tokens
    model.generation_config.decoder_start_token_id = tok.cls_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.sep_token_id
    model.generation_config.max_new_tokens = None
    _cache[name] = (model, tok)
    return model, tok


def run_ocr(image: Image.Image, model_name: str, beams: int = 4) -> str:
    model, tok = _load(model_name)
    print(f"[ocr] image size={image.size} mode={image.mode} model={model_name} beams={beams}", flush=True)
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    else:
        image = image.convert("RGB")
    arr = np.array(image)
    print(f"[ocr] after conversion: min={arr.min()} max={arr.max()} mean={arr.mean():.1f}", flush=True)
    pixel_values = _processor([image])["pixel_values"].to(_device)
    print(f"[ocr] pixel_values shape={pixel_values.shape} device={pixel_values.device}", flush=True)
    with torch.no_grad():
        ids = model.generate(pixel_values, num_beams=beams, max_length=128)
    print(f"[ocr] generated ids={ids.tolist()}", flush=True)
    result = tok.batch_decode(ids, skip_special_tokens=True)[0]
    print(f"[ocr] result={repr(result)}", flush=True)
    return result
