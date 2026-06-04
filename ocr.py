# -*- coding: utf-8 -*-
"""Model loading + inference for the Hebrew handwriting OCR web app."""

import os
import torch
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
    tok = AutoTokenizer.from_pretrained(path)
    # make sure generation has its special tokens
    model.generation_config.decoder_start_token_id = tok.cls_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.sep_token_id
    model.generation_config.max_new_tokens = None
    _cache[name] = (model, tok)
    return model, tok


def run_ocr(image: Image.Image, model_name: str, beams: int = 4) -> str:
    model, tok = _load(model_name)
    pixel_values = _processor([image.convert("RGB")])["pixel_values"].to(_device)
    with torch.no_grad():
        ids = model.generate(pixel_values, num_beams=beams, max_length=128)
    return tok.batch_decode(ids, skip_special_tokens=True)[0]
