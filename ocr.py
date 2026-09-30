# -*- coding: utf-8 -*-
"""Model loading + inference for the Hebrew handwriting OCR web app."""

import os
import re
import torch
import numpy as np
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer

from block_processor import HebrewBlockProcessor
import mishkefet_backend
import exp32_backend

MODELS_DIR = "models"

# Optional allow-list: OCR_MODELS="name1,name2". When set, only these models are listed and
# served, and a request for any other name (e.g. an old frontend's default) is served by the
# first one -- so a deployed frontend never breaks when the backend model changes.
ALLOWED = [m.strip() for m in os.environ.get("OCR_MODELS", "").split(",") if m.strip()]

_processor = HebrewBlockProcessor()
_cache = {}          # model_name -> (model, tokenizer )
_device = "cuda" if torch.cuda.is_available() else "cpu"


def list_models():
    """Folder names under models/ that look like a saved HF model (or the OCR_MODELS allow-list)."""
    if ALLOWED:
        return list(ALLOWED)
    if not os.path.isdir(MODELS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(MODELS_DIR)):
        path = os.path.join(MODELS_DIR, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
            out.append(name)
    out.append(mishkefet_backend.NAME)     # CTC model from the HF cache, not a models/ folder
    out.extend(exp32_backend.MODELS)       # exp7 encoder + CTC head (TrOCR_Hebrew/exp32)
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


def resolve_model(name: str) -> str:
    """With an allow-list, any name outside it maps to the first allowed model."""
    if ALLOWED and name not in ALLOWED:
        return ALLOWED[0]
    return name


def run_ocr(image: Image.Image, model_name: str, beams: int = 4):
    """Return (text, confidence) where confidence is the geometric-mean
    per-token probability of the decoded sequence, in [0, 1]."""
    model_name = resolve_model(model_name)
    if model_name == mishkefet_backend.NAME:
        return mishkefet_backend.run(image, beams)
    if model_name in exp32_backend.MODELS:
        return exp32_backend.run(image, model_name, beams)
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
        gen = model.generate(
            pixel_values,
            num_beams=beams,
            max_length=128,
            output_scores=True,
            return_dict_in_generate=True,
        )
    ids = gen.sequences
    print(f"[ocr] generated ids={ids.tolist()}", flush=True)
    result = tok.batch_decode(ids, skip_special_tokens=True)[0]

    # per-token log-probabilities of the chosen tokens (handles greedy & beam)
    beam_indices = getattr(gen, "beam_indices", None)
    transition = model.compute_transition_scores(
        gen.sequences, gen.scores, beam_indices, normalize_logits=True
    )
    mask = torch.isfinite(transition)          # padding after EOS is -inf
    safe = torch.where(mask, transition, torch.zeros_like(transition))
    n_tokens = mask.sum(dim=1).clamp(min=1)
    confidence = float((safe.sum(dim=1) / n_tokens).exp()[0])

    # A result with no Hebrew letter (e.g. "-", ".", stray punctuation) is almost
    # always a false-positive detection — force 0 confidence so the UI flags it red
    # instead of showing a spurious high score.
    if not re.search(r"[א-ת]", result):
        confidence = 0.0

    print(f"[ocr] result={repr(result)} confidence={confidence:.3f}", flush=True)
    return result, confidence
