# -*- coding: utf-8 -*-
"""exp32 backend: exp7's ViT encoder + a CTC head (TrOCR_Hebrew/exp32), weights on the public
HF repo cyttic/exp32-exp7-encoder-ctc-<arm> (safetensors -- no pickle).

Pipeline, identical to training:
  exp7's HebrewBlockProcessor (AUTOCROP=False: mirror -> 64 px -> tile into 384x384)
  -> ViT (exp7 architecture, fine-tuned) -> 24x24 patch grid regrouped into reading order
  (one frame per 16 px column, split into 2 sub-frames) -> CTC over Mishkefet's 198 chars
  -> beam-12 + Mishkefet's char 6-gram LM (always; the UI's beams value is ignored).

Mishkefet's charset/decoder/LM code comes through mishkefet_backend (same safe loading).
"""

import json
import math
import os
import re
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image, ImageOps

import mishkefet_backend

MODELS = {"exp32-ctc-realsyn": "cyttic/exp32-exp7-encoder-ctc-realsyn",
          "exp32-ctc-real": "cyttic/exp32-exp7-encoder-ctc-real",
          "exp32-ctc-seqhead": "cyttic/exp32-exp7-encoder-ctc-seqhead",   # 2-layer transformer head
          "exp33-ctc-allreal": "cyttic/exp33-exp7-encoder-ctc-seqhead-allreal",   # seqhead, all real Hebrew
          "exp34-ctc-synpunct-iam": "cyttic/exp34-exp7-encoder-ctc-seqhead-allreal-synpunct-iam",  # + synth digits/punct + IAM
          "exp34-ctc-synpunct": "cyttic/exp34-exp7-encoder-ctc-seqhead-allreal-synpunct",          # + synth digits/punct
          "exp34-ctc-synpunct-punctdec": "cyttic/exp34-exp7-encoder-ctc-seqhead-allreal-synpunct",  # same weights, punctuation-aware decoder
          "exp35-ctc-base-en": "cyttic/exp35-trocr-base-en-ctc-seqhead-allreal-synpunct",     # TrOCR-base encoder, no Hebrew pretraining
          "exp35-ctc-large-en": "cyttic/exp35-trocr-large-en-ctc-seqhead-allreal-synpunct",   # TrOCR-large encoder (ViT-L, 24 layers)
          "exp35-ctc-large-en-punctdec": "cyttic/exp35-trocr-large-en-ctc-seqhead-allreal-synpunct"}
# Punctuation-aware decoding (TrOCR_Hebrew/exp34/tune_punct_decoder.py, chosen on HHD validation):
# the char LM scores only Hebrew letters and spaces; punctuation, digits and Latin get a flat
# score (PUNCT_FLAT) so the image decides them, and are stripped from the LM's context.
# HHD val CER 4.84 -> 4.15; benchmark . , errors 79% -> 49%.
PUNCT_DECODER = {"exp34-ctc-synpunct-punctdec", "exp35-ctc-large-en-punctdec"}
LM_WEIGHT, PUNCT_FLAT = 0.4, 0.0
ENC_CONFIG_SRC = "cyttic/trocr-hebrew-matan-exp7"
SUB, PROJ = 2, 512

_lock = threading.Lock()
_cache = {}
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _sinusoid(n, d):
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


class _EncoderCTC(nn.Module):
    """Same module as in the exp32/exp33 notebooks (state-dict compatible). seq_layers > 0 adds
    the transformer over reading-order sub-frames (arm seqhead, exp33)."""

    def __init__(self, vit, n_classes, sub=SUB, proj=PROJ, seq_layers=0):
        super().__init__()
        self.vit = vit
        self.Hd, self.sub, self.proj_dim = vit.config.hidden_size, sub, proj
        self.proj = nn.Sequential(nn.Linear(4 * self.Hd, sub * proj), nn.GELU(), nn.Dropout(0.0))
        self.seq = None
        if seq_layers:
            self.register_buffer("pe", _sinusoid(6 * 24 * sub, proj), persistent=False)
            layer = nn.TransformerEncoderLayer(proj, nhead=8, dim_feedforward=4 * proj, dropout=0.0,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.seq = nn.TransformerEncoder(layer, num_layers=seq_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(proj)
        self.head = nn.Linear(proj, n_classes)

    def forward(self, pixel_values, cols=None):
        t = self.vit(pixel_values=pixel_values).last_hidden_state[:, 1:, :]
        B = t.shape[0]
        g = t.reshape(B, 24, 24, self.Hd).reshape(B, 6, 4, 24, self.Hd).permute(0, 1, 3, 2, 4)
        g = g.reshape(B, 6 * 24, 4 * self.Hd)
        y = self.proj(g).reshape(B, 6 * 24 * self.sub, self.proj_dim)
        if self.seq is not None:
            T = y.shape[1]
            pad = (torch.arange(T, device=y.device)[None, :] >= (cols.to(y.device) * self.sub)[:, None]) \
                if cols is not None else None
            y = self.seq(y + self.pe[:T].to(y.dtype), src_key_padding_mask=pad)
        return F.log_softmax(self.head(self.norm(y)).float(), dim=-1)


class _PunctAwareLM:
    """Wraps the char LM for beam_decode (which only calls .logprob). beam_decode multiplies by
    lm_weight, so the flat score is pre-divided to make a protected character's bonus exactly
    PUNCT_FLAT + length_bonus."""

    def __init__(self, base, lm_weight=LM_WEIGHT, flat=PUNCT_FLAT):
        self.base, self.flat = base, flat / lm_weight

    def logprob(self, context, char):
        if not re.match(r"[א-ת ]", char):
            return self.flat
        return self.base.logprob(re.sub(r" +", " ", re.sub(r"[^א-ת ]", "", context)), char)


def _preprocess(image):
    """exp7's HebrewBlockProcessor (AUTOCROP=False) + number of 16-px columns in use."""
    image = ImageOps.mirror(image.convert("RGB")); w, h = image.size
    nw = max(1, round(w * 64 / h)); image = image.resize((nw, 64), Image.LANCZOS)
    cont = Image.new("RGB", (384, 384), (255, 255, 255)); arr = np.array(image); sx = dx = dy = 0
    while sx < nw and dy < 384:
        cw = min(nw - sx, 384 - dx); cont.paste(Image.fromarray(arr[:, sx:sx + cw]), (dx, dy))
        sx += cw; dx += cw
        if dx >= 384:
            dx = 0; dy += 64
    t = torch.tensor(np.array(cont), dtype=torch.float32).permute(2, 0, 1) / 255.0
    return (t - 0.5) / 0.5, math.ceil(min(nw, 6 * 384) / 16)


def _load(name):
    repo = MODELS[name]
    with _lock:
        if repo in _cache:
            return _cache[repo]
        mishkefet_backend._repo()                      # puts hebocr on sys.path, safe LM unpickler
        from hebocr.charset import Charset
        from hebocr.lm import CharNGramLM
        from safetensors.torch import load_file
        from transformers import VisionEncoderDecoderConfig, ViTModel

        path = snapshot_download(repo)
        charset = Charset.load(os.path.join(path, "charset.json"))
        # the encoder architecture comes from the checkpoint the run started from (exp35+ record
        # it in results.json; older runs are all exp7's ViT-base)
        enc_src = ENC_CONFIG_SRC
        if os.path.exists(os.path.join(path, "results.json")):
            with open(os.path.join(path, "results.json")) as fh:
                enc_src = json.load(fh).get("encoder_source", ENC_CONFIG_SRC)
        enc_cfg = VisionEncoderDecoderConfig.from_pretrained(os.path.dirname(
            hf_hub_download(enc_src, "config.json"))).encoder
        state = load_file(os.path.join(path, "model.safetensors"))
        seq_layers = len({k.split(".")[2] for k in state if k.startswith("seq.layers.")})   # 0 = linear head
        model = _EncoderCTC(ViTModel(enc_cfg, add_pooling_layer=False), charset.n_classes, seq_layers=seq_layers)
        model.load_state_dict(state, strict=True)
        model.to(_device).eval()
        lm = CharNGramLM.load(os.path.join(mishkefet_backend._repo(), "hebrew_char6.pkl"))
        _cache[repo] = (model, charset, lm)
        return _cache[repo]


@torch.no_grad()
def run(image: Image.Image, model_name: str, beams: int = 4):
    """Return (text, confidence): beam-12 + char LM; confidence = mean best-path probability
    over emitting frames (same definition as the Mishkefet backend)."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    model, charset, lm = _load(model_name)
    from hebocr.decode import beam_decode, greedy_confidence   # on sys.path only after _load()
    px, cols = _preprocess(image)
    with torch.autocast("cuda", dtype=torch.float16, enabled=_device == "cuda"):
        lp = model(px.unsqueeze(0).to(_device), torch.tensor([cols]))
    lens = torch.tensor([cols * SUB])
    dec_lm = _PunctAwareLM(lm) if model_name in PUNCT_DECODER else lm
    text = beam_decode(lp.cpu(), lens, charset, beam_width=12, lm=dec_lm, lm_weight=LM_WEIGHT)[0]
    conf = float(greedy_confidence(lp.cpu(), lens, emitting_only=True)[0])
    if not re.search(r"[א-ת]", text):
        conf = 0.0
    print(f"[{model_name}] size={image.size} beams={beams} result={text!r} confidence={conf:.3f}", flush=True)
    return text, conf
