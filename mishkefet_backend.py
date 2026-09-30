# -*- coding: utf-8 -*-
"""Mishkefet-v1 backend: a CNN + ViT + CTC Hebrew line recognizer (itayinbar/Mishkefet-v1 on
the Hugging Face Hub, after HTR-VT). Its own code and weights ship in that HF repo, so they
are loaded from the local HF cache -- nothing lives in models/.

Safety: the repo's loader uses torch.load(weights_only=False) and plain pickles for the
language model, both of which can execute code on load. Here torch.load is forced to
weights_only=True and the LM unpickler rejects every global (the files were inspected: the
checkpoint needs only tensor globals and the LM pickles none).

Decoding is ALWAYS beam-12 with the character 6-gram LM (the configuration the model's
author tuned); the UI's `beams` value is logged but ignored. Weights are CC-BY-NC-SA-4.0.
"""

import os
import pickle
import re
import sys
import threading
import time
import types

import torch
from huggingface_hub import snapshot_download
from PIL import Image

NAME = "mishkefet-v1"
REPO_ID = "itayinbar/Mishkefet-v1"

_lock = threading.Lock()
_state = {}          # "repo", "greedy", "beam" -> loaded objects


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def _repo():
    if "repo" not in _state:
        path = snapshot_download(REPO_ID)          # uses the local HF cache when present
        if path not in sys.path:
            sys.path.insert(0, path)
        safe = types.SimpleNamespace(load=lambda fh: _NoGlobals(fh).load(), dump=pickle.dump,
                                     HIGHEST_PROTOCOL=pickle.HIGHEST_PROTOCOL)
        import hebocr.lm as lm_mod
        import hebocr.wordlm as wlm_mod
        lm_mod.pickle = safe
        wlm_mod.pickle = safe
        _state["repo"] = path
    return _state["repo"]


def _recognizer(beam: bool):
    key = "beam" if beam else "greedy"
    with _lock:
        if key not in _state:
            path = _repo()
            from hebocr.lm import CharNGramLM
            from hebocr.recognize import Recognizer
            kw = dict(beam_width=0)
            if beam:
                kw = dict(beam_width=12, lm=CharNGramLM.load(os.path.join(path, "hebrew_char6.pkl")),
                          lm_weight=0.4)
            real_load = torch.load
            torch.load = lambda f, *a, **k: real_load(f, *a, **{**k, "weights_only": True})
            try:
                _state[key] = Recognizer(os.path.join(path, "mishkefet-v1.pt"), **kw)
            finally:
                torch.load = real_load
    return _state[key]


def run(image: Image.Image, beams: int = 4):
    """Return (text, confidence). Confidence = mean probability of the best CTC path over
    the frames that emit a character, in [0, 1]."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        bg = Image.new("RGB", image.size, (255, 255, 255))     # canvas drawings are transparent
        bg.paste(image, mask=image.split()[3])
        image = bg
    else:
        image = image.convert("RGB")
    rec = _recognizer(beam=True)          # always beam-12 + char LM; the UI's beams value is ignored
    texts, confs = rec.read([image], return_confidence=True, emitting_only=True)
    text, conf = texts[0], float(confs[0])
    if not re.search(r"[א-ת]", text):     # same rule as ocr.run_ocr: no Hebrew letter -> flag it
        conf = 0.0
    print(f"[mishkefet] size={image.size} beams={beams} result={text!r} confidence={conf:.3f}", flush=True)
    if os.environ.get("MISHKEFET_DEBUG") == "1":   # keep what the server actually received
        os.makedirs("debug_inputs", exist_ok=True)
        stem = os.path.join("debug_inputs", f"{time.strftime('%Y%m%d-%H%M%S')}_beams{beams}")
        image.save(stem + ".png")
        with open(stem + ".txt", "w", encoding="utf-8") as fh:
            fh.write(f"{text}\t{conf:.3f}\n")
    return text, conf
