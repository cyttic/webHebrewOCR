FROM python:3.11-slim

WORKDIR /app

# --- python deps -------------------------------------------------------------
# CPU-only PyTorch first (much smaller than the default CUDA build), then the
# rest of the app requirements.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt huggingface_hub

# --- bake the fine-tuned model into the image --------------------------------
# The model is too large for git, so it is pulled from the HuggingFace Hub at
# build time. Override MODEL_REPO via --build-arg if your repo name differs.
# If the HF repo is private, pass --build-arg HF_TOKEN=xxxx.
ARG MODEL_REPO=cyttic/trocr-hebrew-finetuned
ARG MODEL_DIR=first_iteration
ARG HF_TOKEN=""
RUN python -c "import os; from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='${MODEL_REPO}', local_dir='models/${MODEL_DIR}', \
token=(os.environ.get('HF_TOKEN') or None), \
ignore_patterns=['*.zip', '*.bin', '*.msgpack', '*.h5', '*.onnx'])" \
    && ls -la models/${MODEL_DIR}

# --- app code + example images ----------------------------------------------
COPY . .
# populate the gitignored images/ dir from the committed sample set
RUN mkdir -p images && cp -n sample_images/*.png images/ 2>/dev/null || true

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
