FROM python:3.11-slim

WORKDIR /app

# thin frontend deps only — no torch, no model (those live on the notebook)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# seed the gitignored images/ dir from the committed sample set
RUN mkdir -p images && cp -n sample_images/*.png images/ 2>/dev/null || true

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
