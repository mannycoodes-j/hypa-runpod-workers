FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf-cache \
    HYPAAI_ASR_MODEL=hypaai/Hypa-Whisper-Small-v2 \
    PORT=80

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Bake the weights in. The endpoint has no network volume, so a cold worker
# would otherwise re-download ~930 MB from Hugging Face on every start.
RUN python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['HYPAAI_ASR_MODEL'])"

COPY app.py transcriber.py ./

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-80}"]
