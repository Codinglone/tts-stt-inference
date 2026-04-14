# ---------------------------------------------------------------------------
# Stage 1 – base image with CUDA + Python
# ---------------------------------------------------------------------------
# pytorch/pytorch ships with CUDA 12.1, cuDNN 8, and Python 3.10.
# GCP T4 / V100 GPUs are fully compatible with CUDA 12.x.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        ffmpeg \
        libsndfile1 \
        libsndfile1-dev \
        wget \
        git \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && python -m unidic download

# ---------------------------------------------------------------------------
# Application files
# ---------------------------------------------------------------------------
COPY xtt_tts_script.py .
COPY stt_service.py .
COPY main.py .

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
# Directory where the Kinyarwanda model will be downloaded on first start.
# Mount a persistent volume here to avoid re-downloading between container restarts.
ENV MODEL_DIR=/app/models/kinyarwanda_health_model

RUN mkdir -p /app/models

# HuggingFace token — pass at runtime:
#   docker run -e HF_TOKEN=hf_xxx ...
ENV HF_TOKEN=""

# Hugging Face cache → keep models inside the image's /app volume
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8000

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD wget -qO- http://localhost:8000/health || exit 1

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
