FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by torch's CPU threading backend
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# --extra-index-url pulls CPU-only torch/torchvision wheels: this host has no
# GPU, so the default CUDA-bundled wheels would just waste bandwidth and disk.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
