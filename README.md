# mobilenetv2-cifar10 Model Serving

FastAPI service that serves the `mobilenetv2-cifar10` model from the MLflow
Model Registry, with a drag-and-drop web UI for live demos.

## Infrastructure notes

- Runs on CPU only — no GPU on this host, all torch ops use `device="cpu"`.
- MLflow tracking server: `http://localhost:5000` (already running on this host).
- **Port 8000 was already in use on this server**, so the API is bound to
  **port 8001** instead. If you free up 8000 later, edit the `command:` line
  in `docker-compose.yml` back to `--port 8000` (and update the URLs below).

## Build & run

```bash
cd /home/iiitd/mlops-image-clf/serving
docker compose up --build -d
```

Check it came up:

```bash
docker compose ps
docker compose logs -f api
```

Stop it:

```bash
docker compose down
```

Because the container uses `network_mode: host`, it talks to MLflow at
`http://localhost:5000` directly — no port mapping or `host.docker.internal`
needed on Linux.

## Web UI

Open in a browser (including from your phone, if it's on the same network):

```
http://192.168.26.172:8001/
```

Drag an image onto the drop zone (or click to upload) to classify it. The
header shows live model status pulled from `/health`.

## API endpoints

### `GET /health`

```bash
curl http://192.168.26.172:8001/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "3",
  "uptime_seconds": 142.5
}
```

### `POST /predict`

Multipart image upload:

```bash
curl -X POST http://192.168.26.172:8001/predict \
  -F "file=@/path/to/image.jpg"
```

```json
{
  "class_name": "dog",
  "confidence": 0.87,
  "top_3_predictions": [
    {"class_name": "dog", "confidence": 0.87},
    {"class_name": "cat", "confidence": 0.09},
    {"class_name": "deer", "confidence": 0.02}
  ],
  "model_version": "3",
  "inference_time_ms": 42.1
}
```

Returns **503** if no model is currently loaded (e.g. nothing has been
registered in MLflow yet).

### `POST /reload-model`

Triggers a reload of the latest registered version from MLflow without
restarting the service. Use this right after a retraining run registers a
new model version.

```bash
curl -X POST http://192.168.26.172:8001/reload-model
```

```json
{"status": "reloaded", "version": "4", "run_id": "a1b2c3d4..."}
```

Returns **404** if no model is registered under `mobilenetv2-cifar10` yet,
or **503** if MLflow itself is unreachable.

## Model loading behavior

- On startup, the app queries the MLflow Model Registry for the highest
  version number registered under `mobilenetv2-cifar10` and loads it onto
  CPU. The loaded version and run ID are logged.
- If no model is registered yet, the app **starts anyway** (it does not
  crash) — `/health` reports `model_loaded: false` and `/predict` returns
  503 until a model is registered and `/reload-model` is called (or the
  service is restarted).
- CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog,
  horse, ship, truck.

## Rebuilding after code changes

```bash
docker compose up --build -d
```

Rebuilding is only needed for changes to `app/` or dependencies — a new
model version does **not** require a rebuild, just `POST /reload-model`.
