# Deployment Guide — Platform Server + Compute/Laptop Server

This repo (`mlops-image_classification_pipeline_platform_server`) is the
**serving half** of a two-repo, two-role pipeline:

| Role | Repo | What it does | Where |
|---|---|---|---|
| **Platform server** | *this repo* | Runs the FastAPI serving app that loads `mobilenetv2-cifar10` from the MLflow registry and exposes `/predict` + a demo UI | `192.168.26.172` |
| **Compute / laptop server** | [mlops-image_classification_pipeline](https://github.com/himanshu4nagdev/mlops-image_classification_pipeline) | Trains the model and registers new versions into the same MLflow registry | Any GPU/CPU machine (laptop, home box, GPU server) |

The platform server also already runs MLflow's tracking server and a MinIO
artifact store (S3-compatible) — these are **not part of either repo above**;
they're a separate pre-existing stack (`/home/iiitd/mlops_project/mlops-pipeline`
on this host) that both this serving app and every compute server depend on.

```
                    ┌───────────────────────────────────────────┐
                    │            Platform server                │
                    │            192.168.26.172                 │
                    │                                             │
                    │  mlflow server  :5000  (tracking + registry)│
                    │  minio          :9000  (s3://mlflow-artifacts)
                    │  ── pre-existing stack, not this repo ──   │
                    │                                             │
                    │  serving-api (this repo)          :8002    │
                    │    loads models:/mobilenetv2-cifar10/latest │
                    └───────────────▲───────────────┬────────────┘
                                     │ tracking_uri   │ /predict, /reload-model
                          ┌──────────┴──────┐        │
                          │  Compute server  │   ┌────┴─────┐
                          │  (laptop_mx450)  │   │  Browser  │
                          │  trains + registers   │  / curl   │
                          │  mobilenetv2-cifar10  └──────────┘
                          └──────────────────┘
```

**Order matters: Part 1 (platform server) must be done before Part 2 (compute
server)** — the compute server's config needs the platform server's IP and a
reachable MLflow endpoint before it can register anything.

---

## Part 1 — Platform server setup (this machine, `192.168.26.172`)

### 1.1 Verify the MLflow + MinIO stack is already running

This host already has MLflow and MinIO running as a separate docker stack.
**Do not restart or rebuild it** — just confirm it's healthy:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "mlflow|minio"
curl -s http://localhost:5000/health          # expect: OK
curl -s http://localhost:9000/minio/health/live -o /dev/null -w "%{http_code}\n"   # expect: 200
```

If either is down, that's a problem with `/home/iiitd/mlops_project/mlops-pipeline`,
not with this repo — go there to fix it (`docker compose up -d` in that
directory) before continuing.

### 1.2 Clone this repo

```bash
git clone git@github.com:himanshu4nagdev/mlops-image_classification_pipeline_platform_server.git serving
cd serving
```

### 1.3 Configure MinIO/S3 credentials

`docker-compose.yml` reads S3 credentials from a gitignored `.env` (never
committed — see `.env.example` for the required keys):

```bash
cp .env.example .env
```

Fill in the real MinIO credentials (ask whoever set up the MinIO stack, or
check `/home/iiitd/mlops_project/mlops-pipeline` if you have access to it):

```
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=<real value>
AWS_SECRET_ACCESS_KEY=<real value>
```

### 1.4 Build and run

```bash
docker compose up --build -d
```

The app binds to **port 8002** (not 8000) — ports 8000 and 8001 on this host
are already used by the pre-existing `mlops-pipeline` stack's own serving
containers. If you're deploying to a fresh platform server where those ports
are free, change the `command:` line in `docker-compose.yml` back to
`--port 8000`.

### 1.5 Verify

```bash
curl http://192.168.26.172:8002/health
```

Expect `model_loaded: false` until a compute server has registered at least
one version of `mobilenetv2-cifar10` (Part 2). This is normal — the app
starts fine either way and never crashes on a missing model.

Web UI: **http://192.168.26.172:8002/**

---

## Part 2 — Compute / laptop server setup (do this on every training machine)

Full detail lives in the training repo's own
[DEPLOYMENT.md](https://github.com/himanshu4nagdev/mlops-image_classification_pipeline/blob/main/DEPLOYMENT.md).
Summary:

```bash
git clone https://github.com/himanshu4nagdev/mlops-image_classification_pipeline.git
cd mlops-image_classification_pipeline
```

Confirm `config/env_config.yaml` points at this platform server (already set
correctly as of the last check):

```yaml
mlflow:
  tracking_uri: "http://192.168.26.172:5000"
  experiment_name: "image-clf-mobilenetv2"
  registry_model_name: "mobilenetv2-cifar10"
```

Pick the profile matching this machine's GPU (`laptop_mx450`,
`home_server_3gb`, `gpu_server_24gb`, or `cpu_fallback`) via `active_profile`
in the same file, then:

- **Windows:** `setup_and_run.bat` (idempotent, safe to re-run)
- **Linux/Mac:**
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # or omit for CPU
  pip install -r requirements.txt
  python run_pipeline.py --profile <name>
  ```

Sanity-check connectivity before a long run:

```bash
curl http://192.168.26.172:5000/health
```

When the run finishes, it registers a new version under
`mobilenetv2-cifar10` in the same registry the platform server reads from.

---

## Part 3 — After training: pick up the new model on the platform server

The serving app caches the model in memory and does **not** poll for new
versions automatically. After a compute server finishes training:

```bash
curl -X POST http://192.168.26.172:8002/reload-model
```

```json
{"status": "reloaded", "version": "1", "run_id": "..."}
```

Then confirm:

```bash
curl http://192.168.26.172:8002/health
# model_loaded: true, model_version: "1"
```

No rebuild or restart of the serving container is needed — `/reload-model`
picks up the new version live.

---

## Troubleshooting

**`/health` shows `model_loaded: false` after training finished** — the
model registered under a different name than `mobilenetv2-cifar10` (check
`curl http://localhost:5000/api/2.0/mlflow/registered-models/search`), or
`/reload-model` wasn't called yet.

**`/reload-model` returns 404** — nothing is registered yet under
`mobilenetv2-cifar10`; check the compute server's run actually completed and
registered (not just trained).

**`/reload-model` returns 503** — the serving container can't reach MLflow
at `http://localhost:5000` (Part 1.1) or MinIO for the artifact download —
check both are up and `.env` has the right MinIO credentials.

**Port 8002 already in use on a different platform server** — pick any free
port and update the `command:` line in `docker-compose.yml` accordingly;
nothing else needs to change since the container uses `network_mode: host`.
