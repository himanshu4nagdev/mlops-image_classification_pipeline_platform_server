"""FastAPI serving application for mobilenetv2-cifar10."""

import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from app.model_manager import ModelNotLoadedError, model_manager
from app.schemas import HealthResponse, PredictionResponse, TopPrediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("serving")

STATIC_DIR = Path(__file__).parent / "static"

# Same preprocessing used at training time: resize to 224x224, normalize
# with ImageNet statistics (the backbone was ImageNet-pretrained).
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    result = model_manager.reload_model()
    if result["status"] != "ok":
        logger.warning("Startup model load did not succeed: %s", result.get("detail"))
    yield


app = FastAPI(title="MobileNetV2 CIFAR-10 Classifier", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded,
        model_version=model_manager.version,
        uptime_seconds=time.time() - START_TIME,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet — train and register a model in MLflow first.",
        )

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    tensor = PREPROCESS(image).unsqueeze(0)

    start = time.perf_counter()
    try:
        top3, version = model_manager.predict(tensor)
    except ModelNotLoadedError:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet — train and register a model in MLflow first.",
        )
    inference_time_ms = (time.perf_counter() - start) * 1000

    top_class, top_confidence = top3[0]
    response = PredictionResponse(
        class_name=top_class,
        confidence=top_confidence,
        top_3_predictions=[TopPrediction(class_name=c, confidence=p) for c, p in top3],
        model_version=version,
        inference_time_ms=inference_time_ms,
    )

    logger.info(
        "predict filename=%s class=%s confidence=%.3f time_ms=%.1f model_version=%s",
        file.filename, top_class, top_confidence, inference_time_ms, version,
    )
    return response


@app.post("/reload-model")
async def reload_model():
    result = model_manager.reload_model()
    if result["status"] == "ok":
        return {"status": "reloaded", "version": result["version"], "run_id": result["run_id"]}
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=result["detail"])
    raise HTTPException(status_code=503, detail=result["detail"])
