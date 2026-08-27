"""Load, cache, and hot-reload the mobilenetv2-cifar10 model from the MLflow Model Registry."""

import logging
import os
import threading
import time
from typing import List, Optional, Tuple

import mlflow
import torch
from mlflow import MlflowClient

logger = logging.getLogger("model_manager")

MODEL_NAME = "mobilenetv2-cifar10"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
DEVICE = "cpu"

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested but no model is currently loaded."""


class ModelManager:
    """Owns the currently loaded model, protected by a lock so reloads can't
    race with in-flight predictions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Optional[torch.nn.Module] = None
        self._version: Optional[str] = None
        self._run_id: Optional[str] = None
        self._loaded_at: Optional[float] = None

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self._client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> Optional[str]:
        return self._version

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    def _fetch_latest_version(self):
        """Return the ModelVersion object with the highest version number.

        Uses search_model_versions instead of the deprecated/stage-based
        get_latest_versions so this works across MLflow server versions.
        """
        versions = self._client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            return None
        return max(versions, key=lambda v: int(v.version))

    def reload_model(self) -> dict:
        """Fetch and load the latest registered version from MLflow.

        Safe to call at startup (when no model may exist yet) and via the
        /reload-model endpoint (to pick up a newly registered version).
        Never raises for "no model registered" or "MLflow unreachable" —
        callers should check the returned dict's "status" key.
        """
        try:
            latest = self._fetch_latest_version()
        except Exception as exc:
            logger.warning("Could not reach MLflow at %s: %s", MLFLOW_TRACKING_URI, exc)
            return {"status": "error", "detail": f"MLflow unreachable: {exc}"}

        if latest is None:
            logger.warning("No registered versions found for model '%s'", MODEL_NAME)
            return {"status": "not_found", "detail": f"No registered model named '{MODEL_NAME}' yet"}

        model_uri = f"models:/{MODEL_NAME}/{latest.version}"
        try:
            model = mlflow.pytorch.load_model(model_uri, map_location=DEVICE)
        except Exception as exc:
            logger.error("Failed to load model '%s' version %s: %s", MODEL_NAME, latest.version, exc)
            return {"status": "error", "detail": f"Failed to load model: {exc}"}

        model.to(DEVICE)
        model.eval()

        with self._lock:
            self._model = model
            self._version = latest.version
            self._run_id = latest.run_id
            self._loaded_at = time.time()

        logger.info(
            "Loaded model '%s' version=%s run_id=%s onto %s",
            MODEL_NAME, latest.version, latest.run_id, DEVICE,
        )
        return {"status": "ok", "version": latest.version, "run_id": latest.run_id}

    def predict(self, tensor: torch.Tensor) -> Tuple[List[Tuple[str, float]], str]:
        """Run inference on a preprocessed (1, C, H, W) CPU tensor.

        Returns a list of (class_name, confidence) sorted descending by
        confidence (top-3), plus the model version used for the prediction.
        Raises ModelNotLoadedError if no model is currently cached.
        """
        with self._lock:
            if self._model is None:
                raise ModelNotLoadedError("No model is currently loaded")
            model = self._model
            version = self._version

        with torch.no_grad():
            logits = model(tensor.to(DEVICE))
            probs = torch.nn.functional.softmax(logits[0], dim=0)

        top3_probs, top3_idx = torch.topk(probs, k=min(3, probs.shape[0]))
        top3 = [
            (CIFAR10_CLASSES[idx.item()], prob.item())
            for prob, idx in zip(top3_probs, top3_idx)
        ]
        return top3, version


model_manager = ModelManager()
