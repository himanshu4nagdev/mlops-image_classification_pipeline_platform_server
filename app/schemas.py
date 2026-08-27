"""Pydantic request/response schemas for the serving API."""

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Optional metadata that may accompany an image upload.

    The image itself is sent as multipart/form-data, not JSON, so every
    field here is optional.
    """

    filename: Optional[str] = None


class TopPrediction(BaseModel):
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_3_predictions: List[TopPrediction]
    model_version: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    uptime_seconds: float
