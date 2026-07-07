# api/schemas.py
"""Pydantic request / response models for the prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    """A single class prediction with its confidence score."""

    class_name: str = Field(..., description="Crop class name")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")


class PredictionResponse(BaseModel):
    """Response from the /predict endpoint."""

    predicted_class: str = Field(..., description="Top-1 predicted crop class")
    confidence: float = Field(..., ge=0, le=1, description="Top-1 confidence")
    top_5: list[PredictionItem] = Field(..., description="Top-5 predictions")


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    num_classes: int = Field(..., description="Number of crop classes")
    model_type: str = Field(..., description="Model architecture type")


class ClassesResponse(BaseModel):
    """Response from the /classes endpoint."""

    num_classes: int
    classes: list[str]
