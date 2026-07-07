# api/routes.py
"""API route handlers — /predict, /health, /classes."""

from __future__ import annotations

import io
import numpy as np
import tensorflow as tf
from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import (
    ClassesResponse,
    HealthResponse,
    PredictionItem,
    PredictionResponse,
)

router = APIRouter()

# ── These are populated by main.py at startup ────────────────
_model: tf.keras.Model | None = None
_class_names: list[str] = []
_image_size: int = 224
_model_type: str = "unknown"


def set_model(
    model: tf.keras.Model | None,
    class_names: list[str],
    image_size: int,
    model_type: str,
) -> None:
    """Called once at application startup to inject the loaded model."""
    global _model, _class_names, _image_size, _model_type
    _model = model
    _class_names = class_names
    _image_size = image_size
    _model_type = model_type


# ── Endpoints ────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health-check endpoint."""
    return HealthResponse(
        status="healthy" if _model is not None else "model_not_loaded",
        model_loaded=_model is not None,
        num_classes=len(_class_names),
        model_type=_model_type,
    )


@router.get("/classes", response_model=ClassesResponse, tags=["Info"])
async def list_classes():
    """Return the list of all crop class names the model can recognise."""
    return ClassesResponse(
        num_classes=len(_class_names),
        classes=_class_names,
    )


@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(..., description="Image file (jpg/png)")):
    """Upload an image and receive crop classification predictions."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Upload an image.",
        )

    try:
        contents = await file.read()
        img = tf.image.decode_image(contents, channels=3)
        img = tf.image.resize(img, (_image_size, _image_size))
        img = img / 255.0
        img_batch = tf.expand_dims(img, axis=0)

        preds = _model.predict(img_batch, verbose=0)[0]
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process image: {str(e)}",
        )

    # Top-5
    top_indices = np.argsort(preds)[::-1][:5]
    top_5 = [
        PredictionItem(
            class_name=_class_names[i],
            confidence=round(float(preds[i]), 4),
        )
        for i in top_indices
    ]

    best = top_indices[0]
    return PredictionResponse(
        predicted_class=_class_names[best],
        confidence=round(float(preds[best]), 4),
        top_5=top_5,
    )
