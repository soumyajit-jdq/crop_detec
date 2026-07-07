# api/main.py
"""FastAPI application — crop classification inference server.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

if "DML_VISIBLE_DEVICES" not in os.environ:
    os.environ["DML_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router, set_model
from src.config import load_config
from src.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model at startup."""
    cfg = load_config()

    # Determine model path
    model_name = f"crop_cnn_{cfg.model_type}"
    if cfg.model_type == "transfer":
        model_name += f"_{cfg.backbone}"
    model_path = os.path.join(cfg.model_save_dir, f"{model_name}.h5")

    # Load class names
    class_names_path = os.path.join(cfg.model_save_dir, "class_names.json")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.warning("API starting without model — /predict will return 503")
        set_model(None, [], cfg.image_size, cfg.model_type)
    elif not os.path.exists(class_names_path):
        logger.error(f"class_names.json not found: {class_names_path}")
        logger.warning("API starting without model — train first")
        set_model(None, [], cfg.image_size, cfg.model_type)
    else:
        logger.info(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        with open(class_names_path, "r", encoding="utf-8") as fp:
            class_names = json.load(fp)
        set_model(model, class_names, cfg.image_size, cfg.model_type)
        logger.info(
            f"Model loaded — {len(class_names)} classes, "
            f"type={cfg.model_type}"
        )

    yield  # Application runs here

    logger.info("Shutting down API server")


# ── Application ──────────────────────────────────────────────

app = FastAPI(
    title="🌾 Crop Detection API",
    description=(
        "Upload an image of a crop/plant and get the predicted crop type "
        "with confidence scores. Powered by CNN (TensorFlow/Keras)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — redirect hint to docs."""
    return {
        "message": "Crop Detection API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
