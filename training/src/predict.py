# src/predict.py
"""CLI utility for single-image prediction.

Usage:
    python -m src.predict --image path/to/image.jpg
    python -m src.predict --image path/to/image.jpg --model models/crop_cnn_transfer_efficientnetb3.h5
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

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

from src.config import load_config
from src.utils.logger import logger


def load_class_names(model_dir: str) -> list[str]:
    """Load class names from the JSON saved during training."""
    path = os.path.join(model_dir, "class_names.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"class_names.json not found in {model_dir}. Train the model first."
        )
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def predict_image(
    image_path: str,
    model: tf.keras.Model,
    class_names: list[str],
    image_size: int = 224,
    top_k: int = 5,
) -> dict:
    """Run inference on a single image and return predictions.

    Returns
    -------
    dict with keys:
        - ``predicted_class`` : str
        - ``confidence``      : float
        - ``top_k``           : list[dict]  (class_name, confidence)
    """
    # Load & preprocess
    img = tf.keras.utils.load_img(image_path, target_size=(image_size, image_size))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_batch, verbose=0)[0]

    # Top-K
    top_indices = np.argsort(preds)[::-1][:top_k]
    top_predictions = [
        {"class_name": class_names[i], "confidence": round(float(preds[i]), 4)}
        for i in top_indices
    ]

    best_idx = top_indices[0]
    return {
        "predicted_class": class_names[best_idx],
        "confidence": round(float(preds[best_idx]), 4),
        "top_k": top_predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict crop type from image")
    parser.add_argument(
        "--image", type=str, required=True, help="Path to input image"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Path to .h5 model file"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config.yaml"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top predictions"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Resolve model path
    if args.model is None:
        model_name = f"crop_cnn_{cfg.model_type}"
        if cfg.model_type == "transfer":
            model_name += f"_{cfg.backbone}"
        model_path = os.path.join(cfg.model_save_dir, f"{model_name}.h5")
    else:
        model_path = args.model

    logger.info(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, safe_mode=False)

    class_names = load_class_names(cfg.model_save_dir)
    logger.info(f"Loaded {len(class_names)} class names")

    result = predict_image(
        args.image, model, class_names, cfg.image_size, args.top_k
    )

    print("\n" + "=" * 50)
    print(f"  🌾 Predicted Crop : {result['predicted_class']}")
    print(f"  📊 Confidence     : {result['confidence']:.2%}")
    print(f"\n  Top-{args.top_k} Predictions:")
    for i, pred in enumerate(result["top_k"], 1):
        bar = "█" * int(pred["confidence"] * 30)
        print(f"    {i}. {pred['class_name']:<45s} {pred['confidence']:.2%} {bar}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
