# src/export.py
"""Export a trained .h5 model to ONNX and SavedModel formats."""

from __future__ import annotations

import argparse
import os

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


def export_saved_model(model: tf.keras.Model, out_dir: str) -> str:
    """Export to TensorFlow SavedModel format."""
    path = os.path.join(out_dir, "crop_cnn_saved_model")
    model.export(path)
    logger.info(f"SavedModel exported → {path}")
    return path


def export_onnx(model: tf.keras.Model, out_dir: str, opset: int = 13) -> str:
    """Export to ONNX via tf2onnx."""
    try:
        import tf2onnx
    except ImportError:
        logger.error("tf2onnx not installed — run: pip install tf2onnx")
        raise

    onnx_path = os.path.join(out_dir, "crop_cnn.onnx")
    spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input_image"),)

    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=opset
    )

    import onnx
    onnx.save(model_proto, onnx_path)
    logger.info(f"ONNX model exported → {onnx_path}")
    return onnx_path


def export_tflite(model: tf.keras.Model, out_dir: str) -> str:
    """Export to TFLite (quantised)."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = os.path.join(out_dir, "crop_cnn.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    logger.info(f"TFLite model exported → {tflite_path}")
    return tflite_path


def export_all(
    model_path: str | None = None,
    config_path: str | None = None,
) -> dict[str, str]:
    """Export the trained model to all configured formats."""
    cfg = load_config(config_path)

    # ── Determine model path ─────────────────────────────────
    if model_path is None:
        model_name = f"crop_cnn_{cfg.model_type}"
        if cfg.model_type == "transfer":
            model_name += f"_{cfg.backbone}"
        model_path = os.path.join(cfg.model_save_dir, f"{model_name}.h5")

    logger.info(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, safe_mode=False)

    out_dir = cfg.model_save_dir
    exported: dict[str, str] = {}

    for fmt in cfg.export_formats:
        fmt = fmt.lower().strip()
        try:
            if fmt == "h5":
                # Already in .h5 — just note it
                exported["h5"] = model_path
                logger.info(f"H5 model already at → {model_path}")

            elif fmt == "saved_model":
                exported["saved_model"] = export_saved_model(model, out_dir)

            elif fmt == "onnx":
                exported["onnx"] = export_onnx(model, out_dir)

            elif fmt == "tflite":
                exported["tflite"] = export_tflite(model, out_dir)

            else:
                logger.warning(f"Unknown export format '{fmt}' — skipping")

        except Exception as e:
            logger.error(f"Export to {fmt} failed: {e}")

    logger.info(f"Export complete: {list(exported.keys())}")
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained crop classifier")
    parser.add_argument("--model", type=str, default=None, help="Path to .h5 model")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    export_all(model_path=args.model, config_path=args.config)
    logger.info("✅ Export complete!")


if __name__ == "__main__":
    main()
