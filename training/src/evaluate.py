# src/evaluate.py
"""Evaluate a trained model on the test set — metrics, confusion matrix, report."""

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
from sklearn.metrics import classification_report, confusion_matrix

from src.config import load_config
from src.dataset import load_datasets
from src.utils.logger import logger



def _save_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    out_path: str,
) -> None:
    """Save a confusion-matrix heatmap to *out_path* using matplotlib."""
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns

    # For 139 classes the full matrix is huge — save a summarised version
    n = len(class_names)
    fig_size = max(12, n * 0.15)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if n <= 30 else False,
        yticklabels=class_names if n <= 30 else False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {out_path}")


def evaluate(
    model_path: str | None = None,
    config_path: str | None = None,
) -> dict:
    """Evaluate the model and return a metrics dict."""
    cfg = load_config(config_path)
    _, _, test_ds, class_names = load_datasets(cfg)
    num_classes = len(class_names)

    # Load model 
    if model_path is None:
        model_name = f"crop_cnn_{cfg.model_type}"
        if cfg.model_type == "transfer":
            model_name += f"_{cfg.backbone}"
        model_path = os.path.join(cfg.model_save_dir, f"{model_name}.h5")

    logger.info(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, safe_mode=False)

    # Built-in evaluation
    results = model.evaluate(test_ds, return_dict=True)
    logger.info(f"Test results: {results}")

    # Detailed per-class metrics 
    y_true_list: list[int] = []
    y_pred_list: list[int] = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_list.extend(np.argmax(labels.numpy(), axis=1).tolist())
        y_pred_list.extend(np.argmax(preds, axis=1).tolist())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    # Classification report
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    logger.info(f"\n{report_text}")

    # Save report
    outputs_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "outputs"
    )
    os.makedirs(outputs_dir, exist_ok=True)

    report_path = os.path.join(outputs_dir, "classification_report.json")
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report_dict, fp, indent=2)
    logger.info(f"Classification report saved → {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(outputs_dir, "confusion_matrix.png")
    _save_confusion_matrix(cm, class_names, cm_path)

    return {
        "test_loss": float(results.get("loss", 0)),
        "test_accuracy": float(results.get("accuracy", 0)),
        "test_top_5_accuracy": float(results.get("top_5_accuracy", 0)),
        "num_classes": num_classes,
        "total_test_samples": len(y_true),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained crop classifier")
    parser.add_argument("--model", type=str, default=None, help="Path to .h5 model")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    metrics = evaluate(model_path=args.model, config_path=args.config)
    logger.info(f"Evaluation summary: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
