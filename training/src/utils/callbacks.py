# src/utils/callbacks.py
"""Custom Keras callbacks for training progress logging."""

from __future__ import annotations

import tensorflow as tf
from src.utils.logger import logger


class LoguruCallback(tf.keras.callbacks.Callback):
    """Log epoch metrics to Loguru at the end of each epoch."""

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        parts = [f"Epoch {epoch + 1:>3d}"]
        for key in ("loss", "accuracy", "val_loss", "val_accuracy",
                     "top_5_accuracy", "val_top_5_accuracy", "lr"):
            if key in logs:
                val = logs[key]
                if "accuracy" in key:
                    parts.append(f"{key}={val:.4f}")
                elif key == "lr":
                    parts.append(f"lr={val:.2e}")
                else:
                    parts.append(f"{key}={val:.4f}")
        logger.info(" | ".join(parts))

    def on_train_begin(self, logs: dict | None = None) -> None:
        logger.info("━━━ Training started ━━━")

    def on_train_end(self, logs: dict | None = None) -> None:
        logger.info("━━━ Training finished ━━━")


class LRSchedulerLogger(tf.keras.callbacks.Callback):
    """Log whenever ReduceLROnPlateau fires."""

    def __init__(self) -> None:
        super().__init__()
        self._prev_lr: float | None = None

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        if self._prev_lr is not None and current_lr != self._prev_lr:
            logger.warning(
                f"Learning rate reduced: {self._prev_lr:.2e} → {current_lr:.2e}"
            )
        self._prev_lr = current_lr
