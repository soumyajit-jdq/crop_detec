# src/models/__init__.py
"""Model factory — build the right architecture from config."""

from __future__ import annotations

import math

import tensorflow as tf

from src.config import Config
from src.models.custom_cnn import build_custom_cnn
from src.models.transfer_cnn import build_transfer_cnn
from src.utils.logger import logger


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay to min_lr."""

    def __init__(self, base_lr: float, total_steps: int, warmup_steps: int, min_lr: float = 1e-5):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        # Linear warmup
        warmup_lr = self.base_lr * (step / tf.maximum(warmup_steps, 1.0))

        # Cosine decay
        decay_steps = total_steps - warmup_steps
        progress = (step - warmup_steps) / tf.maximum(decay_steps, 1.0)
        progress = tf.minimum(progress, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + tf.cos(math.pi * progress))

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }


def build_model(cfg: Config, num_classes: int, steps_per_epoch: int = 0) -> tf.keras.Model:
    """Return a compiled Keras model based on ``cfg.model_type``.

    Parameters
    ----------
    cfg : Config
        Project configuration (model_type, backbone, image_size, ...).
    num_classes : int
        Number of output classes (auto-discovered from the dataset).
    steps_per_epoch : int
        Number of training steps per epoch (needed for cosine LR schedule).
    """
    input_shape = (cfg.image_size, cfg.image_size, 3 if cfg.color_mode == "rgb" else 1)

    if cfg.model_type == "custom":
        logger.info("Building > Custom CNN from scratch")
        model = build_custom_cnn(input_shape, num_classes)
    elif cfg.model_type == "transfer":
        logger.info(f"Building > Transfer Learning ({cfg.backbone})")
        model = build_transfer_cnn(input_shape, num_classes, backbone=cfg.backbone)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    # Build optimizer with LR schedule
    if cfg.lr_schedule == "cosine" and steps_per_epoch > 0:
        total_steps = steps_per_epoch * cfg.epochs
        warmup_steps = steps_per_epoch * cfg.warmup_epochs
        lr = WarmupCosineDecay(
            base_lr=cfg.learning_rate,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=cfg.cosine_min_lr,
        )
        logger.info(
            f"LR schedule: cosine decay (warmup={cfg.warmup_epochs} epochs, "
            f"base_lr={cfg.learning_rate}, min_lr={cfg.cosine_min_lr})"
        )
    else:
        lr = cfg.learning_rate
        logger.info(f"LR schedule: constant lr={cfg.learning_rate}")

    if cfg.optimizer == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = tf.keras.optimizers.SGD(
            learning_rate=lr, momentum=0.9, nesterov=True
        )

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg.label_smoothing
        ),
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy"),
        ],
    )

    total_params = model.count_params()
    trainable = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    logger.info(
        f"Model ready - {total_params:,} params ({trainable:,} trainable)"
    )
    return model
