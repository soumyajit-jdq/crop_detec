# src/models/__init__.py
"""Model factory — build the right architecture from config."""

from __future__ import annotations

import tensorflow as tf

from src.config import Config
from src.models.custom_cnn import build_custom_cnn
from src.models.transfer_cnn import build_transfer_cnn
from src.utils.logger import logger


def build_model(cfg: Config, num_classes: int) -> tf.keras.Model:
    """Return a compiled Keras model based on ``cfg.model_type``.

    Parameters
    ----------
    cfg : Config
        Project configuration (model_type, backbone, image_size, …).
    num_classes : int
        Number of output classes (auto-discovered from the dataset).
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

    # Compile
    if cfg.optimizer == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    else:
        opt = tf.keras.optimizers.SGD(
            learning_rate=cfg.learning_rate, momentum=0.9, nesterov=True
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
