# src/models/transfer_cnn.py
"""Transfer-learning models (ResNet50 / EfficientNetB3) for crop classification."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model

from src.utils.logger import logger

# Map config string → Keras application
_BACKBONES = {
    "resnet50": (
        tf.keras.applications.ResNet50,
        tf.keras.applications.resnet50.preprocess_input,
    ),
    "efficientnetb3": (
        tf.keras.applications.EfficientNetV2B3,
        tf.keras.applications.efficientnet_v2.preprocess_input,
    ),
}


def build_transfer_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    backbone: str = "efficientnetb3",
) -> Model:
    """Build a transfer-learning model.

    The backbone is loaded with ImageNet weights and **frozen** by default.
    Call ``unfreeze_top_layers(model, n)`` after initial training to fine-tune.

    Architecture
    ------------
    Input → Preprocess → Frozen Backbone → GAP → Dense(512) →
    BN → ReLU → Dropout(0.5) → Dense(softmax)
    """
    if backbone not in _BACKBONES:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Choose from {list(_BACKBONES)}"
        )

    BackboneClass, preprocess_fn = _BACKBONES[backbone]

    inputs = layers.Input(shape=input_shape, name="input_image")

    # Backbone-specific preprocessing (handles rescaling internally)
    x = layers.Lambda(
        lambda img: preprocess_fn(img * 255.0),  # undo 0-1 rescaling for backbone preprocess
        name="backbone_preprocess",
    )(inputs)

    base_model = BackboneClass(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    base_model.trainable = False  # Freeze for initial training
    logger.info(
        f"Backbone {backbone} loaded — {len(base_model.layers)} layers (frozen)"
    )

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, use_bias=False, name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.ReLU(name="fc1_relu")(x)
    x = layers.Dropout(0.5, name="fc1_drop")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name=f"CropCNN_{backbone}")
    return model


def unfreeze_top_layers(model: Model, num_layers: int) -> None:
    """Unfreeze the top *num_layers* of the backbone for fine-tuning.

    Call this after the initial training phase, then recompile with a
    lower learning rate.
    """
    # The backbone is the first "functional" sub-model inside our model
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break

    if backbone is None:
        logger.warning("No backbone sub-model found — nothing to unfreeze")
        return

    backbone.trainable = True
    total = len(backbone.layers)
    freeze_up_to = max(0, total - num_layers)

    for layer in backbone.layers[:freeze_up_to]:
        layer.trainable = False

    trainable_count = sum(1 for l in backbone.layers if l.trainable)
    logger.info(
        f"Unfroze top {trainable_count} / {total} backbone layers for fine-tuning"
    )
