# src/models/custom_cnn.py
"""Deep custom CNN architecture for crop classification (built from scratch)."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model  # type: ignore[import]


def _conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    dropout_rate: float = 0.25,
    name_prefix: str = "",
) -> tf.Tensor:
    """Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPool → Dropout."""
    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        name=f"{name_prefix}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu1")(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        name=f"{name_prefix}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu2")(x)

    x = layers.MaxPooling2D(pool_size=2, name=f"{name_prefix}_pool")(x)
    x = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop")(x)
    return x


def build_custom_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
) -> Model:
    """Build a 4-block custom CNN.

    Architecture
    ------------
    Block 1:  2× Conv(32)  → BN → ReLU → MaxPool → Dropout(0.25)
    Block 2:  2× Conv(64)  → BN → ReLU → MaxPool → Dropout(0.25)
    Block 3:  2× Conv(128) → BN → ReLU → MaxPool → Dropout(0.25)
    Block 4:  2× Conv(256) → BN → ReLU → MaxPool → Dropout(0.25)
    Head:     GAP → Dense(512) → BN → ReLU → Dropout(0.5) → Dense(softmax)
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    x = _conv_block(inputs, 32, name_prefix="block1")
    x = _conv_block(x, 64, name_prefix="block2")
    x = _conv_block(x, 128, name_prefix="block3")
    x = _conv_block(x, 256, name_prefix="block4")

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, use_bias=False, name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.ReLU(name="fc1_relu")(x)
    x = layers.Dropout(0.5, name="fc1_drop")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="CropCNN_Custom")
    return model
