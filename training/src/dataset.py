# src/dataset.py
"""Data loading, augmentation, and tf.data pipeline construction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import tensorflow as tf

from src.config import Config
from src.utils.logger import logger



# ── Augmentation layer ───────────────────────────────────────

def build_augmentation_layer(cfg: Config) -> tf.keras.Sequential:
    """Return a ``tf.keras.Sequential`` of random augmentation layers."""
    aug = cfg.augmentation
    layers = []

    if aug.horizontal_flip:
        layers.append(tf.keras.layers.RandomFlip("horizontal"))
    if aug.vertical_flip:
        layers.append(tf.keras.layers.RandomFlip("vertical"))
    if aug.rotation_range > 0:
        layers.append(tf.keras.layers.RandomRotation(aug.rotation_range))
    if aug.zoom_range > 0:
        layers.append(tf.keras.layers.RandomZoom((-aug.zoom_range, aug.zoom_range)))
    if aug.brightness_range > 0:
        layers.append(tf.keras.layers.RandomBrightness(aug.brightness_range))
    if aug.contrast_range > 0:
        layers.append(tf.keras.layers.RandomContrast(aug.contrast_range))

    return tf.keras.Sequential(layers, name="data_augmentation")


# ── Dataset builders ─────────────────────────────────────────

def _load_split(
    data_dir: str,
    split: str,
    cfg: Config,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """Load a single split (train / val / test) using
    ``image_dataset_from_directory``."""
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        image_size=(cfg.image_size, cfg.image_size),
        batch_size=cfg.batch_size,
        label_mode="categorical",
        color_mode=cfg.color_mode,
        shuffle=shuffle,
        seed=42,
    )
    return ds


def load_datasets(
    cfg: Config,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Return ``(train_ds, val_ds, test_ds, class_names)``.

    * Training data is augmented, shuffled, cached and prefetched.
    * Validation / test data is only normalised, cached and prefetched.
    * ``class_names`` is the sorted list of folder names (= crop labels).
    """
    data_dir = cfg.data_dir
    logger.info(f"Loading dataset from: {data_dir}")

    # ── Load raw splits ──────────────────────────────────────
    train_ds = _load_split(data_dir, "train", cfg, shuffle=True)
    val_ds = _load_split(data_dir, "val", cfg, shuffle=False)
    test_ds = _load_split(data_dir, "test", cfg, shuffle=False)

    # Class names come from the first loaded dataset
    class_names: list[str] = sorted(train_ds.class_names)
    num_classes = len(class_names)
    logger.info(f"Discovered {num_classes} classes")
    logger.debug(f"Classes: {class_names[:10]}{'…' if num_classes > 10 else ''}")

    # Persist class names for inference
    class_names_path = os.path.join(cfg.model_save_dir, "class_names.json")
    with open(class_names_path, "w", encoding="utf-8") as fp:
        json.dump(class_names, fp, indent=2, ensure_ascii=False)
    logger.info(f"Saved class names → {class_names_path}")

    # ── Normalisation (0-255 → 0-1) ─────────────────────────
    rescale = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling")

    # ── Augmentation (train only) ────────────────────────────
    aug_layer = build_augmentation_layer(cfg)

    def _prepare_train(image, label):
        image = rescale(image)
        image = aug_layer(image, training=True)
        return image, label

    def _prepare_eval(image, label):
        image = rescale(image)
        return image, label

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        train_ds
        .map(_prepare_train, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_ds
        .map(_prepare_eval, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    test_ds = (
        test_ds
        .map(_prepare_eval, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    # Log split sizes
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        n_batches = tf.data.experimental.cardinality(ds).numpy()
        logger.info(f"  {name}: ~{n_batches} batches (batch_size={cfg.batch_size})")

    return train_ds, val_ds, test_ds, class_names
