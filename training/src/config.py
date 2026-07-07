# src/config.py
"""Load and validate the project configuration from config/config.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class AugmentationConfig:
    """Data-augmentation hyper-parameters."""

    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_range: float = 0.15
    zoom_range: float = 0.15
    brightness_range: float = 0.2
    contrast_range: float = 0.2


@dataclass
class Config:
    """Top-level project configuration."""

    # Paths
    data_dir: str = "../dataset/RGB_224x224/RGB_224x224"
    model_save_dir: str = "../models"
    log_dir: str = "outputs/logs"
    tensorboard_dir: str = "outputs/tensorboard"

    # Model
    model_type: str = "transfer"  # "custom" | "transfer"
    backbone: str = "efficientnetb3"  # "resnet50" | "efficientnetb3"

    # Image
    image_size: int = 224
    color_mode: str = "rgb"

    # Training
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    optimizer: str = "adam"
    label_smoothing: float = 0.1

    # Learning rate schedule
    lr_schedule: str = "cosine"        # "cosine" or "plateau"
    warmup_epochs: int = 3
    cosine_min_lr: float = 1e-5

    # Fine-tuning (transfer only)
    fine_tune_layers: int = 30
    fine_tune_lr: float = 0.0001
    fine_tune_epochs: int = 20

    # Augmentation
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    # Callbacks
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    reduce_lr_min: float = 1e-6

    # Export
    export_formats: List[str] = field(default_factory=lambda: ["h5", "saved_model", "onnx"])

    # Derived (set after loading)
    num_classes: int = 0

    def __post_init__(self) -> None:
        """Validate critical fields."""
        if self.model_type not in ("custom", "transfer"):
            raise ValueError(
                f"model_type must be 'custom' or 'transfer', got '{self.model_type}'"
            )
        if self.backbone not in ("resnet50", "efficientnetb0", "efficientnetb3"):
            raise ValueError(
                f"backbone must be 'resnet50', 'efficientnetb0', or 'efficientnetb3', got '{self.backbone}'"
            )
        if self.image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {self.image_size}")


def _resolve_path(path_str: str, base: Path) -> str:
    """Resolve a possibly-relative path against *base*."""
    p = Path(path_str)
    if not p.is_absolute():
        p = (base / p).resolve()
    return str(p)


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from *config_path* (default: ``config/config.yaml``).

    Relative paths inside the YAML are resolved against the ``training/``
    directory so the config works regardless of the caller's cwd.
    """
    if config_path is None:
        # Walk up from this file → src/ → training/
        training_dir = Path(__file__).resolve().parent.parent
        config_path = training_dir / "config" / "config.yaml"
    else:
        config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    training_dir = config_path.resolve().parent.parent  # config/ → training/

    with open(config_path, "r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f)

    # Resolve relative paths
    for key in ("data_dir", "model_save_dir", "log_dir", "tensorboard_dir"):
        if key in raw:
            raw[key] = _resolve_path(raw[key], training_dir)

    # Pull out nested augmentation dict
    aug_dict = raw.pop("augmentation", {})
    aug = AugmentationConfig(**aug_dict) if aug_dict else AugmentationConfig()

    cfg = Config(**raw, augmentation=aug)

    # Ensure output directories exist
    for d in (cfg.model_save_dir, cfg.log_dir, cfg.tensorboard_dir):
        os.makedirs(d, exist_ok=True)

    return cfg
