# src/train.py
"""Main training orchestrator — run with ``python -m src.train``."""

from __future__ import annotations

import argparse
import json
import os
import sys

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

from src.config import Config, load_config
from src.dataset import load_datasets
from src.models import build_model
from src.models.transfer_cnn import unfreeze_top_layers
from src.utils.callbacks import LoguruCallback, LRSchedulerLogger
from src.utils.logger import logger


def _state_path(cfg: Config) -> str:
    """Return the path to the training state JSON file."""
    return os.path.join(cfg.model_save_dir, "training_state.json")


def _save_state(cfg: Config, phase: str, epoch: int) -> None:
    """Persist the last completed epoch so training can resume."""
    state = {"phase": phase, "last_epoch": epoch}
    with open(_state_path(cfg), "w") as f:
        json.dump(state, f)


def _load_state(cfg: Config) -> dict:
    """Load the saved training state, or return defaults."""
    path = _state_path(cfg)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"phase": "phase1_initial", "last_epoch": 0}


class _EpochStateCallback(tf.keras.callbacks.Callback):
    """Save training state after every epoch so we can resume."""

    def __init__(self, cfg: Config, phase: str):
        super().__init__()
        self._cfg = cfg
        self._phase = phase

    def on_epoch_end(self, epoch: int, logs=None):
        # epoch is 0-indexed, save as 1-indexed completed count
        _save_state(self._cfg, self._phase, epoch + 1)


def _get_callbacks(cfg: Config, phase: str = "initial") -> list:
    """Build the Keras callbacks list."""
    model_name = f"crop_cnn_{cfg.model_type}"
    if cfg.model_type == "transfer":
        model_name += f"_{cfg.backbone}"

    checkpoint_path = os.path.join(
        cfg.model_save_dir, f"{model_name}_best.h5"
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(cfg.tensorboard_dir, phase),
            histogram_freq=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(cfg.log_dir, f"training_{phase}.csv"),
            append=True,
        ),
        LoguruCallback(),
        LRSchedulerLogger(),
        _EpochStateCallback(cfg, phase),
    ]

    # Only add ReduceLROnPlateau if not using cosine schedule
    if cfg.lr_schedule != "cosine":
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg.reduce_lr_factor,
                patience=cfg.reduce_lr_patience,
                min_lr=cfg.reduce_lr_min,
                verbose=1,
            )
        )

    return callbacks


def train(cfg: Config, resume: bool = False) -> tf.keras.Model:
    """Run the full training pipeline and return the trained model."""
    # Data
    train_ds, val_ds, _test_ds, class_names = load_datasets(cfg)
    num_classes = len(class_names)
    cfg.num_classes = num_classes

    # Calculate steps per epoch for LR schedule
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()

    # Model
    model_name = f"crop_cnn_{cfg.model_type}"
    if cfg.model_type == "transfer":
        model_name += f"_{cfg.backbone}"
    checkpoint_path = os.path.join(cfg.model_save_dir, f"{model_name}_best.h5")

    initial_epoch = 0
    if resume and os.path.exists(checkpoint_path):
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        try:
            model = tf.keras.models.load_model(checkpoint_path, safe_mode=False)
        except Exception as e:
            logger.warning(f"Could not load full model config: {e}. Rebuilding architecture and loading weights only.")
            model = build_model(cfg, num_classes, steps_per_epoch=steps_per_epoch)
            model.load_weights(checkpoint_path)
        state = _load_state(cfg)
        initial_epoch = state.get("last_epoch", 0)
        logger.info(f"Resuming from epoch {initial_epoch + 1}/{cfg.epochs} (phase: {state.get('phase', 'unknown')})")
    else:
        logger.info("Building new model.")
        model = build_model(cfg, num_classes, steps_per_epoch=steps_per_epoch)
        model.summary(print_fn=logger.info)

    # Phase 1 — Initial training
    if initial_epoch < cfg.epochs:
        logger.info(f"Phase 1: Training epochs {initial_epoch + 1}-{cfg.epochs} (lr={cfg.learning_rate})")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.epochs,
            initial_epoch=initial_epoch,
            callbacks=_get_callbacks(cfg, phase="phase1_initial"),
        )
    else:
        logger.info(f"Phase 1 already complete ({initial_epoch} epochs). Skipping.")

    # Phase 2 — Fine-tuning (transfer only)
    if cfg.model_type == "transfer" and cfg.fine_tune_layers > 0:
        logger.info(
            f"Phase 2: Fine-tuning top {cfg.fine_tune_layers} backbone layers "
            f"for {cfg.fine_tune_epochs} epochs (lr={cfg.fine_tune_lr})"
        )
        unfreeze_top_layers(model, cfg.fine_tune_layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.fine_tune_lr),
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=cfg.label_smoothing
            ),
            metrics=[
                "accuracy",
                tf.keras.metrics.TopKCategoricalAccuracy(
                    k=5, name="top_5_accuracy"
                ),
            ],
        )
        ft_initial = 0
        if resume:
            state = _load_state(cfg)
            if state.get("phase") == "phase2_finetune":
                ft_initial = state.get("last_epoch", 0)
                logger.info(f"Resuming fine-tuning from epoch {ft_initial + 1}/{cfg.fine_tune_epochs}")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.fine_tune_epochs,
            initial_epoch=ft_initial,
            callbacks=_get_callbacks(cfg, phase="phase2_finetune"),
        )

    # Save final model
    model_name = f"crop_cnn_{cfg.model_type}"
    if cfg.model_type == "transfer":
        model_name += f"_{cfg.backbone}"

    final_path = os.path.join(cfg.model_save_dir, f"{model_name}.h5")
    model.save(final_path)
    logger.info(f"Final model saved -> {final_path}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN crop classifier")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the best checkpoint if it exists.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info(f"Config loaded - model_type={cfg.model_type}, backbone={cfg.backbone}")
    logger.info(f"Data dir: {cfg.data_dir}")
    logger.info(f"Model save dir: {cfg.model_save_dir}")

    train(cfg, resume=args.resume)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
