# src/utils/logger.py
"""Centralized logging with Loguru — adapted from the original project logger."""

from __future__ import annotations

import os
import sys

from loguru import logger

# ── Constants ────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "logs")

# Formatting
_FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
    "<level>{message}</level>"
)


def setup_logging(
    log_level: str = "DEBUG",
    retention: str = "1 week",
    rotation: str = "100 MB",
    log_format: str = _FMT,
) -> None:
    """Configure Loguru handlers for console + file output."""
    os.makedirs(LOG_DIR, exist_ok=True)

    # Reset existing handlers
    logger.remove()

    # Console
    logger.add(
        sys.stdout,
        level=log_level,
        format=log_format,
        backtrace=True,
        diagnose=True,
    )

    # Structured JSON log
    logger.add(
        os.path.join(LOG_DIR, "app.log"),
        level="INFO",
        serialize=True,
        rotation=rotation,
        retention=retention,
        compression="zip",
    )

    # Error-only log
    logger.add(
        os.path.join(LOG_DIR, "error.log"),
        level="ERROR",
        rotation=rotation,
        retention=retention,
        compression="zip",
        backtrace=True,
        diagnose=True,
    )


# Auto-initialise on import
setup_logging()
