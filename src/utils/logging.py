"""Project-wide logging configuration using Loguru.

This module centralizes logger configuration to ensure consistent formatting
and levels across the project. Call `setup_logging` once at app start.
"""

from __future__ import annotations

import sys
from typing import Optional

from loguru import logger


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure the global Loguru logger.

    Parameters
    ----------
    log_level: str
        Minimum level to output (e.g., "DEBUG", "INFO", "WARNING").
    log_file: Optional[str]
        Optional file path to also write logs to. If provided, a rotating file
        sink is configured in addition to stderr.
    """

    # Remove default handlers to avoid duplicate logs if reconfigured
    logger.remove()

    # Console sink
    logger.add(
        sys.stderr,
        level=log_level,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
    )

    # Optional file sink
    if log_file:
        logger.add(
            log_file,
            rotation="10 MB",
            retention="7 days",
            level=log_level,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
        )


__all__ = ["setup_logging", "logger"]
