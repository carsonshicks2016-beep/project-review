"""Single logging helper used across the project."""

from __future__ import annotations

import logging

from . import config

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a timestamped logger at the configured level.

    Idempotent: the root handler is configured once, on first call.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
            format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
