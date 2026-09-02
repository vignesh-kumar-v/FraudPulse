"""Uniform logging so producer/consumer/API output is readable side by side."""

from __future__ import annotations

import logging
import os

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.environ.get("FP_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=lvl,
        format="%(name)-28s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=False)],
    )
    # these are chatty and drown out our own logs
    for noisy in ("kafka", "urllib3", "botocore", "feast", "py4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
