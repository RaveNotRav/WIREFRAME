"""Centralized logging configuration for WIREFRAME."""

import logging
import os
import sys


def setup_logging() -> None:
    """Configure logging. Reads WIREFRAME_LOG_LEVEL from env (default: INFO)."""
    level_name = os.environ.get("WIREFRAME_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
