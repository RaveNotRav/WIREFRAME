"""Global state for the FastAPI app."""
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_state_lock = threading.Lock()
_model = None
_model_available = False
_config_failed = False


def set_model(model: Any):
    global _model, _model_available
    with _state_lock:
        _model = model
        _model_available = model is not None


def get_model() -> Any:
    return _model


def set_model_available(available: bool):
    global _model_available
    with _state_lock:
        _model_available = available


def is_model_available() -> bool:
    from backend.mock_detector import is_mock_mode
    if is_mock_mode():
        return True
    if not _model_available:
        return False
    return _model is not None


def load_model_once(model_path: str | None = None) -> Any:
    """Thread-safely load and cache the YOLO model once if not already loaded."""
    global _model, _model_available
    with _state_lock:
        if _model is not None:
            return _model
        resolved_path = model_path or os.environ.get("WIREFRAME_MODEL_PATH") or "backend/models/wireframe_seg.pt"
        if not os.path.isfile(resolved_path):
            _model = None
            _model_available = False
            logger.warning("Model file not found: %s. Uploads will return 503.", resolved_path)
            return None
        try:
            from ultralytics import YOLO
            loaded = YOLO(resolved_path)
            _model = loaded
            _model_available = True
            logger.info("Model loaded: %s", resolved_path)
            return _model
        except Exception as exc:
            _model = None
            _model_available = False
            logger.error("Failed to load model: %s", exc)
            return None


def set_config_failed(failed: bool):
    global _config_failed
    _config_failed = failed


def is_config_failed() -> bool:
    return _config_failed


