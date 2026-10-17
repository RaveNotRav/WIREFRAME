"""Global state for the FastAPI app."""
from typing import Any

_model = None
_model_available = False
_config_failed = False

def set_model(model: Any):
    global _model, _model_available
    _model = model
    _model_available = model is not None

def get_model() -> Any:
    return _model

def set_model_available(available: bool):
    global _model_available
    _model_available = available

def is_model_available() -> bool:
    from backend.mock_detector import is_mock_mode
    if is_mock_mode():
        return True
    if not _model_available:
        return False
    return _model is not None

def set_config_failed(failed: bool):
    global _config_failed
    _config_failed = failed

def is_config_failed() -> bool:
    return _config_failed

