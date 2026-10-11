"""Global state for the FastAPI app."""
from typing import Any

_model = None
_config_failed = False

def set_model(model: Any):
    global _model
    _model = model

def get_model() -> Any:
    return _model

def set_config_failed(failed: bool):
    global _config_failed
    _config_failed = failed

def is_config_failed() -> bool:
    return _config_failed
