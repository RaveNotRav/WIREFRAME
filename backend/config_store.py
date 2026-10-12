"""Configuration store backed by SQLite settings table with in-memory cache."""

import json
import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

_cache: dict[str, Any] = {}
_conn: sqlite3.Connection | None = None

DEFAULT_SETTINGS: dict[str, Any] = {
    # PROVISIONAL: values must be recalibrated against measured sag_to_span field data with adviser agreement
    "crs_params": {
        "sag_offset": 0.03,
        "sag_scale": 0.17,
        "residual_offset": 2.0,
        "residual_gain": 5.0,
        "penalty_cap": 20,
    },
    "site_risk_weights": {
        "vni": 0.45,
        "crs": 0.45,
        "dead": 0.10,
    },
    "risk_bands": [
        {"min": 0.00, "max": 0.33, "label": "Low"},
        {"min": 0.34, "max": 0.66, "label": "Moderate"},
        {"min": 0.67, "max": 0.85, "label": "High"},
        {"min": 0.86, "max": 1.00, "label": "Critical"},
    ],
    "class_weights": {
        "power_line": 1.0,
        "telecom_cable": 1.0,
        "fiber_optic": 1.0,
        "dead_abandoned": 1.0,
    },
    "pole_heights": [9, 10, 12],
    "pec_min_clearance_by_class": {
        "power_line": 5.5,
        "telecom_cable": 5.5,
        "fiber_optic": 5.5,
        "dead_abandoned": 5.5,
        "default": 5.5
    },
    "margin_of_error_pct": 20,
    "model_path": "backend/models/wireframe_seg.pt",
    "inference_params": {
        "conf": 0.25,
        "iou": 0.7,
        "imgsz": 1024,
        "retina_masks": True,
    },
    "severity_weights": {
        "high": 1.0,
        "medium": 0.5,
        "low": 0.1,
    },
    "severity_thresholds": {
        "high_confidence_min": 0.7,
        "medium_confidence_min": 0.4,
    },
    "safety_mapping": {
        "crs_rank": {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3, "unavailable": 0},
        "clearance_rank": {"ok": 0, "marginal": 1, "below": 2, "unavailable": 0},
        "rank_to_level": {"0": "Safe", "1": "Caution", "2": "Unsafe", "3": "Critical"}
    },
    "action_mapping": {
        "low": "No immediate action required. Routine inspection.",
        "high": "Schedule maintenance soon to resolve clearance or sag issues.",
        "critical": "Immediate dispatch required. Clear hazard present.",
        "neutral": "Review data. Unable to confidently clear site.",
        "no_wires": "No wires detected. Site clear."
    }
}


def init_config_store(conn: sqlite3.Connection) -> None:
    """Initialize config store: store connection, seed defaults, load cache."""
    global _conn, _cache
    _conn = conn
    _cache = {}
    _seed_defaults()
    _load_cache()
    logger.info("Config store initialized with %d settings", len(_cache))


def _seed_defaults() -> None:
    if _conn is None:
        raise RuntimeError("Config store not initialized")
    for key, value in DEFAULT_SETTINGS.items():
        exists = _conn.execute(
            "SELECT 1 FROM settings WHERE key = ?", (key,)
        ).fetchone()
        if exists is None:
            _conn.execute(
                "INSERT INTO settings (key, value_json, updated_by) VALUES (?, ?, ?)",
                (key, json.dumps(value), "system"),
            )
            logger.debug("Seeded default: %s", key)
    _conn.commit()


def _load_cache() -> None:
    if _conn is None:
        raise RuntimeError("Config store not initialized")
    _cache.clear()
    for row in _conn.execute("SELECT key, value_json FROM settings").fetchall():
        _cache[row["key"]] = json.loads(row["value_json"])


def get_setting(key: str, default: Any = None) -> Any:
    """Get a setting value from cache, falling back to DB then defaults."""
    if key in _cache:
        return _cache[key]
    if _conn is not None:
        row = _conn.execute(
            "SELECT value_json FROM settings WHERE key = ?", (key,)
        ).fetchone()
        if row is not None:
            value = json.loads(row["value_json"])
            _cache[key] = value
            return value
    return DEFAULT_SETTINGS.get(key, default)


def set_setting(key: str, value: Any, updated_by: str = "system") -> None:
    """Persist a setting and update the cache."""
    if _conn is None:
        raise RuntimeError("Config store not initialized")
    _conn.execute(
        """INSERT INTO settings (key, value_json, updated_at, updated_by)
           VALUES (?, ?, datetime('now'), ?)
           ON CONFLICT(key) DO UPDATE SET
               value_json = excluded.value_json,
               updated_at = excluded.updated_at,
               updated_by = excluded.updated_by""",
        (key, json.dumps(value), updated_by),
    )
    _conn.commit()
    _cache[key] = value
    logger.info("Setting updated: %s (by %s)", key, updated_by)


def reset_config_store() -> None:
    """Reset module state. For testing only."""
    global _conn, _cache
    _conn = None
    _cache = {}


def clear_cache() -> None:
    """Clear in-memory cache, forcing next read from DB."""
    _cache.clear()
