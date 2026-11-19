"""Configuration store backed by SQLite settings table with in-memory cache."""

import json
import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

_cache: dict[str, Any] = {}
_conn: sqlite3.Connection | None = None

NO_WIRES_ACTION_TEXT = "No wires were detected in this image. This is not a clearance or safety certification. Retake the photo if wires are visible."

DEFAULT_SETTINGS: dict[str, Any] = {
    # PROVISIONAL: values must be recalibrated against measured sag_to_span field data with adviser agreement
    "crs_params": {
        "sag_offset": 0.03,
        "sag_scale": 0.17,
        "residual_offset": 2.0,
        "residual_gain": 5.0,
        "penalty_cap": 20,
    },
    "vni_params": {
        "size": 640,
        "gaussian_kernel": 5,
        "gaussian_sigma": 1.0,
        "radius_factor": 0.15,
        "remove_mean": True,
        "q01": 0.034001,
        "q99": 0.243397,
        "bands": {
            "Low": 0.25,
            "Moderate": 0.50,
            "High": 0.75,
            "Severe": 1.00,
        },
    },
    "paper_crs_params": {
        "delta_lim0": {
            "power_distribution_line": 0.030,
            "communication_cable": 0.040,
            "dead_wire": 0.025,
            "power_line": 0.030,
            "telecom_cable": 0.040,
            "fiber_optic": 0.040,
            "dead_abandoned": 0.025,
            "default": 0.040,
        },
        "low_below": 3.0,
        "high_above": 6.0,
    },
    "mspi_weights": {
        "alpha": 0.3333333333333333,
        "beta": 0.3333333333333333,
        "gamma": 0.3333333333333334,
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
        "power_distribution_line": 1.0,
        "communication_cable": 1.0,
        "dead_wire": 1.0,
        "power_line": 1.0,
        "telecom_cable": 1.0,
        "fiber_optic": 1.0,
        "dead_abandoned": 1.0,
    },
    "pole_heights": [9, 10, 12],
    "pec_min_clearance_by_class": {
        "power_distribution_line": 5.5,
        "communication_cable": 5.5,
        "dead_wire": 5.5,
        "power_line": 5.5,
        "telecom_cable": 5.5,
        "fiber_optic": 5.5,
        "dead_abandoned": 5.5,
        "default": 5.5
    },
    "margin_of_error_pct": 20,
    "model_path": "backend/models/wireframe_seg.pt",
    "class_aliases": {
        "utility_pole": "pole",
        "pole_utility": "pole",
        "electric_pole": "pole",
        "power_pole": "pole",
        "wooden_pole": "pole",
        "concrete_pole": "pole",
        "utility_post": "pole",
        "post": "pole",
        "power_distribution_line": "power_distribution_line",
        "power-distribution-lines": "power_distribution_line",
        "power-lines": "power_distribution_line",
        "power_line": "power_distribution_line",
        "power-line": "power_distribution_line",
        "power line": "power_distribution_line",
        "power": "power_distribution_line",
        "powerline": "power_distribution_line",
        "power_cable": "power_distribution_line",
        "power_wire": "power_distribution_line",
        "electric_line": "power_distribution_line",
        "electric_wire": "power_distribution_line",
        "electrical_line": "power_distribution_line",
        "electrical_wire": "power_distribution_line",
        "high_voltage": "power_distribution_line",
        "low_voltage": "power_distribution_line",
        "primary_line": "power_distribution_line",
        "secondary_line": "power_distribution_line",
        "service_drop": "power_distribution_line",
        "communication_cable": "communication_cable",
        "communication_wire": "communication_cable",
        "telecom_cable": "communication_cable",
        "telecom-wires": "communication_cable",
        "telecom-cable": "communication_cable",
        "telecom cable": "communication_cable",
        "telecom": "communication_cable",
        "telco": "communication_cable",
        "telco_cable": "communication_cable",
        "telecom_wire": "communication_cable",
        "communications_cable": "communication_cable",
        "coax": "communication_cable",
        "coaxial": "communication_cable",
        "coaxial_cable": "communication_cable",
        "catv": "communication_cable",
        "telephone_line": "communication_cable",
        "fiber_optic": "communication_cable",
        "fiber_optic_drop": "communication_cable",
        "fiber-optic": "communication_cable",
        "fiber optic": "communication_cable",
        "fiber": "communication_cable",
        "fibre": "communication_cable",
        "fiber_cable": "communication_cable",
        "fiber_optic_cable": "communication_cable",
        "optical_fiber": "communication_cable",
        "fo_cable": "communication_cable",
        "dead_wire": "dead_wire",
        "dead_abandoned": "dead_wire",
        "dead-abandoned": "dead_wire",
        "dead abandoned": "dead_wire",
        "dead": "dead_wire",
        "dead_cable": "dead_wire",
        "abandoned": "dead_wire",
        "abandoned_wire": "dead_wire",
        "abandoned-wire": "dead_wire",
        "abandoned_cable": "dead_wire",
        "unused_wire": "dead_wire",
        "dangling_wire": "dead_wire",
        "vegetation": "vegetation",
        "tree": "vegetation",
        "trees": "vegetation",
        "foliage": "vegetation",
        "branch": "vegetation",
        "plants": "vegetation",
    },
    "inference_params": {
        "conf": 0.02,
        "iou": 0.70,
        "imgsz": 1024,
        "retina_masks": True,
        "device": "auto",
        "tiled": False,
        "tile_size": 640,
        "tile_overlap": 0.3,
        "tile_iou": 0.9,
        "max_det": 1000,
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
        "no_wires": NO_WIRES_ACTION_TEXT
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
        row = _conn.execute(
            "SELECT value_json, updated_by FROM settings WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            _conn.execute(
                "INSERT INTO settings (key, value_json, updated_by) VALUES (?, ?, ?)",
                (key, json.dumps(value), "system"),
            )
            logger.debug("Seeded default: %s", key)
        elif key == "inference_params" and row["updated_by"] == "system":
            # Migrate stale system-seeded defaults (e.g. conf=0.25/0.08, imgsz=640, tiled=True)
            try:
                existing = json.loads(row["value_json"]) if row["value_json"] else {}
            except Exception:
                existing = {}
            if isinstance(existing, dict) and existing != value:
                _conn.execute(
                    "UPDATE settings SET value_json = ?, updated_at = datetime('now') WHERE key = ?",
                    (json.dumps(value), key),
                )
                logger.info("Migrated system default for setting: %s", key)
    _conn.commit()


def _load_cache() -> None:
    if _conn is None:
        raise RuntimeError("Config store not initialized")
    _cache.clear()
    for row in _conn.execute("SELECT key, value_json FROM settings").fetchall():
        _cache[row["key"]] = json.loads(row["value_json"])


def _apply_env_overrides(key: str, value: Any) -> Any:
    """Apply explicit environment variable overrides so env vars always take precedence over SQLite settings."""
    if key != "inference_params" or not isinstance(value, dict):
        return value
    import os
    merged = dict(DEFAULT_SETTINGS["inference_params"])
    merged.update(value)

    env_conf = os.getenv("WIREFRAME_CONF")
    if env_conf is not None and env_conf.strip() != "":
        try:
            merged["conf"] = float(env_conf)
        except (ValueError, TypeError):
            pass

    env_iou = os.getenv("WIREFRAME_IOU")
    if env_iou is not None and env_iou.strip() != "":
        try:
            merged["iou"] = float(env_iou)
        except (ValueError, TypeError):
            pass

    env_imgsz = os.getenv("WIREFRAME_IMGSZ")
    if env_imgsz is not None and env_imgsz.strip() != "":
        try:
            merged["imgsz"] = int(env_imgsz)
        except (ValueError, TypeError):
            pass

    env_device = os.getenv("WIREFRAME_DEVICE")
    if env_device is not None and env_device.strip() != "":
        merged["device"] = env_device.strip()

    env_tiled = os.getenv("WIREFRAME_TILED")
    if env_tiled is not None and env_tiled.strip() != "":
        merged["tiled"] = env_tiled.strip().lower() in ("1", "true", "yes")

    return merged


def get_setting(key: str, default: Any = None) -> Any:
    """Get a setting value from cache, falling back to DB then defaults."""
    if key in _cache:
        return _apply_env_overrides(key, _cache[key])
    if _conn is not None:
        row = _conn.execute(
            "SELECT value_json FROM settings WHERE key = ?", (key,)
        ).fetchone()
        if row is not None:
            value = json.loads(row["value_json"])
            _cache[key] = value
            return _apply_env_overrides(key, value)
    return _apply_env_overrides(key, DEFAULT_SETTINGS.get(key, default))


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
