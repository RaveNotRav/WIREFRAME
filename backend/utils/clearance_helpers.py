"""Shared clearance and calibration recompute helpers used by pipeline and analysis routes."""

import ast
import json
import logging
from typing import Any

from backend.config_store import get_setting
from backend.formulas import (
    compute_clearance_status,
    compute_crs_band,
    compute_safety_level,
)
from backend.utils.cv.types import normalize_class_name
from backend.utils.cv.calibration import estimate_clearance

logger = logging.getLogger(__name__)


def parse_params_json(val: str | dict | None) -> dict[str, Any]:
    """Safely parse wire params_json, supporting JSON and legacy str(dict).
    
    Malicious code strings are rejected and never executed.
    """
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if not isinstance(val, str) or not val.strip():
        return {}
    try:
        res = json.loads(val)
        if isinstance(res, dict):
            return res
    except Exception:
        pass
    try:
        res = ast.literal_eval(val)
        if isinstance(res, dict):
            return res
    except Exception:
        pass
    return {}


def parse_centerline_json(val: str | list | None) -> list[Any]:
    """Safely parse wire centerline_json, supporting JSON and legacy str(list)."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if not isinstance(val, str) or not val.strip():
        return []
    try:
        res = json.loads(val)
        if isinstance(res, list):
            return res
    except Exception:
        pass
    try:
        res = ast.literal_eval(val)
        if isinstance(res, list):
            return res
    except Exception:
        pass
    return []


def get_lowest_y(polyline: list[Any]) -> float:
    """Return the maximum y coordinate (lowest vertical point) in image space."""
    if not polyline:
        return 0.0
    try:
        return float(max(pt[1] for pt in polyline))
    except Exception:
        return 0.0


def evaluate_wire(
    class_name: str,
    confidence: float,
    sag_px: float,
    span_px: float,
    crs_weighted: float,
    lowest_y: float,
    pole_bbox: list[float] | None,
    meters_per_px: float | None,
    image_height: int | None,
    pec_min_map: dict[str, float],
    params_dict: dict[str, Any],
) -> dict[str, Any]:
    """Pure per-wire evaluation of clearance, status, and safety level."""
    clear_est = estimate_clearance(lowest_y, pole_bbox, meters_per_px, image_height)

    updated_params = dict(params_dict)
    updated_params["clearance_low_m"] = clear_est["clearance_low_m"]
    updated_params["clearance_high_m"] = clear_est["clearance_high_m"]

    cls_key = normalize_class_name(class_name)
    pec_min = pec_min_map.get(cls_key, pec_min_map.get("default", 5.5))

    if clear_est["status"] != "unavailable":
        clearance_status = compute_clearance_status(
            clear_est["clearance_m"],
            clear_est["clearance_low_m"],
            clear_est["clearance_high_m"],
            pec_min,
        )
        est_str = (
            f"Estimated clearance {clear_est['clearance_m']:.2f} m "
            f"(range {clear_est['clearance_low_m']:.2f}-{clear_est['clearance_high_m']:.2f} m)"
        )
    else:
        clearance_status = "unavailable"
        est_str = "Clearance unavailable."

    band = compute_crs_band(crs_weighted)
    safety = compute_safety_level(band, clearance_status, est_str)
    wire_safety = safety["safety_level"]

    sag_m = sag_px * meters_per_px if meters_per_px is not None and sag_px is not None else None

    return {
        "class_name": class_name,
        "confidence": confidence,
        "sag_px": sag_px,
        "span_px": span_px,
        "crs": crs_weighted,
        "band": band,
        "sag_m": sag_m,
        "clearance_m": clear_est["clearance_m"],
        "clearance_low_m": clear_est["clearance_low_m"],
        "clearance_high_m": clear_est["clearance_high_m"],
        "clearance_status": clearance_status,
        "safety_level": wire_safety,
        "params_dict": updated_params,
        "params_json": json.dumps(updated_params),
    }


def rollup_analysis_clearance(
    evaluated_wires: list[dict[str, Any]],
    pec_min_map: dict[str, float],
    margin_pct: float,
) -> dict[str, Any]:
    """Roll up per-wire clearance metrics into analysis-level summary fields."""
    valid_clearances = [
        fw for fw in evaluated_wires
        if fw.get("clearance_m") is not None and fw.get("clearance_status") in ("ok", "below", "marginal")
    ]

    if valid_clearances:
        lowest_wire = min(valid_clearances, key=lambda w: w["clearance_m"])
        clearance_available = lowest_wire["clearance_m"]
        status_str = lowest_wire.get("clearance_status")
        if status_str == "ok":
            clearance_ok = 1
        elif status_str == "marginal":
            clearance_ok = 2
        elif status_str == "below":
            clearance_ok = 0
        else:
            clearance_ok = None

        cls_key = normalize_class_name(lowest_wire["class_name"])
        pec_min_clearance_m = pec_min_map.get(cls_key, pec_min_map.get("default", 5.5))

        sags_m = [fw["sag_m"] for fw in evaluated_wires if fw.get("sag_m") is not None]
        sag_meters = max(sags_m) if sags_m else None
        margin_error_m = clearance_available * margin_pct if clearance_available is not None else None
    else:
        clearance_ok = None
        clearance_available = None
        pec_min_clearance_m = None
        sag_meters = None
        margin_error_m = None
        lowest_wire = None

    return {
        "clearance_available": clearance_available,
        "clearance_ok": clearance_ok,
        "pec_min_clearance_m": pec_min_clearance_m,
        "sag_meters": sag_meters,
        "margin_error_m": margin_error_m,
        "lowest_wire": lowest_wire,
    }


def determine_overall_safety(
    wire_safeties: list[str],
    max_wire_band: str | None = None,
    analysis_band: str | None = None,
    clearance_status: str | None = None,
) -> str:
    """Worst-of overall safety level determination.

    Severity order: Critical > Unsafe > Caution > Safe.
    Applies a wire-based floor from max_wire_band (Critical -> Critical, High -> at least Unsafe)
    and optionally combines clearance_status x analysis_band when provided.
    "Safe" is returned only when every wire is "Safe" and no escalation applies.
    """
    base = "Unknown"
    if wire_safeties:
        if "Critical" in wire_safeties:
            base = "Critical"
        elif "Unsafe" in wire_safeties:
            base = "Unsafe"
        elif "Caution" in wire_safeties:
            base = "Caution"
        elif all(s == "Safe" for s in wire_safeties):
            base = "Safe"

    rank_map = {"Unknown": 0, "Incomplete": 0, "Safe": 1, "Caution": 2, "Unsafe": 3, "Critical": 4}
    inv_map = {1: "Safe", 2: "Caution", 3: "Unsafe", 4: "Critical"}

    current_rank = rank_map.get(base, 0)

    # Optional clearance x analysis_band matrix evaluation
    if clearance_status or analysis_band:
        clr_rank = 0
        if clearance_status in ("below", 0):
            clr_rank = 4
        elif clearance_status in ("near", "marginal", 2):
            clr_rank = 2
        elif clearance_status in ("above", "ok", 1):
            clr_rank = 1

        band_rank = 0
        if analysis_band == "Critical":
            band_rank = 4
        elif analysis_band == "High":
            band_rank = 3
        elif analysis_band == "Moderate":
            band_rank = 2
        elif analysis_band == "Low":
            band_rank = 1

        if clearance_status in ("no_pole", "no_wires"):
            matrix_rank = band_rank if band_rank >= 3 else 0
        elif clearance_status in ("unavailable", None) and not wire_safeties:
            matrix_rank = band_rank if band_rank >= 2 else 0
        else:
            matrix_rank = max(clr_rank, band_rank)

        current_rank = max(current_rank, matrix_rank)

    # Wire-based floor from max_wire_band: Critical -> Critical (4); High -> at least Unsafe (3)
    if max_wire_band == "Critical":
        current_rank = max(current_rank, 4)
    elif max_wire_band == "High":
        current_rank = max(current_rank, 3)

    if current_rank >= 1:
        return inv_map[current_rank]
    return base
