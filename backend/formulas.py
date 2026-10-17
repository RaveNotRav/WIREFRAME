"""Pure computation functions for WIREFRAME risk assessment.

All thresholds read from config_store.
    TODO(THESIS-3.5) - formula constants from thesis Section 3.5
    TODO(VERIFY-WITH-ADVISER) - values needing adviser confirmation
"""

import logging
from typing import Any

from backend.config_store import get_setting

logger = logging.getLogger(__name__)


def compute_catenary_risk(sag_to_span: float, residual: float) -> float:
    """Catenary Risk Score (CRS), 0-100.

    CRS = clip(sag_term + residual_penalty, 0, 100)
      sag_term     = clip((sag_to_span - offset) / scale * 100, 0, 100)
      residual_pen = clip((residual - r_offset) * r_gain, 0, penalty_cap)

    TODO(THESIS-3.5): Verify constants with adviser.
    """
    p = get_setting("crs_params")
    sag_term = max(0.0, min(100.0,
        (sag_to_span - p["sag_offset"]) / p["sag_scale"] * 100))
    residual_penalty = max(0.0, min(float(p["penalty_cap"]),
        (residual - p["residual_offset"]) * p["residual_gain"]))
    crs = max(0.0, min(100.0, sag_term + residual_penalty))
    logger.debug("CRS: sag_to_span=%.4f term=%.2f residual=%.4f pen=%.2f -> %.2f",
                 sag_to_span, sag_term, residual, residual_penalty, crs)
    return crs


def compute_barangay_risk(
    vni: float, max_crs: float, dead_wire_count: int, wire_count: int,
) -> float:
    """Per-image barangay risk, 0-100.

    risk = clip(w_vni*VNI*100 + w_crs*max_CRS + w_dead*density*100, 0, 100)
    density = clip(dead_wire_count / max(1, wire_count), 0, 1)
    """
    w = get_setting("site_risk_weights")
    density = min(1.0, max(0.0, dead_wire_count / max(1, wire_count)))
    risk = w["vni"] * vni * 100 + w["crs"] * max_crs + w["dead"] * density * 100
    risk = max(0.0, min(100.0, risk))
    logger.debug("Barangay risk: vni=%.4f crs=%.2f dead=%d/%d -> %.2f",
                 vni, max_crs, dead_wire_count, wire_count, risk)
    return risk


def get_severity(confidence: float) -> str:
    """Map confidence to severity (INVERTED: high confidence = low risk).

    TODO(THESIS-3.5): Verify threshold boundaries.
    """
    t = get_setting("severity_thresholds")
    if confidence >= t["high_confidence_min"]:
        return "low"
    if confidence >= t["medium_confidence_min"]:
        return "medium"
    return "high"


def get_severity_weight(severity: str) -> float:
    """Numeric weight for a severity label."""
    return get_setting("severity_weights").get(severity, 0.5)


def compute_detection_risk(severities: list[str]) -> float:
    """Detection risk from severity labels, 0-100.

    detection_risk = mean(weight per severity) * 100
    """
    if not severities:
        return 0.0
    sw = get_setting("severity_weights")
    total = sum(sw.get(s, 0.5) for s in severities)
    return max(0.0, min(100.0, (total / len(severities)) * 100))


def compute_risk_score(detection_risk: float, barangay_risk: float) -> float:
    """Final risk score = 0.5*detection + 0.5*barangay, clipped 0-100."""
    return max(0.0, min(100.0, 0.5 * detection_risk + 0.5 * barangay_risk))


def compute_crs_band(crs_raw: float) -> str:
    """Convert raw 0-100 CRS to band (Low, Moderate, High, Critical).
    Reads thresholds from risk_bands DB config.
    Returns 'unavailable' if crs_raw is None or negative.
    """
    if crs_raw is None or crs_raw < 0:
        return "unavailable"
        
    val = round(crs_raw, 2) / 100.0
    val = max(0.0, min(1.0, val))
    
    bands = get_setting("risk_bands")
    for b in bands:
        if val <= b["max"]:
            return b["label"]
            
    return bands[-1]["label"] if bands else "Unknown"


def compute_clearance_status(
    clearance_m: float, 
    low_m: float, 
    high_m: float, 
    pec_min: float
) -> str:
    """Determine clearance status against PEC minimum.
    
    lower >= min -> ok
    upper < min -> below
    straddling -> marginal
    """
    if clearance_m is None or low_m is None or high_m is None:
        return "unavailable"
        
    if low_m >= pec_min:
        return "ok"
    elif high_m < pec_min:
        return "below"
    return "marginal"


def compute_safety_level(crs_band: str, clearance_status: str, clearance_estimate_str: str) -> dict:
    """Determine safety level and plain language reasons using worst-of logic.
    
    Returns dict with 'safety_level' and 'reasons'.
    """
    mapping = get_setting("safety_mapping")
    crs_rank = mapping["crs_rank"].get(crs_band, 0)
    clear_rank = mapping["clearance_rank"].get(clearance_status, 0)
    
    reasons = []
    
    if crs_band == "unavailable" and clearance_status == "unavailable":
        safety_level = "Unknown"
        reasons.append("Both CRS and clearance are unavailable. Cannot determine safety.")
    elif clearance_status == "unavailable":
        # Caps based on CRS only, but cannot confirm safe without clearance
        rank = crs_rank
        level = mapping["rank_to_level"].get(str(rank), "Unknown")
        if level == "Safe":
            safety_level = "Unknown"
            reasons.append("Clearance could not be estimated; cannot confirm safe.")
        else:
            safety_level = level
            reasons.append("Clearance could not be estimated; verdict is based on CRS only.")
        reasons.append(f"CRS band is {crs_band}.")
    elif crs_band == "unavailable":
        # Clearance alone caps at Unsafe (rank 2)
        rank = min(2, clear_rank)
        safety_level = mapping["rank_to_level"].get(str(rank), "Unknown")
        reasons.append("CRS could not be estimated; verdict is based on clearance only.")
        reasons.append(clearance_estimate_str)
    else:
        rank = max(crs_rank, clear_rank)
        # If clearance drives it, but clearance rank is 3? Clearance rank only goes up to 2 (below)
        safety_level = mapping["rank_to_level"].get(str(rank), "Unknown")
        reasons.append(f"CRS band is {crs_band}.")
        reasons.append(clearance_estimate_str)
        
    return {
        "safety_level": safety_level,
        "reasons": reasons
    }


def compute_metric_clearance(
    sag_px: float,
    pole_bbox_height_px: float,
    assumed_pole_height_m: float,
) -> dict[str, Any]:
    """Deprecated thin wrapper over new calibration logic for backwards compatibility.
    
    TODO(VERIFY-WITH-ADVISER): Proper clearance checks bottom of wire to ground. 
    This legacy function compares sag directly to the ground clearance threshold!
    """
    pec_min_map = get_setting("pec_min_clearance_by_class", {})
    pec_min = pec_min_map.get("default", 5.5)
    margin: float = get_setting("margin_of_error_pct")
    
    from backend.utils.cv.calibration import calibrate_scale
    
    if pole_bbox_height_px <= 0:
        logger.warning("Invalid pole bbox height: %.2f px", pole_bbox_height_px)
        return {
            "sag_meters": 0.0,
            "clearance_ok": False,
            "clearance_available": 0.0,
            "pec_min_clearance_m": pec_min,
            "assumed_pole_height_m": assumed_pole_height_m,
            "meters_per_px": 0.0,
        }
        
    scale_res = calibrate_scale(pole_bbox=[0, 0, 0, pole_bbox_height_px], image_height=9999)
    meters_per_px = scale_res.meters_per_px or 0.0
    
    sag_meters = sag_px * meters_per_px
    clearance_available = sag_meters
    worst_case = sag_meters * (1.0 + margin / 100.0)
    clearance_ok = worst_case <= pec_min
    
    return {
        "sag_meters": round(sag_meters, 4),
        "clearance_ok": clearance_ok,
        "clearance_available": round(clearance_available, 4),
        "pec_min_clearance_m": pec_min,
        "assumed_pole_height_m": assumed_pole_height_m,
        "meters_per_px": round(meters_per_px, 6),
    }

def rank_sites(sites_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank sites based on max image score, with mean score as a tie-breaker.
    
    TODO(THESIS-3.5): Verify site aggregation formula.
    """
    def site_key(site):
        # We assume site data has a list of image_scores
        scores = site.get("image_scores", [0.0])
        if not scores:
            return (0.0, 0.0)
        max_score = max(scores)
        mean_score = sum(scores) / len(scores)
        return (max_score, mean_score)
        
    return sorted(sites_data, key=site_key, reverse=True)
