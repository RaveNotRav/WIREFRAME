"""Pure computation functions for WIREFRAME risk assessment.

All thresholds read from config_store.
Includes both the 0-100 UI gauge formulas and the thesis-paper-conformant
Equation 9 (allowable sag), Equation 10 (0-10 CRS), and Equation 11 (MSPI + 66-point sensitivity grid).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.stats import spearmanr

from backend.config_store import get_setting
from backend.utils.cv.types import normalize_class_name, normalize_wire_class

logger = logging.getLogger(__name__)

# Illustrative conductor reference properties (Section 3.6.5 / Table 3.8)
_DEFAULT_CONDUCTORS: dict[str, dict[str, float]] = {
    "power_distribution_line": {
        "diameter_m": 0.0101,
        "area_m2": 6.32e-5,
        "weight_n_per_m": 2.11,
        "elastic_modulus_pa": 6.8e10,
        "thermal_expansion_per_c": 1.91e-5,
        "reference_tension_n": 3600.0,
    },
    "communication_cable": {
        "diameter_m": 0.0140,
        "area_m2": 1.10e-4,
        "weight_n_per_m": 1.47,
        "elastic_modulus_pa": 1.2e10,
        "thermal_expansion_per_c": 2.0e-5,
        "reference_tension_n": 2200.0,
    },
    "dead_wire": {
        "diameter_m": 0.0080,
        "area_m2": 5.0e-5,
        "weight_n_per_m": 1.20,
        "elastic_modulus_pa": 1.2e10,
        "thermal_expansion_per_c": 2.0e-5,
        "reference_tension_n": 1500.0,
    },
}


def compute_catenary_risk(sag_to_span: float, residual: float) -> float:
    """Catenary Risk Score (CRS), 0-100.

    CRS = clip(sag_term + residual_penalty, 0, 100)
      sag_term     = clip((sag_to_span - offset) / scale * 100, 0, 100)
      residual_pen = clip((residual - r_offset) * r_gain, 0, penalty_cap)
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


def _change_of_state_tension(
    h1: float,
    w1: float,
    t1: float,
    w2: float,
    t2: float,
    span: float,
    e_mod: float,
    area: float,
    alpha: float,
) -> float:
    """Positive real root H2 of the cubic change-of-state equation (Kiessling et al., 2003)."""
    ea = e_mod * area
    k = h1 - ea * (w1**2) * (span**2) / (24.0 * (h1**2)) - ea * alpha * (t2 - t1)
    rhs = ea * (w2**2) * (span**2) / 24.0

    def f(h: float) -> float:
        return h * h * (h - k) - rhs

    lo = max(k, 0.0)
    hi = max(lo, 1.0)
    while f(hi) < 0:
        hi *= 2.0
        if hi > 1e12:
            raise ValueError("change-of-state root not bracketed")
    return float(brentq(f, lo if f(lo) < 0 else 1e-9, hi, xtol=1e-9, rtol=1e-12))


def compute_allowable_sag(
    class_name: str,
    span_m: float | None = None,
    context: dict[str, Any] | None = None,
) -> tuple[float, list[str]]:
    """Equation 9 / Table 3.8 allowable sag ratio delta_lim.

    delta_lim = delta_lim,0 * (delta_exp / delta_exp,0)

    When span_m or engineering context is not provided, returns delta_lim,0
    and includes the `"default_parameters"` flag.
    """
    paper_cfg = get_setting("paper_crs_params", {}) or {}
    d0_map = paper_cfg.get("delta_lim0", {}) if isinstance(paper_cfg, dict) else {}
    norm_cls = normalize_class_name(class_name)
    canon_cls = normalize_wire_class(norm_cls)
    d0 = float(
        d0_map.get(
            canon_cls,
            d0_map.get(norm_cls, d0_map.get("default", 0.04)),
        )
    )

    flags: list[str] = []
    ctx = context or {}
    effective_span = span_m if span_m is not None else ctx.get("span_m")
    if effective_span is None or float(effective_span) <= 0:
        flags.append("default_parameters")
        return d0, flags

    span = float(effective_span)
    cond = _DEFAULT_CONDUCTORS.get(canon_cls, _DEFAULT_CONDUCTORS["communication_cable"])
    t_ref = 25.0
    temp_c = float(ctx["temperature_c"]) if ctx.get("temperature_c") is not None else t_ref
    wind_mps = float(ctx["wind_speed_mps"]) if ctx.get("wind_speed_mps") is not None else 0.0
    if ctx.get("temperature_c") is None:
        flags.append("default_temperature")
    if ctx.get("wind_speed_mps") is None:
        flags.append("default_wind")

    w1 = float(ctx.get("weight_n_per_m") or cond["weight_n_per_m"])
    e_mod = float(ctx.get("elastic_modulus_pa") or cond["elastic_modulus_pa"])
    area = cond["area_m2"]
    alpha = cond["thermal_expansion_per_c"]
    h1 = cond["reference_tension_n"]
    diam = cond["diameter_m"]

    # Reference state sag ratio
    d_exp0 = (h1 / w1) * (math.cosh(w1 * span / (2.0 * h1)) - 1.0) / span
    if d_exp0 <= 0:
        flags.append("default_parameters")
        return d0, flags

    # Capture state with wind & temperature
    rho_air = 1.225
    cd = 1.0
    w_wind = 0.5 * rho_air * (wind_mps**2) * cd * diam
    w_res = math.hypot(w1, w_wind)
    try:
        h2 = _change_of_state_tension(h1, w1, t_ref, w_res, temp_c, span, e_mod, area, alpha)
        d_exp = (h2 / w_res) * (math.cosh(w_res * span / (2.0 * h2)) - 1.0) / span
        delta_lim = float(d0 * (d_exp / d_exp0))
        return delta_lim, flags
    except Exception:
        flags.append("default_parameters")
        return d0, flags


def compute_paper_crs(
    sag_to_span: float,
    delta_lim: float = 0.04,
    class_weight: float = 1.0,
    *,
    fit_failed: bool = False,
) -> tuple[float | None, str | None]:
    """Equation 10 & Table 1.2 paper-conformant 0-10 Catenary Risk Score and risk level.

        CRS = min{10, 10 * (delta_rel / delta_lim) * w_class}

        Low (green)     if CRS < 3.0
        Medium (yellow) if 3.0 <= CRS <= 6.0
        High (red)      if CRS > 6.0

    When fit_failed is True (Algorithm 5 line 4), CRS is None and level is None.
    """
    if fit_failed or sag_to_span is None:
        return None, None
    if delta_lim <= 0:
        raise ValueError("delta_lim must be positive")
    delta_rel = max(0.0, float(sag_to_span))
    crs_10 = min(10.0, 10.0 * (delta_rel / float(delta_lim)) * float(class_weight))
    crs_10 = round(float(crs_10), 4)

    paper_cfg = get_setting("paper_crs_params", {}) or {}
    low_below = float(paper_cfg.get("low_below", 3.0)) if isinstance(paper_cfg, dict) else 3.0
    high_above = float(paper_cfg.get("high_above", 6.0)) if isinstance(paper_cfg, dict) else 6.0

    if crs_10 < low_below:
        level = "Low"
    elif crs_10 <= high_above:
        level = "Medium"
    else:
        level = "High"
    return crs_10, level


def compute_mspi(
    vni: float,
    wire_crs_10_list: list[float],
    alpha: float = 1.0 / 3.0,
    beta: float = 1.0 / 3.0,
    gamma: float = 1.0 / 3.0,
) -> dict[str, Any]:
    """Equation 11 / Algorithm 6 Multi-Site Priority Index (MSPI).

        MSPI_s = alpha * VNI_s + beta * mean_i(CRS_i) / 10 + gamma * max_i(CRS_i) / 10,
                 alpha + beta + gamma = 1

    When wire_crs_10_list is empty, returns mspi=None and flags=["NO_VALID_WIRES"].
    """
    valid_crs = [float(c) for c in wire_crs_10_list if c is not None and math.isfinite(float(c))]
    if not valid_crs:
        return {
            "mspi": None,
            "mean_crs_10": None,
            "max_crs_10": None,
            "paper_risk_level": None,
            "n_wires": 0,
            "flags": ["NO_VALID_WIRES"],
        }

    total_w = float(alpha) + float(beta) + float(gamma)
    if total_w <= 0 or not math.isclose(total_w, 1.0, abs_tol=1e-6):
        raise ValueError("alpha + beta + gamma must equal 1.0")

    vni_clamped = max(0.0, min(1.0, float(vni or 0.0)))
    mean_crs = float(np.mean(valid_crs))
    max_crs = float(np.max(valid_crs))
    mspi_val = float(alpha * vni_clamped + beta * (mean_crs / 10.0) + gamma * (max_crs / 10.0))
    mspi_val = round(max(0.0, min(1.0, mspi_val)), 6)
    _, max_level = compute_paper_crs(max_crs / 10.0, delta_lim=1.0, class_weight=1.0)
    # Determine level directly from max_crs on 0-10 scale
    paper_cfg = get_setting("paper_crs_params", {}) or {}
    low_below = float(paper_cfg.get("low_below", 3.0)) if isinstance(paper_cfg, dict) else 3.0
    high_above = float(paper_cfg.get("high_above", 6.0)) if isinstance(paper_cfg, dict) else 6.0
    if max_crs < low_below:
        max_level = "Low"
    elif max_crs <= high_above:
        max_level = "Medium"
    else:
        max_level = "High"

    return {
        "mspi": mspi_val,
        "mean_crs_10": round(mean_crs, 4),
        "max_crs_10": round(max_crs, 4),
        "paper_risk_level": max_level,
        "n_wires": len(valid_crs),
        "flags": [],
    }


def mspi_weight_grid(step: float = 0.1) -> list[tuple[float, float, float]]:
    """All (alpha, beta, gamma) triples on the simplex with step 0.1 (66 triples)."""
    k = round(1.0 / step)
    if not math.isclose(k * step, 1.0, abs_tol=1e-9):
        raise ValueError("step must divide 1")
    return [
        (round(i / k, 4), round(j / k, 4), round((k - i - j) / k, 4))
        for i in range(k + 1)
        for j in range(k + 1 - i)
    ]


def compute_mspi_sensitivity_grid(
    sites: list[dict[str, Any]],
    baseline: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    step: float = 0.1,
    k: int = 5,
) -> dict[str, Any]:
    """Table 3.11 / Section 3.6.6 sensitivity analysis over the 66-point weight grid.

    Each site dict in `sites` should provide:
      - `site_id`: str | int
      - `vni`: float
      - `crs_10_list` (or `crs`): list[float] of per-wire 0-10 CRS values
    """
    valid_sites = []
    for s in sites:
        crs_list = s.get("crs_10_list") if "crs_10_list" in s else s.get("crs", [])
        clean_crs = [float(c) for c in (crs_list or []) if c is not None and math.isfinite(float(c))]
        if clean_crs:
            valid_sites.append({
                "site_id": str(s.get("site_id", s.get("id", ""))),
                "vni": float(s.get("vni") or 0.0),
                "crs": clean_crs,
            })

    def _order_for_weights(a: float, b: float, g: float) -> list[str]:
        scored = []
        for s in valid_sites:
            res = compute_mspi(s["vni"], s["crs"], alpha=a, beta=b, gamma=g)
            scored.append((s["site_id"], res["mspi"], res["max_crs_10"]))
        scored.sort(key=lambda item: (-item[1], -item[2], item[0]))
        return [item[0] for item in scored]

    base_order = _order_for_weights(*baseline)
    top_k = min(k, len(base_order))
    base_pos = {sid: idx for idx, sid in enumerate(base_order)}
    ranks_by_site: dict[str, list[int]] = {sid: [] for sid in base_order}
    rows: list[dict[str, Any]] = []

    for a, b, g in mspi_weight_grid(step):
        order = _order_for_weights(a, b, g)
        for idx, sid in enumerate(order):
            ranks_by_site[sid].append(idx + 1)

        rho = None
        overlap = None
        if len(order) >= 2:
            x = [base_pos[sid] for sid in order]
            res = spearmanr(x, list(range(len(order))))
            if np.isfinite(res.statistic):
                rho = round(float(res.statistic), 6)
        if top_k > 0:
            overlap = round(len(set(order[:top_k]) & set(base_order[:top_k])) / top_k, 6)

        rows.append({
            "alpha": a,
            "beta": b,
            "gamma": g,
            "spearman_rho": rho,
            "top_k_overlap": overlap,
            "order": order,
        })

    rhos = [r["spearman_rho"] for r in rows if r["spearman_rho"] is not None]
    return {
        "baseline": baseline,
        "n_sites": len(base_order),
        "k": top_k,
        "grid_size": len(rows),
        "rows": rows,
        "min_rho": min(rhos) if rhos else None,
        "mean_rho": round(float(np.mean(rhos)), 6) if rhos else None,
        "rank_range": {sid: (min(v), max(v)) for sid, v in ranks_by_site.items() if v},
    }


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
    """Map confidence to severity (INVERTED: high confidence = low risk)."""
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
        
    val = float(crs_raw) / 100.0
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
        rank = min(2, clear_rank)
        safety_level = mapping["rank_to_level"].get(str(rank), "Unknown")
        reasons.append("CRS could not be estimated; verdict is based on clearance only.")
        reasons.append(clearance_estimate_str)
    else:
        rank = max(crs_rank, clear_rank)
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
    """Deprecated thin wrapper over new calibration logic for backwards compatibility."""
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
    """Rank sites based on max image score, with mean score as a tie-breaker."""
    def site_key(site):
        scores = site.get("image_scores", [0.0])
        if not scores:
            return (0.0, 0.0)
        max_score = max(scores)
        mean_score = sum(scores) / len(scores)
        return (max_score, mean_score)
        
    return sorted(sites_data, key=site_key, reverse=True)


def compute_unified_hazard_breakdown(
    *,
    clearance_m: float | None,
    pec_road_min_m: float = 5.5,
    pec_pedestrian_min_m: float = 4.6,
    max_wire_crs: float = 0.0,
    max_crs_10: float = 0.0,
    vni: float = 0.0,
    vegetation_count: int | None = None,
    min_veg_dist_m: float | None = None,
    min_veg_dist_px: float | None = None,
    dead_wire_count: int = 0,
    free_hanging_count: int = 0,
    wire_count: int = 0,
    image_area_mp: float = 1.0,
) -> dict[str, Any]:
    """Compute a single non-contradictory 5-factor hazard breakdown (0-100 each) + PEC clearance check.

    Factors:
    1. `clearance_risk`: PEC 5.5m road / 4.6m pedestrian check (`0` if >= 5.5m, `35-65` if 4.6-5.5m, `75-100` if < 4.6m).
    2. `catenary_sag_risk`: fused from 0-100 CRS and 0-10 Paper CRS (`max(max_wire_crs, max_crs_10 * 10)`).
    3. `vegetation_proximity_risk`: fused from VNI and direct wire-to-vegetation minimum distance (`0m` contact = 95, `< 1.0m` = 65).
    4. `dead_wire_risk`: based on `dead_wire` instances and topological unattached free-hanging strands.
    5. `bundle_density_risk`: congestion / spaghetti density from wire count and spans per megapixel.
    """
    # 1. Clearance Risk & PEC Check
    if clearance_m is not None and clearance_m > 0:
        road_ok = bool(clearance_m >= pec_road_min_m)
        ped_ok = bool(clearance_m >= pec_pedestrian_min_m)
        if clearance_m >= pec_road_min_m + 0.5:
            clr_risk = 5.0
        elif clearance_m >= pec_road_min_m:
            clr_risk = 20.0
        elif clearance_m >= pec_pedestrian_min_m:
            frac = (pec_road_min_m - clearance_m) / max(0.1, pec_road_min_m - pec_pedestrian_min_m)
            clr_risk = 38.0 + 27.0 * frac
        else:
            deficit = min(1.0, (pec_pedestrian_min_m - clearance_m) / max(1.0, pec_pedestrian_min_m))
            clr_risk = 75.0 + 25.0 * deficit
    else:
        road_ok = None
        ped_ok = None
        clr_risk = 15.0

    # 2. Catenary Sag Risk
    sag_risk = float(max(0.0, min(100.0, max(float(max_wire_crs or 0.0), float(max_crs_10 or 0.0) * 8.5))))

    # 3. Vegetation Proximity Risk (gated when vegetation_count == 0 and min_veg_dist_px is None)
    if vegetation_count == 0 and min_veg_dist_px is None:
        veg_risk = round(min(10.0, float(vni or 0.0) * 25.0), 1)
    else:
        veg_risk = float(max(0.0, min(100.0, float(vni or 0.0) * 60.0)))
        if min_veg_dist_px is not None:
            if min_veg_dist_px <= 1.0:
                veg_risk = max(veg_risk, 90.0)
            elif min_veg_dist_m is not None and min_veg_dist_m <= 0.6:
                veg_risk = max(veg_risk, 75.0)
            elif min_veg_dist_m is not None and min_veg_dist_m <= 1.5:
                veg_risk = max(veg_risk, 50.0)
            elif min_veg_dist_px <= 20.0:
                veg_risk = max(veg_risk, 55.0)

    # 4. Dangling / Dead Wire Risk
    eff_dead = max(int(dead_wire_count or 0), int(free_hanging_count or 0))
    if eff_dead == 0:
        dead_risk = 0.0
    elif eff_dead == 1:
        dead_risk = 65.0
    else:
        dead_risk = min(100.0, 65.0 + 17.5 * (eff_dead - 1))

    # 5. Bundle Density / Congestion Risk
    mp = max(0.15, float(image_area_mp or 1.0))
    spans_per_mp = float(wire_count) / mp
    density_risk = min(100.0, max(0.0, (wire_count / 12.0) * 55.0 + min(45.0, spans_per_mp * 1.5)))

    # Weighted composite + worst-hazard floor so critical clearance/contact is never diluted
    weighted = (
        0.28 * clr_risk
        + 0.27 * sag_risk
        + 0.20 * veg_risk
        + 0.15 * dead_risk
        + 0.10 * density_risk
    )
    peak_floor = 0.65 * max(clr_risk, sag_risk, veg_risk, dead_risk)
    fused_score = round(max(0.0, min(100.0, max(weighted, peak_floor))), 2)

    pec_check = {
        "road_min_m": pec_road_min_m,
        "pedestrian_min_m": pec_pedestrian_min_m,
        "estimated_clearance_m": round(float(clearance_m), 3) if clearance_m is not None else None,
        "road_ok": road_ok,
        "pedestrian_ok": ped_ok,
    }
    breakdown = {
        "clearance_risk": round(clr_risk, 1),
        "catenary_sag_risk": round(sag_risk, 1),
        "vegetation_proximity_risk": round(veg_risk, 1),
        "dead_wire_risk": round(dead_risk, 1),
        "bundle_density_risk": round(density_risk, 1),
        "fused_risk_score": fused_score,
    }
    return {"hazard_breakdown": breakdown, "pec_clearance_check": pec_check, "fused_risk_score": fused_score}

