"""Calibration module to scale pixels to meters and calculate ground clearance."""

import logging
from dataclasses import dataclass
from typing import Any

from backend.config_store import get_setting

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    meters_per_px: float | None
    source: str
    reasons: list[str]


def calibrate_scale(
    manual_pt1: tuple[int, int] | None = None,
    manual_pt2: tuple[int, int] | None = None,
    manual_length_m: float | None = None,
    pole_bbox: list[float] | None = None,
    image_height: int | None = None
) -> CalibrationResult:
    """Derive meters_per_px prioritizing manual > pole > unavailable.
    
    If pole_bbox touches the bottom of the image (y2 >= image_height - 5),
    it is considered unavailable for clearance since the ground is not visible.
    """
    reasons = []
    
    # 1. Manual Calibration Priority
    if manual_pt1 and manual_pt2 and manual_length_m and manual_length_m > 0:
        dx = manual_pt1[0] - manual_pt2[0]
        dy = manual_pt1[1] - manual_pt2[1]
        px_dist = (dx**2 + dy**2)**0.5
        if px_dist > 0:
            scale = manual_length_m / px_dist
            reasons.append(f"Scale derived from manual calibration: {manual_length_m}m / {px_dist:.1f}px = {scale:.6f} m/px")
            return CalibrationResult(meters_per_px=scale, source="manual", reasons=reasons)
            
    # 2. Pole BBox Priority
    if pole_bbox and len(pole_bbox) >= 4 and image_height:
        _, y1, _, y2 = pole_bbox
        pole_px_height = y2 - y1
        
        if y2 >= image_height - 5:
            reasons.append("Pole bounding box touches the bottom of the image; ground is not fully visible.")
        elif pole_px_height > 0:
            # Assumed pole height from settings (default 9m for typical utility pole)
            assumed_h = get_setting("pole_heights")[0]
            scale = assumed_h / pole_px_height
            reasons.append(f"Scale derived from pole detection: {assumed_h}m / {pole_px_height:.1f}px = {scale:.6f} m/px")
            return CalibrationResult(meters_per_px=scale, source="pole", reasons=reasons)
            
    reasons.append("No valid scale source available.")
    return CalibrationResult(meters_per_px=None, source="unavailable", reasons=reasons)


def estimate_clearance(
    lowest_wire_y: float,
    pole_bbox: list[float] | None,
    meters_per_px: float | None,
    image_height: int | None
) -> dict[str, Any]:
    """Estimate ground clearance.
    
    ground_y = bottom of reference pole bbox.
    clearance = (ground_y - lowest_wire_y) * meters_per_px.
    """
    if not meters_per_px or not pole_bbox or not image_height:
        return {
            "clearance_m": None,
            "clearance_low_m": None,
            "clearance_high_m": None,
            "status": "unavailable"
        }
        
    ground_y = pole_bbox[3]
    
    if ground_y >= image_height - 5:
        # Ground not fully visible
        return {
            "clearance_m": None,
            "clearance_low_m": None,
            "clearance_high_m": None,
            "status": "unavailable"
        }
    
    # In image coords, y goes down. So ground_y > lowest_wire_y
    if lowest_wire_y > ground_y:
        logger.warning(f"Wire ({lowest_wire_y}) is below ground reference ({ground_y})!")
        clearance_px = 0.0
    else:
        clearance_px = ground_y - lowest_wire_y
        
    clearance_m = clearance_px * meters_per_px
    margin_pct = get_setting("margin_of_error_pct", 20) / 100.0
    
    low_m = clearance_m * (1.0 - margin_pct)
    high_m = clearance_m * (1.0 + margin_pct)
    
    return {
        "clearance_m": clearance_m,
        "clearance_low_m": low_m,
        "clearance_high_m": high_m,
        "status": "estimated"
    }
