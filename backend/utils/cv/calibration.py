"""Calibration module to scale pixels to meters, calculate ground clearance,
and provide classical Hough pole fallback, span stitching, and loose-end detection.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from backend.config_store import get_setting

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    meters_per_px: float | None
    source: str
    reasons: list[str]


def detect_pole_hough_fallback(bgr_image: np.ndarray) -> list[float] | None:
    """Classical Hough vertical-structure pole detector fallback when YOLO does not detect a pole.

    Looks for strong near-vertical parallel edge pairs or vertical structural columns
    spanning at least 35% of the image height, and returns [x1, y1, x2, y2] of the best candidate.
    """
    if bgr_image is None or bgr_image.size == 0:
        return None
    h, w = bgr_image.shape[:2]
    if h < 80 or w < 80:
        return None

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY) if bgr_image.ndim == 3 else bgr_image
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    vert_strength = np.clip(np.abs(gx) - 0.6 * np.abs(gy), 0, None)
    if float(vert_strength.max()) < 20.0:
        return None

    edges = cv2.Canny(gray, 50, 150)
    min_line_len = max(30, int(0.30 * h))
    lines = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=40,
        minLineLength=min_line_len,
        maxLineGap=max(10, int(0.05 * h)),
    )
    if lines is None or len(lines) == 0:
        return None

    # Keep near-vertical segments (within 8 degrees of vertical)
    vertical_segs: list[tuple[float, float, float, float, float]] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, line)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_line_len:
            continue
        angle_from_vert = math.degrees(math.atan2(abs(dx), max(abs(dy), 1e-6)))
        if angle_from_vert <= 8.0:
            xm = 0.5 * (x1 + x2)
            ym_top = min(y1, y2)
            ym_bot = max(y1, y2)
            vertical_segs.append((xm, ym_top, ym_bot, length, angle_from_vert))

    if not vertical_segs:
        return None

    # Cluster x positions within pole width tolerance (1% - 10% of image width)
    min_w = max(4.0, 0.008 * w)
    max_w = max(20.0, 0.10 * w)
    best_score = -1.0
    best_bbox: list[float] | None = None

    for i, (xm1, top1, bot1, len1, _) in enumerate(vertical_segs):
        # Also allow a single very strong vertical cluster or paired edges
        partner_xs = [xm1]
        tops = [top1]
        bots = [bot1]
        total_len = len1
        for j, (xm2, top2, bot2, len2, _) in enumerate(vertical_segs):
            if i == j:
                continue
            dist_x = abs(xm2 - xm1)
            overlap_y = min(bot1, bot2) - max(top1, top2)
            if dist_x <= max_w and overlap_y > 0.25 * min(len1, len2):
                partner_xs.append(xm2)
                tops.append(top2)
                bots.append(bot2)
                total_len += len2

        y_top = float(min(tops))
        y_bot = float(max(bots))
        span_h = y_bot - y_top
        if span_h < 0.35 * h:
            continue

        x_left = float(min(partner_xs))
        x_right = float(max(partner_xs))
        if x_right - x_left < min_w:
            half_pad = max(4.0, 0.012 * w)
            x_center = 0.5 * (x_left + x_right)
            x_left = max(0.0, x_center - half_pad)
            x_right = min(float(w - 1), x_center + half_pad)

        score = total_len * (span_h / h)
        if score > best_score:
            best_score = score
            best_bbox = [round(x_left, 1), round(y_top, 1), round(x_right, 1), round(y_bot, 1)]

    return best_bbox


def bridge_across_poles(
    wire_mask: np.ndarray,
    pole_masks: list[np.ndarray],
    dilate_px: int = 12,
) -> np.ndarray:
    """Bridge wire mask interruptions caused by utility poles occluding a continuous span."""
    if not pole_masks or wire_mask is None or not np.any(wire_mask):
        return wire_mask
    out = (wire_mask > 0).astype(np.uint8)
    pole_union = np.zeros_like(out)
    for pm in pole_masks:
        if pm is not None and pm.shape == out.shape:
            pole_union |= (pm > 0).astype(np.uint8)
    if not np.any(pole_union):
        return wire_mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    bridged = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    out = out | (bridged & pole_union)
    return out.astype(wire_mask.dtype)


def stitch_spans(
    polylines: list[list[tuple[float, float]]],
    max_gap_px: float = 90.0,
    max_angle_deg: float = 12.0,
    max_offset_px: float = 14.0,
) -> list[list[tuple[float, float]]]:
    """Collinearly stitch fragmented span polylines separated by small occlusion gaps."""
    valid = [list(p) for p in polylines if len(p) >= 2]
    if len(valid) <= 1:
        return valid

    # Sort polylines by left endpoint x
    valid.sort(key=lambda pts: min(pts[0][0], pts[-1][0]))
    normalized: list[list[tuple[float, float]]] = []
    for pts in valid:
        if pts[0][0] > pts[-1][0]:
            pts = list(reversed(pts))
        normalized.append(pts)

    used = [False] * len(normalized)
    stitched: list[list[tuple[float, float]]] = []

    def _end_tangent(pts: list[tuple[float, float]], at_end: bool, k: int = 15) -> tuple[float, float]:
        sub = pts[-min(len(pts), k):] if at_end else pts[:min(len(pts), k)]
        dx = sub[-1][0] - sub[0][0]
        dy = sub[-1][1] - sub[0][1]
        norm = math.hypot(dx, dy) or 1.0
        return dx / norm, dy / norm

    for i, base in enumerate(normalized):
        if used[i]:
            continue
        used[i] = True
        cur = list(base)
        while True:
            merged_any = False
            x1, y1 = cur[-1]
            tx1, ty1 = _end_tangent(cur, at_end=True)
            for j, cand in enumerate(normalized):
                if used[j]:
                    continue
                x2, y2 = cand[0]
                dx, dy = x2 - x1, y2 - y1
                gap = math.hypot(dx, dy)
                if dx < -5.0 or gap > max_gap_px or gap < 1e-6:
                    continue
                tx2, ty2 = _end_tangent(cand, at_end=False)
                dot = max(-1.0, min(1.0, tx1 * tx2 + ty1 * ty2))
                angle = math.degrees(math.acos(dot))
                # Lateral offset from tangent line
                perp = abs(-ty1 * dx + tx1 * dy)
                if angle <= max_angle_deg and perp <= max_offset_px:
                    cur.extend(cand)
                    used[j] = True
                    merged_any = True
                    break
            if not merged_any:
                break
        stitched.append(cur)

    return stitched


def detect_loose_ends(
    wire_masks: list[np.ndarray],
    image_shape: tuple[int, int],
    *,
    min_dangle_px: int = 35,
    min_pitch_deg: float = 50.0,
) -> list[dict[str, Any]]:
    """Detect steeply hanging free-end wire segments (dangling / cut wires)."""
    from backend.utils.cv.centerline import pixel_graph, zhang_suen

    h, w = image_shape[:2]
    margin = max(12, int(0.02 * min(h, w)))
    results: list[dict[str, Any]] = []

    for idx, mask in enumerate(wire_masks):
        if mask is None or not np.any(mask):
            continue
        skel = zhang_suen(mask > 0)
        adj = pixel_graph(skel)
        endpoints = [pt for pt, nbs in adj.items() if len(nbs) == 1]
        for ep in endpoints:
            ex, ey = ep
            # Must be in interior (not clipped by border)
            if ex <= margin or ey <= margin or ex >= w - 1 - margin or ey >= h - 1 - margin:
                continue
            # Trace from endpoint up to min_dangle_px steps
            chain = [ep]
            prev = ep
            cur = next(iter(adj[ep]))
            while len(chain) < min_dangle_px and len(adj.get(cur, ())) == 2:
                chain.append(cur)
                nxt = next(n for n in adj[cur] if n != prev)
                prev, cur = cur, nxt
            if len(chain) < min_dangle_px:
                continue
            # Free end must be the lower end (ey > cur_y in image coords)
            rx, ry = chain[-1]
            dy = ey - ry
            dx = abs(ex - rx)
            if dy <= 0:
                continue
            pitch = math.degrees(math.atan2(dy, max(dx, 1e-6)))
            if pitch >= min_pitch_deg:
                results.append({
                    "wire_index": idx,
                    "endpoint": (int(ex), int(ey)),
                    "root": (int(rx), int(ry)),
                    "length_px": len(chain),
                    "pitch_deg": round(pitch, 1),
                })
    return results


def calibrate_scale(
    manual_pt1: tuple[int, int] | None = None,
    manual_pt2: tuple[int, int] | None = None,
    manual_length_m: float | None = None,
    pole_bbox: list[float] | None = None,
    image_height: int | None = None,
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
    image_height: int | None,
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
            "status": "unavailable",
        }
        
    ground_y = pole_bbox[3]
    
    if ground_y >= image_height - 5:
        # Ground not fully visible
        return {
            "clearance_m": None,
            "clearance_low_m": None,
            "clearance_high_m": None,
            "status": "unavailable",
        }
    
    # In image coords, y goes down. So ground_y > lowest_wire_y
    if lowest_wire_y > ground_y:
        logger.warning(f"Wire ({lowest_wire_y}) is below ground reference ({ground_y})!")
        return {
            "clearance_m": None,
            "clearance_low_m": None,
            "clearance_high_m": None,
            "status": "unavailable",
            "reason": "wire below ground reference (invalid geometry)",
        }
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
        "status": "estimated",
    }
