"""Unified All-In-One Annotated Overlay Renderer for WIREFRAME.

Renders all detected elements together into a single primary annotated output image:
- Color-coded translucent segmentation masks and crisp contour outlines for all 4 canonical classes
  (`power_distribution_line`, `communication_cable`, `dead_wire`, `vegetation`).
- Class + confidence badges on every wire, vegetation region, and utility pole.
- Utility pole bounding boxes (`Pole`) and pole-to-pole span bridge / chord lines.
- Fitted catenary / centerline curves, sag indicator arrows/lines, and ground clearance reference lines
  so the entire scene context (wires + poles + vegetation + measurements) is immediately visible in one view.
"""

from __future__ import annotations

import ast
from typing import Any

import cv2
import numpy as np

from backend.utils.cv.types import normalize_wire_class

# RGB color palette matching frontend tokens.css:
# --wire-power_distribution_line: #ef4444 (Red-Orange / High Voltage) -> (239, 68, 68)
# --wire-communication_cable:     #3b82f6 (Royal Blue)                -> (59, 130, 246)
# --wire-dead_wire:               #f59e0b (Amber / Gold)              -> (245, 158, 11)
# --wire-vegetation:              #10b981 (Emerald Green)             -> (16, 185, 129)
# Pole:                           #38bdf8 (Telemetry Cyan)            -> (56, 189, 248)
CLASS_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "power_distribution_line": (239, 68, 68),
    "communication_cable": (59, 130, 246),
    "dead_wire": (245, 158, 11),
    "vegetation": (16, 185, 129),
    "pole": (56, 189, 248),
    "crossarm": (20, 184, 166),
    "insulator": (244, 114, 182),
    "building": (148, 163, 184),
    "other": (168, 85, 247),
}

CLASS_SHORT_LABELS: dict[str, str] = {
    "power_distribution_line": "Power Line",
    "communication_cable": "Communication",
    "dead_wire": "Dead Wire",
    "vegetation": "Vegetation",
    "pole": "Pole",
    "crossarm": "Crossarm",
    "insulator": "Insulator",
    "building": "Building",
    "other": "Other Wire",
}


def _draw_badge(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    bg_rgb: tuple[int, int, int],
    text_rgb: tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.52,
    thickness: int = 1,
) -> None:
    """Draw a high-contrast pill/rectangle badge with text at (x, y)."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 6, 4
    bx1 = max(2, min(w - tw - 2 * pad_x - 2, x))
    by1 = max(2, min(h - th - 2 * pad_y - 2, y - th - 2 * pad_y))
    bx2 = min(w - 2, bx1 + tw + 2 * pad_x)
    by2 = min(h - 2, by1 + th + 2 * pad_y + baseline // 2)

    # Dark backing with colored border for readability on both bright sky and dark foliage
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (15, 23, 42), -1)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), bg_rgb, 2)
    cv2.putText(
        img,
        text,
        (bx1 + pad_x, by2 - pad_y - baseline // 2),
        font,
        font_scale,
        text_rgb,
        thickness,
        cv2.LINE_AA,
    )


def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 8,
) -> None:
    """Draw a dashed line between pt1 and pt2."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = float(np.hypot(dx, dy))
    if dist < 1.0:
        return
    steps = max(1, int(dist // dash_len))
    for i in range(0, steps, 2):
        t0 = i / steps
        t1 = min(1.0, (i + 1) / steps)
        p0 = (int(round(pt1[0] + dx * t0)), int(round(pt1[1] + dy * t0)))
        p1 = (int(round(pt1[0] + dx * t1)), int(round(pt1[1] + dy * t1)))
        cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)


def _parse_polyline(raw: Any) -> list[tuple[float, float]]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [(float(p[0]), float(p[1])) for p in raw if len(p) >= 2]
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [(float(p[0]), float(p[1])) for p in parsed if len(p) >= 2]
        except Exception:
            return []
    return []


def render_unified_overlay(
    rgb_image: np.ndarray,
    *,
    analyzed_wires: list[dict[str, Any]] | None = None,
    vegetation_detections: list[dict[str, Any]] | None = None,
    pole_bbox: list[float] | None = None,
    pole_confidence: float | None = None,
    pole_masks: list[np.ndarray] | None = None,
    crossarm_boxes: list[list[float]] | None = None,
    insulator_boxes: list[list[float]] | None = None,
    building_boxes: list[list[float]] | None = None,
    virtual_ground_y: float | None = None,
    meters_per_px: float | None = None,
) -> np.ndarray:
    """Render all detections (masks, contours, centerlines, pole boxes, span chords, sag & clearance indicators)
    into a single unified RGB image.
    """
    if rgb_image is None or rgb_image.size == 0:
        return rgb_image

    canvas = rgb_image.copy()
    overlay = canvas.copy()
    h, w = canvas.shape[:2]

    wires_list = analyzed_wires or []
    veg_list = vegetation_detections or []
    p_masks = pole_masks or []

    # 1. Translucent segmentation mask fills (vegetation + poles + wires)
    has_mask_pixels = False

    for pm in p_masks:
        if pm is not None and pm.shape[:2] == (h, w):
            m_bool = pm > 0
            if np.any(m_bool):
                overlay[m_bool] = CLASS_COLORS_RGB["pole"]
                has_mask_pixels = True

    for veg in veg_list:
        vm = veg.get("mask")
        if vm is not None and vm.shape[:2] == (h, w):
            m_bool = vm > 0
            if np.any(m_bool):
                overlay[m_bool] = CLASS_COLORS_RGB["vegetation"]
                has_mask_pixels = True

    for wire in wires_list:
        wm = wire.get("mask")
        cls_key = normalize_wire_class(wire.get("class_name", "other"))
        color = CLASS_COLORS_RGB.get(cls_key, CLASS_COLORS_RGB["other"])
        if wm is not None and wm.shape[:2] == (h, w):
            m_bool = wm > 0
            if np.any(m_bool):
                overlay[m_bool] = color
                has_mask_pixels = True

    if has_mask_pixels:
        canvas = cv2.addWeighted(overlay, 0.32, canvas, 0.68, 0)

    # 2. Crisp mask contours & bounding boxes for Vegetation
    for idx, veg in enumerate(veg_list, start=1):
        v_color = CLASS_COLORS_RGB["vegetation"]
        vm = veg.get("mask")
        if vm is not None and vm.shape[:2] == (h, w):
            contours, _ = cv2.findContours((vm > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(canvas, contours, -1, v_color, 2, cv2.LINE_AA)
        v_bbox = veg.get("bbox")
        if v_bbox and len(v_bbox) >= 4:
            vx1, vy1, vx2, vy2 = [int(round(v)) for v in v_bbox[:4]]
            cv2.rectangle(canvas, (vx1, vy1), (vx2, vy2), v_color, 2)
            v_conf = float(veg.get("confidence") or 0.0)
            v_label = f"Vegetation {int(round(v_conf * 100))}%" if v_conf > 0 else "Vegetation"
            _draw_badge(canvas, v_label, vx1, max(24, vy1), v_color)

    # 3. Utility Pole bounding box & ground reference line
    ground_y: int | None = None
    pole_center_x: int | None = None
    if pole_bbox and len(pole_bbox) >= 4:
        p_color = CLASS_COLORS_RGB["pole"]
        px1, py1, px2, py2 = [int(round(v)) for v in pole_bbox[:4]]
        cv2.rectangle(canvas, (px1, py1), (px2, py2), p_color, 2)
        if pole_confidence and pole_confidence >= 0.25:
            p_badge = f"Pole {int(round(pole_confidence * 100))}%"
        elif pole_confidence and pole_confidence > 0:
            p_badge = "Pole (Detected)"
        else:
            p_badge = "Pole"
        _draw_badge(canvas, p_badge, px1, max(24, py1), p_color)
        if py2 < h - 5:
            ground_y = py2
            pole_center_x = (px1 + px2) // 2
            # Draw horizontal ground datum line at base of pole
            gx1 = max(4, px1 - 45)
            gx2 = min(w - 4, px2 + 45)
            _draw_dashed_line(canvas, (gx1, ground_y), (gx2, ground_y), p_color, thickness=2, dash_len=8)
        elif virtual_ground_y is not None:
            pole_center_x = (px1 + px2) // 2

    # 3b. Crossarms, Insulators & Building Context Outlines
    for cb in crossarm_boxes or []:
        if len(cb) >= 4:
            cx1, cy1, cx2, cy2 = [int(round(v)) for v in cb[:4]]
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), CLASS_COLORS_RGB["crossarm"], 2)
            _draw_badge(canvas, "Crossarm", cx1, max(24, cy1), CLASS_COLORS_RGB["crossarm"], font_scale=0.44)

    for ib in insulator_boxes or []:
        if len(ib) >= 4:
            ix1, iy1, ix2, iy2 = [int(round(v)) for v in ib[:4]]
            icx, icy = (ix1 + ix2) // 2, (iy1 + iy2) // 2
            cv2.circle(canvas, (icx, icy), max(4, (ix2 - ix1) // 2), CLASS_COLORS_RGB["insulator"], 2, cv2.LINE_AA)

    for bb in (building_boxes or [])[:2]:
        if len(bb) >= 4:
            bx1, by1, bx2, by2 = [int(round(v)) for v in bb[:4]]
            _draw_dashed_line(canvas, (bx1, by1), (bx2, by1), CLASS_COLORS_RGB["building"], thickness=1, dash_len=8)

    # 4. Wires: continuous smooth mask contours, endpoint chords, centerline / catenary curves, sag vectors & badges
    lowest_wire_pt: tuple[int, int] | None = None
    for idx, wire in enumerate(wires_list, start=1):
        cls_key = normalize_wire_class(wire.get("class_name", "other"))
        w_color = CLASS_COLORS_RGB.get(cls_key, CLASS_COLORS_RGB["other"])
        short_lbl = CLASS_SHORT_LABELS.get(cls_key, cls_key.replace("_", " ").title())
        w_conf = float(wire.get("confidence") or 0.0)

        wm = wire.get("mask")
        if wm is not None and wm.shape[:2] == (h, w):
            # Smooth 1-2px gaps in rendered contour so thin diagonal wires render as continuous smooth ribbons
            k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            wm_smooth = cv2.morphologyEx((wm > 0).astype(np.uint8), cv2.MORPH_CLOSE, k_smooth)
            contours, _ = cv2.findContours(wm_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(canvas, contours, -1, w_color, 1, cv2.LINE_AA)

        w_bbox = wire.get("bbox")
        if w_bbox and len(w_bbox) >= 4:
            bx1, by1, bx2, by2 = [int(round(v)) for v in w_bbox[:4]]
        else:
            bx1, by1 = 10, 25 * idx

        polyline = _parse_polyline(wire.get("centerline_json") or wire.get("polyline_json") or wire.get("polyline"))
        if len(polyline) > 1:
            pts_arr = np.round(polyline).astype(np.int32)
            # Endpoint chord (span reference line)
            p_start = (int(pts_arr[0][0]), int(pts_arr[0][1]))
            p_end = (int(pts_arr[-1][0]), int(pts_arr[-1][1]))
            _draw_dashed_line(canvas, p_start, p_end, (226, 232, 240), thickness=1, dash_len=10)

            # Crisp centerline curve (dark halo + vibrant class color core)
            pts_cv = pts_arr.reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts_cv], isClosed=False, color=(15, 23, 42), thickness=4, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [pts_cv], isClosed=False, color=w_color, thickness=2, lineType=cv2.LINE_AA)

            # Endpoints
            cv2.circle(canvas, p_start, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, p_start, 4, w_color, 1, cv2.LINE_AA)
            cv2.circle(canvas, p_end, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, p_end, 4, w_color, 1, cv2.LINE_AA)

            # Sag indicator from chord to lowest sag point on the curve
            max_y_idx = int(np.argmax(pts_arr[:, 1]))
            sag_pt = (int(pts_arr[max_y_idx][0]), int(pts_arr[max_y_idx][1]))
            if lowest_wire_pt is None or sag_pt[1] > lowest_wire_pt[1]:
                lowest_wire_pt = sag_pt

            dx_chord = p_end[0] - p_start[0]
            if abs(dx_chord) > 5:
                t_frac = (sag_pt[0] - p_start[0]) / float(dx_chord)
                if 0.0 <= t_frac <= 1.0:
                    chord_y_at_sag = int(round(p_start[1] + t_frac * (p_end[1] - p_start[1])))
                    if sag_pt[1] - chord_y_at_sag >= 4:
                        cv2.line(canvas, (sag_pt[0], chord_y_at_sag), sag_pt, (250, 204, 21), 2, cv2.LINE_AA)
                        cv2.circle(canvas, sag_pt, 4, (250, 204, 21), -1, cv2.LINE_AA)

            label_x, label_y = int(pts_arr[0][0]), int(pts_arr[0][1])
        else:
            label_x, label_y = bx1, max(24, by1)

        cat = wire.get("catenary")
        sag_px_val = getattr(cat, "sag_px", None) if cat is not None else wire.get("sag_px")
        sag_suffix = ""
        if sag_px_val is not None and float(sag_px_val) > 0:
            if meters_per_px:
                sag_suffix = f" | Sag {float(sag_px_val) * meters_per_px:.2f}m"
            else:
                sag_suffix = f" | Sag {float(sag_px_val):.0f}px"

        badge_text = f"#{idx} {short_lbl} {int(round(w_conf * 100))}%{sag_suffix}"
        _draw_badge(canvas, badge_text, label_x, max(24, label_y - 4), w_color)

        # Vegetation encroachment indicator line when vegetation is within 80 px of this wire
        veg_dist_px = wire.get("vegetation_distance_px")
        w_pt = wire.get("veg_wire_point")
        v_pt = wire.get("veg_point")
        if veg_dist_px is not None and float(veg_dist_px) <= 80.0 and w_pt and v_pt:
            wp_t = (int(round(w_pt[0])), int(round(w_pt[1])))
            vp_t = (int(round(v_pt[0])), int(round(v_pt[1])))
            _draw_dashed_line(canvas, wp_t, vp_t, (16, 185, 129), thickness=2, dash_len=5)
            cv2.circle(canvas, wp_t, 3, (16, 185, 129), -1, cv2.LINE_AA)
            cv2.circle(canvas, vp_t, 3, (16, 185, 129), -1, cv2.LINE_AA)

    # 5. Ground clearance indicator line from lowest wire sag point to pole / virtual ground datum
    if ground_y is not None and lowest_wire_pt is not None and ground_y > lowest_wire_pt[1]:
        cx = lowest_wire_pt[0]
        _draw_dashed_line(canvas, (cx, lowest_wire_pt[1]), (cx, ground_y), (56, 189, 248), thickness=2, dash_len=6)
        if pole_center_x is not None:
            _draw_dashed_line(canvas, (cx, ground_y), (pole_center_x, ground_y), (56, 189, 248), thickness=1, dash_len=6)
        clr_px = ground_y - lowest_wire_pt[1]
        if meters_per_px:
            clr_text = f"Clearance ~{clr_px * meters_per_px:.2f}m"
        else:
            clr_text = f"Clearance {clr_px}px"
        mid_y = (lowest_wire_pt[1] + ground_y) // 2
        _draw_badge(canvas, clr_text, min(w - 160, cx + 8), mid_y, (56, 189, 248))
    elif virtual_ground_y is not None and lowest_wire_pt is not None and virtual_ground_y > lowest_wire_pt[1] and meters_per_px:
        cx = lowest_wire_pt[0]
        bottom_y = h - 4
        _draw_dashed_line(canvas, (cx, lowest_wire_pt[1]), (cx, bottom_y), (56, 189, 248), thickness=2, dash_len=6)
        est_clr_m = (float(virtual_ground_y) - lowest_wire_pt[1]) * float(meters_per_px)
        clr_text = f"Est. Clearance ~{est_clr_m:.2f}m (Prior)"
        mid_y = (lowest_wire_pt[1] + bottom_y) // 2
        _draw_badge(canvas, clr_text, max(6, min(w - 220, cx - 60)), mid_y, (56, 189, 248))

    return canvas
