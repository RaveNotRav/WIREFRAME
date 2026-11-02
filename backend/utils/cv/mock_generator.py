import hashlib
import os

import cv2
import numpy as np

from backend.mock_detector import get_mock_scenario, resolve_active_mock_scenario
from backend.utils.cv.types import SegmentationResult


def _generate_scenario_segmentation(
    active_scenario: str,
    original_shape: tuple[int, int],
) -> SegmentationResult:
    """Shape synthetic pole & wire masks for low/moderate/high/critical scenarios.

    All masks and confidences flow through the unmodified CV pipeline
    (extract_centerline -> calculate_sag_metrics -> compute_catenary_risk ->
    compute_detection_risk -> compute_barangay_risk -> compute_risk_score).
    """
    h, w = original_shape
    names = {0: "telecom_cable", 1: "power_line", 2: "dead_abandoned", 3: "pole"}

    boxes: list[list[float]] = []
    classes: list[int] = []
    confidences: list[float] = []
    masks: list[np.ndarray] = []

    # Pole bbox: height = 0.80 * h, ground_y = 0.90 * h -> meters_per_px = 9.0 / (0.80 * h)
    boxes.append([w * 0.8, h * 0.1, w * 0.95, h * 0.9])
    classes.append(3)
    confidences.append(0.95)
    pole_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(pole_mask, (int(w * 0.8), int(h * 0.1)), (int(w * 0.95), int(h * 0.9)), 255, -1)
    masks.append(pole_mask)

    x_end = max(20, int(w * 0.8))
    center_x = x_end * 0.5
    max_wire_y = int(h * 0.9 - 10)

    # Configure wire geometry (sag proportional to horizontal span x_end so sag_to_span is aspect-ratio invariant),
    # attachment height (controls ground clearance relative to ground_y = 0.90 * h),
    # wire classes (0=telecom, 1=power, 2=dead_abandoned), and confidences (<0.40 high, 0.40-0.69 medium, >=0.70 low).
    if active_scenario == "low":
        # Low band (risk_score <= 33):
        # - conf >= 0.70 on all wires -> detection_risk = 10.0
        # - sag_to_span ~ 0.055 -> max_crs ~ 15 (Low wire band)
        # - high attachment -> lowest_y ~ 0.25*h -> clearance ~ 7.3m (clearance_low ~ 5.85m >= 5.5m -> status 'ok')
        # - 0 dead wires -> Overall Safety = Safe
        wire_specs = [
            {"y_attach": int(h * 0.14), "sag_ratio": 0.048, "cls": 0, "conf": 0.92},
            {"y_attach": int(h * 0.17), "sag_ratio": 0.052, "cls": 0, "conf": 0.90},
            {"y_attach": int(h * 0.20), "sag_ratio": 0.055, "cls": 1, "conf": 0.88},
        ]
    elif active_scenario == "moderate":
        # Moderate band (33 < risk_score <= 66):
        # - 2 medium conf (0.55, 0.52) + 1 high conf (0.85) -> detection_risk = 36.67
        # - sag_to_span ~ 0.115 -> max_crs ~ 50 (Moderate wire band)
        # - 1 dead wire -> risk_score ~ 40-52 (Moderate)
        # - clearance 'ok' / 'marginal' + Moderate wire -> Overall Safety = Caution
        wire_specs = [
            {"y_attach": int(h * 0.16), "sag_ratio": 0.095, "cls": 0, "conf": 0.85},
            {"y_attach": int(h * 0.20), "sag_ratio": 0.105, "cls": 1, "conf": 0.55},
            {"y_attach": int(h * 0.24), "sag_ratio": 0.115, "cls": 2, "conf": 0.52},
        ]
    elif active_scenario == "high":
        # High band (66 < risk_score <= 85):
        # - 2 low conf (0.32, 0.35 -> 'high' severity 1.0) + 1 medium conf (0.55 -> 0.5) -> detection_risk = 83.33
        # - sag_to_span ~ 0.158 -> max_crs ~ 75 (High wire band, <= 85)
        # - 2 dead wires -> risk_score ~ 70-83 (High)
        # - lowest_y kept above 'below' threshold so clearance is 'ok'/'marginal' -> Overall Safety = Unsafe
        wire_specs = [
            {"y_attach": int(h * 0.14), "sag_ratio": 0.135, "cls": 1, "conf": 0.55},
            {"y_attach": int(h * 0.17), "sag_ratio": 0.148, "cls": 2, "conf": 0.35},
            {"y_attach": int(h * 0.20), "sag_ratio": 0.158, "cls": 2, "conf": 0.32},
        ]
    else:  # "critical"
        # Critical band (risk_score > 85):
        # - all 3 wires conf < 0.40 -> detection_risk = 100.0
        # - sag_to_span ~ 0.21 -> max_crs = 100.0 (Critical wire band)
        # - all 3 wires dead_abandoned -> dead_ratio = 1.0 -> risk_score ~ 88-99 (Critical)
        # - low attachment so lowest_y ~ 0.65*h -> clearance ~ 2.8m (clearance_high ~ 3.4m < 5.5m -> status 'below')
        # - Critical wire + below clearance -> Overall Safety = Critical
        wire_specs = [
            {"y_attach": int(h * 0.34), "sag_ratio": 0.180, "cls": 2, "conf": 0.31},
            {"y_attach": int(h * 0.38), "sag_ratio": 0.195, "cls": 2, "conf": 0.29},
            {"y_attach": int(h * 0.42), "sag_ratio": 0.210, "cls": 2, "conf": 0.28},
        ]

    step_x = max(1, min(5, x_end // 40))
    for spec in wire_specs:
        mask = np.zeros((h, w), dtype=np.uint8)
        y_attach = int(spec["y_attach"])
        wire_sag = max(4, int(x_end * float(spec["sag_ratio"])))
        # Ensure vertex y_attach + wire_sag does not get clamped by max_wire_y so sag_px stays exact
        if y_attach + wire_sag > max_wire_y:
            y_attach = max(10, max_wire_y - wire_sag)

        pts = []
        for x in range(0, x_end + 1, step_x):
            u = (x - center_x) / center_x if center_x > 0 else 0.0
            y = int(round(y_attach + wire_sag * (1.0 - u**2)))
            y = max(0, min(y, max_wire_y))
            pts.append((x, y))

        if len(pts) > 2:
            pts_arr = np.array(pts, np.int32)
            cv2.polylines(mask, [pts_arr], False, 255, 6)
            bx, by, bw, bh = cv2.boundingRect(pts_arr)
            boxes.append([bx, by, bx + bw, by + bh])
            classes.append(int(spec["cls"]))
            confidences.append(float(spec["conf"]))
            masks.append(mask)

    return SegmentationResult(
        mask_available=True,
        boxes=boxes,
        classes=classes,
        confidences=confidences,
        masks=masks,
        names=names,
    )


def generate_mock_segmentation(image_path: str, original_shape: tuple[int, int]) -> SegmentationResult:
    """
    Generates deterministic synthetic segmentation outputs based on the file content
    or WIREFRAME_MOCK_SCENARIO when configured.
    Filename keywords override when WIREFRAME_MOCK_SCENARIO is 'random' (default).
    """
    h, w = original_shape

    configured_scenario = get_mock_scenario()
    if configured_scenario in ("low", "moderate", "high", "critical", "cycle"):
        active_scenario = resolve_active_mock_scenario()
        return _generate_scenario_segmentation(active_scenario, original_shape)
    
    # 1. Determine scenario based on filename override OR content hash
    filename = os.path.basename(image_path).lower()
    scenario = None
    
    if "safe" in filename and "unsafe" not in filename:
        scenario = "safe"
    elif "unsafe" in filename:
        scenario = "unsafe"
    elif "caution" in filename:
        scenario = "caution"
    elif "critical" in filename:
        scenario = "critical"
    elif "nowires" in filename:
        scenario = "nowires"
    elif "reject" in filename:
        scenario = "reject"
    elif "nopole" in filename:
        scenario = "nopole"
    else:
        # Seed from content hash
        try:
            with open(image_path, "rb") as f:
                content = f.read()
            hash_val = int(hashlib.sha256(content).hexdigest(), 16)
        except Exception:
            hash_val = 0
            
        scenarios = ["safe", "caution", "safe", "caution"]
        scenario = scenarios[hash_val % len(scenarios)]
        
    if scenario == "reject":
        # We handle reject upstream usually, but if we got here, we just act like no_wires
        return SegmentationResult(mask_available=False)
        
    if scenario == "nowires":
        return SegmentationResult(
            mask_available=True,
            boxes=[],
            classes=[],
            confidences=[],
            masks=[],
            names={0: "telecom_cable", 1: "power_line", 2: "dead_abandoned", 3: "pole"}
        )
        
    # Generate mock pole
    has_pole = (scenario != "nopole")
    boxes = []
    classes = []
    confidences = []
    masks = []
    
    names = {0: "telecom_cable", 1: "power_line", 2: "dead_abandoned", 3: "pole"}
    
    if has_pole:
        # Pole bounding box on the right side
        boxes.append([w * 0.8, h * 0.1, w * 0.95, h * 0.9])
        classes.append(3)
        confidences.append(0.95)
        # Empty mask for pole (we don't use pole mask in centerline anyway)
        pole_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(pole_mask, (int(w*0.8), int(h*0.1)), (int(w*0.95), int(h*0.9)), 255, -1)
        masks.append(pole_mask)
        
    # Generate mock wires
    x_end = int(w * 0.8) if has_pole else w
    center_x = x_end * 0.5
    
    if scenario == "unsafe":
        # Below-PEC scenario: lower attachments and deeper sag (~2.5 - 3.5m clearance)
        base_y = int(h * 0.38)
        y_step = int(h * 0.08)
        sag_px_base = int(h * 0.12)
    else:
        # Standard mock: realistic clearance target 5 - 9 m above ground
        base_y = int(h * 0.16)
        y_step = int(h * 0.04)
        sag_px_base = int(h * 0.05)
    
    num_wires = 3
    max_wire_y = int(h * 0.9 - 15) if has_pole else int(h - 15)
    
    for i in range(num_wires):
        mask = np.zeros((h, w), dtype=np.uint8)
        y_attach = base_y + i * y_step
        wire_sag = sag_px_base + int(i * h * 0.01)
        
        pts = []
        for x in range(0, x_end, 5):
            u = (x - center_x) / center_x if center_x > 0 else 0
            y = int(y_attach + wire_sag * (1.0 - u**2))
            y = min(y, max_wire_y)
            if 0 <= y < h:
                pts.append((x, y))
                
        if len(pts) > 2:
            pts_arr = np.array(pts, np.int32)
            cv2.polylines(mask, [pts_arr], False, 255, 6)
            
            # Find bounding box
            x, y, bw, bh = cv2.boundingRect(pts_arr)
            boxes.append([x, y, x + bw, y + bh])
            classes.append(0 if i < 2 else 2) # First two telecom, 3rd dead
            confidences.append(0.85 + i * 0.03)
            masks.append(mask)

    return SegmentationResult(
        mask_available=True,
        boxes=boxes,
        classes=classes,
        confidences=confidences,
        masks=masks,
        names=names
    )
