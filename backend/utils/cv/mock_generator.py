import hashlib
import os

import cv2
import numpy as np

from backend.utils.cv.types import SegmentationResult


def generate_mock_segmentation(image_path: str, original_shape: tuple[int, int]) -> SegmentationResult:
    """
    Generates deterministic synthetic segmentation outputs based on the file content.
    Filename keywords override the deterministic scenario.
    """
    h, w = original_shape
    
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
