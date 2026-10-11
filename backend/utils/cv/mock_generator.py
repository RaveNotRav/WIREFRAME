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
            
        scenarios = ["safe", "caution", "unsafe", "critical", "nowires", "safe", "caution"]
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
            names={0: "telecom_wire", 1: "power_wire", 2: "dead_abandoned", 3: "pole"}
        )
        
    # Generate mock pole
    has_pole = (scenario != "nopole")
    boxes = []
    classes = []
    confidences = []
    masks = []
    
    names = {0: "telecom_wire", 1: "power_wire", 2: "dead_abandoned", 3: "pole"}
    
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
    # Depending on scenario, we change sag amount
    sag_factor = {
        "safe": 0.05,
        "caution": 0.15,
        "unsafe": 0.30,
        "critical": 0.50,
        "nopole": 0.20
    }.get(scenario, 0.1)
    
    num_wires = 3
    
    for i in range(num_wires):
        # Create a cosh curve
        # y = a * cosh((x - center_x) / a)
        mask = np.zeros((h, w), dtype=np.uint8)
        
        y_offset = int(h * (0.2 + i * 0.15))
        a = w / (sag_factor * 10.0) # Fake parameter
        
        pts = []
        for x in range(0, int(w * 0.8) if has_pole else w, 5):
            center_x = w * 0.4
            y = int(a * np.cosh((x - center_x) / a) - a) + y_offset
            if y < h:
                pts.append((x, y))
                
        if len(pts) > 2:
            pts = np.array(pts, np.int32)
            cv2.polylines(mask, [pts], False, 255, 10)
            
            # Find bounding box
            x, y, bw, bh = cv2.boundingRect(pts)
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
