"""Phase 3 Checkpoint CLI script."""

import argparse
import json
import sys

import cv2
import numpy as np

from backend.config_store import init_config_store
from backend.database import get_connection, run_migrations
from backend.utils.analysis_engine import analyze_image
from backend.utils.cv.catenary import cosh_func
from backend.utils.cv.types import SegmentationResult


class DummyModel:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, image, **kwargs):
        return [self]


def _generate_synthetic_data(w, h):
    # Image with some noise
    image = np.full((h, w, 3), 200, dtype=np.uint8)
    
    # 1. Pole: 100px tall
    # Let's say ground is at y=500
    cv2.rectangle(image, (10, 400), (30, 500), (100, 100, 100), -1)
    
    # 2. Wire: catenary
    a = 200.0
    x0 = 320.0
    b = 100.0 - a  # min y = 100
    
    x_vals = np.arange(50, 590)
    y_vals = cosh_func(x_vals, a, b, x0)
    
    for x, y in zip(x_vals, y_vals):
        cv2.circle(image, (int(x), int(y)), 2, (0, 0, 0), -1)
        
    cv2.imwrite("synthetic_phase3.jpg", image)
    
    # Generate perfect segmentation mask for the model
    mask_wire = np.zeros((h, w), dtype=np.uint8)
    for x, y in zip(x_vals, y_vals):
        cv2.circle(mask_wire, (int(x), int(y)), 4, 1, -1)
        
    return image, mask_wire


class SyntheticYOLO:
    def __init__(self, mask_wire, box_wire, box_pole):
        self.mask_wire = mask_wire
        self.box_wire = box_wire
        self.box_pole = box_pole
        
    def __call__(self, image, **kwargs):
        class MockResult:
            pass
        res = MockResult()
        
        class MockMasks:
            pass
        m = MockMasks()
        m.data = np.stack([self.mask_wire, np.zeros_like(self.mask_wire)], axis=0) # 2 masks
        res.masks = m
        
        class MockBoxes:
            pass
        b = MockBoxes()
        # [x1, y1, x2, y2]
        b.data = np.array([self.box_wire, self.box_pole])
        b.cls = np.array([0, 1])
        b.conf = np.array([0.9, 0.95])
        res.boxes = b
        
        res.names = {0: "power_line", 1: "pole"}
        
        # Monkey patch segment_image to return this directly, because 
        # segment.py expects ultralytics Result objects which are complex to mock perfectly.
        return [res]


def patch_segment_image(mask_wire, box_wire, box_pole):
    from backend.utils import analysis_engine
    def mock_segment(img, model, orig_shape):
        masks = [mask_wire]
        return SegmentationResult(
            mask_available=True,
            masks=masks,
            classes=[0, 1],
            names={0: "power_line", 1: "pole"},
            confidences=[0.9, 0.95],
            boxes=[box_wire, box_pole]
        )
    analysis_engine.segment_image = mock_segment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Input image path (ignored if --synthetic)")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic generated data")
    args = parser.parse_args()

    # Init DB and configs
    conn = get_connection()
    run_migrations(conn)
    init_config_store(conn)
    
    if args.synthetic:
        print("Generating synthetic image (synthetic_phase3.jpg)...")
        w, h = 640, 640
        _, mask_wire = _generate_synthetic_data(w, h)
        
        # wire box
        # y min is 100, y max is roughly cosh(270/200)*200 + b ~ 200*2.05 - 100 = 310
        box_wire = [50, 100, 590, 310]
        # pole box, ground is 500, top is 400
        box_pole = [10, 400, 30, 500]
        
        patch_segment_image(mask_wire, box_wire, box_pole)
        image_path = "synthetic_phase3.jpg"
        model = None
    else:
        if not args.image:
            print("Error: must provide image path or --synthetic")
            sys.exit(1)
        image_path = args.image
        from ultralytics import YOLO
        try:
            model = YOLO("backend/models/wireframe_seg.pt")
        except Exception:
            print("Could not load YOLO model.")
            sys.exit(1)

    print("--- RUN 1: Standard Pipeline (Pole Calibration) ---")
    conn.execute("INSERT OR IGNORE INTO analyses (id, filename, status) VALUES (999, 'synthetic.jpg', 'processing')")
    conn.commit()
    res1 = analyze_image(image_path, model, analysis_id=999, db_conn=conn)
    
    if res1["status"] != "success":
        print(f"Analysis failed: {res1.get('reason')}")
        sys.exit(1)
        
    print(f"\nImage Safety Level: {res1['image_safety_level']}")
    print(f"Limitation: {res1['limitation']}\n")
    
    print(f"{'Class':<15} | {'CRS':<5} | {'Band':<10} | {'Clearance (m)':<15} | {'Status':<10} | {'Safety'}")
    print("-" * 80)
    for w in res1["wires"]:
        params = json.loads(w["params_json"])
        clear_str = f"{w['clearance_m']:.2f} [{params['clearance_low_m']:.2f}-{params['clearance_high_m']:.2f}]" if w['clearance_m'] else "N/A"
        print(f"{w['class_name']:<15} | {w['crs']:<5.1f} | {w['band']:<10} | {clear_str:<15} | {w['clearance_status']:<10} | {w['safety_level']}")
        for reason in w['safety_reasons']:
            print(f"  - {reason}")
            

    print("\n\n--- RUN 2: Manual Calibration Override ---")
    # Patch calibrate_scale to simulate manual points override
    from backend.utils.cv import calibration
    orig_calibrate = calibration.calibrate_scale
    def mock_calibrate_scale(**kwargs):
        return orig_calibrate(
            manual_pt1=(0,0), manual_pt2=(100,0), manual_length_m=5.0, # 0.05 m/px
            pole_bbox=kwargs.get("pole_bbox"),
            image_height=kwargs.get("image_height")
        )
    calibration.calibrate_scale = mock_calibrate_scale
    
    conn.execute("INSERT OR IGNORE INTO analyses (id, filename, status) VALUES (1000, 'synthetic.jpg', 'processing')")
    conn.commit()
    res2 = analyze_image(image_path, model, analysis_id=1000, db_conn=conn)
    
    print(f"\nImage Safety Level: {res2['image_safety_level']}")
    print(f"{'Class':<15} | {'CRS':<5} | {'Band':<10} | {'Clearance (m)':<15} | {'Status':<10} | {'Safety'}")
    print("-" * 80)
    for w in res2["wires"]:
        params = json.loads(w["params_json"])
        clear_str = f"{w['clearance_m']:.2f} [{params['clearance_low_m']:.2f}-{params['clearance_high_m']:.2f}]" if w['clearance_m'] else "N/A"
        print(f"{w['class_name']:<15} | {w['crs']:<5.1f} | {w['band']:<10} | {clear_str:<15} | {w['clearance_status']:<10} | {w['safety_level']}")

    print("\n(Check database to verify records were inserted for analysis_id=999 and 1000)")

if __name__ == "__main__":
    main()
