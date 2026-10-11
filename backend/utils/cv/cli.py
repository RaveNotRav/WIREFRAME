"""CLI for testing the CV pipeline visually.

Usage: python -m backend.utils.cv.cli sample.jpg --out out/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from backend.utils.cv.catenary import calculate_sag_metrics
from backend.utils.cv.centerline import extract_centerline
from backend.utils.cv.preprocess import validate_and_preprocess
from backend.utils.cv.segment import segment_image
from backend.utils.cv.vni import compute_vni


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--out", required=True, help="Output directory for visual overlays")
    parser.add_argument("--model", default="backend/models/wireframe_seg.pt", help="Model path")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Preprocess
    prep = validate_and_preprocess(args.image)
    if not prep.quality_ok:
        print(f"Image rejected: {prep.reason}")
        return
        
    print("Preprocessing OK.")
    
    # VNI
    vni_res = compute_vni(prep.gray)
    print(f"VNI: {vni_res.vni:.4f}")
    if vni_res.spectrum_png:
        with open(out_dir / "spectrum.png", "wb") as f:
            f.write(vni_res.spectrum_png)
            
    # Segment
    print(f"Loading model {args.model}...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None
        
    overlay = prep.rgb.copy()
    
    if model:
        seg_res = segment_image(prep.rgb, model, prep.original_shape)
        if seg_res.mask_available:
            print(f"Found {len(seg_res.masks)} masks.")
            # Overlay masks
            for i, mask in enumerate(seg_res.masks):
                # Colorize mask
                color = (0, 255, 0)
                colored_mask = np.zeros_like(prep.rgb)
                colored_mask[mask > 0] = color
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)
                
                # Centerline
                cline = extract_centerline(mask, seg_res.boxes[i])
                if cline.is_valid:
                    print(f"Wire {i}: Extracted centerline ({len(cline.polyline)} points)")
                    # Draw centerline
                    for pt in cline.polyline:
                        cv2.circle(overlay, pt, 1, (255, 0, 0), -1)
                        
                    # Catenary
                    cat = calculate_sag_metrics(cline.polyline)
                    print(f"  Sag to Span: {cat.sag_to_span:.4f}")
                    # Draw fit
                    if cat.fitted_points:
                        pts = np.array(cat.fitted_points, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(overlay, [pts], False, (0, 0, 255), 2)
        else:
            print("No masks available (detect-only).")
            
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / "overlay.jpg"), overlay_bgr)
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
