"""Audit interannotator agreement on an overlapping sample."""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def compute_mask_iou(box1, box2):
    # Simplified IoU for bounding boxes representing masks
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def extract_bbox_from_polygon(parts):
    # Extrapolates bounding box from polygon coords to compute simplified IoU
    coords = [float(x) for x in parts]
    xs = coords[0::2]
    ys = coords[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]

def audit_agreement(ann1_dir: Path, ann2_dir: Path, iou_thresh: float = 0.5):
    ann1_files = set([f.name for f in ann1_dir.glob("*.txt")])
    ann2_files = set([f.name for f in ann2_dir.glob("*.txt")])
    
    overlap = ann1_files.intersection(ann2_files)
    logger.info(f"Found {len(overlap)} overlapping images for audit.")
    
    if not overlap:
        return
        
    agreements = 0
    total_comparisons = 0
    total_iou = 0.0
    
    for filename in overlap:
        with open(ann1_dir / filename) as f1, open(ann2_dir / filename) as f2:
            lines1 = [l.strip().split() for l in f1 if l.strip()]
            lines2 = [l.strip().split() for l in f2 if l.strip()]
            
            # Simple bipartite matching logic for IoU
            matched2 = set()
            for p1 in lines1:
                cls1 = int(p1[0])
                bbox1 = extract_bbox_from_polygon(p1[1:])
                
                best_iou = 0
                best_match = -1
                
                for j, p2 in enumerate(lines2):
                    if j in matched2:
                        continue
                    bbox2 = extract_bbox_from_polygon(p2[1:])
                    iou = compute_mask_iou(bbox1, bbox2)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = j
                        
                if best_iou >= iou_thresh and best_match != -1:
                    matched2.add(best_match)
                    cls2 = int(lines2[best_match][0])
                    if cls1 == cls2:
                        agreements += 1
                    total_iou += best_iou
                total_comparisons += 1
                
    if total_comparisons > 0:
        class_agreement = agreements / total_comparisons
        avg_iou = total_iou / total_comparisons
        logger.info(f"Class Agreement: {class_agreement:.2f}")
        logger.info(f"Average IoU (Mask proxy): {avg_iou:.2f}")
    else:
        logger.info("No comparable instances found.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann1", required=True, help="Annotator 1 labels directory")
    parser.add_argument("--ann2", required=True, help="Annotator 2 labels directory")
    args = parser.parse_args()
    
    audit_agreement(Path(args.ann1), Path(args.ann2))

if __name__ == "__main__":
    main()
