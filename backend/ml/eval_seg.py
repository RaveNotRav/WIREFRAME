"""Evaluate YOLO11-seg model on the held-out test split."""

import argparse
import logging

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model (.pt)")
    parser.add_argument("--config", default="backend/ml/wireframe_seg.yaml", help="Dataset config")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate on")
    args = parser.parse_args()

    logger.info(f"Loading model {args.model}")
    model = YOLO(args.model)
    
    logger.info(f"Evaluating on {args.split} split from {args.config}")
    metrics = model.val(data=args.config, split=args.split)
    
    # Print metrics
    logger.info("\n--- Evaluation Results ---")
    logger.info(f"mAP50:    {metrics.seg.map50:.4f}")
    logger.info(f"mAP50-95: {metrics.seg.map:.4f}")
    
    # Per class metrics usually available in metrics.class_result or similar
    # For ultralytics YOLO:
    if hasattr(metrics.seg, 'p') and len(metrics.seg.p) > 0:
        logger.info(f"Precision: {metrics.seg.p.mean():.4f}")
        logger.info(f"Recall:    {metrics.seg.r.mean():.4f}")
        logger.info(f"F1 Score:  {metrics.seg.f1.mean():.4f}")
    
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()
