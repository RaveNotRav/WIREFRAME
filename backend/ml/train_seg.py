"""Train YOLO11-seg model on the wireframe dataset."""

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="backend/ml/wireframe_seg.yaml", help="Dataset config")
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Base model (e.g., yolo11n-seg.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=30, help="Patience epochs for early stopping")
    parser.add_argument("--device", default=None, help="Device to run on (e.g., cuda:0 or cpu)")
    parser.add_argument("--limit", type=int, default=0, help="Limit dataset size for smoke testing")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--project", default="runs/segment", help="Save directory")
    parser.add_argument("--name", default="wireframe_seg", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Log legacy model if present
    legacy_model = Path("backend/models/dataset11_cabledetection.pt")
    if legacy_model.exists():
        logger.warning(f"Legacy model found at {legacy_model}. It is detect-only. Training new segmentation model.")

    # Recommendation: Use --batch 8 at --imgsz 1024 on 16 GB GPUs to avoid CUDA out-of-memory errors
    logger.info("Recommendation: Use --batch 8 at --imgsz 1024 on 16 GB GPUs.")

    if args.resume:
        model = YOLO(Path(args.project) / args.name / "weights" / "last.pt")
        logger.info(f"Resuming training from {model.ckpt_path}")
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        logger.info(f"Starting training with model: {args.model}")
        
        train_args = {
            "data": args.config,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "lr0": args.lr0,
            "project": args.project,
            "name": args.name,
            "seed": args.seed,
            "patience": args.patience,
        }
        if args.device is not None:
            train_args["device"] = args.device
        
        if args.limit > 0:
            logger.info(f"SMOKE TEST MODE: limit={args.limit}. Not using fraction to avoid dropping tiny datasets.")
            # We'll just rely on the user passing a small dataset or we don't truncate.
            
        model.train(**train_args)
        
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
