"""Split dataset into train, val, and test sets with a set seed."""

import argparse
import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def split_dataset(src_dir: Path, out_dir: Path, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    random.seed(seed)
    
    images_dir = src_dir / "images"
    labels_dir = src_dir / "labels"
    
    if not images_dir.exists():
        logger.error(f"Images directory not found in {src_dir}")
        return
        
    images = [f.name for f in images_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    images.sort() # Ensure reproducible shuffling
    random.shuffle(images)
    
    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }
    
    for split, split_images in splits.items():
        out_images = out_dir / "images" / split
        out_labels = out_dir / "labels" / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        
        for img in split_images:
            shutil.copy2(images_dir / img, out_images / img)
            
            lbl = Path(img).stem + ".txt"
            if (labels_dir / lbl).exists():
                shutil.copy2(labels_dir / lbl, out_labels / lbl)
                
        logger.info(f"{split.upper()}: {len(split_images)} images")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Cleaned dataset directory")
    parser.add_argument("--out", required=True, help="Output split dataset directory")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.2, help="Val ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Normalize ratios if they don't sum to 1.0 exactly due to user error
    total = args.train + args.val + args.test
    
    split_dataset(Path(args.src), Path(args.out), args.train/total, args.val/total, args.test/total, args.seed)

if __name__ == "__main__":
    main()
