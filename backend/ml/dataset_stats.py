"""Count images and instances per class per split."""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CANONICAL_CLASSES = ["power_line", "telecom_cable", "fiber_optic", "dead_abandoned", "pole"]

def analyze_split(split_dir: Path) -> dict:
    stats = {
        "images": 0,
        "instances": defaultdict(int)
    }
    
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        return stats
        
    for label_file in labels_dir.glob("*.txt"):
        # Check if corresponding image exists
        img_exists = False
        for ext in [".jpg", ".jpeg", ".png"]:
            if (images_dir / (label_file.stem + ext)).exists():
                img_exists = True
                break
        
        if not img_exists:
            continue
            
        stats["images"] += 1
        
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_idx = int(parts[0])
                    stats["instances"][cls_idx] += 1
                    
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Dataset directory to analyze")
    args = parser.parse_args()
    
    dataset_dir = Path(args.dir)
    
    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            # Sometimes data is just in dataset_dir/images/train
            split_dir = dataset_dir
            if not (split_dir / "images" / split).exists():
                continue
            
            # Use nested structure
            stats = {
                "images": len(list((dataset_dir / "images" / split).glob("*.*"))),
                "instances": defaultdict(int)
            }
            labels_dir = dataset_dir / "labels" / split
            if labels_dir.exists():
                for label_file in labels_dir.glob("*.txt"):
                    with open(label_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                stats["instances"][int(parts[0])] += 1
        else:
            stats = analyze_split(split_dir)
            
        if stats["images"] > 0:
            logger.info(f"\nSplit: {split.upper()}")
            logger.info(f"  Images: {stats['images']}")
            logger.info("  Instances:")
            for i, count in sorted(stats["instances"].items()):
                name = CANONICAL_CLASSES[i] if i < len(CANONICAL_CLASSES) else str(i)
                logger.info(f"    - {name}: {count}")

if __name__ == "__main__":
    main()
