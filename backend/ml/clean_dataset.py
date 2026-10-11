"""Clean dataset: remove duplicates (dHash), corrupt images, too dark, or too blurry."""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def dhash(image: np.ndarray, hash_size: int = 8) -> str:
    """Compute difference hash (dHash) for perceptual deduplication."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def clean_dataset(dataset_dir: Path, out_dir: Path):
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    
    report = {
        "total_processed": 0,
        "corrupt": [],
        "duplicates": [],
        "too_dark": [],
        "too_blurry": [],
        "kept": 0
    }
    
    seen_hashes = {}
    
    for img_path in images_dir.glob("*.*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
            
        report["total_processed"] += 1
        
        # 1. Corrupt check
        img = cv2.imread(str(img_path))
        if img is None:
            report["corrupt"].append(img_path.name)
            continue
            
        # 2. Duplicate check (Perceptual Hash)
        h = dhash(img)
        if h in seen_hashes:
            report["duplicates"].append({"file": img_path.name, "duplicate_of": seen_hashes[h]})
            continue
            
        # 3. Brightness check
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_bright = np.mean(gray)
        if mean_bright < 15.0: # Extremely dark
            report["too_dark"].append(img_path.name)
            continue
            
        # 4. Blur check (Laplacian Variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50.0: # Very blurry
            report["too_blurry"].append(img_path.name)
            continue
            
        # Keep image and its label
        seen_hashes[h] = img_path.name
        report["kept"] += 1
        
        # Copy image
        import shutil
        shutil.copy2(img_path, out_images / img_path.name)
        
        # Copy label if exists
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy2(label_path, out_labels / label_path.name)
            
    with open(out_dir / "cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Cleaning complete. Kept {report['kept']}/{report['total_processed']} images.")
    logger.info(f"Report saved to {out_dir / 'cleaning_report.json'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source dataset directory (with images/ and labels/)")
    parser.add_argument("--out", required=True, help="Output cleaned dataset directory")
    args = parser.parse_args()
    
    clean_dataset(Path(args.src), Path(args.out))

if __name__ == "__main__":
    main()
