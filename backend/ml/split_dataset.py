"""Split dataset into train, val, and test sets with a set seed."""

import argparse
from collections import defaultdict
import hashlib
import cv2
import logging
from pathlib import Path
import random
import re
import shutil

try:
    from backend.ml.clean_dataset import dhash
except ImportError:
    from clean_dataset import dhash

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def compute_file_hash(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_image_dhash(filepath: Path) -> int:
    img = cv2.imread(str(filepath))
    if img is None:
        raise ValueError(f"Could not read image for perceptual hash: {filepath}")
    return dhash(img)

def get_group_key(name: str, regex: str | None) -> str:
    if not regex:
        return name
    match = re.search(regex, name)
    if not match:
        return name
    if match.lastindex:
        return match.group(1)
    return match.group(0)

def split_dataset(
    src_dir: Path,
    out_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_regex: str | None = None,
    fixed_test_dir: Path | None = None,
    near_dup_threshold: int | None = None,
):
    random.seed(seed)
    
    if not group_regex:
        logger.warning("No --group-regex specified: leakage cannot be detected and split is random per image.")

    images_dir = src_dir / "images"
    labels_dir = src_dir / "labels"
    
    if not images_dir.exists():
        logger.error(f"Images directory not found in {src_dir}")
        return
        
    src_images = [f.name for f in images_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    fixed_test_names: set[str] = set()
    
    if fixed_test_dir:
        fixed_img_dir = fixed_test_dir / "images"
        fixed_lbl_dir = fixed_test_dir / "labels"
        if not fixed_img_dir.exists():
            logger.error(f"Fixed test images directory not found in {fixed_test_dir}")
            return
            
        fixed_images = [f.name for f in fixed_img_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        splits["test"] = sorted(fixed_images)
        fixed_test_names = set(fixed_images)
        
        # Check duplicate content between fixed test and source pool
        fixed_hashes: dict[str, str] = {}
        for img in splits["test"]:
            h = compute_file_hash(fixed_img_dir / img)
            fixed_hashes[h] = img

        pool = [img for img in src_images if img not in fixed_test_names]
        for img in pool:
            h = compute_file_hash(images_dir / img)
            if h in fixed_hashes:
                dup = fixed_hashes[h]
                raise ValueError(
                    f"Duplicate image content detected between fixed-test ({dup}) and dataset pool ({img}) [SHA-256: {h}]"
                )

        if near_dup_threshold is not None:
            fixed_dhashes: dict[str, int] = {}
            for img in splits["test"]:
                fixed_dhashes[img] = compute_image_dhash(fixed_img_dir / img)

            pool_dhashes: dict[str, int] = {}
            for img in pool:
                pool_dhashes[img] = compute_image_dhash(images_dir / img)

            near_dup_pairs = []
            for f_img, f_h in fixed_dhashes.items():
                for p_img, p_h in pool_dhashes.items():
                    dist = (f_h ^ p_h).bit_count()
                    if dist <= near_dup_threshold:
                        near_dup_pairs.append((f_img, p_img, dist))

            if near_dup_pairs:
                pairs_str = ", ".join(f"({f}, {p}, dist={d})" for f, p, d in near_dup_pairs)
                raise ValueError(
                    f"Near-duplicate image content detected between fixed-test and train/val (threshold {near_dup_threshold}): {pairs_str}"
                )

        # Copy fixed test set directly
        out_test_img = out_dir / "images" / "test"
        out_test_lbl = out_dir / "labels" / "test"
        out_test_img.mkdir(parents=True, exist_ok=True)
        out_test_lbl.mkdir(parents=True, exist_ok=True)
        for img in splits["test"]:
            shutil.copy2(fixed_img_dir / img, out_test_img / img)
            lbl = Path(img).stem + ".txt"
            if (fixed_lbl_dir / lbl).exists():
                shutil.copy2(fixed_lbl_dir / lbl, out_test_lbl / lbl)
                
        # Re-normalize train/val ratios to sum to 1.0
        tv_total = train_ratio + val_ratio
        r_train = train_ratio / tv_total if tv_total > 0 else 1.0
        r_val = val_ratio / tv_total if tv_total > 0 else 0.0
        r_test = 0.0
    else:
        pool = list(src_images)
        r_train = train_ratio
        r_val = val_ratio
        r_test = test_ratio

    if group_regex:
        groups = defaultdict(list)
        for img in pool:
            key = get_group_key(img, group_regex)
            groups[key].append(img)
            
        sorted_keys = sorted(groups.keys())
        random.shuffle(sorted_keys)
        
        total_items = len(pool)
        target_train = max(1, int(total_items * r_train)) if r_train > 0 else 0
        target_val = max(1, int(total_items * r_val)) if r_val > 0 else 0
        
        for key_idx, key in enumerate(sorted_keys):
            key_imgs = groups[key]
            if r_test > 0 and key_idx == len(sorted_keys) - 1 and not splits["test"]:
                splits["test"].extend(key_imgs)
            elif r_test > 0 and key_idx == len(sorted_keys) - 2 and not splits["val"]:
                splits["val"].extend(key_imgs)
            elif len(splits["train"]) < target_train or (not splits["train"] and target_train > 0):
                splits["train"].extend(key_imgs)
            elif (r_test > 0 and len(splits["val"]) < target_val) or (r_test == 0 and len(splits["val"]) + len(key_imgs) <= target_val + 1):
                splits["val"].extend(key_imgs)
            elif r_test > 0:
                splits["test"].extend(key_imgs)
            else:
                splits["val"].extend(key_imgs)
    else:
        pool.sort()
        random.shuffle(pool)
        n = len(pool)
        n_train = int(n * r_train)
        n_val = int(n * r_val)
        splits["train"] = pool[:n_train]
        splits["val"] = pool[n_train:n_train + n_val]
        if not fixed_test_dir:
            splits["test"] = pool[n_train + n_val:]

    # Content-hash check between train and val
    train_hashes: dict[str, str] = {}
    for img in splits["train"]:
        h = compute_file_hash(images_dir / img)
        train_hashes[h] = img

    for img in splits["val"]:
        h = compute_file_hash(images_dir / img)
        if h in train_hashes:
            dup = train_hashes[h]
            raise ValueError(
                f"Duplicate image content detected between train ({dup}) and val ({img}) [SHA-256: {h}]"
            )

    if not fixed_test_dir and splits["test"]:
        train_val_hashes = dict(train_hashes)
        for img in splits["val"]:
            train_val_hashes[compute_file_hash(images_dir / img)] = f"val:{img}"
        for img in splits["test"]:
            h = compute_file_hash(images_dir / img)
            if h in train_val_hashes:
                dup = train_val_hashes[h]
                raise ValueError(
                    f"Duplicate image content detected between test ({img}) and {dup} [SHA-256: {h}]"
                )

    if near_dup_threshold is not None and not fixed_test_dir:
        train_dhashes = {img: compute_image_dhash(images_dir / img) for img in splits["train"]}
        val_dhashes = {img: compute_image_dhash(images_dir / img) for img in splits["val"]}
        test_dhashes = {img: compute_image_dhash(images_dir / img) for img in splits["test"]}

        near_dup_pairs = []
        for tr_img, tr_h in train_dhashes.items():
            for v_img, v_h in val_dhashes.items():
                dist = (tr_h ^ v_h).bit_count()
                if dist <= near_dup_threshold:
                    near_dup_pairs.append((f"train:{tr_img}", f"val:{v_img}", dist))

        for t_img, t_h in test_dhashes.items():
            for tv_split, dhashes in [("train", train_dhashes), ("val", val_dhashes)]:
                for tv_img, tv_h in dhashes.items():
                    dist = (t_h ^ tv_h).bit_count()
                    if dist <= near_dup_threshold:
                        near_dup_pairs.append((f"{tv_split}:{tv_img}", f"test:{t_img}", dist))

        if near_dup_pairs:
            pairs_str = ", ".join(f"({a}, {b}, dist={d})" for a, b, d in near_dup_pairs)
            raise ValueError(
                f"Near-duplicate image content detected between splits (threshold {near_dup_threshold}): {pairs_str}"
            )

    # Copy train and val (and test if not fixed) from src_dir
    for split in ["train", "val", "test"]:
        if split == "test" and fixed_test_dir:
            continue
        out_images = out_dir / "images" / split
        out_labels = out_dir / "labels" / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        for img in splits[split]:
            shutil.copy2(images_dir / img, out_images / img)
            lbl = Path(img).stem + ".txt"
            if (labels_dir / lbl).exists():
                shutil.copy2(labels_dir / lbl, out_labels / lbl)

    # Group counts & Leakage check
    if group_regex:
        split_groups: dict[str, set[str]] = {}
        for split, split_images in splits.items():
            keys = {get_group_key(img, group_regex) for img in split_images}
            split_groups[split] = keys
            logger.info(f"{split.upper()}: {len(split_images)} images across {len(keys)} groups")
            
        overlap_train_val = split_groups["train"] & split_groups["val"]
        overlap_train_test = split_groups["train"] & split_groups["test"]
        overlap_val_test = split_groups["val"] & split_groups["test"]
        all_overlap = overlap_train_val | overlap_train_test | overlap_val_test
        
        if all_overlap:
            logger.error(f"LEAKAGE DETECTED! Overlapping groups: {all_overlap}")
            raise ValueError(f"Group leakage detected between splits: {all_overlap}")
        else:
            logger.info("Leakage check passed: no group key in more than one split.")
    else:
        for split, split_images in splits.items():
            logger.info(f"{split.upper()}: {len(split_images)} images")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Cleaned dataset directory")
    parser.add_argument("--out", required=True, help="Output split dataset directory")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.2, help="Val ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--group-regex", default=None, help="Regex pattern to group images into same split")
    parser.add_argument("--fixed-test", default=None, help="Directory containing fixed test images/ and labels/")
    parser.add_argument("--near-dup-threshold", type=int, default=None, help="Perceptual hash Hamming distance threshold to reject near-duplicate images across splits")
    args = parser.parse_args()
    
    total = args.train + args.val + args.test
    fixed_test_path = Path(args.fixed_test) if args.fixed_test else None
    
    split_dataset(
        Path(args.src),
        Path(args.out),
        args.train / total,
        args.val / total,
        args.test / total,
        args.seed,
        group_regex=args.group_regex,
        fixed_test_dir=fixed_test_path,
        near_dup_threshold=args.near_dup_threshold,
    )

if __name__ == "__main__":
    main()

