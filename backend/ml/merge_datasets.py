"""Merge multiple YOLO datasets into one, applying class mapping and flagging bboxes."""

import argparse
from collections import defaultdict
import logging
from pathlib import Path
import shutil

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_CLASSES = ["power_line", "telecom_cable", "fiber_optic", "dead_abandoned", "pole"]

def load_class_map(map_path: Path) -> dict[str, str]:
    if not map_path.exists():
        logger.warning(f"Class map {map_path} not found. Using empty map.")
        return {}
    with open(map_path, 'r') as f:
        return yaml.safe_load(f) or {}

def parse_yolo_yaml(yaml_path: Path) -> dict[int, str]:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    if 'names' in data:
        if isinstance(data['names'], list):
            return {i: name for i, name in enumerate(data['names'])}
        elif isinstance(data['names'], dict):
            return data['names']
    return {}

def ensure_clean_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        marker = out_dir / ".wireframe_merge"
        has_contents = any(out_dir.iterdir())
        if has_contents and not marker.exists():
            raise RuntimeError(
                f"Output directory '{out_dir}' is non-empty and missing '.wireframe_merge'. "
                f"Refusing to delete. Provide an empty directory or clean it manually."
            )
        logger.info(f"Clearing output directory {out_dir}")
        shutil.rmtree(out_dir)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".wireframe_merge").touch()

def _process_image_files(
    src_images: Path,
    src_labels: Path,
    src_class_names: dict[int, str],
    prefix: str,
    out_images: Path,
    out_labels: Path,
    class_map: dict[str, str],
    needs_polygon_images: list[str],
    unmapped_images: list[tuple[str, list[str]]],
    stats: dict,
    box_policy: str = "skip",
    keep_empty: bool = False,
    allow_unmapped: bool = False,
) -> None:
    for label_file in src_labels.rglob("*.txt"):
        img_file = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = src_images / label_file.parent.relative_to(src_labels) / (label_file.stem + ext)
            if candidate.exists():
                img_file = candidate
                break
        
        if not img_file:
            continue

        stats["images"] += 1

        with open(label_file, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        has_box_line = False
        
        image_kept_per_class = defaultdict(int)
        unmapped_in_image = set()
        
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            cls_idx = int(parts[0])
            src_class_name = src_class_names.get(cls_idx, str(cls_idx))
            
            canonical_name = class_map.get(src_class_name)
            if not canonical_name:
                stats["unmapped_lines"] += 1
                stats["unmapped_names"].add(src_class_name)
                unmapped_in_image.add(src_class_name)
                continue
                
            if canonical_name == "DROP":
                stats["dropped_lines"] += 1
                continue
                
            if canonical_name not in CANONICAL_CLASSES:
                logger.warning(f"Mapped class {canonical_name} is not in CANONICAL_CLASSES.")
                continue
                
            new_idx = CANONICAL_CLASSES.index(canonical_name)
            
            if len(parts) == 5:
                # BBox format: class cx cy w h
                has_box_line = True
                if box_policy == "skip":
                    stats["box_skipped"] += 1
                    continue
                else:
                    _, cx, cy, w, h = map(float, parts)
                    x1, y1 = cx - w/2, cy - h/2
                    x2, y2 = cx + w/2, cy + h/2
                    new_line = f"{new_idx} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}\n"
                    new_lines.append(new_line)
                    image_kept_per_class[canonical_name] += 1
            else:
                new_line = f"{new_idx} " + " ".join(parts[1:]) + "\n"
                new_lines.append(new_line)
                image_kept_per_class[canonical_name] += 1
            
        prefixed_img_name = f"{prefix}{img_file.name}"
        if unmapped_in_image and not allow_unmapped:
            stats["skipped_unmapped"] += 1
            unmapped_images.append((prefixed_img_name, sorted(unmapped_in_image)))
            continue

        if has_box_line:
            needs_polygon_images.append(prefixed_img_name)
            if box_policy == "skip":
                stats["partially_labeled_skipped"] += 1
                continue
            
        if len(new_lines) == 0:
            stats["empty_label_images"].append(prefixed_img_name)
            if not keep_empty:
                continue
                
        # Copy image and write new label with prefix
        dest_img = out_images / prefixed_img_name
        dest_lbl = out_labels / f"{prefix}{label_file.name}"
        if not dest_img.exists():
            shutil.copy2(img_file, dest_img)
        with open(dest_lbl, "w") as f:
            f.writelines(new_lines)

        stats["written_images"] += 1
        stats["written_lines_total"] += len(new_lines)
        for cls_name, count in image_kept_per_class.items():
            stats["kept_per_class"][cls_name] += count

def process_dataset(
    src_dir: Path,
    out_dir: Path,
    class_map: dict[str, str],
    needs_polygon_file: Path,
    box_policy: str = "skip",
    keep_empty: bool = False,
    empty_labels_file: Path | None = None,
    allow_unmapped: bool = False,
    unmapped_images_file: Path | None = None,
) -> dict:

    stats = {
        "dataset": src_dir.name,
        "images": 0,
        "written_images": 0,
        "written_lines_total": 0,
        "kept_per_class": {cls_name: 0 for cls_name in CANONICAL_CLASSES},
        "dropped_lines": 0,
        "unmapped_lines": 0,
        "unmapped_names": set(),
        "skipped_unmapped": 0,
        "box_skipped": 0,
        "partially_labeled_skipped": 0,
        "empty_label_images": [],
    }

    # Find data.yaml in root of src_dir
    data_yaml = src_dir / "data.yaml"
    src_class_names = {}
    if data_yaml.exists():
        src_class_names = parse_yolo_yaml(data_yaml)

    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    needs_polygon_images = []
    unmapped_images = []

    # Detect Roboflow-style layout: train, valid, test (or val) subfolders
    candidate_splits = []
    for split_name in ["train", "valid", "test", "val"]:
        s_dir = src_dir / split_name
        if (s_dir / "images").exists() and (s_dir / "labels").exists():
            candidate_splits.append((split_name, s_dir))

    if candidate_splits:
        for split_name, s_dir in candidate_splits:
            split_yaml = s_dir / "data.yaml"
            split_class_names = parse_yolo_yaml(split_yaml) if split_yaml.exists() else src_class_names
            prefix = f"{src_dir.name}_{split_name}_"
            _process_image_files(
                s_dir / "images",
                s_dir / "labels",
                split_class_names,
                prefix,
                out_images,
                out_labels,
                class_map,
                needs_polygon_images,
                unmapped_images,
                stats,
                box_policy=box_policy,
                keep_empty=keep_empty,
                allow_unmapped=allow_unmapped,
            )
    else:
        src_images = src_dir / "images"
        src_labels = src_dir / "labels"
        if not src_images.exists() or not src_labels.exists():
            logger.warning(f"Skipping {src_dir}: missing images/ or labels/")
            return stats
        if not src_class_names and (src_dir / "data.yaml").exists():
            src_class_names = parse_yolo_yaml(src_dir / "data.yaml")
        prefix = f"{src_dir.name}_"
        _process_image_files(
            src_images,
            src_labels,
            src_class_names,
            prefix,
            out_images,
            out_labels,
            class_map,
            needs_polygon_images,
            unmapped_images,
            stats,
            box_policy=box_policy,
            keep_empty=keep_empty,
            allow_unmapped=allow_unmapped,
        )

    if needs_polygon_images:
        with open(needs_polygon_file, "a") as f:
            f.writelines(f"{img}\n" for img in needs_polygon_images)

    if empty_labels_file and stats["empty_label_images"]:
        with open(empty_labels_file, "a") as f:
            f.writelines(f"{img}\n" for img in stats["empty_label_images"])

    if unmapped_images:
        if unmapped_images_file is None:
            unmapped_images_file = out_dir / "unmapped_images.txt"
        with open(unmapped_images_file, "a") as f:
            for img, u_classes in unmapped_images:
                f.write(f"{img}: {', '.join(u_classes)}\n")

    # Log per-dataset report
    logger.info(f"\n--- Dataset: {src_dir.name} ---")
    logger.info(f"  Images processed: {stats['images']}")
    logger.info(f"  Images written: {stats['written_images']}")
    logger.info(f"  Lines written total: {stats['written_lines_total']}")
    logger.info(f"  Lines kept per class: {stats['kept_per_class']}")
    logger.info(f"  Lines dropped (DROP): {stats['dropped_lines']}")
    unmapped_str = ", ".join(sorted(stats["unmapped_names"])) if stats["unmapped_names"] else "none"
    logger.info(f"  Lines unmapped: {stats['unmapped_lines']} (classes: {unmapped_str})")
    logger.info(f"  Skipped for unmapped: {stats['skipped_unmapped']}")
    logger.info(f"  Box-only skipped: {stats['box_skipped']}")
    logger.info(f"  Partially labeled skipped: {stats['partially_labeled_skipped']}")
    logger.info(f"  Images with empty labels: {len(stats['empty_label_images'])}")

    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", nargs="+", required=True, help="Source dataset directories")
    parser.add_argument("--out", required=True, help="Output dataset directory")
    parser.add_argument("--map", default="backend/ml/classes_map.yaml", help="Class mapping YAML")
    parser.add_argument("--box-policy", choices=["skip", "convert"], default="skip", help="How to handle bounding boxes: skip or convert to polygon")
    parser.add_argument("--keep-empty", action="store_true", help="Keep images with empty labels as background negatives")
    parser.add_argument("--allow-unmapped", action="store_true", help="Allow images containing unmapped classes to be written with unmapped lines omitted")
    args = parser.parse_args()

    out_dir = Path(args.out)
    map_path = Path(args.map)
    needs_poly = out_dir / "needs_polygon.txt"
    empty_labels = out_dir / "empty_labels.txt"
    unmapped_file = out_dir / "unmapped_images.txt"
    
    ensure_clean_output_dir(out_dir)
    class_map = load_class_map(map_path)
    
    for src in args.src:
        logger.info(f"Processing dataset: {src}")
        process_dataset(
            Path(src),
            out_dir,
            class_map,
            needs_poly,
            box_policy=args.box_policy,
            keep_empty=args.keep_empty,
            empty_labels_file=empty_labels,
            allow_unmapped=args.allow_unmapped,
            unmapped_images_file=unmapped_file,
        )

if __name__ == "__main__":
    main()

