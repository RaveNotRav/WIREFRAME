"""Merge multiple YOLO datasets into one, applying class mapping and flagging bboxes."""

import argparse
import logging
import shutil
from pathlib import Path

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

def process_dataset(src_dir: Path, out_dir: Path, class_map: dict[str, str], needs_polygon_file: Path) -> None:
    src_images = src_dir / "images"
    src_labels = src_dir / "labels"
    
    if not src_images.exists() or not src_labels.exists():
        logger.warning(f"Skipping {src_dir}: missing images/ or labels/")
        return

    # Find data.yaml
    data_yaml = src_dir / "data.yaml"
    src_class_names = {}
    if data_yaml.exists():
        src_class_names = parse_yolo_yaml(data_yaml)
    else:
        logger.warning(f"No data.yaml in {src_dir}. Cannot map classes accurately by index!")

    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    unmapped_classes = set()
    needs_polygon_images = []

    for label_file in src_labels.rglob("*.txt"):
        # In a real script we'd check the images dir for the actual extension.
        img_file = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = src_images / label_file.parent.relative_to(src_labels) / (label_file.stem + ext)
            if candidate.exists():
                img_file = candidate
                break
        
        if not img_file:
            continue

        with open(label_file, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        is_bbox_only = False
        
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            cls_idx = int(parts[0])
            src_class_name = src_class_names.get(cls_idx, str(cls_idx))
            
            canonical_name = class_map.get(src_class_name)
            if not canonical_name:
                unmapped_classes.add(src_class_name)
                continue
                
            if canonical_name == "DROP":
                continue
                
            if canonical_name not in CANONICAL_CLASSES:
                logger.warning(f"Mapped class {canonical_name} is not in CANONICAL_CLASSES.")
                continue
                
            new_idx = CANONICAL_CLASSES.index(canonical_name)
            
            if len(parts) == 5:
                # BBox format: class cx cy w h
                is_bbox_only = True
                _, cx, cy, w, h = map(float, parts)
                # Convert to rectangular polygon: x1 y1 x2 y1 x2 y2 x1 y2
                x1, y1 = cx - w/2, cy - h/2
                x2, y2 = cx + w/2, cy + h/2
                new_line = f"{new_idx} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}\n"
            else:
                # Polygon format: class x1 y1 x2 y2 ...
                new_line = f"{new_idx} " + " ".join(parts[1:]) + "\n"
                
            new_lines.append(new_line)
            
        if is_bbox_only:
            needs_polygon_images.append(img_file.name)
            
        # Copy image and write new label
        dest_img = out_images / img_file.name
        dest_lbl = out_labels / label_file.name
        if not dest_img.exists():
            shutil.copy2(img_file, dest_img)
        with open(dest_lbl, "w") as f:
            f.writelines(new_lines)
            
    if unmapped_classes:
        logger.warning(f"Unmapped classes found in {src_dir.name}: {', '.join(unmapped_classes)}")
        
    if needs_polygon_images:
        with open(needs_polygon_file, "a") as f:
            f.writelines(f"{img}\n" for img in needs_polygon_images)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", nargs="+", required=True, help="Source dataset directories")
    parser.add_argument("--out", required=True, help="Output dataset directory")
    parser.add_argument("--map", default="backend/ml/classes_map.yaml", help="Class mapping YAML")
    args = parser.parse_args()

    out_dir = Path(args.out)
    map_path = Path(args.map)
    needs_poly = out_dir / "needs_polygon.txt"
    
    if out_dir.exists():
        logger.info(f"Clearing output directory {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    
    class_map = load_class_map(map_path)
    
    for src in args.src:
        logger.info(f"Processing dataset: {src}")
        process_dataset(Path(src), out_dir, class_map, needs_poly)

if __name__ == "__main__":
    main()
