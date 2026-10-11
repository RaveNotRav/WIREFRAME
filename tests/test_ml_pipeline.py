"""Tests for the ML Pipeline scripts."""

import json

import numpy as np
import pytest
import yaml

from backend.ml.audit_agreement import compute_mask_iou, extract_bbox_from_polygon
from backend.ml.clean_dataset import clean_dataset
from backend.ml.merge_datasets import load_class_map, process_dataset
from backend.ml.split_dataset import split_dataset


@pytest.fixture
def synthetic_dataset(tmp_path):
    # Create a tiny synthetic YOLO dataset
    dataset_dir = tmp_path / "synthetic_dataset"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # data.yaml
    data_yaml = dataset_dir / "data.yaml"
    with open(data_yaml, "w") as f:
        yaml.dump({"names": ["Cable", "tree", "power_pole"]}, f)

    # 3 images
    import cv2
    # Use random noise for img1 and img2 so they pass the blur check
    rng = np.random.RandomState(42)
    img1 = rng.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    img2 = img1.copy() # exact duplicate of img1
    img3 = np.full((100, 100, 3), 10, dtype=np.uint8)  # too dark

    cv2.imwrite(str(images_dir / "img1.jpg"), img1)
    cv2.imwrite(str(images_dir / "img2.jpg"), img2)
    cv2.imwrite(str(images_dir / "img3.jpg"), img3)

    # Labels
    # img1: 1 polygon (Cable)
    with open(labels_dir / "img1.txt", "w") as f:
        f.write("0 0.1 0.1 0.2 0.2 0.3 0.3\n")
    # img2: 1 bbox (power_pole)
    with open(labels_dir / "img2.txt", "w") as f:
        f.write("2 0.5 0.5 0.2 0.2\n")
    # img3: 1 polygon (tree - to be dropped)
    with open(labels_dir / "img3.txt", "w") as f:
        f.write("1 0.1 0.1 0.2 0.2 0.3 0.3\n")
        
    return dataset_dir

@pytest.fixture
def class_map(tmp_path):
    map_file = tmp_path / "map.yaml"
    with open(map_file, "w") as f:
        yaml.dump({
            "Cable": "telecom_cable",
            "power_pole": "pole",
            "tree": "DROP"
        }, f)
    return load_class_map(map_file)


def test_merge_datasets(synthetic_dataset, class_map, tmp_path):
    out_dir = tmp_path / "merged"
    needs_poly = out_dir / "needs_polygon.txt"
    out_dir.mkdir()
    
    process_dataset(synthetic_dataset, out_dir, class_map, needs_poly)
    
    out_labels = out_dir / "labels"
    assert (out_labels / "img1.txt").exists()
    assert (out_labels / "img2.txt").exists()
    
    # img1 Cable -> telecom_cable (idx 1)
    with open(out_labels / "img1.txt") as f:
        assert f.read().startswith("1 0.1 0.1")
        
    # img2 power_pole -> pole (idx 4), bbox converted to poly
    with open(out_labels / "img2.txt") as f:
        line = f.read().strip()
        parts = line.split()
        assert int(parts[0]) == 4
        assert len(parts) == 9 # 1 class + 8 coords
        
    # img3 tree -> dropped
    with open(out_labels / "img3.txt") as f:
        assert f.read().strip() == ""
        
    assert needs_poly.exists()
    assert "img2.jpg" in needs_poly.read_text()


def test_clean_dataset(synthetic_dataset, tmp_path):
    out_dir = tmp_path / "cleaned"
    clean_dataset(synthetic_dataset, out_dir)
    
    report_file = out_dir / "cleaning_report.json"
    assert report_file.exists()
    with open(report_file) as f:
        report = json.load(f)
        
    assert report["total_processed"] == 3
    assert len(report["too_dark"]) == 1
    assert "img3.jpg" in report["too_dark"]
    
    assert len(report["duplicates"]) == 1
    
    assert report["kept"] == 1


def test_split_dataset(tmp_path):
    # Dummy dataset
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)
    
    for i in range(10):
        (src / "images" / f"img{i}.jpg").touch()
        with open(src / "labels" / f"img{i}.txt", "w") as f:
            f.write("0 0.5 0.5 0.1 0.1\n")
            
    out = tmp_path / "split"
    split_dataset(src, out, 0.7, 0.2, 0.1, 42)
    
    assert len(list((out / "images" / "train").glob("*.jpg"))) == 7
    assert len(list((out / "images" / "val").glob("*.jpg"))) == 2
    assert len(list((out / "images" / "test").glob("*.jpg"))) == 1


def test_audit_agreement():
    # bbox format: xmin, ymin, xmax, ymax
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    iou = compute_mask_iou(box1, box2)
    assert iou == pytest.approx(25 / 175) # inter 25, union 100+100-25=175
    
    parts = ["0.0", "0.0", "10.0", "0.0", "10.0", "10.0", "0.0", "10.0"]
    bbox = extract_bbox_from_polygon(parts)
    assert bbox == [0.0, 0.0, 10.0, 10.0]
