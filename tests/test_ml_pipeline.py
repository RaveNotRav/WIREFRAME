"""Tests for the ML Pipeline scripts."""

import json
import re

import numpy as np
import pytest
import yaml

from backend.ml.audit_agreement import compute_mask_iou, extract_bbox_from_polygon
from backend.ml.clean_dataset import clean_dataset
from backend.ml.merge_datasets import ensure_clean_output_dir, load_class_map, process_dataset
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
    empty_labels = out_dir / "empty_labels.txt"
    out_dir.mkdir()
    
    process_dataset(synthetic_dataset, out_dir, class_map, needs_poly, box_policy="convert", keep_empty=True, empty_labels_file=empty_labels)
    
    out_labels = out_dir / "labels"
    assert (out_labels / "synthetic_dataset_img1.txt").exists()
    assert (out_labels / "synthetic_dataset_img2.txt").exists()
    
    # img1 Cable -> telecom_cable (idx 1)
    with open(out_labels / "synthetic_dataset_img1.txt") as f:
        assert f.read().startswith("1 0.1 0.1")
        
    # img2 power_pole -> pole (idx 4), bbox converted to poly
    with open(out_labels / "synthetic_dataset_img2.txt") as f:
        line = f.read().strip()
        parts = line.split()
        assert int(parts[0]) == 4
        assert len(parts) == 9 # 1 class + 8 coords
        
    # img3 tree -> dropped, keep_empty=True so file exists but is empty
    with open(out_labels / "synthetic_dataset_img3.txt") as f:
        assert f.read().strip() == ""
        
    assert needs_poly.exists()
    assert "synthetic_dataset_img2.jpg" in needs_poly.read_text()


def test_merge_datasets_box_skip_and_empty_report(synthetic_dataset, class_map, tmp_path):
    out_dir = tmp_path / "merged_skip"
    needs_poly = out_dir / "needs_polygon.txt"
    empty_labels = out_dir / "empty_labels.txt"
    out_dir.mkdir()
    
    # Default box_policy="skip", keep_empty=False
    stats = process_dataset(
        synthetic_dataset,
        out_dir,
        class_map,
        needs_poly,
        box_policy="skip",
        keep_empty=False,
        empty_labels_file=empty_labels,
    )
    
    out_labels = out_dir / "labels"
    # img1 kept
    assert (out_labels / "synthetic_dataset_img1.txt").exists()
    # img2 had only a box, so it is skipped. With keep_empty=False, no label file created
    assert not (out_labels / "synthetic_dataset_img2.txt").exists()
    # img3 tree dropped -> empty, not written
    assert not (out_labels / "synthetic_dataset_img3.txt").exists()
    
    # needs_poly has img2
    assert needs_poly.exists()
    assert "synthetic_dataset_img2.jpg" in needs_poly.read_text()
    
    # empty_labels has img3 (tree dropped)
    assert empty_labels.exists()
    empty_content = empty_labels.read_text()
    assert "synthetic_dataset_img3.jpg" in empty_content
    assert stats["box_skipped"] == 1


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
        (src / "images" / f"img{i}.jpg").write_bytes(f"img_content_{i}".encode())
        with open(src / "labels" / f"img{i}.txt", "w") as f:
            f.write("0 0.5 0.5 0.1 0.1\n")
            
    out = tmp_path / "split"
    split_dataset(src, out, 0.7, 0.2, 0.1, 42)
    
    assert len(list((out / "images" / "train").glob("*.jpg"))) == 7
    assert len(list((out / "images" / "val").glob("*.jpg"))) == 2
    assert len(list((out / "images" / "test").glob("*.jpg"))) == 1


def test_split_dataset_group_regex_and_fixed_test(tmp_path):
    src = tmp_path / "src_grouped"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)
    
    # Create images grouped by site prefix with unique content
    for site in ["locA", "locB", "locC", "locD"]:
        for idx in range(3):
            fname = f"{site}_frame_{idx}.jpg"
            (src / "images" / fname).write_bytes(f"content_{fname}".encode())
            (src / "labels" / f"{site}_frame_{idx}.txt").write_text("0 0.1 0.1 0.2 0.2\n")
            
    # Fixed test directory
    fixed_test = tmp_path / "fixed_test"
    (fixed_test / "images").mkdir(parents=True)
    (fixed_test / "labels").mkdir(parents=True)
    (fixed_test / "images" / "benchmark_1.jpg").write_bytes(b"benchmark_content_1")
    (fixed_test / "labels" / "benchmark_1.txt").write_text("0 0.2 0.2 0.3 0.3\n")
    
    out = tmp_path / "split_out"
    split_dataset(
        src,
        out,
        train_ratio=0.7,
        val_ratio=0.3,
        test_ratio=0.0,
        seed=42,
        group_regex=r"^(loc[A-D])",
        fixed_test_dir=fixed_test,
    )
    
    # Check fixed test set was copied
    assert (out / "images" / "test" / "benchmark_1.jpg").exists()
    assert (out / "labels" / "test" / "benchmark_1.txt").exists()
    
    train_imgs = [p.name for p in (out / "images" / "train").glob("*.jpg")]
    val_imgs = [p.name for p in (out / "images" / "val").glob("*.jpg")]
    
    assert len(train_imgs) > 0
    assert len(val_imgs) > 0
    
    # Verify group regex kept all frames of a site in the same split
    train_sites = {re.match(r"^(loc[A-D])", f).group(1) for f in train_imgs}
    val_sites = {re.match(r"^(loc[A-D])", f).group(1) for f in val_imgs}
    fixed_sites = {"benchmark_1"}
    
    # No leakage between train and val
    assert len(train_sites & val_sites) == 0
    # No leakage with fixed test
    assert len(train_sites & fixed_sites) == 0
    assert len(val_sites & fixed_sites) == 0


def test_split_dataset_duplicate_content_raises(tmp_path):
    src = tmp_path / "src_dup"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)
    (src / "images" / "pool_img.jpg").write_bytes(b"identical_image_bytes_12345")
    (src / "labels" / "pool_img.txt").write_text("0 0.1 0.1 0.2 0.2\n")

    fixed_test = tmp_path / "fixed_dup"
    (fixed_test / "images").mkdir(parents=True)
    (fixed_test / "labels").mkdir(parents=True)
    (fixed_test / "images" / "fixed_img.jpg").write_bytes(b"identical_image_bytes_12345")
    (fixed_test / "labels" / "fixed_img.txt").write_text("0 0.1 0.1 0.2 0.2\n")

    out = tmp_path / "out_dup"
    with pytest.raises(ValueError, match="Duplicate image content detected"):
        split_dataset(src, out, 0.7, 0.3, 0.0, 42, fixed_test_dir=fixed_test)


def test_merge_datasets_partially_labeled_skipped(tmp_path, class_map):
    ds = tmp_path / "partially_labeled_ds"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    (ds / "data.yaml").write_text("names: ['Cable', 'power_pole']\n")

    (ds / "images" / "mixed.jpg").write_bytes(b"mixed_image_bytes")
    # mixed.txt has 1 polygon line (Cable) and 1 box line (power_pole)
    (ds / "labels" / "mixed.txt").write_text(
        "0 0.1 0.1 0.2 0.2 0.3 0.3\n"
        "1 0.5 0.5 0.2 0.2\n"
    )

    out_skip = tmp_path / "out_skip"
    out_skip.mkdir()
    needs_poly = out_skip / "needs_polygon.txt"
    stats_skip = process_dataset(ds, out_skip, class_map, needs_poly, box_policy="skip")

    # Image should NOT be written under box_policy="skip"
    assert not (out_skip / "images" / "partially_labeled_ds_mixed.jpg").exists()
    assert not (out_skip / "labels" / "partially_labeled_ds_mixed.txt").exists()
    assert stats_skip["partially_labeled_skipped"] == 1
    assert "partially_labeled_ds_mixed.jpg" in needs_poly.read_text()

    # Under box_policy="convert", image SHOULD be written
    out_convert = tmp_path / "out_convert"
    out_convert.mkdir()
    needs_poly_conv = out_convert / "needs_polygon.txt"
    stats_conv = process_dataset(ds, out_convert, class_map, needs_poly_conv, box_policy="convert")
    assert (out_convert / "images" / "partially_labeled_ds_mixed.jpg").exists()
    assert (out_convert / "labels" / "partially_labeled_ds_mixed.txt").exists()
    assert stats_conv["partially_labeled_skipped"] == 0


def test_merge_datasets_roboflow_layout_no_collision(tmp_path, class_map):
    ds = tmp_path / "rf_dataset"
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "data.yaml").write_text("names: ['Cable']\n")

    # Two splits: train and valid, sharing identical filenames
    (ds / "train" / "images").mkdir(parents=True)
    (ds / "train" / "labels").mkdir(parents=True)
    (ds / "train" / "images" / "frame_1.jpg").write_bytes(b"train_frame_1")
    (ds / "train" / "labels" / "frame_1.txt").write_text("0 0.1 0.1 0.2 0.2 0.3 0.3\n")

    (ds / "valid" / "images").mkdir(parents=True)
    (ds / "valid" / "labels").mkdir(parents=True)
    (ds / "valid" / "images" / "frame_1.jpg").write_bytes(b"valid_frame_1")
    (ds / "valid" / "labels" / "frame_1.txt").write_text("0 0.2 0.2 0.4 0.4 0.5 0.5\n")

    out = tmp_path / "rf_merged"
    out.mkdir()
    needs_poly = out / "needs_polygon.txt"
    stats = process_dataset(ds, out, class_map, needs_poly)

    assert stats["images"] == 2
    # Verify prefix includes split name and prevents collision
    assert (out / "images" / "rf_dataset_train_frame_1.jpg").exists()
    assert (out / "images" / "rf_dataset_valid_frame_1.jpg").exists()
    assert (out / "labels" / "rf_dataset_train_frame_1.txt").exists()
    assert (out / "labels" / "rf_dataset_valid_frame_1.txt").exists()


def test_rmtree_guard_refuses_non_empty_dir(tmp_path):
    protected_dir = tmp_path / "user_dir"
    protected_dir.mkdir()
    (protected_dir / "user_file.txt").write_text("precious user data")

    # Refuses because directory is non-empty and missing .wireframe_merge
    with pytest.raises(RuntimeError, match="missing '.wireframe_merge'"):
        ensure_clean_output_dir(protected_dir)

    # When marked with .wireframe_merge, clean succeeds
    (protected_dir / ".wireframe_merge").touch()
    ensure_clean_output_dir(protected_dir)
    assert protected_dir.exists()
    assert not (protected_dir / "user_file.txt").exists()
    assert (protected_dir / ".wireframe_merge").exists()


def test_audit_agreement():
    # bbox format: xmin, ymin, xmax, ymax
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    iou = compute_mask_iou(box1, box2)
    assert iou == pytest.approx(25 / 175) # inter 25, union 100+100-25=175
    
    parts = ["0.0", "0.0", "10.0", "0.0", "10.0", "10.0", "0.0", "10.0"]
    bbox = extract_bbox_from_polygon(parts)
    assert bbox == [0.0, 0.0, 10.0, 10.0]


def test_merge_datasets_kept_counts_exclude_skipped_images(tmp_path, class_map):
    ds = tmp_path / "kept_count_ds"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    (ds / "data.yaml").write_text("names: ['Cable']\n")

    # Image 1: Pure polygon (Cable -> telecom_cable)
    (ds / "images" / "pure.jpg").write_bytes(b"pure_img_bytes")
    (ds / "labels" / "pure.txt").write_text("0 0.1 0.1 0.2 0.2 0.3 0.3\n")

    # Image 2: Mixed (1 polygon + 1 bbox), which should be skipped under box_policy='skip'
    (ds / "images" / "mixed.jpg").write_bytes(b"mixed_img_bytes")
    (ds / "labels" / "mixed.txt").write_text("0 0.1 0.1 0.2 0.2 0.3 0.3\n0 0.5 0.5 0.2 0.2\n")

    out = tmp_path / "out_kept"
    out.mkdir()
    needs_poly = out / "needs_polygon.txt"
    stats = process_dataset(ds, out, class_map, needs_poly, box_policy="skip")

    # Only 1 image actually written; kept_per_class must NOT include lines from the skipped mixed.jpg
    assert stats["written_images"] == 1
    assert stats["written_lines_total"] == 1
    assert stats["kept_per_class"]["telecom_cable"] == 1
    assert stats["partially_labeled_skipped"] == 1


def test_merge_datasets_unmapped_policy(tmp_path, class_map):
    ds = tmp_path / "unmapped_policy_ds"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    (ds / "data.yaml").write_text("names: ['Cable', 'unmapped_obj']\n")

    # Image has 1 mapped polygon line and 1 unmapped line
    (ds / "images" / "sample.jpg").write_bytes(b"sample_bytes")
    (ds / "labels" / "sample.txt").write_text("0 0.1 0.1 0.2 0.2 0.3 0.3\n1 0.4 0.4 0.5 0.5 0.6 0.6\n")

    # Default: allow_unmapped=False -> image skipped
    out_default = tmp_path / "out_unmapped_default"
    out_default.mkdir()
    needs_poly_a = out_default / "needs_polygon.txt"
    stats_default = process_dataset(ds, out_default, class_map, needs_poly_a, allow_unmapped=False)

    assert stats_default["skipped_unmapped"] == 1
    assert stats_default["written_images"] == 0
    assert not (out_default / "images" / "unmapped_policy_ds_sample.jpg").exists()
    unmapped_file = out_default / "unmapped_images.txt"
    assert unmapped_file.exists()
    assert "unmapped_policy_ds_sample.jpg: unmapped_obj" in unmapped_file.read_text()

    # With allow_unmapped=True -> image written with unmapped line dropped
    out_allowed = tmp_path / "out_unmapped_allowed"
    out_allowed.mkdir()
    needs_poly_b = out_allowed / "needs_polygon.txt"
    stats_allowed = process_dataset(ds, out_allowed, class_map, needs_poly_b, allow_unmapped=True)

    assert stats_allowed["skipped_unmapped"] == 0
    assert stats_allowed["written_images"] == 1
    assert stats_allowed["kept_per_class"]["telecom_cable"] == 1
    assert (out_allowed / "images" / "unmapped_policy_ds_sample.jpg").exists()
    lbl_content = (out_allowed / "labels" / "unmapped_policy_ds_sample.txt").read_text()
    assert len(lbl_content.strip().splitlines()) == 1


def test_split_dataset_near_dup_detection(tmp_path):
    import cv2
    from backend.ml.split_dataset import compute_file_hash

    # Generate image with texture/pattern
    rng = np.random.RandomState(999)
    img_data = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    src = tmp_path / "src_neardup"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)
    png_path = src / "images" / "pattern.png"
    cv2.imwrite(str(png_path), img_data)
    (src / "labels" / "pattern.txt").write_text("0 0.1 0.1 0.2 0.2\n")

    fixed = tmp_path / "fixed_neardup"
    (fixed / "images").mkdir(parents=True)
    (fixed / "labels").mkdir(parents=True)
    jpg_path = fixed / "images" / "pattern.jpg"
    cv2.imwrite(str(jpg_path), img_data, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    (fixed / "labels" / "pattern.txt").write_text("0 0.1 0.1 0.2 0.2\n")

    # Due to JPEG compression, exact SHA-256 differs
    assert compute_file_hash(png_path) != compute_file_hash(jpg_path)

    # Without near-dup threshold, split succeeds
    out_no_thresh = tmp_path / "out_no_thresh"
    split_dataset(src, out_no_thresh, 0.7, 0.3, 0.0, 42, fixed_test_dir=fixed, near_dup_threshold=None)

    # With near-dup threshold, raises ValueError
    out_thresh = tmp_path / "out_thresh"
    with pytest.raises(ValueError, match="Near-duplicate image content detected"):
        split_dataset(src, out_thresh, 0.7, 0.3, 0.0, 42, fixed_test_dir=fixed, near_dup_threshold=5)


def test_split_dataset_group_regex_dynamic_test_ratio(tmp_path):
    src = tmp_path / "src_dyn"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)

    for g in ["grpA", "grpB", "grpC", "grpD", "grpE", "grpF"]:
        for i in range(2):
            fname = f"{g}_frame_{i}.jpg"
            (src / "images" / fname).write_bytes(f"unique_content_{fname}".encode())
            (src / "labels" / f"{g}_frame_{i}.txt").write_text("0 0.1 0.1 0.2 0.2\n")

    out = tmp_path / "out_dyn"
    split_dataset(
        src,
        out,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        group_regex=r"^(grp[A-F])",
    )

    train_imgs = list((out / "images" / "train").glob("*.jpg"))
    val_imgs = list((out / "images" / "val").glob("*.jpg"))
    test_imgs = list((out / "images" / "test").glob("*.jpg"))

    # Assert non-empty train, val, and test splits
    assert len(train_imgs) > 0
    assert len(val_imgs) > 0
    assert len(test_imgs) > 0

    train_groups = {re.match(r"^(grp[A-F])", f.name).group(1) for f in train_imgs}
    val_groups = {re.match(r"^(grp[A-F])", f.name).group(1) for f in val_imgs}
    test_groups = {re.match(r"^(grp[A-F])", f.name).group(1) for f in test_imgs}

    # Assert no group overlap across splits
    assert len(train_groups & val_groups) == 0
    assert len(train_groups & test_groups) == 0
    assert len(val_groups & test_groups) == 0


def test_merge_datasets_normalized_canonical(tmp_path):
    from backend.ml.merge_datasets import load_class_map, process_dataset
    
    # Empty custom map; canonical classes should map as identities automatically
    empty_map_file = tmp_path / "empty_map.yaml"
    empty_map_file.write_text("")
    class_map = load_class_map(empty_map_file)
    
    ds = tmp_path / "norm_ds"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    # Source dataset with non-canonical case and formatting
    (ds / "data.yaml").write_text("names: ['Pole', 'Power Line']\n")
    (ds / "images" / "test.jpg").write_bytes(b"test_img")
    (ds / "labels" / "test.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"
        "1 0.3 0.3 0.4 0.3 0.4 0.4 0.3 0.4\n"
    )

    out = tmp_path / "norm_out"
    out.mkdir()
    stats = process_dataset(
        ds,
        out,
        class_map,
        needs_polygon_file=out / "needs_poly.txt",
    )

    assert stats["written_images"] == 1
    assert stats["unmapped_lines"] == 0
    assert stats["kept_per_class"]["pole"] == 1
    assert stats["kept_per_class"]["power_line"] == 1


def test_merge_datasets_dry_run(tmp_path, caplog, monkeypatch):
    import sys
    from backend.ml.merge_datasets import main

    ds = tmp_path / "dry_run_ds"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    (ds / "data.yaml").write_text("names: ['Pole', 'UnknownClass', 'tree']\n")
    (ds / "images" / "sample.jpg").write_bytes(b"sample")
    (ds / "labels" / "sample.txt").write_text(
        "0 0.1 0.1 0.2 0.2\n"
        "1 0.3 0.3 0.4 0.4\n"
        "2 0.5 0.5 0.6 0.6\n"
    )

    out = tmp_path / "must_not_exist"
    test_args = [
        "merge_datasets.py",
        "--src", str(ds),
        "--dry-run"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with caplog.at_level("INFO"):
        main()

    # Assert output directory was never created
    assert not out.exists()
    # Check stdout/logs for class status
    log_text = caplog.text
    assert "Class 'Pole': mapped -> pole" in log_text
    assert "Class 'UnknownClass': unmapped" in log_text
    assert "Class 'tree': DROP" in log_text






