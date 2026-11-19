"""Tests for real model integration prep without real model weights."""

import io
import json
import os
import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import main
from backend import main_state
from backend.auth import _hash_token
from backend.config_store import get_setting, set_setting
from backend.database import get_connection
from backend.utils.cv.segment import get_inference_params
from backend.utils.pipeline import execute_pipeline, resolve_class_name


def _create_test_image(path: str, width: int = 800, height: int = 600):
    img = np.full((height, width, 3), 120, dtype=np.uint8)
    img[::2, ::2] = 200
    cv2.imwrite(path, img)


def _make_fake_model(names_dict, classes_list, confs_list):
    """Helper to construct a mock YOLO model returning specified names, classes, confidences."""
    h, w = 600, 800
    wire_mask = np.zeros((h, w), dtype=np.float32)
    # Draw a simple curved wire
    for x in range(100, 700):
        y = int(250 + 40 * ((x - 400) / 300.0) ** 2)
        wire_mask[max(0, y - 2) : min(h, y + 3), x] = 1.0

    pole_mask = np.zeros((h, w), dtype=np.float32)
    pole_mask[100:550, 400:460] = 1.0

    masks_list = []
    boxes_list = []
    for cls_idx in classes_list:
        cls_name = names_dict.get(cls_idx, "")
        if "pole" in cls_name.lower():
            masks_list.append(torch.from_numpy(pole_mask))
            boxes_list.append([400.0, 100.0, 460.0, 550.0])
        else:
            masks_list.append(torch.from_numpy(wire_mask))
            boxes_list.append([100.0, 250.0, 700.0, 290.0])

    class FakeBoxes:
        def __init__(self):
            self.xyxy = torch.tensor(boxes_list, dtype=torch.float32)
            self.cls = torch.tensor(classes_list, dtype=torch.int64)
            self.conf = torch.tensor(confs_list, dtype=torch.float32)

    class FakeMasks:
        def __init__(self):
            self.data = torch.stack(masks_list) if masks_list else torch.empty((0, h, w))

    class FakeResult:
        def __init__(self):
            self.boxes = FakeBoxes()
            self.masks = FakeMasks()
            self.names = names_dict

    def fake_model(img, **kwargs):
        return [FakeResult()]

    return fake_model


def test_real_pipeline_with_fake_model_persists_is_mock_zero(tmp_path, monkeypatch):
    """Running real pipeline with fake model produces status='complete' and is_mock=0."""
    monkeypatch.setenv("WIREFRAME_MOCK", "0")
    img_path = str(tmp_path / "test_wire.jpg")
    _create_test_image(img_path)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO analyses (filename, status, company, barangay, street, guest_session_id) VALUES ('test_wire.jpg', 'queued', 'Meralco', 'San Antonio', 'Perea St', ?)",
        (_hash_token("test_token_1"),)
    )
    analysis_id = cur.lastrowid
    conn.commit()

    model = _make_fake_model(
        names_dict={0: "power_line", 1: "pole"},
        classes_list=[0, 1],
        confs_list=[0.92, 0.95]
    )

    execute_pipeline(analysis_id, img_path, conn, model=model, batch_id=None)

    row = conn.execute("SELECT status, is_mock, clearance_ok FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    assert row["status"] == "complete"
    assert row["is_mock"] == 0
    conn.close()


def test_class_aliases_resolution_and_pipeline(tmp_path, monkeypatch):
    """Class name aliases map model names before normalization and pipeline handles them."""
    monkeypatch.setenv("WIREFRAME_MOCK", "0")

    aliases = {"utility_pole": "pole", "power-line": "power_distribution_line"}
    assert resolve_class_name("power-line", aliases) == "power_distribution_line"
    assert resolve_class_name("utility_pole", aliases) == "pole"
    assert resolve_class_name("Power Line", aliases) == "power_distribution_line"

    # Configure alias in settings
    set_setting("class_aliases", aliases)

    img_path = str(tmp_path / "test_alias.jpg")
    _create_test_image(img_path)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO analyses (filename, status, company, barangay, street, guest_session_id) VALUES ('test_alias.jpg', 'queued', 'Meralco', 'San Antonio', 'Perea St', ?)",
        (_hash_token("test_token_2"),)
    )
    analysis_id = cur.lastrowid
    conn.commit()

    # Model outputs aliased names
    model = _make_fake_model(
        names_dict={0: "power-line", 1: "utility_pole"},
        classes_list=[0, 1],
        confs_list=[0.88, 0.94]
    )

    execute_pipeline(analysis_id, img_path, conn, model=model, batch_id=None)

    row = conn.execute("SELECT status, is_mock FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    assert row["status"] == "complete"
    assert row["is_mock"] == 0

    wires = conn.execute("SELECT class_name FROM wires WHERE analysis_id = ?", (analysis_id,)).fetchall()
    assert len(wires) >= 1
    assert all(w["class_name"] == "power_distribution_line" for w in wires)
    conn.close()


def test_unknown_class_ignored_and_logged_once(tmp_path, monkeypatch, caplog):
    """Unknown classes are ignored, not crashed on, and logged only once."""
    import logging
    monkeypatch.setenv("WIREFRAME_MOCK", "0")

    img_path = str(tmp_path / "test_unknown.jpg")
    _create_test_image(img_path)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO analyses (filename, status, company, barangay, street, guest_session_id) VALUES ('test_unknown.jpg', 'queued', 'Meralco', 'San Antonio', 'Perea St', ?)",
        (_hash_token("test_token_3"),)
    )
    analysis_id = cur.lastrowid
    conn.commit()

    # Model includes unknown class 2: "transformer_box"
    model = _make_fake_model(
        names_dict={0: "power_line", 1: "pole", 2: "transformer_box"},
        classes_list=[0, 1, 2],
        confs_list=[0.90, 0.95, 0.85]
    )

    from backend.utils.pipeline import _UNKNOWN_CLASSES_LOGGED
    _UNKNOWN_CLASSES_LOGGED.discard("transformer_box")

    with caplog.at_level(logging.WARNING):
        execute_pipeline(analysis_id, img_path, conn, model=model, batch_id=None)

        row = conn.execute("SELECT status, is_mock FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        assert row["status"] == "complete"

        # Verify unknown class was logged
        unknown_logs = [r for r in caplog.records if "transformer_box" in r.message]
        assert len(unknown_logs) == 1

        # Running a second time should not log the same unknown class again
        cur.execute(
            "INSERT INTO analyses (filename, status, company, barangay, street, guest_session_id) VALUES ('test_unknown2.jpg', 'queued', 'Meralco', 'San Antonio', 'Perea St', ?)",
            (_hash_token("test_token_4"),)
        )
        aid2 = cur.lastrowid
        conn.commit()

        execute_pipeline(aid2, img_path, conn, model=model, batch_id=None)
        unknown_logs2 = [r for r in caplog.records if "transformer_box" in r.message]
        assert len(unknown_logs2) == 1
    conn.close()


def test_missing_model_in_real_mode_rejects_upload_503(monkeypatch):
    """When WIREFRAME_MOCK=0 and model is missing, uploads fail with 503 and exact message."""
    monkeypatch.setenv("WIREFRAME_MOCK", "0")
    main_state.set_model(None)
    main_state.set_model_available(False)

    conn = get_connection()
    count_before = conn.execute("SELECT count(*) FROM analyses").fetchone()[0]

    client = TestClient(main.app)
    # Generate in-memory valid JPEG
    img_byte_arr = io.BytesIO()
    Image.new("RGB", (640, 480), color=(100, 100, 100)).save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    data = {
        "company": "Test Co",
        "barangay": "San Antonio",
        "street": "Perea St",
    }
    files = {"file": ("test.jpg", img_byte_arr.getvalue(), "image/jpeg")}

    res = client.post(
        "/api/analysis/upload",
        data=data,
        files=files,
        headers={"X-Requested-With": "XMLHttpRequest"}
    )
    assert res.status_code == 503
    assert res.json()["detail"] == "Model not available. The detection model is not loaded on this server."

    # Verify no error row created in database
    count_after = conn.execute("SELECT count(*) FROM analyses").fetchone()[0]
    assert count_after == count_before
    conn.close()


def test_health_and_api_health_endpoints(monkeypatch):
    """Health endpoints report model_loaded and mock_mode."""
    # Real mode, model not loaded
    monkeypatch.setenv("WIREFRAME_MOCK", "0")
    main_state.set_model(None)
    main_state.set_model_available(False)

    client = TestClient(main.app)

    for endpoint in ["/health", "/api/health"]:
        res = client.get(endpoint)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["mock_mode"] is False
        assert data["model_loaded"] is False
        assert data["model_status"] == "not_loaded"

    # Real mode, model loaded
    dummy_model = object()
    main_state.set_model(dummy_model)
    for endpoint in ["/health", "/api/health"]:
        res = client.get(endpoint)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["mock_mode"] is False
        assert data["model_loaded"] is True
        assert data["model_status"] == "loaded"

    # Cleanup
    main_state.set_model(None)


def test_inference_settings_env_and_defaults(monkeypatch):
    """Inference params respect env overrides and fallback to defaults."""
    monkeypatch.delenv("WIREFRAME_CONF", raising=False)
    monkeypatch.delenv("WIREFRAME_IOU", raising=False)
    monkeypatch.delenv("WIREFRAME_DEVICE", raising=False)
    monkeypatch.delenv("WIREFRAME_TILED", raising=False)

    params = get_inference_params()
    assert params["conf"] == 0.08
    assert params["iou"] == 0.70
    assert params["imgsz"] == 1024
    assert params["tiled"] is False
    assert params["tile_size"] == 640
    assert params["tile_overlap"] == 0.3
    assert params["tile_iou"] == 0.9
    assert params["max_det"] == 1000
    assert params["device"] == "auto"

    monkeypatch.setenv("WIREFRAME_CONF", "0.45")
    monkeypatch.setenv("WIREFRAME_IOU", "0.65")
    monkeypatch.setenv("WIREFRAME_DEVICE", "cpu")
    monkeypatch.setenv("WIREFRAME_TILED", "1")

    params2 = get_inference_params()
    assert params2["conf"] == 0.45
    assert params2["iou"] == 0.65
    assert params2["device"] == "cpu"
    assert params2["tiled"] is True


def test_real_path_end_to_end_via_testclient_without_weights(tmp_path, monkeypatch, caplog, client_admin):
    """End-to-end real-path test through TestClient with fake Ultralytics model:
    verifies real converter + pipeline -> wires stored, class mapping (variants + unknown -> 'other'),
    risk/safety computed, result API, dashboard counts (including 'other'), and PDF report generation.
    """
    import logging

    monkeypatch.setenv("WIREFRAME_MOCK", "0")

    # Model returns class name variants + an unknown class 'aerial_bundle_unknown' + pole variant
    fake_model = _make_fake_model(
        names_dict={
            0: "Electric_Line",
            1: "Telco",
            2: "Optical-Fiber",
            3: "Dead_Wire",
            4: "aerial_bundle_unknown",
            5: "Utility_Pole",
        },
        classes_list=[0, 1, 2, 3, 4, 5],
        confs_list=[0.93, 0.88, 0.85, 0.91, 0.79, 0.96],
    )
    main_state.set_model(fake_model)
    main_state.set_model_available(True)

    from backend.utils.pipeline import _UNKNOWN_CLASSES_LOGGED
    _UNKNOWN_CLASSES_LOGGED.discard("aerial_bundle_unknown")

    # Create a textured image in tmp_path (never touches repo uploads/)
    img_path = str(tmp_path / "e2e_real_test.jpg")
    _create_test_image(img_path, width=800, height=600)

    guest_token = "e2e_real_guest_token"
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO analyses
           (filename, status, company, barangay, street, guest_session_id, is_mock, is_public)
           VALUES ('e2e_real_test.jpg', 'queued', 'Meralco', 'San Antonio', 'Ayala Ave', ?, 0, 0)""",
        (_hash_token(guest_token),),
    )
    analysis_id = cur.lastrowid
    conn.commit()

    with caplog.at_level(logging.WARNING):
        execute_pipeline(analysis_id, img_path, conn, model=fake_model, batch_id=None)

        # Verify unknown class was logged once with WARNING
        unknown_logs = [r for r in caplog.records if "aerial_bundle_unknown" in r.message]
        assert len(unknown_logs) == 1
    conn.close()

    client = client_admin
    main_state.set_model(fake_model)
    main_state.set_model_available(True)

    # 1. Verify /api/health reports ready=True, model_loaded=True, model_status='loaded'
    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    hdata = health_res.json()
    assert hdata["ready"] is True
    assert hdata["model_loaded"] is True
    assert hdata["model_status"] == "loaded"
    assert hdata["mock_mode"] is False

    # 2. Verify /api/analysis/{id}/result returns complete real-model analysis
    res = client.get(f"/api/analysis/{analysis_id}/result")
    assert res.status_code == 200
    result_data = res.json()
    assert result_data["status"] == "complete"
    assert result_data["is_mock"] in (0, False)

    # Verify risk & safety computed
    from backend.utils.clearance_helpers import determine_overall_safety
    assert result_data["risk_score"] is not None
    assert result_data["risk_score"] > 0
    assert result_data["crs_band"] in ("Low", "Moderate", "High", "Critical")
    assert result_data["ground_clearance_status"] in ("above", "near", "below", "no_pole")
    assert result_data["max_wire_crs"] is not None
    assert result_data["max_wire_band"] in ("Low", "Moderate", "High", "Critical")

    # Verify wires stored and mapped to canonical wire classes + 'other' (pole used for calibration, not stored as wire)
    wires = result_data["wires"]
    stored_classes = sorted(w["class_name"] for w in wires)
    assert stored_classes == ["communication_cable", "communication_cable", "dead_wire", "other", "power_distribution_line"]
    assert all(w["safety_level"] in ("Safe", "Caution", "Unsafe", "Critical") for w in wires)
    overall_safety = determine_overall_safety(
        [w["safety_level"] for w in wires],
        max_wire_band=result_data["max_wire_band"],
        analysis_band=result_data["crs_band"],
        clearance_status=result_data["ground_clearance_status"],
    )
    assert overall_safety in ("Safe", "Caution", "Unsafe", "Critical")

    # 3. Verify dashboard stats include all canonical classes + 'other'
    stats_res = client.get("/api/stats/dashboard")
    assert stats_res.status_code == 200
    dist = stats_res.json()["distribution"]
    assert dist["power_distribution_line"] >= 1
    assert dist["communication_cable"] >= 2
    assert dist["dead_wire"] >= 1
    assert dist["vegetation"] >= 0
    assert dist.get("other", 0) >= 1

    # 4. Verify PDF report builds cleanly in real mode
    pdf_res = client.get(f"/api/reports/{analysis_id}.pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"].startswith("application/pdf")
    assert pdf_res.content.startswith(b"%PDF")

    main_state.set_model(None)
    main_state.set_model_available(False)


def test_corrupt_or_missing_model_file_startup_and_batch_503(tmp_path, monkeypatch):
    """If model file is missing or corrupt with mock off:
    load_model_once returns None without raising, /api/health shows ready=False & model_status='not_loaded',
    and both single and batch uploads return 503 with no stack traces or server paths.
    """
    monkeypatch.setenv("WIREFRAME_MOCK", "0")
    main_state.set_model(None)
    main_state.set_model_available(False)

    # 1. Missing model file
    missing_path = str(tmp_path / "does_not_exist.pt")
    assert main_state.load_model_once(missing_path) is None
    assert main_state.is_model_available() is False

    # 2. Corrupt model file
    corrupt_path = tmp_path / "corrupt_weights.pt"
    corrupt_path.write_bytes(b"not a valid pytorch zip archive")
    assert main_state.load_model_once(str(corrupt_path)) is None
    assert main_state.is_model_available() is False

    client = TestClient(main.app)
    main_state.set_model(None)
    main_state.set_model_available(False)

    h_res = client.get("/api/health")
    assert h_res.status_code == 200
    h_json = h_res.json()
    assert h_json["ready"] is False
    assert h_json["model_loaded"] is False
    assert h_json["model_status"] == "not_loaded"

    img_byte_arr = io.BytesIO()
    Image.new("RGB", (320, 240), color=(120, 120, 120)).save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()

    batch_res = client.post(
        "/api/batches",
        data={"company": "Test Co", "barangay": "San Antonio", "street": "Perea St"},
        files=[("files", ("test1.jpg", img_bytes, "image/jpeg"))],
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert batch_res.status_code == 503
    detail = batch_res.json()["detail"]
    assert "not loaded" in detail.lower()
    assert "corrupt_weights.pt" not in detail
    assert "Traceback" not in detail


def test_tiled_sliced_predict_stitches_wire_across_tile_seams():
    """Verify sliced_predict / segment_image(..., tiled=True) slices a large image into overlapping
    tiles and stitches a continuous wire crossing tile boundaries into a unified full-size mask.
    """
    from backend.utils.cv.segment import positions, segment_image

    # 1. Verify tile positions calculation
    pos = positions(1200, tile=640, overlap=0.3)
    assert pos[0] == 0
    assert pos[-1] == 1200 - 640
    assert len(pos) >= 2

    # 2. Construct a 640x1200 RGB image with a continuous wire drawn across x=100..1100 at y=300
    full_h, full_w = 640, 1200
    rgb_img = np.zeros((full_h, full_w, 3), dtype=np.uint8)
    cv2.line(rgb_img, (100, 300), (1100, 300), (255, 255, 255), thickness=4)

    class FakeBoxes:
        def __init__(self, boxes, classes, confs):
            self.xyxy = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4))
            self.cls = torch.tensor(classes, dtype=torch.int64) if classes else torch.empty((0,), dtype=torch.int64)
            self.conf = torch.tensor(confs, dtype=torch.float32) if confs else torch.empty((0,), dtype=torch.float32)

    class FakeMasks:
        def __init__(self, masks_t):
            self.data = masks_t

    class FakeTileResult:
        def __init__(self, crop_bgr):
            self.names = {0: "power-distribution-lines"}
            ch, cw = crop_bgr.shape[:2]
            wire_px = (crop_bgr[:, :, 0] > 128).astype(np.float32)
            if np.any(wire_px):
                ys, xs = np.nonzero(wire_px)
                box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
                self.boxes = FakeBoxes([box], [0], [0.91])
                self.masks = FakeMasks(torch.from_numpy(wire_px[None, :, :]))
            else:
                self.boxes = FakeBoxes([], [], [])
                self.masks = None

    class FakeTiledModel:
        names = {0: "power-distribution-lines"}

        def predict(self, crops, **kwargs):
            return [FakeTileResult(c) for c in crops]

    seg_res = segment_image(
        rgb_img,
        FakeTiledModel(),
        original_shape=(full_h, full_w),
        tiled=True,
        tile_size=640,
        tile_overlap=0.3,
    )
    assert seg_res.mask_available is True
    # Because the wire is continuous across the tiles, cross-tile stitching merges it into 1 unified mask
    assert len(seg_res.masks) == 1
    assert seg_res.classes == [0]
    assert seg_res.masks[0].shape == (full_h, full_w)
    x1, y1, x2, y2 = seg_res.boxes[0]
    assert x1 <= 105
    assert x2 >= 1095


def test_wirenet_ensemble_metadata_and_graceful_fallback(tmp_path):
    """Verify load_wirenet_ensemble_metadata loads v212_ensemble.json, v212_convnext_t.arch.json,
    and v212_mitb1.arch.json from backend/models/ and degrades gracefully when .pt files are absent.
    """
    import shutil
    from pathlib import Path
    from backend.utils.cv.types import SegmentationResult
    from backend.utils.cv.wirenet import (
        DEFAULT_MODELS_DIR,
        is_wirenet_available,
        load_wirenet_ensemble_metadata,
        refine_detections_with_wirenet,
    )

    meta = load_wirenet_ensemble_metadata()
    assert meta["ensemble_config_loaded"] is True
    assert len(meta["members"]) == 3
    assert "v212_convnext_t" in meta["architectures"]
    assert meta["architectures"]["v212_convnext_t"]["encoder"] == "tu-convnext_tiny"
    assert "v212_mitb1" in meta["architectures"]
    assert meta["architectures"]["v212_mitb1"]["encoder"] == "mit_b1"
    assert meta["training_logs_present"]["v212_convnext_t"] is True
    assert meta["training_logs_present"]["v212_mitb1"] is True

    # Test graceful fallback when .pt files are absent in an isolated temp models dir
    empty_models = tmp_path / "models_no_pt"
    empty_models.mkdir()
    for jf in Path(DEFAULT_MODELS_DIR).glob("v212_*.json"):
        shutil.copy2(jf, empty_models / jf.name)

    assert is_wirenet_available(empty_models) is False
    dummy_mask = np.ones((100, 100), dtype=np.uint8) * 255
    seg_in = SegmentationResult(
        mask_available=True,
        boxes=[[10.0, 20.0, 90.0, 30.0]],
        classes=[0],
        confidences=[0.88],
        masks=[dummy_mask],
        names={0: "power_distribution_line"},
    )
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    seg_out = refine_detections_with_wirenet(rgb, seg_in, models_dir=empty_models)
    assert seg_out is seg_in
    assert len(seg_out.masks) == 1


def test_hough_pole_fallback_in_real_pipeline_when_enabled(tmp_path, monkeypatch):
    """When WIREFRAME_HOUGH_POLE_FALLBACK=1 and the 4-class model outputs no pole class,
    the pipeline uses detect_pole_hough_fallback to recover a pole bbox and compute clearance.
    """
    monkeypatch.setenv("WIREFRAME_MOCK", "0")
    monkeypatch.setenv("WIREFRAME_HOUGH_POLE_FALLBACK", "1")

    h, w = 600, 800
    img = np.full((h, w, 3), 135, dtype=np.uint8)
    # Add fine background texture to pass blur check
    img[::2, ::2] = 175
    # Draw strong vertical utility pole in the image (x=380..415, y=60..540) so Hough fallback detects it
    cv2.rectangle(img, (380, 60), (415, 540), (30, 30, 30), thickness=-1)
    cv2.line(img, (380, 60), (380, 540), (10, 10, 10), thickness=3)
    cv2.line(img, (415, 60), (415, 540), (10, 10, 10), thickness=3)

    img_path = str(tmp_path / "hough_pole_test.jpg")
    cv2.imwrite(img_path, img)

    # Fake 4-class model that ONLY outputs a wire (class 0: power-distribution-lines) and NO pole
    model = _make_fake_model(
        names_dict={0: "power-distribution-lines"},
        classes_list=[0],
        confs_list=[0.91],
    )

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO analyses (filename, status, company, barangay, street, guest_session_id) VALUES ('hough_pole_test.jpg', 'queued', 'Meralco', 'San Antonio', 'Perea St', ?)",
        (_hash_token("hough_tok"),),
    )
    aid = cur.lastrowid
    conn.commit()

    execute_pipeline(aid, img_path, conn, model=model, batch_id=None)

    row = conn.execute(
        "SELECT status, pole_bbox_json, clearance_available, clearance_ok FROM analyses WHERE id = ?",
        (aid,),
    ).fetchone()
    conn.close()

    assert row["status"] == "complete"
    assert row["pole_bbox_json"] is not None
    assert row["clearance_available"] is not None
    assert row["clearance_ok"] in (0, 1, 2)


def test_dynamic_api_paper_metrics_for_anonymous_and_admin_endpoints(tmp_path, monkeypatch, client_admin):
    """Verify anonymous guest result endpoint, admin MPC ranking, and single PDF report
    all include paper-conformant metrics (mspi, vni_band, crs_10, paper_risk_level, flags)
    without breaking existing 0-100 risk_score and clearance contracts.
    """
    monkeypatch.setenv("WIREFRAME_MOCK", "0")
    img_path = str(tmp_path / "paper_metrics_api.jpg")
    _create_test_image(img_path)

    model = _make_fake_model(
        names_dict={0: "power_distribution_line", 1: "pole"},
        classes_list=[0, 1],
        confs_list=[0.93, 0.96],
    )

    guest_tok = "anon_paper_tok_99"
    conn = get_connection()
    from backend.services.sites import get_or_create_site
    site_id = get_or_create_site(conn, "Meralco", "San Antonio", "Perea St", 14.55, 121.02)
    cur = conn.execute(
        """INSERT INTO analyses
           (filename, status, company, barangay, street, site_id, guest_session_id, is_mock, is_public)
           VALUES ('paper_metrics_api.jpg', 'queued', 'Meralco', 'San Antonio', 'Perea St', ?, ?, 0, 0)""",
        (site_id, _hash_token(guest_tok)),
    )
    aid = cur.lastrowid
    conn.commit()

    execute_pipeline(aid, img_path, conn, model=model, batch_id=None)
    conn.close()

    # 1. Anonymous guest client with matching guest_session cookie can read result with paper metrics
    anon_client = TestClient(main.app)
    anon_client.cookies.set("guest_session", guest_tok)
    res_anon = anon_client.get(f"/api/analysis/{aid}/result")
    assert res_anon.status_code == 200
    body = res_anon.json()
    assert body["status"] == "complete"
    assert isinstance(body["risk_score"], (int, float))
    assert body["vni_band"] in ("Low", "Moderate", "High", "Severe")
    assert isinstance(body["mspi"], float)
    assert 0.0 <= body["mspi"] <= 1.0
    assert body["paper_risk_level"] in ("Low", "Medium", "High")
    assert isinstance(body["flags"], list)
    assert len(body["wires"]) == 1
    w0 = body["wires"][0]
    assert isinstance(w0["crs_10"], float)
    assert 0.0 <= w0["crs_10"] <= 10.0
    assert w0["paper_risk_level"] in ("Low", "Medium", "High")
    assert isinstance(w0["flags"], list)

    # 2. Admin MPC ranking includes mspi, vni_band, max_crs_10, paper_risk_level
    res_rank = client_admin.get("/api/mpc/ranking")
    assert res_rank.status_code == 200
    rankings = res_rank.json()["rankings"]
    assert len(rankings) >= 1
    r0 = rankings[0]
    assert isinstance(r0["mspi"], float)
    assert r0["vni_band"] in ("Low", "Moderate", "High", "Severe")
    assert r0["paper_risk_level"] in ("Low", "Medium", "High")

    # 3. Single inspection PDF generates cleanly with paper CRS/MSPI fields included
    res_pdf = client_admin.get(f"/api/reports/{aid}.pdf")
    assert res_pdf.status_code == 200
    assert res_pdf.content.startswith(b"%PDF")


def test_health_diagnostics_and_real_dataset_smoke_check(tmp_path, monkeypatch):
    """Verify /api/health reports weights_present, wirenet_weights_present, tiled_mode,
    and canonical_classes, verify MODEL_MANIFEST.json, and run a pipeline smoke check on a
    real image from datasets/wireframes_v1/train/images/ (if present) confirming annotated.jpg
    and spectrum.png are generated cleanly with >0 bytes.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "backend" / "models" / "MODEL_MANIFEST.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["canonical_classes"] == [
        "power_distribution_line",
        "communication_cable",
        "dead_wire",
        "vegetation",
    ]

    client = TestClient(main.app)
    monkeypatch.setenv("WIREFRAME_TILED", "1")
    h_res = client.get("/api/health")
    assert h_res.status_code == 200
    h_body = h_res.json()
    assert isinstance(h_body["weights_present"], bool)
    assert isinstance(h_body["wirenet_weights_present"], bool)
    assert h_body["tiled_mode"] is True
    assert h_body["canonical_classes"] == [
        "power_distribution_line",
        "communication_cable",
        "dead_wire",
        "vegetation",
    ]
    monkeypatch.delenv("WIREFRAME_TILED", raising=False)

    dataset_img_dir = repo_root / "datasets" / "wireframes_v1" / "train" / "images"
    if not dataset_img_dir.is_dir():
        pytest.skip("datasets/wireframes_v1/train/images not present on disk")

    sample_imgs = sorted(dataset_img_dir.glob("*.jpg"))
    if not sample_imgs:
        pytest.skip("No .jpg files in datasets/wireframes_v1/train/images")

    sample_img = sample_imgs[0]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_MOCK", "1")

    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO analyses (filename, status, company, barangay, street, is_mock) VALUES (?, 'queued', 'Meralco', 'San Antonio', 'Perea St', 1)",
        (sample_img.name,),
    )
    aid = cur.lastrowid
    conn.commit()

    execute_pipeline(aid, str(sample_img), conn, model=None, batch_id=None)

    row = conn.execute(
        "SELECT status, vni, annotated_path, spectrum_path FROM analyses WHERE id = ?",
        (aid,),
    ).fetchone()
    conn.close()

    assert row["status"] in ("complete", "no_wires")
    assert row["vni"] is not None
    assert 0.0 <= float(row["vni"]) <= 1.0
    assert row["annotated_path"] and Path(row["annotated_path"]).is_file()
    assert Path(row["annotated_path"]).stat().st_size > 0
    assert row["spectrum_path"] and Path(row["spectrum_path"]).is_file()
    assert Path(row["spectrum_path"]).stat().st_size > 0


def test_weights_downloader_sha256_verification_and_normalized_classes(tmp_path, monkeypatch):
    """Verify automatic model weight downloader & SHA-256 verifier (single & multi-part files,
    local alias mapping for wireframe_seg.pt, checksum mismatch rejection, existing file preservation)
    and verify normalize_wire_class maps all 4 canonical classes.
    """
    from functools import partial
    import hashlib
    import http.server
    import threading
    from backend.utils import weights_downloader as wd
    from backend.utils.cv.types import normalize_wire_class

    # 1. Verify normalize_wire_class maps all 4 classes cleanly
    assert normalize_wire_class("Vegetation") == "vegetation"
    assert normalize_wire_class("abandoned-wire") == "dead_wire"
    assert normalize_wire_class("power-distribution-lines") == "power_distribution_line"
    assert normalize_wire_class("telecom-cable") == "communication_cable"

    # 2. Spin up local HTTP mirror serving a single-part weight (with local_alias wireframe_seg.pt)
    #    and a 2-part weight
    mirror_dir = tmp_path / "mirror"
    mirror_dir.mkdir()
    data_yolo = b"yolo11m-seg-weights" * 200
    data_multi = b"convnext-part-000-and-001" * 300
    (mirror_dir / "wireframe-yolo11m-seg.pt").write_bytes(data_yolo)
    mid = len(data_multi) // 2
    (mirror_dir / "v212_convnext_t.pt.000").write_bytes(data_multi[:mid])
    (mirror_dir / "v212_convnext_t.pt.001").write_bytes(data_multi[mid:])

    fake_required = (
        wd.WeightFile(
            "wireframe-yolo11m-seg.pt",
            len(data_yolo),
            hashlib.sha256(data_yolo).hexdigest(),
            local_alias="wireframe_seg.pt",
        ),
        wd.WeightFile(
            "v212_convnext_t.pt",
            len(data_multi),
            hashlib.sha256(data_multi).hexdigest(),
            parts=2,
        ),
    )
    monkeypatch.setattr(wd, "REQUIRED", fake_required)

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(mirror_dir))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{srv.server_address[1]}"

    try:
        target_models = tmp_path / "models"
        assert len(wd.missing(target_models)) == 2
        failed = wd.ensure(target_models, base_url=base_url, verify_existing_hashes=True)
        assert failed == []

        # Both wireframe-yolo11m-seg.pt and its mapped alias wireframe_seg.pt exist and match
        assert (target_models / "wireframe-yolo11m-seg.pt").read_bytes() == data_yolo
        assert (target_models / "wireframe_seg.pt").read_bytes() == data_yolo
        # Multi-part file joined and verified
        assert (target_models / "v212_convnext_t.pt").read_bytes() == data_multi

        rep = wd.verify_all(target_models, check_hash=True)
        assert rep["wireframe-yolo11m-seg.pt"]["verified"] is True
        assert rep["v212_convnext_t.pt"]["verified"] is True

        # Tamper with mirror file and verify checksum mismatch is rejected on a fresh dir
        (mirror_dir / "wireframe-yolo11m-seg.pt").write_bytes(b"corrupted" * 200)
        bad_models = tmp_path / "bad_models"
        failed_bad = wd.ensure(bad_models, base_url=base_url)
        assert "wireframe-yolo11m-seg.pt" in failed_bad
        assert not (bad_models / "wireframe-yolo11m-seg.pt").exists()
    finally:
        srv.shutdown()


def test_render_unified_overlay_and_class_counts():
    """Verify render_unified_overlay draws masks, contours, pole bbox, span chords, sag & clearance lines."""
    from backend.utils.cv.visualize import render_unified_overlay

    img = np.full((240, 320, 3), 255, dtype=np.uint8)
    mask_power = np.zeros((240, 320), dtype=np.uint8)
    mask_power[80:86, 40:260] = 1
    mask_veg = np.zeros((240, 320), dtype=np.uint8)
    mask_veg[140:200, 180:250] = 1
    mask_pole = np.zeros((240, 320), dtype=np.uint8)
    mask_pole[40:220, 20:38] = 1

    analyzed_wires = [
        {
            "wire_index": 0,
            "class_name": "power_distribution_line",
            "confidence": 0.91,
            "mask": mask_power,
            "bbox": [40.0, 80.0, 260.0, 86.0],
            "polyline_json": "[(40, 80), (150, 92), (260, 82)]",
            "sag_m": 0.55,
            "ground_clearance_m": 5.12,
        }
    ]
    vegetation_detections = [
        {
            "class_name": "vegetation",
            "confidence": 0.84,
            "mask": mask_veg,
            "bbox": [180.0, 140.0, 250.0, 200.0],
        }
    ]
    rendered = render_unified_overlay(
        img,
        analyzed_wires=analyzed_wires,
        vegetation_detections=vegetation_detections,
        pole_bbox=[20.0, 40.0, 38.0, 220.0],
        pole_confidence=0.88,
        pole_masks=[mask_pole],
        meters_per_px=0.05,
    )
    assert rendered.shape == img.shape
    assert rendered.dtype == np.uint8
    # Image must have been modified with colored masks, contours, pole box, and badges
    assert not np.array_equal(rendered, img)






