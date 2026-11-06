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

    aliases = {"utility_pole": "pole", "power-line": "power_line"}
    assert resolve_class_name("power-line", aliases) == "power_line"
    assert resolve_class_name("utility_pole", aliases) == "pole"
    assert resolve_class_name("Power Line", aliases) == "power_line"

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
    assert all(w["class_name"] == "power_line" for w in wires)
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

    params = get_inference_params()
    assert params["conf"] == 0.25
    assert params["iou"] == 0.70
    assert params["device"] == "auto"

    monkeypatch.setenv("WIREFRAME_CONF", "0.45")
    monkeypatch.setenv("WIREFRAME_IOU", "0.65")
    monkeypatch.setenv("WIREFRAME_DEVICE", "cpu")

    params2 = get_inference_params()
    assert params2["conf"] == 0.45
    assert params2["iou"] == 0.65
    assert params2["device"] == "cpu"


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

    # Verify wires stored and mapped to 4 UI classes + 'other' (pole used for calibration, not stored as wire)
    wires = result_data["wires"]
    stored_classes = sorted(w["class_name"] for w in wires)
    assert stored_classes == ["dead_abandoned", "fiber_optic", "other", "power_line", "telecom_cable"]
    assert all(w["safety_level"] in ("Safe", "Caution", "Unsafe", "Critical") for w in wires)
    overall_safety = determine_overall_safety(
        [w["safety_level"] for w in wires],
        max_wire_band=result_data["max_wire_band"],
        analysis_band=result_data["crs_band"],
        clearance_status=result_data["ground_clearance_status"],
    )
    assert overall_safety in ("Safe", "Caution", "Unsafe", "Critical")

    # 3. Verify dashboard stats include all 4 canonical classes + 'other'
    stats_res = client.get("/api/stats/dashboard")
    assert stats_res.status_code == 200
    dist = stats_res.json()["distribution"]
    assert dist["power_line"] >= 1
    assert dist["telecom_cable"] >= 1
    assert dist["fiber_optic"] >= 1
    assert dist["dead_abandoned"] >= 1
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

