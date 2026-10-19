"""Tests for Audit Batch B fixes (#4, #1, #5, #11).

- #4: MPC ranking excludes public tips (is_public = 1).
- #1: Trend query includes boundary day in SQLite datetime format.
- #5: Batch GPS validation mirrors single upload (only lat -> 422, lat=999 -> 422, valid -> 200).
- #11: Purge mock files deletes mock files resolving in uploads/outputs and preserves paths outside.
"""

import uuid
from pathlib import Path
import cv2
import pytest


def test_mpc_ranking_excludes_public_tips(client_admin, _init_settings):
    conn = _init_settings
    cur = conn.execute(
        "INSERT INTO sites (name, barangay, street, barangay_norm, street_norm) "
        "VALUES ('Test Utility', 'Brgy San Jose', 'Rizal St', 'san jose', 'rizal st')"
    )
    site_id = cur.lastrowid

    # Verified inspector record: is_public = 0, risk = 85.0
    conn.execute(
        "INSERT INTO analyses (filename, site_id, status, is_public, risk_score, is_mock) "
        "VALUES ('insp.jpg', ?, 'complete', 0, 85.0, 0)",
        (site_id,)
    )

    # Public submission (unverified tip): is_public = 1, risk = 15.0
    conn.execute(
        "INSERT INTO analyses (filename, site_id, status, is_public, risk_score, is_mock) "
        "VALUES ('tip.jpg', ?, 'complete', 1, 15.0, 0)",
        (site_id,)
    )
    conn.commit()

    res = client_admin.get("/api/mpc/ranking?include_mock=false")
    assert res.status_code == 200
    rankings = res.json()["rankings"]

    assert len(rankings) == 1
    # Only verified record is counted: count == 1 and avg_risk == 85.0
    assert rankings[0]["analysis_count"] == 1
    assert rankings[0]["avg_risk"] == 85.0


def test_stats_trend_includes_boundary_day(client_admin, _init_settings):
    conn = _init_settings

    # Analysis created on the boundary day (now - 7 days + 1 hour)
    conn.execute(
        "INSERT INTO analyses (filename, status, is_public, risk_score, is_mock, created_at) "
        "VALUES ('boundary.jpg', 'complete', 0, 42.0, 0, datetime('now', '-7 days', '+1 hour'))"
    )
    # Another analysis created outside the window (now - 8 days)
    conn.execute(
        "INSERT INTO analyses (filename, status, is_public, risk_score, is_mock, created_at) "
        "VALUES ('old.jpg', 'complete', 0, 50.0, 0, datetime('now', '-8 days'))"
    )
    conn.commit()

    res = client_admin.get("/api/stats/dashboard?days=7&include_mock=false")
    assert res.status_code == 200
    trend = res.json()["trend"]

    # Trend must include the record created within the 7-day window
    total_in_trend = sum(t["count"] for t in trend)
    assert total_in_trend == 1
    assert trend[0]["avg_risk"] == 42.0


def test_batch_gps_validation(client_admin, clean_image):
    image_bytes = cv2.imencode('.jpg', clean_image)[1].tobytes()

    # 1. Only lat provided (missing lng) -> 422
    data_only_lat = {
        "company": "Power Corp",
        "barangay": "San Jose",
        "street": "Mabini St",
        "gps_lat": "14.5995",
        "gps_lng": ""
    }
    files = [("files", ("image1.jpg", image_bytes, "image/jpeg"))]
    res_only_lat = client_admin.post("/api/batches", data=data_only_lat, files=files)
    assert res_only_lat.status_code == 422
    assert "both" in res_only_lat.text.lower()

    # 2. lat = 999 (out of world bounds) -> 422 Invalid latitude (batch vs single parity)
    data_invalid_lat = {
        "company": "Power Corp",
        "barangay": "San Jose",
        "street": "Mabini St",
        "gps_lat": "999",
        "gps_lng": "120.9842"
    }
    res_batch_invalid = client_admin.post("/api/batches", data=data_invalid_lat, files=files)
    assert res_batch_invalid.status_code == 422
    assert "latitude" in res_batch_invalid.text.lower()

    single_file = [("file", ("image1.jpg", image_bytes, "image/jpeg"))]
    res_single_invalid = client_admin.post("/api/analysis/upload", data=data_invalid_lat, files=single_file)
    assert res_single_invalid.status_code == 422
    assert res_single_invalid.json()["detail"] == res_batch_invalid.json()["detail"]

    # 3. Valid coordinates inside PH -> 200 without warning
    data_valid = {
        "company": "Power Corp",
        "barangay": "San Jose",
        "street": "Mabini St",
        "gps_lat": "14.5995",
        "gps_lng": "120.9842"
    }
    res_valid = client_admin.post("/api/batches", data=data_valid, files=files)
    assert res_valid.status_code == 200
    body_valid = res_valid.json()
    assert "batch_id" in body_valid
    assert "warning" not in body_valid

    # 4. Valid coordinates outside PH -> 200 with warning
    data_outside_ph = {
        "company": "Power Corp",
        "barangay": "San Jose",
        "street": "Mabini St",
        "gps_lat": "35.6762",
        "gps_lng": "139.6503"
    }
    res_outside = client_admin.post("/api/batches", data=data_outside_ph, files=files)
    assert res_outside.status_code == 200
    body_outside = res_outside.json()
    assert "batch_id" in body_outside
    assert "warning" in body_outside
    assert "philippines" in body_outside["warning"].lower()


def test_purge_mock_data_files(client_admin, _init_settings, tmp_path):
    conn = _init_settings

    uploads_dir = Path("uploads")
    outputs_dir = Path("outputs")
    uploads_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    test_uid = uuid.uuid4().hex[:8]
    up_path = uploads_dir / f"purge_test_{test_uid}.jpg"
    ann_path = outputs_dir / f"purge_test_ann_{test_uid}.png"
    spec_path = outputs_dir / f"purge_test_spec_{test_uid}.png"
    outside_path = tmp_path / f"outside_sensitive_{test_uid}.txt"

    up_path.write_bytes(b"mock original image data")
    ann_path.write_bytes(b"mock annotated image data")
    spec_path.write_bytes(b"mock spectrum data")
    outside_path.write_bytes(b"protected system file outside uploads/outputs")

    # Insert mock records
    conn.execute(
        "INSERT INTO analyses (filename, original_path, annotated_path, spectrum_path, is_mock, status) "
        "VALUES (?, ?, ?, ?, 1, 'complete')",
        (up_path.name, str(up_path), str(ann_path), str(spec_path))
    )
    conn.execute(
        "INSERT INTO analyses (filename, original_path, is_mock, status) "
        "VALUES ('outside.jpg', ?, 1, 'complete')",
        (str(outside_path),)
    )
    conn.commit()

    # Ensure files exist before purge
    assert up_path.is_file()
    assert ann_path.is_file()
    assert spec_path.is_file()
    assert outside_path.is_file()

    res = client_admin.delete("/api/admin/purge_mock")
    assert res.status_code == 200
    assert res.json()["status"] == "success"

    # Mock files inside uploads/outputs must be deleted
    assert not up_path.exists(), "uploads mock file should be deleted"
    assert not ann_path.exists(), "outputs annotated mock file should be deleted"
    assert not spec_path.exists(), "outputs spectrum mock file should be deleted"

    # File outside uploads/outputs must NEVER be deleted
    assert outside_path.is_file(), "file outside uploads/outputs must NOT be deleted"

    # Mock DB records deleted
    remaining_mock = conn.execute("SELECT COUNT(*) as c FROM analyses WHERE is_mock = 1").fetchone()["c"]
    assert remaining_mock == 0
