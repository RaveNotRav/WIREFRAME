"""Tests for excluding invalid submissions (rejected, no_wires) from authority views."""

import json
import csv
import io
import pytest
from fastapi.testclient import TestClient
from backend.database import get_connection
from tests.conftest import _get_auth_client, _create_user
from main import app


def test_history_default_excludes_rejected_and_no_wires(_init_settings):
    conn = _init_settings
    client_admin = _get_auth_client(conn, "admin", "hist_admin")

    # Create analyses with various statuses
    statuses = ["complete", "error", "interrupted", "rejected", "no_wires", "queued", "processing"]
    for i, s in enumerate(statuses, start=100):
        conn.execute(
            """INSERT INTO analyses (id, filename, status, company, barangay, street, is_mock, is_public)
               VALUES (?, ?, ?, 'Company A', 'Brgy B', 'St C', 1, 0)""",
            (i, f"test_{s}.jpg", s)
        )
    conn.commit()

    res = client_admin.get("/api/analysis?is_mock=1")
    assert res.status_code == 200
    data = res.json()["data"]
    res_statuses = {item["status"] for item in data}

    # complete, error, interrupted, queued, processing must be present
    assert "complete" in res_statuses
    assert "error" in res_statuses
    assert "interrupted" in res_statuses
    assert "queued" in res_statuses
    assert "processing" in res_statuses

    # rejected and no_wires must be excluded by default
    assert "rejected" not in res_statuses
    assert "no_wires" not in res_statuses


def test_history_include_invalid_roles(_init_settings):
    conn = _init_settings
    # 1. Admin with include_invalid=true
    client_admin = _get_auth_client(conn, "admin", "inv_admin")
    conn.execute("INSERT INTO analyses (id, filename, status, is_mock, is_public) VALUES (201, 'a.jpg', 'complete', 1, 0)")
    conn.execute("INSERT INTO analyses (id, filename, status, is_mock, is_public) VALUES (202, 'b.jpg', 'rejected', 1, 0)")
    conn.execute("INSERT INTO analyses (id, filename, status, is_mock, is_public) VALUES (203, 'c.jpg', 'no_wires', 1, 0)")
    conn.commit()

    res = client_admin.get("/api/analysis?is_mock=1&include_invalid=true")
    assert res.status_code == 200
    statuses_admin = {item["status"] for item in res.json()["data"]}
    assert "complete" in statuses_admin
    assert "rejected" in statuses_admin
    assert "no_wires" in statuses_admin

    # 2. LGU Engineer with include_invalid=true
    client_lgu = _get_auth_client(conn, "lgu_engineer", "inv_lgu")
    res_lgu = client_lgu.get("/api/analysis?is_mock=1&include_invalid=true")
    assert res_lgu.status_code == 200
    statuses_lgu = {item["status"] for item in res_lgu.json()["data"]}
    assert "complete" in statuses_lgu
    assert "rejected" in statuses_lgu
    assert "no_wires" in statuses_lgu

    # 3. Field Inspector: created_by matching, include_invalid=true MUST BE IGNORED
    client_insp = _get_auth_client(conn, "field_inspector", "inv_insp")
    insp_row = conn.execute("SELECT id FROM users WHERE username = 'inv_insp'").fetchone()
    insp_id = insp_row["id"]

    conn.execute("INSERT INTO analyses (id, filename, status, is_mock, created_by, is_public) VALUES (204, 'd.jpg', 'complete', 1, ?, 0)", (insp_id,))
    conn.execute("INSERT INTO analyses (id, filename, status, is_mock, created_by, is_public) VALUES (205, 'e.jpg', 'rejected', 1, ?, 0)", (insp_id,))
    conn.execute("INSERT INTO analyses (id, filename, status, is_mock, created_by, is_public) VALUES (206, 'f.jpg', 'no_wires', 1, ?, 0)", (insp_id,))
    conn.commit()

    res_insp = client_insp.get("/api/analysis?is_mock=1&include_invalid=true")
    assert res_insp.status_code == 200
    statuses_insp = {item["status"] for item in res_insp.json()["data"]}
    assert "complete" in statuses_insp
    assert "rejected" not in statuses_insp
    assert "no_wires" not in statuses_insp


def test_stats_and_ranking_exclude_rejected_and_no_wires(_init_settings):
    conn = _init_settings
    client_admin = _get_auth_client(conn, "admin", "stats_admin")

    cur = conn.execute("INSERT INTO sites (name, barangay, street) VALUES ('Site X', 'Brgy X', 'St X')")
    site_id = cur.lastrowid

    # 1 complete with score 80
    conn.execute("INSERT INTO analyses (id, filename, status, site_id, risk_score, is_mock) VALUES (301, 'c.jpg', 'complete', ?, 80.0, 1)", (site_id,))
    # 1 rejected with score 90 (must NOT be counted)
    conn.execute("INSERT INTO analyses (id, filename, status, site_id, risk_score, is_mock) VALUES (302, 'r.jpg', 'rejected', ?, 90.0, 1)", (site_id,))
    # 1 no_wires with score 20 (must NOT be counted)
    conn.execute("INSERT INTO analyses (id, filename, status, site_id, risk_score, is_mock) VALUES (303, 'n.jpg', 'no_wires', ?, 20.0, 1)", (site_id,))
    # Add a wire for 301
    conn.execute("INSERT INTO wires (analysis_id, class_name, safety_level) VALUES (301, 'power_line', 'Critical')")
    # Add wires for rejected/no_wires that should never show in distribution
    conn.execute("INSERT INTO wires (analysis_id, class_name, safety_level) VALUES (302, 'telecom_cable', 'Critical')")
    conn.commit()

    # Stats dashboard
    stats_res = client_admin.get("/api/stats/dashboard?include_mock=true")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert stats["kpis"]["total_analyses"] == 1
    assert stats["kpis"]["average_risk_score"] == 80.0
    assert stats["distribution"]["power_line"] == 1
    assert stats["distribution"]["telecom_cable"] == 0

    # Ranking endpoint
    ranking_res = client_admin.get("/api/mpc/ranking?include_mock=true")
    assert ranking_res.status_code == 200
    rankings = ranking_res.json()["rankings"]
    assert len(rankings) == 1
    assert rankings[0]["analysis_count"] == 1
    assert rankings[0]["avg_risk"] == 80.0


def test_exports_skip_rejected_and_no_wires_by_default(_init_settings):
    conn = _init_settings
    client_admin = _get_auth_client(conn, "admin", "exp_admin")

    conn.execute("INSERT INTO analyses (id, filename, status, is_mock, gps_lat, gps_lng) VALUES (401, 'comp.jpg', 'complete', 1, 14.5, 121.0)")
    conn.execute("INSERT INTO analyses (id, filename, status, is_mock, gps_lat, gps_lng) VALUES (402, 'rej.jpg', 'rejected', 1, 14.5, 121.0)")
    conn.execute("INSERT INTO analyses (id, filename, status, is_mock, gps_lat, gps_lng) VALUES (403, 'nowires.jpg', 'no_wires', 1, 14.5, 121.0)")
    conn.commit()

    aids = [401, 402, 403]

    # --- CSV Export ---
    # Default: skips rejected and no_wires
    res_csv = client_admin.post("/api/reports/csv", json={"analysis_ids": aids, "include_mock": True})
    assert res_csv.status_code == 200
    assert res_csv.headers.get("X-Skipped-Invalid") == "2"
    reader = csv.DictReader(io.StringIO(res_csv.text))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["filename"] == "comp.jpg"

    # include_invalid=true: includes them
    res_csv_all = client_admin.post("/api/reports/csv", json={"analysis_ids": aids, "include_mock": True, "include_invalid": True})
    assert res_csv_all.status_code == 200
    assert res_csv_all.headers.get("X-Skipped-Invalid") == "0"
    reader_all = csv.DictReader(io.StringIO(res_csv_all.text))
    rows_all = list(reader_all)
    assert len(rows_all) == 3

    # --- GeoJSON Export ---
    # Default: skips rejected and no_wires
    res_geo = client_admin.post("/api/reports/geojson", json={"analysis_ids": aids, "include_mock": True})
    assert res_geo.status_code == 200
    assert res_geo.headers.get("X-Skipped-Invalid") == "2"
    geo_data = res_geo.json()
    assert geo_data.get("skipped_invalid") == 2
    assert len(geo_data["features"]) == 1
    assert geo_data["features"][0]["properties"]["id"] == 401

    # include_invalid=true: includes them
    res_geo_all = client_admin.post("/api/reports/geojson", json={"analysis_ids": aids, "include_mock": True, "include_invalid": True})
    assert res_geo_all.status_code == 200
    assert res_geo_all.headers.get("X-Skipped-Invalid") == "0"
    geo_data_all = res_geo_all.json()
    assert len(geo_data_all["features"]) == 3

    # --- Summary PDF Export ---
    # Default: skips rejected and no_wires
    res_pdf = client_admin.post("/api/reports/summary", json={"analysis_ids": aids, "include_mock": True})
    assert res_pdf.status_code == 200
    assert res_pdf.headers.get("X-Skipped-Invalid") == "2"

    # include_invalid=true: includes them
    res_pdf_all = client_admin.post("/api/reports/summary", json={"analysis_ids": aids, "include_mock": True, "include_invalid": True})
    assert res_pdf_all.status_code == 200
    assert res_pdf_all.headers.get("X-Skipped-Invalid") == "0"


def test_field_inspector_cannot_include_invalid_in_export(_init_settings):
    conn = _init_settings
    client_insp = _get_auth_client(conn, "field_inspector", "insp_export_user")
    insp_row = conn.execute("SELECT id FROM users WHERE username = 'insp_export_user'").fetchone()
    insp_id = insp_row["id"]

    conn.execute("INSERT INTO analyses (id, filename, status, is_mock, created_by, gps_lat, gps_lng) VALUES (411, 'c.jpg', 'complete', 1, ?, 14.5, 121.0)", (insp_id,))
    conn.execute("INSERT INTO analyses (id, filename, status, is_mock, created_by, gps_lat, gps_lng) VALUES (412, 'r.jpg', 'rejected', 1, ?, 14.5, 121.0)", (insp_id,))
    conn.execute("INSERT INTO analyses (id, filename, status, is_mock, created_by, gps_lat, gps_lng) VALUES (413, 'n.jpg', 'no_wires', 1, ?, 14.5, 121.0)", (insp_id,))
    conn.commit()

    aids = [411, 412, 413]

    # Even with include_invalid=true, field_inspector cannot bypass the filter
    res_csv = client_insp.post("/api/reports/csv", json={"analysis_ids": aids, "include_mock": True, "include_invalid": True})
    assert res_csv.status_code == 200
    assert res_csv.headers.get("X-Skipped-Invalid") == "2"
    reader = csv.DictReader(io.StringIO(res_csv.text))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["filename"] == "c.jpg"


def test_guest_can_open_own_rejected_or_no_wires_result(_init_settings):
    conn = _init_settings
    from backend.auth import _hash_token
    token = "secret_guest_token_123"
    token_hash = _hash_token(token)

    cur1 = conn.execute(
        """INSERT INTO analyses (filename, status, is_public, guest_session_id, is_mock)
           VALUES ('guest_rej.jpg', 'rejected', 1, ?, 1)""",
        (token_hash,)
    )
    rej_id = cur1.lastrowid

    cur2 = conn.execute(
        """INSERT INTO analyses (filename, status, is_public, guest_session_id, is_mock)
           VALUES ('guest_nowires.jpg', 'no_wires', 1, ?, 1)""",
        (token_hash,)
    )
    nowires_id = cur2.lastrowid
    conn.commit()

    client = TestClient(app)
    client.cookies.set("guest_session", token)

    # Guest opening their own rejected result
    res_rej = client.get(f"/api/analysis/{rej_id}/result")
    assert res_rej.status_code == 200
    data_rej = res_rej.json()
    assert data_rej["status"] == "rejected"
    assert data_rej["ground_clearance_status"] == "unavailable"

    # Guest opening their own no_wires result
    res_nw = client.get(f"/api/analysis/{nowires_id}/result")
    assert res_nw.status_code == 200
    data_nw = res_nw.json()
    assert data_nw["status"] == "no_wires"
    assert data_nw["ground_clearance_status"] == "no_wires"


def test_summary_pdf_with_rejected_items(_init_settings, monkeypatch):
    conn = _init_settings
    client_admin = _get_auth_client(conn, "admin", "pdf_rej_admin")

    conn.execute("INSERT OR IGNORE INTO batches (id, status, created_by) VALUES (1, 'complete', 1)")
    conn.execute(
        """INSERT INTO analyses (id, filename, batch_id, status, summary, is_mock, created_by, max_crs, vni)
           VALUES (501, 'normal.jpg', 1, 'complete', 'Valid run', 1, 1, 0.45, 0.3)"""
    )
    conn.execute(
        """INSERT INTO analyses (id, filename, batch_id, status, summary, is_mock, created_by, max_crs, vni)
           VALUES (502, 'rejected.jpg', 1, 'rejected', 'Image blur exceeds threshold', 1, 1, NULL, NULL)"""
    )
    conn.commit()

    from backend.routes.reporting import _get_analysis_data
    rej_data = _get_analysis_data(conn, 502)
    assert rej_data["safety_level"] == "Rejected"
    assert rej_data["action_text"] == "Rejected: Image blur exceeds threshold"

    from reportlab.platypus import SimpleDocTemplate
    captured_stories = []
    orig_build = SimpleDocTemplate.build

    def spy_build(self, flowables, *args, **kwargs):
        captured_stories.append(list(flowables))
        return orig_build(self, flowables, *args, **kwargs)

    monkeypatch.setattr(SimpleDocTemplate, "build", spy_build)

    res = client_admin.post(
        "/api/reports/summary",
        json={"analysis_ids": [501, 502], "include_mock": True, "include_invalid": True}
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert res.content.startswith(b"%PDF")
    assert res.headers.get("X-Skipped-Invalid") == "0"

    assert len(captured_stories) == 1
    story = captured_stories[0]
    from reportlab.platypus import Table
    tables = [f for f in story if isinstance(f, Table)]
    assert len(tables) >= 1
    found_rej = False
    for row in tables[0]._cellvalues:
        cell_texts = [getattr(cell, "text", str(cell)) for cell in row]
        if any("Rejected: Image blur exceeds threshold" in ct for ct in cell_texts):
            found_rej = True
            assert any("Rejected" in ct for ct in cell_texts)
            break
    assert found_rej, "Expected rejected item row in summary PDF table"


def test_mock_data_realism(tmp_path, monkeypatch):
    from backend.utils.cv.mock_generator import generate_mock_segmentation
    from backend.utils.pipeline import execute_pipeline
    from PIL import Image

    # 1. Standard mock
    std_img_path = str(tmp_path / "standard_pole.jpg")
    img = Image.new("RGB", (1920, 1080), color=(128, 128, 128))
    img.save(std_img_path)

    seg_std = generate_mock_segmentation(std_img_path, (1080, 1920))
    assert seg_std.mask_available is True
    assert 3 in seg_std.classes
    assert len(seg_std.boxes) > 1

    from backend.utils.cv.preprocess import PreprocessResult
    import numpy as np

    monkeypatch.setattr(
        "backend.utils.pipeline.validate_and_preprocess",
        lambda path: PreprocessResult(
            quality_ok=True,
            reason="ok",
            gray=np.zeros((1080, 1920), dtype=np.uint8),
            rgb=np.zeros((1080, 1920, 3), dtype=np.uint8),
            original_shape=(1080, 1920)
        )
    )

    conn = get_connection()
    try:
        conn.execute("INSERT OR IGNORE INTO batches (id, status, created_by) VALUES (10, 'complete', NULL)")
        conn.execute("INSERT INTO analyses (id, filename, batch_id, status, is_mock, created_by) VALUES (601, 'standard_pole.jpg', 10, 'pending', 1, NULL)")
        conn.commit()

        monkeypatch.setenv("WIREFRAME_MOCK", "1")
        execute_pipeline(601, std_img_path, conn, model=None, batch_id=10)

        row_std = conn.execute("SELECT clearance_ok, clearance_available FROM analyses WHERE id = 601").fetchone()
        assert row_std["clearance_ok"] in (1, 2), f"Expected clearance_ok in (1, 2), got {row_std['clearance_ok']}"
        assert 5.0 <= row_std["clearance_available"] <= 9.0, f"Expected clearance 5-9m, got {row_std['clearance_available']}"

        # 2. Unsafe mock
        unsafe_img_path = str(tmp_path / "test_unsafe_wires.jpg")
        img.save(unsafe_img_path)

        conn.execute("INSERT INTO analyses (id, filename, batch_id, status, is_mock, created_by) VALUES (602, 'test_unsafe_wires.jpg', 10, 'pending', 1, NULL)")
        conn.commit()

        execute_pipeline(602, unsafe_img_path, conn, model=None, batch_id=10)

        row_unsafe = conn.execute("SELECT clearance_ok, clearance_available FROM analyses WHERE id = 602").fetchone()
        assert row_unsafe["clearance_ok"] == 0, f"Expected clearance_ok == 0 for unsafe mock, got {row_unsafe['clearance_ok']}"
        assert row_unsafe["clearance_available"] < 5.5, f"Expected clearance < 5.5m, got {row_unsafe['clearance_available']}"

        # 3. No wires scenario
        seg_nowires = generate_mock_segmentation("test_nowires.jpg", (1080, 1920))
        assert len(seg_nowires.boxes) == 0

        # 4. No pole scenario
        seg_nopole = generate_mock_segmentation("test_nopole.jpg", (1080, 1920))
        assert 3 not in seg_nopole.classes
    finally:
        conn.close()

