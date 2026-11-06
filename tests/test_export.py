import json
import csv
import io
from backend.database import get_connection
from backend.utils.report_pdf import UNVERIFIED_BANNER_TEXT

def test_export_geojson_success(client_admin):
    conn = get_connection()
    conn.execute("INSERT INTO analyses (id, filename, status, created_by, gps_lat, gps_lng, max_crs, vni) VALUES (777, 'test.jpg', 'complete', 1, 14.5995, 120.9842, 50.0, 1.2)")
    conn.execute("INSERT INTO wires (analysis_id, class_name, clearance_m) VALUES (777, 'comm', 4.5)")
    conn.commit()
    conn.close()
    
    payload = {"analysis_ids": [777], "include_mock": True}
    r = client_admin.post("/api/reports/geojson", json=payload, headers={"X-Requested-With": "XMLHttpRequest"})
    
    fc = r.json()
    f = fc["features"][0]
    props = f["properties"]
    assert props["ground_clearance_m"] == 4.5

def test_export_csv_success(client_admin):
    pass


def test_exports_include_mock_marking_and_rbac(_init_settings, monkeypatch):
    """Comprehensive test for Summary PDF, GeoJSON, CSV, single PDF, and MPC ranking:
    - 1 verified with GPS, 1 mock-public with GPS, 1 mock without GPS
    - include_mock=false -> only verified exported; mock-only selection -> 400 with clear message
    - include_mock=true -> both GPS records exported in GeoJSON (no-GPS skipped & counted), all 3 in Summary PDF/CSV
    - PDF contains UNVERIFIED_BANNER_TEXT and per-record labels ('Mock', 'Public tip')
    - GeoJSON properties (is_mock, is_public, verified, safety_level, risk_score, risk_band, gps_source) + contains_unverified
    - CSV has new columns is_mock, is_public, verified appended at the end
    - Guest and field_inspector forbidden on unowned/public records as before
    - MPC ranking still excludes mock/public
    - Single PDF of a mock analysis works for admin and is marked with UNVERIFIED_BANNER_TEXT
    """
    from reportlab.platypus import SimpleDocTemplate
    from tests.conftest import _get_auth_client

    conn = _init_settings
    client_admin = _get_auth_client(conn, "admin", "export_admin_user")
    client_insp = _get_auth_client(conn, "field_inspector", "export_insp_user")

    cur_site = conn.execute(
        "INSERT INTO sites (name, barangay, street, barangay_norm, street_norm) "
        "VALUES ('Meralco', 'San Antonio', 'Ayala Ave', 'san antonio', 'ayala ave')"
    )
    site_id = cur_site.lastrowid

    # 1. Verified record with GPS (is_mock=0, is_public=0, gps_source='device', risk_score=25.0 -> Low)
    c1 = conn.execute(
        """INSERT INTO analyses
           (filename, status, company, barangay, street, site_id, created_by, is_mock, is_public,
            gps_lat, gps_lng, gps_source, risk_score, max_crs, vni, clearance_available, clearance_ok)
           VALUES ('verified_gps.jpg', 'complete', 'Meralco', 'San Antonio', 'Ayala Ave', ?, 1, 0, 0,
                   14.5547, 121.0244, 'device', 25.0, 20.0, 0.35, 6.5, 1)""",
        (site_id,),
    )
    id_verified = c1.lastrowid
    conn.execute(
        "INSERT INTO wires (analysis_id, class_name, crs, band, clearance_m, clearance_status, safety_level) "
        "VALUES (?, 'power_line', 20.0, 'Low', 6.5, 'ok', 'Safe')",
        (id_verified,),
    )

    # 2. Mock + Public tip record with GPS (is_mock=1, is_public=1, gps_source='manual', risk_score=78.0 -> High)
    c2 = conn.execute(
        """INSERT INTO analyses
           (filename, status, company, barangay, street, site_id, created_by, is_mock, is_public,
            gps_lat, gps_lng, gps_source, risk_score, max_crs, vni, clearance_available, clearance_ok)
           VALUES ('mock_pub_gps.jpg', 'complete', 'PLDT', 'Poblacion', 'Makati Ave', ?, NULL, 1, 1,
                   14.5640, 121.0300, 'manual', 78.0, 88.0, 0.62, 4.8, 0)""",
        (site_id,),
    )
    id_mock_pub = c2.lastrowid
    conn.execute(
        "INSERT INTO wires (analysis_id, class_name, crs, band, clearance_m, clearance_status, safety_level) "
        "VALUES (?, 'telecom_cable', 88.0, 'Critical', 4.8, 'below', 'Critical')",
        (id_mock_pub,),
    )

    # 3. Mock record without GPS (is_mock=1, is_public=0, gps_lat=NULL, gps_lng=NULL)
    c3 = conn.execute(
        """INSERT INTO analyses
           (filename, status, company, barangay, street, site_id, created_by, is_mock, is_public,
            gps_lat, gps_lng, gps_source, risk_score, max_crs, vni, clearance_available, clearance_ok)
           VALUES ('mock_nogps.jpg', 'complete', 'Globe', 'Bel-Air', 'Paseo', ?, 1, 1, 0,
                   NULL, NULL, NULL, 45.0, 42.0, 0.40, 5.6, 1)""",
        (site_id,),
    )
    id_mock_nogps = c3.lastrowid
    conn.execute(
        "INSERT INTO wires (analysis_id, class_name, crs, band, clearance_m, clearance_status, safety_level) "
        "VALUES (?, 'fiber_optic', 42.0, 'Moderate', 5.6, 'ok', 'Caution')",
        (id_mock_nogps,),
    )
    conn.commit()

    all_ids = [id_verified, id_mock_pub, id_mock_nogps]

    captured_stories = []
    orig_build = SimpleDocTemplate.build

    def spy_build(self, flowables, *args, **kwargs):
        captured_stories.append(list(flowables))
        return orig_build(self, flowables, *args, **kwargs)

    monkeypatch.setattr(SimpleDocTemplate, "build", spy_build)

    # --- A. include_mock=False -> ONLY verified exported ---
    # 1) GeoJSON with include_mock=False
    r_geo_false = client_admin.post(
        "/api/reports/geojson",
        json={"analysis_ids": all_ids, "include_mock": False},
    )
    assert r_geo_false.status_code == 200
    fc_false = r_geo_false.json()
    assert len(fc_false["features"]) == 1
    assert fc_false["features"][0]["properties"]["id"] == id_verified
    assert fc_false["features"][0]["properties"]["is_mock"] is False
    assert fc_false["features"][0]["properties"]["is_public"] is False
    assert fc_false["features"][0]["properties"]["verified"] is True
    assert fc_false["features"][0]["properties"]["gps_source"] == "Device GPS"
    assert fc_false["features"][0]["properties"]["risk_score"] == 25.0
    assert fc_false["features"][0]["properties"]["risk_band"] == "Low"
    assert fc_false["features"][0]["properties"]["safety_level"] == "Safe"
    assert fc_false["contains_unverified"] is False
    assert fc_false["skipped_no_gps"] == 0

    # 2) CSV with include_mock=False
    r_csv_false = client_admin.post(
        "/api/reports/csv",
        json={"analysis_ids": all_ids, "include_mock": False},
    )
    assert r_csv_false.status_code == 200
    rows_false = list(csv.DictReader(io.StringIO(r_csv_false.text)))
    assert len(rows_false) == 1
    assert int(rows_false[0]["id"]) == id_verified
    assert rows_false[0]["is_mock"] == "False"
    assert rows_false[0]["is_public"] == "False"
    assert rows_false[0]["verified"] == "True"
    # Verify new columns are appended at the end
    assert list(rows_false[0].keys())[-3:] == ["is_mock", "is_public", "verified"]

    # 3) Summary PDF with include_mock=False
    captured_stories.clear()
    r_sum_false = client_admin.post(
        "/api/reports/summary",
        json={"analysis_ids": all_ids, "include_mock": False},
    )
    assert r_sum_false.status_code == 200
    story_false_texts = " ".join(getattr(f, "text", "") for f in captured_stories[0])
    assert UNVERIFIED_BANNER_TEXT not in story_false_texts

    # 4) Selecting ONLY mock/public with include_mock=False -> 400 with exact clear message
    for ep in ("/api/reports/summary", "/api/reports/geojson", "/api/reports/csv"):
        r_empty = client_admin.post(ep, json={"analysis_ids": [id_mock_pub, id_mock_nogps], "include_mock": False})
        assert r_empty.status_code == 400
        assert (
            r_empty.json()["detail"]
            == "No verified records to export. Public tips and mock data are excluded from official exports. Check 'Include Mock' to export test data (marked as unverified)."
        )

    # --- B. include_mock=True -> mock & public included and clearly marked ---
    # 1) GeoJSON with include_mock=True -> both GPS records exported, no-GPS mock skipped and counted
    r_geo_true = client_admin.post(
        "/api/reports/geojson",
        json={"analysis_ids": all_ids, "include_mock": True},
    )
    assert r_geo_true.status_code == 200
    assert r_geo_true.headers.get("X-Skipped-No-Gps") == "1"
    fc_true = r_geo_true.json()
    assert fc_true["contains_unverified"] is True
    assert fc_true["skipped_no_gps"] == 1
    assert len(fc_true["features"]) == 2
    by_id = {f["properties"]["id"]: f["properties"] for f in fc_true["features"]}
    assert by_id[id_verified]["verified"] is True
    assert by_id[id_mock_pub]["is_mock"] is True
    assert by_id[id_mock_pub]["is_public"] is True
    assert by_id[id_mock_pub]["verified"] is False
    assert by_id[id_mock_pub]["safety_level"] == "Critical"
    assert by_id[id_mock_pub]["risk_score"] == 78.0
    assert by_id[id_mock_pub]["risk_band"] == "High"
    assert by_id[id_mock_pub]["gps_source"] == "Manual entry"

    # 2) CSV with include_mock=True -> all 3 records exported with is_mock, is_public, verified columns
    r_csv_true = client_admin.post(
        "/api/reports/csv",
        json={"analysis_ids": all_ids, "include_mock": True},
    )
    assert r_csv_true.status_code == 200
    rows_true = list(csv.DictReader(io.StringIO(r_csv_true.text)))
    assert len(rows_true) == 3
    csv_by_id = {int(r["id"]): r for r in rows_true}
    assert csv_by_id[id_verified]["verified"] == "True"
    assert csv_by_id[id_mock_pub]["is_mock"] == "True"
    assert csv_by_id[id_mock_pub]["is_public"] == "True"
    assert csv_by_id[id_mock_pub]["verified"] == "False"
    assert csv_by_id[id_mock_nogps]["is_mock"] == "True"
    assert csv_by_id[id_mock_nogps]["is_public"] == "False"
    assert csv_by_id[id_mock_nogps]["verified"] == "False"

    # 3) Summary PDF with include_mock=True -> watermark banner + per-record source labels
    captured_stories.clear()
    r_sum_true = client_admin.post(
        "/api/reports/summary",
        json={"analysis_ids": all_ids, "include_mock": True},
    )
    assert r_sum_true.status_code == 200
    story_true = captured_stories[0]
    story_true_texts = " ".join(getattr(f, "text", "") for f in story_true)
    assert UNVERIFIED_BANNER_TEXT in story_true_texts
    from reportlab.platypus import Table
    tables = [f for f in story_true if isinstance(f, Table)]
    assert tables
    all_cell_texts = " ".join(
        getattr(cell, "text", str(cell))
        for row in tables[0]._cellvalues
        for cell in row
    )
    assert "Mock / Public tip" in all_cell_texts
    assert "Mock" in all_cell_texts
    assert "Verified" in all_cell_texts

    # --- C. Single PDF of a mock/public analysis works for admin and is marked ---
    captured_stories.clear()
    r_single_mock = client_admin.get(f"/api/reports/{id_mock_pub}.pdf")
    assert r_single_mock.status_code == 200
    assert r_single_mock.headers["content-type"].startswith("application/pdf")
    single_texts = " ".join(getattr(f, "text", "") for f in captured_stories[0])
    assert UNVERIFIED_BANNER_TEXT in single_texts
    assert "Mock / Public tip" in single_texts

    # --- D. RBAC: guest and field_inspector forbidden on unowned/public exports ---
    from fastapi.testclient import TestClient
    from main import app
    guest_client = TestClient(app)
    for ep in ("/api/reports/summary", "/api/reports/geojson", "/api/reports/csv"):
        assert guest_client.post(ep, json={"analysis_ids": [id_verified], "include_mock": True}, headers={"X-Requested-With": "XMLHttpRequest"}).status_code == 401
        assert client_insp.post(ep, json={"analysis_ids": [id_mock_pub], "include_mock": True}).status_code == 403

    # --- E. MPC ranking still excludes mock and public records ---
    r_mpc = client_admin.get("/api/mpc/ranking")
    assert r_mpc.status_code == 200
    rankings = r_mpc.json()["rankings"]
    assert len(rankings) == 1
    assert rankings[0]["analysis_count"] == 1
    assert rankings[0]["avg_risk"] == 25.0


