import pytest
from fastapi.testclient import TestClient

def test_stats_rbac(client_inspector: TestClient, client_engineer: TestClient):
    # inspector should get 403
    resp = client_inspector.get("/api/stats/dashboard")
    assert resp.status_code == 403
    
    # engineer should get 200
    resp = client_engineer.get("/api/stats/dashboard")
    assert resp.status_code == 200

def test_stats_validation(client_engineer: TestClient):
    # test limits
    resp = client_engineer.get("/api/stats/dashboard?limit=200")
    assert resp.status_code == 422
    
    resp = client_engineer.get("/api/stats/dashboard?days=100")
    assert resp.status_code == 422

def test_stats_aggregation(client_admin: TestClient, _init_settings):
    # Seed data
    _init_settings.execute("INSERT INTO sites (name) VALUES ('Site1')")
    site_id = 1
    
    # Mock analysis
    _init_settings.execute(
        "INSERT INTO analyses (filename, status, risk_score, is_mock, created_at) VALUES (?, ?, ?, ?, datetime('now'))",
        ("mock.jpg", "complete", 0.10, 1)
    )
    
    # Real analysis (Low CRS, but Unsafe due to clearance)
    _init_settings.execute(
        "INSERT INTO analyses (filename, status, risk_score, site_id, is_mock, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
        ("real.jpg", "complete", 0.15, site_id, 0)
    )
    analysis_id = _init_settings.execute("SELECT last_insert_rowid()").fetchone()[0]
    
    # Real wire (for distribution and driving to Unsafe due to clearance)
    _init_settings.execute(
        "INSERT INTO wires (analysis_id, class_name, crs, band, clearance_m, clearance_status, safety_level) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (analysis_id, "power_line", 10.0, "Low", 2.0, "below", "Unsafe")
    )
    
    _init_settings.commit()

    # Default (no mock)
    resp = client_admin.get("/api/stats/dashboard")
    assert resp.status_code == 200
    data = resp.json()
    
    assert data["kpis"]["total_analyses"] == 1
    assert data["kpis"]["high_critical_count"] == 1
    assert data["kpis"]["average_risk_score"] == 0.1
    
    assert data["distribution"]["power_distribution_line"] == 1
    assert data["distribution"]["communication_cable"] == 0
    
    assert len(data["heatmap"]) == 1
    assert data["heatmap"][0]["site_name"] == "Site1"
    assert data["heatmap"][0]["risk_score"] == 0.15
    
    assert len(data["trend"]) == 1
    assert data["trend"][0]["count"] == 1
    assert data["trend"][0]["avg_risk"] == 0.1
    
    # With mock
    resp = client_admin.get("/api/stats/dashboard?include_mock=true")
    assert resp.status_code == 200
    data = resp.json()
    assert data["kpis"]["total_analyses"] == 2
    assert data["kpis"]["high_critical_count"] == 1
    assert len(data["heatmap"]) == 2


def test_stats_includes_public_submissions_by_default(client_admin: TestClient, _init_settings):
    # Seed 1 verified inspector analysis (is_public=0) and 1 real public submission (is_public=1, is_mock=0)
    _init_settings.execute(
        "INSERT INTO analyses (filename, status, risk_score, is_mock, is_public, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
        ("verified.jpg", "complete", 20.0, 0, 0)
    )
    _init_settings.execute(
        "INSERT INTO analyses (filename, status, risk_score, is_mock, is_public, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
        ("public_tip.jpg", "complete", 80.0, 0, 1)
    )
    _init_settings.commit()

    # By default (include_mock=False), real Public Submissions ARE included on the Admin Dashboard
    resp = client_admin.get("/api/stats/dashboard")
    assert resp.status_code == 200
    data = resp.json()
    assert data["kpis"]["total_analyses"] == 2
    assert len(data["heatmap"]) == 2
    scores = {h["risk_score"] for h in data["heatmap"]}
    assert scores == {20.0, 80.0}
    assert any(r.get("submission_type") == "Public Submission" for r in data.get("recent_analyses", []))


def test_unauthenticated_post_api_analyze_increments_dashboard_and_history(
    client_unauth: TestClient,
    client_admin: TestClient,
    clean_image,
):
    """Posting an unauthenticated image to /api/analyze saves fallback GPS and immediately updates /api/dashboard & /api/history."""
    import cv2
    from backend.database import DEFAULT_PUBLIC_GPS_LAT, DEFAULT_PUBLIC_GPS_LNG, get_connection

    before_dash = client_admin.get("/api/dashboard").json()
    before_total = before_dash["kpis"]["total_analyses"]

    sharp_img = clean_image.copy()
    cv2.rectangle(sharp_img, (290, 60), (330, 470), (40, 40, 40), -1)
    for y_off in range(80, 260, 25):
        cv2.line(sharp_img, (20, y_off), (620, y_off + 30), (15, 15, 15), 2)
    image_bytes = cv2.imencode(".jpg", sharp_img)[1].tobytes()
    res = client_unauth.post(
        "/api/analyze",
        files={"file": ("public_pole_report.jpg", image_bytes, "image/jpeg")},
    )
    assert res.status_code == 200, res.text
    payload = res.json()
    aid = payload["analysis_id"]
    assert payload["is_public"] == 1
    assert payload["submission_type"] == "Public Submission"
    assert payload["status"] == "complete"
    assert payload["gps_lat"] == pytest.approx(DEFAULT_PUBLIC_GPS_LAT)
    assert payload["gps_lng"] == pytest.approx(DEFAULT_PUBLIC_GPS_LNG)

    # Verify DB row persisted with valid fallback coordinates
    conn = get_connection()
    row = conn.execute("SELECT is_public, gps_lat, gps_lng, gps_source, status FROM analyses WHERE id = ?", (aid,)).fetchone()
    conn.close()
    assert row is not None
    assert row["is_public"] == 1
    assert row["status"] == "complete"
    assert row["gps_lat"] == pytest.approx(DEFAULT_PUBLIC_GPS_LAT)
    assert row["gps_lng"] == pytest.approx(DEFAULT_PUBLIC_GPS_LNG)
    assert row["gps_source"] is None

    # Immediately verify /api/dashboard increments TOTAL ANALYSES and shows item in Recent Analyses
    after_dash = client_admin.get("/api/dashboard").json()
    assert after_dash["kpis"]["total_analyses"] == before_total + 1
    recent_ids = [r["id"] for r in after_dash["recent_analyses"]]
    assert aid in recent_ids
    matched_recent = next(r for r in after_dash["recent_analyses"] if r["id"] == aid)
    assert matched_recent["is_public"] == 1
    assert matched_recent["submission_type"] == "Public Submission"

    # Immediately verify /api/history and /api/analysis return the Public Submission by default
    hist_res = client_admin.get("/api/history")
    assert hist_res.status_code == 200
    hist_items = hist_res.json()["data"]
    assert any(it["id"] == aid and it["is_public"] == 1 and it["submission_type"] == "Public Submission" for it in hist_items)

    list_res = client_admin.get("/api/analysis")
    assert list_res.status_code == 200
    list_items = list_res.json()["data"]
    assert any(it["id"] == aid and it["is_public"] == 1 for it in list_items)


def test_unauthenticated_upload_populates_stats_recent_analyses_and_get_analysis(
    client_unauth: TestClient,
    client_admin: TestClient,
    clean_image,
    monkeypatch,
):
    """Regression: unauthenticated POST /api/analysis/upload populates GET /api/stats/dashboard recent_analyses (>= 1) and GET /api/analysis."""
    import os
    import cv2
    import backend.routes.analysis as analysis_route
    from backend.database import get_connection
    from backend.utils.analysis_engine import process_analysis
    from backend import main_state

    monkeypatch.setattr(analysis_route, "enqueue_batch", lambda _bid: None)

    sharp_img = clean_image.copy()
    cv2.rectangle(sharp_img, (280, 50), (320, 470), (35, 35, 35), -1)
    for y_off in range(70, 250, 25):
        cv2.line(sharp_img, (15, y_off), (625, y_off + 28), (10, 10, 10), 2)
    image_bytes = cv2.imencode(".jpg", sharp_img)[1].tobytes()

    up_res = client_unauth.post(
        "/api/analysis/upload",
        data={"company": "Meralco", "barangay": "San Antonio", "street": "Pearl Drive"},
        files={"file": ("public_upload_check.jpg", image_bytes, "image/jpeg")},
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert up_res.status_code == 200, up_res.text
    aid = up_res.json()["analysis_id"]
    bid = up_res.json()["batch_id"]

    conn = get_connection()
    row = conn.execute("SELECT filename FROM analyses WHERE id = ?", (aid,)).fetchone()
    process_analysis(aid, os.path.join("uploads", row["filename"]), conn, main_state.get_model(), bid)
    # Mark as real non-mock public submission (is_mock=0, is_public=1) to verify default include_mock=false path
    conn.execute("UPDATE analyses SET is_mock = 0 WHERE id = ?", (aid,))
    conn.commit()
    conn.close()

    stats_res = client_admin.get("/api/stats/dashboard")
    assert stats_res.status_code == 200
    stats_body = stats_res.json()
    assert stats_body["kpis"]["total_analyses"] >= 1
    assert len(stats_body["recent_analyses"]) >= 1
    assert stats_body["recent_analyses"][0]["id"] == aid
    assert stats_body["recent_analyses"][0]["is_public"] == 1

    analysis_res = client_admin.get("/api/analysis")
    assert analysis_res.status_code == 200
    analysis_items = analysis_res.json()["data"]
    assert any(it["id"] == aid and it["is_public"] == 1 for it in analysis_items)


def test_dashboard_stats_mock_public_regression(client_admin: TestClient, _init_settings):
    """Regression test: DB with 1 public mock row (is_public=1, is_mock=1, status='complete', risk_score=25.0)

    and 1 verified row (is_public=0, is_mock=0, status='complete', risk_score=80.0).
    - include_mock=false (OFF): Only verified counted (total=1, risk=80.0) across KPIs, heatmap, trend.
    - include_mock=true (ON): Both counted (total=2, avg_risk=52.5) across KPIs, heatmap, trend.
    """
    conn = _init_settings
    # 1 verified inspector record
    conn.execute(
        "INSERT INTO analyses (filename, status, risk_score, is_mock, is_public, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
        ("verified_insp.jpg", "complete", 80.0, 0, 0)
    )
    # 1 public mock row (guest mock upload)
    conn.execute(
        "INSERT INTO analyses (filename, status, risk_score, is_mock, is_public, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
        ("public_guest_mock.jpg", "complete", 25.0, 1, 1)
    )
    conn.commit()

    # OFF: unchecked
    off_res = client_admin.get("/api/stats/dashboard?include_mock=false")
    assert off_res.status_code == 200
    off_data = off_res.json()
    assert off_data["kpis"]["total_analyses"] == 1
    assert off_data["kpis"]["average_risk_score"] == 80.0
    assert len(off_data["heatmap"]) == 1
    assert off_data["heatmap"][0]["risk_score"] == 80.0
    assert sum(t["count"] for t in off_data["trend"]) == 1

    # ON: checked
    on_res = client_admin.get("/api/stats/dashboard?include_mock=true")
    assert on_res.status_code == 200
    on_data = on_res.json()
    assert on_data["kpis"]["total_analyses"] == 2
    assert on_data["kpis"]["average_risk_score"] == 52.5
    assert len(on_data["heatmap"]) == 2
    heatmap_scores = {h["risk_score"] for h in on_data["heatmap"]}
    assert heatmap_scores == {25.0, 80.0}
    assert sum(t["count"] for t in on_data["trend"]) == 2

