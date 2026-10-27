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
    
    assert data["distribution"]["power_line"] == 1
    assert data["distribution"]["telecom_cable"] == 0
    
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


def test_stats_excludes_public_submissions(client_admin: TestClient, _init_settings):
    # Seed 1 verified inspector analysis (is_public=0) and 1 public tip (is_public=1)
    _init_settings.execute(
        "INSERT INTO analyses (filename, status, risk_score, is_mock, is_public, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
        ("verified.jpg", "complete", 20.0, 0, 0)
    )
    _init_settings.execute(
        "INSERT INTO analyses (filename, status, risk_score, is_mock, is_public, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
        ("public_tip.jpg", "complete", 80.0, 0, 1)
    )
    _init_settings.commit()

    resp = client_admin.get("/api/stats/dashboard")
    assert resp.status_code == 200
    data = resp.json()
    assert data["kpis"]["total_analyses"] == 1
    assert len(data["heatmap"]) == 1
    assert data["heatmap"][0]["risk_score"] == 20.0

    # When include_mock=True, mock/public submissions are included in KPIs, heatmap, and trend
    resp_mock = client_admin.get("/api/stats/dashboard?include_mock=true")
    assert resp_mock.status_code == 200
    data_mock = resp_mock.json()
    assert data_mock["kpis"]["total_analyses"] == 2
    assert len(data_mock["heatmap"]) == 2
    scores = {h["risk_score"] for h in data_mock["heatmap"]}
    assert scores == {20.0, 80.0}
    assert sum(t["count"] for t in data_mock["trend"]) == 2


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

