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
