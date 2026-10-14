from backend.database import get_connection

# We can rely on client_admin fixture from conftest.py

def test_pdf_not_found_or_unauthorized(client_admin):
    # Try fetching a non-existent PDF
    r = client_admin.get("/api/reports/999.pdf")
    # API specs say 404 for non-existent (same as unauthorized to avoid leaking existence)
    assert r.status_code == 404

def test_pdf_incomplete_status(client_admin):
    conn = get_connection()
    conn.execute("INSERT INTO analyses (id, filename, status, created_by) VALUES (123, 'test.jpg', 'queued', 1)")
    conn.commit()
    conn.close()
    
    r = client_admin.get("/api/reports/123.pdf")
    assert r.status_code == 409
    assert "detail" in r.json()

def test_pdf_generation_success(client_admin):
    conn = get_connection()
    conn.execute("INSERT INTO batches (id, status, created_by) VALUES (1, 'complete', 1)")
    conn.execute("INSERT INTO analyses (id, filename, batch_id, status, created_by) VALUES (123, 'test.jpg', 1, 'complete', 1)")
    conn.commit()
    conn.close()
    
    r = client_admin.get("/api/reports/123.pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")

def test_summary_pdf(client_admin):
    conn = get_connection()
    # ensure batches exist
    conn.execute("INSERT OR IGNORE INTO batches (id, status, created_by) VALUES (1, 'complete', 1)")
    conn.execute("INSERT OR IGNORE INTO analyses (id, filename, batch_id, status, created_by) VALUES (123, 'test.jpg', 1, 'complete', 1)")
    conn.commit()
    conn.close()

    r = client_admin.post("/api/reports/summary", json={"analysis_ids": ["123"]})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")

def test_pdf_cache_invalidation_on_template_version_change(client_admin, monkeypatch):
    import backend.routes.reporting
    import json
    
    conn = get_connection()
    conn.execute("INSERT OR IGNORE INTO batches (id, status, created_by) VALUES (1, 'complete', 1)")
    conn.execute("INSERT OR IGNORE INTO analyses (id, filename, batch_id, status, created_by) VALUES (456, 'test2.jpg', 1, 'complete', 1)")
    conn.commit()
    conn.close()
    
    # Generate first time
    monkeypatch.setattr(backend.routes.reporting, "TEMPLATE_VERSION", "v1")
    r1 = client_admin.get("/api/reports/456.pdf")
    assert r1.status_code == 200
    
    conn = get_connection()
    rep1 = conn.execute("SELECT metadata FROM reports WHERE analysis_id = 456").fetchone()
    meta1 = json.loads(rep1["metadata"])
    conn.close()
    
    # Generate second time with SAME template version (should hit cache)
    r2 = client_admin.get("/api/reports/456.pdf")
    assert r2.status_code == 200
    
    conn = get_connection()
    rep2 = conn.execute("SELECT metadata FROM reports WHERE analysis_id = 456").fetchone()
    meta2 = json.loads(rep2["metadata"])
    conn.close()
    assert meta1["source_version"] == meta2["source_version"]
    
    # Generate third time with DIFFERENT template version (should regenerate)
    monkeypatch.setattr(backend.routes.reporting, "TEMPLATE_VERSION", "v2")
    r3 = client_admin.get("/api/reports/456.pdf")
    assert r3.status_code == 200
    
    conn = get_connection()
    rep3 = conn.execute("SELECT metadata FROM reports WHERE analysis_id = 456").fetchone()
    meta3 = json.loads(rep3["metadata"])
    conn.close()
    
    assert meta1["source_version"] != meta3["source_version"]

def test_format_pht_timestamp():
    from backend.utils.report_pdf import format_pht_timestamp
    res = format_pht_timestamp("2026-10-11 06:49:00")
    assert res == "2026-10-11 14:49:00 PHT"
    res_date = format_pht_timestamp("2026-10-11 06:49:00", include_time=False)
    assert res_date == "2026-10-11"
    assert format_pht_timestamp(None) == "--"

def test_build_pdf_and_summary_titles(tmp_path, monkeypatch):
    import os
    from backend.utils.report_pdf import build_pdf, build_summary_pdf
    from reportlab.platypus import SimpleDocTemplate

    captured_stories = []
    orig_build = SimpleDocTemplate.build

    def mock_build(self, flowables, *args, **kwargs):
        captured_stories.append(list(flowables))
        return orig_build(self, flowables, *args, **kwargs)

    monkeypatch.setattr(SimpleDocTemplate, "build", mock_build)

    pdf_path = str(tmp_path / "single.pdf")
    build_pdf({"id": "TEST-123"}, pdf_path)
    assert os.path.exists(pdf_path)
    assert any("WIREFRAME Report: TEST-123" in getattr(f, "text", "") for f in captured_stories[0])

    summary_path = str(tmp_path / "summary.pdf")
    build_summary_pdf({"items": []}, summary_path)
    assert os.path.exists(summary_path)
    assert any("WIREFRAME Summary Report" in getattr(f, "text", "") for f in captured_stories[1])


def test_pdf_and_summary_with_and_without_clearance_columns(client_admin, monkeypatch):
    """Assert PDF and summary generation succeed with and without clearance columns, and check pole height."""
    from reportlab.platypus import SimpleDocTemplate

    conn = get_connection()
    conn.execute("INSERT OR IGNORE INTO batches (id, status, created_by) VALUES (1, 'complete', 1)")
    
    # 1. With clearance columns populated (assumed_pole_height_m = 10.0)
    conn.execute("""
        INSERT INTO analyses (
            id, filename, batch_id, status, created_by,
            clearance_ok, clearance_available, pec_min_clearance_m, sag_meters, assumed_pole_height_m
        ) VALUES (801, 'with_clr.jpg', 1, 'complete', 1, 1, 6.2, 5.5, 0.45, 10.0)
    """)
    
    # 2. Without clearance columns (NULL)
    conn.execute("""
        INSERT INTO analyses (
            id, filename, batch_id, status, created_by,
            clearance_ok, clearance_available, pec_min_clearance_m, sag_meters, assumed_pole_height_m
        ) VALUES (802, 'without_clr.jpg', 1, 'complete', 1, NULL, NULL, NULL, NULL, NULL)
    """)
    conn.commit()

    captured_stories = []
    orig_build = SimpleDocTemplate.build

    def mock_build(self, flowables, *args, **kwargs):
        captured_stories.append(list(flowables))
        return orig_build(self, flowables, *args, **kwargs)

    monkeypatch.setattr(SimpleDocTemplate, "build", mock_build)

    # Fetch PDF via API for analysis with clearance columns
    r1 = client_admin.get("/api/reports/801.pdf")
    assert r1.status_code == 200
    assert r1.headers["content-type"] == "application/pdf"
    assert r1.content.startswith(b"%PDF")

    # Fetch PDF via API for analysis without clearance columns
    r2 = client_admin.get("/api/reports/802.pdf")
    assert r2.status_code == 200
    assert r2.headers["content-type"] == "application/pdf"
    assert r2.content.startswith(b"%PDF")

    # Verify flowable content
    story_with = captured_stories[0]
    config_texts_with = [getattr(f, "text", "") for f in story_with if "Config Snapshot:" in getattr(f, "text", "")]
    assert len(config_texts_with) == 1
    assert "Pole height: 10.0m" in config_texts_with[0]

    story_without = captured_stories[1]
    config_texts_without = [getattr(f, "text", "") for f in story_without if "Config Snapshot:" in getattr(f, "text", "")]
    assert len(config_texts_without) == 1
    assert "Pole height: m" in config_texts_without[0]

    # Generate summary report containing both analyses
    r_sum = client_admin.post("/api/reports/summary", json={"analysis_ids": ["801", "802"]})
    assert r_sum.status_code == 200
    assert r_sum.headers["content-type"] == "application/pdf"
    assert r_sum.content.startswith(b"%PDF")

    conn.close()


