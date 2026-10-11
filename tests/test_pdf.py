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
