import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.database import get_connection, run_migrations
from backend.main_state import set_config_failed
from backend.services.worker import stop_worker
from main import app


@pytest.fixture(autouse=True)
def setup_teardown():
    Path("uploads").mkdir(exist_ok=True)
    # clean db
    stop_worker()
    if os.path.exists("wireframe.db"):
        try:
            os.remove("wireframe.db")
        except Exception:
            pass
    conn = get_connection()
    run_migrations(conn)
    conn.close()
    
    set_config_failed(False)
    
    yield
    
    stop_worker()
    if os.path.exists("wireframe.db"):
        try:
            os.remove("wireframe.db")
        except Exception:
            pass

def create_synthetic_image(name):
    img = Image.new('RGB', (100, 100), color = 'red')
    img.save(f"uploads/{name}")
    return f"uploads/{name}"

def test_startup_zombie_recovery():
    # Insert zombies
    conn = get_connection()
    run_migrations(conn)
    conn.execute("INSERT INTO batches (id, status) VALUES (1, 'running')")
    conn.execute("INSERT INTO analyses (id, batch_id, filename, status, company, street) VALUES (1, 1, 'f1.jpg', 'processing', 'C', 'S')")
    conn.execute("INSERT INTO analyses (id, batch_id, filename, status, company, street) VALUES (2, 1, 'f2.jpg', 'queued', 'C', 'S')")
    conn.execute("INSERT OR IGNORE INTO settings (key, value_json) VALUES ('pole_heights', '\"9,10,12\"')")
    conn.commit()
    conn.close()
    
    with TestClient(app):
        pass # triggers lifespan
        
    conn = get_connection()
    b = conn.execute("SELECT status FROM batches WHERE id = 1").fetchone()
    assert b["status"] == "interrupted"
    
    a1 = conn.execute("SELECT status, summary FROM analyses WHERE id = 1").fetchone()
    assert a1["status"] == "error"
    assert "interrupted by server restart" in a1["summary"]
    
    a2 = conn.execute("SELECT status, summary FROM analyses WHERE id = 2").fetchone()
    assert a2["status"] == "error"
    assert "interrupted by server restart" in a2["summary"]
    conn.close()

@pytest.fixture
def test_image(clean_image):
    import io

    from PIL import Image
    img = Image.fromarray(clean_image)
    b = io.BytesIO()
    img.save(b, format="JPEG")
    b.seek(0)
    return b

def test_integration_upload_rejected(client_admin, test_image, monkeypatch):
    from backend import main_state
    monkeypatch.setattr(main_state, "is_config_failed", lambda: True)
    
    res = client_admin.post(
        "/api/analysis/upload",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"barangay": "Test", "sector": "A"}
    )
    assert res.status_code == 503

def test_get_or_create_site_concurrent():
    import threading
    from backend.services.sites import get_or_create_site
    from backend.database import get_connection
    
    
    results = []
    def worker():
        try:
            local_conn = get_connection()
            site_id = get_or_create_site(local_conn, "Test Co", "Brgy. Concurrency", "Thread St")
            results.append(site_id)
            local_conn.close()
        except Exception as e:
            results.append(e)
            
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
    
    # Check that all threads got the exact same site_id without IntegrityError
    assert all(isinstance(r, int) for r in results)
    assert len(set(results)) == 1

def test_worker_survives_exception(caplog):
    import time
    from backend.services.worker import start_worker, stop_worker, batch_queue
    import logging
    
    # Clear any residual None tokens
    batch_queue.queue.clear()
    
    # Ensure worker is started
    start_worker()
    
    # Inject a bad type that causes conn.execute to throw sqlite3.ProgrammingError or similar
    class BadBatchID:
        def __conform__(self, protocol):
            raise Exception("Mock worker failure")
            
        def __str__(self):
            return "BadBatch"
            
    bad_id = BadBatchID()
    
    with caplog.at_level(logging.ERROR):
        # Enqueue bad batch ID
        batch_queue.put(bad_id)
        # Give worker a moment to process and fail
        time.sleep(0.5)
        
    # Thread should still be alive
    from backend.services.worker import _worker_thread
    assert _worker_thread is not None
    assert _worker_thread.is_alive()
    
    # The error should be logged
    assert any("Batch worker exception" in record.message for record in caplog.records)
    
    stop_worker()

def test_upload_corrupt_image_logs_error(caplog, client_admin):
    import logging
    with caplog.at_level(logging.ERROR):
        # Corrupt image bytes (not a valid image)
        bad_image_bytes = b"not an image file"
        
        data = {
            "company": "Test Power Co",
            "barangay": "Brgy. San Jose",
            "street": "Rizal St",
            "sector": "",       
            "gps_lat": "14.0",      
            "gps_lng": "121.0"
        }
        files = {"file": ("test_image.jpg", bad_image_bytes, "image/jpeg")}
        
        res = client_admin.post("/api/analysis/upload", data=data, files=files)
        
        assert res.status_code == 400
        assert res.json()["detail"] == "Corrupt image"
        
        # Verify it logged the exception from the broad except block
        assert any("Invalid image content detected" in record.message for record in caplog.records)

import pytest
from backend.services.sites import _normalize, get_or_create_site

@pytest.mark.parametrize("input_name,expected", [
    ("Brgy. San Jose", "san jose"),
    ("Brgy San Jose", "san jose"),
    ("brgy.san jose", "san jose"),
    ("BARANGAY San Jose", "san jose"),
    ("  San   Jose  ", "san jose"),
    ("Barangay 123", "123"),
    ("Brgy. Niño", "niño"),
    ("Bragancia", "bragancia"),
])
def test_site_normalization_unit(input_name, expected):
    assert _normalize(input_name) == expected

def test_site_normalization_site_id():
    conn = get_connection()
    # Different companies, same location (normalized)
    id1 = get_or_create_site(conn, "Company A", "Brgy. San Jose", "Main St.", 14.0, 121.0)
    id2 = get_or_create_site(conn, "Company B", "brgy.san jose", "Main St.", 14.0, 121.0)
    
    # Assert they resolve to the SAME site_id
    assert id1 == id2
    
    # Different location
    id_diff = get_or_create_site(conn, "Company A", "Bragancia", "Main St.", 14.0, 121.0)
    assert id1 != id_diff
    conn.close()

def test_two_companies_same_site_ranking(client_admin):
    conn = get_connection()
    # Create site
    site_id = get_or_create_site(conn, "Comp A", "Brgy. X", "Y St")
    # Insert 2 complete analyses for this site
    conn.execute("INSERT INTO analyses (site_id, company, barangay, street, sector, filename, status, risk_score) VALUES (?, 'Comp A', 'Brgy. X', 'Y St', '', 'test1.jpg', 'complete', 40.0)", (site_id,))
    conn.execute("INSERT INTO analyses (site_id, company, barangay, street, sector, filename, status, risk_score) VALUES (?, 'Comp B', 'Brgy. X', 'Y St', '', 'test2.jpg', 'complete', 60.0)", (site_id,))
    conn.commit()
    conn.close()
    
    # Check ranking
    res = client_admin.get("/api/mpc/ranking")
    data = res.json()["rankings"]
    sites = [d for d in data if d["barangay"].lower() == "brgy. x".lower() and d["street"].lower() == "y st".lower()]
    assert len(sites) == 1, f"Expected 1 site grouped, got {len(sites)}"
    assert sites[0]["analysis_count"] == 2, "Expected 2 analyses for the site"
    assert sites[0]["avg_risk"] == 50.0, "Expected avg risk to be 50.0"


def test_class_weights_scaling(monkeypatch):
    from backend.config_store import set_setting
    # artificially set power_line to 1.0, telecom_cable to 0.5
    set_setting("class_weights", {
        "power_line": 1.0,
        "telecom_cable": 0.5
    })
    
    # We can invoke execute_pipeline directly or test via upload
    # The easiest is testing the math in pipeline directly or via a mock
    # Wait, the easiest is to just mock the segmentation result and run analyze_image!
    from backend.utils.analysis_engine import analyze_image
    
    class DummyPrep:
        quality_ok = True
        gray = None
        rgb = None
        original_shape = (1000, 1000)
        
    class DummyVNI:
        vni = 1.0
        
    class DummySeg:
        mask_available = True
        masks = [None, None]
        classes = [0, 1]
        names = {0: "power_line", 1: "telecom_cable"}
        confidences = [0.9, 0.9]
        boxes = [[0,0,10,10], [0,0,10,10]]
        
    class DummyCenterline:
        is_valid = True
        polyline = [(0, 500), (10, 500)]
        
    class DummyCat:
        sag_to_span = 0.1
        sag_px = 100
        span_px = 1000
        fitted_a = 0.01
        residual = 1.0
        sag_ratio = 0.15 # This will produce a CRS of roughly 70+
        
    def mock_val(*args, **kwargs): return DummyPrep()
    def mock_vni(*args, **kwargs): return DummyVNI()
    def mock_seg(*args, **kwargs): return DummySeg()
    def mock_extract(*args, **kwargs): return DummyCenterline()
    def mock_cat(*args, **kwargs): return DummyCat()
    
    monkeypatch.setattr("backend.utils.analysis_engine.validate_and_preprocess", mock_val)
    monkeypatch.setattr("backend.utils.analysis_engine.compute_vni", mock_vni)
    monkeypatch.setattr("backend.utils.analysis_engine.segment_image", mock_seg)
    monkeypatch.setattr("backend.utils.analysis_engine.extract_centerline", mock_extract)
    monkeypatch.setattr("backend.utils.analysis_engine.calculate_sag_metrics", mock_cat)
    
    res = analyze_image("dummy.jpg", None, db_conn=None)
    
    wires = res["wires"]
    # We should have 2 wires. They have the identical geometry, but one is power, one is telecom
    w_power = next(w for w in wires if w["class_name"] == "power_line")
    w_telecom = next(w for w in wires if w["class_name"] == "telecom_cable")
    
    assert w_power["crs"] > 0
    # Telecom should be exactly half of Power because weight is 0.5 vs 1.0
    assert abs(w_telecom["crs"] - (w_power["crs"] * 0.5)) < 0.001
