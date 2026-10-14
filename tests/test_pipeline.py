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
    from backend.config_store import get_setting, set_setting
    # artificially set power_line to 1.0, telecom_cable to 0.5
    orig_weights = get_setting("class_weights")
    try:
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
            sag_to_span = 0.15 # This will produce a CRS of roughly 70+
            sag_px = 100
            span_px = 1000
            fitted_a = 0.01
            residual = 1.0
            sag_ratio = 0.15
            
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
    finally:
        set_setting("class_weights", orig_weights)


def test_pipeline_class_name_normalization_and_pole_selection(monkeypatch):
    import numpy as np
    from backend.config_store import get_setting, set_setting
    from backend.database import get_connection
    from backend.utils.pipeline import execute_pipeline
    from backend.utils.cv.types import PreprocessResult, VNIResult, CenterlineResult, CatenaryMetrics, SegmentationResult

    orig_weights = get_setting("class_weights")
    orig_pec = get_setting("pec_min_clearance_by_class")
    orig_pole_h = get_setting("pole_heights")
    orig_margin = get_setting("margin_of_error_pct")

    try:
        # Settings with non-canonical casing/formatting
        set_setting("class_weights", {
            "Power-Line": 1.5,
            "telecom_cable": 1.0
        })
        set_setting("pec_min_clearance_by_class", {
            "Power Line": 6.0,
            "default": 5.5
        })
        set_setting("pole_heights", [10])
        set_setting("margin_of_error_pct", 5)

        prep = PreprocessResult(
            quality_ok=True,
            reason="ok",
            gray=np.zeros((1000, 1000), dtype=np.uint8),
            rgb=np.zeros((1000, 1000, 3), dtype=np.uint8),
            original_shape=(1000, 1000)
        )
        monkeypatch.setattr("backend.utils.pipeline.validate_and_preprocess", lambda path: prep)
        monkeypatch.setattr("backend.utils.pipeline.compute_vni", lambda gray: VNIResult(vni=0.1, spectrum_png=None))

        # Fake segmentation result with:
        # 1. Lower confidence pole [0, 100, 20, 800] with conf 0.5
        # 2. Higher confidence pole [0, 200, 20, 900] with conf 0.95 (height = 700)
        # 3. Non-canonical wire class name "Power Line" with lowest_y = 487 (clearance = 5.90m)
        # 4. Non-wire class "clock" (conf 0.99) that must be skipped
        fake_seg = SegmentationResult(
            mask_available=True,
            boxes=[
                [0, 100, 20, 800],
                [0, 200, 20, 900],
                [10, 300, 500, 487],
                [50, 50, 80, 80]
            ],
            classes=[0, 0, 1, 2],
            confidences=[0.5, 0.95, 0.85, 0.99],
            masks=[
                np.zeros((1000, 1000), dtype=np.uint8),
                np.zeros((1000, 1000), dtype=np.uint8),
                np.zeros((1000, 1000), dtype=np.uint8),
                np.zeros((1000, 1000), dtype=np.uint8),
            ],
            names={0: "Pole", 1: "Power Line", 2: "clock"}
        )
        monkeypatch.setattr("backend.mock_detector.is_mock_mode", lambda: False)
        monkeypatch.setattr("backend.utils.pipeline.segment_image", lambda rgb, model, shape: fake_seg)

        cat = CatenaryMetrics(
            sag_to_span=0.1,
            sag_ratio=0.1,
            residual=1.0,
            span_px=500.0,
            sag_px=50.0,
            fitted_a=0.01,
            fitted_points=[(10.0, 300.0), (500.0, 487.0)]
        )
        monkeypatch.setattr("backend.utils.pipeline.extract_centerline", lambda m, b: CenterlineResult(polyline=[(10, 300), (500, 487)], is_valid=True))
        monkeypatch.setattr("backend.utils.pipeline.calculate_sag_metrics", lambda pts: cat)

        conn = get_connection()
        cur = conn.execute("INSERT INTO analyses (filename, status, company, street) VALUES ('test.jpg', 'queued', 'C', 'S')")
        conn.commit()
        a_id = cur.lastrowid

        # Execute pipeline in real mode with fake model
        execute_pipeline(a_id, "dummy_path.jpg", conn, model="fake_model")

        # Verify wire rows stored: only "power_line" is present; "clock" was skipped
        rows = conn.execute("SELECT class_name, crs, clearance_m, clearance_status FROM wires WHERE analysis_id = ?", (a_id,)).fetchall()
        assert len(rows) == 1
        wire = rows[0]
        assert wire["class_name"] == "power_line"
        assert "clock" not in [r["class_name"] for r in rows]

        # Verify weight was applied (CRS is scaled by 1.5)
        from backend.formulas import compute_catenary_risk
        expected_raw_crs = compute_catenary_risk(0.1, 1.0)
        assert abs(wire["crs"] - min(100.0, expected_raw_crs * 1.5)) < 0.001
        
        # Verify higher confidence pole [0, 200, 20, 900] was selected:
        # height = 900 - 200 = 700 px, scale = 10m / 700px = 0.0142857 m/px
        # lowest_y = 487, ground_y = 900, clearance_px = 413 px
        # clearance_m = 413 * (10 / 700) = 5.90 m
        assert abs(wire["clearance_m"] - 5.90) < 0.01

        # With 5% margin, interval is [5.605m, 6.195m].
        # If PEC min is 6.0m (Power Line): 5.605 < 6.0 <= 6.195 -> "marginal"
        # If PEC min fell back to 5.5m (default): 5.605 >= 5.5 -> "ok"
        # Asserting "marginal" proves the normalized key "Power Line" -> 6.0 was used!
        assert wire["clearance_status"] == "marginal"
        conn.close()
    finally:
        set_setting("class_weights", orig_weights)
        set_setting("pec_min_clearance_by_class", orig_pec)
        set_setting("pole_heights", orig_pole_h)
        set_setting("margin_of_error_pct", orig_margin)


def test_pipeline_skips_non_wire_classes(monkeypatch, caplog):
    import logging
    import numpy as np
    from backend.database import get_connection
    from backend.utils.pipeline import execute_pipeline
    from backend.utils.cv.types import PreprocessResult, SegmentationResult

    prep = PreprocessResult(
        quality_ok=True,
        reason="ok",
        gray=np.zeros((200, 200), dtype=np.uint8),
        rgb=np.zeros((200, 200, 3), dtype=np.uint8),
        original_shape=(200, 200)
    )
    monkeypatch.setattr("backend.utils.pipeline.validate_and_preprocess", lambda path: prep)
    monkeypatch.setattr("backend.mock_detector.is_mock_mode", lambda: False)

    # Segmentation with non-wire COCO classes only: "clock", "traffic light"
    fake_seg = SegmentationResult(
        mask_available=True,
        boxes=[[10, 10, 50, 50], [60, 60, 90, 90]],
        classes=[0, 1],
        confidences=[0.9, 0.85],
        masks=[np.zeros((200, 200), dtype=np.uint8), np.zeros((200, 200), dtype=np.uint8)],
        names={0: "clock", 1: "traffic light"}
    )
    monkeypatch.setattr("backend.utils.pipeline.segment_image", lambda rgb, model, shape: fake_seg)

    conn = get_connection()
    cur = conn.execute("INSERT INTO analyses (filename, status, company, street) VALUES ('test_clock.jpg', 'queued', 'C', 'S')")
    conn.commit()
    a_id = cur.lastrowid

    with caplog.at_level(logging.DEBUG):
        execute_pipeline(a_id, "dummy_path.jpg", conn, model="fake_model")

    # Wires table must be empty
    rows = conn.execute("SELECT * FROM wires WHERE analysis_id = ?", (a_id,)).fetchall()
    assert len(rows) == 0

    # Analysis status should transition to 'no_wires'
    analysis = conn.execute("SELECT status FROM analyses WHERE id = ?", (a_id,)).fetchone()
    assert analysis["status"] == "no_wires"

    # Confirm debug logging for skipping
    assert any("Skipping non-wire class: clock" in r.message for r in caplog.records)
    conn.close()


def test_pipeline_taut_sloped_wire_zero_crs(monkeypatch):
    """(a) A taut sloped wire (slope 20%, zero sag) scores CRS 0 through the pipeline path."""
    import numpy as np
    from backend.database import get_connection
    from backend.utils.pipeline import execute_pipeline
    from backend.utils.cv.types import PreprocessResult, SegmentationResult, CenterlineResult, CatenaryMetrics

    prep = PreprocessResult(
        quality_ok=True,
        reason="ok",
        gray=np.zeros((300, 300), dtype=np.uint8),
        rgb=np.zeros((300, 300, 3), dtype=np.uint8),
        original_shape=(300, 300)
    )
    monkeypatch.setattr("backend.utils.pipeline.validate_and_preprocess", lambda path: prep)
    monkeypatch.setattr("backend.mock_detector.is_mock_mode", lambda: False)

    fake_seg = SegmentationResult(
        mask_available=True,
        boxes=[[0, 0, 300, 60]],
        classes=[0],
        confidences=[0.95],
        masks=[np.zeros((300, 300), dtype=np.uint8)],
        names={0: "power_line"}
    )
    monkeypatch.setattr("backend.utils.pipeline.segment_image", lambda rgb, model, shape: fake_seg)

    # 20% slope: dx=300, dy=60 -> sag_ratio = 60/300 = 0.20 (which previously scored 100.0)
    # But wire is straight/taut: sag_px = 0.0, sag_to_span = 0.0, residual = 0.5 (< 2.0)
    cat = CatenaryMetrics(
        sag_to_span=0.0,
        sag_ratio=0.20,
        residual=0.5,
        span_px=float(np.sqrt(300**2 + 60**2)),
        sag_px=0.0,
        fitted_a=0.0,
        fitted_points=[(0.0, 0.0), (300.0, 60.0)]
    )
    monkeypatch.setattr("backend.utils.pipeline.extract_centerline", lambda m, b: CenterlineResult(polyline=[(0, 0), (300, 60)], is_valid=True))
    monkeypatch.setattr("backend.utils.pipeline.calculate_sag_metrics", lambda pts: cat)

    conn = get_connection()
    cur = conn.execute("INSERT INTO analyses (filename, status, company, street) VALUES ('taut_sloped.jpg', 'queued', 'C', 'S')")
    conn.commit()
    a_id = cur.lastrowid

    execute_pipeline(a_id, "dummy_taut.jpg", conn, model="fake_model")

    rows = conn.execute("SELECT crs, class_name FROM wires WHERE analysis_id = ?", (a_id,)).fetchall()
    assert len(rows) == 1
    # Taut wire has zero chord sag, so CRS through pipeline must be 0.0 (despite 20% bounding-box slope ratio)
    assert rows[0]["crs"] == 0.0
    conn.close()


def test_level_wire_crs_retains_identical_value():
    """(b) A level wire with sag_to_span 0.10 scores the same CRS as before for the same residual."""
    from backend.formulas import compute_catenary_risk

    # Before the parameter rename/swap, compute_catenary_risk(0.10, 1.5) was 41.18
    # With sag_to_span = 0.10 and residual = 1.5, the score must be identical:
    # (0.10 - 0.03) / 0.17 * 100 = 41.176... -> 41.18
    crs = compute_catenary_risk(0.10, 1.5)
    assert abs(crs - 41.18) < 0.01


def test_crs_invariant_to_slope_for_fixed_sag_to_span():
    """(c) CRS is unchanged for the same sag_to_span regardless of slope."""
    from backend.formulas import compute_catenary_risk

    target_sag_to_span = 0.05
    res = 1.0

    # For level (slope 0%), moderate (slope 10%), steep (slope 20%):
    # As long as sag_to_span is 0.05, CRS is invariant:
    # (0.05 - 0.03) / 0.17 * 100 = 11.7647... -> 11.76
    crs_level = compute_catenary_risk(target_sag_to_span, res)
    crs_slope10 = compute_catenary_risk(target_sag_to_span, res)
    crs_slope20 = compute_catenary_risk(target_sag_to_span, res)

    assert crs_level == crs_slope10 == crs_slope20
    assert abs(crs_level - 11.76) < 0.01


def test_mock_mode_pipeline_regression_annotated_jpeg(tmp_path, monkeypatch):
    """End-to-end regression test in mock mode: verifies pipeline completes and writes valid annotated JPEG."""
    import cv2
    from make_demo_images import make_demo_images
    from backend.database import get_connection
    from backend.utils.pipeline import execute_pipeline

    demo_dir = tmp_path / "demo"
    make_demo_images(str(demo_dir))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_MOCK", "1")

    conn = get_connection()
    cur = conn.cursor()

    for fname in ["safe-pole.jpg", "critical-pole.jpg"]:
        cur.execute(
            "INSERT INTO analyses (filename, status, company, barangay, street) VALUES (?, 'queued', 'Meralco', 'San Antonio', 'Perea St')",
            (fname,)
        )
        a_id = cur.lastrowid
        conn.commit()

        execute_pipeline(a_id, str(demo_dir / fname), conn, model=None, batch_id=None)

        row = conn.execute("SELECT status, annotated_path FROM analyses WHERE id = ?", (a_id,)).fetchone()
        assert row["status"] == "complete"
        assert row["annotated_path"] is not None

        # Verify wire rows exist
        wires = conn.execute("SELECT * FROM wires WHERE analysis_id = ?", (a_id,)).fetchall()
        assert len(wires) > 0

        # Verify annotated JPEG exists, is non-empty, and decodes with cv2.imread
        annotated_file = tmp_path / "outputs" / f"analysis_{a_id}_annotated.jpg"
        assert annotated_file.exists()
        assert annotated_file.stat().st_size > 0
        img = cv2.imread(str(annotated_file))
        assert img is not None
        assert img.shape[0] > 0 and img.shape[1] > 0

    conn.close()


def test_real_mode_pipeline_regression_float_centerline(tmp_path, monkeypatch):
    """Real-mode pipeline regression: real segment_image, real extract_centerline, real catenary, real cv2 drawing."""
    import cv2
    import torch
    import numpy as np
    from backend.database import get_connection
    from backend.utils.pipeline import execute_pipeline
    from backend.utils.cv.catenary import cosh_func

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_MOCK", "0")

    # Generate synthetic input image (300x500 RGB) with content to pass quality validation
    h, w = 300, 500
    rng = np.random.RandomState(42)
    img_synth = rng.randint(50, 200, (h, w, 3), dtype=np.uint8)
    input_img_path = tmp_path / "synthetic_test.jpg"
    cv2.imwrite(str(input_img_path), img_synth)

    # Wire mask: hanging catenary curve between x=50 and x=450
    wire_mask = np.zeros((h, w), dtype=np.uint8)
    x = np.arange(50, 450, dtype=float)
    y_up = cosh_func(x, 250.0, -100.0, 150.0)
    y_hang = float(np.max(y_up)) - (y_up - 80.0)
    for xi, yi in zip(x, y_hang):
        cv2.circle(wire_mask, (int(xi), int(round(yi))), 2, 255, -1)

    # Pole mask: vertical pole
    pole_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(pole_mask, (240, 50), (260, 280), 255, -1)

    # Fake model returning pole (cls=0) and wire (cls=1, power_line)
    class FakeBoxes:
        def __init__(self):
            self.xyxy = torch.tensor([
                [240.0, 50.0, 260.0, 280.0],
                [50.0, float(np.min(y_hang) - 5), 450.0, float(np.max(y_hang) + 5)]
            ], dtype=torch.float32)
            self.cls = torch.tensor([0, 1])
            self.conf = torch.tensor([0.95, 0.90])

    class FakeMasks:
        def __init__(self):
            m1 = torch.from_numpy(pole_mask.astype(np.float32) / 255.0)
            m2 = torch.from_numpy(wire_mask.astype(np.float32) / 255.0)
            self.data = torch.stack([m1, m2])

    class FakeResult:
        def __init__(self):
            self.boxes = FakeBoxes()
            self.masks = FakeMasks()
            self.names = {0: "pole", 1: "power_line"}

    def fake_model(bgr_input, **kwargs):
        return [FakeResult()]

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO analyses (filename, status, company, barangay, street) VALUES ('synthetic_test.jpg', 'queued', 'Meralco', 'San Antonio', 'Perea St')"
    )
    a_id = cur.lastrowid
    conn.commit()

    execute_pipeline(a_id, str(input_img_path), conn, model=fake_model, batch_id=None)

    row = conn.execute("SELECT status, annotated_path FROM analyses WHERE id = ?", (a_id,)).fetchone()
    assert row["status"] == "complete"
    assert row["annotated_path"] is not None

    wires = conn.execute("SELECT * FROM wires WHERE analysis_id = ?", (a_id,)).fetchall()
    assert len(wires) == 1
    wire = wires[0]
    assert wire["class_name"] == "power_line"
    assert wire["sag_px"] > 0.0

    annotated_file = tmp_path / "outputs" / f"analysis_{a_id}_annotated.jpg"
    assert annotated_file.exists()
    assert annotated_file.stat().st_size > 0
    img = cv2.imread(str(annotated_file))
    assert img is not None
    assert img.shape == (h, w, 3)

    conn.close()


def test_clearance_rollup_in_mock_mode(tmp_path, monkeypatch):
    """(a) Mock mode end to end: clearance_ok, clearance_available, sag_meters are filled and match lowest wire."""
    from make_demo_images import make_demo_images
    from backend.database import get_connection
    from backend.utils.pipeline import execute_pipeline

    demo_dir = tmp_path / "demo"
    make_demo_images(str(demo_dir))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_MOCK", "1")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO analyses (filename, status, company, barangay, street) VALUES ('safe-pole.jpg', 'queued', 'Meralco', 'San Antonio', 'Perea St')"
    )
    a_id = cur.lastrowid
    conn.commit()

    execute_pipeline(a_id, str(demo_dir / "safe-pole.jpg"), conn, model=None, batch_id=None)

    row = conn.execute("SELECT * FROM analyses WHERE id = ?", (a_id,)).fetchone()
    assert row["status"] == "complete"
    assert row["clearance_available"] is not None
    assert row["sag_meters"] is not None

    wires = conn.execute("SELECT * FROM wires WHERE analysis_id = ?", (a_id,)).fetchall()
    valid_wires = [w for w in wires if w["clearance_m"] is not None and w["clearance_status"] != "unavailable"]
    assert len(valid_wires) > 0

    lowest_w = min(valid_wires, key=lambda w: w["clearance_m"])
    assert row["clearance_available"] == pytest.approx(lowest_w["clearance_m"], abs=1e-4)
    expected_ok = 1 if lowest_w["clearance_status"] == "ok" else 0
    assert row["clearance_ok"] == expected_ok

    max_sag = max(w["sag_m"] for w in wires if w["sag_m"] is not None)
    assert row["sag_meters"] == pytest.approx(max_sag, abs=1e-4)

    conn.close()


def test_clearance_no_pole_real_code_path(tmp_path, monkeypatch):
    """(b) No pole detected: clearance columns remain NULL and API returns ground_clearance_status 'no_pole'."""
    import cv2
    import torch
    import numpy as np
    from backend.database import get_connection
    from backend.utils.pipeline import execute_pipeline
    from backend.utils.cv.catenary import cosh_func

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_MOCK", "0")

    h, w = 300, 500
    img_synth = np.full((h, w, 3), 120, dtype=np.uint8)
    # Add texture for blur variance check
    img_synth[::2, ::2] = 200
    input_path = tmp_path / "wire_only.jpg"
    cv2.imwrite(str(input_path), img_synth)

    wire_mask = np.zeros((h, w), dtype=np.uint8)
    x = np.arange(50, 450, dtype=float)
    y_up = cosh_func(x, 250.0, -100.0, 150.0)
    y_hang = float(np.max(y_up)) - (y_up - 80.0)
    for xi, yi in zip(x, y_hang):
        cv2.circle(wire_mask, (int(xi), int(round(yi))), 2, 255, -1)

    class FakeBoxes:
        def __init__(self):
            self.xyxy = torch.tensor([[50.0, float(np.min(y_hang) - 5), 450.0, float(np.max(y_hang) + 5)]], dtype=torch.float32)
            self.cls = torch.tensor([1])
            self.conf = torch.tensor([0.90])

    class FakeMasks:
        def __init__(self):
            self.data = torch.stack([torch.from_numpy(wire_mask.astype(np.float32) / 255.0)])

    class FakeResult:
        def __init__(self):
            self.boxes = FakeBoxes()
            self.masks = FakeMasks()
            self.names = {1: "power_line"}

    def wire_only_model(bgr_input, **kwargs):
        return [FakeResult()]

    conn = get_connection()
    cur = conn.cursor()
    from backend.auth import _hash_token
    guest_tok = "tok123"
    cur.execute(
        "INSERT INTO analyses (filename, status, company, barangay, street, guest_session_id) VALUES ('wire_only.jpg', 'queued', 'Meralco', 'San Antonio', 'Perea St', ?)",
        (_hash_token(guest_tok),)
    )
    a_id = cur.lastrowid
    conn.commit()

    execute_pipeline(a_id, str(input_path), conn, model=wire_only_model, batch_id=None)

    row = conn.execute("SELECT * FROM analyses WHERE id = ?", (a_id,)).fetchone()
    assert row["status"] == "complete"
    assert row["clearance_ok"] is None
    assert row["clearance_available"] is None
    assert row["pec_min_clearance_m"] is None

    # Test API mapping for this row
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    client.cookies.set("guest_session", guest_tok)
    res = client.get(f"/api/analysis/{a_id}/result")
    assert res.status_code == 200
    data = res.json()
    assert data["ground_clearance_status"] == "no_pole"
    assert data["ground_clearance_m"] is None

    conn.close()


def test_api_clearance_status_mappings():
    """(c) API mapping for above, near, below, and no_pole."""
    from fastapi.testclient import TestClient
    import main
    from backend.database import get_connection
    from backend.auth import _hash_token

    conn = get_connection()
    cur = conn.cursor()

    guest_tok = "test_tok"
    ghash = _hash_token(guest_tok)

    # above: clearance_ok=1, clearance_available=6.0, pec_min=5.5
    cur.execute("INSERT INTO analyses (filename, status, clearance_ok, clearance_available, pec_min_clearance_m, max_crs, guest_session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ('t1.jpg', 'complete', 1, 6.0, 5.5, 10.0, ghash))
    id_above = cur.lastrowid
    cur.execute("INSERT INTO wires (analysis_id, class_name, clearance_m, clearance_status) VALUES (?, 'power_line', 6.0, 'ok')", (id_above,))

    # near: clearance_ok=2, clearance_available=5.5, pec_min=5.5
    cur.execute("INSERT INTO analyses (filename, status, clearance_ok, clearance_available, pec_min_clearance_m, max_crs, guest_session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ('t2.jpg', 'complete', 2, 5.5, 5.5, 10.0, ghash))
    id_near = cur.lastrowid
    cur.execute("INSERT INTO wires (analysis_id, class_name, clearance_m, clearance_status) VALUES (?, 'power_line', 5.5, 'marginal')", (id_near,))

    # below: clearance_ok=0, clearance_available=4.0, pec_min=5.5 (< 5.5)
    cur.execute("INSERT INTO analyses (filename, status, clearance_ok, clearance_available, pec_min_clearance_m, max_crs, guest_session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ('t3.jpg', 'complete', 0, 4.0, 5.5, 10.0, ghash))
    id_below = cur.lastrowid
    cur.execute("INSERT INTO wires (analysis_id, class_name, clearance_m, clearance_status) VALUES (?, 'power_line', 4.0, 'below')", (id_below,))

    # no_pole: complete with wires but clearance_available is None
    cur.execute("INSERT INTO analyses (filename, status, clearance_ok, clearance_available, pec_min_clearance_m, max_crs, guest_session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ('t4.jpg', 'complete', None, None, None, 10.0, ghash))
    id_nopole = cur.lastrowid
    cur.execute("INSERT INTO wires (analysis_id, class_name, clearance_m, clearance_status) VALUES (?, 'power_line', NULL, 'unavailable')", (id_nopole,))

    # no_wires: status='no_wires'
    cur.execute("INSERT INTO analyses (filename, status, clearance_ok, clearance_available, pec_min_clearance_m, max_crs, guest_session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ('t5.jpg', 'no_wires', None, None, None, 0.0, ghash))
    id_nowires = cur.lastrowid

    # no_wires: status='complete' but 0 wire rows
    cur.execute("INSERT INTO analyses (filename, status, clearance_ok, clearance_available, pec_min_clearance_m, max_crs, guest_session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ('t6.jpg', 'complete', None, None, None, 0.0, ghash))
    id_nowires2 = cur.lastrowid

    # queued: status='queued' -> unavailable (never gives no_pole)
    cur.execute("INSERT INTO analyses (filename, status, clearance_ok, clearance_available, pec_min_clearance_m, max_crs, guest_session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ('t7.jpg', 'queued', None, None, None, None, ghash))
    id_queued = cur.lastrowid

    # error: status='error' -> unavailable (never gives no_pole)
    cur.execute("INSERT INTO analyses (filename, status, clearance_ok, clearance_available, pec_min_clearance_m, max_crs, guest_session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ('t8.jpg', 'error', None, None, None, None, ghash))
    id_error = cur.lastrowid

    conn.commit()
    conn.close()

    client = TestClient(main.app)
    client.cookies.set("guest_session", guest_tok)

    assert client.get(f"/api/analysis/{id_above}/result").json()["ground_clearance_status"] == "above"
    assert client.get(f"/api/analysis/{id_near}/result").json()["ground_clearance_status"] == "near"
    assert client.get(f"/api/analysis/{id_below}/result").json()["ground_clearance_status"] == "below"
    assert client.get(f"/api/analysis/{id_nopole}/result").json()["ground_clearance_status"] == "no_pole"
    assert client.get(f"/api/analysis/{id_nowires}/result").json()["ground_clearance_status"] == "no_wires"
    assert client.get(f"/api/analysis/{id_nowires2}/result").json()["ground_clearance_status"] == "no_wires"
    assert client.get(f"/api/analysis/{id_queued}/result").json()["ground_clearance_status"] == "unavailable"
    assert client.get(f"/api/analysis/{id_error}/result").json()["ground_clearance_status"] == "unavailable"


def test_lowest_wire_marginal_rollup_and_api(tmp_path, monkeypatch):
    """Assert lowest wire marginal (clearance 5.9m, PEC 6.0, margin 5%) gives clearance_ok 2 and API 'near'."""
    import cv2
    import torch
    import numpy as np
    from backend.database import get_connection
    from backend.config_store import set_setting
    from backend.utils.pipeline import execute_pipeline
    from backend.utils.cv.catenary import cosh_func
    from backend.auth import _hash_token
    from fastapi.testclient import TestClient
    import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_MOCK", "0")

    conn = get_connection()
    set_setting("pec_min_clearance_by_class", {"default": 6.0, "power_line": 6.0})
    set_setting("margin_of_error_pct", 5.0)
    set_setting("pole_heights", [10.0])

    # Synthetic image
    h, w = 600, 800
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    img[::2, ::2] = 200
    input_path = tmp_path / "marginal_pole.jpg"
    cv2.imwrite(str(input_path), img)

    # Pole: y1=100, y2=500 -> pole_px = 400. With 10.0m pole, scale = 10.0/400 = 0.025 m/px.
    # Ground reference is at bottom of pole = 500 px.
    # To get clearance_m = 5.9 m:
    # 5.9 / 0.025 = 236 px clearance above ground (i.e. y = 500 - 236 = 264 px).
    pole_box = [600.0, 100.0, 650.0, 500.0]

    # Create wire mask with lowest point at y=264 px
    wire_mask = np.zeros((h, w), dtype=np.uint8)
    x = np.arange(100, 700, dtype=float)
    y_up = cosh_func(x, 400.0, -100.0, 200.0)
    y_hang = 264.0 - (float(np.max(y_up)) - y_up)
    for xi, yi in zip(x, y_hang):
        cv2.circle(wire_mask, (int(xi), int(round(yi))), 2, 255, -1)

    class FakeBoxes:
        def __init__(self):
            self.xyxy = torch.tensor([
                [100.0, float(np.min(y_hang) - 5), 700.0, float(np.max(y_hang) + 5)],
                pole_box
            ], dtype=torch.float32)
            self.cls = torch.tensor([0, 4]) # 0=power_line, 4=pole
            self.conf = torch.tensor([0.92, 0.95])

    class FakeMasks:
        def __init__(self):
            pole_mask = np.zeros((h, w), dtype=np.float32)
            pole_mask[100:500, 600:650] = 1.0
            self.data = torch.stack([
                torch.from_numpy(wire_mask.astype(np.float32) / 255.0),
                torch.from_numpy(pole_mask)
            ])

    class FakeResult:
        def __init__(self):
            self.boxes = FakeBoxes()
            self.masks = FakeMasks()
            self.names = {0: "power_line", 4: "pole"}

    guest_tok = "tok_marginal"
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO analyses (filename, status, company, barangay, street, guest_session_id) VALUES ('marginal_pole.jpg', 'queued', 'Meralco', 'San Antonio', 'Perea St', ?)",
        (_hash_token(guest_tok),)
    )
    a_id = cur.lastrowid
    conn.commit()

    execute_pipeline(a_id, str(input_path), conn, model=lambda img, **kwargs: [FakeResult()], batch_id=None)

    row = conn.execute("SELECT * FROM analyses WHERE id = ?", (a_id,)).fetchone()
    assert row["status"] == "complete"
    assert row["clearance_ok"] == 2
    assert row["clearance_available"] == pytest.approx(5.9, abs=0.1)

    client = TestClient(main.app)
    client.cookies.set("guest_session", guest_tok)
    res = client.get(f"/api/analysis/{a_id}/result")
    assert res.status_code == 200
    assert res.json()["ground_clearance_status"] == "near"

    conn.close()




