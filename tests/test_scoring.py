from backend import config_store
from backend.formulas import (
    compute_clearance_status,
    compute_crs_band,
    compute_safety_level,
)
from backend.utils.cv.calibration import calibrate_scale, estimate_clearance


def setup_module(module):
    # For testing without DB, we can just patch get_setting
    settings = config_store.DEFAULT_SETTINGS.copy()
    mapping = settings["safety_mapping"].copy()
    # Simulate JSON load where keys become strings
    mapping["rank_to_level"] = {str(k): v for k, v in mapping["rank_to_level"].items()}
    settings["safety_mapping"] = mapping
    config_store._cache = settings
    

def test_crs_bands():
    assert compute_crs_band(-1) == "unavailable"
    assert compute_crs_band(None) == "unavailable"
    
    # 33.4 -> 33.40 -> 0.334 => Moderate
    assert compute_crs_band(0.0) == "Low"
    assert compute_crs_band(33.0) == "Low"
    assert compute_crs_band(33.1) == "Moderate"
    assert compute_crs_band(33.4) == "Moderate"
    # 33.0 -> 33.00 -> 0.33 => Low
    assert compute_crs_band(33.0) == "Low"
    
    assert compute_crs_band(66.0) == "Moderate"
    assert compute_crs_band(66.1) == "High"
    assert compute_crs_band(67.0) == "High"
    
    assert compute_crs_band(85.0) == "High"
    assert compute_crs_band(85.1) == "Critical"
    assert compute_crs_band(86.0) == "Critical"
    assert compute_crs_band(100.0) == "Critical"
    

def test_clearance_status():
    assert compute_clearance_status(None, None, None, 4.5) == "unavailable"
    
    # Lower >= min -> ok
    assert compute_clearance_status(5.0, 4.5, 5.5, 4.5) == "ok"
    
    # Upper < min -> below
    assert compute_clearance_status(4.0, 3.5, 4.4, 4.5) == "below"
    
    # Straddling -> marginal
    assert compute_clearance_status(4.5, 4.0, 5.0, 4.5) == "marginal"


def test_safety_level():
    # Both unavailable
    res = compute_safety_level("unavailable", "unavailable", "Clearance unavailable")
    assert res["safety_level"] == "Unknown"
    
    # Clearance unavailable, but CRS is High -> Unsafe
    res = compute_safety_level("High", "unavailable", "Clearance unavailable")
    assert res["safety_level"] == "Unsafe"
    assert "based on CRS only" in res["reasons"][0]
    
    # CRS unavailable, but Clearance is below -> Unsafe
    res = compute_safety_level("unavailable", "below", "Est 3.0m")
    assert res["safety_level"] == "Unsafe"
    assert "based on clearance only" in res["reasons"][0]
    
    # Clearance ok, CRS Low -> Safe
    res = compute_safety_level("Low", "ok", "Est 5.0m")
    assert res["safety_level"] == "Safe"
    
    # Clearance marginal, CRS Low -> Caution
    res = compute_safety_level("Low", "marginal", "Est 4.5m")
    assert res["safety_level"] == "Caution"
    
    # Clearance below (Unsafe), CRS Moderate (Caution) -> Unsafe
    res = compute_safety_level("Moderate", "below", "Est 3.0m")
    assert res["safety_level"] == "Unsafe"
    
    # Clearance ok (Safe), CRS Critical -> Critical
    res = compute_safety_level("Critical", "ok", "Est 5.0m")
    assert res["safety_level"] == "Critical"


def test_calibration_priorities():
    # 1. Manual over pole
    res = calibrate_scale(
        manual_pt1=(0, 0), manual_pt2=(100, 0), manual_length_m=10.0,
        pole_bbox=[0, 0, 10, 50], image_height=1000
    )
    assert res.source == "manual"
    assert res.meters_per_px == 0.1
    
    # 2. Pole fallback
    res = calibrate_scale(
        pole_bbox=[0, 0, 10, 90], image_height=1000
    )
    assert res.source == "pole"
    # default assumed pole is 9m / 90px = 0.1
    assert res.meters_per_px == 0.1
    
    # 3. Pole touches bottom edge -> unavailable
    res = calibrate_scale(
        pole_bbox=[0, 0, 10, 996], image_height=1000
    )
    assert res.source == "unavailable"
    assert res.meters_per_px is None


def test_estimate_clearance():
    # Wire at y=500, ground at y=900, scale=0.1
    res = estimate_clearance(500, [0, 0, 10, 900], 0.1, 1000)
    assert res["status"] == "estimated"
    assert res["clearance_m"] == 40.0 # 400 * 0.1
    # 20% margin -> 32.0 to 48.0
    assert res["clearance_low_m"] == 32.0
    assert res["clearance_high_m"] == 48.0
    
    # Ground not fully visible
    res = estimate_clearance(500, [0, 0, 10, 996], 0.1, 1000)
    assert res["status"] == "unavailable"


def test_crs_band_dynamic(monkeypatch):
    from backend.config_store import set_setting
    # Change the bands dynamically to prove it uses config_store
    new_bands = [
        {"min": 0.0, "max": 0.10, "label": "Tiny"},
        {"min": 0.11, "max": 0.90, "label": "Huge"},
        {"min": 0.91, "max": 1.0, "label": "Maxed"}
    ]
    # Set setting sets the cache
    set_setting("risk_bands", new_bands)
    
    # 5.0 -> 0.05
    assert compute_crs_band(5.0) == "Tiny"
    # 50.0 -> 0.50
    assert compute_crs_band(50.0) == "Huge"
    # 95.0 -> 0.95
    assert compute_crs_band(95.0) == "Maxed"
    set_setting("risk_bands", config_store.DEFAULT_SETTINGS["risk_bands"])


def test_analysis_result_uses_risk_score_not_max_crs():
    from fastapi.testclient import TestClient

    from backend import database
    from backend.auth import _hash_token
    from main import app

    config_store.set_setting("risk_bands", config_store.DEFAULT_SETTINGS["risk_bands"])

    guest_token = "test_guest_session_band_consistency"
    guest_hash = _hash_token(guest_token)

    conn = database.get_connection()
    cur = conn.execute(
        """INSERT INTO analyses (filename, status, risk_score, max_crs, clearance_available, clearance_ok, is_public, is_mock, guest_session_id)
           VALUES ('mock_wire.jpg', 'complete', 44.6, 100.0, 4.5, 2, 1, 1, ?)""",
        (guest_hash,),
    )
    analysis_id = cur.lastrowid
    conn.execute(
        """INSERT INTO wires (analysis_id, class_name, confidence, crs, band)
           VALUES (?, 'power_line', 0.92, 100.0, 'Critical')""",
        (analysis_id,),
    )
    conn.commit()
    conn.close()

    client = TestClient(app)
    client.cookies.set("guest_session", guest_token)
    res = client.get(f"/api/analysis/{analysis_id}/result")
    assert res.status_code == 200
    body = res.json()
    assert body["risk_score"] == 44.6
    assert body["crs_band"] == "Moderate"
    assert body["risk_band"] == "Moderate"
    assert abs(body["crs"] - 0.446) < 1e-6
    assert body["max_wire_crs"] == 100.0
    assert body["max_wire_band"] == "Critical"


def test_determine_overall_safety_wire_floor_and_reporting():
    from backend import database
    from backend.routes.reporting import _get_analysis_data
    from backend.utils.clearance_helpers import determine_overall_safety

    # 1. risk_score 44.6 (Moderate) + max_wire_band Critical + near -> Critical
    assert determine_overall_safety(
        ["Caution"],
        max_wire_band="Critical",
        analysis_band="Moderate",
        clearance_status="near",
    ) == "Critical"

    # 2. max_wire_band High + above + Low -> Unsafe
    assert determine_overall_safety(
        ["Safe"],
        max_wire_band="High",
        analysis_band="Low",
        clearance_status="above",
    ) == "Unsafe"

    # 3. max_wire_band Moderate + above + Low -> Safe (no escalation)
    assert determine_overall_safety(
        ["Safe"],
        max_wire_band="Moderate",
        analysis_band="Low",
        clearance_status="above",
    ) == "Safe"

    # 4. no_pole + Critical wire -> Critical
    assert determine_overall_safety(
        ["Unknown"],
        max_wire_band="Critical",
        analysis_band="Low",
        clearance_status="no_pole",
    ) == "Critical"

    # 5. missing max_wire_band -> existing behavior
    assert determine_overall_safety(["Safe", "Caution"]) == "Caution"
    assert determine_overall_safety(["Safe"]) == "Safe"

    # 6. Reporting _get_analysis_data mirrors the rule and reasons
    conn = database.get_connection()
    cur = conn.execute(
        """INSERT INTO analyses (filename, status, risk_score, max_crs, clearance_available, clearance_ok, is_public, is_mock)
           VALUES ('report_floor.jpg', 'complete', 44.6, 100.0, 4.5, 2, 0, 1)"""
    )
    aid = cur.lastrowid
    conn.execute(
        """INSERT INTO wires (analysis_id, class_name, confidence, crs, band, safety_level)
           VALUES (?, 'power_line', 0.92, 100.0, 'Critical', 'Caution')""",
        (aid,),
    )
    conn.commit()
    rep = _get_analysis_data(conn, aid)
    conn.close()

    assert rep["safety_level"] == "Critical"
    assert "Risk score 44.6 (Moderate)" in rep["reasons"]
    assert "Highest wire risk is Critical (100.0)" in rep["reasons"]
    assert "CRS band is Critical" not in rep["reasons"]


def _make_valid_test_image(tmp_path, filename="test_pole_wires.jpg"):
    import cv2
    import numpy as np

    rng = np.random.default_rng(42)
    img = rng.integers(80, 170, size=(600, 800, 3), dtype=np.uint8)
    # Draw crisp lines to guarantee Laplacian variance well above 50
    cv2.line(img, (0, 150), (800, 180), (20, 20, 20), 3)
    cv2.line(img, (0, 220), (800, 260), (240, 240, 240), 3)
    cv2.rectangle(img, (640, 60), (760, 540), (50, 50, 50), 2)
    path = tmp_path / filename
    cv2.imwrite(str(path), img)
    return str(path)


def test_mock_scenarios_bands_and_overall_safety(tmp_path, monkeypatch):
    from backend import database
    from backend.formulas import compute_crs_band
    from backend.utils.clearance_helpers import determine_overall_safety
    from backend.utils.pipeline import execute_pipeline

    config_store.set_setting("risk_bands", config_store.DEFAULT_SETTINGS["risk_bands"])
    monkeypatch.setenv("WIREFRAME_MOCK", "1")
    img_path = _make_valid_test_image(tmp_path)

    expected = {
        "low": ("Low", (0.0, 33.0), "Safe"),
        "moderate": ("Moderate", (33.0001, 66.0), "Caution"),
        "high": ("High", (66.0001, 85.0), "Unsafe"),
        "critical": ("Critical", (85.0001, 100.0), "Critical"),
    }

    clearance_code_to_status = {0: "below", 1: "above", 2: "near", None: "no_pole"}

    for scenario, (exp_band, (lo, hi), exp_safety) in expected.items():
        monkeypatch.setenv("WIREFRAME_MOCK_SCENARIO", scenario)
        conn = database.get_connection()
        cur = conn.execute(
            "INSERT INTO analyses (filename, status, is_mock) VALUES (?, 'queued', 1)",
            (f"scenario_{scenario}.jpg",),
        )
        aid = cur.lastrowid
        conn.commit()

        execute_pipeline(aid, img_path, conn, None)

        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (aid,)).fetchone()
        wires = conn.execute("SELECT * FROM wires WHERE analysis_id = ?", (aid,)).fetchall()
        conn.close()

        assert row["status"] == "complete", f"scenario={scenario} status={row['status']}"
        score = float(row["risk_score"])
        assert lo <= score <= hi, f"scenario={scenario} risk_score={score} not in [{lo}, {hi}]"
        assert compute_crs_band(score) == exp_band

        wire_safety_levels = [w["safety_level"] for w in wires if w["safety_level"]]
        max_wire_crs = float(row["max_crs"])
        max_wire_band = compute_crs_band(max_wire_crs)
        assert max_wire_band == exp_band, f"scenario={scenario} max_wire_band={max_wire_band} (max_crs={max_wire_crs})"

        cl_status = clearance_code_to_status.get(row["clearance_ok"], "no_pole")
        if scenario == "critical":
            assert cl_status in ("near", "below"), f"critical scenario must have near/below clearance, got {cl_status}"

        overall_safety = determine_overall_safety(
            wire_safety_levels,
            max_wire_band=max_wire_band,
            analysis_band=compute_crs_band(score),
            clearance_status=cl_status,
        )
        assert overall_safety == exp_safety, (
            f"scenario={scenario} overall_safety={overall_safety} != {exp_safety} "
            f"(score={score}, max_crs={max_wire_crs}, cl={cl_status}, wire_levels={wire_safety_levels})"
        )


def test_mock_scenario_cycle_order(tmp_path, monkeypatch):
    from backend import database
    from backend.formulas import compute_crs_band
    from backend.mock_detector import reset_mock_cycle
    from backend.utils.pipeline import execute_pipeline

    config_store.set_setting("risk_bands", config_store.DEFAULT_SETTINGS["risk_bands"])
    monkeypatch.setenv("WIREFRAME_MOCK", "1")
    monkeypatch.setenv("WIREFRAME_MOCK_SCENARIO", "cycle")
    reset_mock_cycle()

    img_path = _make_valid_test_image(tmp_path, "cycle_img.jpg")
    observed_bands = []

    for i in range(5):
        conn = database.get_connection()
        cur = conn.execute(
            "INSERT INTO analyses (filename, status, is_mock) VALUES (?, 'queued', 1)",
            (f"cycle_{i}.jpg",),
        )
        aid = cur.lastrowid
        conn.commit()

        execute_pipeline(aid, img_path, conn, None)

        row = conn.execute("SELECT risk_score FROM analyses WHERE id = ?", (aid,)).fetchone()
        conn.close()
        observed_bands.append(compute_crs_band(float(row["risk_score"])))

    assert observed_bands == ["Low", "Moderate", "High", "Critical", "Low"]


def test_mock_scenario_invalid_warns_and_ignored_when_mock_off(monkeypatch, caplog):
    import logging

    from fastapi.testclient import TestClient

    from backend.mock_detector import get_mock_scenario
    from main import app

    monkeypatch.setenv("WIREFRAME_MOCK", "1")
    monkeypatch.setenv("WIREFRAME_MOCK_SCENARIO", "extreme_ultra")
    with caplog.at_level(logging.WARNING):
        val = get_mock_scenario()
    assert val == "random"
    assert any("Invalid WIREFRAME_MOCK_SCENARIO" in rec.message for rec in caplog.records)

    client = TestClient(app)
    monkeypatch.setenv("WIREFRAME_MOCK_SCENARIO", "critical")
    res_mock = client.get("/api/health")
    assert res_mock.status_code == 200
    data_mock = res_mock.json()
    assert data_mock["mock_mode"] is True
    assert data_mock["mock_scenario"] == "critical"

    # When WIREFRAME_MOCK is unset / "0", scenario is ignored and omitted from /api/health
    caplog.clear()
    monkeypatch.delenv("WIREFRAME_MOCK", raising=False)
    monkeypatch.setenv("WIREFRAME_MOCK_SCENARIO", "invalid_should_be_ignored")
    assert get_mock_scenario() == "random"
    assert not any("Invalid WIREFRAME_MOCK_SCENARIO" in rec.message for rec in caplog.records)

    res_real = client.get("/api/health")
    assert res_real.status_code == 200
    data_real = res_real.json()
    assert data_real["mock_mode"] is False
    assert "mock_scenario" not in data_real


def test_blurry_image_still_rejected_under_critical_scenario(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    from backend import database
    from backend.utils.pipeline import execute_pipeline

    monkeypatch.setenv("WIREFRAME_MOCK", "1")
    monkeypatch.setenv("WIREFRAME_MOCK_SCENARIO", "critical")

    # Uniform gray image -> laplacian variance = 0.0 (< 50.0) -> rejected as too_blurry
    blurry = np.full((600, 800, 3), 128, dtype=np.uint8)
    blurry_path = str(tmp_path / "blurry_critical.jpg")
    cv2.imwrite(blurry_path, blurry)

    conn = database.get_connection()
    cur = conn.execute(
        "INSERT INTO analyses (filename, status, is_mock) VALUES ('blurry_critical.jpg', 'queued', 1)"
    )
    aid = cur.lastrowid
    conn.commit()

    execute_pipeline(aid, blurry_path, conn, None)

    row = conn.execute("SELECT status, summary, risk_score FROM analyses WHERE id = ?", (aid,)).fetchone()
    conn.close()

    assert row["status"] == "rejected"
    assert row["summary"] == "too_blurry"
    assert row["risk_score"] is None



