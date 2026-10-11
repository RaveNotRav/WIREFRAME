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
    assert compute_crs_band(33.4) == "Moderate"
    # 33.0 -> 33.00 -> 0.33 => Low
    assert compute_crs_band(33.0) == "Low"
    
    assert compute_crs_band(66.0) == "Moderate"
    assert compute_crs_band(67.0) == "High"
    
    assert compute_crs_band(85.0) == "High"
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
