"""Characterization tests for backend.formulas."""

import pytest

from backend.formulas import (
    compute_barangay_risk,
    compute_catenary_risk,
    compute_detection_risk,
    compute_metric_clearance,
    compute_risk_score,
    get_severity,
    get_severity_weight,
)


class TestComputeCatenaryRisk:
    def test_moderate_sag_low_residual(self):
        # sag_term = (0.10-0.03)/0.17*100 = 41.176..., residual_pen = 0
        result = compute_catenary_risk(0.10, 1.5)
        assert pytest.approx(result, abs=0.01) == 41.18

    def test_zero_at_offsets(self):
        assert compute_catenary_risk(0.03, 2.0) == 0.0

    def test_below_offsets(self):
        assert compute_catenary_risk(0.01, 0.5) == 0.0

    def test_max_saturation(self):
        assert compute_catenary_risk(0.20, 3.5) == 100.0

    def test_sag_plus_residual(self):
        # sag=41.18 + pen=(5-2)*5=15 = 56.18
        result = compute_catenary_risk(0.10, 5.0)
        assert pytest.approx(result, abs=0.01) == 56.18

    def test_residual_penalty_cap(self):
        # pen = (100-2)*5 = 490 capped at 20
        assert compute_catenary_risk(0.03, 100.0) == 20.0


class TestComputeBarangayRisk:
    def test_typical(self):
        # 0.45*0.5*100 + 0.45*60 + 0.10*0.2*100 = 22.5+27+2 = 51.5
        result = compute_barangay_risk(0.5, 60.0, 2, 10)
        assert pytest.approx(result, abs=0.01) == 51.50

    def test_all_zeros(self):
        assert compute_barangay_risk(0.0, 0.0, 0, 0) == 0.0

    def test_maximum(self):
        assert compute_barangay_risk(1.0, 100.0, 5, 5) == 100.0

    def test_no_wires(self):
        # wire_count=0 -> density=1/1=1 -> 0+0+0.10*1*100=10
        result = compute_barangay_risk(0.0, 0.0, 1, 0)
        assert pytest.approx(result, abs=0.01) == 10.0


class TestGetSeverity:
    def test_high_confidence_is_low_severity(self):
        assert get_severity(0.9) == "low"

    def test_boundary_high(self):
        assert get_severity(0.7) == "low"

    def test_medium_confidence(self):
        assert get_severity(0.5) == "medium"

    def test_boundary_medium(self):
        assert get_severity(0.4) == "medium"

    def test_low_confidence_is_high_severity(self):
        assert get_severity(0.2) == "high"

    def test_just_below_high(self):
        assert get_severity(0.69) == "medium"

    def test_just_below_medium(self):
        assert get_severity(0.39) == "high"


class TestGetSeverityWeight:
    def test_known_weights(self):
        assert get_severity_weight("high") == 1.0
        assert get_severity_weight("medium") == 0.5
        assert get_severity_weight("low") == 0.1

    def test_unknown_default(self):
        assert get_severity_weight("unknown") == 0.5


class TestComputeDetectionRisk:
    def test_empty(self):
        assert compute_detection_risk([]) == 0.0

    def test_all_low(self):
        assert compute_detection_risk(["low", "low"]) == pytest.approx(10.0)

    def test_mixed(self):
        # [0.1, 0.1, 0.5] mean=0.2333 -> 23.33
        result = compute_detection_risk(["low", "low", "medium"])
        assert pytest.approx(result, abs=0.01) == 23.33

    def test_all_high(self):
        assert compute_detection_risk(["high", "high"]) == pytest.approx(100.0)


class TestComputeRiskScore:
    def test_balanced(self):
        assert compute_risk_score(30.0, 50.0) == 40.0

    def test_zeros(self):
        assert compute_risk_score(0.0, 0.0) == 0.0

    def test_max(self):
        assert compute_risk_score(100.0, 100.0) == 100.0

    def test_clipped(self):
        assert compute_risk_score(200.0, 200.0) == 100.0


class TestComputeMetricClearance:
    @pytest.fixture(autouse=True)
    def patch_settings(self, monkeypatch):
        # We patch get_setting to return defaults without DB
        from backend.config_store import DEFAULT_SETTINGS
        def _mock_get_setting(key, default=None):
            return DEFAULT_SETTINGS.get(key, default)
        monkeypatch.setattr("backend.formulas.get_setting", _mock_get_setting)
        monkeypatch.setattr("backend.utils.cv.calibration.get_setting", _mock_get_setting)

    def test_within_clearance(self):
        r = compute_metric_clearance(50.0, 400.0, 10.0)
        # pole_height default is 9.0. So 9.0/400.0 = 0.0225
        # sag_meters = 50.0 * 0.0225 = 1.125
        assert r["sag_meters"] == pytest.approx(1.125, abs=0.001)
        assert r["clearance_ok"] is True
        assert r["meters_per_px"] == pytest.approx(0.0225, abs=0.0001)

    def test_exceeds_clearance(self):
        r = compute_metric_clearance(300.0, 400.0, 10.0)
        
        assert r["sag_meters"] == pytest.approx(6.75, abs=0.001)
        assert r["clearance_ok"] is False

    def test_zero_pole_height(self):
        r = compute_metric_clearance(50.0, 0.0, 10.0)
        assert r["clearance_ok"] is False
        assert r["sag_meters"] == 0.0

    def test_uses_assumed_height(self, monkeypatch):
        # We need to mock get_setting("pole_heights") to return [12.0] for this test
        def _mock_get_setting(key, default=None):
            if key == "pole_heights":
                return [12.0]
            from backend.config_store import DEFAULT_SETTINGS
            return DEFAULT_SETTINGS.get(key, default)
        monkeypatch.setattr("backend.formulas.get_setting", _mock_get_setting)
        monkeypatch.setattr("backend.utils.cv.calibration.get_setting", _mock_get_setting)
        
        r = compute_metric_clearance(50.0, 400.0, 12.0)
        assert r["assumed_pole_height_m"] == 12.0
        assert r["meters_per_px"] == pytest.approx(0.03, abs=0.0001)


class TestPaperConformantFormulasAndCV:
    def test_compute_paper_crs_boundaries_and_fit_failed(self):
        from backend.formulas import compute_paper_crs

        # Low (< 3.0): delta_rel=0.008, delta_lim=0.04 -> 10 * 0.2 = 2.0
        crs_low, lvl_low = compute_paper_crs(0.008, delta_lim=0.04, class_weight=1.0)
        assert crs_low == pytest.approx(2.0)
        assert lvl_low == "Low"

        # Exact boundary 3.0 -> Medium
        crs_b3, lvl_b3 = compute_paper_crs(0.012, delta_lim=0.04, class_weight=1.0)
        assert crs_b3 == pytest.approx(3.0)
        assert lvl_b3 == "Medium"

        # Exact boundary 6.0 -> Medium
        crs_b6, lvl_b6 = compute_paper_crs(0.024, delta_lim=0.04, class_weight=1.0)
        assert crs_b6 == pytest.approx(6.0)
        assert lvl_b6 == "Medium"

        # High (> 6.0): delta_rel=0.032, delta_lim=0.04 -> 8.0
        crs_high, lvl_high = compute_paper_crs(0.032, delta_lim=0.04, class_weight=1.0)
        assert crs_high == pytest.approx(8.0)
        assert lvl_high == "High"

        # Cap at 10.0
        crs_cap, lvl_cap = compute_paper_crs(0.10, delta_lim=0.04, class_weight=1.2)
        assert crs_cap == pytest.approx(10.0)
        assert lvl_cap == "High"

        # fit_failed=True -> (None, None)
        crs_fail, lvl_fail = compute_paper_crs(0.05, delta_lim=0.04, fit_failed=True)
        assert crs_fail is None
        assert lvl_fail is None

    def test_compute_allowable_sag_default_and_context(self):
        from backend.formulas import compute_allowable_sag

        d_lim, flags = compute_allowable_sag("power_distribution_line", span_m=None)
        assert d_lim == pytest.approx(0.030)
        assert "default_parameters" in flags

        # With span and higher temperature, expected sag increases -> delta_lim > delta_lim0
        d_lim_hot, flags_hot = compute_allowable_sag(
            "power_distribution_line",
            span_m=40.0,
            context={"temperature_c": 45.0, "wind_speed_mps": 5.0},
        )
        assert d_lim_hot > d_lim
        assert "default_parameters" not in flags_hot

    def test_compute_mspi_and_66_point_sensitivity_grid(self):
        from backend.formulas import compute_mspi, compute_mspi_sensitivity_grid, mspi_weight_grid

        # Empty wire list -> NO_VALID_WIRES
        empty_res = compute_mspi(0.5, [])
        assert empty_res["mspi"] is None
        assert "NO_VALID_WIRES" in empty_res["flags"]

        # Valid wire CRS list on 0-10 scale: [2.0, 6.0] -> mean=4.0, max=6.0
        # MSPI = (1/3)*0.6 + (1/3)*(4.0/10) + (1/3)*(6.0/10) = (0.6 + 0.4 + 0.6)/3 = 1.6/3 = 0.533333
        res = compute_mspi(0.6, [2.0, 6.0])
        assert res["mspi"] == pytest.approx(1.6 / 3.0, abs=1e-5)
        assert res["mean_crs_10"] == pytest.approx(4.0)
        assert res["max_crs_10"] == pytest.approx(6.0)
        assert res["paper_risk_level"] == "Medium"
        assert res["flags"] == []

        grid = mspi_weight_grid(0.1)
        assert len(grid) == 66

        sites = [
            {"site_id": "pole_A", "vni": 0.80, "crs_10_list": [7.5, 8.5]},
            {"site_id": "pole_B", "vni": 0.45, "crs_10_list": [4.0, 5.0]},
            {"site_id": "pole_C", "vni": 0.15, "crs_10_list": [1.5, 2.0]},
        ]
        sens = compute_mspi_sensitivity_grid(sites)
        assert sens["grid_size"] == 66
        assert sens["n_sites"] == 3
        assert len(sens["rows"]) == 66
        assert sens["min_rho"] == pytest.approx(1.0)
        assert sens["rank_range"]["pole_A"] == (1, 1)

    def test_vni_bands_and_hann_window(self):
        import numpy as np
        from backend.utils.cv.vni import compute_vni, get_vni_band, hann_2d

        assert get_vni_band(0.0) == "Low"
        assert get_vni_band(0.25) == "Low"
        assert get_vni_band(0.26) == "Moderate"
        assert get_vni_band(0.50) == "Moderate"
        assert get_vni_band(0.51) == "High"
        assert get_vni_band(0.75) == "High"
        assert get_vni_band(0.76) == "Severe"
        assert get_vni_band(1.00) == "Severe"

        w2d = hann_2d(640, 640)
        assert w2d.shape == (640, 640)
        assert w2d[0, 0] == pytest.approx(0.0)
        assert w2d[320, 320] > 0.99

        rng = np.random.default_rng(123)
        noisy = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
        vni_res = compute_vni(noisy)
        assert 0.0 <= vni_res.vni <= 1.0
        assert vni_res.vni_raw is not None
        assert vni_res.band in ("Low", "Moderate", "High", "Severe")

    def test_centerline_flags_truncated_and_too_short(self):
        import cv2
        import numpy as np
        from backend.utils.cv.centerline import extract_centerline

        # 1. Wire touching left border (x=2 <= 10 px) -> truncated_wire
        mask_trunc = np.zeros((100, 200), dtype=np.uint8)
        cv2.line(mask_trunc, (2, 50), (120, 50), 255, thickness=2)
        res_trunc = extract_centerline(mask_trunc, [0, 40, 130, 60])
        assert res_trunc.is_valid is True
        assert "truncated_wire" in res_trunc.flags
        assert "too_short_to_fit" not in res_trunc.flags

        # 2. Short interior wire (length 12 px < n_min=20, area >= 15 px) -> too_short_to_fit
        mask_short = np.zeros((100, 200), dtype=np.uint8)
        cv2.line(mask_short, (50, 50), (60, 50), 255, thickness=2)
        res_short = extract_centerline(mask_short, [45, 45, 65, 55], n_min=20)
        assert res_short.is_valid is True
        assert "too_short_to_fit" in res_short.flags
        assert "truncated_wire" not in res_short.flags
