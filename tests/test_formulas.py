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
