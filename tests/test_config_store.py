"""Tests for backend.config_store."""

from backend.config_store import (
    DEFAULT_SETTINGS,
    clear_cache,
    get_setting,
    set_setting,
)


class TestGetSetting:
    def test_seeded_defaults(self):
        for key in DEFAULT_SETTINGS:
            assert get_setting(key) == DEFAULT_SETTINGS[key], f"Mismatch for {key}"

    def test_crs_params(self):
        crs = get_setting("crs_params")
        assert crs["sag_offset"] == 0.03
        assert crs["sag_scale"] == 0.17
        assert crs["residual_offset"] == 2.0
        assert crs["residual_gain"] == 5.0
        assert crs["penalty_cap"] == 20

    def test_site_risk_weights(self):
        w = get_setting("site_risk_weights")
        assert w["vni"] == 0.45
        assert w["crs"] == 0.45
        assert w["dead"] == 0.10

    def test_missing_key_returns_default(self):
        assert get_setting("nonexistent", 42) == 42

    def test_missing_key_returns_none(self):
        assert get_setting("nonexistent") is None


class TestSetSetting:
    def test_set_and_get(self):
        set_setting("test_key", {"a": 1}, updated_by="test")
        assert get_setting("test_key") == {"a": 1}

    def test_overwrite(self):
        set_setting("pec_min_clearance_by_class", 5.0, updated_by="test")
        assert get_setting("pec_min_clearance_by_class") == 5.0

    def test_persists_after_cache_clear(self, _init_settings):
        set_setting("test_persist", [1, 2, 3])
        clear_cache()
        assert get_setting("test_persist") == [1, 2, 3]


class TestClearCache:
    def test_forces_db_read(self, _init_settings):
        original = get_setting("pec_min_clearance_by_class")
        _init_settings.execute(
            """UPDATE settings SET value_json = '{"default": 99.9}' WHERE key = 'pec_min_clearance_by_class'"""
        )
        _init_settings.commit()
        assert get_setting("pec_min_clearance_by_class") == original
        clear_cache()
        assert get_setting("pec_min_clearance_by_class") == {"default": 99.9}
