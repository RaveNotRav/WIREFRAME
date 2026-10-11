"""Tests for backend.database migration system."""

from backend.database import (
    MIGRATIONS,
    get_connection,
    get_schema_version,
    run_migrations,
)


class TestMigrations:
    def test_schema_version_at_max(self, _init_settings):
        ver = get_schema_version(_init_settings)
        assert ver == max(v for v, _, _ in MIGRATIONS)

    def test_idempotent(self, _init_settings):
        ver_before = get_schema_version(_init_settings)
        run_migrations(_init_settings)
        assert get_schema_version(_init_settings) == ver_before

    def test_tables_exist(self, _init_settings):
        tables = {
            row[0]
            for row in _init_settings.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for t in ("analyses", "detections", "reports", "settings", "schema_version"):
            assert t in tables, f"Missing table: {t}"

    def test_analyses_columns(self, _init_settings):
        cols = {
            row[1]
            for row in _init_settings.execute(
                "PRAGMA table_info(analyses)"
            ).fetchall()
        }
        for c in ("id", "filename", "status", "risk_score", "vni", "max_crs",
                  "barangay_risk", "sag_meters", "clearance_ok", "created_at"):
            assert c in cols, f"Missing column: {c}"

    def test_detections_columns(self, _init_settings):
        cols = {
            row[1]
            for row in _init_settings.execute(
                "PRAGMA table_info(detections)"
            ).fetchall()
        }
        for c in ("id", "analysis_id", "class_name", "confidence",
                  "bbox_x1", "severity", "crs", "sag_ratio"):
            assert c in cols, f"Missing column: {c}"


class TestConnection:
    def test_wal_mode(self, tmp_path):
        conn = get_connection(str(tmp_path / "wal.db"))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()

    def test_foreign_keys_on(self, tmp_path):
        conn = get_connection(str(tmp_path / "fk.db"))
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
        conn.close()
