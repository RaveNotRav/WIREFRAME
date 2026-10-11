"""SQLite database management with ordered migrations for WIREFRAME."""

import logging
import os
import sqlite3

logger = logging.getLogger(__name__)


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Create a SQLite connection with WAL mode and busy timeout."""
    path = db_path or os.environ.get("WIREFRAME_DB_PATH", "wireframe.db")
    conn = sqlite3.connect(path, timeout=5.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    logger.debug("Database connection opened: %s", path)
    return conn


def _ensure_schema_version_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now')),
            description TEXT
        )
    """)
    conn.commit()


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Return current schema version (0 if no migrations applied)."""
    _ensure_schema_version_table(conn)
    row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    return row["v"] if row["v"] is not None else 0


# ---------------------------------------------------------------------------
# Migration functions
# ---------------------------------------------------------------------------

def _migrate_001_base_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_path TEXT,
            annotated_path TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            barangay TEXT,
            sector TEXT,
            summary TEXT,
            risk_score REAL,
            vni REAL,
            max_crs REAL,
            dead_wire_count INTEGER DEFAULT 0,
            barangay_risk REAL,
            enhancement TEXT,
            authenticity TEXT,
            sag_meters REAL,
            clearance_ok INTEGER,
            clearance_available REAL,
            pec_min_clearance_m REAL,
            assumed_pole_height_m REAL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
            class_name TEXT NOT NULL,
            confidence REAL,
            bbox_x1 REAL,
            bbox_y1 REAL,
            bbox_x2 REAL,
            bbox_y2 REAL,
            severity TEXT,
            crs REAL,
            sag_ratio REAL,
            residual REAL
        );

        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            path TEXT,
            metadata TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_detections_analysis_id ON detections(analysis_id);
        CREATE INDEX IF NOT EXISTS idx_reports_analysis_id ON reports(analysis_id);
        CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses(status);
        CREATE INDEX IF NOT EXISTS idx_analyses_barangay ON analyses(barangay);
    """)


def _migrate_002_settings_table(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_by TEXT NOT NULL DEFAULT 'system'
        );
    """)


def _migrate_003_wires_table(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS wires (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
            class_name TEXT NOT NULL,
            confidence REAL,
            crs REAL,
            band TEXT,
            sag_px REAL,
            span_px REAL,
            sag_m REAL,
            clearance_m REAL,
            clearance_status TEXT,
            safety_level TEXT,
            centerline_json TEXT,
            params_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_wires_analysis_id ON wires(analysis_id);
    """)

def _migrate_004_orchestrator(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL DEFAULT 'queued',
            cancel_requested INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            started_at TEXT,
            finished_at TEXT
        );
        
        CREATE TABLE IF NOT EXISTS pipeline_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
            state TEXT NOT NULL,
            ts TEXT NOT NULL DEFAULT (datetime('now')),
            detail TEXT
        );
        
        ALTER TABLE analyses ADD COLUMN batch_id INTEGER REFERENCES batches(id) ON DELETE CASCADE;
        ALTER TABLE analyses ADD COLUMN current_state TEXT;
        
        CREATE INDEX IF NOT EXISTS idx_analyses_batch_id ON analyses(batch_id);
        CREATE INDEX IF NOT EXISTS idx_pipeline_events_analysis_id_id ON pipeline_events(analysis_id, id);
    """)

def _migrate_005_rbac_and_sites(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_hash TEXT UNIQUE NOT NULL,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            expires_at TEXT NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS sites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            barangay TEXT,
            sector TEXT,
            lat REAL,
            lng REAL
        );
        
        CREATE TABLE IF NOT EXISTS calibrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
            created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            payload_json TEXT NOT NULL,
            sag_m REAL,
            clearance_m REAL,
            safety_level TEXT
        );

        ALTER TABLE batches ADD COLUMN created_by INTEGER REFERENCES users(id) ON DELETE SET NULL;
        ALTER TABLE analyses ADD COLUMN created_by INTEGER REFERENCES users(id) ON DELETE SET NULL;
        ALTER TABLE analyses ADD COLUMN site_id INTEGER REFERENCES sites(id) ON DELETE SET NULL;
        ALTER TABLE analyses ADD COLUMN gps_lat REAL;
        ALTER TABLE analyses ADD COLUMN gps_lng REAL;
        ALTER TABLE analyses ADD COLUMN gps_source TEXT;
        ALTER TABLE analyses ADD COLUMN spectrum_path TEXT;
        ALTER TABLE analyses ADD COLUMN width INTEGER;
        ALTER TABLE analyses ADD COLUMN height INTEGER;
    """)

def _migrate_006_mock_flag(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        ALTER TABLE analyses ADD COLUMN is_mock INTEGER NOT NULL DEFAULT 0;
    """)

def _migrate_007_company_street(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        ALTER TABLE analyses ADD COLUMN company TEXT;
        ALTER TABLE analyses ADD COLUMN street TEXT;
        
        ALTER TABLE sites ADD COLUMN street TEXT;
        ALTER TABLE sites ADD COLUMN barangay_norm TEXT;
        ALTER TABLE sites ADD COLUMN street_norm TEXT;
    """)
    # Backfill
    conn.execute("UPDATE sites SET barangay_norm = LOWER(TRIM(barangay)), street_norm = ''")
    
    conn.executescript("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_sites_norm 
        ON sites(barangay_norm, street_norm) 
        WHERE street_norm <> '';
    """)

def _migrate_008_dashboard_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at);
        CREATE INDEX IF NOT EXISTS idx_analyses_is_mock ON analyses(is_mock);
        CREATE INDEX IF NOT EXISTS idx_analyses_site_id ON analyses(site_id);
    """)

def _migrate_009_analysis_notes(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        ALTER TABLE analyses ADD COLUMN notes TEXT;
    """)


def _migrate_010_public_submissions(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE batches ADD COLUMN guest_session_id TEXT")
    conn.execute("ALTER TABLE batches ADD COLUMN is_public INTEGER DEFAULT 0")
    conn.execute("ALTER TABLE analyses ADD COLUMN guest_session_id TEXT")
    conn.execute("ALTER TABLE analyses ADD COLUMN is_public INTEGER DEFAULT 0")

MIGRATIONS: list[tuple[int, str, callable]] = [

    (1, "create_base_tables", _migrate_001_base_tables),
    (2, "create_settings_table", _migrate_002_settings_table),
    (3, "create_wires_table", _migrate_003_wires_table),
    (4, "create_orchestrator_tables", _migrate_004_orchestrator),
    (5, "rbac_and_sites", _migrate_005_rbac_and_sites),
    (6, "mock_flag", _migrate_006_mock_flag),
    (7, "company_and_street", _migrate_007_company_street),
    (8, "dashboard_indexes", _migrate_008_dashboard_indexes),
    (9, "analysis_notes", _migrate_009_analysis_notes),
    (10, "public_submissions", _migrate_010_public_submissions),
]


def run_migrations(conn: sqlite3.Connection) -> int:
    """Run all pending migrations in order. Returns final schema version."""
    current = get_schema_version(conn)
    applied = 0
    for version, description, migrate_fn in MIGRATIONS:
        if version > current:
            logger.info("Applying migration %d: %s", version, description)
            migrate_fn(conn)
            conn.execute(
                "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                (version, description),
            )
            conn.commit()
            applied += 1
            logger.info("Migration %d applied", version)
    final = get_schema_version(conn)
    if applied:
        logger.info("Migrations complete. Schema v%d (%d applied)", final, applied)
    else:
        logger.info("Schema up to date at v%d", final)
    return final
