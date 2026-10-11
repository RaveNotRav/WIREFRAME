import os
import sqlite3

from backend.database import (
    MIGRATIONS,
    _ensure_schema_version_table,
)


def test_migration_004_with_data():
    if os.path.exists("test_mig.db"):
        os.remove("test_mig.db")
        
    conn = sqlite3.connect("test_mig.db")
    conn.row_factory = sqlite3.Row
    # Apply up to 003
    _ensure_schema_version_table(conn)
    for v, d, fn in MIGRATIONS[:3]:
        fn(conn)
        conn.execute("INSERT INTO schema_version (version, description) VALUES (?, ?)", (v, d))
        conn.commit()
        
    # Insert data
    conn.execute("INSERT INTO analyses (filename, status) VALUES ('old1.jpg', 'complete')")
    conn.execute("INSERT INTO analyses (filename, status) VALUES ('old2.jpg', 'pending')")
    conn.commit()
    
    # Run 004
    MIGRATIONS[3][2](conn)
    conn.commit()
    
    # Verify
    row1 = conn.execute("SELECT batch_id FROM analyses WHERE id=1").fetchone()
    assert row1["batch_id"] is None
    row2 = conn.execute("SELECT batch_id FROM analyses WHERE id=2").fetchone()
    assert row2["batch_id"] is None
    
    # Verify inserting batch
    conn.execute("INSERT INTO batches (status) VALUES ('queued')")
    conn.execute("UPDATE analyses SET batch_id = 1 WHERE id = 1")
    conn.commit()
    
    row = conn.execute("SELECT batch_id FROM analyses WHERE id=1").fetchone()
    assert row["batch_id"] == 1
    
    conn.close()
    os.remove("test_mig.db")

def test_migration_005_with_data():
    if os.path.exists("test_mig5.db"):
        os.remove("test_mig5.db")
        
    conn = sqlite3.connect("test_mig5.db")
    conn.row_factory = sqlite3.Row
    # Apply up to 004
    _ensure_schema_version_table(conn)
    for v, d, fn in MIGRATIONS[:4]:
        fn(conn)
        conn.execute("INSERT INTO schema_version (version, description) VALUES (?, ?)", (v, d))
        conn.commit()
        
    # Insert data
    conn.execute("INSERT INTO analyses (filename, status) VALUES ('old3.jpg', 'complete')")
    conn.execute("INSERT INTO batches (status) VALUES ('done')")
    conn.commit()
    
    # Run 005
    MIGRATIONS[4][2](conn)
    conn.commit()
    
    # Verify new columns
    row = conn.execute("SELECT created_by, site_id FROM analyses WHERE id=1").fetchone()
    assert row["created_by"] is None
    assert row["site_id"] is None
    
    row2 = conn.execute("SELECT created_by FROM batches WHERE id=1").fetchone()
    assert row2["created_by"] is None
    
    conn.close()
    os.remove("test_mig5.db")

def test_migration_007_with_data():
    if os.path.exists("test_mig7.db"):
        os.remove("test_mig7.db")
        
    conn = sqlite3.connect("test_mig7.db")
    conn.row_factory = sqlite3.Row
    # Apply up to 006
    _ensure_schema_version_table(conn)
    for v, d, fn in MIGRATIONS[:6]:
        fn(conn)
        conn.execute("INSERT INTO schema_version (version, description) VALUES (?, ?)", (v, d))
        conn.commit()
        
    # Insert data
    conn.execute("INSERT INTO analyses (filename, status) VALUES ('old3.jpg', 'complete')")
    conn.execute("INSERT INTO sites (name, barangay, sector) VALUES ('Site1', 'Brgy B', 'Sector X')")
    conn.commit()
    
    # Run 007
    MIGRATIONS[6][2](conn)
    conn.commit()
    
    # Verify new columns in analyses
    row = conn.execute("SELECT street, company FROM analyses WHERE id=1").fetchone()
    assert row["street"] is None
    assert row["company"] is None
    
    # Verify new columns in sites
    row2 = conn.execute("SELECT street, barangay_norm, street_norm FROM sites WHERE id=1").fetchone()
    assert row2["street"] is None
    assert row2["barangay_norm"] == "brgy b"
    assert row2["street_norm"] == ""
    
    # Verify unique index prevents duplicates
    try:
        conn.execute("INSERT INTO sites (name, barangay, barangay_norm, street, street_norm) VALUES ('Site2', 'Brgy B', 'b', 'Rizal', 'rizal')")
        conn.execute("INSERT INTO sites (name, barangay, barangay_norm, street, street_norm) VALUES ('Site3', 'Brgy B', 'b', 'Rizal', 'rizal')")
        assert False, "Should raise IntegrityError"
    except sqlite3.IntegrityError:
        pass
    
    conn.close()
    os.remove("test_mig7.db")
