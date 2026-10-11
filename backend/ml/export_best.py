"""Export the best model weights to the backend and register in config store."""

import argparse
import json
import logging
import shutil
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def update_db_config(db_path: Path, model_path: str):
    conn = sqlite3.connect(db_path)
    try:
        # Check if settings table exists
        exists = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='settings'").fetchone()
        if exists:
            # Update model_path setting
            conn.execute(
                """INSERT INTO settings (key, value_json, updated_at, updated_by)
                   VALUES (?, ?, datetime('now'), ?)
                   ON CONFLICT(key) DO UPDATE SET
                       value_json = excluded.value_json,
                       updated_at = excluded.updated_at,
                       updated_by = excluded.updated_by""",
                ("model_path", json.dumps(model_path), "ml_pipeline")
            )
            conn.commit()
            logger.info(f"Database config updated: model_path = {model_path}")
        else:
            logger.warning("Settings table not found in database. Run migrations first.")
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", required=True, help="Path to best.pt from training run")
    parser.add_argument("--db", default="wireframe.db", help="Path to SQLite DB")
    args = parser.parse_args()
    
    src = Path(args.best)
    if not src.exists():
        logger.error(f"Model file not found: {src}")
        return
        
    models_dir = Path("backend/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    dest = models_dir / "wireframe_seg.pt"
    
    logger.info(f"Copying {src} to {dest}")
    shutil.copy2(src, dest)
    
    # Update SQLite database
    db_path = Path(args.db)
    if db_path.exists():
        update_db_config(db_path, str(dest.as_posix()))
    else:
        logger.warning(f"Database {db_path} not found. Could not update settings.")

if __name__ == "__main__":
    main()
