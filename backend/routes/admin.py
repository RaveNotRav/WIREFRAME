import os
from pathlib import Path

from fastapi import APIRouter, Depends

from backend.auth import require_role, verify_csrf
from backend.database import get_connection
from backend.utils.uploads import is_safe_contained_path

router = APIRouter(prefix="/api/admin", tags=["admin"])

@router.delete("/purge_mock", dependencies=[Depends(verify_csrf), Depends(require_role(["admin"]))])
async def purge_mock_data():
    conn = get_connection()
    try:
        # Get all mock file paths from analyses and reports to delete files
        analyses_rows = conn.execute("""
            SELECT original_path, annotated_path, spectrum_path, filename 
            FROM analyses 
            WHERE is_mock = 1
        """).fetchall()
        
        rep_rows = conn.execute("""
            SELECT r.path 
            FROM reports r 
            JOIN analyses a ON r.analysis_id = a.id 
            WHERE a.is_mock = 1
        """).fetchall()

        paths_to_delete = set()
        for r in analyses_rows:
            for col in ("original_path", "annotated_path", "spectrum_path"):
                val = r[col]
                if val:
                    paths_to_delete.add(val)
            if r["filename"]:
                paths_to_delete.add(os.path.join("uploads", r["filename"]))
                paths_to_delete.add(os.path.join("uploads", "thumb_" + r["filename"]))

        for r in rep_rows:
            if r["path"]:
                paths_to_delete.add(r["path"])

        for path_str in paths_to_delete:
            if not is_safe_contained_path(path_str, allowed_dirs=("uploads", "outputs")):
                continue
            p = Path(path_str).resolve()
            if p.is_file():
                try:
                    p.unlink(missing_ok=True)
                except OSError as e:
                    import logging
                    logging.warning(f"Failed to delete mock file {p}: {e}")
        
        # Delete analyses WHERE is_mock = 1
        # ON DELETE CASCADE handles related tables (wires, detections, reports, events, calibrations)
        conn.execute("DELETE FROM analyses WHERE is_mock = 1")
        conn.commit()
        return {"status": "success", "message": "Purged mock data"}
    finally:
        conn.close()
