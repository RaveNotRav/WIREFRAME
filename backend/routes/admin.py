import os

from fastapi import APIRouter, Depends

from backend.auth import require_role, verify_csrf
from backend.database import get_connection

router = APIRouter(prefix="/api/admin", tags=["admin"])

@router.delete("/purge_mock", dependencies=[Depends(verify_csrf), Depends(require_role(["admin"]))])
async def purge_mock_data():
    conn = get_connection()
    try:
        # Get all mock image paths to delete files
        files = conn.execute("SELECT original_path FROM analyses WHERE is_mock = 1").fetchall()
        for f in files:
            if f["original_path"] and os.path.exists(f["original_path"]):
                try:
                    os.remove(f["original_path"])
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to delete mock file {f['original_path']}: {e}")
        
        # Delete analyses WHERE is_mock = 1
        # ON DELETE CASCADE handles related tables (wires, detections, reports, events, calibrations)
        conn.execute("DELETE FROM analyses WHERE is_mock = 1")
        conn.commit()
        return {"status": "success", "message": "Purged mock data"}
    finally:
        conn.close()
