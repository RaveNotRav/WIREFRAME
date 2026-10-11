from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from backend.auth import User, get_current_user, require_role, verify_csrf
from backend.database import get_connection

router = APIRouter(prefix="/api/sites", tags=["sites"])

class SiteCreate(BaseModel):
    name: str
    barangay: str = None
    sector: str = None
    lat: float = None
    lng: float = None

@router.get("")
async def list_sites(
    user: User = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100)
):
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM sites LIMIT ? OFFSET ?", (limit, skip)).fetchall()
        total = conn.execute("SELECT COUNT(*) as c FROM sites").fetchone()["c"]
        return {
            "items": [dict(r) for r in rows],
            "total": total,
            "skip": skip,
            "limit": limit
        }
    finally:
        conn.close()

@router.post("", dependencies=[Depends(verify_csrf), Depends(require_role(["admin", "lgu_engineer"]))])
async def create_site(payload: SiteCreate):
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO sites (name, barangay, sector, lat, lng) VALUES (?, ?, ?, ?, ?)",
            (payload.name, payload.barangay, payload.sector, payload.lat, payload.lng)
        )
        conn.commit()
        return {"status": "success", "id": cur.lastrowid}
    finally:
        conn.close()
