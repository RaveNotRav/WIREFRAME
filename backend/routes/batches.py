import os
import secrets
import shutil
from pathlib import Path

from fastapi import (
    Response,
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from PIL import Image
from PIL.Image import DecompressionBombError

from backend import main_state
from backend.auth import User, check_rate_limit, get_current_user, verify_csrf, get_optional_user, check_anon_rate_limit, _hash_token, verify_analysis_access, cookie_secure
from backend.database import get_connection

router = APIRouter(prefix="/api/batches", tags=["batches"])

from backend.services.sites import get_or_create_site, validate_location, validate_gps_coordinates
from backend.services.worker import enqueue_batch
from backend.utils.uploads import generate_safe_filename


@router.post("", dependencies=[Depends(verify_csrf)])
async def create_batch(
    req: Request,
    files: list[UploadFile] = File(...),
    company: str | None = Form(None),
    barangay: str | None = Form(None),
    street: str | None = Form(None),
    sector: str | None = Form(None),
    gps_lat: str | None = Form(None),
    gps_lng: str | None = Form(None),
    user: User | None = Depends(get_optional_user)
):
    guest_session_id = None
    guest_token = None
    is_public = 0
    if user:
        check_rate_limit(req, "upload", user.username, limit=30, window=60)
    else:
        is_public = 1
        guest_token = req.cookies.get("guest_session")
        if not guest_token:
            guest_token = secrets.token_urlsafe(32)

        guest_session_id = _hash_token(guest_token)

    
    if not files:
        raise HTTPException(status_code=400, detail="Empty batch")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")

    # Limit total batch size to 60MB
    total_size = 0
    for f in files:
        f.file.seek(0, 2)
        total_size += f.file.tell()
        f.file.seek(0)
    if total_size > 60 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Batch exceeds maximum size of 60MB")

    # Parse and validate GPS
    lat_val, lng_val, gps_warning = validate_gps_coordinates(gps_lat, gps_lng)

    # If there's a global config error from startup, reject uploads
    if main_state.is_config_failed():
        raise HTTPException(status_code=503, detail="Server configuration is invalid. Cannot process uploads.")

    from backend.mock_detector import is_mock_mode
    if not is_mock_mode() and not main_state.is_model_available():
        raise HTTPException(
            status_code=503,
            detail="Model not available. The detection model is not loaded on this server."
        )

    conn = get_connection()
    
    # Create batch
    cur = conn.execute("INSERT INTO batches (status, created_by, guest_session_id, is_public) VALUES ('queued', ?, ?, ?)", (user.id if user else None, guest_session_id, is_public))
    batch_id = cur.lastrowid
    
    Path("uploads").mkdir(exist_ok=True)
    
    saved_files = []
    
    try:
        comp, brgy, st = validate_location(company, barangay, street)
        
        for f in files:
            if not f.filename or not f.filename.strip():
                raise HTTPException(status_code=400, detail="Missing or invalid filename")
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png"]:
                raise HTTPException(status_code=400, detail=f"Invalid file type: {f.filename}")
                
            # Limit file size to 15MB
            f.file.seek(0, 2)
            size = f.file.tell()
            f.file.seek(0)
            if size > 15 * 1024 * 1024:
                raise HTTPException(status_code=400, detail=f"File too large: {f.filename}")
                
            try:
                safe_name = generate_safe_filename(f.filename)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            out_path = os.path.join("uploads", safe_name)
            
            with open(out_path, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
                
            try:
                with Image.open(out_path) as img:
                    img.verify()
                # Re-open to generate thumbnail (verify() leaves the file in a bad state for some ops)
                with Image.open(out_path) as img:
                    # Convert to RGB in case of RGBA/P formats before saving as JPEG
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.thumbnail((300, 300))
                    thumb_name = "thumb_" + safe_name
                    if not thumb_name.lower().endswith(('.jpg', '.jpeg')):
                        thumb_name = os.path.splitext(thumb_name)[0] + ".jpg"
                    img.save(os.path.join("uploads", thumb_name), "JPEG", quality=85)
            except DecompressionBombError:
                if os.path.exists(out_path):
                    os.remove(out_path)
                raise HTTPException(status_code=400, detail=f"Image exceeds decompression pixel limit: {f.filename}")
            except Exception:
                if os.path.exists(out_path):
                    os.remove(out_path)
                raise HTTPException(status_code=400, detail=f"Corrupt image: {f.filename}")
                
            saved_files.append(safe_name)
            
        site_id = get_or_create_site(conn, comp, brgy, st, lat_val, lng_val)
            
        is_mock_flag = 1 if os.getenv("WIREFRAME_MOCK") == "1" else 0
        for safe_name in saved_files:
            conn.execute(
                "INSERT INTO analyses (filename, status, company, barangay, street, sector, batch_id, created_by, site_id, is_mock, guest_session_id, is_public, gps_lat, gps_lng, gps_source) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (safe_name, comp, brgy, st, sector, batch_id, user.id if user else None, site_id, is_mock_flag, guest_session_id, is_public, lat_val, lng_val, "manual" if lat_val is not None else None)
            )
            
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    finally:
        conn.close()
        
    try:
        enqueue_batch(batch_id)
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
    
    from fastapi.responses import JSONResponse
    resp_data = {"batch_id": batch_id, "status": "queued", "items_count": len(saved_files)}
    if gps_warning:
        resp_data["warning"] = gps_warning
    resp_obj = JSONResponse(content=resp_data)
    if guest_session_id and guest_token:
        resp_obj.set_cookie(key="guest_session", value=guest_token, httponly=True, secure=cookie_secure(req), samesite="lax", max_age=86400, path="/")
    return resp_obj


@router.get("/{batch_id}")
async def get_batch(batch_id: int, req: Request, user: User | None = Depends(get_optional_user)):
    conn = get_connection()
    b = conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
    if not b:
        conn.close()
        raise HTTPException(status_code=404, detail="Batch not found")
        
    try:
        verify_analysis_access(b, user, req)
    except HTTPException as e:
        conn.close()
        raise e
        

        
    items = conn.execute("SELECT id, status FROM analyses WHERE batch_id = ? ORDER BY id ASC", (batch_id,)).fetchall()
    
    progress = {}
    analyses = []
    for row in items:
        status = row["status"]
        progress[status] = progress.get(status, 0) + 1
        analyses.append({"id": row["id"], "status": status})
    
    conn.close()
    return {
        "id": b["id"],
        "status": b["status"],
        "cancel_requested": bool(b["cancel_requested"]),
        "created_at": b["created_at"],
        "progress": progress,
        "analyses": analyses
    }


@router.post("/{batch_id}/cancel", dependencies=[Depends(verify_csrf)])
async def cancel_batch(batch_id: int, req: Request, user: User | None = Depends(get_optional_user)):
    conn = get_connection()

    b = conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
    if not b:
        conn.close()
        raise HTTPException(status_code=404, detail="Batch not found")
        
    try:
        verify_analysis_access(b, user, req)
    except HTTPException as e:
        conn.close()
        raise e
        
    if b["status"] in ("done", "interrupted"):
        conn.close()
        return {"status": b["status"], "msg": "Already finished or interrupted"}
        
    conn.execute("UPDATE batches SET cancel_requested = 1 WHERE id = ?", (batch_id,))
    conn.commit()
    conn.close()
    
    return {"status": "cancel_requested"}
