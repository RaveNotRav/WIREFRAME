import secrets
import os
import shutil
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse
from PIL import Image

from backend import main_state
from backend.auth import (
    check_anon_rate_limit,
    _hash_token,
    get_optional_user,
    verify_analysis_access,
    User,
    check_rate_limit,
    get_current_user,
    verify_csrf,
)
from backend.database import get_connection
from backend.services.sites import get_or_create_site, validate_location
from backend.services.worker import enqueue_batch

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

def extract_gps(img: Image.Image):
    try:
        exif = img._getexif()
        if not exif:
            return None, None
            
        # 34853 is GPSInfo
        gps_info = exif.get(34853)
        if not gps_info:
            return None, None
            
        def get_decimal(tag_val, ref):
            if not tag_val:
                return None
            d, m, s = tag_val
            val = float(d) + float(m)/60.0 + float(s)/3600.0
            if ref in ['S', 'W']:
                val = -val
            return val
            
        lat = get_decimal(gps_info.get(2), gps_info.get(1))
        lng = get_decimal(gps_info.get(4), gps_info.get(3))
        
        return lat, lng
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to extract GPS: {e}")
        return None, None

@router.post("/upload", dependencies=[Depends(verify_csrf)])
async def upload_image(
    req: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    company: str = Form(None, max_length=200),
    barangay: str = Form(None, max_length=200),
    street: str = Form(None, max_length=200),
    sector: str = Form(None, max_length=100),
    notes: str = Form(None, max_length=2000),
    gps_lat: str = Form(None, max_length=50),
    gps_lng: str = Form(None, max_length=50),
    user: User | None = Depends(get_optional_user)
):
    guest_session_id = None
    guest_token = None
    if user is None:
        
        guest_token = req.cookies.get("guest_session")
        if not guest_token:
            guest_token = secrets.token_urlsafe(32)
        guest_session_id = _hash_token(guest_token)
    else:
        check_rate_limit(req, "upload", user.username, limit=30, window=60)

    if main_state.is_config_failed():
        raise HTTPException(status_code=503, detail="Server configuration is invalid. Cannot process uploads.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Size limit
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")
        
    # Content verification
    from PIL import Image, UnidentifiedImageError
    try:
        img = Image.open(file.file)
        img.verify()
        file.file.seek(0)
    except (IOError, SyntaxError, UnidentifiedImageError) as e:
        import logging
        logging.getLogger(__name__).error(f"Invalid image content detected: {e}")
        raise HTTPException(status_code=400, detail="Corrupt image")

    Path("uploads").mkdir(exist_ok=True)
    import re
    # Sanitize filename to prevent path traversal
    safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', os.path.basename(file.filename))
    if not safe_name:
        safe_name = "upload" + ext
    out_path = os.path.join("uploads", safe_name)

    with open(out_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    comp, brgy, st = validate_location(company, barangay, street)

    def parse_gps(val: str, name: str) -> float | None:
        if not val:
            return None
        try:
            f = float(val)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid {name}")
        return f
        
    parsed_lat = parse_gps(gps_lat, "latitude")
    parsed_lng = parse_gps(gps_lng, "longitude")

    if (parsed_lat is None) != (parsed_lng is None):
        raise HTTPException(status_code=422, detail="Both latitude and longitude must be provided, or neither")

    gps_source = None
    width, height = None, None
    try:
        from PIL import ImageOps
        with Image.open(out_path) as img:
            img.verify()
            
        with Image.open(out_path) as img:
            if parsed_lat is None or parsed_lng is None:
                exif_lat, exif_lng = extract_gps(img)
                if exif_lat is not None and exif_lng is not None:
                    parsed_lat = exif_lat
                    parsed_lng = exif_lng
                    gps_source = 'exif'
            elif parsed_lat is not None and parsed_lng is not None:
                gps_source = 'manual'
            
            img_t = ImageOps.exif_transpose(img)
            width, height = img_t.size
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Image processing error for {out_path}: {e}")
        os.remove(out_path)
        raise HTTPException(status_code=400, detail="Corrupt image")

    if parsed_lat is not None and (not (-90 <= parsed_lat <= 90)):
        os.remove(out_path)
        raise HTTPException(status_code=422, detail="Invalid latitude")
    if parsed_lng is not None and (not (-180 <= parsed_lng <= 180)):
        os.remove(out_path)
        raise HTTPException(status_code=422, detail="Invalid longitude")

    conn = get_connection()
    try:
        site_id = get_or_create_site(conn, comp, brgy, st, parsed_lat, parsed_lng)
        
        cur = conn.execute("INSERT INTO batches (status, created_by, guest_session_id, is_public) VALUES ('queued', ?, ?, ?)", (user.id if user else None, guest_session_id, 1 if user is None else 0))
        batch_id = cur.lastrowid
        
        is_mock_flag = 1 if os.getenv("WIREFRAME_MOCK") == "1" else 0
        is_public = 1 if user is None else 0
        cur = conn.execute(
            """INSERT INTO analyses 
               (filename, status, company, barangay, street, sector, notes, batch_id, created_by, site_id, gps_lat, gps_lng, gps_source, width, height, is_mock, guest_session_id, is_public) 
               VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (safe_name, comp, brgy, st, sector, notes, batch_id, user.id if user else None, site_id, parsed_lat, parsed_lng, gps_source, width, height, is_mock_flag, guest_session_id, is_public)
        )
        analysis_id = cur.lastrowid
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    finally:
        conn.close()

    try:
        enqueue_batch(batch_id)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to enqueue batch {batch_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

    res_body = {"analysis_id": analysis_id, "status": "queued"}
    
    if parsed_lat is not None and parsed_lng is not None:
        if not (4.5 <= parsed_lat <= 21.5) or not (114.0 <= parsed_lng <= 127.0):
            res_body["warning"] = "GPS coordinates are outside the Philippines."
            
    if guest_token and guest_token != req.cookies.get("guest_session"):
        response.set_cookie(
            "guest_session", 
            guest_token, 
            httponly=True, 
            secure=False, # TODO: Change to True before HTTPS deployment
            samesite="lax",
            max_age=86400,
            path="/"
        )
        
    return res_body


@router.get("")
async def get_analyses(
    page: int = 1,
    limit: int = 20,
    status: str = None,
    is_mock: int = None,
    is_public: int = None,
    company: str = None,
    barangay: str = None,
    street: str = None,
    sort_by: str = "created_at",
    sort_dir: str = "desc",
    user: User = Depends(get_current_user)
):
    conn = get_connection()
    try:
        query = "SELECT * FROM analyses"
        params = []
        conds = []
        
        if user.role == "field_inspector":
            if is_public == 1:
                # Field inspectors shouldn't see public tips generally, but if they explicitly ask, we deny or return empty
                conds.append("created_by = -1") # impossible condition
            else:
                conds.append("created_by = ?")
                params.append(user.id)
                if is_public is not None:
                    conds.append("is_public = ?")
                    params.append(is_public)
        else:
            # LGU / Admin
            if is_public is not None:
                conds.append("is_public = ?")
                params.append(is_public)
            
        if status:
            conds.append("status = ?")
            params.append(status)
            
        if is_mock is not None:
            conds.append("is_mock = ?")
            params.append(is_mock)
            
        def add_like_filter(col_name: str, val: str):
            if val:
                val = val.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                conds.append(f"{col_name} LIKE ? ESCAPE '\\'")
                params.append(f"%{val}%")

        add_like_filter("company", company)
        add_like_filter("barangay", barangay)
        add_like_filter("street", street)
            
        if conds:
            query += " WHERE " + " AND ".join(conds)
            
        if sort_by not in ["created_at", "id", "risk_score"]:
            sort_by = "created_at"
        if sort_dir.lower() not in ["asc", "desc"]:
            sort_dir = "desc"
            
        query += f" ORDER BY {sort_by} {sort_dir}"
        
        # count total
        count_query = "SELECT count(*) as c FROM analyses"
        if conds:
            count_query += " WHERE " + " AND ".join(conds)
        total = conn.execute(count_query, params).fetchone()["c"]
        
        offset = (page - 1) * limit
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        items = conn.execute(query, params).fetchall()
        
        # Get thumbnail names (we assume thumb_{filename})
        results = []
        for item in items:
            d = dict(item)
            # Add thumbnail URL logic if needed, or frontend constructs it
            results.append(d)
            
        return {
            "data": results,
            "total": total,
            "page": page,
            "limit": limit
        }
    finally:
        conn.close()

@router.delete("/{analysis_id}", dependencies=[Depends(verify_csrf)])
async def delete_analysis(analysis_id: int, user: User = Depends(get_current_user)):
    conn = get_connection()
    try:
        a = conn.execute("SELECT created_by, original_path, annotated_path, spectrum_path, filename FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        if not a:
            raise HTTPException(status_code=404, detail="Not found")
            
        if user.role != "admin" and (a["created_by"] is None or a["created_by"] != user.id):
            raise HTTPException(status_code=403, detail="Forbidden")
            
        rep = conn.execute("SELECT path FROM reports WHERE analysis_id = ?", (analysis_id,)).fetchone()
        
        # Delete files
        paths = [
            a["original_path"], 
            a["annotated_path"], 
            a["spectrum_path"], 
            os.path.join("uploads", a["filename"]) if a["filename"] else None,
            os.path.join("uploads", "thumb_" + a["filename"]) if a["filename"] else None
        ]
        
        if rep and rep["path"]:
            # Only delete paths inside outputs/
            rep_path = Path(rep["path"]).resolve()
            out_path = Path("outputs").resolve()
            if str(rep_path).startswith(str(out_path)):
                paths.append(str(rep_path))
        
        for p in paths:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Failed to delete file {p}: {e}")
                    
        conn.execute("DELETE FROM reports WHERE analysis_id = ?", (analysis_id,))
        conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        conn.commit()
        
        return {"status": "success"}
    finally:
        conn.close()

@router.get("/{analysis_id}/events", dependencies=[Depends(verify_csrf)])
async def get_events(analysis_id: int, req: Request, user: User | None = Depends(get_optional_user)):
    conn = get_connection()
    a = conn.execute("SELECT created_by, guest_session_id FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    if not a:
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")
        
    try:
        verify_analysis_access(a, user, req)
    except HTTPException as e:
        conn.close()
        raise e
        
    events = conn.execute("SELECT state, ts, detail FROM pipeline_events WHERE analysis_id = ? ORDER BY id ASC", (analysis_id,)).fetchall()
    conn.close()
    return [dict(e) for e in events]


@router.get("/{analysis_id}/wires")
async def get_wires(analysis_id: int, req: Request, user: User | None = Depends(get_optional_user)):
    conn = get_connection()
    a = conn.execute("SELECT created_by, guest_session_id FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    if not a:
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")
    try:
        verify_analysis_access(a, user, req)
    except HTTPException as e:
        conn.close()
        raise e
        
    wires = conn.execute("SELECT * FROM wires WHERE analysis_id = ?", (analysis_id,)).fetchall()
    conn.close()
    return [dict(w) for w in wires]

def serve_image(analysis_id: int, column: str, user: User | None, req: Request):
    conn = get_connection()
    a = conn.execute(f"SELECT {column}, created_by, guest_session_id FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    conn.close()
    
    if not a:
        raise HTTPException(status_code=404, detail="Not found")
    verify_analysis_access(a, user, req)
        
    path_str = a[column]
    if not path_str:
        raise HTTPException(status_code=404, detail="Image not available")
        
    p = Path(path_str).resolve()
    base_uploads = Path("uploads").resolve()
    base_outputs = Path("outputs").resolve()
    
    if not str(p).startswith(str(base_uploads)) and not str(p).startswith(str(base_outputs)):
        raise HTTPException(status_code=403, detail="Path traversal detected")
        
    if not p.is_file():
        raise HTTPException(status_code=404, detail="File does not exist")
        
    return FileResponse(str(p))

@router.get("/{analysis_id}/original")
async def get_original(analysis_id: int, req: Request, user: User | None = Depends(get_optional_user)):
    # original_path wasn't explicitly populated in the old pipeline, but if it is, serve it.
    # We can fallback to 'uploads/' + filename
    conn = get_connection()
    a = conn.execute("SELECT filename, created_by FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    conn.close()
    
    if not a:
        raise HTTPException(status_code=404, detail="Not found")
    verify_analysis_access(a, user, req)
        
    p = Path("uploads") / a["filename"]
    p = p.resolve()
    base_uploads = Path("uploads").resolve()
    if not str(p).startswith(str(base_uploads)):
        raise HTTPException(status_code=403, detail="Path traversal detected")
        
    return FileResponse(str(p))

@router.get("/{analysis_id}/overlay")
async def get_overlay(analysis_id: int, req: Request, user: User | None = Depends(get_optional_user)):
    return serve_image(analysis_id, "annotated_path", user, req)

@router.get("/{analysis_id}/annotated")
async def get_annotated(analysis_id: int, req: Request, user: User | None = Depends(get_optional_user)):
    return serve_image(analysis_id, "annotated_path", user, req)

@router.get("/{analysis_id}/spectrum")
async def get_spectrum(analysis_id: int, req: Request, user: User | None = Depends(get_optional_user)):
    return serve_image(analysis_id, "spectrum_path", user, req)


from pydantic import BaseModel


class CalibratePayload(BaseModel):
    p1: list[float]
    p2: list[float]
    real_length_m: float
    # Can add more

@router.post("/{analysis_id}/calibrate", dependencies=[Depends(verify_csrf)])
async def calibrate_analysis(analysis_id: int, payload: CalibratePayload, req: Request, user: User | None = Depends(get_optional_user)):
    conn = get_connection()
    a = conn.execute("SELECT status, created_by, guest_session_id, width, height FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    if not a:
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")
        
    if a["status"] != "complete":
        conn.close()
        raise HTTPException(status_code=400, detail="Analysis must be complete to calibrate")
        
    try:
        verify_analysis_access(a, user, req)
    except HTTPException as e:
        conn.close()
        raise e
        
    if payload.real_length_m <= 0 or payload.real_length_m > 1000:
        conn.close()
        raise HTTPException(status_code=422, detail="Invalid real length")
        
    w, h = a["width"], a["height"]
    if w and h:
        for p in [payload.p1, payload.p2]:
            if p[0] < 0 or p[0] > w or p[1] < 0 or p[1] > h:
                conn.close()
                raise HTTPException(status_code=422, detail="Points must be within image bounds")

    # In a real implementation we would recompute using the math functions.
    # For now, just save it to calibrations and return dummy updated values 
    # to fulfill the endpoint contract in this mock phase.
    try:
        conn.execute(
            """INSERT INTO calibrations (analysis_id, created_by, payload_json, sag_m, clearance_m, safety_level) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (analysis_id, user.id if user else None, payload.model_dump_json(), 0.5, 5.0, "Safe")
        )
        
        # Invalidate report cache
        rep = conn.execute("SELECT path FROM reports WHERE analysis_id = ?", (analysis_id,)).fetchone()
        if rep and rep["path"]:
            rep_path = Path(rep["path"]).resolve()
            out_path = Path("outputs").resolve()
            if str(rep_path).startswith(str(out_path)) and os.path.exists(str(rep_path)):
                try:
                    os.remove(str(rep_path))
                except OSError as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Failed to delete report {rep_path}: {e}")
        conn.execute("DELETE FROM reports WHERE analysis_id = ?", (analysis_id,))
        
        conn.commit()
    finally:
        conn.close()
        
    return {"status": "success", "sag_m": 0.5, "clearance_m": 5.0, "safety_level": "Safe"}
