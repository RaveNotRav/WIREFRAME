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
    Query,
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
from backend.utils.uploads import generate_safe_filename

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
    from PIL.Image import DecompressionBombError
    try:
        img = Image.open(file.file)
        img.verify()
        file.file.seek(0)
    except DecompressionBombError:
        raise HTTPException(status_code=400, detail="Image exceeds decompression pixel limit")
    except (IOError, SyntaxError, UnidentifiedImageError) as e:
        import logging
        logging.getLogger(__name__).error(f"Invalid image content detected: {e}")
        raise HTTPException(status_code=400, detail="Corrupt image")

    Path("uploads").mkdir(exist_ok=True)
    try:
        safe_name = generate_safe_filename(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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
            secure=(req.url.scheme == "https"),
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
    include_invalid: bool = Query(False, description="Include rejected and no_wires submissions"),
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
            
        allow_invalid = include_invalid and (user.role in ("admin", "lgu_engineer"))
        if not allow_invalid:
            if status:
                if status in ("rejected", "no_wires"):
                    conds.append("1 = 0")
                else:
                    conds.append("status = ?")
                    params.append(status)
            else:
                conds.append("status NOT IN ('rejected', 'no_wires')")
        else:
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
    a = conn.execute("SELECT created_by, guest_session_id, is_public FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
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
    a = conn.execute("SELECT created_by, guest_session_id, is_public FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
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
    a = conn.execute(f"SELECT {column}, created_by, guest_session_id, is_public FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
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
    a = conn.execute("SELECT filename, created_by, guest_session_id, is_public FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
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


import json
import logging
import math
from pydantic import BaseModel

from backend.config_store import get_setting
from backend.utils.cv.types import normalize_class_name
from backend.utils.clearance_helpers import (
    determine_overall_safety,
    evaluate_wire,
    get_lowest_y,
    parse_centerline_json,
    parse_params_json,
    rollup_analysis_clearance,
)


class Point(BaseModel):
    x: float
    y: float


class CalibratePayload(BaseModel):
    points: list[Point]
    real_length_m: float


@router.post("/{analysis_id}/calibrate", dependencies=[Depends(verify_csrf)])
async def calibrate_analysis(
    analysis_id: int,
    payload: CalibratePayload,
    req: Request,
    user: User = Depends(get_current_user),
):
    # Validate payload values
    if not math.isfinite(payload.real_length_m):
        raise HTTPException(status_code=422, detail="Real length must be a finite number")
    if payload.real_length_m < 0.1 or payload.real_length_m > 100.0:
        raise HTTPException(status_code=422, detail="Real length must be between 0.1 and 100 meters")

    if len(payload.points) != 2:
        raise HTTPException(status_code=422, detail="Exactly 2 calibration points required")

    p1, p2 = payload.points[0], payload.points[1]
    if not (math.isfinite(p1.x) and math.isfinite(p1.y) and math.isfinite(p2.x) and math.isfinite(p2.y)):
        raise HTTPException(status_code=422, detail="Points must have finite coordinates")

    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dist = math.hypot(dx, dy)
    if dist < 10.0:
        raise HTTPException(status_code=422, detail="Reference points distance must be at least 10 pixels")

    conn = get_connection()
    try:
        a = conn.execute(
            """SELECT status, is_public, created_by, guest_session_id, width, height, pole_bbox_json,
                      assumed_pole_height_m, sag_meters, clearance_available, clearance_ok, margin_error_m
               FROM analyses WHERE id = ?""",
            (analysis_id,),
        ).fetchone()
        if not a:
            raise HTTPException(status_code=404, detail="Not found")

        verify_analysis_access(a, user, req)

        if a["status"] != "complete":
            raise HTTPException(status_code=409, detail="Analysis must be complete to calibrate")

        w, h = a["width"], a["height"]
        if a["pole_bbox_json"] is None or w is None or h is None:
            raise HTTPException(
                status_code=409,
                detail="This analysis has no stored ground reference; re-analyze the image to enable calibration.",
            )

        try:
            pole_bbox = json.loads(a["pole_bbox_json"])
            if not isinstance(pole_bbox, list) or len(pole_bbox) < 4:
                raise ValueError()
        except Exception:
            raise HTTPException(
                status_code=409,
                detail="This analysis has no stored ground reference; re-analyze the image to enable calibration.",
            )

        for p in [p1, p2]:
            if p.x < 0 or p.x > w or p.y < 0 or p.y > h:
                raise HTTPException(status_code=422, detail="Points must be within image bounds")

        wires = conn.execute(
            "SELECT * FROM wires WHERE analysis_id = ? ORDER BY id",
            (analysis_id,),
        ).fetchall()
        if not wires:
            raise HTTPException(status_code=409, detail="No wires found to calibrate")

        meters_per_px = payload.real_length_m / dist

        raw_pec = get_setting("pec_min_clearance_by_class", {}) or {}
        pec_min_map = {normalize_class_name(k): v for k, v in raw_pec.items()}
        margin_pct = get_setting("margin_of_error_pct", 20) / 100.0
        pole_height = a["assumed_pole_height_m"] or 10.0
        max_allowed_clearance = pole_height * 2.0

        evaluated_wires = []
        for wire_row in wires:
            polyline = parse_centerline_json(wire_row["centerline_json"])
            lowest_y = get_lowest_y(polyline)
            params = parse_params_json(wire_row["params_json"])

            ew = evaluate_wire(
                class_name=wire_row["class_name"],
                confidence=wire_row["confidence"] or 0.0,
                sag_px=wire_row["sag_px"] or 0.0,
                span_px=wire_row["span_px"] or 0.0,
                crs_weighted=wire_row["crs"] or 0.0,
                lowest_y=lowest_y,
                pole_bbox=pole_bbox,
                meters_per_px=meters_per_px,
                image_height=h,
                pec_min_map=pec_min_map,
                params_dict=params,
            )
            # Plausibility check per wire
            if ew["clearance_m"] is not None:
                if ew["clearance_m"] < 0.0 or ew["clearance_m"] > max_allowed_clearance:
                    raise HTTPException(status_code=422, detail="Calibration resulted in implausible clearance")
            ew["id"] = wire_row["id"]
            evaluated_wires.append(ew)

        rollup = rollup_analysis_clearance(evaluated_wires, pec_min_map, margin_pct)
        if rollup["clearance_available"] is not None:
            if rollup["clearance_available"] < 0.0 or rollup["clearance_available"] > max_allowed_clearance:
                raise HTTPException(status_code=422, detail="Calibration resulted in implausible clearance")

        overall_safety = determine_overall_safety([ew["safety_level"] for ew in evaluated_wires])

        # Audit payload with previous values
        previous_state = {
            "sag_meters": a["sag_meters"],
            "clearance_available": a["clearance_available"],
            "clearance_ok": a["clearance_ok"],
            "margin_error_m": a["margin_error_m"],
            "wires": [
                {
                    "id": w["id"],
                    "sag_m": w["sag_m"],
                    "clearance_m": w["clearance_m"],
                    "clearance_status": w["clearance_status"],
                    "safety_level": w["safety_level"],
                    "params_json": w["params_json"],
                }
                for w in wires
            ],
        }
        payload_audit = {
            "points": [{"x": p.x, "y": p.y} for p in payload.points],
            "real_length_m": payload.real_length_m,
            "meters_per_px": meters_per_px,
            "previous_state": previous_state,
        }

        # Single transaction for database modifications
        conn.execute(
            """INSERT INTO calibrations (analysis_id, created_by, payload_json, sag_m, clearance_m, safety_level) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (analysis_id, user.id, json.dumps(payload_audit), rollup["sag_meters"], rollup["clearance_available"], overall_safety),
        )

        for ew in evaluated_wires:
            conn.execute(
                """UPDATE wires
                   SET sag_m = ?, clearance_m = ?, clearance_status = ?, safety_level = ?, params_json = ?
                   WHERE id = ?""",
                (ew["sag_m"], ew["clearance_m"], ew["clearance_status"], ew["safety_level"], ew["params_json"], ew["id"]),
            )

        conn.execute(
            """UPDATE analyses
               SET sag_meters = ?, clearance_available = ?, clearance_ok = ?, pec_min_clearance_m = ?, margin_error_m = ?
               WHERE id = ?""",
            (
                rollup["sag_meters"],
                rollup["clearance_available"],
                rollup["clearance_ok"],
                rollup["pec_min_clearance_m"],
                rollup["margin_error_m"],
                analysis_id,
            ),
        )

        reports_to_delete = conn.execute("SELECT path FROM reports WHERE analysis_id = ?", (analysis_id,)).fetchall()
        conn.execute("DELETE FROM reports WHERE analysis_id = ?", (analysis_id,))

        conn.commit()

        # Invalidate report cache files safely post-commit
        base_outputs = Path("outputs").resolve()
        for rep in reports_to_delete:
            if rep["path"]:
                rep_path = Path(rep["path"]).resolve()
                if rep_path.is_relative_to(base_outputs) and rep_path.exists():
                    try:
                        rep_path.unlink()
                    except OSError as e:
                        logging.getLogger(__name__).warning("Failed to delete report %s: %s", rep_path, e)

        return {
            "status": "success",
            "sag_m": rollup["sag_meters"],
            "clearance_m": rollup["clearance_available"],
            "safety_level": overall_safety,
            "margin_error_m": rollup["margin_error_m"],
        }
    finally:
        conn.close()
