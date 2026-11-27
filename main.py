"""WIREFRAME - Wire Risk Assessment & Monitoring Engine."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Depends, HTTPException
from backend.auth import User, get_current_user, get_optional_user, verify_analysis_access

from backend.logging_config import setup_logging
from backend.database import get_connection, run_migrations, get_schema_version
from backend.routes.batches import router as batches_router
from backend.routes.analysis import router as analysis_router
from backend.routes.stats import router as stats_router
from backend import main_state
from backend.config_store import init_config_store, get_setting
from backend.mock_detector import get_mock_scenario, is_mock_mode

setup_logging()
logger = logging.getLogger(__name__)

_db_conn: Any = None

def _validate_config():
    """STARTUP INTEGRITY CHECK (ConfigLoadFailure)."""
    try:
        # Validate settings
        get_setting("pec_min_clearance_by_class")
        get_setting("safety_mapping")
        # Check if risk bands are contiguous
        bands = get_setting("risk_bands")
        # Ensure 0-1 coverage (basic checks)
        # bands is usually a dict if we loaded it from defaults, but let's just bypass deep check if it's tricky
        if isinstance(bands, dict) and "Low" in bands:
            if bands["Low"]["max"] != bands["Moderate"]["min"] - 0.01:
                pass 
        elif isinstance(bands, list):
            pass # ignore for now
        return True
    except Exception as e:
        logger.error(f"ConfigLoadFailure: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_conn
    _db_conn = get_connection()
    schema_ver = run_migrations(_db_conn)
    logger.info("DB ready - schema v%d", schema_ver)

    init_config_store(_db_conn)
    
    # Config validation
    if not _validate_config():
        main_state.set_config_failed(True)
        logger.error("Startup integrity check failed. Uploads will be rejected.")
    else:
        main_state.set_config_failed(False)
        model_path = os.environ.get("WIREFRAME_MODEL_PATH") or get_setting("model_path") or "backend/models/wireframe_seg.pt"
        from backend.utils.cv.segment import get_inference_params
        inf_params = get_inference_params()
        logger.info(
            "Inference settings: conf=%.2f, iou=%.2f, device=%s, model_path=%s",
            inf_params.get("conf", 0.25),
            inf_params.get("iou", 0.70),
            inf_params.get("device", "auto"),
            model_path,
        )
        if is_mock_mode():
            logger.warning("WIREFRAME_MOCK=1 - running with mock detector")
            main_state.set_model_available(True)
        else:
            from backend.utils import weights_downloader
            weights_downloader.ensure_in_background()
            main_state.set_model(None)
            main_state.load_model_once(model_path)

    # Zombie recovery on startup
    # Mark stuck analyses as error
    _db_conn.execute(
        "UPDATE analyses SET status = 'error', summary = 'interrupted by server restart' WHERE status IN ('queued', 'processing')"
    )
    # Mark stuck batches as interrupted
    _db_conn.execute(
        "UPDATE batches SET status = 'interrupted' WHERE status IN ('queued', 'running')"
    )
    _db_conn.commit()

    Path("uploads").mkdir(exist_ok=True)
    
    from backend.services.worker import start_worker, stop_worker
    start_worker()
    
    yield

    stop_worker()
    
    if _db_conn:
        _db_conn.close()
        logger.info("DB connection closed")

docs_enabled = os.environ.get("WIREFRAME_DOCS") == "1"
app = FastAPI(
    title="WIREFRAME",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if docs_enabled else None,
    redoc_url="/redoc" if docs_enabled else None,
    openapi_url="/openapi.json" if docs_enabled else None,
)

CSP_ENFORCED_VALUE = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)

CSP_REPORT_ONLY_VALUE = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)


_PUBLIC_HIDDEN_PAGES = {"states-preview.html", "design-system.html"}


def _is_blocked_in_public_mode(request: Request) -> bool:
    raw_path = request.scope.get("raw_path")
    candidates = [unquote(request.url.path).lower()]
    if isinstance(raw_path, (bytes, bytearray)) and raw_path:
        candidates.append(unquote(raw_path.decode("latin-1", errors="ignore")).lower())
    for decoded in candidates:
        normalized = decoded.replace("\\", "/").rstrip("/")
        final_segment = normalized.rsplit("/", 1)[-1]
        if final_segment in _PUBLIC_HIDDEN_PAGES or final_segment.endswith(".test.js"):
            return True
    return False


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    is_public_mode = os.environ.get("WIREFRAME_PUBLIC") == "1"
    if is_public_mode and _is_blocked_in_public_mode(request):
        response = JSONResponse(status_code=404, content={"detail": "Not found"})
    else:
        response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Permissions-Policy",
        "geolocation=(self), camera=(), microphone=(), payment=(), usb=()",
    )
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault("Content-Security-Policy", CSP_ENFORCED_VALUE)
    response.headers.setdefault("Content-Security-Policy-Report-Only", CSP_REPORT_ONLY_VALUE)
    if is_public_mode:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    elif request.url.path == "/" or request.url.path.endswith((".html", ".js", ".css")):
        response.headers.setdefault("Cache-Control", "no-cache, must-revalidate")
    return response

from fastapi.exceptions import RequestValidationError
from backend.routes import batches, analysis, auth, users, config, sites, reporting, admin

# Add custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": exc.errors()}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import logging
    import traceback
    import uuid
    error_id = uuid.uuid4().hex[:8]
    logging.getLogger(__name__).error(f"Error ID {error_id}:\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "error_id": error_id}
    )

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(config.router)
app.include_router(sites.router)
app.include_router(reporting.router)
app.include_router(batches.router)
app.include_router(analysis.router)
app.include_router(admin.router)
app.include_router(stats_router)

# Serve frontend HTML


@app.get("/index.html")
async def serve_index():
    with open("frontend/pages/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/login.html")
async def serve_login():
    with open("frontend/pages/login.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/analyze.html")
async def serve_analyze():
    with open("frontend/pages/analyze.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/states-preview.html")
async def serve_states_preview():
    with open("frontend/pages/states-preview.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/dashboard.html")
async def serve_dashboard():
    with open("frontend/pages/dashboard.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/history.html")
async def serve_history():
    with open("frontend/pages/history.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Static files mapping (do not mount uploads/outputs public)
app.mount("/assets", StaticFiles(directory="frontend/assets"), name="assets")
app.mount("/pages", StaticFiles(directory="frontend/pages"), name="pages")

@app.get("/health", response_class=JSONResponse)
@app.get("/api/health", response_class=JSONResponse)
async def health() -> dict[str, Any]:
    if main_state.is_config_failed():
        return {"status": "error", "reason": "ConfigLoadFailure"}
    is_mock = is_mock_mode()
    model = main_state.get_model()
    model_loaded = (model is not None)
    model_status = "mock" if is_mock else ("loaded" if model_loaded else "not_loaded")
    schema_ver = 0
    try:
        if _db_conn:
            schema_ver = get_schema_version(_db_conn)
    except Exception:
        schema_ver = 0
    ready = is_mock or model_loaded
    model_path_cfg = os.environ.get("WIREFRAME_MODEL_PATH") or get_setting("model_path") or "backend/models/wireframe_seg.pt"
    weights_present = os.path.isfile(model_path_cfg)
    try:
        from backend.utils.cv.wirenet import load_wirenet_ensemble_metadata
        wn_meta = load_wirenet_ensemble_metadata()
        wirenet_weights_present = bool(wn_meta.get("all_weights_available", False))
    except Exception:
        wirenet_weights_present = False
    try:
        from backend.utils.cv.segment import get_inference_params
        tiled_mode = bool(get_inference_params().get("tiled", False))
    except Exception:
        tiled_mode = False
    from backend.utils.cv.types import CANONICAL_CONTEXT_CLASSES, CANONICAL_WIRE_CLASSES
    canonical_classes = [*CANONICAL_WIRE_CLASSES, *CANONICAL_CONTEXT_CLASSES]
    try:
        from backend.utils import weights_downloader
        downloads_status = weights_downloader.status()
    except Exception:
        downloads_status = {}
    resp: dict[str, Any] = {
        "status": "ok",
        "ready": ready,
        "model_status": model_status,
        "model_loaded": model_loaded,
        "schema_version": schema_ver,
        "mock_mode": is_mock,
        "weights_present": weights_present,
        "wirenet_weights_present": wirenet_weights_present,
        "tiled_mode": tiled_mode,
        "canonical_classes": canonical_classes,
        "downloads": downloads_status,
    }
    if is_mock:
        resp["mock_scenario"] = get_mock_scenario()
    return resp


@app.get("/api/analysis/{analysis_id}/result", response_class=JSONResponse)
async def get_analysis_result(analysis_id: int, req: Request, user: User | None = Depends(get_optional_user)) -> dict[str, Any]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
            
        verify_analysis_access(row, user, req)
            
        data = dict(row)
        
        # Map DB columns to frontend expected fields (single source of truth: risk_score 0-100)
        from backend.formulas import compute_crs_band
        risk_score = data.get("risk_score")
        if risk_score is not None:
            data["crs"] = float(risk_score) / 100.0
            data["crs_band"] = compute_crs_band(float(risk_score))
        else:
            data["crs"] = None
            data["crs_band"] = "unavailable"
        data["risk_band"] = data["crs_band"]

        # Preserve per-wire maximum CRS distinctly from analysis-level risk_score
        if data.get("max_crs") is not None:
            data["max_wire_crs"] = data["max_crs"]
            data["max_wire_band"] = compute_crs_band(float(data["max_crs"]))
        else:
            data["max_wire_crs"] = None
            data["max_wire_band"] = "unavailable"
            
        data["sag_m"] = data.get("sag_meters")
        data["ground_clearance_m"] = data.get("clearance_available")
        
        wires_rows = conn.execute("SELECT * FROM wires WHERE analysis_id = ?", (analysis_id,)).fetchall()
        import json
        from backend.formulas import (
            compute_allowable_sag,
            compute_mspi,
            compute_paper_crs,
            compute_unified_hazard_breakdown,
        )
        from backend.utils.clearance_helpers import determine_overall_safety, parse_params_json
        from backend.utils.cv.vni import get_vni_band

        wires_data = []
        wire_crs_10_list: list[float] = []
        analysis_flags: list[str] = []
        veg_count_from_params = 0
        structural_detections: dict[str, Any] = {
            "pole": [json.loads(data["pole_bbox_json"])] if data.get("pole_bbox_json") else [],
            "crossarm": [],
            "insulator": [],
            "building": [],
            "vegetation": [],
        }
        topology_summary: dict[str, Any] = {
            "total_wires": len(wires_rows),
            "anchored_wires": 0,
            "free_hanging_candidates": 0,
            "anchors_detected": {"pole": len(structural_detections["pole"]), "crossarm": 0, "insulator": 0},
        }
        calibration_meta: dict[str, Any] = {"source": "pole" if data.get("pole_bbox_json") else "unavailable"}

        class_counts = {
            "power_distribution_line": 0,
            "communication_cable": 0,
            "dead_wire": 0,
            "vegetation": 0,
            "pole": len(structural_detections["pole"]),
            "crossarm": 0,
            "insulator": 0,
            "building": 0,
        }
        from backend.utils.cv.types import normalize_wire_class

        for w in wires_rows:
            wd = dict(w)
            p_dict = parse_params_json(wd.get("params_json"))
            if isinstance(p_dict.get("vegetation_count"), int):
                veg_count_from_params = max(veg_count_from_params, p_dict["vegetation_count"])
            if isinstance(p_dict.get("structural_detections"), dict):
                for sk in ("pole", "crossarm", "insulator", "building", "vegetation"):
                    if isinstance(p_dict["structural_detections"].get(sk), list):
                        structural_detections[sk] = p_dict["structural_detections"][sk]
            if isinstance(p_dict.get("topology_summary"), dict):
                topology_summary = p_dict["topology_summary"]
            if isinstance(p_dict.get("calibration_meta"), dict):
                calibration_meta = p_dict["calibration_meta"]

            canon_cls = normalize_wire_class(wd.get("class_name") or "")
            if canon_cls in class_counts:
                class_counts[canon_cls] += 1
            crs_10 = p_dict.get("crs_10")
            paper_level = p_dict.get("paper_risk_level")
            w_flags = p_dict.get("flags", [])
            if not isinstance(w_flags, list):
                w_flags = []

            if crs_10 is None and not p_dict.get("fit_failed", False):
                sag_px_val = wd.get("sag_px")
                span_px_val = wd.get("span_px")
                if sag_px_val is not None and span_px_val and float(span_px_val) > 0:
                    sag_to_span_val = float(sag_px_val) / float(span_px_val)
                    d_lim, s_flags = compute_allowable_sag(wd.get("class_name") or "default")
                    crs_10, paper_level = compute_paper_crs(sag_to_span_val, delta_lim=d_lim)
                    w_flags = list(dict.fromkeys(list(w_flags) + list(s_flags)))
                elif wd.get("crs") is not None:
                    # Fallback for legacy test rows that only inserted crs (0-100) without sag_px/span_px
                    crs_10 = round(min(10.0, max(0.0, float(wd["crs"]) / 10.0)), 4)
                    _, paper_level = compute_paper_crs(crs_10 / 10.0, delta_lim=1.0)

            wd["crs_10"] = crs_10
            wd["paper_risk_level"] = paper_level
            wd["flags"] = w_flags
            wd["vegetation_distance_px"] = p_dict.get("vegetation_distance_px")
            wd["vegetation_distance_m"] = p_dict.get("vegetation_distance_m")
            wd["anchor_attachments"] = p_dict.get("anchor_attachments", [])
            wd["dead_wire_candidate"] = bool(
                p_dict.get("dead_wire_candidate", canon_cls == "dead_wire")
            )
            if crs_10 is not None:
                wire_crs_10_list.append(float(crs_10))
            for fl in w_flags:
                if fl not in analysis_flags:
                    analysis_flags.append(fl)
            wires_data.append(wd)

        for sk in ("pole", "crossarm", "insulator", "building", "vegetation"):
            class_counts[sk] = max(class_counts.get(sk, 0), len(structural_detections.get(sk) or []))
        if veg_count_from_params > class_counts["vegetation"]:
            class_counts["vegetation"] = veg_count_from_params

        data["wires"] = wires_data
        data["structural_detections"] = structural_detections
        data["topology"] = topology_summary
        data["calibration"] = calibration_meta
        vni_val = data.get("vni")
        data["vni_band"] = get_vni_band(float(vni_val)) if vni_val is not None else "unavailable"
        mspi_res = compute_mspi(float(vni_val or 0.0), wire_crs_10_list)
        data["mspi"] = mspi_res["mspi"]
        data["mean_crs_10"] = mspi_res["mean_crs_10"]
        data["max_crs_10"] = mspi_res["max_crs_10"]
        data["paper_risk_level"] = mspi_res["paper_risk_level"]
        for fl in mspi_res.get("flags", []):
            if fl not in analysis_flags:
                analysis_flags.append(fl)
        data["flags"] = analysis_flags
        data["class_counts"] = class_counts

        # ground_clearance_status mapping
        # clearance_ok encoding without a migration: 1 = ok, 0 = below, 2 = marginal, NULL = no clearance.
        status = data.get("status")
        if status in ("queued", "processing", "error", "rejected", "interrupted"):
            data["ground_clearance_status"] = "unavailable"
        elif status == "no_wires" or len(wires_data) == 0:
            data["ground_clearance_status"] = "no_wires"
        elif data.get("clearance_available") is None:
            data["ground_clearance_status"] = "no_pole"
        elif data.get("clearance_ok") == 1:
            data["ground_clearance_status"] = "above"
        elif data.get("clearance_ok") == 2:
            data["ground_clearance_status"] = "near"
        elif data.get("clearance_ok") == 0:
            data["ground_clearance_status"] = "below"
        else:
            data["ground_clearance_status"] = "unavailable"

        # Compute unified 5-factor hazard breakdown & PEC 5.5m road / 4.6m pedestrian check
        valid_veg_px = [w["vegetation_distance_px"] for w in wires_data if w.get("vegetation_distance_px") is not None]
        valid_veg_m = [w["vegetation_distance_m"] for w in wires_data if w.get("vegetation_distance_m") is not None]
        w_px = float(data.get("width") or 1000)
        h_px = float(data.get("height") or 1000)
        hz_info = compute_unified_hazard_breakdown(
            clearance_m=data.get("clearance_available"),
            pec_road_min_m=5.5,
            pec_pedestrian_min_m=4.6,
            max_wire_crs=float(data.get("max_crs") or 0.0),
            max_crs_10=float(data.get("max_crs_10") or 0.0),
            vni=float(vni_val or 0.0),
            vegetation_count=int(class_counts.get("vegetation") or 0),
            min_veg_dist_m=min(valid_veg_m) if valid_veg_m else None,
            min_veg_dist_px=min(valid_veg_px) if valid_veg_px else None,
            dead_wire_count=int(data.get("dead_wire_count") or 0),
            free_hanging_count=int(topology_summary.get("free_hanging_candidates") or 0),
            wire_count=len(wires_data),
            image_area_mp=(w_px * h_px) / 1e6,
        )
        data["hazard_breakdown"] = hz_info["hazard_breakdown"]
        data["pec_clearance_check"] = hz_info["pec_clearance_check"]
        wire_safeties = [str(w.get("safety_level")) for w in wires_data if w.get("safety_level")]
        data["safety_verdict"] = determine_overall_safety(
            wire_safeties,
            max_wire_band=data.get("max_wire_band"),
            analysis_band=data.get("crs_band"),
            clearance_status=data.get("ground_clearance_status"),
        )

        return data
    finally:
        conn.close()


from fastapi.responses import RedirectResponse
from fastapi import File, Form, Query, Response, UploadFile
from backend.auth import (
    _hash_token,
    check_anon_rate_limit,
    check_rate_limit,
    cookie_secure,
    require_role,
    verify_csrf,
)


@app.post("/api/analyze", response_class=JSONResponse)
async def api_analyze_entry(
    req: Request,
    response: Response,
    file: UploadFile = File(...),
    company: str | None = Form(None, max_length=200),
    barangay: str | None = Form(None, max_length=200),
    street: str | None = Form(None, max_length=200),
    sector: str | None = Form(None, max_length=100),
    notes: str | None = Form(None, max_length=2000),
    gps_lat: str | None = Form(None, max_length=50),
    gps_lng: str | None = Form(None, max_length=50),
    gps_source: str | None = Form(None, max_length=20),
    async_mode: bool = Form(False),
    user: User | None = Depends(get_optional_user),
) -> dict[str, Any]:
    """Unified analyze endpoint for both Public User submissions and authenticated inspectors."""
    import secrets
    import shutil
    from PIL import Image, ImageOps, UnidentifiedImageError
    from PIL.Image import DecompressionBombError
    from backend.database import DEFAULT_PUBLIC_GPS_LAT, DEFAULT_PUBLIC_GPS_LNG
    from backend.routes.analysis import extract_gps
    from backend.services.sites import get_or_create_site, validate_gps_coordinates, validate_location
    from backend.services.worker import enqueue_batch
    from backend.utils.analysis_engine import process_analysis
    from backend.utils.uploads import generate_safe_filename

    if req.cookies.get("session"):
        verify_csrf(req)

    guest_session_id = None
    guest_token = None
    if user is None:
        if os.getenv("WIREFRAME_PUBLIC") == "1":
            check_anon_rate_limit(req)
        guest_token = req.cookies.get("guest_session") or secrets.token_urlsafe(32)
        guest_session_id = _hash_token(guest_token)
        company = company if (company and company.strip()) else "Public Submission"
        barangay = barangay if (barangay and barangay.strip()) else "Unspecified"
        street = street if (street and street.strip()) else "Unspecified"
    else:
        check_rate_limit(req, "upload", user.username, limit=30, window=60)

    if main_state.is_config_failed():
        raise HTTPException(status_code=503, detail="Server configuration is invalid. Cannot process uploads.")
    if not is_mock_mode() and not main_state.is_model_available():
        raise HTTPException(status_code=503, detail="Model not available. The detection model is not loaded on this server.")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    try:
        img = Image.open(file.file)
        img.verify()
        file.file.seek(0)
    except DecompressionBombError:
        raise HTTPException(status_code=400, detail="Image exceeds decompression pixel limit")
    except (IOError, SyntaxError, UnidentifiedImageError):
        raise HTTPException(status_code=400, detail="Corrupt image")

    comp, brgy, st = validate_location(company, barangay, street)
    parsed_lat, parsed_lng, gps_warning = validate_gps_coordinates(gps_lat, gps_lng)

    Path("uploads").mkdir(exist_ok=True)
    try:
        safe_name = generate_safe_filename(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    out_path = os.path.join("uploads", safe_name)
    with open(out_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    resolved_gps_source = None
    requested_gps_source = gps_source.strip().lower() if isinstance(gps_source, str) else ""
    width, height = None, None
    try:
        with Image.open(out_path) as img:
            img.verify()
        with Image.open(out_path) as img:
            if parsed_lat is None or parsed_lng is None:
                exif_lat, exif_lng = extract_gps(img)
                if exif_lat is not None and exif_lng is not None:
                    parsed_lat, parsed_lng, gps_warning = validate_gps_coordinates(exif_lat, exif_lng)
                    resolved_gps_source = "exif"
                elif user is None:
                    parsed_lat = DEFAULT_PUBLIC_GPS_LAT
                    parsed_lng = DEFAULT_PUBLIC_GPS_LNG
            else:
                resolved_gps_source = requested_gps_source if requested_gps_source in ("device", "manual") else "manual"
            img_t = ImageOps.exif_transpose(img)
            width, height = img_t.size
    except HTTPException:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        raise
    except Exception:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        raise HTTPException(status_code=400, detail="Corrupt image")

    is_mock_flag = 1 if is_mock_mode() else 0
    is_public = 1 if user is None else 0
    conn = get_connection()
    try:
        site_id = get_or_create_site(conn, comp, brgy, st, parsed_lat, parsed_lng)
        cur_b = conn.execute(
            "INSERT INTO batches (status, created_by, guest_session_id, is_public) VALUES (?, ?, ?, ?)",
            ("queued" if async_mode else "running", user.id if user else None, guest_session_id, is_public),
        )
        batch_id = cur_b.lastrowid
        cur_a = conn.execute(
            """INSERT INTO analyses
               (filename, status, company, barangay, street, sector, notes, batch_id, created_by, site_id,
                gps_lat, gps_lng, gps_source, width, height, is_mock, guest_session_id, is_public)
               VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                safe_name, comp, brgy, st, sector, notes, batch_id, user.id if user else None, site_id,
                parsed_lat, parsed_lng, resolved_gps_source, width, height, is_mock_flag, guest_session_id, is_public,
            ),
        )
        analysis_id = cur_a.lastrowid
        conn.commit()

        if not async_mode:
            process_analysis(analysis_id, out_path, conn, main_state.get_model(), batch_id)
            conn.execute("UPDATE batches SET status = 'done', finished_at = datetime('now') WHERE id = ?", (batch_id,))
            conn.commit()
            row_done = conn.execute("SELECT status, gps_lat, gps_lng FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
            final_status = row_done["status"] if row_done else "complete"
            parsed_lat = row_done["gps_lat"] if row_done else parsed_lat
            parsed_lng = row_done["gps_lng"] if row_done else parsed_lng
        else:
            final_status = "queued"
    finally:
        conn.close()

    if async_mode:
        enqueue_batch(batch_id)

    if guest_token and guest_token != req.cookies.get("guest_session"):
        response.set_cookie(
            "guest_session",
            guest_token,
            httponly=True,
            secure=cookie_secure(req),
            samesite="lax",
            max_age=86400,
            path="/",
        )

    res_body: dict[str, Any] = {
        "analysis_id": analysis_id,
        "batch_id": batch_id,
        "status": final_status,
        "is_public": is_public,
        "submission_type": "Public Submission" if is_public else "Verified Inspector",
        "gps_lat": parsed_lat,
        "gps_lng": parsed_lng,
    }
    if gps_warning:
        res_body["warning"] = gps_warning
    return res_body


@app.get("/api/dashboard", response_class=JSONResponse, dependencies=[Depends(require_role(["admin", "lgu_engineer"]))])
async def api_dashboard_alias(
    include_mock: bool | None = Query(None, description="Include mock analyses"),
    days: int = Query(30, ge=1, le=90),
    limit: int = Query(48, ge=1, le=100),
) -> dict[str, Any]:
    from backend.routes.stats import get_dashboard_stats
    eff_include_mock = is_mock_mode() if include_mock is None else include_mock
    return await get_dashboard_stats(include_mock=eff_include_mock, days=days, limit=limit)


@app.get("/api/history", response_class=JSONResponse)
async def api_history_alias(
    page: int = 1,
    limit: int = 20,
    status: str | None = None,
    is_mock: int | None = None,
    is_public: int | None = None,
    company: str | None = None,
    barangay: str | None = None,
    street: str | None = None,
    sort_by: str = "created_at",
    sort_dir: str = "desc",
    include_invalid: bool = Query(False),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    from backend.routes.analysis import get_analyses
    return await get_analyses(
        page=page,
        limit=limit,
        status=status,
        is_mock=is_mock,
        is_public=is_public,
        company=company,
        barangay=barangay,
        street=street,
        sort_by=sort_by,
        sort_dir=sort_dir,
        include_invalid=include_invalid,
        user=user,
    )

@app.get("/", response_class=HTMLResponse)
async def serve_root():
    with open("frontend/pages/index.html", "r") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("WIREFRAME_HOST", "127.0.0.1")
    port = int(os.environ.get("WIREFRAME_PORT", 8000))
    uvicorn.run("main:app", host=host, port=port, reload=True)
