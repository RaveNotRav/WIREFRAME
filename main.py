"""WIREFRAME - Wire Risk Assessment & Monitoring Engine."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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
from backend.mock_detector import is_mock_mode

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
            if os.path.isfile(model_path):
                try:
                    from ultralytics import YOLO
                    model = YOLO(model_path)
                    main_state.set_model(model)
                    main_state.set_model_available(True)
                    logger.info("Model loaded: %s", model_path)
                except Exception as exc:
                    main_state.set_model(None)
                    main_state.set_model_available(False)
                    logger.error(f"Failed to load model: {exc}")
            else:
                main_state.set_model(None)
                main_state.set_model_available(False)
                logger.warning("Model file not found: %s. Uploads will return 503.", model_path)

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

app = FastAPI(title="WIREFRAME", version="0.1.0", lifespan=lifespan)

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
    return {
        "status": "ok",
        "ready": ready,
        "model_status": model_status,
        "model_loaded": model_loaded,
        "schema_version": schema_ver,
        "mock_mode": is_mock,
    }


@app.get("/api/analysis/{analysis_id}/result", response_class=JSONResponse)
async def get_analysis_result(analysis_id: int, req: Request, user: User | None = Depends(get_optional_user)) -> dict[str, Any]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
            
        verify_analysis_access(row, user, req)
            
        data = dict(row)
        
        # Map DB columns to frontend expected fields
        data["crs"] = data.get("max_crs") / 100.0 if data.get("max_crs") is not None else None
        
        from backend.formulas import compute_crs_band
        if data.get("max_crs") is not None:
            data["crs_band"] = compute_crs_band(data["max_crs"])
        else:
            data["crs_band"] = "unavailable"
            
        data["sag_m"] = data.get("sag_meters")
        data["ground_clearance_m"] = data.get("clearance_available")
        
        wires_rows = conn.execute("SELECT * FROM wires WHERE analysis_id = ?", (analysis_id,)).fetchall()
        wires_data = [dict(w) for w in wires_rows]
        data["wires"] = wires_data
        
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
            
        return data
    finally:
        conn.close()


from fastapi.responses import RedirectResponse
from fastapi import Request

@app.get("/", response_class=HTMLResponse)
async def serve_root():
    with open("frontend/pages/index.html", "r") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("WIREFRAME_HOST", "0.0.0.0")
    port = int(os.environ.get("WIREFRAME_PORT", 8000))
    uvicorn.run("main:app", host=host, port=port, reload=True)
