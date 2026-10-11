import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend.auth import User, get_current_user, get_optional_user, require_role, verify_csrf
from backend.config_store import _cache
from backend.database import get_connection

router = APIRouter(prefix="/api/config", tags=["config"])

WHITELIST_KEYS = {
    "weights", "thresholds", "risk_bands", "class_weights", 
    "pole_heights", "pec_min_clearance_by_class", "margin_of_error_pct", 
    "model_path", "safety_mapping", "crs_params", "site_risk_weights", "severity_weights"
}

@router.get("", dependencies=[Depends(require_role(["admin"]))])
async def get_config():
    # Return all cached settings
    # To keep it simple, just load them from DB
    conn = get_connection()
    rows = conn.execute("SELECT key, value_json FROM settings").fetchall()
    conn.close()
    
    import json
    return {row["key"]: json.loads(row["value_json"]) for row in rows}

@router.get("/public")
async def get_config_public():
    conn = get_connection()
    rows = conn.execute("SELECT key, value_json FROM settings").fetchall()
    conn.close()
    
    import json
    settings = {row["key"]: json.loads(row["value_json"]) for row in rows}
    
    # Whitelist
    whitelist = ["action_mapping", "risk_bands", "pec_min_clearance_by_class", "margin_of_error_pct", "pole_heights", "safety_mapping"]
    return {k: v for k, v in settings.items() if k in whitelist}

@router.put("", dependencies=[Depends(verify_csrf), Depends(require_role(["admin"]))])
async def update_config(payload: dict[str, Any], user: User = Depends(require_role(["admin"]))):
    import json
    
    # 1. Validate keys
    for k in payload:
        if k not in WHITELIST_KEYS:
            raise HTTPException(status_code=422, detail=f"Unknown or non-editable configuration key: {k}")
            
    # 2. Validate risk_bands
    if "risk_bands" in payload:
        bands = payload["risk_bands"]
        if not isinstance(bands, dict) or "Low" not in bands or "Moderate" not in bands or "High" not in bands or "Critical" not in bands:
            raise HTTPException(status_code=422, detail="risk_bands must contain Low, Moderate, High, Critical")
        if bands["Low"]["max"] != bands["Moderate"]["min"] - 0.01:
            raise HTTPException(status_code=422, detail="risk_bands are not contiguous")
            
    # 3. Validate margin_of_error_pct
    if "margin_of_error_pct" in payload:
        margin = payload["margin_of_error_pct"]
        if not (0 <= margin <= 100):
            raise HTTPException(status_code=422, detail="margin_of_error_pct must be between 0 and 100")
            
    # 4. Validate pec_min_clearance_by_class
    if "pec_min_clearance_by_class" in payload:
        for k, v in payload["pec_min_clearance_by_class"].items():
            if v <= 0:
                raise HTTPException(status_code=422, detail="pec_min_clearance_by_class values must be > 0")
            
    # 5. Validate pole_heights
    if "pole_heights" in payload:
        for val in payload["pole_heights"]:
            if val <= 0:
                raise HTTPException(status_code=422, detail="pole_heights must be positive")
                
    # 6. Validate weights
    if "class_weights" in payload:
        for val in payload["class_weights"].values():
            if val < 0:
                raise HTTPException(status_code=422, detail="weights must be non-negative")
                
    # 7. Validate model_path
    if "model_path" in payload:
        model_path = payload["model_path"]
        base_dir = Path("backend/models").resolve()
        
        try:
            target = Path(model_path).resolve()
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid model_path")
            
        if not str(target).startswith(str(base_dir)):
            raise HTTPException(status_code=422, detail="model_path must be inside backend/models/")
            
        if target.suffix != ".pt":
            raise HTTPException(status_code=422, detail="model_path must end with .pt")
            
        if not target.is_file():
            raise HTTPException(status_code=422, detail="model file does not exist")
            
        # Try loading the model to validate it
        try:
            from ultralytics import YOLO

            from backend import main_state
            from backend.mock_detector import is_mock_mode
            
            if not is_mock_mode():
                model = YOLO(str(target))
                main_state.set_model(model)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to load model: {e!s}")
            
    # Update DB and invalidate cache
    conn = get_connection()
    try:
        for k, v in payload.items():
            conn.execute(
                "UPDATE settings SET value_json = ?, updated_by = ?, updated_at = datetime('now') WHERE key = ?",
                (json.dumps(v), user.username, k)
            )
            # Invalidate cache entry
            if k in _cache:
                del _cache[k]
                
        # Invalidate all PDF reports since config affects output
        reports = conn.execute("SELECT path FROM reports").fetchall()
        out_path = Path("outputs").resolve()
        for r in reports:
            if r["path"]:
                try:
                    rp = Path(r["path"]).resolve()
                    if str(rp).startswith(str(out_path)) and os.path.exists(rp):
                        os.remove(rp)
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to delete cached PDF {r['path']}: {e}")
        conn.execute("DELETE FROM reports")
        
        conn.commit()
    finally:
        conn.close()
        
    return {"status": "success"}
