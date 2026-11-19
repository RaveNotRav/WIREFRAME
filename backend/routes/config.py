import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend.auth import User, get_current_user, get_optional_user, require_role, verify_csrf
from backend import config_store
from backend.config_store import NO_WIRES_ACTION_TEXT
from backend.database import get_connection

router = APIRouter(prefix="/api/config", tags=["config"])

WHITELIST_KEYS = {
    "weights", "thresholds", "risk_bands", "class_weights",
    "pole_heights", "pec_min_clearance_by_class", "margin_of_error_pct",
    "model_path", "safety_mapping", "crs_params", "site_risk_weights",
    "severity_weights", "severity_thresholds", "class_aliases", "inference_params",
    "action_mapping",
}

ALLOWED_CLASS_KEYS = {
    "power_distribution_line",
    "communication_cable",
    "dead_wire",
    "vegetation",
    "power_line",
    "telecom_cable",
    "fiber_optic",
    "dead_abandoned",
    "pole",
    "other",
    "default",
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
    filtered = {k: v for k, v in settings.items() if k in whitelist}
    if "action_mapping" in filtered and isinstance(filtered["action_mapping"], dict):
        filtered["action_mapping"]["no_wires"] = NO_WIRES_ACTION_TEXT
    return filtered

@router.put("", dependencies=[Depends(verify_csrf), Depends(require_role(["admin"]))])
async def update_config(payload: dict[str, Any], user: User = Depends(require_role(["admin"]))):
    import json
    
    # 1. Validate keys
    for k in payload:
        if k not in WHITELIST_KEYS:
            raise HTTPException(status_code=422, detail=f"Unknown or non-editable configuration key: {k}")
            
    # 2. Validate risk_bands (list-of-dicts format or legacy dict format converted to list-of-dicts)
    if "risk_bands" in payload:
        bands = payload["risk_bands"]
        required_labels = ("Low", "Moderate", "High", "Critical")
        if isinstance(bands, list):
            if len(bands) != 4 or not all(isinstance(b, dict) for b in bands):
                raise HTTPException(status_code=422, detail="risk_bands must contain Low, Moderate, High, Critical")
            by_label = {b.get("label"): b for b in bands if isinstance(b.get("label"), str)}
            if any(lbl not in by_label for lbl in required_labels):
                raise HTTPException(status_code=422, detail="risk_bands must contain Low, Moderate, High, Critical")
            ordered = [by_label[lbl] for lbl in required_labels]
            for b in ordered:
                b_min = b.get("min")
                b_max = b.get("max")
                if not isinstance(b_min, (int, float)) or not isinstance(b_max, (int, float)):
                    raise HTTPException(status_code=422, detail="risk_bands min/max must be numeric")
                if b_min < 0.0 or b_max > 1.0 or b_min > b_max:
                    raise HTTPException(status_code=422, detail="risk_bands min/max must be within [0, 1] with min <= max")
            for idx in range(len(ordered) - 1):
                cur_max = float(ordered[idx]["max"])
                nxt_min = float(ordered[idx + 1]["min"])
                if nxt_min < cur_max or round(nxt_min - cur_max, 2) > 0.01:
                    raise HTTPException(status_code=422, detail="risk_bands are not contiguous")
            payload["risk_bands"] = ordered
        elif isinstance(bands, dict):
            if any(lbl not in bands or not isinstance(bands[lbl], dict) for lbl in required_labels):
                raise HTTPException(status_code=422, detail="risk_bands must contain Low, Moderate, High, Critical")
            ordered = []
            for lbl in required_labels:
                b = bands[lbl]
                b_min = b.get("min")
                b_max = b.get("max")
                if not isinstance(b_min, (int, float)) or not isinstance(b_max, (int, float)):
                    raise HTTPException(status_code=422, detail="risk_bands min/max must be numeric")
                if b_min < 0.0 or b_max > 1.0 or b_min > b_max:
                    raise HTTPException(status_code=422, detail="risk_bands min/max must be within [0, 1] with min <= max")
                ordered.append({"min": float(b_min), "max": float(b_max), "label": lbl})
            for idx in range(len(ordered) - 1):
                cur_max = float(ordered[idx]["max"])
                nxt_min = float(ordered[idx + 1]["min"])
                if nxt_min < cur_max or round(nxt_min - cur_max, 2) > 0.01:
                    raise HTTPException(status_code=422, detail="risk_bands are not contiguous")
            payload["risk_bands"] = ordered
        else:
            raise HTTPException(status_code=422, detail="risk_bands must contain Low, Moderate, High, Critical")
            
    # 3. Validate margin_of_error_pct
    if "margin_of_error_pct" in payload:
        margin = payload["margin_of_error_pct"]
        if not isinstance(margin, (int, float)) or not (0 <= margin <= 100):
            raise HTTPException(status_code=422, detail="margin_of_error_pct must be between 0 and 100")
            
    # 4. Validate pec_min_clearance_by_class
    if "pec_min_clearance_by_class" in payload:
        pec_dict = payload["pec_min_clearance_by_class"]
        if not isinstance(pec_dict, dict):
            raise HTTPException(status_code=422, detail="pec_min_clearance_by_class must be an object")
        for k, v in pec_dict.items():
            if not isinstance(v, (int, float)) or v <= 0:
                raise HTTPException(status_code=422, detail="pec_min_clearance_by_class values must be > 0")
            
    # 5. Validate pole_heights
    if "pole_heights" in payload:
        for val in payload["pole_heights"]:
            if not isinstance(val, (int, float)) or val <= 0:
                raise HTTPException(status_code=422, detail="pole_heights must be positive")
                
    # 6. Validate weights
    if "class_weights" in payload:
        cw_dict = payload["class_weights"]
        if not isinstance(cw_dict, dict):
            raise HTTPException(status_code=422, detail="class_weights must be an object")
        for val in cw_dict.values():
            if not isinstance(val, (int, float)) or val < 0:
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
    key_remap = {
        "weights": "site_risk_weights",
        "thresholds": "severity_thresholds",
    }
    conn = get_connection()
    try:
        for k, v in payload.items():
            db_key = key_remap.get(k, k)
            conn.execute(
                """INSERT INTO settings (key, value_json, updated_at, updated_by)
                   VALUES (?, ?, datetime('now'), ?)
                   ON CONFLICT(key) DO UPDATE SET
                       value_json = excluded.value_json,
                       updated_at = excluded.updated_at,
                       updated_by = excluded.updated_by""",
                (db_key, json.dumps(v), user.username)
            )
            # Update in-memory cache immediately so get_setting() reflects the new value
            config_store._cache[db_key] = v
            if k != db_key:
                config_store._cache.pop(k, None)
                
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
