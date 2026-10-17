"""Thin facade wrapping the CV pipeline modules to preserve public function signatures."""

import logging
from typing import Any

from backend.formulas import (
    compute_barangay_risk,
    compute_detection_risk,
    compute_risk_score,
    get_severity,
)
from backend.utils.cv.catenary import calculate_sag_metrics
from backend.utils.cv.centerline import extract_centerline
from backend.utils.cv.preprocess import validate_and_preprocess
from backend.utils.cv.segment import segment_image
from backend.utils.cv.vni import compute_vni

# For backward compatibility, some functions might just call the new modules.

logger = logging.getLogger(__name__)

# To maintain full signature compatibility, we just proxy to the new modules if requested.
# The exact interface of `analysis_engine.py` was not fully specified in Phase 0.
# We will define a main entrypoint `analyze_image` that handles a full pipeline run.

def analyze_image(image_path: str, model, analysis_id: int = None, db_conn = None) -> dict[str, Any]:
    """Run the full CV pipeline on an image and return the final report."""
    # 1. Preprocess
    prep = validate_and_preprocess(image_path)
    if not prep.quality_ok:
        return {"status": "rejected", "reason": prep.reason}
        
    # 2. VNI
    vni_res = compute_vni(prep.gray)
    
    # 3. Segment
    seg_res = segment_image(prep.rgb, model, prep.original_shape)
    
    wires = []
    max_crs = 0.0
    dead_wires = 0
    severities = []
    
    if seg_res.mask_available:
        for i, mask in enumerate(seg_res.masks):
            cls_idx = seg_res.classes[i]
            cls_name = seg_res.names.get(cls_idx, "unknown")
            conf = seg_res.confidences[i]
            
            if cls_name == "dead_abandoned":
                dead_wires += 1
                
            severities.append(get_severity(conf))
            
            # 4. Centerline
            cline = extract_centerline(mask, seg_res.boxes[i])
            
            if cline.is_valid:
                # 5. Catenary
                cat = calculate_sag_metrics(cline.polyline)
                
                # CRS calculated using true sag_to_span (chord sag / span) & residual
                from backend.formulas import (
                    compute_catenary_risk,
                    compute_crs_band,
                    compute_safety_level,
                )
                crs_raw = compute_catenary_risk(cat.sag_to_span, cat.residual)
                
                # Apply class weights
                from backend.config_store import get_setting
                weights = get_setting("class_weights", {})
                cls_weight = weights.get(cls_name, 1.0)
                crs_weighted = max(0.0, min(100.0, crs_raw * cls_weight))
                
                max_crs = max(max_crs, crs_weighted)
                
                crs_band = compute_crs_band(crs_weighted)
                
                # Get lowest wire y for clearance
                lowest_y = max([pt[1] for pt in cline.polyline])
                
                wires.append({
                    "class_name": cls_name,
                    "confidence": conf,
                    "sag_to_span": cat.sag_to_span,
                    "sag_px": cat.sag_px,
                    "span_px": cat.span_px,
                    "crs": crs_weighted,
                    "band": crs_band,
                    "lowest_y": lowest_y,
                    "centerline_json": str(cline.polyline),
                    "params_json": str({"a": cat.fitted_a, "residual": cat.residual})
                })
    else:
        logger.warning("Segmentation masks not available. Skipping sag calculation.")
        
    # Scale calibration and clearance
    from backend.formulas import compute_clearance_status
    from backend.utils.cv.calibration import calibrate_scale, estimate_clearance
    
    # We don't have manual points in standard analyze_image signature yet, but we'll extract pole bbox
    # Let's see if there is a pole detection
    pole_bbox = None
    if seg_res.mask_available:
        for i, cls_idx in enumerate(seg_res.classes):
            if seg_res.names.get(cls_idx) == "pole":
                pole_bbox = seg_res.boxes[i]
                break
                
    scale_res = calibrate_scale(pole_bbox=pole_bbox, image_height=prep.original_shape[0])
    
    pec_min_map = get_setting("pec_min_clearance_by_class", {})
    
    image_safety_level = "Unknown"
    
    for wire in wires:
        lowest_y = wire.pop("lowest_y")
        clear_est = estimate_clearance(lowest_y, pole_bbox, scale_res.meters_per_px, prep.original_shape[0])
        
        wire["sag_m"] = wire["sag_px"] * scale_res.meters_per_px if scale_res.meters_per_px else None
        wire["span_m"] = wire["span_px"] * scale_res.meters_per_px if scale_res.meters_per_px else None
        wire["clearance_m"] = clear_est["clearance_m"]
        
        # Merge range into params
        import json
        from backend.utils.clearance_helpers import parse_params_json
        params = parse_params_json(wire["params_json"])
        params["clearance_low_m"] = clear_est["clearance_low_m"]
        params["clearance_high_m"] = clear_est["clearance_high_m"]
        wire["params_json"] = json.dumps(params)
        
        clearance_status = compute_clearance_status(
            clear_est["clearance_m"], clear_est["clearance_low_m"], clear_est["clearance_high_m"], pec_min_map.get(wire.get("class_name", "unknown"), pec_min_map.get("default", 5.5))
        ) if clear_est["status"] != "unavailable" else "unavailable"
        
        wire["clearance_status"] = clearance_status
        
        est_str = f"Estimated clearance {clear_est['clearance_m']:.2f} m " \
                  f"(range {clear_est['clearance_low_m']:.2f}-{clear_est['clearance_high_m']:.2f} m)" \
                  if clear_est["status"] != "unavailable" else "Clearance unavailable."
                  
        safety = compute_safety_level(wire["band"], clearance_status, est_str)
        wire["safety_level"] = safety["safety_level"]
        wire["safety_reasons"] = safety["reasons"]
        
    from backend.utils.clearance_helpers import determine_overall_safety
    wire_safeties = [w["safety_level"] for w in wires if "safety_level" in w]
    image_safety_level = determine_overall_safety(wire_safeties)

    # Calculate final scores
    detection_risk = compute_detection_risk(severities)
    barangay_risk = compute_barangay_risk(vni_res.vni, max_crs, dead_wires, len(wires))
    final_score = compute_risk_score(detection_risk, barangay_risk)
    
    res = {
        "status": "success",
        "vni": vni_res.vni,
        "max_crs": max_crs,
        "barangay_risk": barangay_risk,
        "final_risk_score": final_score,
        "wire_count": len(wires),
        "dead_wire_count": dead_wires,
        "image_safety_level": image_safety_level,
        "limitation": "estimate only, oblique capture angles reduce accuracy, not a survey measurement. PEC minimum TODO(VERIFY-WITH-ADVISER).",
        "wires": wires
    }
    
    # Store to DB if analysis_id provided
    if analysis_id and db_conn:
        for w in wires:
            db_conn.execute(
                """INSERT INTO wires (
                    analysis_id, class_name, confidence, crs, band, sag_px, span_px, 
                    sag_m, clearance_m, clearance_status, safety_level, centerline_json, params_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    analysis_id, w["class_name"], w["confidence"], w["crs"], w["band"], 
                    w["sag_px"], w["span_px"], w.get("sag_m"), w.get("clearance_m"), 
                    w["clearance_status"], w["safety_level"], w["centerline_json"], w["params_json"]
                )
            )
        db_conn.commit()
        
    return res

def process_analysis(analysis_id: int, image_path: str, conn, model, batch_id: int = None) -> dict[str, Any]:
    """Processing path wrapper to run analysis and persist results."""
    from backend.utils.pipeline import execute_pipeline
    execute_pipeline(analysis_id, image_path, conn, model, batch_id)
    # The return value is no longer synchronously meaningful since it's backgrounded,
    # but we can return basic status
    return {"status": "submitted"}
