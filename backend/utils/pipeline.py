"""State Machine and Orchestrator for WIREFRAME CV Pipeline."""

import enum
import logging
import sqlite3
import traceback
from typing import Any

from backend.config_store import get_setting
from backend.formulas import (
    compute_barangay_risk,
    compute_clearance_status,
    compute_crs_band,
    compute_detection_risk,
    compute_risk_score,
    compute_safety_level,
    get_severity,
)
from backend.utils.cv.calibration import calibrate_scale, estimate_clearance
from backend.utils.cv.catenary import calculate_sag_metrics
from backend.utils.cv.centerline import extract_centerline
from backend.utils.cv.preprocess import validate_and_preprocess
from backend.utils.cv.segment import segment_image
from backend.utils.cv.vni import compute_vni

logger = logging.getLogger(__name__)

class PipelineState(str, enum.Enum):
    INITIALIZING = "Initializing"
    IDLE = "Idle"
    IMAGE_ACQUIRED = "ImageAcquired"
    QUALITY_VALIDATION = "QualityValidation"
    IMAGE_REJECTED = "ImageRejected"
    DETECTING_WIRES = "DetectingWires"
    NO_WIRES_DETECTED = "NoWiresDetected"
    ANALYZING = "Analyzing"
    COMPUTING_RISK = "ComputingRisk"
    GENERATING_REPORT = "GeneratingReport"
    MORE_IMAGES = "MoreImages"
    ERROR_RECOVERY = "ErrorRecovery"
    INTERRUPTED = "Interrupted"


def check_cancel_requested(conn: sqlite3.Connection, batch_id: int | None) -> bool:
    if not batch_id:
        return False
    row = conn.execute("SELECT cancel_requested FROM batches WHERE id = ?", (batch_id,)).fetchone()
    return bool(row and row["cancel_requested"])


def transition_state(conn: sqlite3.Connection, analysis_id: int, new_state: PipelineState, detail: str = ""):
    """Log state transition and commit immediately."""
    logger.info(f"[Analysis {analysis_id}] Transition to {new_state.value}: {detail}")
    conn.execute(
        "INSERT INTO pipeline_events (analysis_id, state, detail) VALUES (?, ?, ?)",
        (analysis_id, new_state.value, detail)
    )
    conn.execute(
        "UPDATE analyses SET current_state = ? WHERE id = ?",
        (new_state.value, analysis_id)
    )
    conn.commit()


def execute_pipeline(analysis_id: int, image_path: str, conn: sqlite3.Connection, model: Any, batch_id: int | None = None) -> None:
    """Core state machine executing the thesis flow for a single analysis."""
    
    # We use a separate connection internally if needed, but since we are given one, we use it.
    # The prompt mandates WAL mode and isolated connections. We assume the caller created `conn` properly.
    
    try:
        if check_cancel_requested(conn, batch_id):
            transition_state(conn, analysis_id, PipelineState.INTERRUPTED, "Batch cancelled before start")
            conn.execute("UPDATE analyses SET status = 'interrupted' WHERE id = ?", (analysis_id,))
            conn.commit()
            return

        transition_state(conn, analysis_id, PipelineState.INITIALIZING, "Starting analysis")
        conn.execute("UPDATE analyses SET status = 'processing' WHERE id = ?", (analysis_id,))
        conn.commit()
        
        transition_state(conn, analysis_id, PipelineState.IMAGE_ACQUIRED, f"Loaded {image_path}")
        
        if check_cancel_requested(conn, batch_id):
            raise InterruptedError("Cancelled during ImageAcquired")

        transition_state(conn, analysis_id, PipelineState.QUALITY_VALIDATION, "Checking quality metrics")
        prep = validate_and_preprocess(image_path)
        if not prep.quality_ok:
            transition_state(conn, analysis_id, PipelineState.IMAGE_REJECTED, prep.reason)
            conn.execute("UPDATE analyses SET status = 'rejected', summary = ? WHERE id = ?", (prep.reason, analysis_id))
            conn.commit()
            transition_state(conn, analysis_id, PipelineState.IDLE, "Pipeline finished (rejected)")
            return

        if check_cancel_requested(conn, batch_id):
            raise InterruptedError("Cancelled before DetectingWires")

        transition_state(conn, analysis_id, PipelineState.DETECTING_WIRES, "Running segmentation")
        from backend.mock_detector import is_mock_mode
        if is_mock_mode():
            try:
                from backend.utils.cv.mock_generator import generate_mock_segmentation
                seg_res = generate_mock_segmentation(image_path, prep.original_shape)
            except Exception as e:
                raise RuntimeError(f"Mock generator failed: {e}")
        else:
            if model is None:
                raise RuntimeError("Model is missing in real mode")
            seg_res = segment_image(prep.rgb, model, prep.original_shape)
        
        wires = []
        dead_wires = 0
        severities = []
        pole_bbox = None
        
        if seg_res.mask_available:
            for i, cls_idx in enumerate(seg_res.classes):
                cls_name = seg_res.names.get(cls_idx, "unknown")
                if cls_name == "pole":
                    pole_bbox = seg_res.boxes[i]
                else:
                    wires.append({
                        "class_name": cls_name,
                        "confidence": seg_res.confidences[i],
                        "mask": seg_res.masks[i],
                        "bbox": seg_res.boxes[i]
                    })
                    if cls_name == "dead_abandoned":
                        dead_wires += 1
                    severities.append(get_severity(seg_res.confidences[i]))

        if len(wires) == 0:
            transition_state(conn, analysis_id, PipelineState.NO_WIRES_DETECTED, "Zero wires found")
            conn.execute("UPDATE analyses SET status = 'no_wires' WHERE id = ?", (analysis_id,))
            conn.commit()
            transition_state(conn, analysis_id, PipelineState.GENERATING_REPORT, "Creating empty report")
            # In a real app we'd build a report PDF here
            transition_state(conn, analysis_id, PipelineState.IDLE, "Pipeline finished (no wires)")
            return
            
        if check_cancel_requested(conn, batch_id):
            raise InterruptedError("Cancelled before Analyzing")
            
        transition_state(conn, analysis_id, PipelineState.ANALYZING, "Computing VNI and extracting centerline geometry")
        vni_res = compute_vni(prep.gray)
        
        analyzed_wires = []
        for w in wires:
            cline = extract_centerline(w["mask"], w["bbox"])
            if cline.is_valid:
                cat = calculate_sag_metrics(cline.polyline)
                w["catenary"] = cat
                w["lowest_y"] = max([pt[1] for pt in cline.polyline])
                w["centerline_json"] = str(cline.polyline)
                w["params_json"] = str({"a": cat.fitted_a, "residual": cat.residual})
                analyzed_wires.append(w)
                
        if check_cancel_requested(conn, batch_id):
            raise InterruptedError("Cancelled before ComputingRisk")
            
        transition_state(conn, analysis_id, PipelineState.COMPUTING_RISK, "Scoring CRS, scaling sag, assigning safety")
        from backend.formulas import compute_catenary_risk
        weights = get_setting("class_weights", {})
        pec_min_map = get_setting("pec_min_clearance_by_class", {})
        
        scale_res = calibrate_scale(pole_bbox=pole_bbox, image_height=prep.original_shape[0])
        
        max_crs = 0.0
        final_wires = []
        for w in analyzed_wires:
            cat = w["catenary"]
            crs_raw = compute_catenary_risk(cat.sag_ratio, cat.residual)
            cls_weight = weights.get(w["class_name"], 1.0)
            crs_weighted = max(0.0, min(100.0, crs_raw * cls_weight))
            max_crs = max(max_crs, crs_weighted)
            
            clear_est = estimate_clearance(w["lowest_y"], pole_bbox, scale_res.meters_per_px, prep.original_shape[0])
            
            import json
            params = eval(w["params_json"])
            params["clearance_low_m"] = clear_est["clearance_low_m"]
            params["clearance_high_m"] = clear_est["clearance_high_m"]
            
            clearance_status = compute_clearance_status(
                clear_est["clearance_m"], clear_est["clearance_low_m"], clear_est["clearance_high_m"], pec_min_map.get(w["class_name"], pec_min_map.get("default", 5.5))
            ) if clear_est["status"] != "unavailable" else "unavailable"
            
            est_str = f"Estimated clearance {clear_est['clearance_m']:.2f} m " \
                      f"(range {clear_est['clearance_low_m']:.2f}-{clear_est['clearance_high_m']:.2f} m)" \
                      if clear_est["status"] != "unavailable" else "Clearance unavailable."
                      
            band = compute_crs_band(crs_weighted)
            safety = compute_safety_level(band, clearance_status, est_str)
            
            final_wires.append({
                "class_name": w["class_name"],
                "confidence": w["confidence"],
                "sag_px": cat.sag_px,
                "span_px": cat.span_px,
                "crs": crs_weighted,
                "band": band,
                "sag_m": cat.sag_px * scale_res.meters_per_px if scale_res.meters_per_px else None,
                "clearance_m": clear_est["clearance_m"],
                "clearance_status": clearance_status,
                "safety_level": safety["safety_level"],
                "centerline_json": w["centerline_json"],
                "params_json": json.dumps(params)
            })
            
        # Store wires
        for fw in final_wires:
            conn.execute(
                """INSERT INTO wires (
                    analysis_id, class_name, confidence, crs, band, sag_px, span_px, 
                    sag_m, clearance_m, clearance_status, safety_level, centerline_json, params_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    analysis_id, fw["class_name"], fw["confidence"], fw["crs"], fw["band"], 
                    fw["sag_px"], fw["span_px"], fw["sag_m"], fw["clearance_m"], 
                    fw["clearance_status"], fw["safety_level"], fw["centerline_json"], fw["params_json"]
                )
            )
        
        # Save spectrum
        import os
        import cv2
        import ast
        from pathlib import Path
        base_outputs = Path("outputs").resolve()
        base_outputs.mkdir(exist_ok=True)
        spectrum_path = ""
        if vni_res.spectrum_png:
            spec_file = base_outputs / f"analysis_{analysis_id}_spectrum.png"
            with open(spec_file, "wb") as f:
                f.write(vni_res.spectrum_png)
            spectrum_path = str(spec_file)

        # Save annotated image
        annotated_img = prep.rgb.copy()
        if pole_bbox:
            px1, py1, px2, py2 = [int(v) for v in pole_bbox]
            cv2.rectangle(annotated_img, (px1, py1), (px2, py2), (255, 0, 0), 2)
            cv2.putText(annotated_img, "Pole", (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
        for w in analyzed_wires:
            bx1, by1, bx2, by2 = [int(v) for v in w["bbox"]]
            cv2.rectangle(annotated_img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(annotated_img, w["class_name"], (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            polyline = ast.literal_eval(w["centerline_json"])
            if len(polyline) > 1:
                for i in range(len(polyline)-1):
                    cv2.line(annotated_img, polyline[i], polyline[i+1], (0, 0, 255), 2)
                    
        annotated_file = base_outputs / f"analysis_{analysis_id}_annotated.jpg"
        cv2.imwrite(str(annotated_file), cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        annotated_path = str(annotated_file)

        # Calculate final scores
        detection_risk = compute_detection_risk(severities)
        barangay_risk = compute_barangay_risk(vni_res.vni, max_crs, dead_wires, len(analyzed_wires))
        final_score = compute_risk_score(detection_risk, barangay_risk)
        
        conn.execute(
            """UPDATE analyses 
               SET status = 'complete', vni = ?, max_crs = ?, barangay_risk = ?, risk_score = ?, dead_wire_count = ?, spectrum_path = ?, annotated_path = ?
               WHERE id = ?""",
            (vni_res.vni, max_crs, barangay_risk, final_score, dead_wires, spectrum_path, annotated_path, analysis_id)
        )
        conn.commit()

        if check_cancel_requested(conn, batch_id):
            raise InterruptedError("Cancelled before GeneratingReport")

        transition_state(conn, analysis_id, PipelineState.GENERATING_REPORT, "Compiling final PDF/JSON records")
        # In a real app we'd generate a report here
        
        transition_state(conn, analysis_id, PipelineState.IDLE, "Pipeline finished successfully")
        
    except InterruptedError as e:
        logger.info(f"Analysis {analysis_id} interrupted: {e}")
        transition_state(conn, analysis_id, PipelineState.INTERRUPTED, str(e))
        conn.execute("UPDATE analyses SET status = 'interrupted' WHERE id = ?", (analysis_id,))
        conn.commit()
    except Exception as e:
        logger.error(f"Pipeline exception for {analysis_id}:\n{traceback.format_exc()}")
        transition_state(conn, analysis_id, PipelineState.ERROR_RECOVERY, f"Exception: {type(e).__name__} - {str(e)[:100]}")
        conn.execute("UPDATE analyses SET status = 'error', summary = ? WHERE id = ?", ("Pipeline failed due to internal error.", analysis_id))
        conn.commit()
        transition_state(conn, analysis_id, PipelineState.IDLE, "Recovered from error")
