"""State Machine and Orchestrator for WIREFRAME CV Pipeline."""

import enum
import json
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
from backend.utils.cv.types import normalize_class_name
from backend.utils.cv.vni import compute_vni
from backend.utils.clearance_helpers import (
    evaluate_wire,
    get_lowest_y,
    parse_params_json,
    rollup_analysis_clearance,
)

logger = logging.getLogger(__name__)

WIRE_CLASSES = {"power_line", "telecom_cable", "fiber_optic", "dead_abandoned"}

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
        best_pole_conf = -1.0
        
        if seg_res.mask_available:
            for i, cls_idx in enumerate(seg_res.classes):
                raw_cls_name = seg_res.names.get(cls_idx, "unknown")
                cls_name = normalize_class_name(raw_cls_name)
                conf = seg_res.confidences[i] if i < len(seg_res.confidences) else 0.0

                if cls_name == "pole":
                    if conf > best_pole_conf:
                        best_pole_conf = conf
                        pole_bbox = seg_res.boxes[i]
                elif cls_name in WIRE_CLASSES:
                    wires.append({
                        "class_name": cls_name,
                        "confidence": conf,
                        "mask": seg_res.masks[i],
                        "bbox": seg_res.boxes[i]
                    })
                    if cls_name == "dead_abandoned":
                        dead_wires += 1
                    severities.append(get_severity(conf))
                else:
                    logger.debug("Skipping non-wire class: %s (confidence: %.2f)", cls_name, conf)

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
                w["lowest_y"] = get_lowest_y(cline.polyline)
                w["centerline_json"] = str(cline.polyline)
                w["params_json"] = json.dumps({"a": cat.fitted_a, "residual": cat.residual})
                analyzed_wires.append(w)
                
        if check_cancel_requested(conn, batch_id):
            raise InterruptedError("Cancelled before ComputingRisk")
            
        transition_state(conn, analysis_id, PipelineState.COMPUTING_RISK, "Scoring CRS, scaling sag, assigning safety")
        from backend.formulas import compute_catenary_risk
        raw_weights = get_setting("class_weights", {}) or {}
        weights = {normalize_class_name(k): v for k, v in raw_weights.items()}
        raw_pec = get_setting("pec_min_clearance_by_class", {}) or {}
        pec_min_map = {normalize_class_name(k): v for k, v in raw_pec.items()}
        
        scale_res = calibrate_scale(pole_bbox=pole_bbox, image_height=prep.original_shape[0])
        
        max_crs = 0.0
        final_wires = []
        for w in analyzed_wires:
            cat = w["catenary"]
            crs_raw = compute_catenary_risk(cat.sag_to_span, cat.residual)
            cls_key = normalize_class_name(w["class_name"])
            cls_weight = weights.get(cls_key, 1.0)
            crs_weighted = max(0.0, min(100.0, crs_raw * cls_weight))
            max_crs = max(max_crs, crs_weighted)
            
            params = parse_params_json(w["params_json"])
            ew = evaluate_wire(
                class_name=w["class_name"],
                confidence=w["confidence"],
                sag_px=cat.sag_px,
                span_px=cat.span_px,
                crs_weighted=crs_weighted,
                lowest_y=w["lowest_y"],
                pole_bbox=pole_bbox,
                meters_per_px=scale_res.meters_per_px,
                image_height=prep.original_shape[0],
                pec_min_map=pec_min_map,
                params_dict=params,
            )
            ew["centerline_json"] = w["centerline_json"]
            final_wires.append(ew)
            
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
        import numpy as np
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
                pts = np.round(polyline).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(annotated_img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
                    
        annotated_file = base_outputs / f"analysis_{analysis_id}_annotated.jpg"
        cv2.imwrite(str(annotated_file), cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        annotated_path = str(annotated_file)

        # Calculate final scores
        detection_risk = compute_detection_risk(severities)
        barangay_risk = compute_barangay_risk(vni_res.vni, max_crs, dead_wires, len(analyzed_wires))
        final_score = compute_risk_score(detection_risk, barangay_risk)
        
        # Roll up clearance and sag metrics
        margin_pct = get_setting("margin_of_error_pct", 20) / 100.0
        clearance_rollup = rollup_analysis_clearance(final_wires, pec_min_map, margin_pct)

        if clearance_rollup["clearance_available"] is not None:
            pole_heights_cfg = get_setting("pole_heights", [10.0]) or [10.0]
            assumed_pole_height_m = pole_heights_cfg[0] if isinstance(pole_heights_cfg, list) and pole_heights_cfg else 10.0
        else:
            assumed_pole_height_m = None

        conn.execute(
            """UPDATE analyses 
               SET status = 'complete', vni = ?, max_crs = ?, barangay_risk = ?, risk_score = ?, dead_wire_count = ?, spectrum_path = ?, annotated_path = ?,
                   clearance_ok = ?, clearance_available = ?, pec_min_clearance_m = ?, sag_meters = ?, assumed_pole_height_m = ?,
                   width = ?, height = ?, pole_bbox_json = ?, margin_error_m = ?
               WHERE id = ?""",
            (
                vni_res.vni, max_crs, barangay_risk, final_score, dead_wires, spectrum_path, annotated_path,
                clearance_rollup["clearance_ok"], clearance_rollup["clearance_available"], clearance_rollup["pec_min_clearance_m"],
                clearance_rollup["sag_meters"], assumed_pole_height_m,
                prep.original_shape[1], prep.original_shape[0], json.dumps(pole_bbox) if pole_bbox else None, clearance_rollup["margin_error_m"],
                analysis_id
            )
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
