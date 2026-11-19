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
from backend.utils.cv.calibration import (
    bridge_across_poles,
    calibrate_scale,
    detect_pole_hough_fallback,
    estimate_clearance,
)
from backend.utils.cv.catenary import calculate_sag_metrics
from backend.utils.cv.centerline import extract_centerline
from backend.utils.cv.preprocess import validate_and_preprocess
from backend.utils.cv.segment import segment_image
from backend.utils.cv.types import (
    CANONICAL_CONTEXT_CLASSES,
    DEFAULT_CLASS_ALIASES,
    IGNORED_NON_WIRE_CLASSES,
    normalize_class_name,
    normalize_wire_class,
)
from backend.utils.cv.vni import compute_vni
from backend.utils.clearance_helpers import (
    evaluate_wire,
    get_lowest_y,
    parse_params_json,
    rollup_analysis_clearance,
)

from backend.utils.uploads import extract_display_filename

logger = logging.getLogger(__name__)

WIRE_CLASSES = {
    "power_distribution_line",
    "communication_cable",
    "dead_wire",
    "power_line",
    "telecom_cable",
    "fiber_optic",
    "dead_abandoned",
    "other",
}
_UNKNOWN_CLASSES_LOGGED: set[str] = set()


def get_effective_class_aliases(aliases: dict[str, str] | None = None) -> dict[str, str]:
    """Merge default class aliases, config store aliases, and optional WIREFRAME_CLASS_ALIASES JSON env."""
    import os
    merged: dict[str, str] = dict(DEFAULT_CLASS_ALIASES)
    cfg_aliases = aliases if aliases is not None else (get_setting("class_aliases", {}) or {})
    if isinstance(cfg_aliases, dict):
        merged.update(cfg_aliases)
    env_aliases = os.getenv("WIREFRAME_CLASS_ALIASES")
    if env_aliases:
        try:
            parsed = json.loads(env_aliases)
            if isinstance(parsed, dict):
                merged.update({str(k): str(v) for k, v in parsed.items()})
        except Exception as e:
            logger.warning("Invalid WIREFRAME_CLASS_ALIASES JSON (%s); ignoring env override.", e)
    return merged


def resolve_class_name(raw_name: str, aliases: dict[str, str] | None = None) -> str:
    """Resolve raw model class name through default + configured aliases and normalize to canonical class."""
    if not raw_name:
        return "unknown"
    effective = get_effective_class_aliases(aliases)
    key = str(raw_name).strip().lower()
    norm_key = normalize_class_name(raw_name)
    resolved = norm_key
    if key in effective:
        resolved = normalize_class_name(effective[key])
    elif str(raw_name) in effective:
        resolved = normalize_class_name(effective[str(raw_name)])
    elif norm_key in effective:
        resolved = normalize_class_name(effective[norm_key])
    return normalize_wire_class(resolved)


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
        
        display_name = extract_display_filename(image_path)
        transition_state(conn, analysis_id, PipelineState.IMAGE_ACQUIRED, f"Loaded {display_name}")
        
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
                logger.error("Model not available for analysis %s", analysis_id)
                transition_state(conn, analysis_id, PipelineState.ERROR_RECOVERY, "Model not available")
                conn.execute(
                    "UPDATE analyses SET status = 'error', summary = 'Model not available', is_mock = 0 WHERE id = ?",
                    (analysis_id,)
                )
                conn.commit()
                transition_state(conn, analysis_id, PipelineState.IDLE, "Pipeline finished (error)")
                return
            seg_res = segment_image(prep.rgb, model, prep.original_shape)
            try:
                from backend.utils.cv.wirenet import refine_detections_with_wirenet
                seg_res = refine_detections_with_wirenet(prep.rgb, seg_res)
            except Exception as wn_exc:
                logger.debug("Optional WireNet hook skipped: %s", wn_exc)
        
        wires = []
        vegetation_detections = []
        dead_wires = 0
        severities = []
        pole_bbox = None
        pole_masks: list[Any] = []
        best_pole_conf = -1.0
        
        aliases = get_setting("class_aliases", {}) or {}
        if seg_res.mask_available:
            for i, cls_idx in enumerate(seg_res.classes):
                raw_cls_name = seg_res.names.get(cls_idx, "unknown")
                cls_name = resolve_class_name(raw_cls_name, aliases)
                conf = seg_res.confidences[i] if i < len(seg_res.confidences) else 0.0

                if cls_name == "pole":
                    if i < len(seg_res.masks) and seg_res.masks[i] is not None:
                        pole_masks.append(seg_res.masks[i])
                    if conf > best_pole_conf:
                        best_pole_conf = conf
                        pole_bbox = seg_res.boxes[i]
                elif cls_name in CANONICAL_CONTEXT_CLASSES or cls_name == "vegetation":
                    vegetation_detections.append({
                        "class_name": "vegetation",
                        "confidence": conf,
                        "mask": seg_res.masks[i],
                        "bbox": seg_res.boxes[i]
                    })
                elif cls_name in WIRE_CLASSES:
                    canon_cls = normalize_wire_class(cls_name)
                    wires.append({
                        "class_name": canon_cls,
                        "confidence": conf,
                        "mask": seg_res.masks[i],
                        "bbox": seg_res.boxes[i]
                    })
                    if canon_cls in ("dead_wire", "dead_abandoned"):
                        dead_wires += 1
                    severities.append(get_severity(conf))
                elif cls_name in IGNORED_NON_WIRE_CLASSES or cls_name in ("drop", "ignore", "none"):
                    logger.debug("Skipping non-wire class: %s (confidence: %.2f)", cls_name, conf)
                else:
                    if raw_cls_name not in _UNKNOWN_CLASSES_LOGGED:
                        _UNKNOWN_CLASSES_LOGGED.add(raw_cls_name)
                        logger.warning(
                            "Unknown model class '%s' (resolved: '%s', id: %s). Grouping as 'other'.",
                            raw_cls_name, cls_name, cls_idx
                        )
                    wires.append({
                        "class_name": "other",
                        "confidence": conf,
                        "mask": seg_res.masks[i],
                        "bbox": seg_res.boxes[i]
                    })
                    severities.append(get_severity(conf))

        if len(wires) == 0:
            transition_state(conn, analysis_id, PipelineState.NO_WIRES_DETECTED, "Zero wires found")
            import cv2 as _cv2_nw
            from pathlib import Path as _Path_nw
            base_outputs_nw = _Path_nw("outputs").resolve()
            base_outputs_nw.mkdir(exist_ok=True)
            vni_res_nw = compute_vni(prep.gray)
            spectrum_path_nw = ""
            if vni_res_nw.spectrum_png:
                spec_file_nw = base_outputs_nw / f"analysis_{analysis_id}_spectrum.png"
                with open(spec_file_nw, "wb") as f:
                    f.write(vni_res_nw.spectrum_png)
                spectrum_path_nw = str(spec_file_nw)
            annotated_img_nw = prep.rgb.copy()
            if pole_bbox:
                px1, py1, px2, py2 = [int(v) for v in pole_bbox]
                _cv2_nw.rectangle(annotated_img_nw, (px1, py1), (px2, py2), (255, 0, 0), 2)
                _cv2_nw.putText(annotated_img_nw, "Pole", (px1, py1 - 10), _cv2_nw.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            for veg in vegetation_detections:
                vx1, vy1, vx2, vy2 = [int(v) for v in veg["bbox"]]
                _cv2_nw.rectangle(annotated_img_nw, (vx1, vy1), (vx2, vy2), (16, 185, 129), 2)
                _cv2_nw.putText(annotated_img_nw, "Vegetation", (vx1, max(15, vy1 - 10)), _cv2_nw.FONT_HERSHEY_SIMPLEX, 0.8, (16, 185, 129), 2)
            annotated_file_nw = base_outputs_nw / f"analysis_{analysis_id}_annotated.jpg"
            _cv2_nw.imwrite(str(annotated_file_nw), _cv2_nw.cvtColor(annotated_img_nw, _cv2_nw.COLOR_RGB2BGR))
            annotated_path_nw = str(annotated_file_nw)
            conn.execute(
                "UPDATE analyses SET status = 'no_wires', vni = ?, spectrum_path = ?, annotated_path = ?, width = ?, height = ? WHERE id = ?",
                (vni_res_nw.vni, spectrum_path_nw, annotated_path_nw, prep.original_shape[1], prep.original_shape[0], analysis_id),
            )
            conn.commit()
            transition_state(conn, analysis_id, PipelineState.GENERATING_REPORT, "Creating empty report")
            # In a real app we'd build a report PDF here
            transition_state(conn, analysis_id, PipelineState.IDLE, "Pipeline finished (no wires)")
            return

        # Optional classical Hough pole fallback in real mode when 4-class model outputs no pole class
        import os as _os
        if pole_bbox is None and not is_mock_mode():
            hough_enabled = (
                _os.getenv("WIREFRAME_HOUGH_POLE_FALLBACK", "0").strip().lower() in ("1", "true", "yes")
                or bool(get_setting("hough_pole_fallback", False))
            )
            if hough_enabled and prep.rgb is not None:
                import cv2 as _cv2
                bgr_for_pole = _cv2.cvtColor(prep.rgb, _cv2.COLOR_RGB2BGR)
                pole_bbox = detect_pole_hough_fallback(bgr_for_pole)
                if pole_bbox is not None and not pole_masks:
                    ph, pw = prep.original_shape[:2]
                    px1, py1, px2, py2 = [max(0, int(round(v))) for v in pole_bbox[:4]]
                    pm = _np_zeros = None
                    import numpy as _np
                    pm = _np.zeros((ph, pw), dtype=_np.uint8)
                    pm[min(ph, py1):min(ph, py2), min(pw, px1):min(pw, px2)] = 255
                    pole_masks.append(pm)

        if check_cancel_requested(conn, batch_id):
            raise InterruptedError("Cancelled before Analyzing")
            
        transition_state(conn, analysis_id, PipelineState.ANALYZING, "Computing VNI and extracting centerline geometry")
        vni_res = compute_vni(prep.gray)
        
        bridge_poles_enabled = (
            _os.getenv("WIREFRAME_BRIDGE_POLES", "0").strip().lower() in ("1", "true", "yes")
            or bool(get_setting("bridge_across_poles", False))
        )
        analyzed_wires = []
        for w in wires:
            w_mask = w["mask"]
            if bridge_poles_enabled and pole_masks and w_mask is not None:
                w_mask = bridge_across_poles(w_mask, pole_masks)
            cline = extract_centerline(w_mask, w["bbox"])
            if cline.is_valid:
                cat = calculate_sag_metrics(cline.polyline)
                w["catenary"] = cat
                w["centerline_flags"] = list(cline.flags)
                w["lowest_y"] = get_lowest_y(cline.polyline)
                w["centerline_json"] = str(cline.polyline)
                w["params_json"] = json.dumps({
                    "a": cat.fitted_a,
                    "residual": cat.residual,
                    "fit_failed": cat.fit_failed,
                    "perp_sag_px": cat.perp_sag_px,
                    "perp_sag_ratio": cat.perp_sag_ratio,
                    "flags": list(dict.fromkeys(list(cline.flags) + list(cat.flags))),
                })
                analyzed_wires.append(w)
                
        if check_cancel_requested(conn, batch_id):
            raise InterruptedError("Cancelled before ComputingRisk")
            
        transition_state(conn, analysis_id, PipelineState.COMPUTING_RISK, "Scoring CRS, scaling sag, assigning safety")
        from backend.formulas import compute_allowable_sag, compute_catenary_risk, compute_paper_crs
        raw_weights = get_setting("class_weights", {}) or {}
        weights: dict[str, float] = {}
        for k, v in raw_weights.items():
            norm_k = normalize_class_name(k)
            weights[norm_k] = v
            weights[normalize_wire_class(norm_k)] = v
        raw_pec = get_setting("pec_min_clearance_by_class", {}) or {}
        pec_min_map: dict[str, float] = {}
        for k, v in raw_pec.items():
            norm_k = normalize_class_name(k)
            pec_min_map[norm_k] = v
            pec_min_map[normalize_wire_class(norm_k)] = v
        
        scale_res = calibrate_scale(pole_bbox=pole_bbox, image_height=prep.original_shape[0])
        
        max_crs = 0.0
        final_wires = []
        for w in analyzed_wires:
            cat = w["catenary"]
            crs_raw = compute_catenary_risk(cat.sag_to_span, cat.residual)
            cls_key = normalize_wire_class(w["class_name"])
            cls_weight = weights.get(cls_key, weights.get(normalize_class_name(w["class_name"]), 1.0))
            crs_weighted = max(0.0, min(100.0, crs_raw * cls_weight))
            max_crs = max(max_crs, crs_weighted)
            
            span_m_est = (cat.span_px * scale_res.meters_per_px) if (scale_res.meters_per_px and cat.span_px) else None
            delta_lim, sag_flags = compute_allowable_sag(cls_key, span_m=span_m_est)
            crs_10, paper_risk_level = compute_paper_crs(
                cat.sag_to_span,
                delta_lim=delta_lim,
                class_weight=cls_weight,
                fit_failed=cat.fit_failed,
            )

            params = parse_params_json(w["params_json"])
            combined_flags = list(dict.fromkeys(list(params.get("flags", [])) + list(sag_flags)))
            params["crs_10"] = crs_10
            params["paper_risk_level"] = paper_risk_level
            params["delta_lim"] = delta_lim
            params["flags"] = combined_flags

            ew = evaluate_wire(
                class_name=cls_key,
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

        for veg in vegetation_detections:
            vx1, vy1, vx2, vy2 = [int(v) for v in veg["bbox"]]
            cv2.rectangle(annotated_img, (vx1, vy1), (vx2, vy2), (16, 185, 129), 2)
            cv2.putText(annotated_img, "Vegetation", (vx1, max(15, vy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (16, 185, 129), 2)
            
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
                   width = ?, height = ?, pole_bbox_json = ?, margin_error_m = ?, is_mock = ?
               WHERE id = ?""",
            (
                vni_res.vni, max_crs, barangay_risk, final_score, dead_wires, spectrum_path, annotated_path,
                clearance_rollup["clearance_ok"], clearance_rollup["clearance_available"], clearance_rollup["pec_min_clearance_m"],
                clearance_rollup["sag_meters"], assumed_pole_height_m,
                prep.original_shape[1], prep.original_shape[0], json.dumps(pole_bbox) if pole_bbox else None, clearance_rollup["margin_error_m"],
                1 if is_mock_mode() else 0,
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
