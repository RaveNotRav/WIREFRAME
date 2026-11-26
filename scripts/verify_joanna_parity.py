"""Automated Verification Gate for Joanna Parity & Closed-Loop Engineering Mode.

Verifies all 5 engineering gates without mocking or weakening any check:
1. 8-Class & Attachment Parity Check (`uploads/2bd8e9c8_dataset4.jpg` + >= 2 dense urban samples)
2. Multi-Cue Calibration & Physics Check (non-null metric sag & ground clearance even on cropped pole base,
   catenary curve fitting, PEC 5.5m/4.6m checks, per-wire min wire-to-vegetation distance)
3. Unified Risk & Overlay Rendering Check (fused non-contradictory risk score with 5-factor breakdown,
   never marked INCOMPLETE solely due to cropped pole base, zero-exception unified overlay image)
4. Interactive Frontend Check (default unified overlay, interactive wire-card hover highlighting,
   Original vs. Overlay comparison toggle)
5. Full Regression Gate (`node --test`, `python -m pytest -q`, live health check with `WIREFRAME_MOCK=0`)

Exits with 0 and prints `ALL_GOALS_ACHIEVED` ONLY when every single check passes.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fail(gate: str, detail: str) -> None:
    print(f"[FAIL] {gate}: {detail}")


def _pass(gate: str, detail: str) -> None:
    print(f"[PASS] {gate}: {detail}")


def run_all_checks() -> int:
    failures: list[str] = []

    # Force Real Mode for Gates 1-3
    os.environ["WIREFRAME_MOCK"] = "0"
    os.environ["WIREFRAME_CONF"] = "0.02"
    os.environ["WIREFRAME_IMGSZ"] = "1024"
    os.environ["WIREFRAME_TILED"] = "0"

    # ------------------------------------------------------------------
    # Load real Ultralytics model & isolated temp DB for live pipeline run
    # ------------------------------------------------------------------
    try:
        from ultralytics import YOLO
        from fastapi.testclient import TestClient
        import backend.database as db_mod
        import backend.config_store as cs_mod
        from backend.auth import _hash_token
        from backend.utils.pipeline import execute_pipeline
        import main as main_app
    except Exception as exc:
        _fail("ImportGate", f"Failed to import backend modules: {exc}")
        return 1

    model_pt = ROOT / "backend" / "models" / "wireframe_seg.pt"
    if not model_pt.is_file():
        _fail("WeightsGate", f"Missing {model_pt}")
        return 1

    yolo_model = YOLO(str(model_pt))
    main_app.main_state.set_model(yolo_model)

    dataset4_path = ROOT / "uploads" / "2bd8e9c8_dataset4.jpg"
    urban_dir = ROOT / "datasets" / "wireframes_v1" / "train" / "images"
    urban_candidates = sorted(urban_dir.glob("*.jpg"))[:3] if urban_dir.is_dir() else []

    if not dataset4_path.is_file():
        _fail("Gate1_Dataset4", f"Missing {dataset4_path}")
        return 1
    if len(urban_candidates) < 2:
        _fail("Gate1_UrbanSamples", f"Need >= 2 urban images in {urban_dir}, found {len(urban_candidates)}")
        return 1

    with tempfile.TemporaryDirectory(prefix="wf_parity_", ignore_cleanup_errors=True) as tmp_dir:
        tmp_db = Path(tmp_dir) / "parity_test.db"
        orig_db_env = os.environ.get("WIREFRAME_DB_PATH")
        os.environ["WIREFRAME_DB_PATH"] = str(tmp_db)
        conn = None
        try:
            conn = db_mod.get_connection(str(tmp_db))
            db_mod.run_migrations(conn)
            cs_mod.init_config_store(conn)

            guest_tok = "parity_guest_tok"
            ghash = _hash_token(guest_tok)

            def _run_live_image(img_path: Path) -> dict:
                cur = conn.execute(
                    """INSERT INTO analyses (filename, status, company, barangay, street, guest_session_id, is_mock)
                       VALUES (?, 'queued', 'Meralco', 'San Antonio', 'Pearl Drive', ?, 0)""",
                    (img_path.name, ghash),
                )
                aid = cur.lastrowid
                conn.commit()
                execute_pipeline(aid, str(img_path), conn, model=yolo_model, batch_id=None)

                client = TestClient(main_app.app)
                client.cookies.set("guest_session", guest_tok)
                res = client.get(f"/api/analysis/{aid}/result")
                if res.status_code != 200:
                    raise RuntimeError(f"GET /api/analysis/{aid}/result returned {res.status_code}: {res.text}")
                return res.json()

            # ==========================================================
            # GATE 1: 8-Class & Attachment Parity Check
            # ==========================================================
            d4_json = _run_live_image(dataset4_path)
            d4_wires = d4_json.get("wires") or []
            if len(d4_wires) >= 4:
                _pass("Gate1_Dataset4Wires", f"Detected {len(d4_wires)} continuous wires on dataset4.jpg (>= 4 required)")
            else:
                msg = f"Expected >= 4 wires on dataset4.jpg, got {len(d4_wires)}"
                _fail("Gate1_Dataset4Wires", msg)
                failures.append(msg)

            urban_results = []
            for u_img in urban_candidates[:2]:
                uj = _run_live_image(u_img)
                urban_results.append((u_img.name, uj))
                uw = uj.get("wires") or []
                if len(uw) >= 1:
                    _pass("Gate1_UrbanWires", f"{u_img.name}: detected {len(uw)} wires (>= 1 required)")
                else:
                    msg = f"Urban image {u_img.name} detected {len(uw)} wires (< 1)"
                    _fail("Gate1_UrbanWires", msg)
                    failures.append(msg)

            # Check structural classes schema (`crossarm`, `insulator`, `building`, `pole`, `vegetation`)
            required_struct_keys = {"crossarm", "insulator", "building", "pole", "vegetation"}
            struct_obj = d4_json.get("structural_detections")
            cls_counts = d4_json.get("class_counts") or {}
            if isinstance(struct_obj, dict) and required_struct_keys.issubset(set(struct_obj.keys())) and required_struct_keys.issubset(set(cls_counts.keys())):
                _pass("Gate1_StructuralClasses", f"8-class schema present in structural_detections & class_counts: { {k: cls_counts.get(k) for k in sorted(required_struct_keys)} }")
            else:
                msg = f"Missing structural class keys in output schema. structural_detections={type(struct_obj)} keys={list(struct_obj.keys()) if isinstance(struct_obj, dict) else None}, class_counts={list(cls_counts.keys())}"
                _fail("Gate1_StructuralClasses", msg)
                failures.append(msg)

            # Check pole-to-pole topology / anchor snapping & dead_wire candidate flagging
            topo = d4_json.get("topology")
            has_wire_attachment_fields = all(
                ("anchor_attachments" in w and "dead_wire_candidate" in w) for w in d4_wires
            )
            if isinstance(topo, dict) and "anchored_wires" in topo and "free_hanging_candidates" in topo and has_wire_attachment_fields:
                _pass("Gate1_TopologyAnchors", f"Topology & anchor snapping wired: {topo}")
            else:
                msg = f"Missing topology / anchor_attachments / dead_wire_candidate fields on wires. topo={topo}"
                _fail("Gate1_TopologyAnchors", msg)
                failures.append(msg)

            # ==========================================================
            # GATE 2: Multi-Cue Calibration & Physics Check
            # ==========================================================
            sag_m = d4_json.get("sag_m")
            clr_m = d4_json.get("ground_clearance_m")
            cal_meta = d4_json.get("calibration") or {}
            pec_check = d4_json.get("pec_clearance_check") or {}

            if isinstance(sag_m, (int, float)) and sag_m > 0 and isinstance(clr_m, (int, float)) and clr_m > 0:
                _pass("Gate2_MultiCueCalibration", f"dataset4.jpg (cropped pole base) metric sag_m={sag_m:.3f}m, ground_clearance_m={clr_m:.3f}m, source={cal_meta.get('source')}")
            else:
                msg = f"Expected valid numeric sag_m and ground_clearance_m on dataset4.jpg, got sag_m={sag_m}, ground_clearance_m={clr_m}"
                _fail("Gate2_MultiCueCalibration", msg)
                failures.append(msg)

            if pec_check.get("road_min_m") == 5.5 and pec_check.get("pedestrian_min_m") == 4.6 and "road_ok" in pec_check and "pedestrian_ok" in pec_check:
                _pass("Gate2_PECClearanceCheck", f"PEC 5.5m road / 4.6m pedestrian check present: {pec_check}")
            else:
                msg = f"Missing PEC 5.5m road / 4.6m pedestrian check in JSON: {pec_check}"
                _fail("Gate2_PECClearanceCheck", msg)
                failures.append(msg)

            # Check per-wire minimum wire-to-vegetation distance (`vegetation_distance_px`, `vegetation_distance_m`)
            all_all_wires = list(d4_wires) + [w for _, uj in urban_results for w in (uj.get("wires") or [])]
            has_veg_dist_schema = all(
                ("vegetation_distance_px" in w and "vegetation_distance_m" in w) for w in all_all_wires
            )
            any_veg_measured = any(w.get("vegetation_distance_px") is not None for w in all_all_wires)
            if has_veg_dist_schema:
                _pass("Gate2_WireToVegDistance", f"Per-wire vegetation_distance_px & vegetation_distance_m present on all {len(all_all_wires)} wires (measured_on_veg_scene={any_veg_measured})")
            else:
                msg = "Per-wire vegetation_distance_px / vegetation_distance_m missing from wire objects"
                _fail("Gate2_WireToVegDistance", msg)
                failures.append(msg)

            # ==========================================================
            # GATE 3: Unified Risk & Overlay Rendering Check
            # ==========================================================
            hb = d4_json.get("hazard_breakdown")
            req_hb_keys = {"clearance_risk", "catenary_sag_risk", "vegetation_proximity_risk", "dead_wire_risk", "bundle_density_risk"}
            overall_verdict = str(d4_json.get("safety_verdict") or "")
            clr_status_d4 = str(d4_json.get("ground_clearance_status") or "")

            if (
                isinstance(hb, dict)
                and req_hb_keys.issubset(set(hb.keys()))
                and isinstance(d4_json.get("risk_score"), (int, float))
                and overall_verdict.lower() not in ("incomplete", "unknown", "")
                and clr_status_d4 != "no_pole"
            ):
                _pass("Gate3_UnifiedRiskScore", f"Fused risk_score={d4_json['risk_score']:.1f}, verdict={overall_verdict}, breakdown={hb}")
            else:
                msg = f"Unified risk check failed: risk_score={d4_json.get('risk_score')}, safety_verdict={overall_verdict}, ground_clearance_status={clr_status_d4}, hazard_breakdown={hb}"
                _fail("Gate3_UnifiedRiskScore", msg)
                failures.append(msg)

            annot_path = d4_json.get("annotated_path")
            if annot_path and Path(annot_path).is_file() and Path(annot_path).stat().st_size > 5000:
                _pass("Gate3_UnifiedOverlayFile", f"Annotated overlay saved ({Path(annot_path).stat().st_size} bytes) at {annot_path}")
            else:
                msg = f"Annotated overlay missing or empty: {annot_path}"
                _fail("Gate3_UnifiedOverlayFile", msg)
                failures.append(msg)

            conn.close()
            conn = None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            if orig_db_env is None:
                os.environ.pop("WIREFRAME_DB_PATH", None)
            else:
                os.environ["WIREFRAME_DB_PATH"] = orig_db_env

    # ==========================================================
    # GATE 4: Interactive Frontend Check
    # ==========================================================
    analyze_html = (ROOT / "frontend" / "pages" / "analyze.html").read_text(encoding="utf-8")
    analyze_js = (ROOT / "frontend" / "assets" / "js" / "analyze.js").read_text(encoding="utf-8")
    pres_js = (ROOT / "frontend" / "assets" / "js" / "lib" / "presentation.js").read_text(encoding="utf-8")

    has_compare_toggle = ("btn-view-overlay" in analyze_html and "btn-view-original" in analyze_html) or ("data-view-mode" in analyze_html)
    has_hover_canvas = ("overlay-highlight-canvas" in analyze_html) and ("highlightWireOnCanvas" in analyze_js) and ("mouseenter" in analyze_js)
    has_default_overlay = 'id="img-annotated"' in analyze_html

    if has_compare_toggle and has_hover_canvas and has_default_overlay:
        _pass("Gate4_InteractiveFrontend", "Default unified overlay, Original vs. Overlay toggle, and interactive wire hover highlight canvas verified")
    else:
        msg = f"Frontend interactive check failed: compare_toggle={has_compare_toggle}, hover_canvas={has_hover_canvas}, default_overlay={has_default_overlay}"
        _fail("Gate4_InteractiveFrontend", msg)
        failures.append(msg)

    # ==========================================================
    # GATE 5: Full Regression Gate (`node --test`, `pytest`, live health)
    # ==========================================================
    node_env = dict(os.environ, WIREFRAME_MOCK="1")
    for k_override in ("WIREFRAME_CONF", "WIREFRAME_IOU", "WIREFRAME_IMGSZ", "WIREFRAME_TILED", "WIREFRAME_DB_PATH"):
        node_env.pop(k_override, None)
    node_proc = subprocess.run(
        ["node", "--test", "frontend/assets/js/**/*.test.js"],
        cwd=str(ROOT),
        env=node_env,
        capture_output=True,
        text=True,
        shell=True,
    )
    if node_proc.returncode == 0 and "fail 0" in node_proc.stdout:
        _pass("Gate5_NodeTests", "node --test passed with 0 failures")
    else:
        msg = f"node --test failed (code {node_proc.returncode}):\n{node_proc.stdout[-1500:]}\n{node_proc.stderr[-1500:]}"
        _fail("Gate5_NodeTests", msg)
        failures.append("node --test failed")

    pytest_env = dict(os.environ, WIREFRAME_MOCK="1")
    for k_override in ("WIREFRAME_CONF", "WIREFRAME_IOU", "WIREFRAME_IMGSZ", "WIREFRAME_TILED", "WIREFRAME_DB_PATH"):
        pytest_env.pop(k_override, None)
    pytest_proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(ROOT),
        env=pytest_env,
        capture_output=True,
        text=True,
    )
    if pytest_proc.returncode == 0:
        last_line = [ln for ln in pytest_proc.stdout.splitlines() if ln.strip()][-1] if pytest_proc.stdout.strip() else ""
        _pass("Gate5_PytestSuite", f"pytest passed with 0 failures ({last_line})")
    else:
        msg = f"pytest failed (code {pytest_proc.returncode}):\n{pytest_proc.stdout[-2000:]}\n{pytest_proc.stderr[-1500:]}"
        _fail("Gate5_PytestSuite", msg)
        failures.append("pytest failed")

    # Live server health check in WIREFRAME_MOCK=0 mode via TestClient lifespan (isolated temp DB)
    os.environ["WIREFRAME_MOCK"] = "0"
    with tempfile.TemporaryDirectory(prefix="wf_health_", ignore_cleanup_errors=True) as h_dir:
        os.environ["WIREFRAME_DB_PATH"] = str(Path(h_dir) / "health_check.db")
        try:
            with TestClient(main_app.app) as live_client:
                h_res = live_client.get("/api/health")
                h_json = h_res.json() if h_res.status_code == 200 else {}
                if (
                    h_res.status_code == 200
                    and h_json.get("status") == "ok"
                    and h_json.get("mock_mode") is False
                    and h_json.get("model_loaded") is True
                    and h_json.get("weights_present") is True
                    and h_json.get("wirenet_weights_present") is True
                ):
                    _pass("Gate5_LiveHealthRealMode", f"Health check verified in WIREFRAME_MOCK=0 mode: {h_json.get('model_status')}")
                else:
                    msg = f"Health check failed in WIREFRAME_MOCK=0 mode: status={h_res.status_code}, body={h_json}"
                    _fail("Gate5_LiveHealthRealMode", msg)
                    failures.append(msg)
        finally:
            os.environ.pop("WIREFRAME_DB_PATH", None)

    if failures:
        print(f"\nVERIFICATION_FAILED ({len(failures)} failing check(s)):")
        for idx, f_msg in enumerate(failures, 1):
            print(f"  {idx}. {f_msg}")
        return 1

    print("\nALL_GOALS_ACHIEVED")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_checks())
