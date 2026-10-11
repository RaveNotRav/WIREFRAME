import hashlib
import json
import os
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
import csv
import io

from backend import main_state
from backend.auth import User, get_current_user, get_optional_user, verify_analysis_access, require_role, verify_csrf
from backend.database import get_connection
from backend.utils.report_pdf import build_pdf, build_summary_pdf, format_pht_timestamp

router = APIRouter(prefix="/api", tags=["reporting"])

@router.get("/mpc/ranking", dependencies=[Depends(require_role(["lgu_engineer", "admin"]))])
async def get_ranking():
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT s.name as company, s.barangay, s.street, AVG(a.risk_score) as avg_risk, COUNT(a.id) as analysis_count
            FROM analyses a
            JOIN sites s ON a.site_id = s.id
            WHERE a.status = 'complete'
            GROUP BY s.id, s.barangay, s.street
            ORDER BY avg_risk DESC
        """).fetchall()
        return {"rankings": [dict(r) for r in rows]}
    finally:
        conn.close()

def _get_analysis_data(conn, analysis_id: int):
    a = conn.execute("""
        SELECT a.*, s.name as site_name
        FROM analyses a
        LEFT JOIN sites s ON a.site_id = s.id
        WHERE a.id = ?
    """, (analysis_id,)).fetchone()
    if not a:
        return None
        
    wires = conn.execute("SELECT * FROM wires WHERE analysis_id = ? ORDER BY id", (analysis_id,)).fetchall()
    calibrations = conn.execute("SELECT * FROM calibrations WHERE analysis_id = ? ORDER BY id", (analysis_id,)).fetchall()
    
    data = dict(a)
    data['site'] = data.get('site_name') or 'Unknown'
    data['wires'] = [dict(w) for w in wires]
    data['calibrations'] = [dict(c) for c in calibrations]
    
    # settings
    settings = conn.execute("SELECT key, value_json FROM settings").fetchall()
    settings_dict = {row['key']: json.loads(row['value_json']) for row in settings}
    data['settings'] = settings_dict
    
    action_mapping = settings_dict.get('action_mapping', {})
    
    # Set safety level properly based on wires if complete
    safety_level = 'Unknown'
    if data['status'] == 'complete':
        # logic from safety presentation
        w_safeties = [w['safety_level'] for w in data['wires']]
        if 'Critical' in w_safeties:
            safety_level = 'Critical'
        elif 'Unsafe' in w_safeties:
            safety_level = 'Unsafe'
        elif 'Caution' in w_safeties:
            safety_level = 'Caution'
        elif len(data['wires']) > 0:
            safety_level = 'Safe'
            
    if data['status'] == 'no_wires':
        safety_level = 'Safe'
        
    data['safety_level'] = safety_level
    data['action_text'] = action_mapping.get(safety_level, '')
    
    reasons = []
    if data['status'] == 'no_wires':
        reasons.append("No wires detected in the image.")
    elif safety_level == 'Critical':
        reasons.append("One or more wires have critical clearance issues.")
    
    data['reasons'] = reasons
    
    data['config'] = {
        'pole_height_m': data.get('assumed_pole_height_m'),
        'pec_min': settings_dict.get('pec_min_clearance_by_class', {}).get('default', 5.5),
        'margin': 0.5
    }
    data['scale_source'] = "Manual Calibration" if data['calibrations'] else "Assumed Pole Height"
    
    model = main_state.get_model()
    data['model_file'] = os.path.basename(model.model.pt_path) if model else "mock"
    if data['is_mock']:
        data['model_file'] = "mock"
        
    return data

TEMPLATE_VERSION = "v3"

def _hash_analysis(data):
    # Hash the parts that affect the PDF
    h = hashlib.sha256()
    h.update(str(data.get('updated_at', '')).encode('utf-8'))
    h.update(str(data.get('status')).encode('utf-8'))
    h.update(str(len(data.get('wires', []))).encode('utf-8'))
    h.update(str(len(data.get('calibrations', []))).encode('utf-8'))
    h.update(str(data.get('settings', {})).encode('utf-8'))
    h.update(str(data.get('is_mock')).encode('utf-8'))
    h.update(TEMPLATE_VERSION.encode('utf-8')) # template version
    return h.hexdigest()

@router.get("/reports/{analysis_id}.pdf")
def get_pdf_report(analysis_id: int, req: Request, user: User | None = Depends(get_optional_user)):
    conn = get_connection()
    try:
        a = conn.execute("SELECT created_by, status, guest_session_id FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        if not a:
            raise HTTPException(status_code=404, detail="Not found")
            
        verify_analysis_access(a, user, req)
            
        status = a["status"]
        if status in ["queued", "processing"]:
            raise HTTPException(status_code=409, detail="Analysis is currently in progress. Cannot generate report yet.")
        if status in ["rejected", "error", "interrupted"]:
            # Need sanitized reason
            reason = "Unknown error"
            event = conn.execute("SELECT detail FROM pipeline_events WHERE analysis_id = ? AND state = ? ORDER BY id DESC LIMIT 1", (analysis_id, status)).fetchone()
            if event and event["detail"]:
                reason = event["detail"]
            raise HTTPException(status_code=409, detail=f"Analysis failed ({status}): {reason}")
            
        # must be complete or no_wires
        data = _get_analysis_data(conn, analysis_id)
        if not data:
            raise HTTPException(status_code=404, detail="Not found")
            
        current_hash = _hash_analysis(data)
        
        # Check cache
        rep = conn.execute("SELECT path, metadata FROM reports WHERE analysis_id = ?", (analysis_id,)).fetchone()
        
        pdf_path = None
        if rep:
            meta = json.loads(rep["metadata"] or "{}")
            if meta.get("source_version") == current_hash and os.path.exists(rep["path"]):
                pdf_path = rep["path"]
                
        if not pdf_path:
            # Generate
            out_dir = os.path.join("outputs", "reports")
            os.makedirs(out_dir, exist_ok=True)
            
            temp_path = os.path.join(out_dir, f"temp_{analysis_id}.pdf")
            final_path = os.path.join(out_dir, f"report_{analysis_id}.pdf")
            
            build_pdf(data, temp_path)
            os.replace(temp_path, final_path)
            
            pdf_path = final_path
            
            meta_json = json.dumps({"source_version": current_hash})
            if rep:
                conn.execute("UPDATE reports SET path = ?, metadata = ?, created_at = datetime('now') WHERE analysis_id = ?", (pdf_path, meta_json, analysis_id))
            else:
                conn.execute("INSERT INTO reports (analysis_id, path, metadata) VALUES (?, ?, ?)", (analysis_id, pdf_path, meta_json))
            conn.commit()
            
        return FileResponse(
            pdf_path, 
            media_type="application/pdf", 
            filename=f"wireframe-report-{analysis_id}.pdf",
            headers={"Cache-Control": "private, no-store"}
        )
    finally:
        conn.close()

class SummaryRequest(BaseModel):
    analysis_ids: list[int]
    include_mock: bool = False

@router.post("/reports/summary", dependencies=[Depends(verify_csrf)])
def generate_summary(payload: SummaryRequest, user: User = Depends(get_current_user)):
    if len(payload.analysis_ids) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 analyses allowed for summary")
        
    conn = get_connection()
    try:
        items = []
        has_mock = False
        
        for aid in payload.analysis_ids:
            data = _get_analysis_data(conn, aid)
            if not data:
                raise HTTPException(status_code=404, detail=f"Analysis {aid} not found")
                
            if user.role == "field_inspector" and data["created_by"] != user.id:
                raise HTTPException(status_code=403, detail=f"Access denied for analysis {aid}")
                
            if user.role in ["lgu_engineer", "admin"] and data["created_by"] is None:
                pass # Legacy allowed for lgu/admin
            elif data["created_by"] is None and user.role == "field_inspector":
                raise HTTPException(status_code=403, detail=f"Access denied for analysis {aid}")
                
            if data["status"] not in ["complete", "no_wires"]:
                raise HTTPException(status_code=409, detail=f"Analysis {aid} is not complete")
                
            if data["is_mock"]:
                if not payload.include_mock:
                    continue
                has_mock = True
                
            items.append({
                "id": data["id"],
                "site": data.get("site", ""),
                "company": data.get("company", ""),
                "barangay": data.get("barangay", ""),
                "street": data.get("street", ""),
                "date": format_pht_timestamp(data.get("created_at"), include_time=False),
                "vni": data.get("vni", "--"),
                "max_crs": data.get("max_crs", "--"),
                "safety_level": data["safety_level"],
                "action": data["action_text"]
            })
            
        # Sort by safety
        risk_map = {"Critical": 4, "Unsafe": 3, "Caution": 2, "Safe": 1, "Unknown": 0}
        items.sort(key=lambda x: (risk_map.get(x["safety_level"], 0), x.get("max_crs") or 0), reverse=True)
        
        out_dir = os.path.join("outputs", "reports")
        os.makedirs(out_dir, exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        temp_path = os.path.join(out_dir, f"temp_summary_{ts}.pdf")
        final_path = os.path.join(out_dir, f"summary_{ts}.pdf")
        
        summary_data = {
            "has_mock": has_mock,
            "items": items
        }
        
        build_summary_pdf(summary_data, temp_path)
        os.replace(temp_path, final_path)
        
        return FileResponse(
            final_path, 
            media_type="application/pdf", 
            filename=f"wireframe-summary-{ts}.pdf",
            headers={"Cache-Control": "private, no-store"}
        )
    finally:
        conn.close()
@router.post("/reports/geojson", dependencies=[Depends(verify_csrf)])
def export_geojson(payload: SummaryRequest, user: User = Depends(get_current_user)):
    if len(payload.analysis_ids) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 analyses allowed for export")
        
    conn = get_connection()
    try:
        features = []
        
        for aid in payload.analysis_ids:
            data = _get_analysis_data(conn, aid)
            if not data:
                continue
            
            if user.role == "field_inspector" and data["created_by"] != user.id:
                if data["created_by"] is not None:
                    continue
            
            if data["status"] not in ["complete", "no_wires"]:
                continue
                
            if data["is_mock"] and not payload.include_mock:
                continue
                
            lat = data.get("gps_lat")
            lng = data.get("gps_lng")
            if lat is None or lng is None:
                continue
            
            wires = data.get("wires", [])
            gc_vals = [w.get("clearance_m") for w in wires if w.get("clearance_m") is not None]
            ground_clearance_m = min(gc_vals) if gc_vals else None
            
            props = {
                "id": data.get("id"),
                "company": data.get("company"),
                "barangay": data.get("barangay"),
                "street": data.get("street"),
                "status": data.get("status"),
                "verdict": data.get("status"), # Mock verdict
                "crs": data.get("max_crs"),
                "ground_clearance_m": ground_clearance_m,
                "vni": data.get("vni"),
                "created_at": format_pht_timestamp(data.get("created_at"), include_time=True)
            }
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lng), float(lat)]
                },
                "properties": props
            }
            features.append(feature)
            
        fc = {
            "type": "FeatureCollection",
            "features": features
        }
        
        return Response(content=json.dumps(fc), media_type="application/geo+json")
    finally:
        conn.close()

@router.post("/reports/csv", dependencies=[Depends(verify_csrf)])
def export_csv(payload: SummaryRequest, user: User = Depends(get_current_user)):
    if len(payload.analysis_ids) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 analyses allowed for export")
        
    conn = get_connection()
    try:
        rows = []
        
        for aid in payload.analysis_ids:
            data = _get_analysis_data(conn, aid)
            if not data:
                continue
                
            if user.role == "field_inspector" and data["created_by"] != user.id:
                if data["created_by"] is not None:
                    continue
            
            if data["status"] not in ["complete", "no_wires"]:
                continue
                
            if data["is_mock"] and not payload.include_mock:
                continue
                
            wires = data.get("wires", [])
            gc_vals = [w.get("clearance_m") for w in wires if w.get("clearance_m") is not None]
            ground_clearance_m = min(gc_vals) if gc_vals else None
            
            rows.append({
                "id": data.get("id"),
                "company": data.get("company"),
                "barangay": data.get("barangay"),
                "street": data.get("street"),
                "status": data.get("status"),
                "verdict": data.get("status"), # Mock verdict
                "crs": data.get("max_crs"),
                "ground_clearance_m": ground_clearance_m,
                "vni": data.get("vni"),
                "created_at": format_pht_timestamp(data.get("created_at"), include_time=True),
                "gps_lat": data.get("gps_lat"),
                "gps_lng": data.get("gps_lng")
            })
            
        output = io.StringIO()
        if rows:
            writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            
        return Response(content=output.getvalue(), media_type="text/csv")
    finally:
        conn.close()

