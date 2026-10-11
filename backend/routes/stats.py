import sqlite3
from typing import List, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query, HTTPException
from backend.auth import User, get_current_user, require_role
from backend.database import get_connection

router = APIRouter(prefix="/api/stats", tags=["stats"])

@router.get("/dashboard", dependencies=[Depends(require_role(["admin", "lgu_engineer"]))])
async def get_dashboard_stats(
    include_mock: bool = Query(False, description="Include mock analyses"),
    days: int = Query(30, ge=1, le=90, description="Number of days for trend"),
    limit: int = Query(48, ge=1, le=100, description="Max cells for heatmap")
):
    conn = get_connection()
    try:
        # Check if we need to filter by mock
        mock_clause = ""
        if not include_mock:
            mock_clause = "AND is_mock = 0"
        
        # 1. KPIs
        kpi_query = f"""
            SELECT 
                COUNT(id) as total_analyses,
                SUM(CASE WHEN EXISTS (
                    SELECT 1 FROM wires w 
                    WHERE w.analysis_id = analyses.id 
                    AND w.safety_level IN ('Unsafe', 'Critical')
                ) THEN 1 ELSE 0 END) as high_critical_count,
                AVG(risk_score) as average_risk_score,
                SUM(CASE WHEN date(created_at, '+8 hours') = date('now', '+8 hours') THEN 1 ELSE 0 END) as processed_today
            FROM analyses
            WHERE status = 'complete' {mock_clause}
        """
        kpi_row = conn.execute(kpi_query).fetchone()
        
        kpis = {
            "total_analyses": kpi_row["total_analyses"] or 0,
            "high_critical_count": kpi_row["high_critical_count"] or 0,
            "average_risk_score": round(kpi_row["average_risk_score"] or 0.0, 1),
            "processed_today": kpi_row["processed_today"] or 0
        }
        
        # 2. Distribution (Wire Classes)
        dist_query = f"""
            SELECT d.class_name, COUNT(d.id) as count
            FROM detections d
            JOIN analyses a ON d.analysis_id = a.id
            WHERE a.status = 'complete' {mock_clause}
            GROUP BY d.class_name
        """
        dist_rows = conn.execute(dist_query).fetchall()
        distribution = {
            "power_line": 0,
            "telecom_cable": 0,
            "fiber_optic": 0,
            "dead_abandoned": 0
        }
        for r in dist_rows:
            if r["class_name"] in distribution:
                distribution[r["class_name"]] = r["count"]
            else:
                distribution[r["class_name"]] = r["count"] # Captures any other classes dynamically just in case

        # 3. Heatmap
        heatmap_query = f"""
            SELECT 
                a.id as analysis_id,
                a.site_id,
                a.risk_score,
                s.name as site_name,
                a.created_at
            FROM analyses a
            LEFT JOIN sites s ON a.site_id = s.id
            WHERE a.status = 'complete' {mock_clause}
            ORDER BY a.created_at DESC
            LIMIT ?
        """
        heatmap_rows = conn.execute(heatmap_query, (limit,)).fetchall()
        
        from backend.config_store import get_setting
        bands = get_setting("risk_bands")
        
        heatmap = []
        for r in heatmap_rows:
            score = r["risk_score"]
            band = "Unknown"
            if score is not None:
                val = score / 100.0
                for b in bands:
                    if val <= b["max"]:
                        band = b["label"]
                        break
                if band == "Unknown" and bands:
                    band = bands[-1]["label"]
            
            heatmap.append({
                "analysis_id": r["analysis_id"],
                "site_id": r["site_id"],
                "site_name": r["site_name"] or f"Analysis #{r['analysis_id']}",
                "risk_score": score,
                "risk_band": band,
                "created_at": r["created_at"]
            })
            
        # 4. Trend
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        trend_query = f"""
            SELECT 
                date(created_at, '+8 hours') as trend_date,
                COUNT(id) as count,
                AVG(risk_score) as avg_risk
            FROM analyses
            WHERE status = 'complete' AND created_at >= ? {mock_clause}
            GROUP BY trend_date
            ORDER BY trend_date ASC
        """
        trend_rows = conn.execute(trend_query, (cutoff_date,)).fetchall()
        trend = [
            {
                "date": r["trend_date"],
                "count": r["count"],
                "avg_risk": round(r["avg_risk"] or 0.0, 1)
            }
            for r in trend_rows
        ]
        
        return {
            "kpis": kpis,
            "distribution": distribution,
            "heatmap": heatmap,
            "trend": trend
        }
    finally:
        conn.close()
