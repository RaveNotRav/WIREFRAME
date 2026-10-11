import json
import csv
import io
from backend.database import get_connection

def test_export_geojson_success(client_admin):
    conn = get_connection()
    conn.execute("INSERT INTO analyses (id, filename, status, created_by, gps_lat, gps_lng, max_crs, vni) VALUES (777, 'test.jpg', 'complete', 1, 14.5995, 120.9842, 50.0, 1.2)")
    conn.execute("INSERT INTO wires (analysis_id, class_name, clearance_m) VALUES (777, 'comm', 4.5)")
    conn.commit()
    conn.close()
    
    payload = {"analysis_ids": [777], "include_mock": True}
    r = client_admin.post("/api/reports/geojson", json=payload, headers={"X-Requested-With": "XMLHttpRequest"})
    
    fc = r.json()
    print("GEOJSON:", fc)
    f = fc["features"][0]
    props = f["properties"]
    assert props["ground_clearance_m"] == 4.5

def test_export_csv_success(client_admin):
    pass

