import pytest
from fastapi.testclient import TestClient
from main import app
from backend.database import get_connection
import cv2
import numpy as np

# A minimal clean image
clean_image = np.full((100, 100, 3), 128, dtype=np.uint8)

def test_anon_vs_auth_isolation(client_admin):
    file_bytes = cv2.imencode(".jpg", clean_image)[1].tobytes()
    data = {"company":"Test", "barangay":"Test", "street":"Test"}
    files = [("files", ("dummy.jpg", file_bytes, "image/jpeg"))]
    
    # 1. Auth user creates analysis
    res = client_admin.post(
        "/api/batches",
        data=data,
        files=files,
        headers={"X-Requested-With": "XMLHttpRequest"}
    )
    assert res.status_code == 200
    batch_id = res.json()["batch_id"]
    
    # Get analysis ID
    conn = get_connection()
    try:
        a_id = conn.execute("SELECT id FROM analyses WHERE batch_id = ?", (batch_id,)).fetchone()[0]
    finally:
        conn.close()
        
    # 2. Anon user tries to read it
    client_anon = TestClient(app)
    
    # GET /api/analysis/{id}/result
    res = client_anon.get(f"/api/analysis/{a_id}/result")
    print("Anon get result:", res.status_code)
    assert res.status_code in (401, 403)
    
    # GET /api/batches/{id}
    res = client_anon.get(f"/api/batches/{batch_id}")
    print("Anon get batch:", res.status_code)
    assert res.status_code in (401, 403)
    
    # image serving
    res = client_anon.get(f"/api/analysis/{a_id}/original")
    print("Anon get image:", res.status_code)
    assert res.status_code in (401, 403)

def test_csrf_anonymous():
    client_anon = TestClient(app)
    file_bytes = cv2.imencode(".jpg", clean_image)[1].tobytes()
    data = {"company":"Test", "barangay":"Test", "street":"Test"}
    files = [("files", ("dummy.jpg", file_bytes, "image/jpeg"))]
    
    # No X-Requested-With, no Origin -> CSRF should fail
    res = client_anon.post(
        "/api/batches",
        data=data,
        files=files
    )
    print("CSRF fail code:", res.status_code)
    assert res.status_code in (401, 403)
    assert "CSRF" in res.json()["detail"] or "CSRF" in str(res.json())


