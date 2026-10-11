import pytest
from fastapi.testclient import TestClient
from main import app
from backend.database import get_connection

def test_anonymous_flow(clean_image):
    import cv2
    file_bytes = cv2.imencode('.jpg', clean_image)[1].tobytes()
    with TestClient(app) as client:
        data = {"company":"Test", "barangay":"Test", "street":"Test"}
        files = [("files", ("dummy.jpg", file_bytes, "image/jpeg"))]
        headers = {"X-Requested-With": "XMLHttpRequest"}
        
        resp = client.post("/api/batches", data=data, files=files, headers=headers)
        
        assert resp.status_code == 200, resp.text
        resp_data = resp.json()
        assert "batch_id" in resp_data
        batch_id = resp_data["batch_id"]
        
        assert "guest_session" in client.cookies
        
        poll_resp = client.get(f"/api/batches/{batch_id}")
        assert poll_resp.status_code == 200
        
        conn = get_connection()
        a_id = conn.execute("SELECT id FROM analyses WHERE batch_id = ?", (batch_id,)).fetchone()[0]
        conn.close()
        
        res = client.get(f"/api/analysis/{a_id}/result")
        assert res.status_code == 200
        
        with TestClient(app) as client2:
            res2 = client2.get(f"/api/analysis/{a_id}/result")
            assert res2.status_code == 403
            
            poll_resp2 = client2.get(f"/api/batches/{batch_id}")
            assert poll_resp2.status_code == 403

def test_authenticated_routes_reject_anon():
    with TestClient(app) as client:
        res = client.get("/api/analysis")
        assert res.status_code == 401