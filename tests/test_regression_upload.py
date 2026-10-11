import time
import hashlib
from fastapi.testclient import TestClient
from main import app
from backend.database import get_connection
from backend.services.worker import start_worker, stop_worker

def setup_test_auth(client):
    conn = get_connection()
    conn.execute("INSERT OR IGNORE INTO users (id, username, password_hash, salt, role, active) VALUES (999, 'admin_test', 'hash', 'salt', 'admin', 1)")
    token_hash = hashlib.sha256(b"test_token").hexdigest()
    conn.execute("INSERT OR REPLACE INTO sessions (token_hash, user_id, expires_at) VALUES (?, 999, datetime('now', '+1 hour'))", (token_hash,))
    conn.commit()
    conn.close()
    client.cookies.set("session", "test_token")

def wait_for_terminal_status(client, analysis_id, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        res = client.get("/api/analysis")
        if res.status_code == 200:
            data = res.json().get("data", [])
            for item in data:
                if item["id"] == analysis_id:
                    if item["status"] in ("complete", "rejected", "failed"):
                        return item
        time.sleep(0.5)
    raise Exception(f"Timeout waiting for terminal status of {analysis_id}")

def test_regression_upload_exact_ui_payload(clean_image):
    import cv2
    image_bytes = cv2.imencode('.jpg', clean_image)[1].tobytes()
    
    with TestClient(app) as client:
        start_worker()
        setup_test_auth(client)
        
        data = {
            "company": "Test Power Co",
            "barangay": "Brgy. San Jose",
            "street": "Rizal St",
            "sector": "",       
            "gps_lat": "",      
            "gps_lng": ""
        }
        files = [
            ("files", ("test_image1.jpg", image_bytes, "image/jpeg")),
            ("files", ("test_image2.jpg", image_bytes, "image/jpeg"))
        ]
        
        response = client.post("/api/batches", data=data, files=files, headers={"X-Requested-With": "XMLHttpRequest"})
        assert response.status_code == 200
        assert "batch_id" in response.json()
        batch_id = response.json()["batch_id"]
        
        conn = get_connection()
        analysis_rows = conn.execute("SELECT id, company, barangay, street FROM analyses WHERE batch_id = ?", (batch_id,)).fetchall()
        conn.close()
        assert len(analysis_rows) == 2, "Should create 2 analyses"
        assert analysis_rows[0]["company"] == "Test Power Co"
        assert analysis_rows[1]["company"] == "Test Power Co"
        assert analysis_rows[0]["barangay"] == "Brgy. San Jose"
        assert analysis_rows[1]["barangay"] == "Brgy. San Jose"
        assert analysis_rows[0]["street"] == "Rizal St"
        assert analysis_rows[1]["street"] == "Rizal St"
        
        analysis_id = analysis_rows[0]["id"]
        
        final_data = wait_for_terminal_status(client, analysis_id)
        assert final_data["status"] in ("complete", "rejected", "no_wires")
        
        conn = get_connection()
        events = conn.execute("SELECT * FROM pipeline_events WHERE analysis_id = ?", (analysis_id,)).fetchall()
        assert len(events) > 0, "Pipeline events should be logged"
        
        wires = conn.execute("SELECT * FROM detections WHERE analysis_id = ?", (analysis_id,)).fetchall()
        if final_data["status"] == "complete":
            assert len(wires) > 0, "Complete status should have wires"
        elif final_data["status"] == "no_wires":
            assert len(wires) == 0, "No wires status should have 0 wires"
        conn.close()
        stop_worker()


def test_regression_upload_invalid_gps(clean_image):
    import cv2
    image_bytes = cv2.imencode('.jpg', clean_image)[1].tobytes()
    with TestClient(app) as client:
        setup_test_auth(client)
        
        data = {
            "company": "Test Power Co",
            "barangay": "Brgy. San Jose",
            "street": "Rizal St",
            "gps_lat": "invalid",
            "gps_lng": "coord"
        }
        files = [("files", ("test_image.jpg", image_bytes, "image/jpeg"))]
        
        response = client.post("/api/batches", data=data, files=files, headers={"X-Requested-With": "XMLHttpRequest"})
        assert response.status_code == 422
        assert "gps" in response.text.lower() or "float" in response.text.lower()


def test_regression_upload_missing_required(clean_image):
    import cv2
    image_bytes = cv2.imencode('.jpg', clean_image)[1].tobytes()
    with TestClient(app) as client:
        setup_test_auth(client)
        
        data = {
            "gps_lat": "",
            "gps_lng": ""
        }
        files = [("files", ("test_image.jpg", image_bytes, "image/jpeg"))]
        
        response = client.post("/api/batches", data=data, files=files, headers={"X-Requested-With": "XMLHttpRequest"})
        assert response.status_code == 422
        assert "company" in response.text.lower() or "barangay" in response.text.lower() or "street" in response.text.lower()
