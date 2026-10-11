import os


def test_config_whitelist(client_admin):
    # Valid
    res = client_admin.put("/api/config", json={"pec_min_clearance_by_class": {"default": 6.0}})
    assert res.status_code == 200
    
    # Invalid key
    res = client_admin.put("/api/config", json={"unknown_key": "hacked"})
    assert res.status_code == 422
    assert "Unknown or non-editable" in res.json()["detail"]

def test_config_model_path_validation(client_admin):
    # Valid
    # Create a dummy model
    os.makedirs("backend/models", exist_ok=True)
    with open("backend/models/dummy.pt", "w") as f:
        f.write("dummy")
        
    # Since we are mocking ultralytics YOLO inside update_config via is_mock_mode(), it won't crash on invalid .pt file.
    # We just need to check the path validation
    res = client_admin.put("/api/config", json={"model_path": "backend/models/dummy.pt"})
    assert res.status_code == 200
    
    # Invalid extension
    res = client_admin.put("/api/config", json={"model_path": "backend/models/dummy.txt"})
    assert res.status_code == 422
    
    # Path traversal
    res = client_admin.put("/api/config", json={"model_path": "backend/models/../../dummy.pt"})
    assert res.status_code == 422
    
    os.remove("backend/models/dummy.pt")

def test_config_auth(client_inspector, client_admin):
    # inspector cannot access /api/config
    res = client_inspector.get("/api/config")
    assert res.status_code == 403
    
    # admin can access /api/config
    res = client_admin.get("/api/config")
    assert res.status_code == 200
    assert "model_path" in res.json()
    
    # inspector can access /api/config/public
    res = client_inspector.get("/api/config/public")
    assert res.status_code == 200
    data = res.json()
    assert "pec_min_clearance_by_class" in data
    assert "model_path" not in data
    assert "users" not in data
