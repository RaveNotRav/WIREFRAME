import os
from backend.config_store import DEFAULT_SETTINGS, get_setting, set_setting


def test_config_whitelist(client_admin):
    # Valid
    res = client_admin.put("/api/config", json={"pec_min_clearance_by_class": {"default": 6.0}})
    assert res.status_code == 200
    
    # Invalid key
    res = client_admin.put("/api/config", json={"unknown_key": "hacked"})
    assert res.status_code == 422
    assert "Unknown or non-editable" in res.json()["detail"]


def test_config_risk_bands_list_of_dicts_and_canonical_classes(client_admin):
    orig_bands = get_setting("risk_bands")
    orig_weights = get_setting("class_weights")
    orig_pec = get_setting("pec_min_clearance_by_class")
    try:
        new_bands = [
            {"min": 0.00, "max": 0.30, "label": "Low"},
            {"min": 0.31, "max": 0.65, "label": "Moderate"},
            {"min": 0.66, "max": 0.85, "label": "High"},
            {"min": 0.86, "max": 1.00, "label": "Critical"},
        ]
        res = client_admin.put(
            "/api/config",
            json={
                "risk_bands": new_bands,
                "site_risk_weights": {"vni": 0.45, "crs": 0.45, "dead": 0.10},
                "severity_thresholds": {"high_confidence_min": 0.7, "medium_confidence_min": 0.4},
                "class_weights": {
                    "power_distribution_line": 1.2,
                    "communication_cable": 1.0,
                    "dead_wire": 1.5,
                    "vegetation": 1.0,
                },
                "pec_min_clearance_by_class": {
                    "power_distribution_line": 6.0,
                    "communication_cable": 5.5,
                    "dead_wire": 5.5,
                    "vegetation": 5.5,
                    "default": 5.5,
                },
            },
        )
        assert res.status_code == 200
        assert get_setting("risk_bands") == new_bands
        assert get_setting("class_weights")["power_distribution_line"] == 1.2
        assert get_setting("pec_min_clearance_by_class")["power_distribution_line"] == 6.0
    finally:
        set_setting("risk_bands", orig_bands or DEFAULT_SETTINGS["risk_bands"])
        set_setting("class_weights", orig_weights or DEFAULT_SETTINGS["class_weights"])
        set_setting("pec_min_clearance_by_class", orig_pec or DEFAULT_SETTINGS["pec_min_clearance_by_class"])


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
