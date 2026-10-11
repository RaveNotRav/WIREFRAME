
def test_rbac_config_admin_only(client_admin, client_engineer, client_inspector):
    # Admin can get config
    assert client_admin.get("/api/config").status_code == 200
    
    # Engineer cannot
    assert client_engineer.get("/api/config").status_code == 403
    
    # Inspector cannot
    assert client_inspector.get("/api/config").status_code == 403

def test_rbac_dashboard_engineer_admin(client_admin, client_engineer, client_inspector):
    # Admin and Engineer can access ranking
    assert client_admin.get("/api/mpc/ranking").status_code == 200
    assert client_engineer.get("/api/mpc/ranking").status_code == 200
    
    # Inspector cannot
    assert client_inspector.get("/api/mpc/ranking").status_code == 403
