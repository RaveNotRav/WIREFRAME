import io
import time
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.services import worker
from backend.services.worker import start_worker, stop_worker
from backend.database import get_connection


def create_test_jpeg(seed: int) -> bytes:
    # 640x480 textured image that passes blur (>50) and brightness (>15) checks
    rng = np.random.RandomState(seed)
    base = rng.randint(40, 200, (480, 640, 3)).astype(np.uint8)
    for y in range(50, 400, 40):
        cv2.line(base, (10, y), (630, y + 20), (255, 255, 255), 2)
    success, encoded = cv2.imencode(".jpg", base)
    assert success
    return encoded.tobytes()


def wait_for_terminal_status(client: TestClient, analysis_id: int, timeout: float = 20.0):
    start = time.time()
    last_res = None
    while time.time() - start < timeout:
        res = client.get("/api/analysis?include_invalid=true&is_mock=1")
        if res.status_code == 200:
            last_res = res.json()
            for item in last_res.get("data", []):
                if item["id"] == analysis_id:
                    if item["status"] in ("complete", "rejected", "no_wires", "error", "interrupted"):
                        return item
        time.sleep(0.2)
    conn = get_connection()
    db_items = conn.execute("SELECT id, status, batch_id, current_state FROM analyses").fetchall()
    db_batches = conn.execute("SELECT id, status FROM batches").fetchall()
    conn.close()
    raise TimeoutError(f"Analysis {analysis_id} did not reach terminal status. In API: {last_res}. In DB analyses: {[dict(x) for x in db_items]}, batches: {[dict(x) for x in db_batches]}")


def ensure_worker_started():
    from backend.services import worker
    worker.stop_worker()
    while not worker.batch_queue.empty():
        try:
            worker.batch_queue.get_nowait()
            worker.batch_queue.task_done()
        except Exception:
            break
    worker.start_worker()


def test_submit_3_sequential_different_images(client_admin: TestClient, _init_settings):
    ensure_worker_started()
    img1 = create_test_jpeg(100)
    img2 = create_test_jpeg(150)
    img3 = create_test_jpeg(200)

    ids = []
    for i, img_bytes in enumerate([img1, img2, img3], start=1):
        data = {
            "company": f"Company {i}",
            "barangay": f"Barangay {i}",
            "street": f"Street {i}"
        }
        files = {"file": (f"test_seq_{i}.jpg", img_bytes, "image/jpeg")}
        resp = client_admin.post("/api/analysis/upload", data=data, files=files)
        assert resp.status_code == 200, resp.text
        res_json = resp.json()
        assert "analysis_id" in res_json
        ids.append(res_json["analysis_id"])

    print("After uploads - worker thread alive:", worker._worker_thread.is_alive() if worker._worker_thread else None)
    print("After uploads - queue size:", worker.batch_queue.qsize())

    assert len(ids) == 3
    assert len(set(ids)) == 3, "Each upload must return a distinct analysis ID"

    # Wait for all 3 to reach terminal status
    for aid in ids:
        item = wait_for_terminal_status(client_admin, aid)
        assert item["status"] in ("complete", "rejected", "no_wires")

    # Check DB rows
    conn = _init_settings
    rows = conn.execute("SELECT id, filename, status, is_mock FROM analyses WHERE id IN (?, ?, ?)", tuple(ids)).fetchall()
    assert len(rows) == 3, f"Expected 3 DB rows for the 3 sequential uploads, got {len(rows)}"

    # Check /api/analysis for admin
    list_resp = client_admin.get("/api/analysis")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    print("Default /api/analysis total:", list_data.get("total"), "count:", len(list_data.get("data", [])))

    # Check /api/analysis?is_mock=1
    mock_resp = client_admin.get("/api/analysis?is_mock=1")
    assert mock_resp.status_code == 200
    mock_data = mock_resp.json()
    print("Mock /api/analysis?is_mock=1 total:", mock_data.get("total"), "count:", len(mock_data.get("data", [])))

    # Check /api/stats/dashboard
    stats_default = client_admin.get("/api/stats/dashboard")
    assert stats_default.status_code == 200
    print("Default /api/stats total_analyses:", stats_default.json()["kpis"]["total_analyses"])

    stats_mock = client_admin.get("/api/stats/dashboard?include_mock=true")
    assert stats_mock.status_code == 200
    print("Mock /api/stats?include_mock=true total_analyses:", stats_mock.json()["kpis"]["total_analyses"])


def test_submit_3_batch_different_images(client_admin: TestClient, _init_settings):
    ensure_worker_started()
    img1 = create_test_jpeg(80)
    img2 = create_test_jpeg(120)
    img3 = create_test_jpeg(160)

    data = {
        "company": "Batch Company",
        "barangay": "Batch Barangay",
        "street": "Batch Street"
    }
    files = [
        ("files", ("batch_1.jpg", img1, "image/jpeg")),
        ("files", ("batch_2.jpg", img2, "image/jpeg")),
        ("files", ("batch_3.jpg", img3, "image/jpeg")),
    ]
    resp = client_admin.post("/api/batches", data=data, files=files)
    assert resp.status_code == 200, resp.text
    batch_id = resp.json()["batch_id"]

    conn = _init_settings
    rows = conn.execute("SELECT id, filename, status FROM analyses WHERE batch_id = ?", (batch_id,)).fetchall()
    assert len(rows) == 3, f"Expected 3 DB rows for the batch, got {len(rows)}"
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == 3

    for aid in ids:
        item = wait_for_terminal_status(client_admin, aid)
        assert item["status"] in ("complete", "rejected", "no_wires")

    list_resp = client_admin.get("/api/analysis?is_mock=1")
    assert list_resp.status_code == 200
    found_ids = {item["id"] for item in list_resp.json()["data"]}
    for aid in ids:
        assert aid in found_ids
