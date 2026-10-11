"""Checkpoint CLI for Phase 4."""

import os

import cv2
import numpy as np

from backend.config_store import init_config_store
from backend.database import get_connection, run_migrations
from backend.utils.pipeline import execute_pipeline


class FakeSegmenter:
    """Deterministic fake segmenter as a test double, independent of MOCK mode."""
    def __init__(self, mode="normal"):
        self.mode = mode
        
    def __call__(self, image, **kwargs):
        class MockResult:
            pass
        res = MockResult()
        
        if self.mode == "no_wires":
            res.mask_available = False
            return [res]
            
        res.mask_available = True
        
        # wire box
        box_wire = [50, 100, 590, 310]
        # pole box
        box_pole = [10, 400, 30, 500]
        
        mask_wire = np.zeros((640, 640), dtype=np.uint8)
        # draw fake wire
        for x in range(50, 590, 10):
            cv2.circle(mask_wire, (x, 200), 4, 1, -1)
            
        class MockMasks:
            pass
        m = MockMasks()
        m.data = np.stack([mask_wire, np.zeros_like(mask_wire)], axis=0)
        res.masks = m
        
        class MockBoxes:
            pass
        b = MockBoxes()
        b.data = np.array([box_wire, box_pole])
        b.cls = np.array([0, 1])
        b.conf = np.array([0.9, 0.95])
        res.boxes = b
        
        res.names = {0: "power_line", 1: "pole"}
        return [res]


def patch_segment_image():
    from backend.utils.cv.types import SegmentationResult
    def mock_segment(img, model, orig_shape):
        if not hasattr(model, "mode"):
            return SegmentationResult(False, [], [], {}, [], [])
            
        if model.mode == "no_wires":
            return SegmentationResult(False, [], [], {}, [], [])
            
        res = model(img)[0]
        return SegmentationResult(
            mask_available=True,
            masks=[res.masks.data[0]],
            classes=[0, 1],
            names={0: "power_line", 1: "pole"},
            confidences=[0.9, 0.95],
            boxes=[[50, 100, 590, 310], [10, 400, 30, 500]]
        )
    from backend.utils import pipeline
    pipeline.segment_image = mock_segment

def main():
    conn = get_connection()
    run_migrations(conn)
    init_config_store(conn)
    
    os.makedirs("uploads", exist_ok=True)
    
    print("Generating batch of 5 dummy images...")
    # 1 blurry
    blurry = np.full((640, 640, 3), 128, dtype=np.uint8)
    # add slight gradient to prevent division by zero, but keep variance low
    cv2.blur(blurry, (50, 50))
    cv2.imwrite("uploads/blurry.jpg", blurry)
    
    # 1 textured no wires
    no_wires = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    cv2.imwrite("uploads/no_wires.jpg", no_wires)
    
    # 3 normal
    for i in range(3):
        normal = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        # make it not blurry
        normal[320:330, :] = 0 
        cv2.imwrite(f"uploads/normal_{i}.jpg", normal)
        
    patch_segment_image()
    
    # create batch
    cur = conn.execute("INSERT INTO batches (status) VALUES ('running')")
    batch_id = cur.lastrowid
    
    files = [
        ("blurry.jpg", "blurry"),
        ("no_wires.jpg", "no_wires"),
        ("normal_0.jpg", "normal"),
        ("normal_1.jpg", "normal"),
        ("normal_2.jpg", "normal")
    ]
    
    analysis_ids = []
    for f, mode in files:
        cur = conn.execute("INSERT INTO analyses (filename, status, batch_id) VALUES (?, 'queued', ?)", (f, batch_id))
        aid = cur.lastrowid
        analysis_ids.append((aid, f, mode))
        
    conn.commit()
    
    print("\nRunning Pipeline Orchestrator on batch...")
    
    for aid, f, mode in analysis_ids:
        model = FakeSegmenter(mode)
        # Note: the pipeline checks for blurry in QualityValidation step. We assume 'blurry.jpg' fails Laplacian check.
        execute_pipeline(aid, f"uploads/{f}", conn, model, batch_id)
        
    print("\n--- BATCH RESULTS ---")
    rows = conn.execute("SELECT id, filename, status FROM analyses WHERE batch_id = ?", (batch_id,)).fetchall()
    print(f"{'ID':<5} | {'Filename':<15} | {'Status'}")
    print("-" * 40)
    for r in rows:
        print(f"{r['id']:<5} | {r['filename']:<15} | {r['status']}")
        
    print("\n--- STATE TRANSITION LOG (Normal Image) ---")
    events = conn.execute("SELECT state, detail FROM pipeline_events WHERE analysis_id = ? ORDER BY id", (analysis_ids[2][0],)).fetchall()
    for e in events:
        print(f"[{e['state']}] {e['detail']}")
        
    print("\n--- STATE TRANSITION LOG (Rejected Image) ---")
    events = conn.execute("SELECT state, detail FROM pipeline_events WHERE analysis_id = ? ORDER BY id", (analysis_ids[0][0],)).fetchall()
    for e in events:
        print(f"[{e['state']}] {e['detail']}")
        
    print("\n--- STATE TRANSITION LOG (No Wires Image) ---")
    events = conn.execute("SELECT state, detail FROM pipeline_events WHERE analysis_id = ? ORDER BY id", (analysis_ids[1][0],)).fetchall()
    for e in events:
        print(f"[{e['state']}] {e['detail']}")
        
    conn.close()

if __name__ == "__main__":
    main()
