import logging
import os
import queue
import sqlite3
import threading

from backend import main_state
from backend.database import get_connection
from backend.utils.analysis_engine import process_analysis

logger = logging.getLogger(__name__)

# Global batch queue
batch_queue = queue.Queue()

def _batch_worker():
    while True:
        try:
            batch_id = batch_queue.get()
            if batch_id is None:
                # Sentinel value to shut down
                break
                
            try:
                conn = get_connection()
                conn.execute("PRAGMA journal_mode=WAL")
                
                # Mark batch running
                conn.execute("UPDATE batches SET status = 'running', started_at = datetime('now') WHERE id = ?", (batch_id,))
                conn.commit()
                
                items = conn.execute("SELECT id, filename FROM analyses WHERE batch_id = ? ORDER BY id", (batch_id,)).fetchall()
                
                for item in items:
                    # Check cancel at batch boundary
                    row = conn.execute("SELECT cancel_requested FROM batches WHERE id = ?", (batch_id,)).fetchone()
                    if row and row["cancel_requested"]:
                        # Mark remaining as interrupted
                        conn.execute("UPDATE analyses SET status = 'interrupted' WHERE batch_id = ? AND status = 'queued'", (batch_id,))
                        conn.commit()
                        break
                        
                    # Handle both cases where the file might be in uploads or just the filename
                    # Note: in a real implementation we would fetch original_path or similar
                    process_analysis(item["id"], os.path.join("uploads", item["filename"]), conn, main_state.get_model(), batch_id)

                # Check final status
                row = conn.execute("SELECT cancel_requested FROM batches WHERE id = ?", (batch_id,)).fetchone()
                final_status = "interrupted" if row and row["cancel_requested"] else "done"
                
                conn.execute("UPDATE batches SET status = ?, finished_at = datetime('now') WHERE id = ?", (final_status, batch_id))
                conn.commit()
                conn.close()
            except Exception as e:
                import traceback
                error_id = str(hash(e))[-6:]
                logger.error(f"Batch worker exception for batch {batch_id} (error_id: {error_id}): {e}\n{traceback.format_exc()}")
                try:
                    conn = get_connection()
                    conn.execute(
                        "UPDATE analyses SET status = 'error', summary = ? WHERE batch_id = ? AND status IN ('queued', 'processing')",
                        (f"Batch worker exception (error_id: {error_id})", batch_id)
                    )
                    conn.execute("UPDATE batches SET status = 'error', finished_at = datetime('now') WHERE id = ?", (batch_id,))
                    conn.commit()
                    conn.close()
                except sqlite3.Error as inner_e:
                    logger.error(f"Failed to mark batch {batch_id} as error: {inner_e}", exc_info=True)
            finally:
                batch_queue.task_done()
        except queue.Empty:
            pass
        except Exception as queue_err:
            logger.error(f"Batch worker critical error: {queue_err}", exc_info=True)

# Start worker thread
_worker_thread = None

def start_worker():
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_batch_worker, daemon=True)
        _worker_thread.start()

def stop_worker():
    global _worker_thread
    batch_queue.put(None)
    if _worker_thread and _worker_thread.is_alive():
        _worker_thread.join(timeout=5)

def enqueue_batch(batch_id: int):
    """Enqueues a batch for processing. To be called AFTER the DB commit."""
    try:
        batch_queue.put(batch_id)
    except Exception as e:
        import traceback
        error_id = str(hash(e))[-6:]
        logger.error(f"Failed to enqueue batch {batch_id} (error_id: {error_id}): {e}\n{traceback.format_exc()}")
        try:
            conn = get_connection()
            conn.execute("UPDATE analyses SET status = 'error', summary = ? WHERE batch_id = ?", (f"Enqueue failed: {e}", batch_id))
            conn.execute("UPDATE batches SET status = 'error' WHERE id = ?", (batch_id,))
            conn.commit()
            conn.close()
        except sqlite3.Error as inner_e:
            logger.error(f"Failed to mark enqueue failure in DB: {inner_e}", exc_info=True)
        raise RuntimeError(f"Could not enqueue batch: {e}")
