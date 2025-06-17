# src/job_manager.py
"""
Background job manager for running benchmarks that persist across page reloads.
Uses threading and SQLite for job tracking and status updates.
"""

import json
import sqlite3
import threading
import time
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

from .paths import RESULTS
from .pipeline import run_test
from .results_db import save_run

# Job status enum
class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Database setup
JOB_DB_PATH = RESULTS / "jobs.sqlite"
JOB_DB_PATH.parent.mkdir(exist_ok=True, parents=True)

# Global thread pool for background jobs
_job_threads: Dict[str, threading.Thread] = {}
_job_lock = threading.Lock()

@contextmanager
def _get_conn():
    """Get a database connection with proper error handling."""
    conn = sqlite3.connect(JOB_DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize the jobs database."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                run_name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                total_docs INTEGER NOT NULL,
                processed_docs INTEGER DEFAULT 0,
                current_doc TEXT,
                folders TEXT NOT NULL,
                models TEXT NOT NULL,
                iterations INTEGER NOT NULL,
                doc_counts TEXT NOT NULL,
                error TEXT,
                results TEXT,
                logs TEXT
            )
        """)
        
        # Create index for faster queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status 
            ON jobs(status, created_at DESC)
        """)
        conn.commit()

# Initialize on import
init_db()

def create_job(
    run_name: str,
    folders: List[str],
    models: List[str],
    iterations: int,
    doc_counts: Dict[str, int],
    total_docs: int
) -> str:
    """Create a new job and return its ID."""
    job_id = f"{run_name}_{int(time.time())}"
    
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO jobs (
                job_id, run_name, status, created_at, total_docs,
                folders, models, iterations, doc_counts, logs
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_id, run_name, JobStatus.PENDING.value, time.time(), total_docs,
            json.dumps(folders), json.dumps(models), iterations,
            json.dumps(doc_counts), json.dumps([])
        ))
        conn.commit()
    
    return job_id

def update_job_status(
    job_id: str,
    status: JobStatus,
    current_doc: Optional[str] = None,
    processed_docs: Optional[int] = None,
    error: Optional[str] = None,
    log_entry: Optional[str] = None
):
    """Update job status and optionally add a log entry."""
    with _get_conn() as conn:
        # Get current logs
        cursor = conn.execute("SELECT logs FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        if row:
            logs = json.loads(row["logs"] or "[]")
            if log_entry:
                logs.append({
                    "timestamp": time.time(),
                    "message": log_entry
                })
        
        # Build update query
        updates = ["status = ?"]
        params = [status.value]
        
        if status == JobStatus.RUNNING and "started_at" not in updates:
            updates.append("started_at = ?")
            params.append(time.time())
        
        if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            updates.append("completed_at = ?")
            params.append(time.time())
        
        if current_doc is not None:
            updates.append("current_doc = ?")
            params.append(current_doc)
        
        if processed_docs is not None:
            updates.append("processed_docs = ?")
            params.append(processed_docs)
        
        if error is not None:
            updates.append("error = ?")
            params.append(error)
        
        if log_entry:
            updates.append("logs = ?")
            params.append(json.dumps(logs))
        
        params.append(job_id)
        conn.execute(f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?", params)
        conn.commit()

def save_job_results(job_id: str, results: List[Dict]):
    """Save the results for a completed job."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE jobs SET results = ? WHERE job_id = ?",
            (json.dumps(results), job_id)
        )
        conn.commit()

def get_job(job_id: str) -> Optional[Dict]:
    """Get job details by ID."""
    with _get_conn() as conn:
        cursor = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
    return None

def get_active_jobs() -> List[Dict]:
    """Get all pending and running jobs."""
    with _get_conn() as conn:
        cursor = conn.execute("""
            SELECT * FROM jobs 
            WHERE status IN (?, ?)
            ORDER BY created_at DESC
        """, (JobStatus.PENDING.value, JobStatus.RUNNING.value))
        return [dict(row) for row in cursor.fetchall()]

def get_recent_jobs(limit: int = 20) -> List[Dict]:
    """Get recent jobs of all statuses."""
    with _get_conn() as conn:
        cursor = conn.execute("""
            SELECT * FROM jobs
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

def cancel_job(job_id: str) -> bool:
    """Cancel a pending or running job."""
    with _job_lock:
        # Check if thread exists and is alive
        if job_id in _job_threads:
            thread = _job_threads[job_id]
            if thread.is_alive():
                # We can't forcefully stop a thread, but we can mark it as cancelled
                # The job runner should check this status periodically
                update_job_status(job_id, JobStatus.CANCELLED, error="Cancelled by user")
                return True
    
    # If no active thread, just update status
    job = get_job(job_id)
    if job and job["status"] in (JobStatus.PENDING.value, JobStatus.RUNNING.value):
        update_job_status(job_id, JobStatus.CANCELLED, error="Cancelled by user")
        return True
    
    return False

def _should_cancel(job_id: str) -> bool:
    """Check if a job has been marked for cancellation."""
    job = get_job(job_id)
    return job and job["status"] == JobStatus.CANCELLED.value

def _job_logger(job_id: str, message: str):
    """Logger function that saves to job logs."""
    update_job_status(job_id, JobStatus.RUNNING, log_entry=message)

def _run_benchmark_job(
    job_id: str,
    run_name: str,
    docs: List[Path],
    models: List[str],
    iterations: int,
    folders: List[str],
    doc_counts: Dict[str, int]
):
    """Background worker function for running benchmarks."""
    try:
        # Update status to running
        update_job_status(job_id, JobStatus.RUNNING, log_entry=f"Starting benchmark: {run_name}")
        
        results = []
        
        for idx, doc_path in enumerate(docs, 1):
            # Check for cancellation
            if _should_cancel(job_id):
                update_job_status(job_id, JobStatus.CANCELLED, 
                                processed_docs=idx-1,
                                log_entry="Job cancelled by user")
                return
            
            # Update progress
            update_job_status(
                job_id, 
                JobStatus.RUNNING,
                current_doc=doc_path.name,
                processed_docs=idx-1,
                log_entry=f"Processing ({idx}/{len(docs)}): {doc_path.name}"
            )
            
            try:
                # Run test with custom logger
                res = run_test(
                    doc_path, 
                    models, 
                    lambda msg: _job_logger(job_id, msg),
                    iterations
                )
                
                if res.get("runs"):
                    results.append(res)
                    update_job_status(
                        job_id,
                        JobStatus.RUNNING,
                        processed_docs=idx,
                        log_entry=f"✅ Completed {doc_path.name} - {len(res['runs'])} drafts"
                    )
                else:
                    update_job_status(
                        job_id,
                        JobStatus.RUNNING,
                        processed_docs=idx,
                        log_entry=f"⚠️ Skipped {doc_path.name} (no paragraphs)"
                    )
                    
            except Exception as e:
                error_msg = f"Error processing {doc_path.name}: {str(e)}"
                update_job_status(
                    job_id,
                    JobStatus.RUNNING,
                    processed_docs=idx,
                    log_entry=f"❌ {error_msg}"
                )
                # Continue with next document instead of failing entire job
                continue
        
        # Save results
        if results:
            save_job_results(job_id, results)
            save_run(
                run_name,
                folders,
                models,
                {"docs": results, "iterations": iterations, "doc_counts": doc_counts}
            )
            update_job_status(
                job_id,
                JobStatus.COMPLETED,
                processed_docs=len(docs),
                log_entry=f"✅ Benchmark completed successfully - {len(results)} documents processed"
            )
        else:
            update_job_status(
                job_id,
                JobStatus.FAILED,
                error="No documents were successfully processed",
                log_entry="❌ Benchmark failed - no documents processed"
            )
            
    except Exception as e:
        error_msg = f"Job failed: {str(e)}\n{traceback.format_exc()}"
        update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_msg,
            log_entry=f"❌ Fatal error: {str(e)}"
        )
    finally:
        # Clean up thread reference
        with _job_lock:
            _job_threads.pop(job_id, None)

def start_benchmark_job(
    run_name: str,
    docs: List[Path],
    folders: List[str],
    models: List[str],
    iterations: int,
    doc_counts: Dict[str, int]
) -> str:
    """Start a benchmark job in the background and return the job ID."""
    # Create job record
    job_id = create_job(
        run_name=run_name,
        folders=folders,
        models=models,
        iterations=iterations,
        doc_counts=doc_counts,
        total_docs=len(docs)
    )
    
    # Start background thread
    thread = threading.Thread(
        target=_run_benchmark_job,
        args=(job_id, run_name, docs, models, iterations, folders, doc_counts),
        daemon=True,
        name=f"benchmark-{job_id}"
    )
    
    with _job_lock:
        _job_threads[job_id] = thread
        thread.start()
    
    return job_id

def cleanup_old_jobs(days: int = 7):
    """Clean up jobs older than specified days."""
    cutoff = time.time() - (days * 24 * 60 * 60)
    
    with _get_conn() as conn:
        conn.execute("""
            DELETE FROM jobs 
            WHERE completed_at < ? 
            AND status IN (?, ?, ?)
        """, (cutoff, JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value))
        conn.commit()