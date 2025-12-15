# src/job_manager.py
"""
Background job manager for running benchmarks that persist across page reloads.
Uses threading and SQLite for job tracking and status updates.
"""

import json
import logging
import sqlite3
import threading
import time
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    wait,
    FIRST_COMPLETED,
)

from .paths import RESULTS
from .config import MAX_PARALLEL_DOCS, LOG_HISTORY_LIMIT
from .pipeline import run_test
from .results_db import save_run
from .temp_results_db import (
    create_temp_run, save_temp_document_result, cleanup_temp_run,
    get_temp_run_progress, get_temp_run_results, recover_from_temp_run,
    list_temp_runs
)

logger = logging.getLogger(__name__)

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
_db_write_lock = threading.Lock()  # Serialize database writes to prevent lock contention

@contextmanager
def _get_conn():
    """Get a database connection with proper error handling and retry logic."""
    max_retries = 5
    retry_delay = 0.1  # Start with 100ms

    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(JOB_DB_PATH, timeout=60, check_same_thread=False)
            conn.row_factory = sqlite3.Row

            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            # Set busy timeout to 60 seconds
            conn.execute("PRAGMA busy_timeout=60000")

            try:
                yield conn
                conn.commit()
                break
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    conn.close()
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                raise
            finally:
                conn.close()
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
                continue
            raise

def init_db():
    """Initialize the jobs database."""
    print(f"[JOB_MANAGER] init_db() called, JOB_DB_PATH: {JOB_DB_PATH}", flush=True)
    with _get_conn() as conn:
        # Check if table exists
        table_exists = any(row[0] == 'jobs' for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
        
        if not table_exists:
            print("[JOB_MANAGER] Table 'jobs' does not exist, creating...", flush=True)
            # Create new table with all columns
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
                    active_docs TEXT DEFAULT '[]',
                    folders TEXT NOT NULL,
                    models TEXT NOT NULL,
                    iterations INTEGER NOT NULL,
                    doc_counts TEXT NOT NULL,
                    include_doc_mode INTEGER DEFAULT 1,
                    use_gptzero INTEGER DEFAULT 1,
                    use_sapling INTEGER DEFAULT 1,
                    user_style_profile TEXT,
                    user_style_profile_mode TEXT,
                    use_style_adherence INTEGER DEFAULT 0,
                    user_style_models TEXT DEFAULT '[]',
                    error TEXT,
                    results TEXT,
                    logs TEXT
                )
            """)
            print("[JOB_MANAGER] [OK] Table 'jobs' created successfully", flush=True)
        else:
            print("[JOB_MANAGER] Table 'jobs' exists, checking for missing columns...", flush=True)
            # Check existing columns and add missing ones
            cursor = conn.execute("PRAGMA table_info(jobs)")
            existing_columns = [row[1] for row in cursor.fetchall()]
            print(f"[JOB_MANAGER] Existing columns: {existing_columns}", flush=True)
            
            # Add missing columns to existing table
            if 'include_doc_mode' not in existing_columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN include_doc_mode INTEGER DEFAULT 1")
                print("[JOB_MANAGER] Added column: include_doc_mode", flush=True)
            
            if 'use_gptzero' not in existing_columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN use_gptzero INTEGER DEFAULT 1")
                print("[JOB_MANAGER] Added column: use_gptzero", flush=True)
            
            if 'use_sapling' not in existing_columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN use_sapling INTEGER DEFAULT 1")
                print("[JOB_MANAGER] Added column: use_sapling", flush=True)
            
            if 'user_style_profile' not in existing_columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN user_style_profile TEXT")
                print("[JOB_MANAGER] Added column: user_style_profile", flush=True)
            
            if 'user_style_profile_mode' not in existing_columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN user_style_profile_mode TEXT")
                print("[JOB_MANAGER] Added column: user_style_profile_mode", flush=True)
            
            if 'use_style_adherence' not in existing_columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN use_style_adherence INTEGER DEFAULT 0")
                print("[JOB_MANAGER] Added column: use_style_adherence", flush=True)
            
            if 'user_style_models' not in existing_columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN user_style_models TEXT DEFAULT '[]'")
                print("[JOB_MANAGER] Added column: user_style_models", flush=True)
            
            # Add active_docs column if it doesn't exist
            if 'active_docs' not in existing_columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN active_docs TEXT DEFAULT '[]'")
                print("[JOB_MANAGER] Added column: active_docs", flush=True)
        
        # Create separate logs table for O(1) insertions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS job_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
            )
        """)
        print("[JOB_MANAGER] [OK] Table 'job_logs' created/verified", flush=True)

        # Create indexes for faster queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status
            ON jobs(status, created_at DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status_created
            ON jobs(status, created_at DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_created
            ON jobs(created_at DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_job_logs_job_ts
            ON job_logs(job_id, timestamp DESC)
        """)
        conn.commit()
        print("[JOB_MANAGER] [OK] init_db() completed successfully", flush=True)

# Initialize on import
init_db()

def create_job(
    run_name: str,
    folders: List[str],
    models: List[str],
    iterations: int,
    doc_counts: Dict[str, int],
    total_docs: int,
    include_doc_mode: bool = True,
    use_gptzero: bool = True,
    use_sapling: bool = True,
    user_style_profile: Optional[str] = None,
    user_style_profile_mode: Optional[str] = None,
    use_style_adherence: bool = False,
    user_style_models: Optional[List[str]] = None
) -> str:
    """Create a new job and return its ID."""
    import random
    
    # Generate unique job_id with timestamp and random component
    # Use microseconds for better uniqueness
    timestamp = time.time()
    random_suffix = random.randint(1000, 9999)
    job_id = f"{run_name}_{int(timestamp * 1000)}_{random_suffix}"
    
    # Retry if job_id already exists (unlikely but possible)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with _get_conn() as conn:
                # Check if job_id already exists
                existing = conn.execute(
                    "SELECT job_id FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                
                if existing:
                    # Regenerate with new random suffix
                    random_suffix = random.randint(1000, 9999)
                    job_id = f"{run_name}_{int(timestamp * 1000)}_{random_suffix}"
                    continue
                
                conn.execute("""
                    INSERT INTO jobs (
                        job_id, run_name, status, created_at, total_docs,
                        folders, models, iterations, doc_counts, include_doc_mode, use_gptzero, use_sapling,
                        user_style_profile, user_style_profile_mode, use_style_adherence, user_style_models
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job_id, run_name, JobStatus.PENDING.value, timestamp, total_docs,
                    json.dumps(folders), json.dumps(models), iterations,
                    json.dumps(doc_counts), int(include_doc_mode), int(use_gptzero), int(use_sapling),
                    user_style_profile, user_style_profile_mode, int(use_style_adherence),
                    json.dumps(user_style_models or [])
                ))
                conn.commit()
            
            return job_id
        except sqlite3.IntegrityError:
            # UNIQUE constraint violation - regenerate and retry
            if attempt < max_retries - 1:
                random_suffix = random.randint(1000, 9999)
                job_id = f"{run_name}_{int(timestamp * 1000)}_{random_suffix}"
                continue
            else:
                raise Exception(f"Failed to create unique job_id after {max_retries} attempts")
    
    return job_id

def update_job_status(
    job_id: str,
    status: JobStatus,
    current_doc: Optional[str] = None,
    current_stage: Optional[str] = None,
    processed_docs: Optional[int] = None,
    error: Optional[str] = None,
    log_entry: Optional[str] = None,
    add_active_doc: Optional[str] = None,
    remove_active_doc: Optional[str] = None
):
    """Update job status and optionally add a log entry to separate logs table (O(1) operation)."""
    # Use write lock to serialize database writes and prevent lock contention
    with _db_write_lock:
        with _get_conn() as conn:
            # Add log entry to separate table (O(1) operation - no read/parse/write overhead)
            if log_entry:
                conn.execute("""
                    INSERT INTO job_logs (job_id, timestamp, message)
                    VALUES (?, ?, ?)
                """, (job_id, time.time(), log_entry))

            # Build update query for job status
            updates = ["status = ?"]
            params = [status.value]

            if status == JobStatus.RUNNING and "started_at" not in updates:
                updates.append("started_at = ?")
                params.append(time.time())

            if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                updates.append("completed_at = ?")
                params.append(time.time())

            if current_doc is not None:
                # Combine document and stage info for current_doc field
                if current_stage:
                    combined_status = f"{current_doc} | {current_stage}"
                else:
                    combined_status = current_doc
                updates.append("current_doc = ?")
                params.append(combined_status)

            # Handle active documents list
            if add_active_doc is not None or remove_active_doc is not None:
                # Get current active docs
                cursor = conn.execute("SELECT active_docs FROM jobs WHERE job_id = ?", (job_id,))
                row = cursor.fetchone()
                if row:
                    try:
                        active_docs = json.loads(row["active_docs"] or "[]")
                    except json.JSONDecodeError:
                        active_docs = []

                    logger.info(f"📋 Current active docs before update: {active_docs}")

                    # Add document to active list
                    if add_active_doc:
                        doc_stage = f"{add_active_doc} | {current_stage}" if current_stage else add_active_doc
                        # Remove any existing entry for this document first
                        active_docs = [doc for doc in active_docs if not doc.startswith(f"{add_active_doc} |")]
                        active_docs.append(doc_stage)
                        logger.info(f"➕ Added '{doc_stage}' to active docs")

                    # Remove document from active list
                    if remove_active_doc:
                        original_count = len(active_docs)
                        active_docs = [doc for doc in active_docs if not doc.startswith(f"{remove_active_doc} |")]
                        logger.info(f"➖ Removed docs starting with '{remove_active_doc}' (removed {original_count - len(active_docs)} entries)")

                    logger.info(f"📋 New active docs after update: {active_docs}")
                    updates.append("active_docs = ?")
                    params.append(json.dumps(active_docs))

            if processed_docs is not None:
                updates.append("processed_docs = ?")
                params.append(processed_docs)

            if error is not None:
                updates.append("error = ?")
                params.append(error)

            params.append(job_id)
            conn.execute(f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?", params)
            # commit() is now handled by _get_conn() context manager

def get_job_logs(job_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get the most recent logs for a job from the separate logs table.
    Fast O(log n) query with index.

    Args:
        job_id: Job ID
        limit: Maximum number of logs to return (default 50)

    Returns:
        List of log entries with timestamp and message
    """
    with _get_conn() as conn:
        cursor = conn.execute("""
            SELECT timestamp, message
            FROM job_logs
            WHERE job_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (job_id, limit))

        logs = [{"timestamp": row[0], "message": row[1]} for row in cursor.fetchall()]
        # Return in chronological order (oldest first)
        return list(reversed(logs))

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
            job_dict = dict(row)
            # Convert include_doc_mode from int to bool
            if 'include_doc_mode' in job_dict:
                job_dict['include_doc_mode'] = bool(job_dict['include_doc_mode'])
            return job_dict
    return None

def get_active_jobs() -> List[Dict]:
    """Get all pending and running jobs."""
    with _get_conn() as conn:
        cursor = conn.execute("""
            SELECT * FROM jobs 
            WHERE status IN (?, ?)
            ORDER BY created_at DESC
        """, (JobStatus.PENDING.value, JobStatus.RUNNING.value))
        jobs = []
        for row in cursor.fetchall():
            job_dict = dict(row)
            # Convert include_doc_mode from int to bool
            if 'include_doc_mode' in job_dict:
                job_dict['include_doc_mode'] = bool(job_dict['include_doc_mode'])
            jobs.append(job_dict)
        return jobs

def get_recent_jobs(limit: int = 20) -> List[Dict]:
    """Get recent jobs of all statuses."""
    with _get_conn() as conn:
        cursor = conn.execute("""
            SELECT * FROM jobs
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        jobs = []
        for row in cursor.fetchall():
            job_dict = dict(row)
            # Convert include_doc_mode from int to bool
            if 'include_doc_mode' in job_dict:
                job_dict['include_doc_mode'] = bool(job_dict['include_doc_mode'])
            jobs.append(job_dict)
        return jobs

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

def _job_logger(job_id: str, message: str, current_doc: str = None, current_stage: str = None):
    """Logger function that saves to job logs and optionally updates current processing status."""
    # Determine if we should add or remove from active docs based on stage
    add_active = None
    remove_active = None
    
    if current_doc and current_stage:
        # Match the ACTUAL pipeline phase messages
        if current_stage in ["Phase 1: Generation", "Phase 2: Detector scoring", "Phase 3: Gemini quality evaluation", "Phase 4: Assembly"]:
            add_active = current_doc
            logger.info(f"🔄 Adding document '{current_doc}' to active list for stage '{current_stage}'")
        elif current_stage in ["Completed", "Failed", "Skipped", "Error"]:
            remove_active = current_doc
            logger.info(f"✅ Removing document '{current_doc}' from active list (stage: '{current_stage}')")
    
    update_job_status(
        job_id, 
        JobStatus.RUNNING, 
        current_doc=current_doc,
        current_stage=current_stage,
        log_entry=message,
        add_active_doc=add_active,
        remove_active_doc=remove_active
    )

def _run_benchmark_job(
    job_id: str,
    run_name: str,
    docs: List[Path],
    models: List[str],
    iterations: int,
    folders: List[str],
    doc_counts: Dict[str, int],
    include_doc_mode: bool = True,
    use_gptzero: bool = True,
    use_sapling: bool = True,
    user_style_profile: Optional[str] = None,
    user_style_profile_mode: Optional[str] = None,
    use_style_adherence: bool = False,
    user_style_models: Optional[List[str]] = None
):
    """Background worker function for running benchmarks."""
    print(f"[BACKGROUND] [START] Starting background benchmark job: {job_id}", flush=True)
    print(f"[BACKGROUND] Run name: {run_name}", flush=True)
    print(f"[BACKGROUND] Docs: {len(docs)}, Models: {models}, Iterations: {iterations}", flush=True)
    
    try:
        # Update status to running
        doc_mode_str = "doc + para modes" if include_doc_mode else "para mode only"
        print(f"[BACKGROUND] Modes: {doc_mode_str}", flush=True)
        update_job_status(job_id, JobStatus.RUNNING, log_entry=f"Starting benchmark: {run_name} ({doc_mode_str})")
        print(f"[BACKGROUND] [OK] Updated job {job_id} status to RUNNING", flush=True)
        
        # Create temporary run record for data persistence
        try:
            create_temp_run(
                job_id=job_id,
                run_name=run_name,
                folders=folders,
                models=models,
                iterations=iterations,
                total_docs=len(docs),
                include_doc_mode=include_doc_mode,
                use_gptzero=use_gptzero,
                use_sapling=use_sapling
            )
            update_job_status(job_id, JobStatus.RUNNING, log_entry="✓ Created temporary database for data persistence")
        except Exception as temp_db_error:
            # Don't fail the job if temp DB creation fails, just log it
            update_job_status(job_id, JobStatus.RUNNING, log_entry=f"⚠️ Failed to create temporary database: {temp_db_error}")
        
        results = []
        processed_counter = 0

        def _process_single(doc_path: Path):
            """Wrapper so we can push work into the pool."""
            try:
                # Update status to show which document is starting
                _job_logger(job_id, f"▶️ Starting {doc_path.name}", 
                           current_doc=doc_path.name, current_stage="Starting")

                # Create a stage-tracking logger for this document
                def stage_logger(message: str, stage: str = None):
                    # Parse stage from message if not explicitly provided
                    if not stage:
                        if "Phase 1: Generation" in message or "humanization" in message.lower():
                            stage = "Stage 1: Humanization"
                        elif "Phase 2: Detector scoring" in message or "detector" in message.lower():
                            stage = "Stage 2: Detector Scoring"
                        elif "Phase 3: Gemini quality" in message or "quality" in message.lower():
                            stage = "Stage 3: Quality Evaluation"
                        elif "Phase 4: Assembly" in message or "assembly" in message.lower():
                            stage = "Stage 4: Assembly"
                        else:
                            stage = "Processing"
                    
                    _job_logger(job_id, message, current_doc=doc_path.name, current_stage=stage)

                res = run_test(
                    doc_path,
                    models,
                    stage_logger,  # Use our enhanced stage logger
                    iterations,
                    include_doc_mode=include_doc_mode,  # Pass the humanization mode parameter
                    use_gptzero=use_gptzero,           # Pass detector selection
                    use_sapling=use_sapling,           # Pass detector selection
                    user_style_profile=user_style_profile,
                    user_style_profile_mode=user_style_profile_mode,
                    use_style_adherence=use_style_adherence,
                    user_style_models=user_style_models or []
                )
                return doc_path, res, None
            except Exception as exc:
                # Log error with document context
                _job_logger(job_id, f"❌ Error in {doc_path.name}: {str(exc)}", 
                           current_doc=doc_path.name, current_stage="Error")
                return doc_path, None, str(exc)

        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_DOCS,
                                thread_name_prefix="doc") as pool:
            fut2doc = {pool.submit(_process_single, p): p for p in docs}

            for fut in as_completed(fut2doc):
                # Early-exit on cancellation
                if _should_cancel(job_id):
                    pool.shutdown(wait=False, cancel_futures=True)
                    update_job_status(job_id, JobStatus.CANCELLED,
                                      processed_docs=processed_counter,
                                      log_entry="Job cancelled by user")
                    return

                processed_counter += 1
                doc_path, res, err = fut.result()

                if err:
                    update_job_status(
                        job_id, JobStatus.RUNNING,
                        current_doc=doc_path.name,
                        current_stage="Failed",
                        processed_docs=processed_counter,
                        log_entry=f"❌ Error {doc_path.name}: {err}"
                    )
                    continue

                if res.get("runs"):
                    results.append(res)
                    
                    # Save to temporary database immediately after processing each document
                    try:
                        save_temp_document_result(
                            job_id=job_id,
                            run_name=run_name,
                            folders=folders,
                            models=models,
                            iterations=iterations,
                            document_result=res,
                            include_doc_mode=include_doc_mode,
                            use_gptzero=use_gptzero,
                            use_sapling=use_sapling
                        )
                        update_job_status(
                            job_id, JobStatus.RUNNING,
                            current_doc=doc_path.name,
                            current_stage="Completed",
                            processed_docs=processed_counter,
                            log_entry=f"✅ Completed {doc_path.name} – {len(res['runs'])} drafts (saved to temp DB)"
                        )
                    except Exception as temp_save_error:
                        # Don't fail the job if temp save fails, just log it
                        update_job_status(
                            job_id, JobStatus.RUNNING,
                            current_doc=doc_path.name,
                            current_stage="Completed (DB Error)",
                            processed_docs=processed_counter,
                            log_entry=f"✅ Completed {doc_path.name} – {len(res['runs'])} drafts (⚠️ temp save failed: {temp_save_error})"
                        )
                else:
                    update_job_status(
                        job_id, JobStatus.RUNNING,
                        current_doc=doc_path.name,
                        current_stage="Skipped",
                        processed_docs=processed_counter,
                        log_entry=f"⚠️ Skipped {doc_path.name} (no paragraphs)"
                    )
        
        # Save results
        if results:
            save_job_results(job_id, results)
            
            # Try to save to persistent database, but don't fail the job if it fails
            try:
                save_run(
                    run_name,
                    folders,
                    models,
                    {
                        "docs": results, 
                        "iterations": iterations, 
                        "doc_counts": doc_counts,
                        "include_doc_mode": include_doc_mode,
                        "use_gptzero": use_gptzero,
                        "use_sapling": use_sapling
                    }
                )
                update_job_status(
                    job_id,
                    JobStatus.COMPLETED,
                    processed_docs=len(docs),
                    log_entry=f"✅ Benchmark completed successfully - {len(results)} documents processed"
                )
                
                # Clean up temporary database data after successful completion
                try:
                    cleanup_temp_run(job_id)
                except Exception as cleanup_error:
                    logging.warning(f"Failed to cleanup temporary data for job '{job_id}': {cleanup_error}")
            except Exception as db_error:
                # Log the database error but don't fail the job
                logging.error(f"Failed to save run to persistent database: {db_error}")
                update_job_status(
                    job_id,
                    JobStatus.COMPLETED,
                    processed_docs=len(docs),
                    log_entry=f"✅ Benchmark completed successfully - {len(results)} documents processed (Warning: Could not save to persistent database)"
                )
                
                # Clean up temporary database data after successful completion
                try:
                    cleanup_temp_run(job_id)
                except Exception as cleanup_error:
                    logging.warning(f"Failed to cleanup temporary data for job '{job_id}': {cleanup_error}")
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
    doc_counts: Dict[str, int],
    include_doc_mode: bool = True,
    use_gptzero: bool = True,
    use_sapling: bool = True,
    user_style_profile: Optional[str] = None,
    user_style_profile_mode: Optional[str] = None,
    use_style_adherence: bool = False,
    user_style_models: Optional[List[str]] = None
) -> str:
    """Start a benchmark job in the background and return the job ID."""
    print(f"[JOB_MANAGER] [CREATE] Creating job record for: {run_name}", flush=True)
    
    # Create job record
    job_id = create_job(
        run_name=run_name,
        folders=folders,
        models=models,
        iterations=iterations,
        doc_counts=doc_counts,
        total_docs=len(docs),
        include_doc_mode=include_doc_mode,
        use_gptzero=use_gptzero,
        use_sapling=use_sapling,
        user_style_profile=user_style_profile,
        user_style_profile_mode=user_style_profile_mode,
        use_style_adherence=use_style_adherence,
        user_style_models=user_style_models
    )
    
    print(f"[JOB_MANAGER] [OK] Created job record: {job_id}", flush=True)
    print(f"[JOB_MANAGER] [THREAD] Starting background thread...", flush=True)
    
    # Start background thread
    thread = threading.Thread(
        target=_run_benchmark_job,
        args=(job_id, run_name, docs, models, iterations, folders, doc_counts, include_doc_mode, use_gptzero, use_sapling, user_style_profile, user_style_profile_mode, use_style_adherence, user_style_models),
        daemon=True,
        name=f"benchmark-{job_id}"
    )
    
    with _job_lock:
        _job_threads[job_id] = thread
        print(f"[JOB_MANAGER] [OK] Registered thread for job {job_id}", flush=True)
        thread.start()
        print(f"[JOB_MANAGER] [OK] Thread started successfully for job {job_id}", flush=True)
    
    print(f"[JOB_MANAGER] [SUCCESS] Job {job_id} fully initialized and running", flush=True)
    
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
    
    # Also clean up old temporary database entries
    try:
        from .temp_results_db import cleanup_old_temp_runs
        cleanup_old_temp_runs(days)
    except Exception as e:
        logging.warning(f"Failed to cleanup old temporary runs: {e}")

# ═══════════════ RECOVERY FUNCTIONS ════════════════════════════════

def get_recoverable_jobs() -> List[Dict[str, Any]]:
    """Get jobs that can be recovered from temporary database."""
    try:
        temp_runs = list_temp_runs()
        recoverable = []
        
        for temp_run in temp_runs:
            # Check if this job still exists in the main jobs table
            job = get_job(temp_run["job_id"])
            if job and job["status"] in (JobStatus.FAILED.value, JobStatus.CANCELLED.value):
                # Job failed but has temporary data
                temp_run["job_status"] = job["status"]
                temp_run["can_recover"] = True
                recoverable.append(temp_run)
            elif not job:
                # Job doesn't exist in main table but has temp data
                temp_run["job_status"] = "unknown"
                temp_run["can_recover"] = True
                recoverable.append(temp_run)
        
        return recoverable
        
    except Exception as e:
        logging.error(f"Failed to get recoverable jobs: {e}")
        return []

def recover_job_from_temp(job_id: str) -> bool:
    """
    Recover a failed job from temporary database.
    Returns True if successful, False otherwise.
    """
    try:
        run_name = recover_from_temp_run(job_id)
        if run_name:
            # Update job status to completed if it exists
            job = get_job(job_id)
            if job:
                update_job_status(
                    job_id,
                    JobStatus.COMPLETED,
                    log_entry=f"[OK] Successfully recovered from temporary database: {run_name}"
                )
            
            # Clean up temporary data after successful recovery
            cleanup_temp_run(job_id)
            return True
        
        return False
        
    except Exception as e:
        logging.error(f"Failed to recover job '{job_id}' from temporary database: {e}")
        return False