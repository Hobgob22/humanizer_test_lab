"""
Temporary database for storing intermediate results during benchmark runs.
This ensures data persistence even if the process fails partway through.

Schema
------
temp_runs (
    job_id       TEXT,
    run_name     TEXT,
    document     TEXT,
    doc_results  TEXT,          -- JSON blob of document results
    processed_at REAL,          -- unix timestamp when processed
    folders      TEXT,          -- comma-sep folder list
    models       TEXT,          -- comma-sep model list
    iterations   INTEGER,
    include_doc_mode INTEGER,
    use_gptzero  INTEGER,
    use_sapling  INTEGER,
    PRIMARY KEY (job_id, document)
)

temp_run_metadata (
    job_id       TEXT PRIMARY KEY,
    run_name     TEXT,
    folders      TEXT,
    models       TEXT,
    iterations   INTEGER,
    include_doc_mode INTEGER,
    use_gptzero  INTEGER,
    use_sapling  INTEGER,
    total_docs   INTEGER,
    processed_docs INTEGER DEFAULT 0,
    created_at   REAL,
    updated_at   REAL
)
"""

import json
import time
import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List, Optional, Any

from .paths import RESULTS

logger = logging.getLogger(__name__)

# Temporary database path
TEMP_DB_PATH = RESULTS / "temp_jobs_backup.sqlite"
TEMP_DB_PATH.parent.mkdir(exist_ok=True, parents=True)

@contextmanager
def _get_temp_conn():
    """Get a temporary database connection with proper error handling."""
    conn = sqlite3.connect(TEMP_DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_temp_db():
    """Initialize the temporary results database."""
    with _get_temp_conn() as conn:
        # Create temp_runs table for individual document results
        conn.execute("""
            CREATE TABLE IF NOT EXISTS temp_runs (
                job_id TEXT NOT NULL,
                run_name TEXT NOT NULL,
                document TEXT NOT NULL,
                doc_results TEXT NOT NULL,
                processed_at REAL NOT NULL,
                folders TEXT NOT NULL,
                models TEXT NOT NULL,
                iterations INTEGER NOT NULL,
                include_doc_mode INTEGER NOT NULL,
                use_gptzero INTEGER NOT NULL,
                use_sapling INTEGER NOT NULL,
                PRIMARY KEY (job_id, document)
            )
        """)
        
        # Create temp_run_metadata table for run-level metadata
        conn.execute("""
            CREATE TABLE IF NOT EXISTS temp_run_metadata (
                job_id TEXT PRIMARY KEY,
                run_name TEXT NOT NULL,
                folders TEXT NOT NULL,
                models TEXT NOT NULL,
                iterations INTEGER NOT NULL,
                include_doc_mode INTEGER NOT NULL,
                use_gptzero INTEGER NOT NULL,
                use_sapling INTEGER NOT NULL,
                total_docs INTEGER NOT NULL,
                processed_docs INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                continuous_blob TEXT
            )
        """)
        
        # Check if continuous_blob column exists, add if not
        cursor = conn.execute("PRAGMA table_info(temp_run_metadata)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'continuous_blob' not in columns:
            conn.execute("ALTER TABLE temp_run_metadata ADD COLUMN continuous_blob TEXT")
        
        # Create indices for faster queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_temp_runs_job_id 
            ON temp_runs(job_id, processed_at DESC)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_temp_runs_processed_at 
            ON temp_runs(processed_at DESC)
        """)
        
        conn.commit()

# Initialize on import
init_temp_db()

def create_temp_run(
    job_id: str,
    run_name: str,
    folders: List[str],
    models: List[str],
    iterations: int,
    total_docs: int,
    include_doc_mode: bool = True,
    use_gptzero: bool = True,
    use_sapling: bool = True
):
    """Create a new temporary run record."""
    try:
        with _get_temp_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO temp_run_metadata (
                    job_id, run_name, folders, models, iterations,
                    include_doc_mode, use_gptzero, use_sapling,
                    total_docs, processed_docs, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """, (
                job_id, run_name, 
                ",".join(folders), ",".join(models), iterations,
                int(include_doc_mode), int(use_gptzero), int(use_sapling),
                total_docs, time.time(), time.time()
            ))
            conn.commit()
            
        logger.info(f"✓ Created temporary run record for job '{job_id}' with {total_docs} documents")
        
    except Exception as e:
        logger.error(f"❌ Failed to create temporary run record for job '{job_id}': {e}")
        raise

def save_temp_document_result(
    job_id: str,
    run_name: str,
    folders: List[str],
    models: List[str],
    iterations: int,
    document_result: Dict[str, Any],
    include_doc_mode: bool = True,
    use_gptzero: bool = True,
    use_sapling: bool = True
):
    """Save a single document's results and update the continuous run blob."""
    try:
        with _get_temp_conn() as conn:
            # Save the individual document result
            conn.execute("""
                INSERT OR REPLACE INTO temp_runs (
                    job_id, run_name, document, doc_results, processed_at,
                    folders, models, iterations, include_doc_mode, use_gptzero, use_sapling
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id, run_name, document_result["document"],
                json.dumps(document_result, ensure_ascii=False), time.time(),
                ",".join(folders), ",".join(models), iterations,
                int(include_doc_mode), int(use_gptzero), int(use_sapling)
            ))
            
            # Get all document results for this job to build the continuous blob
            cursor = conn.execute("""
                SELECT doc_results FROM temp_runs 
                WHERE job_id = ? 
                ORDER BY processed_at ASC
            """, (job_id,))
            
            all_docs = []
            for row in cursor.fetchall():
                try:
                    doc_data = json.loads(row["doc_results"])
                    all_docs.append(doc_data)
                except json.JSONDecodeError:
                    continue
            
            # Calculate doc_counts like the main system
            doc_counts = {}
            for doc in all_docs:
                folder = doc.get("folder", "unknown")
                doc_counts[folder] = doc_counts.get(folder, 0) + 1
            
            # Create the exact same structure as runs.sqlite
            continuous_run_blob = {
                "docs": all_docs,
                "iterations": iterations,
                "doc_counts": doc_counts,
                "include_doc_mode": include_doc_mode,
                "use_gptzero": use_gptzero,
                "use_sapling": use_sapling
            }
            
            # Update metadata with continuous blob
            conn.execute("""
                UPDATE temp_run_metadata 
                SET processed_docs = ?,
                    updated_at = ?,
                    continuous_blob = ?
                WHERE job_id = ?
            """, (len(all_docs), time.time(), json.dumps(continuous_run_blob, ensure_ascii=False), job_id))
            
            conn.commit()
            
        logger.info(f"✓ Saved temporary result for document '{document_result['document']}' in job '{job_id}' (total: {len(all_docs)} docs)")
        
    except Exception as e:
        logger.error(f"❌ Failed to save temporary result for document '{document_result.get('document', 'unknown')}' in job '{job_id}': {e}")
        raise

def get_temp_run_progress(job_id: str) -> Optional[Dict[str, Any]]:
    """Get progress information for a temporary run."""
    try:
        with _get_temp_conn() as conn:
            # Get metadata
            cursor = conn.execute("""
                SELECT * FROM temp_run_metadata WHERE job_id = ?
            """, (job_id,))
            metadata = cursor.fetchone()
            
            if not metadata:
                return None
            
            # Get processed documents count
            cursor = conn.execute("""
                SELECT COUNT(*) as count FROM temp_runs WHERE job_id = ?
            """, (job_id,))
            actual_processed = cursor.fetchone()["count"]
            
            return {
                "job_id": job_id,
                "run_name": metadata["run_name"],
                "folders": metadata["folders"].split(",") if metadata["folders"] else [],
                "models": metadata["models"].split(",") if metadata["models"] else [],
                "iterations": metadata["iterations"],
                "include_doc_mode": bool(metadata["include_doc_mode"]),
                "use_gptzero": bool(metadata["use_gptzero"]),
                "use_sapling": bool(metadata["use_sapling"]),
                "total_docs": metadata["total_docs"],
                "processed_docs": actual_processed,  # Use actual count from temp_runs
                "progress_percent": (actual_processed / metadata["total_docs"]) * 100 if metadata["total_docs"] > 0 else 0,
                "created_at": metadata["created_at"],
                "updated_at": metadata["updated_at"]
            }
            
    except Exception as e:
        logger.error(f"❌ Failed to get temporary run progress for job '{job_id}': {e}")
        return None

def get_temp_run_results(job_id: str) -> Optional[Dict[str, Any]]:
    """Get all results for a temporary run in the exact same format as runs.sqlite."""
    try:
        with _get_temp_conn() as conn:
            # Get metadata including continuous blob
            cursor = conn.execute("""
                SELECT * FROM temp_run_metadata WHERE job_id = ?
            """, (job_id,))
            metadata = cursor.fetchone()
            
            if not metadata:
                return None
            
            # If we have a continuous blob, use it (exact same format as runs.sqlite)
            if metadata["continuous_blob"]:
                try:
                    continuous_data = json.loads(metadata["continuous_blob"])
                    return continuous_data
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Failed to parse continuous blob for job '{job_id}', falling back to individual docs")
            
            # Fallback: build from individual document results
            cursor = conn.execute("""
                SELECT document, doc_results, processed_at 
                FROM temp_runs 
                WHERE job_id = ? 
                ORDER BY processed_at ASC
            """, (job_id,))
            
            docs = []
            doc_counts = {}
            for row in cursor.fetchall():
                try:
                    doc_result = json.loads(row["doc_results"])
                    docs.append(doc_result)
                    
                    # Count by folder
                    folder = doc_result.get("folder", "unknown")
                    doc_counts[folder] = doc_counts.get(folder, 0) + 1
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ Failed to parse document result for '{row['document']}': {e}")
                    continue
            
            return {
                "docs": docs,
                "iterations": metadata["iterations"],
                "doc_counts": doc_counts,
                "include_doc_mode": bool(metadata["include_doc_mode"]),
                "use_gptzero": bool(metadata["use_gptzero"]),
                "use_sapling": bool(metadata["use_sapling"])
            }
            
    except Exception as e:
        logger.error(f"❌ Failed to get temporary run results for job '{job_id}': {e}")
        return None

def list_temp_runs() -> List[Dict[str, Any]]:
    """List all temporary runs with their progress."""
    try:
        with _get_temp_conn() as conn:
            cursor = conn.execute("""
                SELECT job_id FROM temp_run_metadata 
                ORDER BY created_at DESC
            """)
            
            temp_runs = []
            for row in cursor.fetchall():
                progress = get_temp_run_progress(row["job_id"])
                if progress:
                    temp_runs.append(progress)
            
            return temp_runs
            
    except Exception as e:
        logger.error(f"❌ Failed to list temporary runs: {e}")
        return []

def recover_from_temp_run(job_id: str) -> Optional[str]:
    """
    Recover a run from temporary database and save it to the main results database.
    Returns the run name if successful, None otherwise.
    """
    try:
        from .results_db import save_run  # Import here to avoid circular imports
        
        # Get metadata for run name, folders, models
        progress = get_temp_run_progress(job_id)
        if not progress:
            logger.error(f"❌ No temporary metadata found for job '{job_id}'")
            return None
        
        # Get the continuous run data (exact same format as runs.sqlite)
        temp_results = get_temp_run_results(job_id)
        if not temp_results:
            logger.error(f"❌ No temporary results found for job '{job_id}'")
            return None
        
        # Save to main database (temp_results already has the exact structure expected)
        save_run(
            progress["run_name"],
            progress["folders"],
            progress["models"],
            temp_results  # Already in the correct format
        )
        
        logger.info(f"✓ Successfully recovered run '{progress['run_name']}' from temporary database")
        return progress["run_name"]
        
    except Exception as e:
        logger.error(f"❌ Failed to recover run from temporary database for job '{job_id}': {e}")
        return None

def cleanup_temp_run(job_id: str):
    """Clean up temporary data for a completed job."""
    try:
        with _get_temp_conn() as conn:
            conn.execute("DELETE FROM temp_runs WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM temp_run_metadata WHERE job_id = ?", (job_id,))
            conn.commit()
            
        logger.info(f"✓ Cleaned up temporary data for job '{job_id}'")
        
    except Exception as e:
        logger.error(f"❌ Failed to cleanup temporary data for job '{job_id}': {e}")

def cleanup_old_temp_runs(days: int = 7):
    """Clean up temporary runs older than specified days."""
    try:
        cutoff = time.time() - (days * 24 * 60 * 60)
        
        with _get_temp_conn() as conn:
            # Get job_ids to clean up
            cursor = conn.execute("""
                SELECT job_id FROM temp_run_metadata 
                WHERE created_at < ?
            """, (cutoff,))
            
            old_job_ids = [row["job_id"] for row in cursor.fetchall()]
            
            # Clean up old data
            conn.execute("DELETE FROM temp_runs WHERE job_id IN (SELECT job_id FROM temp_run_metadata WHERE created_at < ?)", (cutoff,))
            conn.execute("DELETE FROM temp_run_metadata WHERE created_at < ?", (cutoff,))
            conn.commit()
            
        if old_job_ids:
            logger.info(f"✓ Cleaned up {len(old_job_ids)} old temporary runs: {', '.join(old_job_ids)}")
        
    except Exception as e:
        logger.error(f"❌ Failed to cleanup old temporary runs: {e}") 