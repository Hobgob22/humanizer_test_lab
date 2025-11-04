"""
Tiny SQLite DB for autonomous run storage.

Schema
------
runs (
    name         TEXT PRIMARY KEY,
    ts           REAL,          -- unix timestamp
    folders      TEXT,          -- comma-sep folder list
    models       TEXT,          -- comma-sep model list (display names)
    json_blob    TEXT           -- full result JSON (compressed later?)
)
"""

import json, time, os
from pathlib import Path
from contextlib import contextmanager
import logging
from .paths import RESULTS

# Try to import Turso client, fall back to sqlite3 for local development
try:
    import libsql_experimental as libsql
    TURSO_AVAILABLE = True
except ImportError:
    TURSO_AVAILABLE = False

import sqlite3

logger = logging.getLogger(__name__)

DB_PATH = RESULTS / "runs.sqlite"
DB_PATH.parent.mkdir(exist_ok=True, parents=True)

# Create persistent backup directory for disaster recovery
BACKUP_DIR = RESULTS / "pending"
BACKUP_DIR.mkdir(exist_ok=True, parents=True)

def _save_json_backup(name: str, folders: list[str], models: list[str], data: dict):
    """Save run data as JSON backup for disaster recovery."""
    try:
        backup_data = {
            "name": name,
            "timestamp": time.time(),
            "folders": folders,
            "models": models,
            "data": data,
            "backup_created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }
        
        backup_file = BACKUP_DIR / f"{name}_{int(time.time())}.json"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON backup saved: {backup_file}")
        
        # Clean up old backups (keep last 50 per run name)
        _cleanup_old_backups(name)
        
    except Exception as e:
        logger.error(f"Failed to create JSON backup for run '{name}': {e}")

def _cleanup_old_backups(run_name: str, keep_count: int = 50):
    """Keep only the most recent backups for each run name."""
    try:
        pattern = f"{run_name}_*.json"
        backups = list(BACKUP_DIR.glob(pattern))
        
        if len(backups) > keep_count:
            # Sort by creation time (embedded in filename)
            backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            for old_backup in backups[keep_count:]:
                old_backup.unlink()
                logger.debug(f"Cleaned up old backup: {old_backup}")
                
    except Exception as e:
        logger.warning(f"Failed to cleanup old backups: {e}")

@contextmanager
def _conn(use_local_fallback=False):
    # Check if we're in Streamlit Cloud (has Turso credentials)
    turso_url = os.getenv("TURSO_DATABASE_URL")
    turso_auth_token = os.getenv("TURSO_AUTH_TOKEN")
    
    # In Streamlit Cloud, these would come from st.secrets
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            turso_url = turso_url or st.secrets.get("TURSO_DATABASE_URL")
            turso_auth_token = turso_auth_token or st.secrets.get("TURSO_AUTH_TOKEN")
    except:
        pass
    
    conn = None
    try:
        if not use_local_fallback and TURSO_AVAILABLE and turso_url and turso_auth_token:
            # Use Turso in production
            conn = libsql.connect(turso_url, auth_token=turso_auth_token)
        else:
            # Use local SQLite for development or fallback
            if TURSO_AVAILABLE:
                conn = libsql.connect(str(DB_PATH))
            else:
                conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
        
        # Create table if it doesn't exist
        conn.execute("""CREATE TABLE IF NOT EXISTS runs (
                       name      TEXT PRIMARY KEY,
                       ts        REAL,
                       folders   TEXT,
                       models    TEXT,
                       json_blob TEXT
                     );""")
        
        # Create indexes for faster queries
        try:
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_ts 
                ON runs(ts DESC)
            """)
        except Exception as e:
            # Index might already exist or not supported
            logger.debug(f"Could not create index idx_runs_ts: {e}")
        
        conn.commit()
        
        yield conn
    finally:
        if conn:
            conn.close()

# ───── public helpers ────────────────────────────────────────────────
def save_run(name: str, folders: list[str], models: list[str], data: dict, max_retries=3):
    """Save a run to the database with retry logic for Turso outages."""
    # Always create JSON backup first (most reliable)
    _save_json_backup(name, folders, models, data)
    
    last_error = None
    
    # Enhanced logging for better monitoring
    logger.info(f"Attempting to save run '{name}' with {len(data)} result items")
    
    # First try with Turso
    for attempt in range(max_retries):
        try:
            with _conn() as c:
                c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)",
                          (name, time.time(), ",".join(folders), ",".join(models),
                           json.dumps(data, ensure_ascii=False)))
                c.commit()
                logger.info(f"✓ Successfully saved run '{name}' to Turso database")
                return  # Success!
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            
            # Check if it's a Turso/Hrana API error
            if any(keyword in error_msg for keyword in ['hrana', 'bad gateway', '502', 'turso', 'timeout']):
                logger.warning(f"⚠️ Turso API error on attempt {attempt + 1}/{max_retries} for run '{name}': {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    logger.error(f"❌ Turso unavailable after {max_retries} attempts, falling back to local SQLite")
                    break
            else:
                # For non-Turso errors, don't retry
                logger.error(f"❌ Non-recoverable database error for run '{name}': {e}")
                raise e
    
    # Fallback to local SQLite if Turso is down
    try:
        logger.info(f"🔄 Saving run '{name}' to local SQLite as fallback")
        with _conn(use_local_fallback=True) as c:
            c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)",
                      (name, time.time(), ",".join(folders), ",".join(models),
                       json.dumps(data, ensure_ascii=False)))
            c.commit()
        logger.info(f"✓ Successfully saved run '{name}' to local database")
    except Exception as fallback_error:
        logger.error(f"❌ Failed to save to local database: {fallback_error}")
        logger.error(f"💾 Run '{name}' is preserved in JSON backup at: {BACKUP_DIR}")
        raise Exception(f"Database save failed. Turso error: {last_error}, Local fallback error: {fallback_error}. JSON backup available in {BACKUP_DIR}")

def list_runs(limit: int = None, offset: int = 0) -> tuple[list[dict], int]:
    """
    List all runs with pagination support and fallback logic for Turso outages.

    Returns:
        tuple: (list of runs, total count)
    """
    try:
        with _conn() as c:
            # Get total count
            total = c.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

            # Get paginated results
            if limit is not None:
                cur = c.execute(
                    "SELECT name, ts, folders, models FROM runs ORDER BY ts DESC LIMIT ? OFFSET ?",
                    (limit, offset)
                )
            else:
                cur = c.execute("SELECT name, ts, folders, models FROM runs ORDER BY ts DESC")

            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            logger.debug(f"Listed {len(rows)} runs from primary database (total: {total})")
            return [dict(zip(cols,row)) for row in rows], total
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['hrana', 'bad gateway', '502', 'turso', 'timeout']):
            logger.warning(f"⚠️ Turso API error in list_runs, falling back to local SQLite: {e}")
            try:
                with _conn(use_local_fallback=True) as c:
                    # Get total count
                    total = c.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

                    # Get paginated results
                    if limit is not None:
                        cur = c.execute(
                            "SELECT name, ts, folders, models FROM runs ORDER BY ts DESC LIMIT ? OFFSET ?",
                            (limit, offset)
                        )
                    else:
                        cur = c.execute("SELECT name, ts, folders, models FROM runs ORDER BY ts DESC")

                    cols = [d[0] for d in cur.description]
                    rows = cur.fetchall()
                    logger.info(f"✓ Listed {len(rows)} runs from local fallback database (total: {total})")
                    return [dict(zip(cols,row)) for row in rows], total
            except Exception as fallback_error:
                logger.error(f"❌ Failed to list runs from local database: {fallback_error}")
                logger.info(f"💾 Check JSON backups in: {BACKUP_DIR}")
                return [], 0  # Return empty list and 0 count instead of crashing
        else:
            logger.error(f"❌ Database error in list_runs: {e}")
            raise e

def load_run(name: str) -> dict|None:
    """Load a specific run with fallback logic for Turso outages."""
    try:
        with _conn() as c:
            cur = c.execute("SELECT json_blob FROM runs WHERE name=?", (name,))
            row = cur.fetchone()
            if row:
                logger.debug(f"Loaded run '{name}' from primary database")
                return json.loads(row[0])
            return None
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['hrana', 'bad gateway', '502', 'turso', 'timeout']):
            logger.warning(f"⚠️ Turso API error in load_run, falling back to local SQLite: {e}")
            try:
                with _conn(use_local_fallback=True) as c:
                    cur = c.execute("SELECT json_blob FROM runs WHERE name=?", (name,))
                    row = cur.fetchone()
                    if row:
                        logger.info(f"✓ Loaded run '{name}' from local fallback database")
                        return json.loads(row[0])
                    return None
            except Exception as fallback_error:
                logger.error(f"❌ Failed to load run from local database: {fallback_error}")
                logger.info(f"💾 Check JSON backups in: {BACKUP_DIR}")
                return None
        else:
            logger.error(f"❌ Database error loading run '{name}': {e}")
            raise e

def load_run_summary(name: str) -> dict|None:
    """
    Load ONLY metadata for a run (no document data).
    Much faster than load_run() for list views.
    """
    try:
        with _conn() as c:
            cur = c.execute("SELECT name, ts, folders, models FROM runs WHERE name=?", (name,))
            row = cur.fetchone()
            if row:
                logger.debug(f"Loaded summary for run '{name}' from primary database")
                return {
                    "name": row[0],
                    "ts": row[1],
                    "folders": row[2].split(',') if row[2] else [],
                    "models": row[3].split(',') if row[3] else []
                }
            return None
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['hrana', 'bad gateway', '502', 'turso', 'timeout']):
            logger.warning(f"⚠️ Turso API error in load_run_summary, falling back to local SQLite: {e}")
            try:
                with _conn(use_local_fallback=True) as c:
                    cur = c.execute("SELECT name, ts, folders, models FROM runs WHERE name=?", (name,))
                    row = cur.fetchone()
                    if row:
                        logger.info(f"✓ Loaded summary for run '{name}' from local fallback database")
                        return {
                            "name": row[0],
                            "ts": row[1],
                            "folders": row[2].split(',') if row[2] else [],
                            "models": row[3].split(',') if row[3] else []
                        }
                    return None
            except Exception as fallback_error:
                logger.error(f"❌ Failed to load run summary from local database: {fallback_error}")
                return None
        else:
            logger.error(f"❌ Database error loading run summary '{name}': {e}")
            raise e

def load_run_with_limit(name: str, max_docs: int = None) -> dict|None:
    """
    Load a run with optional document limit for faster preview.

    Args:
        name: Run name
        max_docs: Maximum number of documents to load (None = all)

    Returns:
        Run data with 'truncated' flag if docs were limited
    """
    full_data = load_run(name)
    if not full_data:
        return None

    docs = full_data.get("docs", [])
    total_docs = len(docs)

    if max_docs is not None and total_docs > max_docs:
        logger.info(f"Loading first {max_docs} of {total_docs} documents for run '{name}'")
        return {
            **full_data,
            "docs": docs[:max_docs],
            "total_docs": total_docs,
            "truncated": True
        }

    return {
        **full_data,
        "total_docs": total_docs,
        "truncated": False
    }

def delete_run(name: str):
    """Delete a run with fallback logic for Turso outages."""
    last_error = None
    
    # Try Turso first
    try:
        with _conn() as c:
            c.execute("DELETE FROM runs WHERE name=?", (name,))
            c.commit()
            logger.info(f"✓ Successfully deleted run '{name}' from Turso database")
            return
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['hrana', 'bad gateway', '502', 'turso', 'timeout']):
            logger.warning(f"⚠️ Turso API error in delete_run, falling back to local SQLite: {e}")
            last_error = e
        else:
            logger.error(f"❌ Database error deleting run '{name}': {e}")
            raise e
    
    # Fallback to local SQLite
    try:
        with _conn(use_local_fallback=True) as c:
            c.execute("DELETE FROM runs WHERE name=?", (name,))
            c.commit()
        logger.info(f"✓ Successfully deleted run '{name}' from local database")
    except Exception as fallback_error:
        logger.error(f"❌ Failed to delete run from local database: {fallback_error}")
        raise Exception(f"Delete failed. Turso error: {last_error}, Local fallback error: {fallback_error}")

# ───── disaster recovery helpers ────────────────────────────────────
def export_all_runs(export_path: Path = None) -> Path:
    """Export all runs to a single JSON file for disaster recovery."""
    if export_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        export_path = RESULTS / f"full_export_{timestamp}.json"
    
    try:
        runs_metadata = list_runs()
        export_data = {
            "export_timestamp": time.time(),
            "export_date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_runs": len(runs_metadata),
            "runs": []
        }
        
        for run_meta in runs_metadata:
            run_data = load_run(run_meta['name'])
            if run_data:
                export_data["runs"].append({
                    "name": run_meta['name'],
                    "timestamp": run_meta['ts'],
                    "folders": run_meta['folders'].split(',') if run_meta['folders'] else [],
                    "models": run_meta['models'].split(',') if run_meta['models'] else [],
                    "data": run_data
                })
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Exported {len(export_data['runs'])} runs to: {export_path}")
        return export_path
        
    except Exception as e:
        logger.error(f"❌ Failed to export runs: {e}")
        raise

def list_json_backups() -> list[dict]:
    """List all available JSON backups for manual recovery."""
    try:
        backups = []
        for backup_file in BACKUP_DIR.glob("*.json"):
            try:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
                
                backups.append({
                    "filename": backup_file.name,
                    "filepath": str(backup_file),
                    "run_name": backup_data.get("name", "unknown"),
                    "backup_created": backup_data.get("backup_created", "unknown"),
                    "timestamp": backup_data.get("timestamp", 0),
                    "folders": backup_data.get("folders", []),
                    "models": backup_data.get("models", [])
                })
            except Exception as e:
                logger.warning(f"Could not parse backup file {backup_file}: {e}")
        
        # Sort by timestamp, newest first
        backups.sort(key=lambda x: x["timestamp"], reverse=True)
        logger.info(f"Found {len(backups)} JSON backups")
        return backups
        
    except Exception as e:
        logger.error(f"❌ Failed to list JSON backups: {e}")
        return []

def restore_from_json_backup(backup_filepath: str) -> str:
    """Restore a run from JSON backup."""
    try:
        with open(backup_filepath, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        name = backup_data["name"]
        folders = backup_data["folders"]
        models = backup_data["models"]
        data = backup_data["data"]
        
        # Save without creating another backup (to avoid recursion)
        logger.info(f"🔄 Restoring run '{name}' from backup: {backup_filepath}")
        
        # Try to save directly to database
        with _conn() as c:
            c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)",
                      (name, backup_data["timestamp"], ",".join(folders), ",".join(models),
                       json.dumps(data, ensure_ascii=False)))
            c.commit()
        
        logger.info(f"✓ Successfully restored run '{name}' from JSON backup")
        return name
        
    except Exception as e:
        logger.error(f"❌ Failed to restore from backup {backup_filepath}: {e}")
        raise