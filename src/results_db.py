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
        conn.commit()
        
        yield conn
    finally:
        if conn:
            conn.close()

# ───── public helpers ────────────────────────────────────────────────
def save_run(name: str, folders: list[str], models: list[str], data: dict, max_retries=3):
    """Save a run to the database with retry logic for Turso outages."""
    last_error = None
    
    # First try with Turso
    for attempt in range(max_retries):
        try:
            with _conn() as c:
                c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)",
                          (name, time.time(), ",".join(folders), ",".join(models),
                           json.dumps(data, ensure_ascii=False)))
                c.commit()
                return  # Success!
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            
            # Check if it's a Turso/Hrana API error
            if any(keyword in error_msg for keyword in ['hrana', 'bad gateway', '502', 'turso']):
                logger.warning(f"Turso API error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    logger.error(f"Turso unavailable after {max_retries} attempts, falling back to local SQLite")
                    break
            else:
                # For non-Turso errors, don't retry
                raise e
    
    # Fallback to local SQLite if Turso is down
    try:
        logger.info(f"Saving run '{name}' to local SQLite as fallback")
        with _conn(use_local_fallback=True) as c:
            c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)",
                      (name, time.time(), ",".join(folders), ",".join(models),
                       json.dumps(data, ensure_ascii=False)))
            c.commit()
        logger.info(f"Successfully saved run '{name}' to local database")
    except Exception as fallback_error:
        logger.error(f"Failed to save to local database: {fallback_error}")
        raise Exception(f"Database save failed. Turso error: {last_error}, Local fallback error: {fallback_error}")

def list_runs() -> list[dict]:
    """List all runs with fallback logic for Turso outages."""
    try:
        with _conn() as c:
            cur = c.execute("SELECT name, ts, folders, models FROM runs ORDER BY ts DESC")
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            return [dict(zip(cols,row)) for row in rows]
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['hrana', 'bad gateway', '502', 'turso']):
            logger.warning(f"Turso API error in list_runs, falling back to local SQLite: {e}")
            try:
                with _conn(use_local_fallback=True) as c:
                    cur = c.execute("SELECT name, ts, folders, models FROM runs ORDER BY ts DESC")
                    cols = [d[0] for d in cur.description]
                    rows = cur.fetchall()
                    return [dict(zip(cols,row)) for row in rows]
            except Exception as fallback_error:
                logger.error(f"Failed to list runs from local database: {fallback_error}")
                return []  # Return empty list instead of crashing
        else:
            raise e

def load_run(name: str) -> dict|None:
    """Load a specific run with fallback logic for Turso outages."""
    try:
        with _conn() as c:
            cur = c.execute("SELECT json_blob FROM runs WHERE name=?", (name,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else None
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['hrana', 'bad gateway', '502', 'turso']):
            logger.warning(f"Turso API error in load_run, falling back to local SQLite: {e}")
            try:
                with _conn(use_local_fallback=True) as c:
                    cur = c.execute("SELECT json_blob FROM runs WHERE name=?", (name,))
                    row = cur.fetchone()
                    return json.loads(row[0]) if row else None
            except Exception as fallback_error:
                logger.error(f"Failed to load run from local database: {fallback_error}")
                return None
        else:
            raise e

def delete_run(name: str):
    """Delete a run with fallback logic for Turso outages."""
    last_error = None
    
    # Try Turso first
    try:
        with _conn() as c:
            c.execute("DELETE FROM runs WHERE name=?", (name,))
            c.commit()
            return
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['hrana', 'bad gateway', '502', 'turso']):
            logger.warning(f"Turso API error in delete_run, falling back to local SQLite: {e}")
            last_error = e
        else:
            raise e
    
    # Fallback to local SQLite
    try:
        with _conn(use_local_fallback=True) as c:
            c.execute("DELETE FROM runs WHERE name=?", (name,))
            c.commit()
        logger.info(f"Successfully deleted run '{name}' from local database")
    except Exception as fallback_error:
        logger.error(f"Failed to delete run from local database: {fallback_error}")
        raise Exception(f"Delete failed. Turso error: {last_error}, Local fallback error: {fallback_error}")