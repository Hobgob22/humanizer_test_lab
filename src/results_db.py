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
from .paths import RESULTS

# Try to import Turso client, fall back to sqlite3 for local development
try:
    import libsql_experimental as libsql
    TURSO_AVAILABLE = True
except ImportError:
    import sqlite3
    TURSO_AVAILABLE = False

DB_PATH = RESULTS / "runs.sqlite"
DB_PATH.parent.mkdir(exist_ok=True, parents=True)

def _conn():
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
    
    if TURSO_AVAILABLE and turso_url and turso_auth_token:
        # Use Turso in production
        c = libsql.connect(turso_url, auth_token=turso_auth_token)
    else:
        # Use local SQLite for development
        if TURSO_AVAILABLE:
            c = libsql.connect(str(DB_PATH))
        else:
            c = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    
    c.execute("""CREATE TABLE IF NOT EXISTS runs (
                   name      TEXT PRIMARY KEY,
                   ts        REAL,
                   folders   TEXT,
                   models    TEXT,
                   json_blob TEXT
                 );""")
    c.commit()
    return c

# ───── public helpers ────────────────────────────────────────────────
def save_run(name: str, folders: list[str], models: list[str], data: dict):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)",
                  (name, time.time(), ",".join(folders), ",".join(models),
                   json.dumps(data, ensure_ascii=False)))
        c.commit()

def list_runs() -> list[dict]:
    with _conn() as c:
        cur = c.execute("SELECT name, ts, folders, models FROM runs ORDER BY ts DESC")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols,row)) for row in cur]

def load_run(name: str) -> dict|None:
    with _conn() as c:
        cur = c.execute("SELECT json_blob FROM runs WHERE name=?", (name,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

def delete_run(name: str):
    with _conn() as c:
        c.execute("DELETE FROM runs WHERE name=?", (name,))
        c.commit()