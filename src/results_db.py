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

import sqlite3, json, time
from pathlib import Path
from .paths import RESULTS

DB_PATH = RESULTS / "runs.sqlite"
DB_PATH.parent.mkdir(exist_ok=True, parents=True)

def _conn():
    c = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    c.execute("""CREATE TABLE IF NOT EXISTS runs (
                   name      TEXT PRIMARY KEY,
                   ts        REAL,
                   folders   TEXT,
                   models    TEXT,
                   json_blob TEXT
                 );""")
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
