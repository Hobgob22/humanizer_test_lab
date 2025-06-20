"""
Tiny SQLite cache so we don’t pay twice for the same detector call.
Thread-safe for Streamlit by opening a short-lived connection per access.

NEW (2025-06-17)
----------------
* `@cached` now understands a **keyword flag** ``skip_cache=True`` so callers
  can opt-out of both the *read* and the *write* on a per-call basis.
* Upstream code passes that flag when scoring **humanised drafts**, which
  prevents them from bloating `detector_cache.sqlite`.
"""

from __future__ import annotations

import hashlib
import json
import time
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict

from .paths import CACHE_DIR

DB_PATH = CACHE_DIR / "detector_cache.sqlite"

# ────────────────────────────── Helpers ──────────────────────────────
def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _connect() -> sqlite3.Connection:
    """
    Open a new connection.  `check_same_thread=False` lets us call
    from Streamlit’s various threads safely.
    `timeout=10` avoids “database is locked” on rapid writes.
    """
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS cache (
               detector     TEXT NOT NULL,
               text_hash    TEXT NOT NULL,
               ts           REAL NOT NULL,
               result_json  TEXT NOT NULL,
               PRIMARY KEY(detector, text_hash)
           );"""
    )
    return conn


# ───────────────────────────── Public API ────────────────────────────
def get(detector: str, text: str) -> Dict[str, Any] | None:
    conn = _connect()
    try:
        cur = conn.execute(
            "SELECT result_json FROM cache WHERE detector=? AND text_hash=?",
            (detector, _hash(text)),
        )
        row = cur.fetchone()
        return json.loads(row[0]) if row else None
    finally:
        conn.close()


def put(detector: str, text: str, result: Dict[str, Any]) -> None:
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO cache(detector, text_hash, ts, result_json)"
            " VALUES (?,?,?,?)",
            (detector, _hash(text), time.time(), json.dumps(result)),
        )
        conn.commit()
    finally:
        conn.close()


def cached(detector_name: str):
    """
    Decorator for detector functions that return a JSON dict.

    Extra keyword supported
    -----------------------
    skip_cache : bool, optional
        • ``False`` (default) – normal behaviour  
          → try read, call API on miss, write result  
        • ``True``            – *bypass* the cache  
          → go straight to the wrapped function, **do not** write result
    """

    def decorator(func: Callable[[str], Dict[str, Any]]):
        def wrapper(text: str, *args, **kwargs):
            # New flag – and pop() it so the wrapped function isn’t confused.
            skip_cache: bool = bool(kwargs.pop("skip_cache", False))

            if not skip_cache:
                hit = get(detector_name, text)
                if hit is not None:
                    return hit

            out = func(text, *args, **kwargs)

            if not skip_cache:
                put(detector_name, text, out)
            return out

        return wrapper

    return decorator
