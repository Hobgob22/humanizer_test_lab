# src/api/dependencies.py
"""
Shared dependencies for API endpoints.
Provides database connections, authentication, etc.
"""

from typing import Generator
import sqlite3
from contextlib import contextmanager
from fastapi import HTTPException, Header

from src.paths import RESULTS
from src.config import APP_AUTH_KEY

# Database connection helpers
def get_job_db_conn() -> Generator[sqlite3.Connection, None, None]:
    """Get a connection to the jobs database."""
    db_path = RESULTS / "jobs.sqlite"
    conn = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def get_runs_db_conn():
    """Get a connection to the runs database."""
    from ..results_db import _conn
    with _conn() as conn:
        yield conn

# Authentication dependency
async def verify_api_key(x_api_key: str = Header(None)) -> str:
    """Verify API key from header."""
    if not APP_AUTH_KEY:
        # If no auth key configured, allow all requests (dev mode)
        return "dev"
    
    if not x_api_key or x_api_key != APP_AUTH_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return x_api_key

