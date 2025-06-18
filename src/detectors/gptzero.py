"""
GPTZero detector wrapper

Changes (2025-06-17)
--------------------
* Accepts **`skip_cache`** in the function signature so the caller can
  bypass the cache via the decorator.  The argument is ignored inside
  the function body but must be present to avoid a TypeError.
* Updated rate limit to 500 req/min (30,000 req/hour)
"""

from __future__ import annotations

import requests

from ..cache import cached, get as _cache_get
from ..config import GPTZERO_API_KEY
from ..rate_limiter import wait as _rate_wait


@cached("gptzero")
def detect_ai(
    text: str,
    version: str = "2025-03-13-base",
    *,
    skip_cache: bool = False,  # flag consumed by @cached
):
    """
    Query GPTZero's /predict/text endpoint and return the raw JSON.

    The ``skip_cache`` keyword is swallowed by the decorator and is
    included here only so callers can pass it safely.
    
    Rate limit: 500 requests/minute (30,000 requests/hour)
    """
    _rate_wait("gptzero")  # global 500-req/min token bucket
    url = "https://api.gptzero.me/v2/predict/text"
    headers = {"x-api-key": GPTZERO_API_KEY, "Content-Type": "application/json"}
    data = {"document": text, "version": version, "multilingual": False}
    resp = requests.post(url, headers=headers, json=data, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ----- Public helper: cache-only accessor (unchanged) -----------------
def get(detector: str, text: str):
    return _cache_get(detector, text)