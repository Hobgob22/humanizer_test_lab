"""
Sapling detector wrapper

Changes (2025-06-17)
--------------------
* Accepts a **`skip_cache`** keyword so the caller can bypass the cache.
  The decorator handles the flag; the body ignores it.
"""

from __future__ import annotations

import time
import requests

from ..cache import cached, get as _cache_get
from ..config import SAPLING_API_KEY
from ..rate_limiter import wait as _rate_wait

_MAX_RETRIES = 3
_START_DELAY = 2  # seconds


@cached("sapling")
def detect_ai(text: str, *, skip_cache: bool = False) -> dict:  # new kwarg
    """
    Call Sapling’s /aidetect endpoint with automatic retries.

    ``skip_cache`` is consumed by the decorator; it’s present only
    to keep the signature compatible with pipeline calls.
    """
    url = "https://api.sapling.ai/api/v1/aidetect"
    payload = {"key": SAPLING_API_KEY, "text": text}

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            _rate_wait("sapling")  # 14 requests per minute
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError:
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(_START_DELAY * (2 ** (attempt - 1)))


# ----- Public helper: cache-only accessor (unchanged) -----------------
def get(detector: str, text: str):
    return _cache_get(detector, text)
