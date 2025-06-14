"""
Sapling detector wrapper
------------------------
• Adds 3-attempt retry with exponential back-off (2 s ▸ 4 s ▸ 8 s).
• Respects global rate limits before each attempt.
• Caches successful responses with the @cached decorator.
• Provides a cheap `get()` helper for cache-only look-ups.
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
def detect_ai(text: str) -> dict:
    """
    Call Sapling’s /aidetect endpoint with automatic retries.

    If the request still fails after `_MAX_RETRIES`, the raised
    `HTTPError` is propagated so upstream code can handle it.
    """
    url = "https://api.sapling.ai/api/v1/aidetect"
    payload = {"key": SAPLING_API_KEY, "text": text}

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            _rate_wait("sapling")  # 14 requests per minute
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            if attempt == _MAX_RETRIES:
                raise  # bubble up on final failure
            delay = _START_DELAY * (2 ** (attempt - 1))
            time.sleep(delay)


# -------------------------------------------------------------------------
# Public helper: *cache-only* accessor (no external call)
# -------------------------------------------------------------------------
def get(detector: str, text: str):
    """
    Fetch the cached Sapling response for *text*.

    Parameters
    ----------
    detector : str
        Should be the literal string ``"sapling"`` – mirrors `gptzero.get()`.
    text : str
        The original text whose cached score is requested.

    Returns
    -------
    dict | None
        Cached JSON blob if available, otherwise ``None``.
    """
    return _cache_get(detector, text)
