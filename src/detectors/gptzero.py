"""
GPTZero detector wrapper
------------------------
• Adds local caching via the @cached decorator.
• Exposes a lightweight `get()` helper so other modules can retrieve
  cached results without performing a network request.
"""

import requests

from ..cache import cached, get as _cache_get
from ..config import GPTZERO_API_KEY
from ..rate_limiter import wait as _rate_wait


@cached("gptzero")
def detect_ai(text: str, version: str = "2025-03-13-base"):
    """
    Query GPTZero’s /predict/text endpoint and return the raw JSON.

    Results are automatically stored in the SQLite cache by the
    @cached decorator, so subsequent calls with the same *text*
    are served from disk.
    """
    _rate_wait("gptzero")  # global 14-req/min token bucket
    url = "https://api.gptzero.me/v2/predict/text"
    headers = {
        "x-api-key": GPTZERO_API_KEY,
        "Content-Type": "application/json",
    }
    data = {"document": text, "version": version, "multilingual": False}
    resp = requests.post(url, headers=headers, json=data, timeout=60)
    resp.raise_for_status()
    return resp.json()


# -------------------------------------------------------------------------
# Public helper: *cache-only* accessor (no external call)
# -------------------------------------------------------------------------
def get(detector: str, text: str):
    """
    Fetch the cached GPTZero response for *text*.

    Parameters
    ----------
    detector : str
        Should be the literal string ``"gptzero"`` – included to keep the
        same call signature as `sapling.get()` and allow future variants.
    text : str
        The original text whose cached score is requested.

    Returns
    -------
    dict | None
        Cached JSON blob if available, otherwise ``None``.
    """
    return _cache_get(detector, text)
