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
    version: str = "2025-11-28-base",
    *,
    skip_cache: bool = False,  # flag consumed by @cached
):
    """
    Query GPTZero's /predict/text endpoint and return the raw JSON.

    The ``skip_cache`` keyword is swallowed by the decorator and is
    included here only so callers can pass it safely.
    
    Rate limit: 500 requests/minute (30,000 requests/hour)
    """
    # Validate inputs
    if not text or not text.strip():
        raise ValueError("GPTZero API: text cannot be empty")
    
    if not GPTZERO_API_KEY:
        raise ValueError("GPTZero API: GPTZERO_API_KEY is not set")
    
    # Check text length (GPTZero may have limits)
    if len(text) > 1000000:  # 1MB limit (approximate)
        raise ValueError(f"GPTZero API: text too long ({len(text)} chars, max ~1M)")
    
    _rate_wait("gptzero")  # global 500-req/min token bucket
    url = "https://api.gptzero.me/v2/predict/text"
    headers = {"x-api-key": GPTZERO_API_KEY, "Content-Type": "application/json"}
    data = {"document": text, "version": version, "multilingual": False}
    
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        
        # Enhanced error handling with detailed error messages
        if resp.status_code != 200:
            error_msg = f"GPTZero API error {resp.status_code}"
            try:
                error_detail = resp.json()
                if isinstance(error_detail, dict):
                    error_msg += f": {error_detail.get('message', error_detail.get('error', str(error_detail)))}"
                else:
                    error_msg += f": {error_detail}"
            except:
                error_msg += f": {resp.text[:500]}"  # First 500 chars of response
            
            # Include request details for debugging
            error_msg += f" | Request: text_length={len(text)}, version={version}, key_preview={GPTZERO_API_KEY[:10]}..."
            
            # Raise HTTPError with enhanced message
            raise requests.exceptions.HTTPError(error_msg, response=resp)
        
        return resp.json()
    except requests.exceptions.HTTPError:
        # Re-raise HTTPError (already has enhanced message)
        raise
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"GPTZero API request failed: {str(e)}") from e


# ----- Public helper: cache-only accessor (unchanged) -----------------
def get(detector: str, text: str):
    return _cache_get(detector, text)