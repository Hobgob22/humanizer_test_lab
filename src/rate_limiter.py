"""
Central token-bucket rate-limiter.

• Throttles services with requests-per-minute quotas:
  – Gemini free-tier (≤ 15 req/min)
  – GPTZero          (≤ 15 req/min)
  – Sapling          (≤ 15 req/min)
• Thread-safe and shared by every worker thread.

Simply call  ➜  rate_limiter.wait("api_name")  ➜  then issue the request.
"""

from __future__ import annotations

import threading
import time
from collections import deque

# api_name → (max_requests, window_seconds)
# We keep one request “in hand” (14 < 15) to protect against clock drift.
_LIMITS: dict[str, tuple[int, int]] = {
    "gemini": (14, 60),
    "gptzero": (14, 60),
    "sapling": (14, 60),
}

_queues: dict[str, deque[float]] = {api: deque() for api in _LIMITS}
_lock = threading.Lock()


def wait(api: str) -> None:
    """
    Block until a token is available for *api*.

    Resolution ≈ 10 ms – more than enough for minute-scale quotas.
    """
    if api not in _LIMITS:
        return # No limit for this API

    max_req, window = _LIMITS[api]

    while True:
        with _lock:
            now = time.monotonic()
            q = _queues[api]

            # Purge timestamps that slid out of the window.
            while q and now - q[0] >= window:
                q.popleft()

            if len(q) < max_req:
                q.append(now)
                return                              # token granted ✅

            # Otherwise sleep until the oldest request leaves the window.
            sleep_for = window - (now - q[0]) + 0.05   # 50 ms safety margin

        time.sleep(sleep_for)