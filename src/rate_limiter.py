"""
Central token-bucket rate-limiter.

• Throttles services with requests-per-minute quotas:
  – Gemini (≤ 700 req/min)
  – GPTZero (≤ 500 req/min) 
  – Sapling (≤ 120,000 chars/2min)
  – Claude (≤ 700 req/min)
  – OpenAI (≤ 1500 req/min)
• Thread-safe and shared by every worker thread.

Simply call  ➜  rate_limiter.wait("api_name")  ➜  then issue the request.
For Sapling, call  ➜  rate_limiter.wait_sapling(char_count)  ➜  before requests.
"""

from __future__ import annotations

import threading
import time
from collections import deque

# api_name → (max_requests, window_seconds)
_LIMITS: dict[str, tuple[int, int]] = {
    "gemini": (500, 60),      # 700 req/min
    "gptzero": (200, 60),     # 200 req/min
    "claude": (500, 60),      # 700 req/min
    "openai": (1500, 60),     # 1500 req/min
}

# Special handling for Sapling (character-based limit)
_SAPLING_LIMIT = (120_000, 120)  # 120,000 chars per 2 minutes

_queues: dict[str, deque[float]] = {api: deque() for api in _LIMITS}
_sapling_queue: deque[tuple[float, int]] = deque()  # (timestamp, char_count)
_lock = threading.Lock()


def wait(api: str) -> None:
    """
    Block until a token is available for *api*.

    Resolution ≈ 10 ms – more than enough for minute-scale quotas.
    """
    if api not in _LIMITS:
        return  # No limit for this API

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
                return  # token granted ✅

            # Otherwise sleep until the oldest request leaves the window.
            sleep_for = window - (now - q[0]) + 0.05  # 50 ms safety margin

        time.sleep(sleep_for)


def wait_sapling(char_count: int) -> None:
    """
    Block until the character quota is available for Sapling.
    
    Args:
        char_count: Number of characters in the text to be processed
    """
    max_chars, window = _SAPLING_LIMIT
    
    while True:
        with _lock:
            now = time.monotonic()
            
            # Purge old entries outside the window
            while _sapling_queue and now - _sapling_queue[0][0] >= window:
                _sapling_queue.popleft()
            
            # Calculate current character usage
            current_chars = sum(count for _, count in _sapling_queue)
            
            if current_chars + char_count <= max_chars:
                _sapling_queue.append((now, char_count))
                return  # quota available ✅
            
            # Otherwise sleep until enough quota is freed
            if _sapling_queue:
                sleep_for = window - (now - _sapling_queue[0][0]) + 0.05
            else:
                sleep_for = 0.1  # Short sleep if queue is empty but over limit
                
        time.sleep(sleep_for)