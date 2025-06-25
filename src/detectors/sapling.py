"""
Sapling detector with API key rotation

Features:
- Rotates API keys on 429 errors
- Tracks key freshness (least recently used)
- Reads from cache for original documents
- No cache writes (to avoid bloating with humanized drafts)
- Character-based rate limiting
"""

from __future__ import annotations

import time
import requests
import threading
from collections import deque          # ← NEW
from typing import Dict, List, Optional

from ..config import SAPLING_API_KEYS
from ..rate_limiter import wait_sapling

_MAX_RETRIES = 5
_START_DELAY = 2  # seconds


class SaplingClient:
    """Manages multiple Sapling API keys with rotation on 429 **and**
    per-key 120 000-char / 120-second quota."""

    _CHAR_LIMIT = 120_000            # chars per 120 s
    _WINDOW_SEC = 120                # rolling window
    _COOLDOWN_429 = 300              # 5 min cool-down after any 429

    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("No Sapling API keys provided")

        self.api_keys = api_keys

        # ── per-key state ────────────────────────────────────────────────
        self.key_last_used: Dict[str, float] = {k: 0.0 for k in api_keys}
        self.key_last_429: Dict[str, float] = {k: 0.0 for k in api_keys}
        self.key_usage: Dict[str, deque]   = {k: deque() for k in api_keys}  # (ts, chars)
        # ──────────────────────────────────────────────────────────────────
        self._lock = threading.Lock()
        print(f"[Sapling] Initialized with {len(api_keys)} API keys")

    # ── helpers ──────────────────────────────────────────────────────────
    def _prune_usage(self, key: str, now: float) -> int:
        q = self.key_usage[key]
        while q and now - q[0][0] >= self._WINDOW_SEC:
            q.popleft()
        return sum(c for _, c in q)

    def _key_has_quota(self, key: str, chars: int, now: float) -> bool:
        return self._prune_usage(key, now) + chars <= self._CHAR_LIMIT

    def _select_key(self, chars: int) -> Optional[str]:
        now = time.time()
        with self._lock:
            candidates = [
                k for k in self.api_keys
                if (now - self.key_last_429.get(k, 0) > self._COOLDOWN_429)
                and self._key_has_quota(k, chars, now)
            ]
            if not candidates:
                return None
            return min(candidates, key=lambda k: self.key_last_used.get(k, 0.0))

    def _record_usage(self, key: str, chars: int) -> None:
        with self._lock:
            self.key_usage[key].append((time.time(), chars))
            self.key_last_used[key] = time.time()

    def _note_429(self, key: str) -> None:
        with self._lock:
            self.key_last_429[key] = time.time()
            self.key_last_used[key] = time.time()
            print(f"[Sapling] Key got 429, rotating… (…{key[-6:]})")

    # ── main call ────────────────────────────────────────────────────────
    def detect(self, text: str) -> dict:
        """Call Sapling /aidetect with per-key quota & automatic rotation."""
        url        = "https://api.sapling.ai/api/v1/aidetect"
        char_count = len(text)
        attempts   = 0
        last_error = None

        while attempts < len(self.api_keys) * 2:      # two full rounds
            api_key = self._select_key(char_count)

            if api_key is None:                       # everyone busy/exhausted
                time.sleep(1)
                continue

            attempts += 1
            try:
                resp = requests.post(
                    url, json={"key": api_key, "text": text}, timeout=60
                )

                if resp.status_code == 429:           # short-term OR daily cap
                    self._note_429(api_key)
                    time.sleep(1)
                    continue

                resp.raise_for_status()
                self._record_usage(api_key, char_count)
                return resp.json()

            except requests.RequestException as e:    # network / 5xx, etc.
                last_error = e
                time.sleep(1)                         # quick hop to next key

        raise RuntimeError(
            f"All {len(self.api_keys)} Sapling API keys exhausted. "
            f"Last error: {last_error}"
        )


# Global client instance
_client: Optional[SaplingClient] = None


def _get_client() -> SaplingClient:
    """Get or create the global Sapling client."""
    global _client
    if _client is None:
        if not SAPLING_API_KEYS:
            raise ValueError("No Sapling API keys configured in environment")
        _client = SaplingClient(SAPLING_API_KEYS)
    return _client


def detect_ai(text: str, *, skip_cache: bool = False) -> dict:
    """
    Main entry point - compatible with existing code.
    
    No caching is performed for new API calls (all use rotating keys).
    The skip_cache parameter is kept for compatibility but ignored.
    """
    client = _get_client()
    
    # Retry logic for resilience
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return client.detect(text)
        except RuntimeError as e:
            # All keys exhausted
            if "All" in str(e) and "exhausted" in str(e):
                if attempt == _MAX_RETRIES:
                    raise
                # Wait longer before retrying when all keys are exhausted
                wait_time = min(60 * attempt, 300)  # Max 5 min wait
                print(f"[Sapling] All keys exhausted, waiting {wait_time}s before retry {attempt}/{_MAX_RETRIES}")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(_START_DELAY * (2 ** (attempt - 1)))


# Compatibility function - check cache for reads only
def get(detector: str, text: str):
    """Check cache for existing scores (read-only, no writes)."""
    from ..cache import get as _cache_get
    return _cache_get(detector, text)