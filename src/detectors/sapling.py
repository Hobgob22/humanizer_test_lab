"""
Sapling detector with API key rotation

Features:
- Rotates API keys on 429 errors
- Tracks key freshness (least recently used)
- No caching (as requested)
- Character-based rate limiting
"""

from __future__ import annotations

import time
import requests
import threading
from typing import Dict, List, Optional

from ..config import SAPLING_API_KEYS
from ..rate_limiter import wait_sapling

_MAX_RETRIES = 5
_START_DELAY = 2  # seconds


class SaplingClient:
    """Manages multiple Sapling API keys with rotation on 429."""
    
    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("No Sapling API keys provided")
        
        self.api_keys = api_keys
        self.key_last_used: Dict[str, float] = {key: 0.0 for key in api_keys}
        self.key_last_429: Dict[str, float] = {key: 0.0 for key in api_keys}
        self._lock = threading.Lock()
        
        print(f"[Sapling] Initialized with {len(api_keys)} API keys")
    
    def _get_freshest_key(self) -> str:
        """Get the least recently used key that hasn't had a recent 429."""
        with self._lock:
            now = time.time()
            
            # Filter out keys that got 429 in the last 5 minutes
            available_keys = [
                key for key in self.api_keys
                if now - self.key_last_429.get(key, 0) > 300
            ]
            
            if not available_keys:
                # All keys are in cooldown, use the one with oldest 429
                return min(self.api_keys, key=lambda k: self.key_last_429.get(k, 0))
            
            # Return the least recently used available key
            return min(available_keys, key=lambda k: self.key_last_used.get(k, 0))
    
    def _mark_key_used(self, key: str, got_429: bool = False):
        """Mark a key as used, optionally marking it as rate limited."""
        with self._lock:
            self.key_last_used[key] = time.time()
            if got_429:
                self.key_last_429[key] = time.time()
                print(f"[Sapling] Key got 429, rotating... (key ending in ...{key[-6:]})")
    
    def detect(self, text: str) -> dict:
        """
        Call Sapling's /aidetect endpoint with automatic key rotation.
        
        Returns the raw JSON response.
        """
        url = "https://api.sapling.ai/api/v1/aidetect"
        char_count = len(text)
        
        # Try each key until one works
        attempts = 0
        last_error = None
        
        while attempts < len(self.api_keys) * 2:  # Allow multiple rounds
            attempts += 1
            
            # Get the freshest key
            api_key = self._get_freshest_key()
            
            # Apply rate limiting for character quota
            wait_sapling(char_count)
            
            try:
                payload = {"key": api_key, "text": text}
                resp = requests.post(url, json=payload, timeout=60)
                
                if resp.status_code == 429:
                    # Rate limit hit - mark key and try another
                    self._mark_key_used(api_key, got_429=True)
                    time.sleep(1)  # Brief pause before trying next key
                    continue
                
                resp.raise_for_status()
                
                # Success! Mark key as used (but not rate limited)
                self._mark_key_used(api_key, got_429=False)
                return resp.json()
                
            except requests.HTTPError as e:
                last_error = e
                if resp.status_code != 429:
                    # Non-429 error, might be a real problem
                    raise
            except Exception as e:
                last_error = e
                # Network or other error, try next key
                time.sleep(_START_DELAY)
        
        # All keys exhausted
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
    
    The skip_cache parameter is ignored (no caching).
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


# Compatibility function - no caching
def get(detector: str, text: str):
    """Compatibility function - just calls detect_ai (no cache)."""
    return detect_ai(text)