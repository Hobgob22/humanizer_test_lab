import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# ────────────────────────────────────────────────────────────────
# 1 · ENV FILE LOADING
# ────────────────────────────────────────────────────────────────
env_loaded = load_dotenv(ROOT / ".env")
if not env_loaded:                          # fallback so repo works OOTB
    load_dotenv(ROOT / ".env.example")

# ────────────────────────────────────────────────────────────────
# 2 · APP AUTH KEY
# ────────────────────────────────────────────────────────────────
APP_AUTH_KEY = os.getenv("APP_AUTH_KEY", "")

# ────────────────────────────────────────────────────────────────
# 3 · API KEYS
# ────────────────────────────────────────────────────────────────
OPENAI_API_KEY            = os.getenv("OPENAI_API_KEY", "")
HUMANIZER_OPENAI_API_KEY  = os.getenv("HUMANIZER_OPENAI_API_KEY", "")
GPTZERO_API_KEY           = os.getenv("GPTZERO_API_KEY", "")

# ────────────────────────────────────────────────────────────────
# 3b · SAPLING – primary / fallback key support
# ────────────────────────────────────────────────────────────────
SAPLING_PRIMARY_API_KEY   = os.getenv("SAPLING_PRIMARY_API_KEY", "")

SAPLING_FALLBACK_KEYS_STR = os.getenv("SAPLING_FALLBACK_API_KEYS", "")
SAPLING_FALLBACK_API_KEYS = [
    k.strip() for k in SAPLING_FALLBACK_KEYS_STR.split(",") if k.strip()
]

# Back-compatibility: derive from legacy SAPLING_API_KEYS if new
# vars were not provided.
if not SAPLING_PRIMARY_API_KEY:
    legacy_keys_str = os.getenv("SAPLING_API_KEYS", "")
    legacy_keys = [k.strip() for k in legacy_keys_str.split(",") if k.strip()]
    if legacy_keys:
        SAPLING_PRIMARY_API_KEY  = legacy_keys[0]
        SAPLING_FALLBACK_API_KEYS = legacy_keys[1:]

# Convenience alias retained for existing imports
SAPLING_API_KEYS = (
    ([SAPLING_PRIMARY_API_KEY] if SAPLING_PRIMARY_API_KEY else [])
    + SAPLING_FALLBACK_API_KEYS
)


GEMINI_API_KEY            = os.getenv("GEMINI_API_KEY", "")
CLAUDE_API_KEY            = os.getenv("CLAUDE_API_KEY", "")

# ────────────────────────────────────────────────────────────────
# 4 · GENERAL TUNABLES
# ────────────────────────────────────────────────────────────────
REHUMANIZE_N        = int(os.getenv("REHUMANIZE_N",        5))
ZERO_SHOT_THRESHOLD   = float(os.getenv("ZERO_SHOT_THRESHOLD",      0.10))
MIN_WORDS_PARAGRAPH = int(os.getenv("MIN_WORDS_PARAGRAPH", 15))
MAX_ITERATIONS      = int(os.getenv("MAX_ITER",            5))

# ────────────────────────────────────────────────────────────────
# 5 · THREAD / ASYNC CONCURRENCY CAPS  (env-overrideable)
# ────────────────────────────────────────────────────────────────
# Adjusted based on new rate limits:
# - OpenAI: 1500 req/min (25 req/sec) 
# - Claude: 700 req/min (11.6 req/sec)
# - Gemini: 700 req/min (11.6 req/sec)
# - GPTZero: 500 req/min (8.3 req/sec)
# - Sapling: 120,000 chars/2min (1,000 chars/sec)

HUMANIZER_MAX_WORKERS = int(os.getenv("HUMANIZER_MAX_WORKERS", 100))   # Reduced to prevent rate limiting (OpenAI: 1500 req/min)
GEMINI_MAX_WORKERS    = int(os.getenv("GEMINI_MAX_WORKERS", 70))      # 700 req/min
DETECTOR_MAX_WORKERS  = int(os.getenv("DETECTOR_MAX_WORKERS", 20))   # Mixed detectors

# Sapling-specific limit to prevent character quota exhaustion
# With ~10k chars/doc, limit to 10 concurrent to stay under 120k/2min
SAPLING_MAX_CONCURRENT = int(os.getenv("SAPLING_MAX_CONCURRENT", 20))

# Cap paragraph-level concurrency
PARA_MAX_WORKERS      = int(os.getenv("PARA_MAX_WORKERS", 20))

# ────────────────────────────────────────────────────────────────
# 6 · PIPELINE-LEVEL PARALLELISM
# ────────────────────────────────────────────────────────────────
# Maximum number of documents that may advance through the
# 4-phase pipeline **at the same time**.  Keep conservative –  
# the token-bucket limiter still guards per-API quotas.
MAX_PARALLEL_DOCS     = int(os.getenv("MAX_PARALLEL_DOCS", 10))

# Hard-cap on in-memory log history per job (oldest lines dropped)
LOG_HISTORY_LIMIT     = int(os.getenv("LOG_HISTORY_LIMIT", 500))

# ────────────────────────────────────────────────────────────────
# 7 · API CONFIGURATION
# ────────────────────────────────────────────────────────────────
API_HOST             = os.getenv("API_HOST", "0.0.0.0")
API_PORT             = int(os.getenv("API_PORT", 8000))
API_BASE_URL         = os.getenv("API_BASE_URL", f"http://localhost:{API_PORT}")
WS_URL                = os.getenv("WS_URL", f"ws://localhost:{API_PORT}")
CACHE_TTL_STATISTICS  = int(os.getenv("CACHE_TTL_STATISTICS", 1800))
CACHE_TTL_RUNS        = int(os.getenv("CACHE_TTL_RUNS", 3600))

# ────────────────────────────────────────────────────────────────
# 8 · DEVELOPMENT MODE (for detailed logging)
# ────────────────────────────────────────────────────────────────
# Enable development mode for JSON logging of paragraph mismatches
# Set DEVELOPMENT_MODE=true in .env for local development
IS_DEVELOPMENT = os.getenv("DEVELOPMENT_MODE", "false").lower() in ("true", "1", "yes")