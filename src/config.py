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
SAPLING_API_KEY           = os.getenv("SAPLING_API_KEY", "")
GEMINI_API_KEY            = os.getenv("GEMINI_API_KEY", "")

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
# User-specified limits: OpenAI~50/unlimited, Gemini/Detectors ~15 rpm
HUMANIZER_MAX_WORKERS = 30   # OpenAI (doc mode)
GEMINI_MAX_WORKERS    = 30   # ~15 req/min
DETECTOR_MAX_WORKERS  = 10   # ~15 req/min each

# New: cap paragraph-level concurrency to keep threads in check
PARA_MAX_WORKERS      = 8
