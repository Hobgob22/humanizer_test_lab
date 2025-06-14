import os
from pathlib import Path
from typing import List
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
# 4 · HUMANIZER MODEL LIST  (env-overrideable)
# ────────────────────────────────────────────────────────────────
_DEFAULT_MODELS = [
    "ft:gpt-4o-mini-2024-07-18:litero-ai:v4-short-simple:9oaYlNl2",
    "ft:gpt-4.1-2025-04-14:litero-ai:hum30start:BcBvzILe",
    "ft:gpt-4o-2024-08-06:litero-ai:hum30raw:BcCFkyvO",
]
_env_models = [
    m.strip() for m in os.getenv("HUMANIZER_MODELS", "").split(",") if m.strip()
]
HUMANIZER_MODELS: List[str] = _env_models or _DEFAULT_MODELS

# ────────────────────────────────────────────────────────────────
# 5 · GENERAL TUNABLES
# ────────────────────────────────────────────────────────────────
REHUMANIZE_N        = int(os.getenv("REHUMANIZE_N",        5))
DEFAULT_THRESHOLD   = float(os.getenv("AI_THRESHOLD",      0.25))
MIN_WORDS_PARAGRAPH = int(os.getenv("MIN_WORDS_PARAGRAPH", 15))
MAX_ITERATIONS      = int(os.getenv("MAX_ITER",            5))

# ────────────────────────────────────────────────────────────────
# 6 · THREAD / ASYNC CONCURRENCY CAPS  (env-overrideable)
# ────────────────────────────────────────────────────────────────
# User-specified limits: OpenAI~50/unlimited, Gemini/Detectors ~15 rpm
HUMANIZER_MAX_WORKERS = int(os.getenv("HUMANIZER_MAX_WORKERS", 50))   # OpenAI (doc mode)
GEMINI_MAX_WORKERS    = int(os.getenv("GEMINI_MAX_WORKERS",     5))   # ~15 req/min
DETECTOR_MAX_WORKERS  = int(os.getenv("DETECTOR_MAX_WORKERS",   5))   # ~15 req/min each

# New: cap paragraph-level concurrency to keep threads in check
PARA_MAX_WORKERS      = int(os.getenv("PARA_MAX_WORKERS",       8))
