from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA      = ROOT / "data"
AI_DOCS   = DATA / "ai_texts"
HUMAN_DOCS= DATA / "human_texts"
AI_PARAS = DATA / "ai_paras"
HUMAN_PARAS = DATA / "human_paras"

RESULTS   = ROOT / "results"
CACHE_DIR = ROOT / "cache"
LOG_DIR   = ROOT / "logs"

for p in (RESULTS, CACHE_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)
