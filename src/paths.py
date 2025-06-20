from pathlib import Path
import os
import shutil


ROOT = Path(__file__).resolve().parents[1]


DATA      = ROOT / "data"
AI_DOCS   = DATA / "ai_texts"
HUMAN_DOCS= DATA / "human_texts"
AI_PARAS = DATA / "ai_paras"
HUMAN_PARAS = DATA / "human_paras"

ENV_RESULTS = os.getenv("RESULTS_DIR")
if ENV_RESULTS:
    RESULTS = Path(ENV_RESULTS).expanduser()
    RESULTS.mkdir(parents=True, exist_ok=True)

    repo_db = ROOT / "results" / "runs.sqlite"
    dest_db = RESULTS / "runs.sqlite"
    if repo_db.exists() and not dest_db.exists():
        shutil.copy2(repo_db, dest_db)
    else:
        RESULTS = ROOT / "results"
        RESULTS.mkdir(parents=True, exist_ok=True)

CACHE_DIR = ROOT / "cache"
LOG_DIR   = ROOT / "logs"

for p in (RESULTS, CACHE_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)
