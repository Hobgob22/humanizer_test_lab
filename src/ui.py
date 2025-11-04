# src/ui.py - v6.0 (Refactored with modular pages)
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

from config import OPENAI_API_KEY


# ──────────────────────────── Authentication ─────────────────────────────
from auth import require_login

# enforce login before anything else
require_login()

# Hard-stop if the key is missing so the UI doesn't freeze later.
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is empty – create a .env file or export the variable."
    )

# ─────────────────── project imports / path bootstrap ────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import page functions
from src.pages import page_new_run, page_runs, page_browser
from src.pages.utils import show_performance_metrics

# ───────────────────────── page config ───────────────────────
# Optimize page config for better performance
st.set_page_config(
    page_title="Humanizer Test-Bench", 
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# ─────────────────────────── Navigation & Routing ─────────────────────────
# Import shared sidebar layout
from src.pages._shared_layout import setup_sidebar
setup_sidebar()

# ──────────────────────────── Page rendering ──────────────────────────
# For the main ui.py page, show the new run page by default
page_new_run()
