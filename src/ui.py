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

# ───────────────────────── page config ───────────────────────
st.set_page_config(page_title="Humanizer Test-Bench", layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────── Navigation & Routing ─────────────────────────
st.sidebar.title("🚀 Humanizer Test Bench")
st.sidebar.divider()

PAGE = st.sidebar.radio(
    "Navigation",
    ["New Run", "Benchmark Analysis", "Document Browser"],
    help="Select a page to navigate"
)

st.sidebar.divider()

# Troubleshooting help
with st.sidebar.expander("🛠️ Troubleshooting"):
    st.markdown("""
    **Common Issues:**
    
    **All quality checks showing 0%:**
    - Check if GEMINI_API_KEY is set in .env file
    - Verify Gemini API quota isn't exhausted
    - Enable debug info in draft details
    - Check console/terminal for error messages
    - May need to wait if rate limited
    
    **Paragraph mismatches:**
    - Document structure changed during humanization
    - Try different models or adjust prompts
    - Check MIN_WORDS_PARAGRAPH setting
    
    **High AI scores after humanization:**
    - Model may need fine-tuning
    - Try different humanizer models
    - Increase iterations
    
    **Understanding Zero-shot Success:**
    - Shows % of drafts achieving ≤10% AI detection
    - Higher percentages indicate better performance
    - Both GPTZero and Sapling tracked separately
    
    **Label truncation in tables:**
    - Resize browser window or zoom out
    - Use fullscreen mode
    """)

# Key metrics glossary
with st.sidebar.expander("📖 Metrics Glossary"):
    st.markdown("""
    **AI Detection Scores:**
    - **GPTZero/Sapling**: 0-1 scale (lower = more human-like)
    - **Δ (Delta)**: Change from baseline (negative = improvement)
    
    **Zero-shot Success:**
    - % of drafts with ≤10% AI detection score
    - Higher % = better humanization performance
    
    **Quality Metrics:**
    - **Length OK**: Word count within acceptable range
    - **Same Meaning**: Content meaning preserved
    - **Same Language**: Language consistency maintained
    - **No Missing Info**: All information retained
    - **Citation Preserved**: Academic citations intact
    - **Citation Content OK**: Citation text unchanged
    
    **Modes:**
    - **Doc Mode**: Entire document rewritten at once
    - **Para Mode**: Each paragraph rewritten separately
    """)

st.sidebar.caption(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ──────────────────────────── Page rendering with placeholder ──────────
# Hold all page widgets inside a single persistent placeholder.
# When the user navigates to another tab we EMPTY that placeholder
# first, so the previous page disappears instantly instead of hanging
# around while the new one renders top-to-bottom.

if "page_placeholder" not in st.session_state:
    st.session_state.page_placeholder = st.empty()

# Clear previous page on navigation change
if st.session_state.get("current_page") != PAGE:
    st.session_state.page_placeholder.empty()
    st.session_state.current_page = PAGE

# Render the selected page inside the placeholder container
with st.session_state.page_placeholder.container():
    if PAGE == "New Run":
        page_new_run()
    elif PAGE == "Benchmark Analysis":
        page_runs()
    else:
        page_browser()
