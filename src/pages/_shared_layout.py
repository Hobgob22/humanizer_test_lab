# src/pages/_shared_layout.py
"""
Shared layout components for all pages.
Includes sidebar navigation and authentication.
"""

import sys
import time
from pathlib import Path

import streamlit as st

from config import OPENAI_API_KEY
from auth import require_login
from src.pages.utils import show_performance_metrics

# ─────────────────── project imports / path bootstrap ────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def setup_sidebar():
    """Setup the shared sidebar with navigation tabs."""
    # ──────────────────────────── Authentication ─────────────────────────────
    require_login()
    
    # Hard-stop if the key is missing
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is empty – create a .env file or export the variable."
        )
    
    # ─────────────────────────── Navigation & Routing ─────────────────────────
    st.sidebar.title("🚀 Humanizer Test Bench")
    
    # ─────────────────────────── Unified Tab-Based Sidebar ─────────────────────
    tab_nav, tab_tools, tab_help = st.sidebar.tabs(["📍 Navigation", "🛠️ Tools", "📖 Help"])
    
    # ─────────────────────────── Tab 1: Navigation ────────────────────────────
    with tab_nav:
        st.markdown("##### 📍 Navigation")
        st.info("""
        Use the navigation links above to switch between pages:
        - **New Run** - Create and start a new benchmark run
        - **Preview Results** - Quick model screening (for preview mode runs)
        - **Benchmark Analysis** - View and analyze completed runs
        - **Document Browser** - Browse and analyze individual documents
        - **Job Status** - Monitor active and recent jobs
        """)
        
        st.divider()
        # Cache time display to avoid unnecessary updates
        if "sidebar_time" not in st.session_state or time.time() - st.session_state.get("sidebar_time_updated", 0) > 60:
            st.session_state.sidebar_time = time.strftime('%Y-%m-%d %H:%M:%S')
            st.session_state.sidebar_time_updated = time.time()
        st.caption(f"🕐 {st.session_state.sidebar_time}")
    
    # ─────────────────────────── Tab 2: Tools ─────────────────────────────────
    with tab_tools:
        st.markdown("##### Performance Monitor")
        show_performance_metrics()
        
        st.divider()
        
        st.markdown("##### System Info")
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Memory", f"{memory_info.rss / 1024 / 1024:.1f} MB")
            with col2:
                st.metric("CPU", f"{process.cpu_percent():.1f}%")
        except ImportError:
            st.info("Install `psutil` for system metrics")
    
    # ─────────────────────────── Tab 3: Help ───────────────────────────────────
    with tab_help:
        st.markdown("##### 🛠️ Troubleshooting")
        
        troubleshooting = st.expander("Common Issues", expanded=False)
        with troubleshooting:
            st.markdown("""
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
        
        st.divider()
        
        st.markdown("##### 📖 Metrics Glossary")
        
        glossary = st.expander("Key Metrics Explained", expanded=False)
        with glossary:
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
        
        st.divider()
        
        st.markdown("##### 📚 Documentation")
        st.info("""
        For more information:
        - Check README.md for setup instructions
        - Review API docs at http://localhost:8000/docs
        - See HOT_RELOAD.md for development setup
        """)

