# src/pages/document_browser.py  – fixed: no nested expanders, JSON views now in tabs
from __future__ import annotations

import sys
import concurrent.futures
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import math  # ← pagination helper
import time # Added for retry logic

# ─────────────────── project root ────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pages.utils import (
    ZERO_SHOT_THRESHOLD,
    colored_metric,
    natural_key,  # ← keep natural sort helper untouched
)
from src.pipeline import load_ai_scores
from src.results_db import list_runs, load_run


@st.cache_data(ttl=3600, show_spinner="Loading AI detection scores...")
def _cached_load_ai_scores(doc_path: str) -> Dict:
    """Cached version of load_ai_scores to prevent repeated API calls."""
    return load_ai_scores(Path(doc_path))




# ═════════════════════ DOCUMENTS BROWSER PAGE ═════════════════════════
def page_browser() -> None:
    """Main entry for the "Document Browser" Streamlit page."""
    st.header("📁 Document Browser")
    st.info("Analyze individual documents and compare them with benchmark runs")

    # ── 1 · Folder selection ────────────────────────────────────────────
    folders: List[str] = st.multiselect(
        "Select folders to browse",
        ["ai_texts", "human_texts", "ai_paras", "human_paras"],
        default=["ai_texts"],
        help="Choose which document folders to display",
    )
    if not folders:
        st.warning("Please select at least one folder")
        return

    # ── 2 · Load & sort documents naturally ─────────────────────────────
    docs: List[Path] = sorted(
        [p for f in folders for p in (ROOT / f"data/{f}").glob("*.docx")],
        key=natural_key,
    )
    if not docs:
        st.warning("No .docx files found in the selected folders")
        return

    # ── 3 · Pre-analyse all button (concurrent) ─────────────────────────
    if st.button("⚡ Analyze all documents", use_container_width=True):
        _pre_analyse_all_optimized(docs)

    # ── 4 · Filename filter ─────────────────────────────────────────────
    search_term: str = st.text_input(
        "🔍 Filter documents", placeholder="Type to filter by filename…"
    )
    if search_term:
        docs = [d for d in docs if search_term.lower() in d.name.lower()]

    # ── 5 · Optional benchmark run comparison ───────────────────────────
    run_opts = [r["name"] for r in list_runs()]
    compare_run: str | None = st.selectbox(
        "Compare with benchmark run (optional)",
        ["None"] + run_opts,
        help="Select a benchmark run to compare AI-detection scores",
    )
    if compare_run == "None":
        compare_run = None

    # ── 6 · Page intro & legend  +  pagination  ─────────────────────────
    page_size    = 10
    total_docs   = len(docs)
    total_pages  = max(1, math.ceil(total_docs / page_size))
    curr_page    = st.session_state.get("doc_page", 0)
    curr_page    = max(0, min(curr_page, total_pages - 1))  # clamp inside range

    st.subheader(f"📄 Documents ({total_docs} total) – Page {curr_page + 1}/{total_pages}")

    # ── pagination controls ──────────────────────────────────────────────
    if total_pages > 1:
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        with col2:
            if st.button("⬅️ Previous", disabled=curr_page == 0):
                st.session_state.doc_page = curr_page - 1
                st.rerun()
        with col3:
            # Direct page jump
            new_page = st.number_input(
                "Jump to page:",
                min_value=1,
                max_value=total_pages,
                value=curr_page + 1,
                step=1,
            ) - 1
            if new_page != curr_page:
                st.session_state.doc_page = new_page
                st.rerun()
        with col4:
            if st.button("➡️ Next", disabled=curr_page == total_pages - 1):
                st.session_state.doc_page = curr_page + 1
                st.rerun()

    # ── show current page of documents ───────────────────────────────────
    start_idx = curr_page * page_size
    end_idx = min(start_idx + page_size, total_docs)
    page_docs = docs[start_idx:end_idx]

    for doc_path in page_docs:
        _display_single_doc_optimized(doc_path, compare_run)


# ═════════════════════ helpers ═════════════════════════════════════════
def _pre_analyse_all_optimized(docs: List[Path]) -> None:
    """
    Optimized pre-loading with better progress tracking and error handling.
    """
    # Check which documents need analysis (not in cache)
    remaining = []
    for doc in docs:
        try:
            # Try to get from cache without computing
            _cached_load_ai_scores(str(doc))
        except:
            remaining.append(doc)
    
    if not remaining:
        st.info("✅ All selected documents are already analyzed!")
        return

    st.info(f"🔄 Analyzing {len(remaining)} documents (cached: {len(docs) - len(remaining)})")
    
    progress_bar = st.progress(0.0, text="Starting analysis...")
    status_container = st.container()
    
    # Reduce max workers to prevent API rate limiting
    max_workers = min(4, len(remaining))  # Reduced from 8 to 4
    
    completed = 0
    errors = []
    
    def _load_with_retry(doc_path: Path) -> None:
        """Load with basic retry logic."""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                _cached_load_ai_scores(str(doc_path))
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    errors.append(f"{doc_path.name}: {str(e)}")
                else:
                    time.sleep(1)  # Brief delay before retry

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_doc = {executor.submit(_load_with_retry, doc): doc for doc in remaining}
        
        for future in concurrent.futures.as_completed(future_to_doc):
            completed += 1
            doc = future_to_doc[future]
            
            progress = completed / len(remaining)
            progress_bar.progress(progress, text=f"Analyzed {doc.name} ({completed}/{len(remaining)})")
    
    progress_bar.empty()
    
    if errors:
        with status_container:
            st.warning(f"⚠️ {len(errors)} documents failed analysis:")
            for error in errors[:5]:  # Show first 5 errors
                st.caption(f"• {error}")
            if len(errors) > 5:
                st.caption(f"... and {len(errors) - 5} more")
    else:
        with status_container:
            st.success(f"✅ Successfully analyzed all {len(remaining)} documents!")


def _display_single_doc_optimized(path: Path, compare_run: str | None) -> None:
    """Optimized document display with lazy loading."""
    with st.expander(f"📄 {path.name}", expanded=False):
        try:
            # Use cached loading
            doc = _cached_load_ai_scores(str(path))
        except Exception as e:
            st.error(f"❌ Failed to analyze document: {str(e)}")
            st.caption("This could be due to API rate limits or document processing issues.")
            return

        # ── meta info ──────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Folder", path.parent.name)
        with col2:
            st.metric("Paragraphs", len(doc["segments"]))
        with col3:
            st.metric("Word Count", sum(len(p.split()) for p in doc["segments"]))

        # ── overall AI-detection scores ────────────────────────────────
        st.markdown("### 🎯 Document-level AI detection")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("GPTZero", f"{doc['overall']['gptzero']:.3f}")
        with col2:
            st.metric("Sapling", f"{doc['overall']['sapling']:.3f}")

        if doc["overall"]["gptzero"] <= ZERO_SHOT_THRESHOLD:
            st.success(f"✅ GPTZero zero-shot (≤10 %) : {doc['overall']['gptzero']:.3f}")
        if doc["overall"]["sapling"] <= ZERO_SHOT_THRESHOLD:
            st.success(f"✅ Sapling zero-shot (≤10 %) : {doc['overall']['sapling']:.3f}")

        # ── paragraph table ────────────────────────────────────────────
        st.markdown("### 📊 Paragraph-level analysis")
        para_df = pd.DataFrame(
            {
                "Paragraph": range(1, len(doc["segments"]) + 1),
                "Words": [len(p.split()) for p in doc["segments"]],
                "GPTZero (group)": doc["group_par"]["gptzero"],
                "GPTZero (ind)": doc["ind_par"]["gptzero"],
                "Sapling (group)": doc["group_par"]["sapling"],
                "Sapling (ind)": doc["ind_par"]["sapling"],
            }
        )
        st.dataframe(
            para_df.style.format(
                {
                    "GPTZero (group)": "{:.3f}",
                    "GPTZero (ind)": "{:.3f}",
                    "Sapling (group)": "{:.3f}",
                    "Sapling (ind)": "{:.3f}",
                }
            ).background_gradient(
                subset=[
                    "GPTZero (group)",
                    "GPTZero (ind)",
                    "Sapling (group)",
                    "Sapling (ind)",
                ],
                cmap="RdYlGn_r",
                vmin=0,
                vmax=1,
            ),
            use_container_width=True,
            height=300,
        )

        # ── raw JSON score views (use tabs → no nested expanders) ──────
        st.markdown("### 🗂️ Raw detector scores")
        tab_gz, tab_sp = st.tabs(["🟠 GPTZero", "🟢 Sapling"])

        with tab_gz:
            st.json(
                {
                    "document_score": doc["overall"]["gptzero"],
                    "paragraph_scores_group": doc["group_par"]["gptzero"],
                    "paragraph_scores_ind": doc["ind_par"]["gptzero"],
                }
            )

        with tab_sp:
            st.json(
                {
                    "document_score": doc["overall"]["sapling"],
                    "paragraph_scores_group": doc["group_par"]["sapling"],
                    "paragraph_scores_ind": doc["ind_par"]["sapling"],
                }
            )

        # ── optionally show raw text segments ──────────────────────────
        if st.checkbox(f"Show text segments for {path.name}", key=f"show_seg_{path.name}"):
            for i, seg in enumerate(doc["segments"], 1):
                st.markdown(f"**Paragraph {i}:**")
                st.text_area("", seg, height=100, disabled=True, key=f"seg_{path.name}_{i}")

        # ── comparison with benchmark run (if selected) ────────────────
        if compare_run:
            st.markdown(f"### 🔄 Comparison with benchmark: **{compare_run}**")
            run_data = load_run(compare_run) or {}

            drafts = [
                dr
                for d in run_data.get("docs", [])
                if d["document"] == path.name
                for dr in d["runs"]
            ]
            if not drafts:
                st.info("This document was not processed in the selected run")
                return

            # split by mode
            doc_mode = [d for d in drafts if d["mode"] == "doc"]
            para_mode = [d for d in drafts if d["mode"] == "para"]

            _render_mode_block("Document mode results", doc_mode, doc["overall"], column=0)
            _render_mode_block("Paragraph mode results", para_mode, doc["overall"], column=1)


def _render_mode_block(
    title: str,
    drafts: List[Dict],
    baseline: Dict[str, float],
    column: int,
) -> None:
    """Render a two-column comparison block for a specific mode."""
    if not drafts:
        return

    avg_gz = float(np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in drafts]))
    avg_sp = float(np.mean([d["scores_after"]["group_doc"]["sapling"] for d in drafts]))
    zs_gz = sum(
        1 for d in drafts if d["scores_after"]["group_doc"]["gptzero"] <= ZERO_SHOT_THRESHOLD
    )
    zs_sp = sum(
        1 for d in drafts if d["scores_after"]["group_doc"]["sapling"] <= ZERO_SHOT_THRESHOLD
    )

    cols = st.columns(2)
    with cols[column]:
        st.markdown(f"**{title}:**")
        colored_metric("Avg GPTZero", f"{avg_gz:.3f}", avg_gz - baseline["gptzero"])
        colored_metric("Avg Sapling", f"{avg_sp:.3f}", avg_sp - baseline["sapling"])
        st.metric("Zero-shot success", f"GZ {zs_gz}/{len(drafts)} | SP {zs_sp}/{len(drafts)}")


# ──────────────────────────── Standalone Page Setup ────────────────────
# When this file is executed directly by Streamlit's multi-page system,
# set up the page config and sidebar, then call the page function
# Check if we're being run as a standalone page (not imported)
if __name__ == "__main__":
    # Page config
    st.set_page_config(page_title="Document Browser - Humanizer Test-Bench", layout="wide", initial_sidebar_state="expanded")
    
    # Setup shared sidebar
    from src.pages._shared_layout import setup_sidebar
    setup_sidebar()
    
    # Call the page function
    page_browser()