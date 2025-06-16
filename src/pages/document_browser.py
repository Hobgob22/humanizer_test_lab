# src/pages/document_browser.py - Document browsing and comparison
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

from .utils import colored_metric, ZERO_SHOT_THRESHOLD
from ..results_db import list_runs, load_run
from ..pipeline import load_ai_scores

# ─────────────────── project root ────────────────────
ROOT = Path(__file__).resolve().parents[2]

# ═════════════════════ DOCUMENTS BROWSER PAGE ═════════════════════════
def page_browser():
    st.header("📁 Document Browser")
    st.info("Analyze individual documents and compare with benchmark runs")

    # Folder selection
    folders = st.multiselect(
        "Select folders to browse",
        ["ai_texts", "human_texts", "mixed_texts"],
        default=["ai_texts"],
        help="Choose which document folders to display"
    )
    
    if not folders:
        st.warning("Please select at least one folder")
        return

    # Get documents
    docs = sorted([p for f in folders for p in (ROOT / f"data/{f}").glob("*.docx")])
    if not docs:
        st.warning("No documents found in selected folders")
        return

    # Document filter
    search_term = st.text_input("🔍 Filter documents", placeholder="Type to filter by filename...")
    if search_term:
        docs = [d for d in docs if search_term.lower() in d.name.lower()]

    # Comparison run selection
    run_opts = [r["name"] for r in list_runs()]
    compare_run = st.selectbox(
        "Compare with benchmark run (optional)",
        ["None"] + run_opts,
        help="Select a benchmark run to compare scores"
    )

    # Display documents
    st.subheader(f"📄 Documents ({len(docs)} found)")
    
    with st.expander("ℹ️ Understanding AI detection scores", expanded=False):
        st.markdown("""
        **AI Detection Scores (0-1 scale):**
        - **0.0 - 0.1**: Very low AI detection (appears human-written)
        - **0.1 - 0.3**: Low AI detection
        - **0.3 - 0.7**: Moderate AI detection
        - **0.7 - 0.9**: High AI detection
        - **0.9 - 1.0**: Very high AI detection (clearly AI-generated)
        
        Both GPTZero and Sapling provide document and paragraph-level scores.
        """)
    
    for doc_path in docs:
        _display_single_doc(doc_path, compare_run if compare_run != "None" else None)

def _display_single_doc(path: Path, compare_run: str | None):
    """Enhanced single document display with colored metrics"""
    with st.expander(f"📄 {path.name}", expanded=False):
        # Load and cache AI scores
        if "score_cache" not in st.session_state:
            st.session_state.score_cache = {}
        
        if path.name not in st.session_state.score_cache:
            with st.spinner("Analyzing document..."):
                st.session_state.score_cache[path.name] = load_ai_scores(path)

        doc = st.session_state.score_cache[path.name]
        
        # Document info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Folder", path.parent.name)
        with col2:
            st.metric("Paragraphs", len(doc["segments"]))
        with col3:
            total_words = sum(len(seg.split()) for seg in doc["segments"])
            st.metric("Word Count", total_words)
        
        # Overall scores
        st.markdown("### 🎯 Document-Level AI Detection")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("GPTZero Score", f"{doc['overall']['gptzero']:.3f}",
                     help="Overall AI detection score from GPTZero (0-1)")
        with col2:
            st.metric("Sapling Score", f"{doc['overall']['sapling']:.3f}",
                     help="Overall AI detection score from Sapling (0-1)")
        
        # Zero-shot indicators
        if doc['overall']['gptzero'] <= ZERO_SHOT_THRESHOLD:
            st.success(f"✅ GPTZero zero-shot: Document scores below 10% threshold ({doc['overall']['gptzero']:.3f})")
        if doc['overall']['sapling'] <= ZERO_SHOT_THRESHOLD:
            st.success(f"✅ Sapling zero-shot: Document scores below 10% threshold ({doc['overall']['sapling']:.3f})")
        
        # Paragraph-level analysis
        st.markdown("### 📊 Paragraph-Level Analysis")
        
        para_df = pd.DataFrame({
            "Paragraph": range(1, len(doc["segments"]) + 1),
            "Words": [len(seg.split()) for seg in doc["segments"]],
            "GPTZero (Group)": doc["group_par"]["gptzero"],
            "GPTZero (Ind)": doc["ind_par"]["gptzero"],
            "Sapling (Group)": doc["group_par"]["sapling"],
            "Sapling (Ind)": doc["ind_par"]["sapling"]
        })
        
        # Apply conditional formatting
        st.dataframe(
            para_df.style.format({
                "GPTZero (Group)": "{:.3f}",
                "GPTZero (Ind)": "{:.3f}",
                "Sapling (Group)": "{:.3f}",
                "Sapling (Ind)": "{:.3f}"
            }).background_gradient(subset=["GPTZero (Group)", "GPTZero (Ind)", "Sapling (Group)", "Sapling (Ind)"], 
                                  cmap='RdYlGn_r', vmin=0, vmax=1),
            use_container_width=True,
            height=300
        )
        
        # Show segments
        if st.checkbox(f"Show text segments for {path.name}", key=f"show_seg_{path.name}"):
            for i, seg in enumerate(doc["segments"], 1):
                st.markdown(f"**Paragraph {i}:**")
                st.text_area("", seg, height=100, disabled=True, key=f"seg_{path.name}_{i}")
        
        # Comparison with run
        if compare_run:
            st.markdown(f"### 🔄 Comparison with run: **{compare_run}**")
            
            run_data = load_run(compare_run) or {}
            drafts = [
                dr for d in run_data.get("docs", [])
                if d["document"] == path.name
                for dr in d["runs"]
            ]
            
            if drafts:
                # Summary comparison with colored metrics
                doc_mode_drafts = [d for d in drafts if d["mode"] == "doc"]
                para_mode_drafts = [d for d in drafts if d["mode"] == "para"]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if doc_mode_drafts:
                        avg_gz = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in doc_mode_drafts])
                        avg_sp = np.mean([d["scores_after"]["group_doc"]["sapling"] for d in doc_mode_drafts])
                        
                        # Count zero-shot successes
                        zs_gz = sum(1 for d in doc_mode_drafts if d["scores_after"]["group_doc"]["gptzero"] <= ZERO_SHOT_THRESHOLD)
                        zs_sp = sum(1 for d in doc_mode_drafts if d["scores_after"]["group_doc"]["sapling"] <= ZERO_SHOT_THRESHOLD)
                        
                        st.markdown("**Document Mode Results:**")
                        colored_metric("Avg GPTZero", f"{avg_gz:.3f}", 
                                     avg_gz - doc['overall']['gptzero'])
                        colored_metric("Avg Sapling", f"{avg_sp:.3f}",
                                     avg_sp - doc['overall']['sapling'])
                        st.metric("Zero-shot Success", f"GZ: {zs_gz}/{len(doc_mode_drafts)} | SP: {zs_sp}/{len(doc_mode_drafts)}")
                
                with col2:
                    if para_mode_drafts:
                        avg_gz = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in para_mode_drafts])
                        avg_sp = np.mean([d["scores_after"]["group_doc"]["sapling"] for d in para_mode_drafts])
                        
                        # Count zero-shot successes
                        zs_gz = sum(1 for d in para_mode_drafts if d["scores_after"]["group_doc"]["gptzero"] <= ZERO_SHOT_THRESHOLD)
                        zs_sp = sum(1 for d in para_mode_drafts if d["scores_after"]["group_doc"]["sapling"] <= ZERO_SHOT_THRESHOLD)
                        
                        st.markdown("**Paragraph Mode Results:**")
                        colored_metric("Avg GPTZero", f"{avg_gz:.3f}",
                                     avg_gz - doc['overall']['gptzero'])
                        colored_metric("Avg Sapling", f"{avg_sp:.3f}",
                                     avg_sp - doc['overall']['sapling'])
                        st.metric("Zero-shot Success", f"GZ: {zs_gz}/{len(para_mode_drafts)} | SP: {zs_sp}/{len(para_mode_drafts)}")
            else:
                st.info("This document was not processed in the selected run")