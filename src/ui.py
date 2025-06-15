# src/ui.py - v4.0 (Fixed statistics and improved UI)
from __future__ import annotations

import sys
import time
import threading
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

from src.results_db import list_runs, load_run, delete_run, save_run
from src.pipeline import run_test, load_ai_scores
from src.models import MODEL_REGISTRY

# ───────────────────────── page config & constants ───────────────────────
st.set_page_config(page_title="Humanizer Test-Bench", layout="wide", initial_sidebar_state="expanded")

GEMINI_FLAGS = [
    "length_ok",
    "same_meaning",
    "same_lang",
    "no_missing_info",
    "citation_preserved",
    "citation_content_ok",
]

# Color scheme for consistency
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8'
}

# ────────────────────────── query-param helpers ───────────────────────────
def _qp_get(key: str, default=None):
    val = st.query_params.get(key, default)
    if isinstance(val, list):
        return val[0] if val else default
    return val

def _qp_set(**kwargs):
    qp = dict(st.query_params)
    for k, v in kwargs.items():
        if v is None:
            qp.pop(k, None)
        else:
            qp[k] = v
    st.query_params = qp

# ─────────────────────────── live-log helpers ────────────────────────────
_LOG: list[str] = []
_LOG_LOCK = threading.Lock()

def log(msg: str):
    """Append to the live‐log buffer for display in the UI only."""
    timestamped = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _LOG_LOCK:
        _LOG.append(timestamped)
        # keep only the last 4000 lines
        _LOG[:] = _LOG[-4_000:]

def show_log(box):
    with _LOG_LOCK:
        box.text_area("Live log", "\n".join(_LOG[-400:]), height=300, disabled=True)


# Histogram helper --------------------------------------------------------
def _safe_hist(ax, data, *, bins: int = 20, **kwargs):
    """
    Draw a histogram that *never* raises on zero-range input.

    • Falls back to a single bin when all points are identical
    • Limits the number of bins to the unique value count
    """
    if not data:
        return

    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return

    xmin, xmax = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return

    if math.isclose(xmin, xmax):
        ax.hist(arr, bins=1, **kwargs)
    else:
        bin_cnt = min(bins, len(np.unique(arr)))
        ax.hist(arr, bins=bin_cnt, **kwargs)

# ═════════════════════════════ 1 · NEW RUN PAGE ═══════════════════════════
def page_new_run():
    st.header("⚡️ Launch new benchmark")

    run_name = st.text_input("Unique run name", placeholder="Enter a descriptive name for this benchmark run")

    FOLDERS = {
        "AI texts":    "data/ai_texts",
        "Human texts": "data/human_texts",
        "Mixed texts": "data/mixed_texts",
    }
    folder_labels = st.multiselect("Folders", list(FOLDERS), help="Select which document folders to process")
    folder_paths  = [FOLDERS[f] for f in folder_labels]

    all_models   = list(MODEL_REGISTRY)
    model_labels = st.multiselect("Models", all_models, default=all_models[:3], 
                                   help="Select humanizer models to test")

    iterations = st.slider("Iterations per document", 1, 10, value=5,
                          help="Number of times to humanize each document with each model")

    # Show expected workload
    if folder_paths and model_labels:
        doc_count = sum(len(list((ROOT / fp).glob("*.docx"))) for fp in folder_paths)
        total_drafts = doc_count * len(model_labels) * iterations * 2  # x2 for doc + para modes
        st.info(f"📊 Expected workload: {doc_count} documents × {len(model_labels)} models × {iterations} iterations × 2 modes = **{total_drafts} drafts**")

    # Live-log placeholder
    log_box = st.empty()

    if st.button("🚀 Run benchmark", type="primary", disabled=not (run_name and folder_paths and model_labels)):
        if not run_name.strip():
            st.error("Please provide a run name"); st.stop()
        if load_run(run_name):
            st.error("Run name already exists"); st.stop()
        if not folder_paths or not model_labels:
            st.error("Pick at least one folder and one model"); st.stop()

        docs = [p for fp in folder_paths for p in (ROOT / fp).glob("*.docx")]
        if not docs:
            st.error("No .docx files in the selected folders"); st.stop()

        # initial metadata logs
        log(f"🚀 Benchmark '{run_name}': {len(docs)} docs, {len(model_labels)} models, {iterations} iters")
        log(f"🎯 Total drafts: {len(docs) * len(model_labels) * iterations * 2}")
        log(f"🗂  Folders: {', '.join(folder_labels)}")
        log(f"🤖 Models:   {', '.join(model_labels)}")
        show_log(log_box)

        # status spinner with live updates
        with st.status("Running benchmark …", expanded=True) as status:
            start_time = time.time()
            results = []

            for idx, doc_path in enumerate(docs, start=1):
                status.update(label=f"Processing document {idx}/{len(docs)}: {doc_path.name}")
                st.progress(idx / len(docs))
                log(f"\n📄 Starting document {idx}/{len(docs)}: {doc_path.name}")
                show_log(log_box)

                try:
                    result = run_test(doc_path, model_labels, log, iterations)
                    if result.get("runs"):
                        results.append(result)
                        log(f"✅ Completed {doc_path.name}: {len(result['runs'])} drafts")
                    else:
                        log(f"⚠️  Skipped {doc_path.name} (no paragraphs)")
                except Exception as e:
                    log(f"❌ ERROR in {doc_path.name}: {e}")

                show_log(log_box)

            # Save
            duration = (time.time() - start_time) / 60
            log(f"\n💾 Saving run '{run_name}' (took {duration:.1f} min)…")
            save_run(run_name, folder_labels, model_labels, {"docs": results, "iterations": iterations})
            show_log(log_box)
            status.update(label="Benchmark finished!", state="complete", expanded=False)

        # Final summary
        st.success(f"✅ Run '{run_name}' completed in {duration:.1f} minutes")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Documents", len(results))
        with col2:
            st.metric("📝 Total drafts", sum(len(d['runs']) for d in results))
        with col3:
            st.metric("🤖 Models used", len(model_labels))
        with col4:
            avg_sec = (duration * 60) / len(results) if results else 0
            st.metric("⏱️ Avg time/doc", f"{avg_sec:.1f}s")

# ═════════════════════════ utilities: analytics ════════════════════════════
def _iter_drafts(docs: List[Dict]) -> Tuple[Dict, ...]:
    for doc in docs:
        for d in doc.get("runs", []):
            yield doc, d

def _aggregate_global_kpis(docs: List[Dict]) -> Dict[str, Any]:
    """
    Calculate proper statistics:
    - Use original document scores as baseline (scores_before)
    - Compare with humanized document scores (scores_after) 
    - Separate statistics by mode (doc vs para)
    """
    # Original document baselines (same for all drafts)
    baselines = []
    
    # Humanized scores by mode
    doc_mode_scores = []
    para_mode_scores = []
    
    # Quality metrics by mode
    doc_mode_quality = defaultdict(list)
    para_mode_quality = defaultdict(list)
    
    # Word count changes
    wc_deltas_doc = []
    wc_deltas_para = []
    
    for doc in docs:
        if not doc.get("runs"):
            continue
            
        # Get baseline from first run (same for all)
        first_run = doc["runs"][0]
        baseline = {
            "gptzero": first_run["scores_before"]["group_doc"]["gptzero"],
            "sapling": first_run["scores_before"]["group_doc"]["sapling"],
            "wordcount": first_run["wordcount_before"]
        }
        baselines.append(baseline)
        
        # Process each draft
        for draft in doc["runs"]:
            scores = {
                "gptzero": draft["scores_after"]["group_doc"]["gptzero"],
                "sapling": draft["scores_after"]["group_doc"]["sapling"],
                "wordcount": draft["wordcount_after"]
            }
            
            if draft["mode"] == "doc":
                doc_mode_scores.append(scores)
                wc_deltas_doc.append(draft["wordcount_after"] - draft["wordcount_before"])
                
                # Collect quality flags
                for flag in GEMINI_FLAGS:
                    count = draft.get("flag_counts", {}).get(flag, 0)
                    total = draft.get("para_count_before", 1)
                    doc_mode_quality[flag].append((count / total) * 100 if total > 0 else 0)
                    
            else:  # para mode
                para_mode_scores.append(scores)
                wc_deltas_para.append(draft["wordcount_after"] - draft["wordcount_before"])
                
                # Collect quality flags
                for flag in GEMINI_FLAGS:
                    count = draft.get("flag_counts", {}).get(flag, 0)
                    total = draft.get("para_count_before", 1)
                    para_mode_quality[flag].append((count / total) * 100 if total > 0 else 0)
    
    # Calculate aggregated statistics
    def safe_mean(lst):
        return np.mean(lst) if lst else 0
    
    # Baseline averages
    baseline_stats = {
        "gptzero": safe_mean([b["gptzero"] for b in baselines]),
        "sapling": safe_mean([b["sapling"] for b in baselines]),
        "wordcount": safe_mean([b["wordcount"] for b in baselines])
    }
    
    # Doc mode averages
    doc_stats = {
        "gptzero": safe_mean([s["gptzero"] for s in doc_mode_scores]),
        "sapling": safe_mean([s["sapling"] for s in doc_mode_scores]),
        "wordcount": safe_mean([s["wordcount"] for s in doc_mode_scores]),
        "wc_delta": safe_mean(wc_deltas_doc),
        "quality": {flag: safe_mean(doc_mode_quality[flag]) for flag in GEMINI_FLAGS}
    }
    
    # Para mode averages
    para_stats = {
        "gptzero": safe_mean([s["gptzero"] for s in para_mode_scores]),
        "sapling": safe_mean([s["sapling"] for s in para_mode_scores]),
        "wordcount": safe_mean([s["wordcount"] for s in para_mode_scores]),
        "wc_delta": safe_mean(wc_deltas_para),
        "quality": {flag: safe_mean(para_mode_quality[flag]) for flag in GEMINI_FLAGS}
    }
    
    return {
        "baseline": baseline_stats,
        "doc_mode": doc_stats,
        "para_mode": para_stats,
        "all_baselines": baselines,
        "all_doc_scores": doc_mode_scores,
        "all_para_scores": para_mode_scores
    }

def _folder_summary(docs: List[Dict]) -> pd.DataFrame:
    """Create folder-level summary with proper baseline comparison"""
    rows = []
    groups: DefaultDict[str, List[Dict]] = defaultdict(list)
    for doc in docs:
        groups[doc.get("folder", "(unknown)")].append(doc)

    for folder, folder_docs in groups.items():
        # Collect baselines and scores
        baselines = []
        doc_mode_after = []
        para_mode_after = []
        doc_wc_delta = []
        para_wc_delta = []
        mismatches = 0
        total_drafts = 0
        
        for doc in folder_docs:
            if not doc.get("runs"):
                continue
                
            # Baseline (same for all drafts of this doc)
            first_run = doc["runs"][0]
            baseline_score = first_run["scores_before"]["group_doc"]["gptzero"]
            baselines.append(baseline_score)
            
            for draft in doc["runs"]:
                total_drafts += 1
                after_score = draft["scores_after"]["group_doc"]["gptzero"]
                
                if draft["mode"] == "doc":
                    doc_mode_after.append(after_score)
                    doc_wc_delta.append(draft["wordcount_after"] - draft["wordcount_before"])
                else:
                    para_mode_after.append(after_score)
                    para_wc_delta.append(draft["wordcount_after"] - draft["wordcount_before"])
                    
                if draft["para_mismatch"]:
                    mismatches += 1
        
        # Calculate statistics
        avg_baseline = np.mean(baselines) if baselines else 0
        avg_doc_after = np.mean(doc_mode_after) if doc_mode_after else 0
        avg_para_after = np.mean(para_mode_after) if para_mode_after else 0
        
        rows.append({
            "Folder": folder,
            "Docs": len(folder_docs),
            "Baseline AI": f"{avg_baseline:.3f}",
            "Doc mode AI": f"{avg_doc_after:.3f} ({avg_doc_after - avg_baseline:+.3f})",
            "Para mode AI": f"{avg_para_after:.3f} ({avg_para_after - avg_baseline:+.3f})",
            "Avg WC Δ (doc)": f"{np.mean(doc_wc_delta):+.0f}" if doc_wc_delta else "—",
            "Avg WC Δ (para)": f"{np.mean(para_wc_delta):+.0f}" if para_wc_delta else "—",
            "Mismatches": f"{(mismatches/total_drafts)*100:.1f}%" if total_drafts else "—"
        })

    return pd.DataFrame(rows)

# ═══════════════ 2 · RUN OVERVIEW & DOC PAGE ════════════════════════════
def page_runs():
    run_id   = _qp_get("run")
    doc_name = _qp_get("doc")
    view     = _qp_get("view")

    runs_meta = list_runs()
    if not runs_meta:
        st.info("No benchmarks stored yet. Create a new run to get started!")
        return

    # Run selector with metadata
    run_options = []
    for r in runs_meta:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))
        run_options.append(f"{r['name']} ({ts})")
    
    selected_idx = 0
    if run_id:
        for i, r in enumerate(runs_meta):
            if r["name"] == run_id:
                selected_idx = i
                break
    
    selected = st.selectbox("Select benchmark run", run_options, index=selected_idx)
    run_id = runs_meta[run_options.index(selected)]["name"]
    
    run = load_run(run_id) or {}
    docs = run.get("docs", [])
    if not docs:
        st.warning("Selected run is empty.")
        return

    if view == "doc" and doc_name:
        _page_document(run_id, docs, doc_name)
        return

    # Overview page
    st.header(f"📊 Benchmark Analysis: **{run_id}**")
    
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Documents", len(docs))
    with col2:
        st.metric("📝 Total drafts", sum(len(d.get("runs", [])) for d in docs))
    with col3:
        st.metric("🔄 Iterations", run.get("iterations", "N/A"))
    with col4:
        models_used = set()
        for doc in docs:
            for draft in doc.get("runs", []):
                models_used.add(draft["model"])
        st.metric("🤖 Models", len(models_used))

    # Calculate proper KPIs
    kpi = _aggregate_global_kpis(docs)

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Overview", "📊 Distributions", "📁 By Folder", "📄 Documents"])
    
    with tab1:
        st.subheader("🎯 Original Document Baseline")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("GPTZero Score", f"{kpi['baseline']['gptzero']:.3f}")
        with col2:
            st.metric("Sapling Score", f"{kpi['baseline']['sapling']:.3f}")
        with col3:
            st.metric("Avg Word Count", f"{kpi['baseline']['wordcount']:.0f}")
        
        # Mode comparison
        st.subheader("🔄 Humanization Results by Mode")
        
        # Add explanation of modes
        with st.expander("ℹ️ What's the difference between modes?"):
            st.markdown("""
            **Document Mode:** 
            - The entire document is sent to the humanizer as one piece
            - Better at maintaining overall document coherence and flow
            - May handle transitions between paragraphs more naturally
            
            **Paragraph Mode:**
            - Each paragraph is humanized individually
            - Better at preserving document structure (less likely to merge/split paragraphs)
            - May result in less coherent transitions between paragraphs
            - Headings are preserved as-is (not humanized)
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📄 Document Mode")
            st.markdown("*Entire document humanized at once*")
            
            # Detector scores
            gz_delta = kpi['doc_mode']['gptzero'] - kpi['baseline']['gptzero']
            sp_delta = kpi['doc_mode']['sapling'] - kpi['baseline']['sapling']
            
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric("GPTZero", 
                         f"{kpi['doc_mode']['gptzero']:.3f}",
                         f"{gz_delta:+.3f}",
                         delta_color="inverse")
            with col1b:
                st.metric("Sapling", 
                         f"{kpi['doc_mode']['sapling']:.3f}",
                         f"{sp_delta:+.3f}",
                         delta_color="inverse")
            
            st.metric("Word Count Δ", f"{kpi['doc_mode']['wc_delta']:+.0f}")
            
            # Quality metrics
            st.markdown("**Quality Metrics:**")
            
            # Add explanation of metrics
            with st.expander("ℹ️ What do these metrics mean?"):
                st.markdown("""
                - **Length OK**: Word count change is within acceptable range (-30 to +10 words)
                - **Same Meaning**: The humanized text conveys the same meaning as the original
                - **Same Language**: The text remains in the same language
                - **No Missing Info**: All information from the original is preserved
                - **Citation Preserved**: All citations/references are maintained
                - **Citation Content OK**: Citation text matches exactly
                """)
            
            quality_df = pd.DataFrame({
                'Metric': [f.replace('_', ' ').title() for f in GEMINI_FLAGS],
                'Success Rate': [f"{kpi['doc_mode']['quality'][f]:.1f}%" for f in GEMINI_FLAGS]
            })
            st.dataframe(quality_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("### 📝 Paragraph Mode")
            st.markdown("*Each paragraph humanized separately*")
            
            # Detector scores
            gz_delta = kpi['para_mode']['gptzero'] - kpi['baseline']['gptzero']
            sp_delta = kpi['para_mode']['sapling'] - kpi['baseline']['sapling']
            
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("GPTZero", 
                         f"{kpi['para_mode']['gptzero']:.3f}",
                         f"{gz_delta:+.3f}",
                         delta_color="inverse")
            with col2b:
                st.metric("Sapling", 
                         f"{kpi['para_mode']['sapling']:.3f}",
                         f"{sp_delta:+.3f}",
                         delta_color="inverse")
            
            st.metric("Word Count Δ", f"{kpi['para_mode']['wc_delta']:+.0f}")
            
            # Quality metrics
            st.markdown("**Quality Metrics:**")
            quality_df = pd.DataFrame({
                'Metric': [f.replace('_', ' ').title() for f in GEMINI_FLAGS],
                'Success Rate': [f"{kpi['para_mode']['quality'][f]:.1f}%" for f in GEMINI_FLAGS]
            })
            st.dataframe(quality_df, hide_index=True, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Score Distributions")

        # Build lookup dict
        by_model_mode = defaultdict(lambda: {"doc": [], "para": []})
        for _, draft in _iter_drafts(docs):
            s = draft["scores_after"]["group_doc"]["gptzero"]
            by_model_mode[draft["model"]][draft["mode"]].append(s)

        col1, col2 = st.columns(2)

        # ── (1) GPTZero histograms ───────────────────────────────────────
        with col1:
            st.markdown("### GPTZero Score Distribution")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

            ax1.axvline(
                kpi["baseline"]["gptzero"], color="red", linestyle="--",
                label="Baseline", alpha=0.7,
            )
            for model, d in by_model_mode.items():
                _safe_hist(ax1, d["doc"], bins=20, alpha=0.5, label=model)
            ax1.set_title("Document Mode")
            ax1.set_ylabel("Frequency")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.axvline(
                kpi["baseline"]["gptzero"], color="red", linestyle="--",
                label="Baseline", alpha=0.7,
            )
            for model, d in by_model_mode.items():
                _safe_hist(ax2, d["para"], bins=20, alpha=0.5, label=model)
            ax2.set_title("Paragraph Mode")
            ax2.set_xlabel("GPTZero Score")
            ax2.set_ylabel("Frequency")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

        # ── (2) Word-count delta histograms ─────────────────────────────
        with col2:
            st.markdown("### Word Count Changes")
            wc_by_model_mode = defaultdict(lambda: {"doc": [], "para": []})
            for _, draft in _iter_drafts(docs):
                delta = draft["wordcount_after"] - draft["wordcount_before"]
                wc_by_model_mode[draft["model"]][draft["mode"]].append(delta)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

            ax1.axvline(0, color="black", linestyle="-", alpha=0.3)
            for model, d in wc_by_model_mode.items():
                _safe_hist(ax1, d["doc"], bins=20, alpha=0.5, label=model)
            ax1.set_title("Document Mode Word Count Changes")
            ax1.set_ylabel("Frequency")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.axvline(0, color="black", linestyle="-", alpha=0.3)
            for model, d in wc_by_model_mode.items():
                _safe_hist(ax2, d["para"], bins=20, alpha=0.5, label=model)
            ax2.set_title("Paragraph Mode Word Count Changes")
            ax2.set_xlabel("Word Count Delta")
            ax2.set_ylabel("Frequency")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
    
    with tab3:
        st.subheader("📁 Performance by Folder")
        folder_df = _folder_summary(docs)
        st.dataframe(folder_df, use_container_width=True, hide_index=True)
        
        # Visualize folder comparison
        if len(folder_df) > 1:
            st.markdown("### Folder Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            folders = folder_df['Folder'].tolist()
            x = np.arange(len(folders))
            width = 0.35
            
            # Extract numeric values for plotting
            baselines = []
            doc_scores = []
            para_scores = []
            
            for _, row in folder_df.iterrows():
                baselines.append(float(row['Baseline AI']))
                doc_score = row['Doc mode AI'].split(' ')[0]
                para_score = row['Para mode AI'].split(' ')[0]
                doc_scores.append(float(doc_score))
                para_scores.append(float(para_score))
            
            ax.bar(x - width/2, doc_scores, width, label='Doc Mode', alpha=0.8)
            ax.bar(x + width/2, para_scores, width, label='Para Mode', alpha=0.8)
            ax.plot(x, baselines, 'r--', marker='o', label='Baseline')
            
            ax.set_xlabel('Folder')
            ax.set_ylabel('GPTZero Score')
            ax.set_title('AI Detection Scores by Folder and Mode')
            ax.set_xticks(x)
            ax.set_xticklabels(folders)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab4:
        st.subheader("📄 Document List")
        
        # Group documents by folder
        groups: DefaultDict[str, List[str]] = defaultdict(list)
        for d in docs:
            groups[d.get("folder", "(unknown)")].append(d["document"])
        
        for folder in ["ai_texts", "human_texts", "mixed_texts"]:
            if folder in groups:
                with st.expander(f"📁 {folder.replace('_', ' ').title()} ({len(groups[folder])} documents)", 
                               expanded=(folder == "ai_texts")):
                    for i, fn in enumerate(sorted(groups[folder])):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.text(fn)
                        with col2:
                            if st.button("View", key=f"view_{folder}_{i}"):
                                _qp_set(run=run_id, view="doc", doc=fn)
                                st.rerun()

    # Run management
    st.divider()
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Delete Run", type="secondary"):
            if st.checkbox("Confirm deletion"):
                delete_run(run_id)
                st.warning("Run deleted!")
                _qp_set(run=None, view=None, doc=None)
                st.rerun()

# ──────────────────────────────────────────────────────────────────────────
def _render_draft(draft: Dict, para_total: int, doc_name: str, model: str):
    """Render individual draft with improved UI"""
    wc_delta = draft["wordcount_after"] - draft["wordcount_before"]
    sb = draft["scores_before"]["group_doc"]
    sa = draft["scores_after"]["group_doc"]
    mismatch = draft["para_mismatch"]
    
    # Calculate quality score
    quality_score = 0
    if not mismatch and draft.get("flag_counts"):
        for flag in GEMINI_FLAGS:
            quality_score += draft["flag_counts"].get(flag, 0)
        quality_score = (quality_score / (len(GEMINI_FLAGS) * para_total)) * 100
    
    # Create title with status indicators
    status_emoji = "🔴" if mismatch else ("🟢" if quality_score > 80 else "🟡")
    
    title = (
        f"{status_emoji} Draft {draft['iter']+1} | "
        f"GZ: {sa['gptzero']:.2f} ({sa['gptzero']-sb['gptzero']:+.2f}) | "
        f"SP: {sa['sapling']:.2f} ({sa['sapling']-sb['sapling']:+.2f}) | "
        f"WC: {wc_delta:+d} | "
        f"Quality: {quality_score:.0f}%"
    )

    with st.expander(title, expanded=False):
        if mismatch:
            st.error(f"⚠️ Paragraph count mismatch: {draft['para_count_before']} → {draft['para_count_after']}")
        
        # Download button
        fname = f"{doc_name}_{model}_d{draft['iter']+1}_{draft['mode']}.txt"
        st.download_button(
            "📥 Download draft",
            draft["humanized_text"],
            file_name=fname,
            mime="text/plain",
            key=f"dl_{doc_name}_{model}_{draft['mode']}_{draft['iter']}",
        )
        
        if not mismatch:
            # Quality summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Paragraphs", para_total)
            with col2:
                successful_flags = sum(draft.get("flag_counts", {}).values())
                st.metric("Quality Checks Passed", f"{successful_flags}/{para_total * len(GEMINI_FLAGS)}")
            with col3:
                st.metric("Overall Quality", f"{quality_score:.1f}%")
            
            # Debug info for quality checks (can be removed once working)
            if st.checkbox("Show debug info", key=f"debug_{doc_name}_{model}_{draft['mode']}_{draft['iter']}"):
                st.write("**Debug Information:**")
                st.write(f"- flag_counts: {draft.get('flag_counts', {})}")
                st.write(f"- para_mismatch: {draft.get('para_mismatch', False)}")
                st.write(f"- para_count_before: {draft.get('para_count_before', 0)}")
                st.write(f"- para_count_after: {draft.get('para_count_after', 0)}")
                
                if draft.get("paragraph_details"):
                    st.write("\n**Sample paragraph details:**")
                    for i, para in enumerate(draft["paragraph_details"][:2]):  # Show first 2 paragraphs
                        st.write(f"\nParagraph {i+1}:")
                        st.write(f"- Flags: {para.get('flags', {})}")
                        st.write(f"- AI scores before: {para.get('ai_before', {})}")
                        st.write(f"- AI scores after: {para.get('ai_after', {})}")
                        st.write(f"- Word count: {para.get('wc_before', 0)} → {para.get('wc_after', 0)}")
                
                # Check if quality results are being generated
                st.write("\n**Quality check summary:**")
                total_flags = sum(len(p.get('flags', {})) for p in draft.get('paragraph_details', []))
                total_true = sum(sum(1 for v in p.get('flags', {}).values() if v) for p in draft.get('paragraph_details', []))
                st.write(f"- Total flag checks: {total_flags}")
                st.write(f"- Total passed: {total_true}")
                st.write(f"- Success rate: {(total_true/total_flags*100) if total_flags > 0 else 0:.1f}%")
            
            # Detailed paragraph analysis
            if draft.get("paragraph_details"):
                st.markdown("### 📊 Paragraph-by-Paragraph Analysis")
                
                rows = []
                for p in draft["paragraph_details"]:
                    # Calculate paragraph quality score
                    para_quality = sum(1 for v in p["flags"].values() if v) / len(p["flags"]) * 100
                    
                    row = {
                        "¶": p["paragraph"],
                        "WC Δ": f"{p['wc_after'] - p['wc_before']:+d}",
                        "GZ Before": f"{p['ai_before']['gptzero']:.2f}",
                        "GZ After": f"{p['ai_after']['gptzero']:.2f}",
                        "GZ Δ": f"{p['ai_after']['gptzero'] - p['ai_before']['gptzero']:+.2f}",
                        "Quality": f"{para_quality:.0f}%",
                    }
                    
                    # Add individual flag columns
                    for flag in GEMINI_FLAGS:
                        # Use full names for clarity
                        flag_names = {
                            'length_ok': 'Length OK',
                            'same_meaning': 'Same Meaning',
                            'same_lang': 'Same Language',
                            'no_missing_info': 'No Missing Info',
                            'citation_preserved': 'Citation Preserved',
                            'citation_content_ok': 'Citation Content OK'
                        }
                        flag_name = flag_names.get(flag, flag.replace('_', ' ').title())
                        row[flag_name] = "✅" if p["flags"].get(flag, False) else "❌"
                    
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                
                # Apply conditional formatting
                styled_df = df.style.applymap(
                    lambda x: 'color: green' if isinstance(x, str) and x.startswith('+') else ('color: red' if isinstance(x, str) and x.startswith('-') else ''),
                    subset=['WC Δ', 'GZ Δ']
                ).applymap(
                    lambda x: 'background-color: #90EE90' if x == "✅" else 'background-color: #FFB6C1' if x == "❌" else '',
                    subset=[col for col in df.columns if col in flag_names.values()]
                )
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=min(400, 50 + len(rows) * 35)  # Dynamic height based on rows
                )

def _page_document(run_id: str, docs: List[Dict], doc_name: str):
    """Enhanced document detail page"""
    doc = next((d for d in docs if d["document"] == doc_name), None)
    if not doc:
        st.error("Document not found")
        return

    # Header with navigation
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header(f"📄 {doc_name}")
    with col2:
        if st.button("⬅ Back to Overview"):
            _qp_set(view=None, doc=None)
            st.rerun()

    # Document metadata
    para_total = doc["paragraph_count"]
    baseline_wc = next((r['wordcount_before'] for r in doc['runs'] if r['mode']=='doc'), '—')
    baseline_gz = next((r['scores_before']['group_doc']['gptzero'] for r in doc['runs']), 0)
    baseline_sp = next((r['scores_before']['group_doc']['sapling'] for r in doc['runs']), 0)
    
    # Metadata cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📁 Folder", doc.get('folder', 'unknown'))
    with col2:
        st.metric("📝 Paragraphs", para_total)
    with col3:
        st.metric("📊 Word Count", baseline_wc)
    with col4:
        st.metric("🎯 Baseline GZ", f"{baseline_gz:.3f}")
    with col5:
        st.metric("🎯 Baseline SP", f"{baseline_sp:.3f}")

    # Organize drafts by model and mode
    by_model: DefaultDict[str, Dict[str, List[Dict]]] = defaultdict(lambda: {"doc": [], "para": []})
    for dr in doc["runs"]:
        by_model[dr["model"]][dr["mode"]].append(dr)

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Mode", "📝 Paragraph Mode", "📊 Comparison", "📈 Analysis"])
    
    with tab1:
        st.markdown("### Document-Level Humanization")
        st.info("Each draft represents the entire document rewritten at once")
        
        for model in sorted(by_model):
            if by_model[model]["doc"]:
                st.markdown(f"#### 🤖 Model: {model}")
                
                # Summary stats for this model
                model_drafts = by_model[model]["doc"]
                avg_gz = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in model_drafts])
                avg_sp = np.mean([d["scores_after"]["group_doc"]["sapling"] for d in model_drafts])
                avg_wc_delta = np.mean([d["wordcount_after"] - d["wordcount_before"] for d in model_drafts])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg GPTZero", f"{avg_gz:.3f}", f"{avg_gz - baseline_gz:+.3f}")
                with col2:
                    st.metric("Avg Sapling", f"{avg_sp:.3f}", f"{avg_sp - baseline_sp:+.3f}")
                with col3:
                    st.metric("Avg WC Δ", f"{avg_wc_delta:+.0f}")
                
                # Individual drafts
                for dr in sorted(model_drafts, key=lambda x: x["iter"]):
                    _render_draft(dr, para_total, doc_name, model)
                
                st.divider()

    with tab2:
        st.markdown("### Paragraph-Level Humanization")
        st.info("Each paragraph was rewritten independently and then reassembled")
        
        for model in sorted(by_model):
            if by_model[model]["para"]:
                st.markdown(f"#### 🤖 Model: {model}")
                
                # Summary stats for this model
                model_drafts = by_model[model]["para"]
                avg_gz = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in model_drafts])
                avg_sp = np.mean([d["scores_after"]["group_doc"]["sapling"] for d in model_drafts])
                avg_wc_delta = np.mean([d["wordcount_after"] - d["wordcount_before"] for d in model_drafts])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg GPTZero", f"{avg_gz:.3f}", f"{avg_gz - baseline_gz:+.3f}")
                with col2:
                    st.metric("Avg Sapling", f"{avg_sp:.3f}", f"{avg_sp - baseline_sp:+.3f}")
                with col3:
                    st.metric("Avg WC Δ", f"{avg_wc_delta:+.0f}")
                
                # Individual drafts
                for dr in sorted(model_drafts, key=lambda x: x["iter"]):
                    _render_draft(dr, para_total, doc_name, model)
                
                st.divider()

    with tab3:
        st.markdown("### Model Comparison")
        
        # Prepare comparison data
        comparison_data = []
        for model in sorted(by_model):
            for mode in ["doc", "para"]:
                drafts = by_model[model][mode]
                if drafts:
                    avg_gz = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in drafts])
                    avg_sp = np.mean([d["scores_after"]["group_doc"]["sapling"] for d in drafts])
                    avg_wc = np.mean([d["wordcount_after"] - d["wordcount_before"] for d in drafts])
                    
                    # Calculate average quality
                    quality_scores = []
                    for d in drafts:
                        if not d["para_mismatch"] and d.get("flag_counts"):
                            score = sum(d["flag_counts"].values()) / (len(GEMINI_FLAGS) * para_total) * 100
                            quality_scores.append(score)
                    avg_quality = np.mean(quality_scores) if quality_scores else 0
                    
                    comparison_data.append({
                        "Model": model,
                        "Mode": mode.title(),
                        "Avg GPTZero": f"{avg_gz:.3f}",
                        "Δ GZ": f"{avg_gz - baseline_gz:+.3f}",
                        "Avg Sapling": f"{avg_sp:.3f}",
                        "Δ SP": f"{avg_sp - baseline_sp:+.3f}",
                        "Avg WC Δ": f"{avg_wc:+.0f}",
                        "Avg Quality": f"{avg_quality:.1f}%"
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### GPTZero Score by Model & Mode")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            models = sorted(by_model.keys())
            x = np.arange(len(models))
            width = 0.35
            
            doc_scores = []
            para_scores = []
            
            for model in models:
                doc_drafts = by_model[model]["doc"]
                para_drafts = by_model[model]["para"]
                
                doc_score = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in doc_drafts]) if doc_drafts else baseline_gz
                para_score = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in para_drafts]) if para_drafts else baseline_gz
                
                doc_scores.append(doc_score)
                para_scores.append(para_score)
            
            ax.axhline(baseline_gz, color='red', linestyle='--', label='Baseline', alpha=0.7)
            ax.bar(x - width/2, doc_scores, width, label='Doc Mode', alpha=0.8)
            ax.bar(x + width/2, para_scores, width, label='Para Mode', alpha=0.8)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('GPTZero Score')
            ax.set_title('Average GPTZero Scores by Model and Mode')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Quality Score by Model & Mode")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            doc_quality = []
            para_quality = []
            
            for model in models:
                # Doc mode quality
                doc_drafts = by_model[model]["doc"]
                doc_q_scores = []
                for d in doc_drafts:
                    if not d["para_mismatch"] and d.get("flag_counts"):
                        score = sum(d["flag_counts"].values()) / (len(GEMINI_FLAGS) * para_total) * 100
                        doc_q_scores.append(score)
                doc_quality.append(np.mean(doc_q_scores) if doc_q_scores else 0)
                
                # Para mode quality
                para_drafts = by_model[model]["para"]
                para_q_scores = []
                for d in para_drafts:
                    if not d["para_mismatch"] and d.get("flag_counts"):
                        score = sum(d["flag_counts"].values()) / (len(GEMINI_FLAGS) * para_total) * 100
                        para_q_scores.append(score)
                para_quality.append(np.mean(para_q_scores) if para_q_scores else 0)
            
            ax.bar(x - width/2, doc_quality, width, label='Doc Mode', alpha=0.8)
            ax.bar(x + width/2, para_quality, width, label='Para Mode', alpha=0.8)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Quality Score (%)')
            ax.set_title('Average Quality Scores by Model and Mode')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            
            plt.tight_layout()
            st.pyplot(fig)

    with tab4:
        st.markdown("### Detailed Analysis")
        
        # Score progression over iterations
        st.markdown("#### Score Progression Across Iterations")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        for model in sorted(by_model):
            # Doc mode progression
            doc_drafts = sorted(by_model[model]["doc"], key=lambda x: x["iter"])
            if doc_drafts:
                iterations = [d["iter"] + 1 for d in doc_drafts]
                gz_scores = [d["scores_after"]["group_doc"]["gptzero"] for d in doc_drafts]
                ax1.plot(iterations, gz_scores, marker='o', label=model)
        
        ax1.axhline(baseline_gz, color='red', linestyle='--', label='Baseline', alpha=0.7)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('GPTZero Score')
        ax1.set_title('Document Mode - Score Progression')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        for model in sorted(by_model):
            # Para mode progression
            para_drafts = sorted(by_model[model]["para"], key=lambda x: x["iter"])
            if para_drafts:
                iterations = [d["iter"] + 1 for d in para_drafts]
                gz_scores = [d["scores_after"]["group_doc"]["gptzero"] for d in para_drafts]
                ax2.plot(iterations, gz_scores, marker='o', label=model)
        
        ax2.axhline(baseline_gz, color='red', linestyle='--', label='Baseline', alpha=0.7)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('GPTZero Score')
        ax2.set_title('Paragraph Mode - Score Progression')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Quality metrics breakdown
        st.markdown("#### Quality Metrics Breakdown")
        
        quality_breakdown = defaultdict(lambda: defaultdict(list))
        
        for model in sorted(by_model):
            for mode in ["doc", "para"]:
                for draft in by_model[model][mode]:
                    if not draft["para_mismatch"] and draft.get("flag_counts"):
                        for flag in GEMINI_FLAGS:
                            success_rate = (draft["flag_counts"].get(flag, 0) / para_total) * 100
                            quality_breakdown[f"{model} ({mode})"][flag].append(success_rate)
        
        # Create quality heatmap data
        heatmap_data = []
        model_mode_labels = []
        
        for model_mode, flags in quality_breakdown.items():
            model_mode_labels.append(model_mode)
            row = [np.mean(flags.get(flag, [0])) for flag in GEMINI_FLAGS]
            heatmap_data.append(row)
        
        if heatmap_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Convert to numpy array for better handling
            heatmap_array = np.array(heatmap_data)
            
            # Create heatmap
            im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            
            ax.set_xticks(np.arange(len(GEMINI_FLAGS)))
            ax.set_yticks(np.arange(len(model_mode_labels)))
            ax.set_xticklabels([f.replace('_', ' ').title() for f in GEMINI_FLAGS], rotation=45, ha='right')
            ax.set_yticklabels(model_mode_labels)
            
            # Add text annotations
            for i in range(len(model_mode_labels)):
                for j in range(len(GEMINI_FLAGS)):
                    value = heatmap_array[i, j]
                    text_color = "white" if value < 50 else "black"
                    text = ax.text(j, i, f'{value:.0f}%',
                                   ha="center", va="center", color=text_color, fontsize=9)
            
            ax.set_title("Quality Metrics Success Rate Heatmap")
            cbar = fig.colorbar(im, ax=ax, label='Success Rate (%)')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No quality data available for heatmap visualization")

# ═════════════════════ 3 · DOCUMENTS BROWSER PAGE ═════════════════════════
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
    
    for doc_path in docs:
        _display_single_doc(doc_path, compare_run if compare_run != "None" else None)

def _display_single_doc(path: Path, compare_run: str | None):
    """Enhanced single document display"""
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
            st.metric("GPTZero Score", f"{doc['overall']['gptzero']:.3f}")
        with col2:
            st.metric("Sapling Score", f"{doc['overall']['sapling']:.3f}")
        
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
                # Summary comparison
                doc_mode_drafts = [d for d in drafts if d["mode"] == "doc"]
                para_mode_drafts = [d for d in drafts if d["mode"] == "para"]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if doc_mode_drafts:
                        avg_gz = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in doc_mode_drafts])
                        avg_sp = np.mean([d["scores_after"]["group_doc"]["sapling"] for d in doc_mode_drafts])
                        
                        st.markdown("**Document Mode Results:**")
                        st.metric("Avg GPTZero", f"{avg_gz:.3f}", 
                                 f"{avg_gz - doc['overall']['gptzero']:+.3f}")
                        st.metric("Avg Sapling", f"{avg_sp:.3f}",
                                 f"{avg_sp - doc['overall']['sapling']:+.3f}")
                
                with col2:
                    if para_mode_drafts:
                        avg_gz = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in para_mode_drafts])
                        avg_sp = np.mean([d["scores_after"]["group_doc"]["sapling"] for d in para_mode_drafts])
                        
                        st.markdown("**Paragraph Mode Results:**")
                        st.metric("Avg GPTZero", f"{avg_gz:.3f}",
                                 f"{avg_gz - doc['overall']['gptzero']:+.3f}")
                        st.metric("Avg Sapling", f"{avg_sp:.3f}",
                                 f"{avg_sp - doc['overall']['sapling']:+.3f}")
            else:
                st.info("This document was not processed in the selected run")

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
    
    **Label truncation in tables:**
    - Resize browser window or zoom out
    - Use fullscreen mode
    """)

st.sidebar.caption(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if PAGE == "New Run":
    page_new_run()
elif PAGE == "Benchmark Analysis":
    page_runs()
else:
    page_browser()