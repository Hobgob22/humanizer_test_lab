# src/ui.py - v5.0 (Enhanced statistics view)
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

# Helper for colored metrics
def colored_metric(label: str, value: str, delta: float = None, help_text: str = None):
    """Display a metric with custom coloring for AI score differences"""
    if delta is not None:
        # For AI scores, negative is good (green), positive is bad (red)
        if delta < 0:
            delta_color = "off"  # This will show green arrow
            delta_str = f"{delta:+.3f}"
        elif delta > 0:
            delta_color = "inverse"  # This will show red arrow
            delta_str = f"{delta:+.3f}"
        else:
            delta_color = "off"
            delta_str = "0.000"
        st.metric(label, value, delta_str, delta_color=delta_color, help=help_text)
    else:
        st.metric(label, value, help=help_text)

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

def _aggregate_statistics_by_model_mode_folder(docs: List[Dict]) -> Dict[str, Any]:
    """
    Calculate detailed statistics separated by model, mode, and folder.
    Returns nested structure: folder -> model -> mode -> stats
    """
    # Initialize nested structure
    stats: DefaultDict[str, DefaultDict[str, DefaultDict[str, Dict]]] = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # First, collect folder baselines (one per document, not per draft)
    folder_baselines: DefaultDict[str, List[Dict]] = defaultdict(list)
    for doc in docs:
        folder = doc.get("folder", "unknown")
        if doc.get("runs"):
            first_run = doc["runs"][0]
            folder_baselines[folder].append({
                "gptzero": first_run["scores_before"]["group_doc"]["gptzero"],
                "sapling": first_run["scores_before"]["group_doc"]["sapling"],
                "wordcount": first_run["wordcount_before"]
            })
    
    # Calculate average baselines per folder
    folder_avg_baselines = {}
    for folder, baselines in folder_baselines.items():
        folder_avg_baselines[folder] = {
            "gptzero": np.mean([b["gptzero"] for b in baselines]),
            "sapling": np.mean([b["sapling"] for b in baselines]),
            "wordcount": np.mean([b["wordcount"] for b in baselines])
        }
    
    # Collect data
    for doc in docs:
        folder = doc.get("folder", "unknown")
        if not doc.get("runs"):
            continue
        
        for draft in doc["runs"]:
            model = draft["model"]
            mode = draft["mode"]
            
            # Initialize lists if not exists
            if "scores" not in stats[folder][model][mode]:
                stats[folder][model][mode] = {
                    "after_scores": [],
                    "wc_deltas": [],
                    "quality_flags": defaultdict(list),
                    "draft_count": 0,
                    "mismatch_count": 0
                }
            
            s = stats[folder][model][mode]
            
            # Add after scores
            s["after_scores"].append({
                "gptzero": draft["scores_after"]["group_doc"]["gptzero"],
                "sapling": draft["scores_after"]["group_doc"]["sapling"],
                "wordcount": draft["wordcount_after"]
            })
            
            # Word count delta
            s["wc_deltas"].append(draft["wordcount_after"] - draft["wordcount_before"])
            
            # Quality flags
            if not draft["para_mismatch"]:
                for flag in GEMINI_FLAGS:
                    count = draft.get("flag_counts", {}).get(flag, 0)
                    total = draft.get("para_count_before", 1)
                    s["quality_flags"][flag].append((count / total) * 100 if total > 0 else 0)
            
            s["draft_count"] += 1
            if draft["para_mismatch"]:
                s["mismatch_count"] += 1
    
    # Calculate aggregated statistics
    result = {}
    for folder, models in stats.items():
        result[folder] = {}
        folder_baseline = folder_avg_baselines.get(folder, {"gptzero": 0, "sapling": 0})
        
        for model, modes in models.items():
            result[folder][model] = {}
            for mode, data in modes.items():
                # Calculate averages
                after_gz = np.mean([a["gptzero"] for a in data["after_scores"]])
                after_sp = np.mean([a["sapling"] for a in data["after_scores"]])
                
                result[folder][model][mode] = {
                    "baseline": {
                        "gptzero": folder_baseline["gptzero"],
                        "sapling": folder_baseline["sapling"],
                    },
                    "after": {
                        "gptzero": after_gz,
                        "sapling": after_sp,
                    },
                    "deltas": {
                        "gptzero": after_gz - folder_baseline["gptzero"],
                        "sapling": after_sp - folder_baseline["sapling"],
                        "wordcount": np.mean(data["wc_deltas"]) if data["wc_deltas"] else 0
                    },
                    "quality": {
                        flag: np.mean(data["quality_flags"][flag]) if data["quality_flags"][flag] else 0
                        for flag in GEMINI_FLAGS
                    },
                    "draft_count": data["draft_count"],
                    "mismatch_rate": (data["mismatch_count"] / data["draft_count"] * 100) if data["draft_count"] > 0 else 0
                }
    
    return result

def _create_model_comparison_table(stats: Dict[str, Any], folder: str) -> pd.DataFrame:
    """Create a comparison table for all models in a specific folder"""
    rows = []
    
    if folder not in stats:
        return pd.DataFrame()
    
    for model, modes in stats[folder].items():
        for mode in ["doc", "para"]:
            if mode in modes:
                s = modes[mode]
                rows.append({
                    "Model": model,
                    "Mode": mode.title(),
                    "Drafts": s["draft_count"],
                    "Baseline GZ": f"{s['baseline']['gptzero']:.3f}",
                    "After GZ": f"{s['after']['gptzero']:.3f}",
                    "Δ GZ": s['deltas']['gptzero'],
                    "Baseline SP": f"{s['baseline']['sapling']:.3f}",
                    "After SP": f"{s['after']['sapling']:.3f}",
                    "Δ SP": s['deltas']['sapling'],
                    "Avg WC Δ": f"{s['deltas']['wordcount']:+.0f}",
                    "Quality %": f"{np.mean(list(s['quality'].values())):.1f}%",
                    "Mismatch %": f"{s['mismatch_rate']:.1f}%"
                })
    
    df = pd.DataFrame(rows)
    return df

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

    # Calculate detailed statistics
    detailed_stats = _aggregate_statistics_by_model_mode_folder(docs)

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 By Folder & Model", 
        "📈 Model Performance", 
        "📁 Folder Summary", 
        "📊 Distributions",
        "📄 Documents"
    ])
    
    with tab1:
        st.subheader("🎯 Detailed Statistics by Folder, Model, and Mode")
        
        # Group folders by type
        folder_order = ["ai_texts", "human_texts", "mixed_texts"]
        available_folders = [f for f in folder_order if f in detailed_stats]
        other_folders = [f for f in detailed_stats if f not in folder_order]
        all_folders = available_folders + other_folders
        
        for folder in all_folders:
            with st.expander(f"📁 **{folder.replace('_', ' ').title()}**", expanded=(folder == "ai_texts")):
                if folder in detailed_stats:
                    df = _create_model_comparison_table(detailed_stats, folder)
                    if not df.empty:
                        # Style the dataframe with color coding
                        def style_delta(val):
                            if isinstance(val, (int, float)):
                                if val < 0:
                                    return 'color: green'
                                elif val > 0:
                                    return 'color: red'
                            return ''
                        
                        styled_df = df.style.applymap(style_delta, subset=['Δ GZ', 'Δ SP'])
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                        
                        # Visualizations for this folder
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### GPTZero Score Changes")
                            fig, ax = plt.subplots(figsize=(8, 6))
                            
                            models = df['Model'].unique()
                            x = np.arange(len(models))
                            width = 0.35
                            
                            for i, mode in enumerate(['Doc', 'Para']):
                                mode_df = df[df['Mode'] == mode]
                                deltas = []
                                for model in models:
                                    model_data = mode_df[mode_df['Model'] == model]
                                    if not model_data.empty:
                                        deltas.append(model_data['Δ GZ'].iloc[0])
                                    else:
                                        deltas.append(0)
                                
                                bars = ax.bar(x + (i-0.5)*width, deltas, width, 
                                             label=f'{mode} Mode', alpha=0.8)
                                
                                # Color bars based on value
                                for bar, delta in zip(bars, deltas):
                                    if delta < 0:
                                        bar.set_color('green')
                                    else:
                                        bar.set_color('red')
                            
                            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                            ax.set_xlabel('Model')
                            ax.set_ylabel('GPTZero Score Change')
                            ax.set_title(f'GPTZero Score Changes - {folder.replace("_", " ").title()}')
                            ax.set_xticks(x)
                            ax.set_xticklabels(models, rotation=45, ha='right')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        with col2:
                            st.markdown("#### Quality Scores")
                            fig, ax = plt.subplots(figsize=(8, 6))
                            
                            quality_data = []
                            labels = []
                            for _, row in df.iterrows():
                                quality = float(row['Quality %'].rstrip('%'))
                                quality_data.append(quality)
                                labels.append(f"{row['Model']}\n({row['Mode']})")
                            
                            bars = ax.bar(range(len(quality_data)), quality_data, alpha=0.8)
                            
                            # Color bars based on quality
                            for bar, q in zip(bars, quality_data):
                                if q >= 80:
                                    bar.set_color('green')
                                elif q >= 60:
                                    bar.set_color('orange')
                                else:
                                    bar.set_color('red')
                            
                            ax.set_xlabel('Model & Mode')
                            ax.set_ylabel('Average Quality Score (%)')
                            ax.set_title(f'Quality Scores - {folder.replace("_", " ").title()}')
                            ax.set_xticks(range(len(labels)))
                            ax.set_xticklabels(labels, rotation=45, ha='right')
                            ax.set_ylim(0, 100)
                            ax.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                else:
                    st.info("No data for this folder")
    
    with tab2:
        st.subheader("📈 Model Performance Across All Folders")
        
        # Aggregate model performance across folders
        model_perf = defaultdict(lambda: {
            "doc": {"gz_deltas": [], "sp_deltas": [], "quality": [], "count": 0},
            "para": {"gz_deltas": [], "sp_deltas": [], "quality": [], "count": 0}
        })
        
        for folder, models in detailed_stats.items():
            for model, modes in models.items():
                for mode, stats in modes.items():
                    model_perf[model][mode]["gz_deltas"].append(stats["deltas"]["gptzero"])
                    model_perf[model][mode]["sp_deltas"].append(stats["deltas"]["sapling"])
                    model_perf[model][mode]["quality"].append(np.mean(list(stats["quality"].values())))
                    model_perf[model][mode]["count"] += stats["draft_count"]
        
        # Create summary table
        rows = []
        for model, modes in model_perf.items():
            for mode in ["doc", "para"]:
                if modes[mode]["count"] > 0:
                    rows.append({
                        "Model": model,
                        "Mode": mode.title(),
                        "Total Drafts": modes[mode]["count"],
                        "Avg Δ GZ": np.mean(modes[mode]["gz_deltas"]),
                        "Avg Δ SP": np.mean(modes[mode]["sp_deltas"]),
                        "Avg Quality": f"{np.mean(modes[mode]['quality']):.1f}%",
                        "Folders": len(modes[mode]["gz_deltas"])
                    })
        
        perf_df = pd.DataFrame(rows)
        if not perf_df.empty:
            # Style the dataframe
            def style_delta(val):
                if isinstance(val, (int, float)):
                    if val < 0:
                        return 'color: green; font-weight: bold'
                    elif val > 0:
                        return 'color: red; font-weight: bold'
                return ''
            
            styled_perf = perf_df.style.applymap(style_delta, subset=['Avg Δ GZ', 'Avg Δ SP']).format({
                'Avg Δ GZ': '{:.3f}',
                'Avg Δ SP': '{:.3f}'
            })
            st.dataframe(styled_perf, use_container_width=True, hide_index=True)
            
            # Best performers
            st.markdown("### 🏆 Best Performers")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Best GPTZero Reduction")
                best_gz = perf_df.nsmallest(5, 'Avg Δ GZ')[['Model', 'Mode', 'Avg Δ GZ']]
                st.dataframe(best_gz, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### Best Quality Scores")
                perf_df['Quality_numeric'] = perf_df['Avg Quality'].str.rstrip('%').astype(float)
                best_quality = perf_df.nlargest(5, 'Quality_numeric')[['Model', 'Mode', 'Avg Quality']]
                st.dataframe(best_quality, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("📁 Performance Summary by Folder")
        
        folder_summary = []
        for folder, models in detailed_stats.items():
            total_drafts = 0
            all_gz_deltas = []
            all_sp_deltas = []
            all_quality = []
            
            for model, modes in models.items():
                for mode, stats in modes.items():
                    total_drafts += stats["draft_count"]
                    all_gz_deltas.append(stats["deltas"]["gptzero"])
                    all_sp_deltas.append(stats["deltas"]["sapling"])
                    all_quality.append(np.mean(list(stats["quality"].values())))
            
            if total_drafts > 0:
                folder_summary.append({
                    "Folder": folder.replace('_', ' ').title(),
                    "Total Drafts": total_drafts,
                    "Models": len(models),
                    "Avg Δ GZ": np.mean(all_gz_deltas),
                    "Avg Δ SP": np.mean(all_sp_deltas),
                    "Avg Quality": f"{np.mean(all_quality):.1f}%"
                })
        
        folder_df = pd.DataFrame(folder_summary)
        if not folder_df.empty:
            # Style the dataframe
            styled_folder = folder_df.style.applymap(
                lambda x: 'color: green; font-weight: bold' if isinstance(x, (int, float)) and x < 0 else ('color: red; font-weight: bold' if isinstance(x, (int, float)) and x > 0 else ''),
                subset=['Avg Δ GZ', 'Avg Δ SP']
            ).format({
                'Avg Δ GZ': '{:.3f}',
                'Avg Δ SP': '{:.3f}'
            })
            st.dataframe(styled_folder, use_container_width=True, hide_index=True)
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # GPTZero changes by folder
            folders = folder_df['Folder'].tolist()
            gz_deltas = folder_df['Avg Δ GZ'].tolist()
            
            bars1 = ax1.bar(folders, gz_deltas, alpha=0.8)
            for bar, delta in zip(bars1, gz_deltas):
                if delta < 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax1.set_xlabel('Folder')
            ax1.set_ylabel('Average GPTZero Change')
            ax1.set_title('Average GPTZero Score Changes by Folder')
            ax1.grid(True, alpha=0.3)
            
            # Quality by folder
            quality_vals = [float(q.rstrip('%')) for q in folder_df['Avg Quality']]
            bars2 = ax2.bar(folders, quality_vals, alpha=0.8)
            
            for bar, q in zip(bars2, quality_vals):
                if q >= 80:
                    bar.set_color('green')
                elif q >= 60:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            ax2.set_xlabel('Folder')
            ax2.set_ylabel('Average Quality Score (%)')
            ax2.set_title('Average Quality Scores by Folder')
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab4:
        st.subheader("📊 Score Distributions")

        # Build lookup dict
        by_model_mode_folder = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        # Also calculate folder baselines
        folder_baselines = defaultdict(list)
        for doc in docs:
            folder = doc.get("folder", "unknown")
            if doc.get("runs"):
                # Get baseline from first run (all runs have same baseline)
                baseline_gz = doc["runs"][0]["scores_before"]["group_doc"]["gptzero"]
                folder_baselines[folder].append(baseline_gz)
                
                for draft in doc.get("runs", []):
                    s = draft["scores_after"]["group_doc"]["gptzero"]
                    by_model_mode_folder[folder][draft["model"]][draft["mode"]].append(s)

        # Create distribution plots by folder
        for folder in ["ai_texts", "human_texts", "mixed_texts"]:
            if folder in by_model_mode_folder:
                st.markdown(f"### 📁 {folder.replace('_', ' ').title()}")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Get average baseline for this folder
                if folder in folder_baselines:
                    avg_baseline_gz = np.mean(folder_baselines[folder])
                    ax1.axvline(avg_baseline_gz, color="red", linestyle="--", label=f"Baseline (avg: {avg_baseline_gz:.3f})", alpha=0.7)
                    ax2.axvline(avg_baseline_gz, color="red", linestyle="--", label=f"Baseline (avg: {avg_baseline_gz:.3f})", alpha=0.7)
                
                # Doc mode distributions
                for model, modes in by_model_mode_folder[folder].items():
                    if "doc" in modes:
                        _safe_hist(ax1, modes["doc"], bins=20, alpha=0.5, label=model)
                
                ax1.set_title(f"Document Mode - {folder.replace('_', ' ').title()}")
                ax1.set_xlabel("GPTZero Score")
                ax1.set_ylabel("Frequency")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Para mode distributions
                for model, modes in by_model_mode_folder[folder].items():
                    if "para" in modes:
                        _safe_hist(ax2, modes["para"], bins=20, alpha=0.5, label=model)
                
                ax2.set_title(f"Paragraph Mode - {folder.replace('_', ' ').title()}")
                ax2.set_xlabel("GPTZero Score")
                ax2.set_ylabel("Frequency")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                st.divider()
    
    with tab5:
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
    """Render individual draft with improved UI and colored metrics"""
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
    
    # Color code the deltas in the title
    gz_delta = sa['gptzero'] - sb['gptzero']
    sp_delta = sa['sapling'] - sb['sapling']
    
    gz_color = "🟢" if gz_delta < 0 else "🔴" if gz_delta > 0 else "⚪"
    sp_color = "🟢" if sp_delta < 0 else "🔴" if sp_delta > 0 else "⚪"
    
    title = (
        f"{status_emoji} Draft {draft['iter']+1} | "
        f"GZ: {sa['gptzero']:.2f} ({gz_color}{gz_delta:+.2f}) | "
        f"SP: {sa['sapling']:.2f} ({sp_color}{sp_delta:+.2f}) | "
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
            # Quality summary with colored metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Paragraphs", para_total)
            with col2:
                colored_metric("GPTZero Change", f"{gz_delta:+.3f}", gz_delta)
            with col3:
                colored_metric("Sapling Change", f"{sp_delta:+.3f}", sp_delta)
            with col4:
                st.metric("Overall Quality", f"{quality_score:.1f}%")
            
            # Detailed paragraph analysis
            if draft.get("paragraph_details"):
                st.markdown("### 📊 Paragraph-by-Paragraph Analysis")
                
                rows = []
                for p in draft["paragraph_details"]:
                    # Calculate paragraph quality score
                    para_quality = sum(1 for v in p["flags"].values() if v) / len(p["flags"]) * 100
                    
                    # Calculate deltas
                    gz_delta_para = p['ai_after']['gptzero'] - p['ai_before']['gptzero']
                    
                    row = {
                        "¶": p["paragraph"],
                        "WC Δ": f"{p['wc_after'] - p['wc_before']:+d}",
                        "GZ Before": f"{p['ai_before']['gptzero']:.2f}",
                        "GZ After": f"{p['ai_after']['gptzero']:.2f}",
                        "GZ Δ": gz_delta_para,
                        "Quality": f"{para_quality:.0f}%",
                    }
                    
                    # Add individual flag columns
                    for flag in GEMINI_FLAGS:
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
                def color_delta(val):
                    if isinstance(val, (int, float)):
                        if val < 0:
                            return 'color: green; font-weight: bold'
                        elif val > 0:
                            return 'color: red; font-weight: bold'
                    return ''
                
                styled_df = df.style.applymap(
                    lambda x: 'color: green' if isinstance(x, str) and x.startswith('+') else ('color: red' if isinstance(x, str) and x.startswith('-') else ''),
                    subset=['WC Δ']
                ).applymap(
                    color_delta,
                    subset=['GZ Δ']
                ).applymap(
                    lambda x: 'background-color: #90EE90' if x == "✅" else 'background-color: #FFB6C1' if x == "❌" else '',
                    subset=[col for col in df.columns if col in ['Length OK', 'Same Meaning', 'Same Language', 'No Missing Info', 'Citation Preserved', 'Citation Content OK']]
                ).format({
                    'GZ Δ': '{:+.3f}'
                })
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=min(400, 50 + len(rows) * 35)  # Dynamic height based on rows
                )

def _page_document(run_id: str, docs: List[Dict], doc_name: str):
    """Enhanced document detail page with colored metrics"""
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
                
                # Summary stats for this model with colored metrics
                model_drafts = by_model[model]["doc"]
                avg_gz = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in model_drafts])
                avg_sp = np.mean([d["scores_after"]["group_doc"]["sapling"] for d in model_drafts])
                avg_wc_delta = np.mean([d["wordcount_after"] - d["wordcount_before"] for d in model_drafts])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    colored_metric("Avg GPTZero", f"{avg_gz:.3f}", avg_gz - baseline_gz)
                with col2:
                    colored_metric("Avg Sapling", f"{avg_sp:.3f}", avg_sp - baseline_sp)
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
                
                # Summary stats for this model with colored metrics
                model_drafts = by_model[model]["para"]
                avg_gz = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in model_drafts])
                avg_sp = np.mean([d["scores_after"]["group_doc"]["sapling"] for d in model_drafts])
                avg_wc_delta = np.mean([d["wordcount_after"] - d["wordcount_before"] for d in model_drafts])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    colored_metric("Avg GPTZero", f"{avg_gz:.3f}", avg_gz - baseline_gz)
                with col2:
                    colored_metric("Avg Sapling", f"{avg_sp:.3f}", avg_sp - baseline_sp)
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
                        "Δ GZ": avg_gz - baseline_gz,
                        "Avg Sapling": f"{avg_sp:.3f}",
                        "Δ SP": avg_sp - baseline_sp,
                        "Avg WC Δ": f"{avg_wc:+.0f}",
                        "Avg Quality": f"{avg_quality:.1f}%"
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Style with color coding
        def style_delta(val):
            if isinstance(val, (int, float)):
                if val < 0:
                    return 'color: green; font-weight: bold'
                elif val > 0:
                    return 'color: red; font-weight: bold'
            return ''
        
        styled_comparison = comparison_df.style.applymap(
            style_delta, subset=['Δ GZ', 'Δ SP']
        ).format({
            'Δ GZ': '{:+.3f}',
            'Δ SP': '{:+.3f}'
        })
        
        st.dataframe(styled_comparison, use_container_width=True, hide_index=True)
        
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
            
            # Color bars based on performance vs baseline
            doc_bars = ax.bar(x - width/2, doc_scores, width, label='Doc Mode', alpha=0.8)
            para_bars = ax.bar(x + width/2, para_scores, width, label='Para Mode', alpha=0.8)
            
            for bar, score in zip(doc_bars, doc_scores):
                if score < baseline_gz:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            for bar, score in zip(para_bars, para_scores):
                if score < baseline_gz:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
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
                # Summary comparison with colored metrics
                doc_mode_drafts = [d for d in drafts if d["mode"] == "doc"]
                para_mode_drafts = [d for d in drafts if d["mode"] == "para"]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if doc_mode_drafts:
                        avg_gz = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in doc_mode_drafts])
                        avg_sp = np.mean([d["scores_after"]["group_doc"]["sapling"] for d in doc_mode_drafts])
                        
                        st.markdown("**Document Mode Results:**")
                        colored_metric("Avg GPTZero", f"{avg_gz:.3f}", 
                                     avg_gz - doc['overall']['gptzero'])
                        colored_metric("Avg Sapling", f"{avg_sp:.3f}",
                                     avg_sp - doc['overall']['sapling'])
                
                with col2:
                    if para_mode_drafts:
                        avg_gz = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in para_mode_drafts])
                        avg_sp = np.mean([d["scores_after"]["group_doc"]["sapling"] for d in para_mode_drafts])
                        
                        st.markdown("**Paragraph Mode Results:**")
                        colored_metric("Avg GPTZero", f"{avg_gz:.3f}",
                                     avg_gz - doc['overall']['gptzero'])
                        colored_metric("Avg Sapling", f"{avg_sp:.3f}",
                                     avg_sp - doc['overall']['sapling'])
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