# src/pages/benchmark_analysis.py - Analysis and reporting
from __future__ import annotations

import time
from typing import Dict, List, Any, Tuple, DefaultDict
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.pages.utils import (
    qp_get, qp_set, colored_metric, render_draft, safe_hist,
    GEMINI_FLAGS, ZERO_SHOT_THRESHOLD
)

from src.results_db import list_runs, load_run, delete_run

# ─────────────────── project root ────────────────────
ROOT = Path(__file__).resolve().parents[2]

# ═════════════════════ utilities: analytics ════════════════════════════
def _iter_drafts(docs: List[Dict]) -> Tuple[Dict, ...]:
    for doc in docs:
        for d in doc.get("runs", []):
            yield doc, d

def _aggregate_statistics_by_model_mode_folder(docs: List[Dict]) -> Dict[str, Any]:
    """
    Calculate detailed statistics separated by model, mode, and folder.
    Returns nested structure: folder -> model -> mode -> stats
    Now includes zero-shot success rates for both detectors.
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
                    "mismatch_count": 0,
                    "zero_shot_gz": 0,  # Count of drafts with GPTZero <= 10%
                    "zero_shot_sp": 0,  # Count of drafts with Sapling <= 10%
                }
            
            s = stats[folder][model][mode]
            
            # Add after scores
            gz_after = draft["scores_after"]["group_doc"]["gptzero"]
            sp_after = draft["scores_after"]["group_doc"]["sapling"]
            
            s["after_scores"].append({
                "gptzero": gz_after,
                "sapling": sp_after,
                "wordcount": draft["wordcount_after"]
            })
            
            # Count zero-shot successes
            if gz_after <= ZERO_SHOT_THRESHOLD:
                s["zero_shot_gz"] += 1
            if sp_after <= ZERO_SHOT_THRESHOLD:
                s["zero_shot_sp"] += 1
            
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
                    "mismatch_rate": (data["mismatch_count"] / data["draft_count"] * 100) if data["draft_count"] > 0 else 0,
                    "zero_shot_success": {
                        "gptzero": (data["zero_shot_gz"] / data["draft_count"] * 100) if data["draft_count"] > 0 else 0,
                        "sapling": (data["zero_shot_sp"] / data["draft_count"] * 100) if data["draft_count"] > 0 else 0,
                    }
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
                    "Zero-shot GZ": f"{s['zero_shot_success']['gptzero']:.1f}%",
                    "Baseline SP": f"{s['baseline']['sapling']:.3f}",
                    "After SP": f"{s['after']['sapling']:.3f}",
                    "Δ SP": s['deltas']['sapling'],
                    "Zero-shot SP": f"{s['zero_shot_success']['sapling']:.1f}%",
                    "Avg WC Δ": f"{s['deltas']['wordcount']:+.0f}",
                    "Quality %": f"{np.mean(list(s['quality'].values())):.1f}%",
                    "Mismatch %": f"{s['mismatch_rate']:.1f}%"
                })
    
    df = pd.DataFrame(rows)
    return df

# ═══════════════ RUN OVERVIEW & DOC PAGE ════════════════════════════
def page_runs():
    run_id   = qp_get("run")
    doc_name = qp_get("doc")
    view     = qp_get("view")

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
        st.metric("📄 Documents", len(docs), help="Total number of documents processed in this benchmark")
    with col2:
        st.metric("📝 Total drafts", sum(len(d.get("runs", [])) for d in docs), 
                  help="Total humanized drafts generated (documents × models × iterations × 2 modes)")
    with col3:
        st.metric("🔄 Iterations", run.get("iterations", "N/A"),
                  help="Number of times each document was humanized with each model")
    with col4:
        models_used = set()
        for doc in docs:
            for draft in doc.get("runs", []):
                models_used.add(draft["model"])
        st.metric("🤖 Models", len(models_used), help="Number of different humanizer models tested")

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
        
        with st.expander("ℹ️ Understanding the metrics", expanded=False):
            st.markdown("""
            **Key Metrics:**
            - **Δ GZ/SP**: Change in AI detection score (negative = better)
            - **Zero-shot**: % of drafts scoring ≤10% on AI detection
            - **Quality %**: Average of all quality checks (meaning, citations, etc.)
            - **Mismatch %**: Drafts where paragraph count changed
            """)
        
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
                        
                        def style_zero_shot(val):
                            if isinstance(val, str) and val.endswith('%'):
                                num_val = float(val.rstrip('%'))
                                if num_val >= 80:
                                    return 'color: green; font-weight: bold'
                                elif num_val >= 50:
                                    return 'color: orange'
                                else:
                                    return 'color: red'
                            return ''
                        
                        styled_df = df.style.applymap(
                            style_delta, subset=['Δ GZ', 'Δ SP']
                        ).applymap(
                            style_zero_shot, subset=['Zero-shot GZ', 'Zero-shot SP']
                        )
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                        
                        # Visualizations for this folder
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### AI Detection Score Changes")
                            st.caption("Shows how much the AI detection scores changed after humanization")
                            
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
                            
                            models = df['Model'].unique()
                            x = np.arange(len(models))
                            width = 0.35
                            
                            # GPTZero changes
                            for i, mode in enumerate(['Doc', 'Para']):
                                mode_df = df[df['Mode'] == mode]
                                deltas = [ mode_df[mode_df['Model']==m]['Δ GZ'].iloc[0]
                                        if not mode_df[mode_df['Model']==m].empty else 0
                                        for m in models ]
                                
                                is_para = (mode == 'Para')
                                bars = ax1.bar(
                                    x + (i-0.5)*width, deltas, width,
                                    label=f'{mode} Mode',
                                    facecolor='none' if is_para else None,
                                    edgecolor='black' if is_para else None,
                                    hatch='-' if is_para else None,
                                    alpha=0.3 if is_para else 0.8,
                                    linewidth=1 if is_para else 0
                                )
                                
                                # fill color for Doc mode, and for Para use black‐edge hatch over colored face
                                for bar, delta in zip(bars, deltas):
                                    if is_para:
                                        bar.set_facecolor('green' if delta < 0 else 'red')
                                    else:
                                        bar.set_color('green' if delta < 0 else 'red')
                            
                            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                            ax1.set_xlabel('Model')
                            ax1.set_ylabel('GPTZero Score Change')
                            ax1.set_title(f'GPTZero Score Changes - {folder.replace("_", " ").title()}')
                            ax1.set_xticks(x)
                            ax1.set_xticklabels(models, rotation=45, ha='right')
                            ax1.legend()
                            ax1.grid(True, alpha=0.3)
                            
                            # Sapling changes
                            for i, mode in enumerate(['Doc', 'Para']):
                                mode_df = df[df['Mode'] == mode]
                                deltas = [ mode_df[mode_df['Model']==m]['Δ SP'].iloc[0]
                                        if not mode_df[mode_df['Model']==m].empty else 0
                                        for m in models ]
                                
                                is_para = (mode == 'Para')
                                bars = ax2.bar(
                                    x + (i-0.5)*width, deltas, width,
                                    label=f'{mode} Mode',
                                    facecolor='none' if is_para else None,
                                    edgecolor='black' if is_para else None,
                                    hatch='-' if is_para else None,
                                    alpha=0.3 if is_para else 0.8,
                                    linewidth=1 if is_para else 0
                                )
                                
                                for bar, delta in zip(bars, deltas):
                                    if is_para:
                                        bar.set_facecolor('green' if delta < 0 else 'red')
                                    else:
                                        bar.set_color('green' if delta < 0 else 'red')
                            
                            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                            ax2.set_xlabel('Model')
                            ax2.set_ylabel('Sapling Score Change')
                            ax2.set_title(f'Sapling Score Changes - {folder.replace("_", " ").title()}')
                            ax2.set_xticks(x)
                            ax2.set_xticklabels(models, rotation=45, ha='right')
                            ax2.legend()
                            ax2.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)

                        
                        with col2:
                            st.markdown("#### Zero-shot Success & Quality")
                            st.caption("Zero-shot: % of drafts with ≤10% AI score | Quality: content preservation")
                            
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
                            
                            # Zero-shot success rates
                            bar_width = 0.2
                            x = np.arange(len(df['Model'].unique()))
                            
                            for i, (mode, detector) in enumerate([('Doc', 'GZ'), ('Doc', 'SP'), ('Para', 'GZ'), ('Para', 'SP')]):
                                mode_df = df[df['Mode'] == mode.title()]
                                col_name = f'Zero-shot {detector}'
                                values = []
                                for model in models:
                                    model_data = mode_df[mode_df['Model'] == model]
                                    if not model_data.empty:
                                        val = float(model_data[col_name].iloc[0].rstrip('%'))
                                        values.append(val)
                                    else:
                                        values.append(0)
                                
                                ax1.bar(x + (i-1.5)*bar_width, values, bar_width, 
                                       label=f'{mode} {detector}', alpha=0.8)
                            
                            ax1.set_xlabel('Model')
                            ax1.set_ylabel('Zero-shot Success Rate (%)')
                            ax1.set_title(f'Zero-shot Success Rates - {folder.replace("_", " ").title()}')
                            ax1.set_xticks(x)
                            ax1.set_xticklabels(models, rotation=45, ha='right')
                            ax1.legend()
                            ax1.set_ylim(0, 100)
                            ax1.grid(True, alpha=0.3)
                            
                            # Quality scores
                            quality_data = []
                            labels = []
                            for _, row in df.iterrows():
                                quality = float(row['Quality %'].rstrip('%'))
                                quality_data.append(quality)
                                labels.append(f"{row['Model']}\n({row['Mode']})")
                            
                            bars = ax2.bar(range(len(quality_data)), quality_data, alpha=0.8)
                            
                            # Color bars based on quality
                            for bar, q in zip(bars, quality_data):
                                if q >= 80:
                                    bar.set_color('green')
                                elif q >= 60:
                                    bar.set_color('orange')
                                else:
                                    bar.set_color('red')
                            
                            ax2.set_xlabel('Model & Mode')
                            ax2.set_ylabel('Average Quality Score (%)')
                            ax2.set_title(f'Quality Scores - {folder.replace("_", " ").title()}')
                            ax2.set_xticks(range(len(labels)))
                            ax2.set_xticklabels(labels, rotation=45, ha='right')
                            ax2.set_ylim(0, 100)
                            ax2.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                else:
                    st.info("No data for this folder")
    
    with tab2:
        st.subheader("📈 Model Performance Across All Folders")
        
        with st.expander("ℹ️ About this view", expanded=False):
            st.markdown("""
            This view aggregates model performance across all document folders,
            helping identify which models perform best overall. Lower AI detection
            scores and higher zero-shot success rates indicate better performance.
            """)
        
        # Aggregate model performance across folders
        model_perf = defaultdict(lambda: {
            "doc": {"gz_deltas": [], "sp_deltas": [], "quality": [], "count": 0, 
                    "zero_shot_gz": [], "zero_shot_sp": []},
            "para": {"gz_deltas": [], "sp_deltas": [], "quality": [], "count": 0,
                     "zero_shot_gz": [], "zero_shot_sp": []}
        })
        
        for folder, models in detailed_stats.items():
            for model, modes in models.items():
                for mode, stats in modes.items():
                    model_perf[model][mode]["gz_deltas"].append(stats["deltas"]["gptzero"])
                    model_perf[model][mode]["sp_deltas"].append(stats["deltas"]["sapling"])
                    model_perf[model][mode]["quality"].append(np.mean(list(stats["quality"].values())))
                    model_perf[model][mode]["count"] += stats["draft_count"]
                    model_perf[model][mode]["zero_shot_gz"].append(stats["zero_shot_success"]["gptzero"])
                    model_perf[model][mode]["zero_shot_sp"].append(stats["zero_shot_success"]["sapling"])
        
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
                        "Zero-shot GZ": f"{np.mean(modes[mode]['zero_shot_gz']):.1f}%",
                        "Zero-shot SP": f"{np.mean(modes[mode]['zero_shot_sp']):.1f}%",
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
            
            def style_zero_shot(val):
                if isinstance(val, str) and val.endswith('%'):
                    num_val = float(val.rstrip('%'))
                    if num_val >= 80:
                        return 'color: green; font-weight: bold'
                    elif num_val >= 50:
                        return 'color: orange'
                    else:
                        return 'color: red'
                return ''
            
            styled_perf = perf_df.style.applymap(
                style_delta, subset=['Avg Δ GZ', 'Avg Δ SP']
            ).applymap(
                style_zero_shot, subset=['Zero-shot GZ', 'Zero-shot SP']
            ).format({
                'Avg Δ GZ': '{:.3f}',
                'Avg Δ SP': '{:.3f}'
            })
            st.dataframe(styled_perf, use_container_width=True, hide_index=True)
            
            # Best performers
            st.markdown("### 🏆 Best Performers")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Best AI Score Reduction")
                st.caption("Models achieving the largest decrease in AI detection scores")
                
                # GPTZero
                best_gz = perf_df.nsmallest(5, 'Avg Δ GZ')[['Model', 'Mode', 'Avg Δ GZ']]
                st.markdown("**GPTZero:**")
                st.dataframe(best_gz, use_container_width=True, hide_index=True)
                
                # Sapling
                best_sp = perf_df.nsmallest(5, 'Avg Δ SP')[['Model', 'Mode', 'Avg Δ SP']]
                st.markdown("**Sapling:**")
                st.dataframe(best_sp, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### Best Zero-shot Success")
                st.caption("Models with highest % of drafts scoring ≤10% AI detection")
                
                # GPTZero zero-shot
                perf_df['ZS_GZ_numeric'] = perf_df['Zero-shot GZ'].str.rstrip('%').astype(float)
                best_zs_gz = perf_df.nlargest(5, 'ZS_GZ_numeric')[['Model', 'Mode', 'Zero-shot GZ']]
                st.markdown("**GPTZero Zero-shot:**")
                st.dataframe(best_zs_gz, use_container_width=True, hide_index=True)
                
                # Sapling zero-shot
                perf_df['ZS_SP_numeric'] = perf_df['Zero-shot SP'].str.rstrip('%').astype(float)
                best_zs_sp = perf_df.nlargest(5, 'ZS_SP_numeric')[['Model', 'Mode', 'Zero-shot SP']]
                st.markdown("**Sapling Zero-shot:**")
                st.dataframe(best_zs_sp, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("📁 Performance Summary by Folder")
        
        with st.expander("ℹ️ About folder types", expanded=False):
            st.markdown("""
            - **AI texts**: Documents originally generated by AI
            - **Human texts**: Documents originally written by humans
            - **Mixed texts**: Documents with both AI and human content
            
            Performance varies by folder type - AI texts typically show larger improvements.
            """)
        
        folder_summary = []
        for folder, models in detailed_stats.items():
            total_drafts = 0
            all_gz_deltas = []
            all_sp_deltas = []
            all_quality = []
            all_zero_shot_gz = []
            all_zero_shot_sp = []
            
            for model, modes in models.items():
                for mode, stats in modes.items():
                    total_drafts += stats["draft_count"]
                    all_gz_deltas.append(stats["deltas"]["gptzero"])
                    all_sp_deltas.append(stats["deltas"]["sapling"])
                    all_quality.append(np.mean(list(stats["quality"].values())))
                    all_zero_shot_gz.append(stats["zero_shot_success"]["gptzero"])
                    all_zero_shot_sp.append(stats["zero_shot_success"]["sapling"])
            
            if total_drafts > 0:
                folder_summary.append({
                    "Folder": folder.replace('_', ' ').title(),
                    "Total Drafts": total_drafts,
                    "Models": len(models),
                    "Avg Δ GZ": np.mean(all_gz_deltas),
                    "Avg Δ SP": np.mean(all_sp_deltas),
                    "Zero-shot GZ": f"{np.mean(all_zero_shot_gz):.1f}%",
                    "Zero-shot SP": f"{np.mean(all_zero_shot_sp):.1f}%",
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
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            folders = folder_df['Folder'].tolist()
            
            # AI Score changes by folder
            gz_deltas = folder_df['Avg Δ GZ'].tolist()
            sp_deltas = folder_df['Avg Δ SP'].tolist()

            x = np.arange(len(folders))
            width = 0.35

            bars1 = ax1.bar(
                x - width/2, gz_deltas, width,
                label='GPTZero',
                color='green',
                alpha=0.8
            )

            bars2 = ax1.bar(
                x + width/2, sp_deltas, width,
                label='Sapling',
                hatch='-',
                edgecolor='black',
                alpha=0.8
            )

            for bar, delta in zip(bars2, sp_deltas):
                bar.set_facecolor('green' if delta < 0 else 'red')

            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax1.set_xlabel('Folder')
            ax1.set_ylabel('Average Score Change')
            ax1.set_title('Average AI Detection Score Changes by Folder')
            ax1.set_xticks(x)
            ax1.set_xticklabels(folders)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            
            # Zero-shot success by folder
            zs_gz_vals = [float(q.rstrip('%')) for q in folder_df['Zero-shot GZ']]
            zs_sp_vals = [float(q.rstrip('%')) for q in folder_df['Zero-shot SP']]
            
            bars3 = ax2.bar(x - width/2, zs_gz_vals, width, label='GPTZero', alpha=0.8)
            bars4 = ax2.bar(x + width/2, zs_sp_vals, width, label='Sapling', alpha=0.8)
            
            ax2.set_xlabel('Folder')
            ax2.set_ylabel('Zero-shot Success Rate (%)')
            ax2.set_title('Zero-shot Success Rates by Folder')
            ax2.set_xticks(x)
            ax2.set_xticklabels(folders)
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Quality by folder
            quality_vals = [float(q.rstrip('%')) for q in folder_df['Avg Quality']]
            bars5 = ax3.bar(folders, quality_vals, alpha=0.8)
            
            for bar, q in zip(bars5, quality_vals):
                if q >= 80:
                    bar.set_color('green')
                elif q >= 60:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            ax3.set_xlabel('Folder')
            ax3.set_ylabel('Average Quality Score (%)')
            ax3.set_title('Average Quality Scores by Folder')
            ax3.set_ylim(0, 100)
            ax3.grid(True, alpha=0.3)
            
            # Combined metric radar chart
            categories = ['GZ Reduction', 'SP Reduction', 'ZS GZ', 'ZS SP', 'Quality']
            
            for i, folder_row in folder_df.iterrows():
                # Normalize values for radar (0-1 scale)
                values = [
                    max(0, -folder_row['Avg Δ GZ'] / 0.5),  # Normalize reduction (0.5 = max expected)
                    max(0, -folder_row['Avg Δ SP'] / 0.5),
                    zs_gz_vals[i] / 100,
                    zs_sp_vals[i] / 100,
                    quality_vals[i] / 100
                ]
                values += values[:1]  # Complete the circle
                
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
                angles = np.concatenate([angles, [angles[0]]])
                
                ax4.plot(angles, values, 'o-', linewidth=2, label=folder_row['Folder'])
                ax4.fill(angles, values, alpha=0.25)
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(categories)
            ax4.set_ylim(0, 1)
            ax4.set_title('Normalized Performance Metrics by Folder')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab4:
        st.subheader("📊 Score Distributions")
        
        with st.expander("ℹ️ Understanding distributions", expanded=False):
            st.markdown("""
            These histograms show the spread of AI detection scores after humanization.
            - **Red line**: Average baseline score (before humanization)
            - **Bars**: Distribution of scores after humanization
            - **Left is better**: Lower scores indicate better humanization
            """)

        # Build lookup dict
        by_model_mode_folder = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"gz": [], "sp": []})))
        
        # Also calculate folder baselines
        folder_baselines = defaultdict(lambda: {"gz": [], "sp": []})
        for doc in docs:
            folder = doc.get("folder", "unknown")
            if doc.get("runs"):
                # Get baseline from first run (all runs have same baseline)
                baseline_gz = doc["runs"][0]["scores_before"]["group_doc"]["gptzero"]
                baseline_sp = doc["runs"][0]["scores_before"]["group_doc"]["sapling"]
                folder_baselines[folder]["gz"].append(baseline_gz)
                folder_baselines[folder]["sp"].append(baseline_sp)
                
                for draft in doc.get("runs", []):
                    gz_score = draft["scores_after"]["group_doc"]["gptzero"]
                    sp_score = draft["scores_after"]["group_doc"]["sapling"]
                    by_model_mode_folder[folder][draft["model"]][draft["mode"]]["gz"].append(gz_score)
                    by_model_mode_folder[folder][draft["model"]][draft["mode"]]["sp"].append(sp_score)

        # Create distribution plots by folder
        for folder in ["ai_texts", "human_texts", "mixed_texts"]:
            if folder in by_model_mode_folder:
                st.markdown(f"### 📁 {folder.replace('_', ' ').title()}")
                
                # GPTZero distributions
                st.markdown("#### GPTZero Score Distributions")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Get average baseline for this folder
                if folder in folder_baselines:
                    avg_baseline_gz = np.mean(folder_baselines[folder]["gz"])
                    ax1.axvline(avg_baseline_gz, color="red", linestyle="--", 
                               label=f"Baseline (avg: {avg_baseline_gz:.3f})", alpha=0.7)
                    ax2.axvline(avg_baseline_gz, color="red", linestyle="--", 
                               label=f"Baseline (avg: {avg_baseline_gz:.3f})", alpha=0.7)
                
                # Doc mode distributions
                for model, modes in by_model_mode_folder[folder].items():
                    if "doc" in modes and modes["doc"]["gz"]:
                        safe_hist(ax1, modes["doc"]["gz"], bins=20, alpha=0.5, label=model)
                
                ax1.set_title(f"Document Mode - {folder.replace('_', ' ').title()}")
                ax1.set_xlabel("GPTZero Score")
                ax1.set_ylabel("Frequency")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Para mode distributions
                for model, modes in by_model_mode_folder[folder].items():
                    if "para" in modes and modes["para"]["gz"]:
                        safe_hist(ax2, modes["para"]["gz"], bins=20, alpha=0.5, label=model)
                
                ax2.set_title(f"Paragraph Mode - {folder.replace('_', ' ').title()}")
                ax2.set_xlabel("GPTZero Score")
                ax2.set_ylabel("Frequency")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Sapling distributions
                st.markdown("#### Sapling Score Distributions")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Get average baseline for this folder
                if folder in folder_baselines:
                    avg_baseline_sp = np.mean(folder_baselines[folder]["sp"])
                    ax1.axvline(avg_baseline_sp, color="red", linestyle="--", 
                               label=f"Baseline (avg: {avg_baseline_sp:.3f})", alpha=0.7)
                    ax2.axvline(avg_baseline_sp, color="red", linestyle="--", 
                               label=f"Baseline (avg: {avg_baseline_sp:.3f})", alpha=0.7)
                
                # Doc mode distributions
                for model, modes in by_model_mode_folder[folder].items():
                    if "doc" in modes and modes["doc"]["sp"]:
                        safe_hist(ax1, modes["doc"]["sp"], bins=20, alpha=0.5, label=model)
                
                ax1.set_title(f"Document Mode - {folder.replace('_', ' ').title()}")
                ax1.set_xlabel("Sapling Score")
                ax1.set_ylabel("Frequency")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Para mode distributions
                for model, modes in by_model_mode_folder[folder].items():
                    if "para" in modes and modes["para"]["sp"]:
                        safe_hist(ax2, modes["para"]["sp"], bins=20, alpha=0.5, label=model)
                
                ax2.set_title(f"Paragraph Mode - {folder.replace('_', ' ').title()}")
                ax2.set_xlabel("Sapling Score")
                ax2.set_ylabel("Frequency")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                st.divider()
    
    with tab5:
        st.subheader("📄 Document List")
        
        with st.expander("ℹ️ About documents", expanded=False):
            st.markdown("""
            Click "View" to see detailed results for any document, including:
            - All humanized drafts
            - Paragraph-by-paragraph analysis
            - Quality check results
            - Score comparisons
            """)
        
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
                                qp_set(run=run_id, view="doc", doc=fn)
                                st.rerun()

    # Run management
    st.divider()
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Delete Run", type="secondary"):
            if st.checkbox("Confirm deletion"):
                delete_run(run_id)
                st.warning("Run deleted!")
                qp_set(run=None, view=None, doc=None)
                st.rerun()

# ──────────────────────────────────────────────────────────────────────────
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
            qp_set(view=None, doc=None)
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
                zero_shot_gz = sum(1 for d in model_drafts if d["scores_after"]["group_doc"]["gptzero"] <= ZERO_SHOT_THRESHOLD)
                zero_shot_sp = sum(1 for d in model_drafts if d["scores_after"]["group_doc"]["sapling"] <= ZERO_SHOT_THRESHOLD)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    colored_metric("Avg GPTZero", f"{avg_gz:.3f}", avg_gz - baseline_gz)
                with col2:
                    colored_metric("Avg Sapling", f"{avg_sp:.3f}", avg_sp - baseline_sp)
                with col3:
                    st.metric("Avg WC Δ", f"{avg_wc_delta:+.0f}")
                with col4:
                    st.metric("Zero-shot", f"GZ:{zero_shot_gz}/{len(model_drafts)} SP:{zero_shot_sp}/{len(model_drafts)}")
                
                # Individual drafts
                for dr in sorted(model_drafts, key=lambda x: x["iter"]):
                    render_draft(dr, para_total, doc_name, model)
                
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
                zero_shot_gz = sum(1 for d in model_drafts if d["scores_after"]["group_doc"]["gptzero"] <= ZERO_SHOT_THRESHOLD)
                zero_shot_sp = sum(1 for d in model_drafts if d["scores_after"]["group_doc"]["sapling"] <= ZERO_SHOT_THRESHOLD)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    colored_metric("Avg GPTZero", f"{avg_gz:.3f}", avg_gz - baseline_gz)
                with col2:
                    colored_metric("Avg Sapling", f"{avg_sp:.3f}", avg_sp - baseline_sp)
                with col3:
                    st.metric("Avg WC Δ", f"{avg_wc_delta:+.0f}")
                with col4:
                    st.metric("Zero-shot", f"GZ:{zero_shot_gz}/{len(model_drafts)} SP:{zero_shot_sp}/{len(model_drafts)}")
                
                # Individual drafts
                for dr in sorted(model_drafts, key=lambda x: x["iter"]):
                    render_draft(dr, para_total, doc_name, model)
                
                st.divider()

    with tab3:
        st.markdown("### Model Comparison")
        
        with st.expander("ℹ️ Understanding comparisons", expanded=False):
            st.markdown("""
            This table compares all models tested on this document:
            - **Δ GZ/SP**: Change from baseline (negative = improvement)
            - **Zero-shot**: Number of drafts achieving ≤10% AI detection
            - **Quality**: Average content preservation score
            """)
        
        # Prepare comparison data
        comparison_data = []
        for model in sorted(by_model):
            for mode in ["doc", "para"]:
                drafts = by_model[model][mode]
                if drafts:
                    avg_gz = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in drafts])
                    avg_sp = np.mean([d["scores_after"]["group_doc"]["sapling"] for d in drafts])
                    avg_wc = np.mean([d["wordcount_after"] - d["wordcount_before"] for d in drafts])
                    
                    # Count zero-shot successes
                    zero_shot_gz = sum(1 for d in drafts if d["scores_after"]["group_doc"]["gptzero"] <= ZERO_SHOT_THRESHOLD)
                    zero_shot_sp = sum(1 for d in drafts if d["scores_after"]["group_doc"]["sapling"] <= ZERO_SHOT_THRESHOLD)
                    
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
                        "Drafts": len(drafts),
                        "Avg GPTZero": f"{avg_gz:.3f}",
                        "Δ GZ": avg_gz - baseline_gz,
                        "Zero-shot GZ": f"{zero_shot_gz}/{len(drafts)}",
                        "Avg Sapling": f"{avg_sp:.3f}",
                        "Δ SP": avg_sp - baseline_sp,
                        "Zero-shot SP": f"{zero_shot_sp}/{len(drafts)}",
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
            st.markdown("#### AI Detection Scores by Model & Mode")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
            
            models = sorted(by_model.keys())
            x = np.arange(len(models))
            width = 0.35
            
            # GPTZero scores
            doc_scores_gz = []
            para_scores_gz = []
            
            for model in models:
                doc_drafts = by_model[model]["doc"]
                para_drafts = by_model[model]["para"]
                
                doc_score = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in doc_drafts]) if doc_drafts else baseline_gz
                para_score = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in para_drafts]) if para_drafts else baseline_gz
                
                doc_scores_gz.append(doc_score)
                para_scores_gz.append(para_score)
            
            ax1.axhline(baseline_gz, color='red', linestyle='--', label='Baseline', alpha=0.7)
            
            # Color bars based on performance vs baseline
            doc_bars = ax1.bar(x - width/2, doc_scores_gz, width, label='Doc Mode', alpha=0.8)
            para_bars = ax1.bar(x + width/2, para_scores_gz, width, label='Para Mode', alpha=0.8)
            
            for bar, score in zip(doc_bars, doc_scores_gz):
                if score < baseline_gz:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            for bar, score in zip(para_bars, para_scores_gz):
                if score < baseline_gz:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            ax1.set_xlabel('Model')
            ax1.set_ylabel('GPTZero Score')
            ax1.set_title('Average GPTZero Scores by Model and Mode')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Sapling scores
            doc_scores_sp = []
            para_scores_sp = []
            
            for model in models:
                doc_drafts = by_model[model]["doc"]
                para_drafts = by_model[model]["para"]
                
                doc_score = np.mean([d["scores_after"]["group_doc"]["sapling"] for d in doc_drafts]) if doc_drafts else baseline_sp
                para_score = np.mean([d["scores_after"]["group_doc"]["sapling"] for d in para_drafts]) if para_drafts else baseline_sp
                
                doc_scores_sp.append(doc_score)
                para_scores_sp.append(para_score)
            
            ax2.axhline(baseline_sp, color='red', linestyle='--', label='Baseline', alpha=0.7)
            
            # Color bars based on performance vs baseline
            doc_bars = ax2.bar(x - width/2, doc_scores_sp, width, label='Doc Mode', alpha=0.8)
            para_bars = ax2.bar(x + width/2, para_scores_sp, width, label='Para Mode', alpha=0.8)
            
            for bar, score in zip(doc_bars, doc_scores_sp):
                if score < baseline_sp:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            for bar, score in zip(para_bars, para_scores_sp):
                if score < baseline_sp:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Sapling Score')
            ax2.set_title('Average Sapling Scores by Model and Mode')
            ax2.set_xticks(x)
            ax2.set_xticklabels(models, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
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
        st.caption("Shows how AI detection scores vary across multiple humanization attempts")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # GPTZero Doc mode
        for model in sorted(by_model):
            doc_drafts = sorted(by_model[model]["doc"], key=lambda x: x["iter"])
            if doc_drafts:
                iterations = [d["iter"] + 1 for d in doc_drafts]
                gz_scores = [d["scores_after"]["group_doc"]["gptzero"] for d in doc_drafts]
                ax1.plot(iterations, gz_scores, marker='o', label=model)
        
        ax1.axhline(baseline_gz, color='red', linestyle='--', label='Baseline', alpha=0.7)
        ax1.axhline(ZERO_SHOT_THRESHOLD, color='green', linestyle=':', label='Zero-shot threshold', alpha=0.7)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('GPTZero Score')
        ax1.set_title('Document Mode - GPTZero Score Progression')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # GPTZero Para mode
        for model in sorted(by_model):
            para_drafts = sorted(by_model[model]["para"], key=lambda x: x["iter"])
            if para_drafts:
                iterations = [d["iter"] + 1 for d in para_drafts]
                gz_scores = [d["scores_after"]["group_doc"]["gptzero"] for d in para_drafts]
                ax2.plot(iterations, gz_scores, marker='o', label=model)
        
        ax2.axhline(baseline_gz, color='red', linestyle='--', label='Baseline', alpha=0.7)
        ax2.axhline(ZERO_SHOT_THRESHOLD, color='green', linestyle=':', label='Zero-shot threshold', alpha=0.7)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('GPTZero Score')
        ax2.set_title('Paragraph Mode - GPTZero Score Progression')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Sapling Doc mode
        for model in sorted(by_model):
            doc_drafts = sorted(by_model[model]["doc"], key=lambda x: x["iter"])
            if doc_drafts:
                iterations = [d["iter"] + 1 for d in doc_drafts]
                sp_scores = [d["scores_after"]["group_doc"]["sapling"] for d in doc_drafts]
                ax3.plot(iterations, sp_scores, marker='o', label=model)
        
        ax3.axhline(baseline_sp, color='red', linestyle='--', label='Baseline', alpha=0.7)
        ax3.axhline(ZERO_SHOT_THRESHOLD, color='green', linestyle=':', label='Zero-shot threshold', alpha=0.7)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Sapling Score')
        ax3.set_title('Document Mode - Sapling Score Progression')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Sapling Para mode
        for model in sorted(by_model):
            para_drafts = sorted(by_model[model]["para"], key=lambda x: x["iter"])
            if para_drafts:
                iterations = [d["iter"] + 1 for d in para_drafts]
                sp_scores = [d["scores_after"]["group_doc"]["sapling"] for d in para_drafts]
                ax4.plot(iterations, sp_scores, marker='o', label=model)
        
        ax4.axhline(baseline_sp, color='red', linestyle='--', label='Baseline', alpha=0.7)
        ax4.axhline(ZERO_SHOT_THRESHOLD, color='green', linestyle=':', label='Zero-shot threshold', alpha=0.7)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Sapling Score')
        ax4.set_title('Paragraph Mode - Sapling Score Progression')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Quality metrics breakdown
        st.markdown("#### Quality Metrics Breakdown")
        st.caption("Heatmap showing success rate for each quality check across models and modes")
        
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