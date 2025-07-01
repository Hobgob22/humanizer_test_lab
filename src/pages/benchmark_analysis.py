# src/pages/benchmark_analysis.py
from __future__ import annotations

###############################################################################
#  Benchmark Analysis – extended metrics with MERGE functionality
#  • Per-flag quality columns (length_ok, same_meaning, …)
#  • Word-count-difference columns:
#        – Within 10 words %   – Within 20 words %
#        – % Longer            – % Shorter
#  • Per-folder word-count-delta histogram + summary
#  • Grammar quality scoring with percentage display
#  • Merge multiple runs into a single view
###############################################################################

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import ZERO_SHOT_THRESHOLD
from src.pages.utils import (
    GEMINI_FLAGS,
    colored_metric,
    qp_get,
    qp_set,
    render_draft,
    safe_hist,
)
from src.results_db import delete_run, list_runs, load_run

# ────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

# ────────────────────────── helpers ─────────────────────────────────────
_EXPECTED_FLAGS = (
    "length_ok",
    "same_meaning",
    "same_lang",
    "no_missing_info",
    "citation_preserved",
    "citation_content_ok",
)


def _iter_drafts(docs: List[Dict]) -> Tuple[Dict, ...]:
    """Yield ``(doc, draft)`` pairs for every draft in *docs*."""
    for doc in docs:
        for dr in doc.get("runs", []):
            yield doc, dr


def _merge_runs_data(run_ids: List[str]) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Merge data from multiple runs, handling duplicate models.
    Returns merged docs list and metadata about the merge.
    """
    merged_docs = []
    model_sources = defaultdict(set)  # Track which runs each model came from
    doc_by_name = defaultdict(lambda: {"runs": []})  # Group by document name
    
    for run_id in run_ids:
        run_data = load_run(run_id)
        if not run_data:
            continue
            
        docs = run_data.get("docs", [])
        
        for doc in docs:
            doc_name = doc["document"]
            
            # Copy doc metadata if not already present
            if "document" not in doc_by_name[doc_name]:
                doc_by_name[doc_name].update({
                    "document": doc_name,
                    "folder": doc.get("folder", "unknown"),
                    "paragraph_count": doc.get("paragraph_count", 0),
                    "error": doc.get("error"),
                    "warning": doc.get("warning"),
                    "empty": doc.get("empty", False),
                    "phase_failed": doc.get("phase_failed"),
                })
            
            # Add runs from this doc, tracking model sources
            for run in doc.get("runs", []):
                model = run.get("model", "unknown")
                mode = run.get("mode", "unknown")
                iter_num = run.get("iter", 0)
                
                # Create a unique key for this draft
                draft_key = (model, mode, iter_num)
                model_sources[model].add(run_id)
                
                # Add the run with source metadata
                run_copy = run.copy()
                run_copy["_source_run"] = run_id
                doc_by_name[doc_name]["runs"].append(run_copy)
    
    # Convert back to list format
    merged_docs = list(doc_by_name.values())
    
    # Create metadata about the merge
    merge_metadata = {
        "total_runs": len(run_ids),
        "model_sources": dict(model_sources),
        "total_models": len(model_sources),
        "models_list": sorted(model_sources.keys()),
    }
    
    return merged_docs, merge_metadata


# ╔════════════════════════ analytics ══════════════════════════════════╗
def _aggregate_statistics_by_model_mode_folder(docs: List[Dict]) -> Dict[str, Any]:
    """
    Build nested dict  folder → model → mode → stats
    and attach word-count-difference metrics + quality-flag rates.
    """
    stats: DefaultDict[str, DefaultDict[str, DefaultDict[str, Dict]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )

    # ── baselines -------------------------------------------------------
    folder_baselines: DefaultDict[str, List[Dict]] = defaultdict(list)
    for doc in docs:
        if not doc.get("runs"):
            continue
        first = doc["runs"][0]
        if (
            "scores_before" in first
            and "group_doc" in first["scores_before"]
            and "gptzero" in first["scores_before"]["group_doc"]
        ):
            folder_baselines[doc.get("folder", "unknown")].append(
                {
                    "gptzero": first["scores_before"]["group_doc"]["gptzero"],
                    "sapling": first["scores_before"]["group_doc"]["sapling"],
                    "wordcount": first.get("wordcount_before", 0),
                }
            )

    # ── ignore None / NaN values ───────────────────────────────────
    _valid = lambda v: v is not None and not (isinstance(v, float) and np.isnan(v))

    folder_avg_baselines = {
        f: {
            "gptzero": np.nanmean([b["gptzero"]  for b in bl if _valid(b["gptzero"])]),
            "sapling": np.nanmean([b["sapling"]  for b in bl if _valid(b["sapling"])]),
            "wordcount": np.nanmean([b["wordcount"] for b in bl if _valid(b["wordcount"])]),
        }
        for f, bl in folder_baselines.items()
    }

    # ── per-draft collection -------------------------------------------
    for doc, dr in _iter_drafts(docs):
        folder = doc.get("folder", "unknown")
        if "scores_after" not in dr or "group_doc" not in dr["scores_after"]:
            continue
        model = dr.get("model", "unknown")
        mode = dr.get("mode", "unknown")

        bucket = stats[folder][model].setdefault(
            mode,
            {
                "after_scores": [],
                "wc_deltas": [],
                "quality_flags": defaultdict(list),
                "grammar_scores": [],  # NEW: track grammar scores
                "draft_count": 0,
                "mismatch_count": 0,
                "zs_hits": {"gptzero": 0, "sapling": 0},
                # NEW – keep raw series for the nerd-stats tab
                "series": defaultdict(list),
                # Track source runs for merged views
                "source_runs": set(),
            },
        )

        # Track source run if present
        if "_source_run" in dr:
            bucket["source_runs"].add(dr["_source_run"])

        # detector scores
        gz = dr["scores_after"]["group_doc"].get("gptzero", 0)
        sp = dr["scores_after"]["group_doc"].get("sapling", 0)
        bucket["after_scores"].append({"gptzero": gz, "sapling": sp})
        if gz <= ZERO_SHOT_THRESHOLD:
            bucket["zs_hits"]["gptzero"] += 1
        if sp <= ZERO_SHOT_THRESHOLD:
            bucket["zs_hits"]["sapling"] += 1

        # word-count delta
        delta_wc = dr.get("wordcount_after", 0) - dr.get("wordcount_before", 0)
        bucket["wc_deltas"].append(delta_wc)

        # ── NEW: store raw series for extended-stats tab ──────────────────────
        bucket["series"]["after_gz"].append(gz)
        bucket["series"]["after_sp"].append(sp)
        bucket["series"]["wc"].append(delta_wc)
        
        # quality flags and grammar (skip drafts with paragraph mismatch)
        if not dr.get("para_mismatch", False):
            total = dr.get("para_count_before", 1)
            
            # Boolean quality flags
            for flag in _EXPECTED_FLAGS:
                cnt = dr.get("flag_counts", {}).get(flag, 0)
                bucket["quality_flags"][flag].append((cnt / total) * 100 if total else 0)
            
            # Grammar score (NEW)
            grammar_score = dr.get("flag_counts", {}).get("grammar_score")
            if grammar_score is not None:
                bucket["grammar_scores"].append(grammar_score)
                bucket["series"]["grammar"].append(grammar_score)
            
            # Calculate overall quality percentage (boolean flags only for backward compatibility)
            qual_pct = (
                sum(dr.get("flag_counts", {}).get(f, 0) for f in _EXPECTED_FLAGS)
                / (total * len(_EXPECTED_FLAGS))
                * 100
                if total
                else 0
            )
            bucket["series"]["quality"].append(qual_pct)

        bucket["draft_count"] += 1
        if dr.get("para_mismatch", False):
            bucket["mismatch_count"] += 1

    # ── aggregate bucket data ------------------------------------------
    result: Dict[str, Any] = {}
    for folder, models in stats.items():
        baseline = folder_avg_baselines.get(folder, {"gptzero": 0.5, "sapling": 0.5})
        result[folder] = {}
        for model, modes in models.items():
            result[folder][model] = {}
            for mode, data in modes.items():
                if not data["draft_count"]:
                    continue

                _ok = lambda v: v is not None and not (isinstance(v, float) and np.isnan(v))

                after_gz = np.nanmean([s["gptzero"]  for s in data["after_scores"] if _ok(s["gptzero"])])
                after_sp = np.nanmean([s["sapling"]  for s in data["after_scores"] if _ok(s["sapling"])])

                zs_gz_pct = data["zs_hits"]["gptzero"] / data["draft_count"] * 100
                zs_sp_pct = data["zs_hits"]["sapling"] / data["draft_count"] * 100

                # word-count diff metrics
                deltas = np.array(data["wc_deltas"])
                within10 = (np.abs(deltas) <= 10).mean() * 100
                within20 = (np.abs(deltas) <= 20).mean() * 100
                pct_longer = (deltas > 0).mean() * 100
                pct_shorter = (deltas < 0).mean() * 100

                # Grammar score average (NEW)
                avg_grammar = np.mean(data["grammar_scores"]) if data["grammar_scores"] else None

                result[folder][model][mode] = {
                    "baseline": baseline,
                    "after": {"gptzero": after_gz, "sapling": after_sp},
                    "deltas": {
                        "gptzero": after_gz - baseline["gptzero"],
                        "sapling": after_sp - baseline["sapling"],
                        "wordcount": deltas.mean() if deltas.size else 0,
                    },
                    "quality": {
                        flag: np.mean(vals) if vals else 0
                        for flag, vals in data["quality_flags"].items()
                    },
                    "grammar_score": avg_grammar,  # NEW
                    "draft_count": data["draft_count"],
                    "mismatch_rate": data["mismatch_count"] / data["draft_count"] * 100,
                    "zs_hits": data["zs_hits"],
                    "zero_shot_success": {"gptzero": zs_gz_pct, "sapling": zs_sp_pct},
                    "wc_diff": {
                        "within10": within10,
                        "within20": within20,
                        "pct_longer": pct_longer,
                        "pct_shorter": pct_shorter,
                    },
                    "wc_deltas": data["wc_deltas"],      # for histograms
                    "series": data["series"],             # 🔑 keep raw numbers
                    "source_runs": data["source_runs"],   # Track which runs contributed
                }
    return result


# ═══════════ helper: build model‑perf dataframe ═══════════════════════

def _compute_model_perf(
    stats: Dict[str, Any], restrict_folders: Set[str] | None = None
) -> pd.DataFrame:
    """Summarise performance, using exact hit counts (no rounding error)."""
    agg: DefaultDict[str, DefaultDict[str, Dict[str, Any]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "gz_deltas": [],
                "sp_deltas": [],
                "quality": [],
                "grammar_scores": [],  # NEW
                "drafts": 0,
                "zs_gz_hits": 0,
                "zs_sp_hits": 0,
                "folders": set(),
                "source_runs": set(),
            }
        )
    )

    for folder, models in stats.items():
        if restrict_folders and folder not in restrict_folders:
            continue
        for model, modes in models.items():
            for mode, s in modes.items():
                bucket = agg[model][mode]
                bucket["gz_deltas"].append(s["deltas"]["gptzero"])
                if not (isinstance(s["deltas"]["sapling"], float) and np.isnan(s["deltas"]["sapling"])):
                    bucket["sp_deltas"].append(s["deltas"]["sapling"])
                bucket["quality"].append(np.mean(list(s["quality"].values())))
                
                # Add grammar score if available (NEW)
                if s.get("grammar_score") is not None:
                    bucket["grammar_scores"].append(s["grammar_score"])
                
                bucket["drafts"] += s["draft_count"]
                bucket["zs_gz_hits"] += s["zs_hits"]["gptzero"]
                bucket["zs_sp_hits"] += s["zs_hits"]["sapling"]
                bucket["folders"].add(folder)
                bucket["source_runs"].update(s.get("source_runs", set()))

    rows = []
    for model, modes in agg.items():
        for mode in ("doc", "para"):
            m = modes.get(mode)
            if not m or m["drafts"] == 0:
                continue
            
            # Calculate average grammar score as percentage (NEW)
            avg_grammar = np.mean(m["grammar_scores"]) if m["grammar_scores"] else None
            
            row_data = {
                "Model": model,
                "Mode": mode.title(),
                "Total Drafts": m["drafts"],
                "Avg Δ GZ": np.nanmean(m["gz_deltas"]) if m["gz_deltas"] else np.nan,
                "Avg Δ SP": np.nanmean(m["sp_deltas"]) if m["sp_deltas"] else np.nan,
                "Zero-shot GZ": f"{m['zs_gz_hits'] / m['drafts'] * 100:.1f}%",
                "Zero-shot SP": f"{m['zs_sp_hits'] / m['drafts'] * 100:.1f}%",
                "Avg Quality": f"{np.mean(m['quality']):.1f}%",
                "Avg Grammar": f"{avg_grammar * 10:.0f}%" if avg_grammar is not None else "—",  # Convert to %
                "Folders": len(m["folders"]),
            }
            
            # Only add Source Runs column if we have multiple sources
            if len(m["source_runs"]) > 0:
                row_data["Source Runs"] = len(m["source_runs"])
                
            rows.append(row_data)
            
    return pd.DataFrame(rows)


# ╔═══════════════════ extended-stats helpers ═══════════════════════════╗
def _describe(arr):
    """Return min, 25-perc, median, mean, 75-perc, max for *arr* (list-like)."""
    if not arr:
        return {"Min": 0, "P25": 0, "Median": 0, "Mean": 0, "P75": 0, "Max": 0}
    # Drop None / NaN first
    clean = [v for v in arr if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return {"Min": 0, "P25": 0, "Median": 0, "Mean": 0, "P75": 0, "Max": 0}
    a = np.asarray(clean, dtype=float)
    return {
        "Min":    float(np.min(a)),
        "P25":    float(np.percentile(a, 25)),
        "Median": float(np.median(a)),
        "Mean":   float(np.mean(a)),
        "P75":    float(np.percentile(a, 75)),
        "Max":    float(np.max(a)),
    }


def _build_extended_stats(stats):
    """
    Build DataFrame with descriptive statistics for
    GPTZero, Sapling, word-count Δ, quality %, and grammar score.
    """
    rows = []
    for folder, models in stats.items():
        for model, modes in models.items():
            for mode, s in modes.items():
                ser = s.get("series", {})
                if not ser:
                    continue
                    
                # Convert grammar scores to percentages for display
                grammar_series = ser.get("grammar", [])
                grammar_pct_series = [g * 10 for g in grammar_series if g is not None]
                
                rows.append(
                    {
                        "Folder": folder,
                        "Model":  model,
                        "Mode":   mode.title(),
                        **{f"GPTZero {k}": v for k, v in _describe(ser.get("after_gz", [])).items()},
                        **{f"Sapling {k}": v for k, v in _describe(ser.get("after_sp", [])).items()},
                        **{f"WC Δ {k}":  v for k, v in _describe(ser.get("wc", [])).items()},
                        **{f"Quality {k}": v for k, v in _describe(ser.get("quality", [])).items()},
                        **{f"Grammar % {k}": v for k, v in _describe(grammar_pct_series).items()},  # NEW as %
                    }
                )
    return pd.DataFrame(rows)


# ╔═════════════════════ styling helpers ════════════════════════════════╗
def _style_delta(v):
    if isinstance(v, (int, float)):
        return "color: green; font-weight:bold" if v < 0 else "color: red; font-weight:bold" if v > 0 else ""
    return ""


def _style_zs(v):
    if isinstance(v, str) and v.endswith("%"):
        f = float(v[:-1])
        if f >= 80:
            return "color: green; font-weight:bold"
        if f >= 50:
            return "color: orange"
        return "color: red"
    return ""


def _style_quality(v):
    if isinstance(v, str) and v.endswith("%"):
        f = float(v[:-1])
        if f >= 90:
            return "color: green"
        if f >= 70:
            return "color: orange"
        return "color: red"
    return ""

def _style_grammar(v):
    """Style grammar scores with color coding (now expects percentage format)."""
    if isinstance(v, str) and v.endswith("%") and v != "—":
        try:
            score = float(v.rstrip("%"))
            if score >= 90:
                return "color: green; font-weight: bold"
            elif score >= 70:
                return "color: orange"
            elif score < 50:
                return "color: red; font-weight: bold"
        except:
            pass
    return ""

# ╔════════════ word-count distribution plot ════════════════════════════╗
def _plot_wordcount_distribution(detailed_stats: Dict[str, Any], folder: str) -> None:
    """Histogram & summary of word-count deltas (per folder)."""
    deltas = []
    for model in detailed_stats[folder].values():
        for mode_stats in model.values():
            deltas.extend(mode_stats["wc_deltas"])
    if not deltas:
        st.info("No word-count data for this folder.")
        return

    arr = np.array(deltas)
    summary = pd.DataFrame(
        [
            {
                "Drafts": len(arr),
                "Mean Δ": f"{arr.mean():+.1f}",
                "Median Δ": f"{np.median(arr):+.1f}",
                "Within 10 words %": f"{(np.abs(arr) <= 10).mean()*100:.1f}%",
                "Within 20 words %": f"{(np.abs(arr) <= 20).mean()*100:.1f}%",
                "% Longer": f"{(arr > 0).mean()*100:.1f}%",
                "% Shorter": f"{(arr < 0).mean()*100:.1f}%",
            }
        ]
    )
    st.markdown("##### Word-count change summary")
    st.dataframe(summary, hide_index=True, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    safe_hist(ax, deltas, bins=30, alpha=0.7)
    ax.axvline(0, color="black", linewidth=0.8, label="No change")
    ax.set_xlabel("Word-count Δ (after − before)")
    ax.set_ylabel("Drafts")
    ax.set_title(f"Word-count change distribution – {folder.replace('_',' ').title()}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)


def _render_model_perf(df: pd.DataFrame, title_suffix: str = "") -> None:
    """Render summary table + leaderboards for a given DataFrame."""
    if df.empty:
        st.info("No data available for this selection.")
        return

    # Check if this is a merged view (has Source Runs column)
    is_merged = "Source Runs" in df.columns
    
    styled = (
        df.style.applymap(_style_delta, subset=["Avg Δ GZ", "Avg Δ SP"])
        .applymap(_style_zs, subset=["Zero-shot GZ", "Zero-shot SP"])
        .applymap(_style_grammar, subset=["Avg Grammar"])  # NEW
        .format({"Avg Δ GZ": "{:.3f}", "Avg Δ SP": "{:.3f}"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── leaderboards ----------------------------------------------------------
    st.markdown(f"### 🏆 Best Performers {title_suffix}")
    col1, col2 = st.columns(2)

    df_num = df.copy()
    df_num["ZS_GZ_num"] = df_num["Zero-shot GZ"].str.rstrip("%").astype(float)
    df_num["ZS_SP_num"] = df_num["Zero-shot SP"].str.rstrip("%").astype(float)

    with col1:
        st.markdown("#### Best AI Score Reduction")
        st.caption("Largest negative Δ scores")
        st.markdown("**GPTZero:**")
        st.dataframe(
            df_num.nsmallest(5, "Avg Δ GZ")[["Model", "Mode", "Avg Δ GZ"]],
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("**Sapling:**")
        st.dataframe(
            df_num.nsmallest(5, "Avg Δ SP")[["Model", "Mode", "Avg Δ SP"]],
            hide_index=True,
            use_container_width=True,
        )

    with col2:
        st.markdown("#### Best Zero-shot Success")
        st.caption("Highest percentage of drafts ≤ 10 % AI detection")
        st.markdown("**GPTZero:**")
        st.dataframe(
            df_num.nlargest(5, "ZS_GZ_num")[["Model", "Mode", "Zero-shot GZ"]],
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("**Sapling:**")
        st.dataframe(
            df_num.nlargest(5, "ZS_SP_num")[["Model", "Mode", "Zero-shot SP"]],
            hide_index=True,
            use_container_width=True,
        )


# ╔════════════════ helper – folder table ══════════════════════════════╗
def _create_model_comparison_table(stats: Dict[str, Any], folder: str) -> pd.DataFrame:
    """Detailed table (one folder)."""
    rows = []
    if folder not in stats:
        return pd.DataFrame()

    for model, modes in stats[folder].items():
        for mode in ("doc", "para"):
            if mode not in modes:
                continue
            s = modes[mode]
            row = {
                "Model": model,
                "Mode": mode.title(),
                "Drafts": s["draft_count"],
                "Baseline GZ": f"{s['baseline']['gptzero']:.3f}",
                "After GZ": f"{s['after']['gptzero']:.3f}",
                "Δ GZ": s["deltas"]["gptzero"],
                "Zero-shot GZ": f"{s['zero_shot_success']['gptzero']:.1f}%",
                "Baseline SP": f"{s['baseline']['sapling']:.3f}",
                "After SP": f"{s['after']['sapling']:.3f}",
                "Δ SP": s["deltas"]["sapling"],
                "Zero-shot SP": f"{s['zero_shot_success']['sapling']:.1f}%",
                "Avg WC Δ": f"{s['deltas']['wordcount']:+.0f}",
                "Within 10 words %": f"{s['wc_diff']['within10']:.1f}%",
                "Within 20 words %": f"{s['wc_diff']['within20']:.1f}%",
                "% Longer": f"{s['wc_diff']['pct_longer']:.1f}%",
                "% Shorter": f"{s['wc_diff']['pct_shorter']:.1f}%",
                "Quality %": f"{np.mean(list(s['quality'].values())):.1f}%",
                "Grammar %": f"{s['grammar_score'] * 10:.0f}%" if s.get('grammar_score') is not None else "—",  # Convert to %
                "Mismatch %": f"{s['mismatch_rate']:.1f}%",
            }
            # per-flag columns
            for flag in _EXPECTED_FLAGS:
                row[f"{flag.replace('_',' ').title()} %"] = f"{s['quality'].get(flag, 0):.1f}%"
            rows.append(row)

    # column order
    qual_cols = [f"{f.replace('_',' ').title()} %" for f in _EXPECTED_FLAGS]
    base_cols = [
        "Model","Mode","Drafts",
        "Baseline GZ","After GZ","Δ GZ","Zero-shot GZ",
        "Baseline SP","After SP","Δ SP","Zero-shot SP",
        "Avg WC Δ","Within 10 words %","Within 20 words %","% Longer","% Shorter",
        "Quality %","Grammar %","Mismatch %",  # Grammar as %
    ]
    return pd.DataFrame(rows)[base_cols + qual_cols]

# ═══════════════ RUN OVERVIEW & DOC PAGE (main) ═══════════════════════
def page_runs() -> None:
    # Get all available runs
    runs_meta = list_runs()
    if not runs_meta:
        st.info("No benchmarks stored yet. Create a new run to get started!")
        return

    # Check for view mode
    view_mode = qp_get("view_mode", "single")  # single or merged
    
    # View mode selector
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("📊 Benchmark Analysis")
    with col2:
        view_options = ["Single Run", "Merge Runs"]
        selected_view = st.radio(
            "View Mode",
            view_options,
            index=0 if view_mode == "single" else 1,
            horizontal=True,
            key="view_mode_selector"
        )
        new_view_mode = "single" if selected_view == "Single Run" else "merged"
        if new_view_mode != view_mode:
            qp_set(view_mode=new_view_mode)
            st.rerun()

    # Handle different view modes
    if view_mode == "merged":
        _page_merged_runs(runs_meta)
    else:
        _page_single_run(runs_meta)


def _page_merged_runs(runs_meta: List[Dict]) -> None:
    """Handle merged runs view."""
    st.subheader("🔀 Merge Multiple Runs")
    
    with st.expander("ℹ️ About Merged View", expanded=False):
        st.markdown("""
        **Merged View** allows you to combine results from multiple benchmark runs:
        - Select multiple runs to merge their results
        - Models are automatically deduplicated
        - All statistics are recalculated for the combined dataset
        - Original runs remain unchanged (this is just a view)
        
        **Use cases:**
        - Compare models tested in different runs
        - See aggregate statistics across multiple experiments
        - Analyze performance without re-running benchmarks
        """)
    
    # Multi-select for runs
    run_labels = [
        f"{r['name']} ({time.strftime('%Y-%m-%d %H:%M', time.localtime(r['ts']))})"
        for r in runs_meta
    ]
    
    selected_labels = st.multiselect(
        "Select runs to merge",
        run_labels,
        default=run_labels[:2] if len(run_labels) >= 2 else run_labels,
        help="Choose 2 or more runs to merge their results"
    )
    
    if len(selected_labels) < 2:
        st.warning("Please select at least 2 runs to merge.")
        return
    
    # Get selected run IDs
    selected_indices = [run_labels.index(label) for label in selected_labels]
    selected_run_ids = [runs_meta[i]["name"] for i in selected_indices]
    
    # Merge the data
    with st.spinner("Merging runs..."):
        merged_docs, merge_metadata = _merge_runs_data(selected_run_ids)
    
    if not merged_docs:
        st.error("No data found in selected runs.")
        return
    
    # Display merge info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔀 Merged Runs", merge_metadata["total_runs"])
    with col2:
        st.metric("🤖 Unique Models", merge_metadata["total_models"])
    with col3:
        successful_docs = sum(1 for d in merged_docs if d.get("runs"))
        st.metric("📄 Documents", len(merged_docs))
    with col4:
        total_drafts = sum(len(d.get("runs", [])) for d in merged_docs)
        st.metric("📝 Total Drafts", total_drafts)
    
    # Show which models came from which runs
    with st.expander("📊 Model Sources", expanded=False):
        model_source_df = pd.DataFrame([
            {"Model": model, "Source Runs": len(sources), "Run Names": ", ".join(sorted(sources))}
            for model, sources in merge_metadata["model_sources"].items()
        ])
        st.dataframe(model_source_df, use_container_width=True, hide_index=True)
    
    # Now display the merged analysis using the same code as single run
    _display_analysis(merged_docs, f"Merged: {len(selected_run_ids)} runs", is_merged=True)


def _page_single_run(runs_meta: List[Dict]) -> None:
    """Handle single run view (original functionality)."""
    # --- run selection --------------------------------------------------------
    run_id = qp_get("run")
    doc_name = qp_get("doc")
    view = qp_get("view")

    run_labels = [
        f"{r['name']} ({time.strftime('%Y-%m-%d %H:%M', time.localtime(r['ts']))})"
        for r in runs_meta
    ]
    default_idx = next((i for i, r in enumerate(runs_meta) if r["name"] == run_id), 0)
    selected = st.selectbox("Select benchmark run", run_labels, index=default_idx)
    run_id = runs_meta[run_labels.index(selected)]["name"]

    run = load_run(run_id) or {}
    docs: List[Dict] = run.get("docs", [])
    if not docs:
        st.warning("Selected run is empty.")
        return

    # single-document deep-dive
    if view == "doc" and doc_name:
        _page_document(run_id, docs, doc_name)
        return

    # Display analysis for single run
    _display_analysis(docs, run_id, is_merged=False)
    
    # Run management (only for single runs)
    st.divider()
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Delete Run", type="secondary"):
            if st.checkbox("Confirm deletion"):
                delete_run(run_id)
                st.warning("Run deleted!")
                qp_set(run=None, view=None, doc=None)
                st.rerun()


def _display_analysis(docs: List[Dict], run_name: str, is_merged: bool = False) -> None:
    """Display the analysis tabs for either single or merged runs."""
    # --- overview header ------------------------------------------------------
    st.header(f"📊 Analysis: **{run_name}**")

    successful_docs = sum(1 for d in docs if d.get("runs"))
    failed_docs = len(docs) - successful_docs
    total_drafts = sum(len(d.get("runs", [])) for d in docs)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📄 Documents", len(docs))
    with col2:
        st.metric("✅ Successful", successful_docs)
    with col3:
        st.metric("❌ Failed", failed_docs)
    with col4:
        st.metric("📝 Total drafts", total_drafts)
    with col5:
        models_used = {
            draft.get("model", "unknown")
            for doc in docs
            for draft in doc.get("runs", [])
        }
        st.metric("🤖 Models", len(models_used))

    if failed_docs:
        st.warning(
            f"⚠️ {failed_docs} documents failed processing. "
            "They are excluded from statistics but listed in the Documents tab."
        )

    # ── calculate analytics once ---------------------------------------------
    detailed_stats = _aggregate_statistics_by_model_mode_folder(docs)

    # --- main tabs ------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 By Folder & Model",
            "📈 Model Performance",
            "📐 Extended Stats",
            "📁 Folder Summary",
            "📊 Distributions",
            "📄 Documents",
        ]
    )

    # ════════════════════════════════════════════════════════════════════════
    # Tab 1 – Detailed per-folder / model table + charts (+NEW wc histogram)
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("🎯 Detailed Statistics by Folder, Model, and Mode")

        with st.expander("ℹ️ Understanding the metrics", expanded=False):
            st.markdown(
                """
                **Key Metrics:**  
                • **Δ GZ / Δ SP** – change in AI-detection score (negative = better)  
                • **Zero-shot** – % drafts ≤ 10 % on detector  
                • **Quality %** – average of all quality checks  
                • **Grammar %** – average grammatical correctness score  
                • **Within 10 / 20 words** – word-count distance from original  
                • **% Longer / % Shorter** – drafts that grew / shrank  
                • **Mismatch %** – paragraph-count mismatches
                """
            )

        # Folder ordering
        folder_order = ["ai_texts", "human_texts", "ai_paras", "human_paras"]
        available_folders = [f for f in folder_order if f in detailed_stats]
        other_folders = [f for f in detailed_stats if f not in folder_order]
        all_folders = available_folders + other_folders
        
        # ── iterate folders ----------------------------------------------
        for folder in all_folders:
            with st.expander(
                f"📁 **{folder.replace('_', ' ').title()}**",
                expanded=(folder == "ai_texts"),
            ):
                if folder not in detailed_stats:
                    st.info("No data for this folder.")
                    continue

                df = _create_model_comparison_table(detailed_stats, folder)
                if df.empty:
                    st.info("No drafts for this folder.")
                    continue

                # Style dataframe incl. new columns
                qual_cols = [c for c in df.columns if c.endswith(" %") and c not in
                             ("Zero-shot GZ","Zero-shot SP","Quality %","Grammar %","Mismatch %")]
                styled_df = (
                    df.style.applymap(_style_delta, subset=["Δ GZ", "Δ SP"])
                    .applymap(_style_zs, subset=["Zero-shot GZ", "Zero-shot SP"])
                    .applymap(_style_quality, subset=qual_cols + ["Quality %"])
                    .applymap(_style_grammar, subset=["Grammar %"])  # NEW
                    .format({"Δ GZ": "{:+.3f}", "Δ SP": "{:+.3f}"})
                )
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

                # ── side-by-side charts ---------------------------------
                col1, col2 = st.columns(2)

                # 1️⃣ AI-detector Δ-score bars (unchanged)
                with col1:
                    st.markdown("#### AI-detector score changes")
                    st.caption("Negative bars = improvement")

                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
                    models = df["Model"].unique()
                    x = np.arange(len(models))
                    width = 0.35

                    for i, mode in enumerate(["Doc", "Para"]):
                        mdf = df[df["Mode"] == mode]
                        deltas_gz = [
                            mdf[mdf["Model"] == m]["Δ GZ"].iloc[0] if not mdf[mdf["Model"] == m].empty else 0
                            for m in models
                        ]
                        deltas_sp = [
                            mdf[mdf["Model"] == m]["Δ SP"].iloc[0] if not mdf[mdf["Model"] == m].empty else 0
                            for m in models
                        ]
                        shift = x + (i - 0.5) * width
                        ax1.bar(
                            shift,
                            deltas_gz,
                            width,
                            label=f"{mode}",
                            color=["green" if v < 0 else "red" for v in deltas_gz],
                            alpha=0.8 if mode == "Doc" else 0.3,
                            hatch="-" if mode == "Para" else None,
                            edgecolor="black" if mode == "Para" else None,
                        )
                        ax2.bar(
                            shift,
                            deltas_sp,
                            width,
                            label=f"{mode}",
                            color=["green" if v < 0 else "red" for v in deltas_sp],
                            alpha=0.8 if mode == "Doc" else 0.3,
                            hatch="-" if mode == "Para" else None,
                            edgecolor="black" if mode == "Para" else None,
                        )

                    for ax, title, ylabel in (
                        (ax1, "Δ GPTZero", "Δ Score"),
                        (ax2, "Δ Sapling", "Δ Score"),
                    ):
                        ax.axhline(0, color="black", linewidth=0.8)
                        ax.set_xticks(x)
                        ax.set_xticklabels(models, rotation=45, ha="right")
                        ax.set_title(title)
                        ax.set_ylabel(ylabel)
                        ax.grid(True, alpha=0.3)
                        ax.legend()

                    plt.tight_layout()
                    st.pyplot(fig)

                # 2️⃣ Zero-shot & quality bars (updated with grammar)
                with col2:
                    st.markdown("#### Zero-shot success & quality")
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

                    # zero-shot
                    bar_w = 0.2
                    models = df["Model"].unique()
                    x = np.arange(len(models))
                    combos = [("Doc", "GZ"), ("Doc", "SP"), ("Para", "GZ"), ("Para", "SP")]

                    for i, (mode, det) in enumerate(combos):
                        mdf = df[df["Mode"] == mode]
                        vals = [
                            float(
                                mdf[mdf["Model"] == m][f"Zero-shot {det}"].str.rstrip("%").iloc[0]
                            )
                            if not mdf[mdf["Model"] == m].empty
                            else 0
                            for m in models
                        ]
                        ax1.bar(
                            x + (i - 1.5) * bar_w,
                            vals,
                            bar_w,
                            label=f"{mode} {det}",
                            alpha=0.8,
                        )

                    ax1.set_ylim(0, 100)
                    ax1.set_xticks(x)
                    ax1.set_xticklabels(models, rotation=45, ha="right")
                    ax1.set_ylabel("%")
                    ax1.set_title("Zero-shot success")
                    ax1.legend(fontsize="small")
                    ax1.grid(True, alpha=0.3)

                    # quality and grammar
                    bar_width = 0.35
                    qualities = []
                    grammars = []
                    
                    for m in models:
                        mdf = df[df["Model"] == m]
                        if not mdf.empty:
                            qual_val = float(mdf.iloc[0]["Quality %"].rstrip("%"))
                            qualities.append(qual_val)
                            
                            gram_val = mdf.iloc[0]["Grammar %"]
                            if gram_val != "—":
                                grammars.append(float(gram_val.rstrip("%")))
                            else:
                                grammars.append(0)
                        else:
                            qualities.append(0)
                            grammars.append(0)
                    
                    # Quality bars
                    bars1 = ax2.bar(
                        x - bar_width/2,
                        qualities,
                        bar_width,
                        label="Quality",
                        color=["green" if q >= 80 else "orange" if q >= 60 else "red" for q in qualities],
                    )
                    
                    # Grammar bars
                    bars2 = ax2.bar(
                        x + bar_width/2,
                        grammars,
                        bar_width,
                        label="Grammar",
                        color=["green" if g >= 90 else "orange" if g >= 70 else "red" if g > 0 else "gray" for g in grammars],
                    )
                    
                    ax2.set_ylim(0, 100)
                    ax2.set_xticks(x)
                    ax2.set_xticklabels(models, rotation=45, ha="right")
                    ax2.set_ylabel("%")
                    ax2.set_title("Average quality & grammar scores")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)

                # Word-count distribution plot (NEW)
                st.divider()
                _plot_wordcount_distribution(detailed_stats, folder)


    with tab2:
        st.subheader("📈 Model Performance")
        with st.expander("ℹ️ About this view", expanded=False):
            st.markdown(
                """
                Compare humanizer models on different document sets.  
                Lower **Δ GZ / Δ SP** values and higher **Zero-shot** rates are better.
                Higher **Grammar %** indicates better grammatical quality.
                """
            )

        folder_order = ["ai_texts", "human_texts", "ai_paras", "human_paras"]
        available_folders = [f for f in folder_order if f in detailed_stats]

        sub_tabs = ["All Folders"] + [
            f.replace("_", " ").title() for f in available_folders
        ]
        st_subtabs = st.tabs(sub_tabs)

        # All folders combined
        with st_subtabs[0]:
            df_all = _compute_model_perf(detailed_stats)
            _render_model_perf(df_all, title_suffix="– All Folders")

        # per-folder
        for idx, folder in enumerate(available_folders, start=1):
            with st_subtabs[idx]:
                pn = folder.replace("_", " ").title()
                st.markdown(f"### 📂 {pn}")
                df_folder = _compute_model_perf(detailed_stats, {folder})
                _render_model_perf(df_folder, title_suffix=f"– {pn}")

    
    with tab3:
        st.subheader("📐 Extended Statistics")
        st.caption("Min, 25-percentile, median, mean, 75-percentile and max for each metric.")
        ext_df = _build_extended_stats(detailed_stats)
        if ext_df.empty:
            st.info("No data available for extended statistics.")
        else:
            st.dataframe(ext_df, use_container_width=True, hide_index=True)

    with tab4:
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
            all_grammar_scores = []  # NEW
            
            for model, modes in models.items():
                for mode, stats in modes.items():
                    total_drafts += stats["draft_count"]
                    all_gz_deltas.append(stats["deltas"]["gptzero"])
                    all_sp_deltas.append(stats["deltas"]["sapling"])
                    all_quality.append(np.mean(list(stats["quality"].values())))
                    all_zero_shot_gz.append(stats["zero_shot_success"]["gptzero"])
                    all_zero_shot_sp.append(stats["zero_shot_success"]["sapling"])
                    
                    # Add grammar scores (NEW)
                    if stats.get("grammar_score") is not None:
                        all_grammar_scores.append(stats["grammar_score"])
            
            if total_drafts > 0:
                folder_summary.append({
                    "Folder": folder.replace('_', ' ').title(),
                    "Total Drafts": total_drafts,
                    "Models": len(models),
                    "Avg Δ GZ": np.nanmean(all_gz_deltas),
                    "Avg Δ SP": np.nanmean(all_sp_deltas),
                    "Zero-shot GZ": f"{np.mean(all_zero_shot_gz):.1f}%",
                    "Zero-shot SP": f"{np.mean(all_zero_shot_sp):.1f}%",
                    "Avg Quality": f"{np.mean(all_quality):.1f}%",
                    "Avg Grammar": f"{np.mean(all_grammar_scores) * 10:.0f}%" if all_grammar_scores else "—",  # NEW as %
                })
        
        folder_df = pd.DataFrame(folder_summary)
        if not folder_df.empty:
            # Style the dataframe
            styled_folder = folder_df.style.applymap(
                lambda x: 'color: green; font-weight: bold' if isinstance(x, (int, float)) and x < 0 else ('color: red; font-weight: bold' if isinstance(x, (int, float)) and x > 0 else ''),
                subset=['Avg Δ GZ', 'Avg Δ SP']
            ).applymap(
                _style_grammar,
                subset=['Avg Grammar']
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
            
            # Quality and Grammar by folder (UPDATED)
            quality_vals = [float(q.rstrip('%')) for q in folder_df['Avg Quality']]
            grammar_vals = []
            for g in folder_df['Avg Grammar']:
                if g != "—":
                    grammar_vals.append(float(g.rstrip('%')))
                else:
                    grammar_vals.append(0)
            
            bar_width = 0.35
            bars5 = ax3.bar(x - bar_width/2, quality_vals, bar_width, label='Quality', alpha=0.8)
            bars6 = ax3.bar(x + bar_width/2, grammar_vals, bar_width, label='Grammar', alpha=0.8)
            
            # Color quality bars
            for bar, q in zip(bars5, quality_vals):
                if q >= 80:
                    bar.set_color('green')
                elif q >= 60:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            # Color grammar bars
            for bar, g in zip(bars6, grammar_vals):
                if g >= 90:
                    bar.set_color('green')
                elif g >= 70:
                    bar.set_color('orange')
                elif g > 0:
                    bar.set_color('red')
                else:
                    bar.set_color('gray')
            
            ax3.set_xlabel('Folder')
            ax3.set_ylabel('Score (%)')
            ax3.set_title('Average Quality and Grammar Scores by Folder')
            ax3.set_xticks(x)
            ax3.set_xticklabels(folders)
            ax3.set_ylim(0, 100)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Combined metric radar chart (UPDATED with Grammar)
            categories = ['GZ Reduction', 'SP Reduction', 'ZS GZ', 'ZS SP', 'Quality', 'Grammar']
            
            for i, folder_row in folder_df.iterrows():
                # Normalize values for radar (0-1 scale)
                values = [
                    max(0, -folder_row['Avg Δ GZ'] / 0.5),  # Normalize reduction (0.5 = max expected)
                    max(0, -folder_row['Avg Δ SP'] / 0.5),
                    zs_gz_vals[i] / 100,
                    zs_sp_vals[i] / 100,
                    quality_vals[i] / 100,
                    grammar_vals[i] / 100,  # NEW
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
    
    with tab5:
        st.subheader("📊 Score & Word-count Distributions")

        with st.expander("ℹ️ How to read these charts", expanded=False):
            st.markdown(
                """
                • **Overall charts** — distribution of *all* drafts in the folder  
                • **Per-model / mode charts** — one histogram per model for *Doc* and *Para* mode  
                • **Word-count Δ charts** — histogram of the change in word-count (after − before)  
                Red dashed vertical line = average baseline detector score.  
                Black solid vertical line = no word-count change.
                """
            )

        # ── collect data ─────────────────────────────────────────────
        by_model_mode_folder = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: {"gz": [], "sp": [], "wc": []}))
        )
        folder_baselines = defaultdict(lambda: {"gz": [], "sp": []})

        for doc in docs:
            folder = doc.get("folder", "unknown")
            runs   = doc.get("runs", [])
            if not runs:
                continue

            first = runs[0]
            if "scores_before" in first and "group_doc" in first["scores_before"]:
                folder_baselines[folder]["gz"].append(first["scores_before"]["group_doc"]["gptzero"])
                folder_baselines[folder]["sp"].append(first["scores_before"]["group_doc"]["sapling"])

            for dr in runs:
                if "scores_after" not in dr or "group_doc" not in dr["scores_after"]:
                    continue
                model = dr.get("model", "unknown")
                mode  = dr.get("mode",  "unknown")
                by_model_mode_folder[folder][model][mode]["gz"].append(
                    dr["scores_after"]["group_doc"]["gptzero"]
                )
                by_model_mode_folder[folder][model][mode]["sp"].append(
                    dr["scores_after"]["group_doc"]["sapling"]
                )
                # word-count delta
                if "wordcount_after" in dr and "wordcount_before" in dr:
                    by_model_mode_folder[folder][model][mode]["wc"].append(
                        dr["wordcount_after"] - dr["wordcount_before"]
                    )

        # ── plotting ─────────────────────────────────────────────────
        for folder in ["ai_texts", "human_texts", "ai_paras", "human_paras"]:
            if folder not in by_model_mode_folder:
                continue

            st.markdown(f"### 📁 {folder.replace('_', ' ').title()}")
            base_gz = np.mean(folder_baselines[folder]["gz"]) if folder_baselines[folder]["gz"] else None
            base_sp = np.mean(folder_baselines[folder]["sp"]) if folder_baselines[folder]["sp"] else None

            # ----------  A. folder-level detector distributions  ----------
            with st.expander("Overall detector-score distributions", expanded=False):
                for detector in ("gz", "sp"):
                    fig, (ax_doc, ax_para) = plt.subplots(1, 2, figsize=(12, 4))
                    for model, modes in by_model_mode_folder[folder].items():
                        if modes["doc"][detector]:
                            safe_hist(ax_doc, modes["doc"][detector], bins=20, alpha=0.4, label=model)
                        if modes["para"][detector]:
                            safe_hist(ax_para, modes["para"][detector], bins=20, alpha=0.4, label=model)

                    bl = base_gz if detector == "gz" else base_sp
                    if bl is not None:
                        for ax in (ax_doc, ax_para):
                            ax.axvline(bl, color="red", linestyle="--", alpha=0.7, label=f"Baseline {bl:.3f}")

                    title = "GPTZero" if detector == "gz" else "Sapling"
                    ax_doc.set_title(f"{title} – Document mode")
                    ax_para.set_title(f"{title} – Paragraph mode")
                    for ax in (ax_doc, ax_para):
                        ax.set_xlabel("Score")
                        ax.set_ylabel("Drafts")
                        ax.grid(True, alpha=0.3)
                    ax_doc.legend(fontsize="small")
                    plt.tight_layout()
                    st.pyplot(fig)

            # ----------  B. per-model / mode detector distributions ----------
            with st.expander("Per-model / mode detector-score distributions", expanded=False):
                for model, modes in by_model_mode_folder[folder].items():
                    for mode_key in ("doc", "para"):
                        scores_gz = modes[mode_key]["gz"]
                        scores_sp = modes[mode_key]["sp"]
                        if not scores_gz and not scores_sp:
                            continue

                        st.markdown(f"**{model}** – {mode_key.title()} mode")
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                        # GPTZero
                        if scores_gz:
                            safe_hist(ax1, scores_gz, bins=20, alpha=0.7)
                        if base_gz is not None:
                            ax1.axvline(base_gz, color="red", linestyle="--", alpha=0.7, label=f"Baseline {base_gz:.3f}")
                        ax1.set_xlabel("GPTZero Score")
                        ax1.set_ylabel("Drafts")
                        ax1.grid(True, alpha=0.3)
                        ax1.legend()

                        # Sapling
                        if scores_sp:
                            safe_hist(ax2, scores_sp, bins=20, alpha=0.7)
                        if base_sp is not None:
                            ax2.axvline(base_sp, color="red", linestyle="--", alpha=0.7, label=f"Baseline {base_sp:.3f}")
                        ax2.set_xlabel("Sapling Score")
                        ax2.set_ylabel("Drafts")
                        ax2.grid(True, alpha=0.3)
                        ax2.legend()

                        plt.tight_layout()
                        st.pyplot(fig)

            # ----------  C. per-model / mode word-count-Δ distributions ----------
            with st.expander("Per-model / mode Word-count Δ distributions", expanded=False):
                for model, modes in by_model_mode_folder[folder].items():
                    for mode_key in ("doc", "para"):
                        wc_deltas = modes[mode_key]["wc"]
                        if not wc_deltas:
                            continue

                        st.markdown(f"**{model}** – {mode_key.title()} mode")
                        fig, ax = plt.subplots(figsize=(8, 4))
                        safe_hist(ax, wc_deltas, bins=30, alpha=0.75)
                        ax.axvline(0, color="black", linewidth=1.2, label="No change")
                        ax.set_xlabel("Word-count Δ (after − before)")
                        ax.set_ylabel("Drafts")
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig)

            st.divider()
                
    with tab6:
        st.subheader("📄 Document List")
        
        with st.expander("ℹ️ About documents", expanded=False):
            st.markdown("""
            Click "View" to see detailed results for any document, including:
            - All humanized drafts
            - Paragraph-by-paragraph analysis
            - Quality check results
            - Score comparisons
            
            Documents marked with ❌ failed processing and have no results.
            """)
        
        # Only show view buttons for single run mode
        show_view_buttons = not is_merged
        
        # Group documents by folder and status
        groups: DefaultDict[str, List[Dict]] = defaultdict(list)
        for d in docs:
            groups[d.get("folder", "(unknown)")].append(d)
        
        for folder in ["ai_texts", "human_texts", "ai_paras", "human_paras"]:
            if folder in groups:
                folder_docs = groups[folder]
                successful = sum(1 for d in folder_docs if d.get("runs"))
                failed = len(folder_docs) - successful
                
                with st.expander(f"📁 {folder.replace('_', ' ').title()} ({successful} ✅, {failed} ❌)", 
                               expanded=(folder == "ai_texts")):
                    for i, doc in enumerate(sorted(folder_docs, key=lambda x: x["document"])):
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col1:
                            if doc.get("runs"):
                                st.success("✅")
                            else:
                                st.error("❌")
                        with col2:
                            st.text(doc["document"])
                            if doc.get("error"):
                                st.caption(f"Error: {doc['error']}")
                            elif doc.get("warning"):
                                st.caption(f"Warning: {doc['warning']}")
                        with col3:
                            if show_view_buttons and doc.get("runs"):
                                if st.button("View", key=f"view_{folder}_{i}"):
                                    qp_set(run=run_name, view="doc", doc=doc["document"])
                                    st.rerun()
                            elif doc.get("runs"):
                                st.caption("(View in single run mode)")
                            else:
                                st.button("View", key=f"view_{folder}_{i}", disabled=True)


# ──────────────────────────────────────────────────────────────────────────
def _page_document(run_id: str, docs: List[Dict], doc_name: str):
    """Enhanced document detail page with colored metrics and grammar info"""
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

    # Check if document has results
    if not doc.get("runs"):
        st.error("This document failed processing and has no results.")
        if doc.get("error"):
            st.error(f"Error: {doc['error']}")
        if doc.get("warning"):
            st.warning(f"Warning: {doc['warning']}")
        return

    # Document metadata
    para_total = doc["paragraph_count"]
    baseline_wc = next((r.get('wordcount_before', 0) for r in doc['runs'] if r.get('mode')=='doc'), 0)
    baseline_gz = next((r['scores_before']['group_doc']['gptzero'] for r in doc['runs'] if 'scores_before' in r), 0)
    baseline_sp = next((r['scores_before']['group_doc']['sapling'] for r in doc['runs'] if 'scores_before' in r), 0)
    
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
        model = dr.get("model", "unknown")
        mode = dr.get("mode", "unknown")
        by_model[model][mode].append(dr)

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
                valid_drafts = [d for d in model_drafts if "scores_after" in d and "group_doc" in d["scores_after"]]
                
                if valid_drafts:
                    avg_gz = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in valid_drafts])
                    avg_sp = np.mean([d["scores_after"]["group_doc"]["sapling"] for d in valid_drafts])
                    avg_wc_delta = np.mean([d.get("wordcount_after", 0) - d.get("wordcount_before", 0) for d in valid_drafts])
                    zero_shot_gz = sum(1 for d in valid_drafts if d["scores_after"]["group_doc"]["gptzero"] <= ZERO_SHOT_THRESHOLD)
                    zero_shot_sp = sum(1 for d in valid_drafts if d["scores_after"]["group_doc"]["sapling"] <= ZERO_SHOT_THRESHOLD)
                    
                    # Calculate average grammar score (NEW)
                    grammar_scores = [d.get("flag_counts", {}).get("grammar_score") for d in valid_drafts if d.get("flag_counts", {}).get("grammar_score") is not None]
                    avg_grammar = np.mean(grammar_scores) if grammar_scores else None
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        colored_metric("Avg GPTZero", f"{avg_gz:.3f}", avg_gz - baseline_gz)
                    with col2:
                        colored_metric("Avg Sapling", f"{avg_sp:.3f}", avg_sp - baseline_sp)
                    with col3:
                        st.metric("Avg WC Δ", f"{avg_wc_delta:+.0f}")
                    with col4:
                        st.metric("Zero-shot", f"GZ:{zero_shot_gz}/{len(valid_drafts)} SP:{zero_shot_sp}/{len(valid_drafts)}")
                    with col5:
                        if avg_grammar is not None:
                            st.metric("Avg Grammar", f"{avg_grammar * 10:.0f}%")
                        else:
                            st.metric("Avg Grammar", "—")
                
                # Individual drafts
                for dr in sorted(model_drafts, key=lambda x: x.get("iter", 0)):
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
                valid_drafts = [d for d in model_drafts if "scores_after" in d and "group_doc" in d["scores_after"]]
                
                if valid_drafts:
                    avg_gz = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in valid_drafts])
                    avg_sp = np.mean([d["scores_after"]["group_doc"]["sapling"] for d in valid_drafts])
                    avg_wc_delta = np.mean([d.get("wordcount_after", 0) - d.get("wordcount_before", 0) for d in valid_drafts])
                    zero_shot_gz = sum(1 for d in valid_drafts if d["scores_after"]["group_doc"]["gptzero"] <= ZERO_SHOT_THRESHOLD)
                    zero_shot_sp = sum(1 for d in valid_drafts if d["scores_after"]["group_doc"]["sapling"] <= ZERO_SHOT_THRESHOLD)
                    
                    # Calculate average grammar score (NEW)
                    grammar_scores = [d.get("flag_counts", {}).get("grammar_score") for d in valid_drafts if d.get("flag_counts", {}).get("grammar_score") is not None]
                    avg_grammar = np.mean(grammar_scores) if grammar_scores else None
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        colored_metric("Avg GPTZero", f"{avg_gz:.3f}", avg_gz - baseline_gz)
                    with col2:
                        colored_metric("Avg Sapling", f"{avg_sp:.3f}", avg_sp - baseline_sp)
                    with col3:
                        st.metric("Avg WC Δ", f"{avg_wc_delta:+.0f}")
                    with col4:
                        st.metric("Zero-shot", f"GZ:{zero_shot_gz}/{len(valid_drafts)} SP:{zero_shot_sp}/{len(valid_drafts)}")
                    with col5:
                        if avg_grammar is not None:
                            st.metric("Avg Grammar", f"{avg_grammar * 10:.0f}%")
                        else:
                            st.metric("Avg Grammar", "—")
                
                # Individual drafts
                for dr in sorted(model_drafts, key=lambda x: x.get("iter", 0)):
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
            - **Grammar**: Average grammatical correctness score
            """)
        
        # Prepare comparison data
        comparison_data = []
        for model in sorted(by_model):
            for mode in ["doc", "para"]:
                drafts = by_model[model][mode]
                valid_drafts = [d for d in drafts if "scores_after" in d and "group_doc" in d["scores_after"]]
                
                if valid_drafts:
                    avg_gz = np.mean([d["scores_after"]["group_doc"]["gptzero"] for d in valid_drafts])
                    avg_sp = np.mean([d["scores_after"]["group_doc"]["sapling"] for d in valid_drafts])
                    avg_wc = np.mean([d.get("wordcount_after", 0) - d.get("wordcount_before", 0) for d in valid_drafts])
                    
                    # Count zero-shot successes
                    zero_shot_gz = sum(1 for d in valid_drafts if d["scores_after"]["group_doc"]["gptzero"] <= ZERO_SHOT_THRESHOLD)
                    zero_shot_sp = sum(1 for d in valid_drafts if d["scores_after"]["group_doc"]["sapling"] <= ZERO_SHOT_THRESHOLD)
                    
                    # Calculate average quality
                    quality_scores = []
                    grammar_scores = []
                    for d in valid_drafts:
                        if not d.get("para_mismatch", False) and d.get("flag_counts"):
                            score = sum(d["flag_counts"].get(f, 0) for f in GEMINI_FLAGS) / (len(GEMINI_FLAGS) * para_total) * 100
                            quality_scores.append(score)
                            
                            # Grammar score
                            gs = d["flag_counts"].get("grammar_score")
                            if gs is not None:
                                grammar_scores.append(gs)
                    
                    avg_quality = np.mean(quality_scores) if quality_scores else 0
                    avg_grammar = np.mean(grammar_scores) if grammar_scores else None
                    
                    comparison_data.append({
                        "Model": model,
                        "Mode": mode.title(),
                        "Drafts": len(valid_drafts),
                        "Avg GPTZero": f"{avg_gz:.3f}",
                        "Δ GZ": avg_gz - baseline_gz,
                        "Zero-shot GZ": f"{zero_shot_gz}/{len(valid_drafts)}",
                        "Avg Sapling": f"{avg_sp:.3f}",
                        "Δ SP": avg_sp - baseline_sp,
                        "Zero-shot SP": f"{zero_shot_sp}/{len(valid_drafts)}",
                        "Avg WC Δ": f"{avg_wc:+.0f}",
                        "Avg Quality": f"{avg_quality:.1f}%",
                        "Avg Grammar": f"{avg_grammar * 10:.0f}%" if avg_grammar is not None else "—"
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
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
            ).applymap(
                _style_grammar, subset=['Avg Grammar']
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
                    doc_drafts = [d for d in by_model[model]["doc"] if "scores_after" in d]
                    para_drafts = [d for d in by_model[model]["para"] if "scores_after" in d]
                    
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
                    doc_drafts = [d for d in by_model[model]["doc"] if "scores_after" in d]
                    para_drafts = [d for d in by_model[model]["para"] if "scores_after" in d]
                    
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
                st.markdown("#### Quality & Grammar Scores by Model & Mode")
                fig, ax = plt.subplots(figsize=(8, 6))
                
                doc_quality = []
                doc_grammar = []
                para_quality = []
                para_grammar = []
                
                for model in models:
                    # Doc mode
                    doc_drafts = by_model[model]["doc"]
                    doc_q_scores = []
                    doc_g_scores = []
                    for d in doc_drafts:
                        if not d.get("para_mismatch", False) and d.get("flag_counts"):
                            score = sum(d["flag_counts"].get(f, 0) for f in GEMINI_FLAGS) / (len(GEMINI_FLAGS) * para_total) * 100
                            doc_q_scores.append(score)
                            
                            gs = d["flag_counts"].get("grammar_score")
                            if gs is not None:
                                doc_g_scores.append(gs * 10)  # Convert to %
                    
                    doc_quality.append(np.mean(doc_q_scores) if doc_q_scores else 0)
                    doc_grammar.append(np.mean(doc_g_scores) if doc_g_scores else 0)
                    
                    # Para mode
                    para_drafts = by_model[model]["para"]
                    para_q_scores = []
                    para_g_scores = []
                    for d in para_drafts:
                        if not d.get("para_mismatch", False) and d.get("flag_counts"):
                            score = sum(d["flag_counts"].get(f, 0) for f in GEMINI_FLAGS) / (len(GEMINI_FLAGS) * para_total) * 100
                            para_q_scores.append(score)
                            
                            gs = d["flag_counts"].get("grammar_score")
                            if gs is not None:
                                para_g_scores.append(gs * 10)  # Convert to %
                    
                    para_quality.append(np.mean(para_q_scores) if para_q_scores else 0)
                    para_grammar.append(np.mean(para_g_scores) if para_g_scores else 0)
                
                # Create grouped bar chart
                bar_width = 0.2
                r1 = x - bar_width * 1.5
                r2 = x - bar_width * 0.5
                r3 = x + bar_width * 0.5
                r4 = x + bar_width * 1.5
                
                ax.bar(r1, doc_quality, bar_width, label='Doc Quality', color='blue', alpha=0.8)
                ax.bar(r2, doc_grammar, bar_width, label='Doc Grammar', color='cyan', alpha=0.8)
                ax.bar(r3, para_quality, bar_width, label='Para Quality', color='green', alpha=0.8)
                ax.bar(r4, para_grammar, bar_width, label='Para Grammar', color='lime', alpha=0.8)
                
                ax.set_xlabel('Model')
                ax.set_ylabel('Score (%)')
                ax.set_title('Quality and Grammar Scores by Model and Mode')
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
            doc_drafts = [d for d in by_model[model]["doc"] if "scores_after" in d]
            doc_drafts = sorted(doc_drafts, key=lambda x: x.get("iter", 0))
            if doc_drafts:
                iterations = [d.get("iter", 0) + 1 for d in doc_drafts]
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
            para_drafts = [d for d in by_model[model]["para"] if "scores_after" in d]
            para_drafts = sorted(para_drafts, key=lambda x: x.get("iter", 0))
            if para_drafts:
                iterations = [d.get("iter", 0) + 1 for d in para_drafts]
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
            doc_drafts = [d for d in by_model[model]["doc"] if "scores_after" in d]
            doc_drafts = sorted(doc_drafts, key=lambda x: x.get("iter", 0))
            if doc_drafts:
                iterations = [d.get("iter", 0) + 1 for d in doc_drafts]
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
            para_drafts = [d for d in by_model[model]["para"] if "scores_after" in d]
            para_drafts = sorted(para_drafts, key=lambda x: x.get("iter", 0))
            if para_drafts:
                iterations = [d.get("iter", 0) + 1 for d in para_drafts]
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
                    if not draft.get("para_mismatch", False) and draft.get("flag_counts"):
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
        
        # Grammar progression chart (NEW)
        st.markdown("#### Grammar Score Progression")
        st.caption("Shows how grammar scores vary across iterations")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Doc mode grammar
        for model in sorted(by_model):
            doc_drafts = [d for d in by_model[model]["doc"] if "scores_after" in d and not d.get("para_mismatch", False)]
            doc_drafts = sorted(doc_drafts, key=lambda x: x.get("iter", 0))
            
            grammar_scores = []
            iterations = []
            for d in doc_drafts:
                gs = d.get("flag_counts", {}).get("grammar_score")
                if gs is not None:
                    grammar_scores.append(gs * 10)  # Convert to %
                    iterations.append(d.get("iter", 0) + 1)
            
            if grammar_scores:
                ax1.plot(iterations, grammar_scores, marker='o', label=model)
        
        ax1.axhline(90, color='green', linestyle=':', alpha=0.5, label='Excellent (90%)')
        ax1.axhline(70, color='orange', linestyle=':', alpha=0.5, label='Good (70%)')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Grammar Score (%)')
        ax1.set_title('Document Mode - Grammar Score Progression')
        ax1.set_ylim(0, 100)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Para mode grammar
        for model in sorted(by_model):
            para_drafts = [d for d in by_model[model]["para"] if "scores_after" in d and not d.get("para_mismatch", False)]
            para_drafts = sorted(para_drafts, key=lambda x: x.get("iter", 0))
            
            grammar_scores = []
            iterations = []
            for d in para_drafts:
                gs = d.get("flag_counts", {}).get("grammar_score")
                if gs is not None:
                    grammar_scores.append(gs * 10)  # Convert to %
                    iterations.append(d.get("iter", 0) + 1)
            
            if grammar_scores:
                ax2.plot(iterations, grammar_scores, marker='o', label=model)
        
        ax2.axhline(90, color='green', linestyle=':', alpha=0.5, label='Excellent (90%)')
        ax2.axhline(70, color='orange', linestyle=':', alpha=0.5, label='Good (70%)')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Grammar Score (%)')
        ax2.set_title('Paragraph Mode - Grammar Score Progression')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)