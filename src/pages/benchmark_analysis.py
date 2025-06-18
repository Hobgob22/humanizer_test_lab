# src/pages/benchmark_analysis.py
from __future__ import annotations

###############################################################################
#  Benchmark Analysis – extended metrics
#  • Per-flag quality columns (length_ok, same_meaning, …)
#  • Word-count-difference columns:
#        – Within 10 words %   – Within 20 words %
#        – % Longer            – % Shorter
#  • Per-folder word-count-delta histogram + summary
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

    folder_avg_baselines = {
        f: {
            "gptzero": np.mean([b["gptzero"] for b in bl]),
            "sapling": np.mean([b["sapling"] for b in bl]),
            "wordcount": np.mean([b["wordcount"] for b in bl]),
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
                "draft_count": 0,
                "mismatch_count": 0,
                "zs_hits": {"gptzero": 0, "sapling": 0},
            },
        )

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

        # quality flags (skip drafts with paragraph mismatch)
        if not dr.get("para_mismatch", False):
            total = dr.get("para_count_before", 1)
            for flag in _EXPECTED_FLAGS:
                cnt = dr.get("flag_counts", {}).get(flag, 0)
                bucket["quality_flags"][flag].append((cnt / total) * 100 if total else 0)

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

                after_gz = np.mean([s["gptzero"] for s in data["after_scores"]])
                after_sp = np.mean([s["sapling"] for s in data["after_scores"]])

                zs_gz_pct = data["zs_hits"]["gptzero"] / data["draft_count"] * 100
                zs_sp_pct = data["zs_hits"]["sapling"] / data["draft_count"] * 100

                # word-count diff metrics
                deltas = np.array(data["wc_deltas"])
                within10 = (np.abs(deltas) <= 10).mean() * 100
                within20 = (np.abs(deltas) <= 20).mean() * 100
                pct_longer = (deltas > 0).mean() * 100
                pct_shorter = (deltas < 0).mean() * 100

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
                    "wc_deltas": data["wc_deltas"],  # for histograms
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
                "drafts": 0,
                "zs_gz_hits": 0,
                "zs_sp_hits": 0,
                "folders": set(),
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
                bucket["sp_deltas"].append(s["deltas"]["sapling"])
                bucket["quality"].append(np.mean(list(s["quality"].values())))
                bucket["drafts"] += s["draft_count"]
                bucket["zs_gz_hits"] += s["zs_hits"]["gptzero"]
                bucket["zs_sp_hits"] += s["zs_hits"]["sapling"]
                bucket["folders"].add(folder)

    rows = []
    for model, modes in agg.items():
        for mode in ("doc", "para"):
            m = modes.get(mode)
            if not m or m["drafts"] == 0:
                continue
            rows.append(
                {
                    "Model": model,
                    "Mode": mode.title(),
                    "Total Drafts": m["drafts"],
                    "Avg Δ GZ": np.mean(m["gz_deltas"]),
                    "Avg Δ SP": np.mean(m["sp_deltas"]),
                    "Zero-shot GZ": f"{m['zs_gz_hits'] / m['drafts'] * 100:.1f}%",
                    "Zero-shot SP": f"{m['zs_sp_hits'] / m['drafts'] * 100:.1f}%",
                    "Avg Quality": f"{np.mean(m['quality']):.1f}%",
                    "Folders": len(m["folders"]),
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

    styled = (
        df.style.applymap(_style_delta, subset=["Avg Δ GZ", "Avg Δ SP"])
        .applymap(_style_zs, subset=["Zero-shot GZ", "Zero-shot SP"])
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
        "Quality %","Mismatch %",
    ]
    return pd.DataFrame(rows)[base_cols + qual_cols]

# ═══════════════ RUN OVERVIEW & DOC PAGE (main) ═══════════════════════
def page_runs() -> None:
    # --- run selection --------------------------------------------------------
    run_id = qp_get("run")
    doc_name = qp_get("doc")
    view = qp_get("view")

    runs_meta = list_runs()
    if not runs_meta:
        st.info("No benchmarks stored yet. Create a new run to get started!")
        return

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

    # --- overview header ------------------------------------------------------
    st.header(f"📊 Benchmark Analysis: **{run_id}**")

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 By Folder & Model",
            "📈 Model Performance",
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
                • **Within 10 / 20 words** – word-count distance from original  
                • **% Longer / % Shorter** – drafts that grew / shrank  
                • **Mismatch %** – paragraph-count mismatches
                """
            )

        # Folder ordering
        folder_order = ["ai_texts", "human_texts", "mixed_texts"]
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
                             ("Zero-shot GZ","Zero-shot SP","Quality %","Mismatch %")]
                styled_df = (
                    df.style.applymap(_style_delta, subset=["Δ GZ", "Δ SP"])
                    .applymap(_style_zs, subset=["Zero-shot GZ", "Zero-shot SP"])
                    .applymap(_style_quality, subset=qual_cols + ["Quality %"])
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

                # 2️⃣ Zero-shot & quality bars (unchanged)
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

                    # quality
                    qualities = [
                        float(df[df["Model"] == m]["Quality %"].iloc[0].rstrip("%"))
                        for m in models
                    ]
                    ax2.bar(
                        x,
                        qualities,
                        0.4,
                        color=["green" if q >= 80 else "orange" if q >= 60 else "red" for q in qualities],
                    )
                    ax2.set_ylim(0, 100)
                    ax2.set_xticks(x)
                    ax2.set_xticklabels(models, rotation=45, ha="right")
                    ax2.set_ylabel("%")
                    ax2.set_title("Average quality")
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)


    with tab2:
        st.subheader("📈 Model Performance")
        with st.expander("ℹ️ About this view", expanded=False):
            st.markdown(
                """
                Compare humanizer models on different document sets.  
                Lower **Δ GZ / Δ SP** values and higher **Zero-shot** rates are better.
                """
            )

        folder_order = ["ai_texts", "human_texts", "mixed_texts"]
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

        with st.expander("ℹ️ How to read these charts", expanded=False):
            st.markdown(
                """
                *Upper charts* – overall distribution for each **folder**  
                *Lower charts* – separate distribution for **each model / mode**  
                A red dashed line shows the *baseline* detector score before humanisation.
                """
            )

        # ── collect data ─────────────────────────────────────────────
        by_model_mode_folder = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: {"gz": [], "sp": []}))
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

        # ── plotting ─────────────────────────────────────────────────
        for folder in ["ai_texts", "human_texts", "mixed_texts"]:
            if folder not in by_model_mode_folder:
                continue

            st.markdown(f"### 📁 {folder.replace('_', ' ').title()}")
            base_gz = np.mean(folder_baselines[folder]["gz"]) if folder_baselines[folder]["gz"] else None
            base_sp = np.mean(folder_baselines[folder]["sp"]) if folder_baselines[folder]["sp"] else None

            # ----------  A. folder-level distributions  ----------
            with st.expander("Overall distribution charts", expanded=False):
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

            # ----------  B. per-model-per-mode distributions ----------
            with st.expander("Per-model / mode distribution charts", expanded=False):
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
            
            Documents marked with ❌ failed processing and have no results.
            """)
        
        # Group documents by folder and status
        groups: DefaultDict[str, List[Dict]] = defaultdict(list)
        for d in docs:
            groups[d.get("folder", "(unknown)")].append(d)
        
        for folder in ["ai_texts", "human_texts", "mixed_texts"]:
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
                            if doc.get("runs"):
                                if st.button("View", key=f"view_{folder}_{i}"):
                                    qp_set(run=run_id, view="doc", doc=doc["document"])
                                    st.rerun()
                            else:
                                st.button("View", key=f"view_{folder}_{i}", disabled=True)

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
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        colored_metric("Avg GPTZero", f"{avg_gz:.3f}", avg_gz - baseline_gz)
                    with col2:
                        colored_metric("Avg Sapling", f"{avg_sp:.3f}", avg_sp - baseline_sp)
                    with col3:
                        st.metric("Avg WC Δ", f"{avg_wc_delta:+.0f}")
                    with col4:
                        st.metric("Zero-shot", f"GZ:{zero_shot_gz}/{len(valid_drafts)} SP:{zero_shot_sp}/{len(valid_drafts)}")
                
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
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        colored_metric("Avg GPTZero", f"{avg_gz:.3f}", avg_gz - baseline_gz)
                    with col2:
                        colored_metric("Avg Sapling", f"{avg_sp:.3f}", avg_sp - baseline_sp)
                    with col3:
                        st.metric("Avg WC Δ", f"{avg_wc_delta:+.0f}")
                    with col4:
                        st.metric("Zero-shot", f"GZ:{zero_shot_gz}/{len(valid_drafts)} SP:{zero_shot_sp}/{len(valid_drafts)}")
                
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
                    for d in valid_drafts:
                        if not d.get("para_mismatch", False) and d.get("flag_counts"):
                            score = sum(d["flag_counts"].values()) / (len(GEMINI_FLAGS) * para_total) * 100
                            quality_scores.append(score)
                    avg_quality = np.mean(quality_scores) if quality_scores else 0
                    
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
                        "Avg Quality": f"{avg_quality:.1f}%"
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
                st.markdown("#### Quality Score by Model & Mode")
                fig, ax = plt.subplots(figsize=(8, 6))
                
                doc_quality = []
                para_quality = []
                
                for model in models:
                    # Doc mode quality
                    doc_drafts = by_model[model]["doc"]
                    doc_q_scores = []
                    for d in doc_drafts:
                        if not d.get("para_mismatch", False) and d.get("flag_counts"):
                            score = sum(d["flag_counts"].values()) / (len(GEMINI_FLAGS) * para_total) * 100
                            doc_q_scores.append(score)
                    doc_quality.append(np.mean(doc_q_scores) if doc_q_scores else 0)
                    
                    # Para mode quality
                    para_drafts = by_model[model]["para"]
                    para_q_scores = []
                    for d in para_drafts:
                        if not d.get("para_mismatch", False) and d.get("flag_counts"):
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