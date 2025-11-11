# src/pages/preview_results.py
"""
Preview Results - Simplified view for quick model comparison
Shows key metrics to identify best and poor models quickly
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

# ─────────────────── project root ────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ZERO_SHOT_THRESHOLD
from src.api_client import cached_list_runs, cached_get_run


def _calculate_model_stats(docs: List[Dict]) -> Dict[str, Dict]:
    """Calculate key statistics for each model from run data."""
    model_stats = {}

    for doc in docs:
        for run in doc.get("runs", []):
            model = run.get("model_label", "unknown")

            if model not in model_stats:
                model_stats[model] = {
                    "total_drafts": 0,
                    "gptzero_scores": [],
                    "sapling_scores": [],
                    "zero_shot_passes": 0,
                    "same_meaning_scores": [],
                    "grammar_scores": [],
                    "errors": 0,
                }

            stats = model_stats[model]
            stats["total_drafts"] += 1

            # Check for errors
            if run.get("error"):
                stats["errors"] += 1
                continue

            # GPTZero scores
            gptzero_after = run.get("gptzero_after")
            if gptzero_after is not None:
                stats["gptzero_scores"].append(gptzero_after)
                if gptzero_after <= ZERO_SHOT_THRESHOLD:
                    stats["zero_shot_passes"] += 1

            # Sapling scores
            sapling_after = run.get("sapling_after")
            if sapling_after is not None:
                stats["sapling_scores"].append(sapling_after)

            # Quality metrics
            quality = run.get("quality", {})
            if quality:
                same_meaning = quality.get("same_meaning", {}).get("level")
                if same_meaning is not None:
                    stats["same_meaning_scores"].append(same_meaning)

                grammar = quality.get("grammar", {}).get("level")
                if grammar is not None:
                    stats["grammar_scores"].append(grammar)

    # Calculate averages
    for model, stats in model_stats.items():
        stats["avg_gptzero"] = sum(stats["gptzero_scores"]) / len(stats["gptzero_scores"]) if stats["gptzero_scores"] else None
        stats["avg_sapling"] = sum(stats["sapling_scores"]) / len(stats["sapling_scores"]) if stats["sapling_scores"] else None
        stats["zero_shot_rate"] = (stats["zero_shot_passes"] / stats["total_drafts"] * 100) if stats["total_drafts"] > 0 else 0
        stats["avg_same_meaning"] = sum(stats["same_meaning_scores"]) / len(stats["same_meaning_scores"]) if stats["same_meaning_scores"] else None
        stats["avg_grammar"] = sum(stats["grammar_scores"]) / len(stats["grammar_scores"]) if stats["grammar_scores"] else None
        stats["error_rate"] = (stats["errors"] / stats["total_drafts"] * 100) if stats["total_drafts"] > 0 else 0

    return model_stats


def _render_model_card(model: str, stats: Dict, rank: int):
    """Render a model performance card with color-coded metrics."""
    # Determine overall performance
    gptzero_avg = stats.get("avg_gptzero", 1.0)
    zero_shot_rate = stats.get("zero_shot_rate", 0)
    error_rate = stats.get("error_rate", 0)

    # Color coding
    if error_rate > 10:
        card_color = "🔴"  # High error rate
        performance = "Poor"
    elif gptzero_avg is not None and gptzero_avg <= 0.15 and zero_shot_rate >= 70:
        card_color = "🟢"  # Excellent
        performance = "Excellent"
    elif gptzero_avg is not None and gptzero_avg <= 0.25 and zero_shot_rate >= 50:
        card_color = "🟡"  # Good
        performance = "Good"
    else:
        card_color = "🟠"  # Needs improvement
        performance = "Needs Improvement"

    with st.container():
        col1, col2, col3, col4, col5, col6 = st.columns([1, 3, 2, 2, 2, 2])

        with col1:
            st.markdown(f"### {card_color}")

        with col2:
            st.markdown(f"**#{rank}. {model}**")
            st.caption(f"Performance: {performance}")

        with col3:
            if gptzero_avg is not None:
                st.metric("Avg GPTZero", f"{gptzero_avg:.3f}")
            else:
                st.metric("Avg GPTZero", "N/A")

        with col4:
            st.metric("Zero-Shot Rate", f"{zero_shot_rate:.1f}%")

        with col5:
            same_meaning = stats.get("avg_same_meaning")
            if same_meaning is not None:
                st.metric("Meaning", f"{same_meaning:.1f}/10")
            else:
                st.metric("Meaning", "N/A")

        with col6:
            grammar = stats.get("avg_grammar")
            if grammar is not None:
                st.metric("Grammar", f"{grammar:.1f}/10")
            else:
                st.metric("Grammar", "N/A")

        st.divider()


def page_preview_results():
    """Preview results page for quick model comparison."""
    st.header("🔬 Preview Mode Results")

    st.markdown("""
    **Quick Model Screening:** This simplified view helps you identify the best and poorest performing models at a glance.

    - 🟢 **Excellent:** Low AI detection, high zero-shot rate
    - 🟡 **Good:** Decent performance, worth further testing
    - 🟠 **Needs Improvement:** Higher AI scores or lower success rates
    - 🔴 **Poor:** High error rates or very poor metrics
    """)

    st.divider()

    # Run selector
    try:
        runs = cached_list_runs()
    except Exception as e:
        st.error(f"Could not load runs: {e}")
        return

    if not runs:
        st.info("No benchmark runs found. Create one from the 'New Run' page.")
        return

    # Filter for preview mode runs if possible
    preview_runs = [r for r in runs if r.get("preview_mode", False)]
    all_run_names = [r["run_name"] for r in (preview_runs if preview_runs else runs)]

    if preview_runs:
        st.info(f"📋 Showing {len(preview_runs)} preview mode runs")

    selected_run = st.selectbox(
        "Select a benchmark run to analyze",
        all_run_names,
        help="Choose which benchmark run to view"
    )

    if not selected_run:
        return

    # Load run data
    try:
        with st.spinner("Loading run data..."):
            run_data = cached_get_run(selected_run)
    except Exception as e:
        st.error(f"Could not load run data: {e}")
        return

    if not run_data:
        st.error("Run data not found")
        return

    docs = run_data.get("docs", [])
    if not docs:
        st.warning("No documents found in this run")
        return

    # Calculate statistics
    model_stats = _calculate_model_stats(docs)

    if not model_stats:
        st.warning("No model statistics available")
        return

    # Sort models by performance (lower GPTZero avg = better)
    sorted_models = sorted(
        model_stats.items(),
        key=lambda x: (
            x[1].get("error_rate", 100),  # Prioritize low error rate
            x[1].get("avg_gptzero", 1.0),  # Then low GPTZero score
            -x[1].get("zero_shot_rate", 0)  # Then high zero-shot rate
        )
    )

    st.markdown(f"## 📊 Model Rankings ({len(sorted_models)} models tested)")
    st.caption("Models ranked by AI detection performance (lower is better)")

    # Display top performers
    st.markdown("### 🏆 Top Performers")
    for rank, (model, stats) in enumerate(sorted_models[:5], 1):
        _render_model_card(model, stats, rank)

    # Display bottom performers if more than 5 models
    if len(sorted_models) > 5:
        st.markdown("### ⚠️ Needs Improvement")
        for rank, (model, stats) in enumerate(sorted_models[-3:], len(sorted_models) - 2):
            _render_model_card(model, stats, rank)

    # Summary statistics table
    st.markdown("### 📈 Detailed Statistics")

    df_data = []
    for model, stats in sorted_models:
        df_data.append({
            "Model": model,
            "Drafts": stats["total_drafts"],
            "Avg GPTZero": f"{stats['avg_gptzero']:.3f}" if stats['avg_gptzero'] is not None else "N/A",
            "Zero-Shot %": f"{stats['zero_shot_rate']:.1f}%",
            "Avg Sapling": f"{stats['avg_sapling']:.3f}" if stats['avg_sapling'] is not None else "N/A",
            "Meaning (0-10)": f"{stats['avg_same_meaning']:.1f}" if stats['avg_same_meaning'] is not None else "N/A",
            "Grammar (0-10)": f"{stats['avg_grammar']:.1f}" if stats['avg_grammar'] is not None else "N/A",
            "Error Rate": f"{stats['error_rate']:.1f}%",
        })

    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Next steps
    st.markdown("### 🎯 Next Steps")
    st.info("""
    **For Top Performers:** Run a full benchmark with more documents and iterations to confirm results.

    **For Poor Performers:** Consider removing from further testing to save resources.
    """)

    # Button to full analysis
    if st.button("📊 View Detailed Analysis", type="primary"):
        st.session_state.page = "Benchmark Analysis"
        st.rerun()


# ──────────────────────────── Standalone Page Setup ────────────────────
if __name__ == "__main__":
    st.set_page_config(page_title="Preview Results", layout="wide")

    from src.pages._shared_layout import setup_sidebar
    setup_sidebar()

    page_preview_results()
