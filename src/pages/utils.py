# src/pages/utils.py - Shared utilities and helpers
from __future__ import annotations

import time
import threading
import math
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.config import ZERO_SHOT_THRESHOLD

# ───────────────────────── Constants ───────────────────────
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

# ────────────────────────── query-param helpers ───────────────────────────
def qp_get(key: str, default=None):
    val = st.query_params.get(key, default)
    if isinstance(val, list):
        return val[0] if val else default
    return val

def qp_set(**kwargs):
    qp = dict(st.query_params)
    for k, v in kwargs.items():
        if v is None:
            qp.pop(k, None)
        else:
            qp[k] = v
    st.query_params = qp

# Histogram helper --------------------------------------------------------
def safe_hist(ax, data, *, bins: int = 20, **kwargs):
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

# ──────────────────────────────────────────────────────────────────────────
def render_draft(draft: Dict, para_total: int, doc_name: str, model: str):
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
    
    # Check zero-shot success
    gz_zeroshot = "✅" if sa['gptzero'] <= ZERO_SHOT_THRESHOLD else ""
    sp_zeroshot = "✅" if sa['sapling'] <= ZERO_SHOT_THRESHOLD else ""
    
    title = (
        f"{status_emoji} Draft {draft['iter']+1} | "
        f"GZ: {sa['gptzero']:.2f} ({gz_color}{gz_delta:+.2f}) {gz_zeroshot} | "
        f"SP: {sa['sapling']:.2f} ({sp_color}{sp_delta:+.2f}) {sp_zeroshot} | "
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
            
            # Zero-shot indicators
            if sa['gptzero'] <= ZERO_SHOT_THRESHOLD or sa['sapling'] <= ZERO_SHOT_THRESHOLD:
                st.success(f"✅ Zero-shot success: " + 
                          (f"GPTZero ({sa['gptzero']:.3f}) " if sa['gptzero'] <= ZERO_SHOT_THRESHOLD else "") +
                          (f"Sapling ({sa['sapling']:.3f})" if sa['sapling'] <= ZERO_SHOT_THRESHOLD else ""))
            
            # Detailed paragraph analysis
            if draft.get("paragraph_details"):
                st.markdown("### 📊 Paragraph-by-Paragraph Analysis")
                
                rows = []
                for p in draft["paragraph_details"]:
                    # Calculate paragraph quality score
                    para_quality = sum(1 for v in p["flags"].values() if v) / len(p["flags"]) * 100
                    
                    # Calculate deltas
                    gz_delta_para = p['ai_after']['gptzero'] - p['ai_before']['gptzero']
                    sp_delta_para = p['ai_after']['sapling'] - p['ai_before']['sapling']
                    
                    row = {
                        "¶": p["paragraph"],
                        "WC Δ": f"{p['wc_after'] - p['wc_before']:+d}",
                        "GZ Before": f"{p['ai_before']['gptzero']:.2f}",
                        "GZ After": f"{p['ai_after']['gptzero']:.2f}",
                        "GZ Δ": gz_delta_para,
                        "SP Before": f"{p['ai_before']['sapling']:.2f}",
                        "SP After": f"{p['ai_after']['sapling']:.2f}",
                        "SP Δ": sp_delta_para,
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
                    subset=['GZ Δ', 'SP Δ']
                ).applymap(
                    lambda x: 'background-color: #90EE90' if x == "✅" else 'background-color: #FFB6C1' if x == "❌" else '',
                    subset=[col for col in df.columns if col in ['Length OK', 'Same Meaning', 'Same Language', 'No Missing Info', 'Citation Preserved', 'Citation Content OK']]
                ).format({
                    'GZ Δ': '{:+.3f}',
                    'SP Δ': '{:+.3f}'
                })
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=min(400, 50 + len(rows) * 35)  # Dynamic height based on rows
                )