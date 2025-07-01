# src/pages/utils.py
from __future__ import annotations

import re
import time
import threading
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlencode

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import ZERO_SHOT_THRESHOLD

# ───────────────────────── Gemini validation flags ─────────────────────────
GEMINI_FLAGS = [
    "length_ok",
    "same_meaning",
    "same_lang",
    "no_missing_info",
    "citation_preserved",
    "citation_content_ok",
]

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

# ───────────────────────── histogram safety wrapper ────────────────────────
def safe_hist(ax, data, bins=20, **kwargs):
    """
    Safely plot histogram even with identical values.
    Falls back to single bar if all values are the same.
    """
    if not data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    
    data_range = max(data) - min(data)
    if data_range == 0:
        # All values identical
        value = data[0]
        width = abs(value) * 0.1 if value != 0 else 1
        ax.bar([value], [len(data)], width=width, **kwargs)
        ax.set_xlim(value - width * 2, value + width * 2)
    else:
        # Normal histogram
        ax.hist(data, bins=bins, **kwargs)

# ─────────────────────── query-parameter helpers ─────────────────────────
def qp_get(key: str, default: Any = None) -> Any:
    """Read value from query-params; returns default if missing."""
    qp = st.query_params
    if key in qp:
        val = qp[key]
        # Handle various types
        if val == "None" or val == "null":
            return None
        if val == "True":
            return True
        if val == "False":
            return False
        # Try to parse as int if it looks numeric
        if val and isinstance(val, str) and val.isdigit():
            try:
                return int(val)
            except:
                pass
        return val
    return default

def qp_set(**kwargs):
    """Update multiple query params at once."""
    current = dict(st.query_params)
    current.update(kwargs)
    # Clean up None values
    current = {k: v for k, v in current.items() if v is not None}
    st.query_params.update(current)

# ─────────────────────── document listing ───────────────────────────────
def list_docx_files(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.docx"), key=natural_key)

# ─────────────────────── colored metric helper ───────────────────────────
def colored_metric(label: str, value: str, delta: float = None, help_text: str = None):
    """Display a metric with colored delta indicator."""
    if delta is not None:
        if delta < 0:
            st.metric(label, value, f"{delta:.3f}", delta_color="inverse", help=help_text)
        elif delta > 0:
            st.metric(label, value, f"+{delta:.3f}", delta_color="normal", help=help_text)
        else:
            st.metric(label, value, "0.000", help=help_text)
    else:
        st.metric(label, value, help=help_text)

# ─────────────────────── draft renderer ───────────────────────────────────
def render_draft(draft: Dict, para_total: int, doc_name: str, model: str):
    """Render a single draft with enhanced UI including grammar info."""
    iter_num = draft.get("iter", 0) + 1
    mode = draft.get("mode", "unknown")
    
    # Get scores
    gz_before = draft["scores_before"]["group_doc"]["gptzero"]
    sp_before = draft["scores_before"]["group_doc"]["sapling"]
    gz_after = draft["scores_after"]["group_doc"]["gptzero"]
    sp_after = draft["scores_after"]["group_doc"]["sapling"]
    
    wc_before = draft.get("wordcount_before", 0)
    wc_after = draft.get("wordcount_after", 0)
    wc_delta = wc_after - wc_before
    
    # Check for mismatch
    para_mismatch = draft.get("para_mismatch", False)
    
    # Draft header with iteration info
    expander_title = f"🔄 Iteration {iter_num} • {mode.title()} mode"
    if gz_after <= ZERO_SHOT_THRESHOLD or sp_after <= ZERO_SHOT_THRESHOLD:
        expander_title += " • ✨ Zero-shot!"
    
    with st.expander(expander_title, expanded=True):
        # Key metrics
        cols = st.columns([2, 2, 2, 2, 2])
        
        with cols[0]:
            colored_metric("GPTZero", f"{gz_after:.3f}", gz_after - gz_before)
            if gz_after <= ZERO_SHOT_THRESHOLD:
                st.success(f"✅ Zero-shot!")
        
        with cols[1]:
            colored_metric("Sapling", f"{sp_after:.3f}", sp_after - sp_before)
            if sp_after <= ZERO_SHOT_THRESHOLD:
                st.success(f"✅ Zero-shot!")
        
        with cols[2]:
            st.metric("Word Count Δ", f"{wc_delta:+d}")
            st.caption(f"{wc_before} → {wc_after}")
        
        with cols[3]:
            # Quality score calculation with grammar
            if not para_mismatch and draft.get("flag_counts"):
                # Calculate boolean quality (existing logic)
                bool_quality = sum(draft["flag_counts"].get(f, 0) for f in GEMINI_FLAGS) / (para_total * len(GEMINI_FLAGS)) * 100
                
                # Get grammar score if available
                grammar_score = draft["flag_counts"].get("grammar_score")
                
                if grammar_score is not None:
                    # Composite score: 80% boolean checks + 20% grammar
                    composite_quality = bool_quality * 0.8 + (grammar_score / 10) * 20
                    st.metric("Quality", f"{composite_quality:.1f}%")
                    # Convert grammar score to percentage
                    grammar_pct = grammar_score * 10
                    st.caption(f"Bool: {bool_quality:.0f}% | Grammar: {grammar_pct:.0f}%")
                else:
                    # Old runs without grammar
                    st.metric("Quality", f"{bool_quality:.1f}%")
                    st.caption("Legacy (no grammar)")
            else:
                st.metric("Quality", "—")
                if para_mismatch:
                    st.caption("Para mismatch")
        
        with cols[4]:
            para_count_after = draft.get("para_count_after", 0)
            if para_count_after != para_total:
                st.metric("Paragraphs", f"{para_count_after}", f"{para_count_after - para_total:+d}")
                st.warning("⚠️ Mismatch")
            else:
                st.metric("Paragraphs", f"{para_count_after}")
                st.success("✅ Match")
        
        # Quality breakdown
        if draft.get("flag_counts") and not para_mismatch:
            st.divider()
            st.markdown("##### Quality Breakdown")
            
            # Boolean flags
            flag_cols = st.columns(len(GEMINI_FLAGS))
            for idx, flag in enumerate(GEMINI_FLAGS):
                count = draft["flag_counts"].get(flag, 0)
                pct = (count / para_total * 100) if para_total else 0
                with flag_cols[idx]:
                    flag_name = flag.replace('_', ' ').title()
                    if pct >= 90:
                        st.success(f"**{flag_name}**\n{count}/{para_total} ({pct:.0f}%)")
                    elif pct >= 70:
                        st.warning(f"**{flag_name}**\n{count}/{para_total} ({pct:.0f}%)")
                    else:
                        st.error(f"**{flag_name}**\n{count}/{para_total} ({pct:.0f}%)")
            
            # Grammar section (if available)
            grammar_errors = draft["flag_counts"].get("grammar_errors", [])
            if grammar_errors:
                st.divider()
                st.markdown("##### ⚠️ Grammar Issues")
                error_summary = f"Found {len(grammar_errors)} grammar issue{'s' if len(grammar_errors) != 1 else ''}"
                st.warning(error_summary)
                
                # Show errors in columns for better layout
                error_cols = st.columns(2)
                for idx, error in enumerate(grammar_errors):
                    with error_cols[idx % 2]:
                        st.markdown(f"• {error}")
        
        # Paragraph details
        if draft.get("paragraph_details") and not para_mismatch:
            st.divider()
            st.markdown("##### 📊 Paragraph-by-paragraph analysis")
            
            para_data = []
            for detail in draft["paragraph_details"]:
                # Extract grammar info
                grammar_score = detail.get("grammar_score")
                grammar_errors = detail.get("grammar_errors", [])
                
                para_info = {
                    "Para #": detail["paragraph"],
                    "WC Before": detail["wc_before"],
                    "WC After": detail["wc_after"],
                    "WC Δ": detail["wc_after"] - detail["wc_before"],
                    "GZ Before": f"{detail['ai_before']['gptzero']:.3f}" if detail['ai_before'].get('gptzero') is not None else "—",
                    "GZ After": f"{detail['ai_after']['gptzero']:.3f}" if detail['ai_after'].get('gptzero') is not None else "—",
                    "SP Before": f"{detail['ai_before']['sapling']:.3f}" if detail['ai_before'].get('sapling') is not None else "—",
                    "SP After": f"{detail['ai_after']['sapling']:.3f}" if detail['ai_after'].get('sapling') is not None else "—",
                }
                
                # Add quality flags
                for flag in GEMINI_FLAGS:
                    flag_name = flag.replace('_', ' ').title()
                    para_info[flag_name] = "✅" if detail["flags"].get(flag, False) else "❌"
                
                # Add grammar info if available (as percentage)
                if grammar_score is not None:
                    grammar_pct = grammar_score * 10
                    para_info["Grammar"] = f"{grammar_pct:.0f}%"
                    if grammar_errors:
                        para_info["Grammar Issues"] = f"{len(grammar_errors)} issue(s)"
                
                para_data.append(para_info)
            
            df = pd.DataFrame(para_data)
            
            # Apply styling
            def style_delta(val):
                if isinstance(val, (int, float)):
                    if val < 0:
                        return 'color: green; font-weight: bold'
                    elif val > 0:
                        return 'color: red; font-weight: bold'
                return ''
            
            def style_grammar(val):
                if isinstance(val, str) and val.endswith('%') and val != "—":
                    try:
                        pct = float(val.rstrip('%'))
                        if pct >= 90:
                            return 'color: green; font-weight: bold'
                        elif pct >= 70:
                            return 'color: orange'
                        elif pct < 50:
                            return 'color: red; font-weight: bold'
                    except:
                        pass
                return ''
            
            style_cols = ['GZ Before', 'GZ After', 'SP Before', 'SP After', 'WC Δ']
            flag_cols = [col for col in df.columns if col in [f.replace('_', ' ').title() for f in GEMINI_FLAGS]]
            
            styled_df = df.style
            
            # Apply delta styling
            if 'WC Δ' in df.columns:
                styled_df = styled_df.applymap(style_delta, subset=['WC Δ'])
            
            # Apply grammar styling
            if 'Grammar' in df.columns:
                styled_df = styled_df.applymap(style_grammar, subset=['Grammar'])
            
            # Apply flag styling
            for col in flag_cols:
                if col in df.columns:
                    styled_df = styled_df.applymap(
                        lambda x: 'background-color: #d4f8d4' if x == "✅" else 'background-color: #f8d4d4' if x == "❌" else '',
                        subset=[col]
                    )
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # View humanized text
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("##### 📝 Humanized text")
        with col2:
            # Download button
            fname = f"{doc_name}_{model}_iter{iter_num}_{mode}.txt"
            st.download_button(
                "📥 Download",
                draft.get("humanized_text", "No text available"),
                file_name=fname,
                mime="text/plain",
                key=f"dl_{doc_name}_{model}_{mode}_{iter_num}"
            )
        
        # Text area for viewing
        st.text_area(
            "Humanized version",
            draft.get("humanized_text", "No text available"),
            height=300,
            disabled=True,
            label_visibility="collapsed"
        )
        
        # Debug info at the bottom
        st.markdown("##### 🐛 Debug info")
        debug_data = {
            "model": model,
            "mode": mode,
            "iteration": iter_num,
            "scores_before": draft.get("scores_before"),
            "scores_after": draft.get("scores_after"),
            "flag_counts": draft.get("flag_counts"),
            "para_mismatch": draft.get("para_mismatch"),
            "para_count_before": draft.get("para_count_before"),
            "para_count_after": draft.get("para_count_after"),
        }
        st.json(debug_data)


# ─────────────────────── run name generator ───────────────────────────────
def generate_run_name(folders: List[str], models: List[str]) -> str:
    """Generate a descriptive run name based on folders and models."""
    import time
    
    # Shorten folder names
    folder_parts = []
    for f in folders[:2]:  # Max 2 folders in name
        if "ai_texts" in f:
            folder_parts.append("AI")
        elif "human_texts" in f:
            folder_parts.append("Human")
        elif "ai_paras" in f:
            folder_parts.append("AI-P")
        elif "human_paras" in f:
            folder_parts.append("Human-P")
        else:
            folder_parts.append(f[:6])
    
    # Shorten model names
    model_parts = []
    for m in models[:2]:  # Max 2 models in name
        if "gpt" in m.lower():
            model_parts.append("GPT")
        elif "claude" in m.lower():
            model_parts.append("Claude")
        elif "gemini" in m.lower():
            model_parts.append("Gemini")
        else:
            model_parts.append(m[:6])
    
    timestamp = time.strftime("%m%d_%H%M")
    
    folder_str = "+".join(folder_parts) if folder_parts else "NoFolder"
    model_str = "+".join(model_parts) if model_parts else "NoModel"
    
    if len(models) > 2:
        model_str += f"+{len(models)-2}more"
    
    return f"{folder_str}_{model_str}_{timestamp}"

# ─────────────────────── progress helpers ───────────────────────────────
def calculate_progress(log_history: List[str]) -> Dict[str, Any]:
    """Calculate progress from log history."""
    progress = {
        "current_doc": 0,
        "total_docs": 0,
        "current_phase": "Starting",
        "phase_progress": 0,
        "current_action": "",
    }
    
    # Parse log history
    for line in log_history:
        # Document progress
        if "Processing document:" in line:
            progress["current_doc"] += 1
        
        # Phase detection
        if "Phase 1: Generation" in line:
            progress["current_phase"] = "Phase 1: Generation"
        elif "Phase 2: Detector scoring" in line:
            progress["current_phase"] = "Phase 2: Detector scoring"
        elif "Phase 3: Gemini quality" in line:
            progress["current_phase"] = "Phase 3: Quality evaluation"
        elif "Phase 4: Assembly" in line:
            progress["current_phase"] = "Phase 4: Assembly"
        
        # Detailed progress
        if "Progress:" in line and "/" in line:
            try:
                parts = line.split("Progress:")[1].strip().split()
                if parts and "/" in parts[0]:
                    current, total = parts[0].split("/")
                    progress["phase_progress"] = int(current) / int(total)
            except:
                pass
        
        # Current action
        if "▶️" in line:
            progress["current_action"] = line.split("▶️")[1].strip()
    
    return progress

# ─────────────────────── natural sorting ─────────────────────────
_num = re.compile(r'(\d+)')

def natural_key(p: Path | str):
    """
    Split the filename into text/number chunks so that
    'AI_text_100.docx' > 'AI_text_11.docx' > 'AI_text_2.docx'.
    Works with Path objects *or* plain strings.
    """
    s = p.name if isinstance(p, Path) else str(p)
    return [int(tok) if tok.isdigit() else tok.lower() for tok in _num.split(s)]