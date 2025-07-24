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
    Safely plot histogram even with identical values or None values.
    Falls back to single bar if all values are the same.
    Filters out None values from disabled detectors.
    """
    # Filter out None values (from disabled detectors)
    filtered_data = [x for x in data if x is not None]
    
    if not filtered_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    
    data_range = max(filtered_data) - min(filtered_data)
    if data_range == 0:
        # All values identical
        value = filtered_data[0]
        width = abs(value) * 0.1 if value != 0 else 1
        ax.bar([value], [len(filtered_data)], width=width, **kwargs)
        ax.set_xlim(value - width * 2, value + width * 2)
    else:
        # Normal histogram
        ax.hist(filtered_data, bins=bins, **kwargs)

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
    """
    Update query-string parameters.

    • Pass a value to **set / overwrite** a parameter  
    • Pass **None** to **remove** a parameter  

    Example  
    --------  
    >>> qp_set(view=None, doc=None)     # ← removes “view” and “doc”  
    >>> qp_set(run="latest")            # ← sets/updates “run”
    """
    qp = st.query_params

    for key, val in kwargs.items():
        if val is None:
            # Explicitly delete the key if it exists
            try:
                del qp[key]
            except KeyError:
                pass
        else:
            qp[key] = val

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
    
    # Check zero-shot success only for enabled detectors
    is_zero_shot = False
    if gz_after is not None and gz_after <= ZERO_SHOT_THRESHOLD:
        is_zero_shot = True
    elif sp_after is not None and sp_after <= ZERO_SHOT_THRESHOLD:
        is_zero_shot = True
    
    if is_zero_shot:
        expander_title += " • ✨ Zero-shot!"
    
    with st.expander(expander_title, expanded=True):
        # Key metrics
        cols = st.columns([2, 2, 2, 2, 2])
        
        with cols[0]:
            if gz_after is not None:
                colored_metric("GPTZero", f"{gz_after:.3f}", gz_after - gz_before)
                if gz_after <= ZERO_SHOT_THRESHOLD:
                    st.success(f"✅ Zero-shot!")
            else:
                st.metric("GPTZero", "N/A", help="GPTZero detector was disabled")
        
        with cols[1]:
            if sp_after is not None:
                colored_metric("Sapling", f"{sp_after:.3f}", sp_after - sp_before)
                if sp_after <= ZERO_SHOT_THRESHOLD:
                    st.success(f"✅ Zero-shot!")
            else:
                st.metric("Sapling", "N/A", help="Sapling detector was disabled")
        
        with cols[2]:
            st.metric("Word Count Δ", f"{wc_delta:+d}")
            st.caption(f"{wc_before} → {wc_after}")
        
        with cols[3]:
            # Quality score calculation with grammar
            if not para_mismatch and draft.get("flag_counts"):
                flag_counts = draft["flag_counts"]
                content_count = flag_counts.get("content_paragraph_count", para_total)
                total_segments = flag_counts.get("total_segments", para_total)
                
                # Calculate boolean quality (using content paragraphs only)
                bool_quality = sum(flag_counts.get(f, 0) for f in GEMINI_FLAGS) / (content_count * len(GEMINI_FLAGS)) * 100 if content_count > 0 else 0
                
                # Get grammar score if available
                grammar_score = flag_counts.get("grammar_score")
                
                if grammar_score is not None:
                    # Show grammar as 0-10 scale (not percentage)
                    st.metric("Quality", f"{bool_quality:.1f}%")
                    st.caption(f"Grammar: {grammar_score:.1f}/10")
                else:
                    # Old runs without grammar
                    st.metric("Quality", f"{bool_quality:.1f}%")
                    st.caption("Legacy (no grammar)")
                
                # Show content vs total paragraph structure
                if total_segments > content_count:
                    headings = total_segments - content_count
                    st.caption(f"{content_count} content + {headings} headings")
                else:
                    st.caption(f"{content_count} content paras")
            else:
                st.metric("Quality", "—")
                if para_mismatch:
                    st.caption("Para mismatch")
        
        with cols[4]:
            para_count_after = draft.get("para_count_after", 0)
            para_mismatch = draft.get("para_mismatch", False)
            mismatch_reason = draft.get("mismatch_reason")
            
            if para_mismatch:
                st.metric("Paragraphs", f"{para_count_after}", f"{para_count_after - para_total:+d}")
                st.warning("⚠️ Mismatch")
                if mismatch_reason:
                    st.caption(f"Reason: {mismatch_reason}")
            else:
                st.metric("Paragraphs", f"{para_count_after}")
                st.success("✅ Match")
        
        # Quality breakdown
        if draft.get("flag_counts") and not para_mismatch:
            st.divider()
            st.markdown("##### Quality Breakdown")
            
            flag_counts = draft["flag_counts"]
            content_count = flag_counts.get("content_paragraph_count", para_total)
            total_segments = flag_counts.get("total_segments", para_total)
            
            # Show paragraph structure info
            if total_segments > content_count:
                headings_count = total_segments - content_count
                st.info(f"📊 **Document structure:** {content_count} content paragraphs + {headings_count} headings = {total_segments} total segments")
                st.caption("Quality metrics are calculated only on content paragraphs (headings excluded)")
            
            # Boolean flags (calculated on content paragraphs only)
            flag_cols = st.columns(len(GEMINI_FLAGS))
            for idx, flag in enumerate(GEMINI_FLAGS):
                count = flag_counts.get(flag, 0)
                pct = (count / content_count * 100) if content_count else 0
                with flag_cols[idx]:
                    flag_name = flag.replace('_', ' ').title()
                    if pct >= 90:
                        st.success(f"**{flag_name}**\n{count}/{content_count} ({pct:.0f}%)")
                    elif pct >= 70:
                        st.warning(f"**{flag_name}**\n{count}/{content_count} ({pct:.0f}%)")
                    else:
                        st.error(f"**{flag_name}**\n{count}/{content_count} ({pct:.0f}%)")
            
            # Show numeric quality levels if available
            meaning_level = flag_counts.get("same_meaning_level_avg")
            missing_level = flag_counts.get("missing_info_level_avg")
            grammar_score = flag_counts.get("grammar_score")
            
            if any(x is not None for x in [meaning_level, missing_level, grammar_score]):
                st.divider()
                st.markdown("##### 🎯 Quality Levels (0-10 scale)")
                level_cols = st.columns(3)
                
                with level_cols[0]:
                    if meaning_level is not None:
                        if meaning_level >= 8:
                            st.success(f"**Same Meaning**\n{meaning_level:.1f}/10")
                        elif meaning_level >= 6:
                            st.warning(f"**Same Meaning**\n{meaning_level:.1f}/10")
                        else:
                            st.error(f"**Same Meaning**\n{meaning_level:.1f}/10")
                    else:
                        st.info("**Same Meaning**\nNot evaluated")
                
                with level_cols[1]:
                    if missing_level is not None:
                        # Lower is better for missing info (0 = no missing info)
                        if missing_level <= 2:
                            st.success(f"**Missing Info**\n{missing_level:.1f}/10")
                        elif missing_level <= 4:
                            st.warning(f"**Missing Info**\n{missing_level:.1f}/10")
                        else:
                            st.error(f"**Missing Info**\n{missing_level:.1f}/10")
                    else:
                        st.info("**Missing Info**\nNot evaluated")
                
                with level_cols[2]:
                    if grammar_score is not None:
                        if grammar_score >= 8:
                            st.success(f"**Grammar**\n{grammar_score:.1f}/10")
                        elif grammar_score >= 6:
                            st.warning(f"**Grammar**\n{grammar_score:.1f}/10")
                        else:
                            st.error(f"**Grammar**\n{grammar_score:.1f}/10")
                    else:
                        st.info("**Grammar**\nNot evaluated")
            
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
                # Extract paragraph type and grammar info
                para_type = detail.get("type", "content")
                is_heading = para_type == "heading"
                grammar_score = detail.get("grammar_score")
                grammar_errors = detail.get("grammar_errors", [])
                
                para_info = {
                    "Para #": detail["paragraph"],
                    "Type": "📋 Heading" if is_heading else "📄 Content",
                    "WC Before": detail["wc_before"],
                    "WC After": detail["wc_after"],
                    "WC Δ": detail["wc_after"] - detail["wc_before"],
                    "GZ Before": f"{detail['ai_before']['gptzero']:.3f}" if detail['ai_before'].get('gptzero') is not None else "—",
                    "GZ After": f"{detail['ai_after']['gptzero']:.3f}" if detail['ai_after'].get('gptzero') is not None else "—",
                    "SP Before": f"{detail['ai_before']['sapling']:.3f}" if detail['ai_before'].get('sapling') is not None else "—",
                    "SP After": f"{detail['ai_after']['sapling']:.3f}" if detail['ai_after'].get('sapling') is not None else "—",
                }
                
                # Add quality flags (empty for headings)
                for flag in GEMINI_FLAGS:
                    flag_name = flag.replace('_', ' ').title()
                    if is_heading:
                        para_info[flag_name] = "—"  # No quality evaluation for headings
                    else:
                        para_info[flag_name] = "✅" if detail["flags"].get(flag, False) else "❌"
                
                # Add numeric quality levels for content paragraphs
                if not is_heading:
                    meaning_level = detail.get("same_meaning_level")
                    missing_level = detail.get("missing_info_level")
                    para_info["Meaning Lv"] = f"{meaning_level:.1f}" if meaning_level is not None else "—"
                    para_info["Missing Lv"] = f"{missing_level:.1f}" if missing_level is not None else "—"
                    
                    # Add meaning details if available
                    meaning_details = detail.get("same_meaning_details", "")
                    if meaning_details:
                        para_info["Meaning Details"] = meaning_details[:50] + ("..." if len(meaning_details) > 50 else "")
                    else:
                        para_info["Meaning Details"] = "—"
                        
                    # Add missing info details
                    missing_items = detail.get("missing_items", [])
                    added_items = detail.get("added_items", [])
                    if missing_items or added_items:
                        details_parts = []
                        if missing_items:
                            details_parts.append(f"Missing: {', '.join(missing_items[:2])}")
                        if added_items:
                            details_parts.append(f"Added: {', '.join(added_items[:2])}")
                        para_info["Info Changes"] = "; ".join(details_parts)[:60] + ("..." if len("; ".join(details_parts)) > 60 else "")
                    else:
                        para_info["Info Changes"] = "—"
                else:
                    para_info["Meaning Lv"] = "—"
                    para_info["Missing Lv"] = "—"
                    para_info["Meaning Details"] = "—"
                    para_info["Info Changes"] = "—"
                
                # Add grammar info (0-10 scale for content, empty for headings)
                if not is_heading:
                    if grammar_score is not None:
                        para_info["Grammar Lv"] = f"{grammar_score:.1f}"
                    else:
                        para_info["Grammar Lv"] = "—"
                        
                    if grammar_errors:
                        para_info["Grammar Issues"] = f"{len(grammar_errors)} issue(s)"
                        # Show first few errors
                        error_preview = "; ".join(grammar_errors[:2])
                        if len(error_preview) > 50:
                            error_preview = error_preview[:50] + "..."
                        para_info["Error Details"] = error_preview
                    else:
                        para_info["Grammar Issues"] = "None"
                        para_info["Error Details"] = "—"
                else:
                    para_info["Grammar Lv"] = "—"
                    para_info["Grammar Issues"] = "—"
                    para_info["Error Details"] = "—"
                
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
            
            def style_levels(val):
                """Style the 0-10 level columns (Meaning Lv, Missing Lv)."""
                if isinstance(val, str) and val != "—":
                    try:
                        level = float(val)
                        if level >= 8:
                            return 'color: green; font-weight: bold'
                        elif level >= 6:
                            return 'color: orange'
                        elif level < 4:
                            return 'color: red; font-weight: bold'
                    except:
                        pass
                return ''
            
            def style_type(val):
                """Style the paragraph type column."""
                if val == "📋 Heading":
                    return 'background-color: #e8f4f8; font-weight: bold'
                elif val == "📄 Content":
                    return 'background-color: #f8f8f8'
                return ''
            
            style_cols = ['GZ Before', 'GZ After', 'SP Before', 'SP After', 'WC Δ']
            flag_cols = [col for col in df.columns if col in [f.replace('_', ' ').title() for f in GEMINI_FLAGS]]
            level_cols = ['Meaning Lv', 'Missing Lv']
            
            styled_df = df.style
            
            # Apply type styling
            if 'Type' in df.columns:
                styled_df = styled_df.applymap(style_type, subset=['Type'])
            
            # Apply delta styling
            if 'WC Δ' in df.columns:
                styled_df = styled_df.applymap(style_delta, subset=['WC Δ'])
            
            # Apply grammar styling (now for Grammar Lv column)
            if 'Grammar Lv' in df.columns:
                styled_df = styled_df.applymap(style_levels, subset=['Grammar Lv'])  # Use levels styling for 0-10 scale
            
            # Apply level styling for meaning and missing info
            for col in level_cols:
                if col in df.columns:
                    styled_df = styled_df.applymap(style_levels, subset=[col])
            
            # Apply flag styling
            for col in flag_cols:
                if col in df.columns:
                    styled_df = styled_df.applymap(
                        lambda x: 'background-color: #d4f8d4' if x == "✅" else 'background-color: #f8d4d4' if x == "❌" else 'background-color: #f0f0f0' if x == "—" else '',
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

# ─────────────────────── performance helpers ─────────────────────────
@st.cache_data(ttl=1800)
def cached_safe_hist(data: List[float], bins: int = 20):
    """Cached version of safe histogram data preparation."""
    if not data:
        return None, "No data"
    
    data_range = max(data) - min(data)
    if data_range == 0:
        # All values identical
        value = data[0]
        width = abs(value) * 0.1 if value != 0 else 1
        return {
            'type': 'bar',
            'value': value,
            'count': len(data),
            'width': width
        }, None
    else:
        # Normal histogram
        hist, bin_edges = np.histogram(data, bins=bins)
        return {
            'type': 'hist',
            'hist': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }, None

def render_cached_histogram(ax, cached_data):
    """Render histogram from cached data."""
    if cached_data is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    
    if cached_data['type'] == 'bar':
        # Single bar for identical values
        ax.bar([cached_data['value']], [cached_data['count']], 
               width=cached_data['width'])
        ax.set_xlim(cached_data['value'] - cached_data['width'] * 2, 
                   cached_data['value'] + cached_data['width'] * 2)
    else:
        # Normal histogram
        bin_edges = cached_data['bin_edges']
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
        widths = [bin_edges[i+1] - bin_edges[i] for i in range(len(bin_edges)-1)]
        ax.bar(bin_centers, cached_data['hist'], width=widths)

# ─────────────────────── optimized draft renderer ────────────────────────
@st.cache_data(ttl=900, show_spinner=False)  # Cache for 15 minutes
def _compute_draft_metrics(draft: Dict, para_total: int) -> Dict:
    """Pre-compute expensive draft metrics for caching."""
    metrics = {}
    
    # Basic scores
    if "scores_before" in draft and "scores_after" in draft:
        metrics["gz_before"] = draft["scores_before"]["group_doc"]["gptzero"]
        metrics["sp_before"] = draft["scores_before"]["group_doc"]["sapling"]
        metrics["gz_after"] = draft["scores_after"]["group_doc"]["gptzero"]
        metrics["sp_after"] = draft["scores_after"]["group_doc"]["sapling"]
        
        # Handle None values for disabled detectors in delta calculations
        if metrics["gz_after"] is not None and metrics["gz_before"] is not None:
            metrics["gz_delta"] = metrics["gz_after"] - metrics["gz_before"]
        else:
            metrics["gz_delta"] = 0
            
        if metrics["sp_after"] is not None and metrics["sp_before"] is not None:
            metrics["sp_delta"] = metrics["sp_after"] - metrics["sp_before"]
        else:
            metrics["sp_delta"] = 0
    else:
        metrics.update({
            "gz_before": 0, "sp_before": 0, "gz_after": 0, "sp_after": 0,
            "gz_delta": 0, "sp_delta": 0
        })
    
    # Word counts
    metrics["wc_before"] = draft.get("wordcount_before", 0)
    metrics["wc_after"] = draft.get("wordcount_after", 0)
    metrics["wc_delta"] = metrics["wc_after"] - metrics["wc_before"] 
    
    # Zero-shot detection (only for enabled detectors)
    metrics["gz_zero_shot"] = metrics["gz_after"] is not None and metrics["gz_after"] <= ZERO_SHOT_THRESHOLD
    metrics["sp_zero_shot"] = metrics["sp_after"] is not None and metrics["sp_after"] <= ZERO_SHOT_THRESHOLD
    
    # Quality metrics
    para_mismatch = draft.get("para_mismatch", False)
    if not para_mismatch and draft.get("flag_counts"):
        flag_counts = draft["flag_counts"]
        content_count = flag_counts.get("content_paragraph_count", para_total)
        
        # Boolean quality percentage
        bool_quality = sum(flag_counts.get(f, 0) for f in GEMINI_FLAGS) / (content_count * len(GEMINI_FLAGS)) * 100 if content_count > 0 else 0
        metrics["quality_pct"] = bool_quality
        
        # Grammar score
        metrics["grammar_score"] = flag_counts.get("grammar_score")
        
        # Structure info
        total_segments = flag_counts.get("total_segments", para_total)
        metrics["content_count"] = content_count
        metrics["heading_count"] = total_segments - content_count
        metrics["total_segments"] = total_segments
    else:
        metrics.update({
            "quality_pct": None, "grammar_score": None,
            "content_count": 0, "heading_count": 0, "total_segments": 0
        })
    
    # Paragraph structure
    metrics["para_mismatch"] = para_mismatch
    metrics["mismatch_reason"] = draft.get("mismatch_reason")
    metrics["para_count_after"] = draft.get("para_count_after", 0)
    metrics["para_count_delta"] = metrics["para_count_after"] - para_total
    
    return metrics

def render_draft_optimized(draft: Dict, para_total: int, doc_name: str, model: str):
    """Optimized draft renderer using pre-computed metrics."""
    iter_num = draft.get("iter", 0) + 1
    mode = draft.get("mode", "unknown")
    
    # Get cached metrics
    metrics = _compute_draft_metrics(draft, para_total)
    
    # Draft header with iteration info
    expander_title = f"🔄 Iteration {iter_num} • {mode.title()} mode"
    if metrics["gz_zero_shot"] or metrics["sp_zero_shot"]:
        expander_title += " • ✨ Zero-shot!"
    
    with st.expander(expander_title, expanded=True):
        # Key metrics row
        cols = st.columns([2, 2, 2, 2, 2])
        
        with cols[0]:
            colored_metric("GPTZero", f"{metrics['gz_after']:.3f}", metrics['gz_delta'])
            if metrics["gz_zero_shot"]:
                st.success("✅ Zero-shot!")
        
        with cols[1]:
            colored_metric("Sapling", f"{metrics['sp_after']:.3f}", metrics['sp_delta'])
            if metrics["sp_zero_shot"]:
                st.success("✅ Zero-shot!")
        
        with cols[2]:
            st.metric("Word Count Δ", f"{metrics['wc_delta']:+d}")
            st.caption(f"{metrics['wc_before']} → {metrics['wc_after']}")
        
        with cols[3]:
            # Quality score
            if metrics["quality_pct"] is not None:
                st.metric("Quality", f"{metrics['quality_pct']:.1f}%")
                if metrics["grammar_score"] is not None:
                    st.caption(f"Grammar: {metrics['grammar_score']:.1f}/10")
                else:
                    st.caption("Legacy (no grammar)")
                
                # Structure info
                if metrics["total_segments"] > metrics["content_count"]:
                    st.caption(f"{metrics['content_count']} content + {metrics['heading_count']} headings")
                else:
                    st.caption(f"{metrics['content_count']} content paras")
            else:
                st.metric("Quality", "—")
                if metrics["para_mismatch"]:
                    st.caption("Para mismatch")
        
        with cols[4]:
            # Paragraph structure
            if metrics["para_mismatch"]:
                st.metric("Paragraphs", f"{metrics['para_count_after']}", f"{metrics['para_count_delta']:+d}")
                st.warning("⚠️ Mismatch")
                if metrics["mismatch_reason"]:
                    st.caption(f"Reason: {metrics['mismatch_reason']}")
            else:
                st.metric("Paragraphs", f"{metrics['para_count_after']}")
                st.success("✅ Match")
        
        # For detailed breakdown, only show if expanded and not cached
        if st.checkbox(f"Show detailed breakdown", key=f"detail_{doc_name}_{model}_{mode}_{iter_num}"):
            # Use the original detailed render for the expanded view
            render_draft(draft, para_total, doc_name, model)

# ─────────────────────── cache management helpers ─────────────────────────
def clear_streamlit_cache():
    """Clear all Streamlit caches to free memory and force recomputation."""
    try:
        st.cache_data.clear()
        return True, "✅ Successfully cleared all cached data"
    except Exception as e:
        return False, f"❌ Error clearing cache: {str(e)}"

def get_cache_stats():
    """Get basic cache statistics for performance monitoring."""
    try:
        # This is approximate since Streamlit doesn't expose detailed cache stats
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "memory_usage_mb": memory_info.rss / 1024 / 1024,
            "memory_percentage": process.memory_percent(),
            "cpu_percentage": process.cpu_percent(),
            "cache_available": hasattr(st, 'cache_data')
        }
    except ImportError:
        return {
            "memory_usage_mb": "Unknown",
            "memory_percentage": "Unknown", 
            "cpu_percentage": "Unknown",
            "cache_available": hasattr(st, 'cache_data')
        }
    except Exception:
        return {"cache_available": hasattr(st, 'cache_data')}

def show_performance_metrics():
    """Display performance metrics in the sidebar or main area."""
    st.markdown("##### 🚀 Performance")
    
    stats = get_cache_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        if isinstance(stats.get("memory_usage_mb"), (int, float)):
            st.metric("Memory", f"{stats['memory_usage_mb']:.1f} MB")
        else:
            st.metric("Memory", "Unknown")
    
    with col2:
        if isinstance(stats.get("cpu_percentage"), (int, float)):
            st.metric("CPU", f"{stats['cpu_percentage']:.1f}%")
        else:
            st.metric("CPU", "Unknown")
    
    # Cache management
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Cache", help="Clear all cached data to free memory"):
            success, message = clear_streamlit_cache()
            if success:
                st.success(message)
            else:
                st.error(message)
            st.rerun()
    
    with col2:
        if st.button("♻️ Refresh Stats"):
            st.rerun()

# ─────────────────────── lazy loading helpers ─────────────────────────
class LazyLoader:
    """Helper class for lazy loading of expensive computations."""
    
    def __init__(self, compute_func, cache_key: str, ttl: int = 1800):
        self.compute_func = compute_func
        self.cache_key = cache_key
        self.ttl = ttl
        self._cached_func = st.cache_data(ttl=ttl)(compute_func)
    
    def load(self, *args, **kwargs):
        """Load data with caching and progress indication."""
        with st.spinner(f"Loading {self.cache_key}..."):
            return self._cached_func(*args, **kwargs)
    
    def clear_cache(self):
        """Clear cache for this specific loader."""
        try:
            self._cached_func.clear()
            return True
        except:
            return False

# ─────────────────────── progress tracking ─────────────────────────
def show_loading_progress(current: int, total: int, operation: str = "Processing"):
    """Show a simple progress indicator."""
    if total > 0:
        progress = current / total
        st.progress(progress, text=f"{operation}: {current}/{total} ({progress*100:.1f}%)")
    else:
        st.info(f"{operation}...")