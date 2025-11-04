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

import sys
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────── project root ────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ZERO_SHOT_THRESHOLD
from src.pages.utils import (
    GEMINI_FLAGS,
    colored_metric,
    qp_get,
    qp_set,
    render_draft,
    safe_hist,
)
from src.api_client import get_client, cached_list_runs, cached_get_run
from src.results_db import delete_run  # Keep for deletion

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


@st.cache_data(ttl=600, show_spinner=False)
def _cached_load_run_preview(run_id: str, max_docs: int = 20):
    """Cached loader for run preview with limited documents (fast initial load)."""
    from src.results_db import load_run_with_limit
    return load_run_with_limit(run_id, max_docs=max_docs)

@st.cache_data(ttl=3600, show_spinner="Merging run data...")
def _merge_runs_data(run_ids: List[str]) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Merge data from multiple runs, handling duplicate models.
    Returns merged docs list and metadata about the merge.
    """
    merged_docs = []
    model_sources = defaultdict(set)  # Track which runs each model came from
    doc_by_name = defaultdict(lambda: {"runs": []})  # Group by document name
    
    for run_id in run_ids:
        try:
            run_data = cached_get_run(run_id)
        except Exception as e:
            st.warning(f"Could not load run {run_id}: {e}")
            continue
            
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
        "total_runs_merged": len(run_ids),
        "model_sources": dict(model_sources),
        "total_documents": len(merged_docs),
        "total_drafts": sum(len(doc.get("runs", [])) for doc in merged_docs)
    }
    
    return merged_docs, merge_metadata


@st.cache_data(ttl=1800, show_spinner="Computing statistical analysis...")
def _aggregate_statistics_by_model_mode_folder(docs: List[Dict]) -> Dict[str, Any]:
    """
    Build nested dict  folder → model → mode → stats
    and attach word-count-difference metrics + quality-flag rates.
    """
    stats: DefaultDict[str, DefaultDict[str, DefaultDict[str, Dict]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    
    # First pass: collect document-level paragraph counts for each model/mode combination
    doc_paragraphs_by_model_mode = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

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

    for doc in docs:
        folder = doc.get("folder", "unknown")
        doc_name = doc.get("document", "unknown")
        doc_para_count = doc.get("paragraph_count", 0)
        
        if not doc.get("runs"):
            continue
            
        for dr in doc["runs"]:
            if "scores_after" not in dr or "group_doc" not in dr["scores_after"]:
                continue
            model = dr.get("model", "unknown")
            mode = dr.get("mode", "unknown")
            
            # Track this document for this model/mode combination
            doc_paragraphs_by_model_mode[folder][model][mode].add((doc_name, doc_para_count))

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
                "grammar_scores": [],  # backward compatibility
                "draft_count": 0,
                "mismatch_count": 0,          # (legacy: kept for back‑compat)
                "doc_mismatch_count": 0,      # NEW: whole‑document structure mismatches
                "doc_total_drafts": 0,        # NEW: drafts seen – needed for %
                "para_level_mismatch_count": 0,  # paragraph 1→N mismatches in para mode
                "total_content_paragraphs": 0,  # NEW: total content paragraphs processed (calculated from documents)
                "para_level_mismatched_paragraphs": 0,  # NEW: count of mismatched paragraphs
                "zs_hits": {"gptzero": 0, "sapling": 0},
                # NEW: extended metrics
                "draft_length_deviations": [],  # percentage deviations at draft level
                "para_length_deviations": [],   # paragraph level deviations
                "same_meaning_levels": [],      # 0-10 numeric levels
                "missing_info_levels": [],      # 0-10 numeric levels
                "citation_preservation_rates": [],  # paragraph citation preservation
                "citation_exact_match_rates": [],   # citation exact match rates
                "content_paragraph_counts": [],     # actual content paragraphs (excluding headings)
                "total_paragraphs": [],             # includes headings for mismatch detection
                # raw series for extended-stats tab
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
        
        # Only check zero-shot threshold if detector was actually used (not None)
        if gz is not None and gz <= ZERO_SHOT_THRESHOLD:
            bucket["zs_hits"]["gptzero"] += 1
        if sp is not None and sp <= ZERO_SHOT_THRESHOLD:
            bucket["zs_hits"]["sapling"] += 1

        # word-count delta
        delta_wc = dr.get("wordcount_after", 0) - dr.get("wordcount_before", 0)
        bucket["wc_deltas"].append(delta_wc)

        # NEW: collect draft-level length deviation
        draft_length_deviation = dr.get("draft_length_deviation", 0)
        bucket["draft_length_deviations"].append(draft_length_deviation)
        
        # Store raw series for extended-stats tab
        bucket["series"]["after_gz"].append(gz)
        bucket["series"]["after_sp"].append(sp)
        bucket["series"]["wc"].append(delta_wc)
        bucket["series"]["draft_length_dev"].append(draft_length_deviation)
        
        # Initialize flag_counts for later use
        flag_counts = {}
        
        # Paragraph and quality metrics (skip drafts with paragraph mismatch)
        if not dr.get("para_mismatch", False):
            flag_counts = dr.get("flag_counts", {})
            total_segments = flag_counts.get("total_segments", dr.get("para_count_before", 1))
            content_paragraphs = flag_counts.get("content_paragraph_count", total_segments)
            
            # Track paragraph counts
            bucket["content_paragraph_counts"].append(content_paragraphs)
            bucket["total_paragraphs"].append(total_segments)
            
            # Boolean quality flags (calculated on content paragraphs only)
            for flag in _EXPECTED_FLAGS:
                cnt = flag_counts.get(flag, 0)
                bucket["quality_flags"][flag].append((cnt / content_paragraphs) * 100 if content_paragraphs else 0)
            
            # NEW: Numeric quality levels (0-10 scale)
            same_meaning_level = flag_counts.get("same_meaning_level_avg")
            missing_info_level = flag_counts.get("missing_info_level_avg")
            grammar_score = flag_counts.get("grammar_score")
            
            if same_meaning_level is not None:
                bucket["same_meaning_levels"].append(same_meaning_level)
                bucket["series"]["same_meaning_level"].append(same_meaning_level)
            
            if missing_info_level is not None:
                bucket["missing_info_levels"].append(missing_info_level)
                bucket["series"]["missing_info_level"].append(missing_info_level)
            
            if grammar_score is not None:
                bucket["grammar_scores"].append(grammar_score)
                bucket["series"]["grammar"].append(grammar_score)
            
            # NEW: Length deviation metrics
            para_length_deviation = flag_counts.get("para_length_deviation_avg", 0)
            bucket["para_length_deviations"].append(para_length_deviation)
            bucket["series"]["para_length_dev"].append(para_length_deviation)
            
            # NEW: Citation preservation metrics
            citation_preservation_rate = flag_counts.get("paragraph_citation_preservation_rate", 100)
            citation_exact_match_rate = flag_counts.get("citation_exact_match_rate", 100)
            bucket["citation_preservation_rates"].append(citation_preservation_rate)
            bucket["citation_exact_match_rates"].append(citation_exact_match_rate)
            bucket["series"]["citation_preservation"].append(citation_preservation_rate)
            bucket["series"]["citation_exact_match"].append(citation_exact_match_rate)
            
            # Calculate overall quality percentage (boolean flags only for backward compatibility)
            qual_pct = (
                sum(flag_counts.get(f, 0) for f in _EXPECTED_FLAGS)
                / (content_paragraphs * len(_EXPECTED_FLAGS))
                * 100
                if content_paragraphs
                else 0
            )
            bucket["series"]["quality"].append(qual_pct)

        bucket["draft_count"] += 1

        # ── document‑structure mismatch (all modes) ────────────────
        if dr.get("para_mismatch", False):
            bucket["doc_mismatch_count"] += 1
        bucket["doc_total_drafts"] += 1

        # legacy counter (kept only so nothing else breaks)
        if dr.get("para_mismatch", False) and dr.get("mode") == "para":
            bucket["mismatch_count"] += 1

        
        # Track paragraph-level mismatches in para mode
        if dr.get("mode") == "para":
            # Track drafts with paragraph-level mismatches
            if dr.get("has_para_level_mismatches", False):
                bucket["para_level_mismatch_count"] += 1
            
            # Track individual mismatched paragraphs
            para_level_mismatches = dr.get("para_level_mismatches", 0)
            bucket["para_level_mismatched_paragraphs"] += para_level_mismatches

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

                # Calculate average metrics
                avg_grammar = np.mean(data["grammar_scores"]) if data["grammar_scores"] else None
                avg_same_meaning_level = np.mean(data["same_meaning_levels"]) if data["same_meaning_levels"] else None
                avg_missing_info_level = np.mean(data["missing_info_levels"]) if data["missing_info_levels"] else None
                
                # NEW: Length deviation metrics
                draft_length_devs = np.array(data["draft_length_deviations"])
                para_length_devs = np.array(data["para_length_deviations"])
                
                # Calculate length deviation percentile bands
                len_within_10_pct = (np.abs(draft_length_devs) <= 10).mean() * 100 if draft_length_devs.size else 0
                len_within_15_pct = (np.abs(draft_length_devs) <= 15).mean() * 100 if draft_length_devs.size else 0
                len_within_20_pct = (np.abs(draft_length_devs) <= 20).mean() * 100 if draft_length_devs.size else 0
                
                # NEW: Citation preservation metrics
                avg_citation_preservation = np.mean(data["citation_preservation_rates"]) if data["citation_preservation_rates"] else 100
                avg_citation_exact_match = np.mean(data["citation_exact_match_rates"]) if data["citation_exact_match_rates"] else 100
                
                # Average paragraph counts (from non-mismatched drafts)
                avg_content_paras = np.mean(data["content_paragraph_counts"]) if data["content_paragraph_counts"] else 0
                total_content_paras = sum(data["content_paragraph_counts"]) if data["content_paragraph_counts"] else 0
                
                # NEW: Calculate total paragraphs from document-level counts (consistent across models)
                doc_level_total_paras = sum(para_count for _, para_count in doc_paragraphs_by_model_mode[folder][model][mode])
                
                # Set the consistent total in the bucket for future use
                data["total_content_paragraphs"] = doc_level_total_paras

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
                    # Legacy metrics (backward compatibility)
                    "grammar_score": avg_grammar,
                    
                    # NEW: Numeric quality levels (0-10 scale)
                    "same_meaning_level_avg": avg_same_meaning_level,
                    "missing_info_level_avg": avg_missing_info_level,
                    
                    # NEW: Length deviation metrics
                    "draft_length_deviation_avg": draft_length_devs.mean() if draft_length_devs.size else 0,
                    "para_length_deviation_avg": para_length_devs.mean() if para_length_devs.size else 0,
                    "length_within_10_pct": len_within_10_pct,
                    "length_within_15_pct": len_within_15_pct,
                    "length_within_20_pct": len_within_20_pct,
                    
                    # NEW: Citation metrics  
                    "citation_preservation_rate_avg": avg_citation_preservation,
                    "citation_exact_match_rate_avg": avg_citation_exact_match,
                    
                    # Paragraph counts
                    "avg_content_paragraphs": avg_content_paras,
                    "total_content_paragraphs": doc_level_total_paras,
                    
                    # Existing metrics
                    "draft_count": data["draft_count"],
                    "mismatch_rate": data["mismatch_count"] / data["draft_count"] * 100,
                    # ── NEW mismatch metrics ─────────────────────────────
                    "draft_with_para_mismatch_pct": (
                        data["para_level_mismatch_count"] / data["draft_count"] * 100
                        if data["draft_count"] else 0
                    ),
                    "draft_with_doc_mismatch_pct": (
                        data["doc_mismatch_count"] / data["doc_total_drafts"] * 100
                        if data["doc_total_drafts"] else 0
                    ),
                    "mismatched_paragraphs_pct": (
                        data["para_level_mismatched_paragraphs"] / doc_level_total_paras * 100
                        if doc_level_total_paras else 0
                    ),
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

@st.cache_data(ttl=1800, show_spinner="Computing model performance...")
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
                "Avg Grammar": f"{avg_grammar:.1f}" if avg_grammar is not None else "—",  # 0-10 scale
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


@st.cache_data(ttl=1800, show_spinner="Building extended statistics...")
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
                    
                # Grammar scores are already on 0-10 scale
                grammar_series = ser.get("grammar", [])
                
                rows.append(
                    {
                        "Folder": folder,
                        "Model":  model,
                        "Mode":   mode.title(),
                        **{f"GPTZero {k}": v for k, v in _describe(ser.get("after_gz", [])).items()},
                        **{f"Sapling {k}": v for k, v in _describe(ser.get("after_sp", [])).items()},
                        **{f"WC Δ {k}":  v for k, v in _describe(ser.get("wc", [])).items()},
                        **{f"Quality {k}": v for k, v in _describe(ser.get("quality", [])).items()},
                        **{f"Grammar Lv {k}": v for k, v in _describe(grammar_series).items()},  # NEW as 0-10
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
    """Style grammar scores with color coding (now expects 0-10 scale).""" 
    if isinstance(v, str) and v != "—":
        try:
            score = float(v)
            if score >= 8:
                return "color: green; font-weight: bold"
            elif score >= 6:
                return "color: orange"
            elif score < 4:
                return "color: red; font-weight: bold"
        except:
            pass
    return ""

def _style_levels(v):
    """Style 0-10 level columns (Same Meaning Lv, Missing Info Lv)."""
    if isinstance(v, str) and v != "—":
        try:
            level = float(v)
            # For meaning levels, higher is better
            if level >= 8:
                return "color: green; font-weight: bold"
            elif level >= 6:
                return "color: orange"
            elif level < 4:
                return "color: red; font-weight: bold"
        except:
            pass
    return ""

def _style_missing_info_levels(v):
    """Style missing info levels where lower is better."""
    if isinstance(v, str) and v != "—":
        try:
            level = float(v)
            # For missing info, lower is better (0 = no missing info)
            if level <= 2:
                return "color: green; font-weight: bold"
            elif level <= 4:
                return "color: orange"
            elif level > 6:
                return "color: red; font-weight: bold"
        except:
            pass
    return ""

def _style_citation_metrics(v):
    """Style citation preservation metrics (percentages)."""
    if isinstance(v, str) and v.endswith("%") and v != "—":
        try:
            pct = float(v.rstrip("%"))
            if pct >= 95:
                return "color: green; font-weight: bold"
            elif pct >= 80:
                return "color: orange"
            elif pct < 60:
                return "color: red; font-weight: bold"
        except:
            pass
    return ""

def _style_length_deviation(v):
    """Style length deviation percentage columns."""
    if isinstance(v, str) and v.endswith("%") and v != "—":
        try:
            pct = float(v.rstrip("%"))
            if pct >= 80:  # Good: within target range
                return "color: green; font-weight: bold"
            elif pct >= 60:
                return "color: orange"
            elif pct < 40:
                return "color: red; font-weight: bold"
        except:
            pass
    return ""

def _style_mismatch_percentages(v):
    """Style for mismatch percentages - red for any mismatches"""
    if isinstance(v, str) and v.endswith("%") and v != "—":
        try:
            pct = float(v.rstrip("%"))
            if pct <= 0:
                return "color: green; font-weight: bold"  # Green for no mismatches
            elif pct <= 10:
                return "color: orange; font-weight: bold"  # Yellow for low mismatches
            else:
                return "color: red; font-weight: bold"  # Red for high mismatches
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
        .applymap(_style_levels, subset=["Avg Grammar"])  # Changed to _style_levels for 0-10 scale
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
@st.cache_data(ttl=1800, show_spinner="Creating comparison table...")
def _create_model_comparison_table(stats: Dict[str, Any], folder: str) -> pd.DataFrame:
    """
    Create the new 33-column detailed table matching the specification.
    All columns as defined in the user's requirements.
    """
    rows = []
    if folder not in stats:
        return pd.DataFrame()

    for model, modes in stats[folder].items():
        for mode in ("doc", "para"):
            if mode not in modes:
                continue
            s = modes[mode]
            
            # Column calculations following exact specification
            baseline_gz = s['baseline']['gptzero']
            baseline_sp = s['baseline']['sapling']
            after_gz = s['after']['gptzero']
            after_sp = s['after']['sapling']
            
            row = {
                # Columns 1-3: Basic info
                "Model": model,
                "Mode": mode.title(),
                "Drafts": s["draft_count"],
                
                # Column 4: Content paragraphs (excluding headings)
                "Paragraphs": int(s.get("total_content_paragraphs", 0)),
                
                # Columns 5-12: AI Detection scores  
                "Baseline GZ": f"{baseline_gz:.3f}",
                "After GZ": f"{after_gz:.3f}",
                "Δ GZ": s["deltas"]["gptzero"],
                "Zero-shot GZ": f"{s['zero_shot_success']['gptzero']:.1f}%",
                "Baseline SP": f"{baseline_sp:.3f}",
                "After SP": f"{after_sp:.3f}",
                "Δ SP": s["deltas"]["sapling"],
                "Zero-shot SP": f"{s['zero_shot_success']['sapling']:.1f}%",
                
                # Columns 13-17: Length deviation metrics (NEW)
                "Avg Draft Δ %": f"{s.get('draft_length_deviation_avg', 0):.1f}%",
                "Avg Para Δ %": f"{s.get('para_length_deviation_avg', 0):.1f}%",
                "Len ±10 %": f"{s.get('length_within_10_pct', 0):.1f}%",
                "Len ±15 %": f"{s.get('length_within_15_pct', 0):.1f}%", 
                "Len ±20 %": f"{s.get('length_within_20_pct', 0):.1f}%",
                
                # Columns 18-22: Word count metrics (existing)
                "Avg WC Δ": f"{s['deltas']['wordcount']:+.0f}",
                "Within 10 words %": f"{s['wc_diff']['within10']:.1f}%",
                "Within 20 words %": f"{s['wc_diff']['within20']:.1f}%",
                "% Longer": f"{s['wc_diff']['pct_longer']:.1f}%",
                "% Shorter": f"{s['wc_diff']['pct_shorter']:.1f}%",
                
                # Columns 23-25: Quality & Grammar
                "Quality %": f"{np.mean(list(s['quality'].values())):.1f}%",
                "Grammar Lv": f"{s.get('grammar_score', 0):.1f}" if s.get('grammar_score') is not None else "—",
                # Paragraph-level mismatch metrics
                "Drafts w/ para‑split %":  f"{s.get('draft_with_para_mismatch_pct', 0):.1f}%",
                "Drafts w/ doc‑mismatch %": f"{s.get('draft_with_doc_mismatch_pct', 0):.1f}%",
                "Mismatched Paragraphs %":   f"{s.get('mismatched_paragraphs_pct', 0):.1f}%",

                # Columns 26-27: NEW numeric quality levels 
                "Same Meaning Lv": f"{s.get('same_meaning_level_avg', 0):.1f}" if s.get('same_meaning_level_avg') is not None else "—",
                "Missing Info Lv": f"{s.get('missing_info_level_avg', 0):.1f}" if s.get('missing_info_level_avg') is not None else "—",
            }
            
            # Columns 28-31: Boolean quality flags (existing)
            for flag in _EXPECTED_FLAGS:
                col_name = f"{flag.replace('_',' ').title()} %"
                row[col_name] = f"{s['quality'].get(flag, 0):.1f}%"
            
            # Columns 32-33: NEW citation preservation metrics
            row["Citation Preserved %"] = f"{s.get('citation_preservation_rate_avg', 100):.1f}%"
            row["Citation Exact %"] = f"{s.get('citation_exact_match_rate_avg', 100):.1f}%"
            
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    
    # Define the exact 33-column order as specified
    column_order = [
        # Basic info (1-4)
        "Model", "Mode", "Drafts", "Paragraphs",
        
        # AI Detection (5-12)  
        "Baseline GZ", "After GZ", "Δ GZ", "Zero-shot GZ",
        "Baseline SP", "After SP", "Δ SP", "Zero-shot SP",
        
        # Length deviations (13-17)
        "Avg Draft Δ %", "Avg Para Δ %", "Len ±10 %", "Len ±15 %", "Len ±20 %",
        
        # Word count metrics (18-22)
        "Avg WC Δ", "Within 10 words %", "Within 20 words %", "% Longer", "% Shorter",
        
        # Quality metrics (23‑27)
        "Quality %", "Grammar Lv",
        "Drafts w/ para‑split %",   # NEW
        "Drafts w/ doc‑mismatch %", # NEW
        "Mismatched Paragraphs %",  # NEW
        "Same Meaning Lv", "Missing Info Lv",

        # Boolean flags (28-31)
        "Length Ok %", "Same Meaning %", "Same Lang %", "No Missing Info %",
        
        # Citation metrics (32-33)
        "Citation Preserved %", "Citation Exact %",
    ]
    
    df = pd.DataFrame(rows)
    # Return columns in the specified order, handling any missing columns
    available_cols = [col for col in column_order if col in df.columns]
    return df[available_cols]

# ═══════════════ RUN OVERVIEW & DOC PAGE (main) ═══════════════════════

def page_runs() -> None:
    # Get all available runs via API
    try:
        runs_response = cached_list_runs(limit=100, offset=0)
        runs_meta = []
        for r in runs_response.get("runs", []):
            # Parse folders and models from JSON strings (API returns them as strings)
            try:
                folders_str = r.get("folders", "")
                models_str = r.get("models", "")
                
                if isinstance(folders_str, str) and folders_str:
                    folders = json.loads(folders_str) if folders_str.startswith("[") else folders_str.split(",")
                else:
                    folders = folders_str if isinstance(folders_str, list) else []
                
                if isinstance(models_str, str) and models_str:
                    models = json.loads(models_str) if models_str.startswith("[") else models_str.split(",")
                else:
                    models = models_str if isinstance(models_str, list) else []
            except Exception as parse_err:
                # Fallback to string values
                folders = r.get("folders", "")
                models = r.get("models", "")
            
            runs_meta.append({
                "name": r["name"],
                "ts": r["timestamp"],
                "folders": folders,
                "models": models
            })
    except Exception as e:
        st.warning(f"Could not load runs: {e}")
        runs_meta = []
    
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
        st.metric("🔀 Merged Runs", merge_metadata["total_runs_merged"])
    with col2:
        st.metric("📁 Documents", merge_metadata["total_documents"])
    with col3:
        successful_docs = sum(1 for d in merged_docs if d.get("runs"))
        st.metric("✅ Successful", successful_docs)
    with col4:
        st.metric("📝 Total Drafts", merge_metadata["total_drafts"])
    
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
    
    # Run selector and delete button at the top
    col1, col2 = st.columns([4, 1])
    with col1:
        selected = st.selectbox("Select benchmark run", run_labels, index=default_idx)
    with col2:
        run_id = runs_meta[run_labels.index(selected)]["name"]
        delete_key = f"delete_run_{run_id}"
        if st.button("🗑️ Delete Run", type="secondary", key=delete_key):
            # Use session state for confirmation instead of checkbox
            if f"{delete_key}_confirmed" not in st.session_state:
                st.session_state[f"{delete_key}_confirmed"] = False
            
            if not st.session_state[f"{delete_key}_confirmed"]:
                st.session_state[f"{delete_key}_confirmed"] = True
                st.warning("⚠️ Click delete again to confirm")
            else:
                # Delete via API
                try:
                    get_client().delete_run(run_id)
                    st.success("✅ Run deleted successfully!")
                    # Clear session state
                    st.session_state[f"{delete_key}_confirmed"] = False
                    # Update query params without reload
                    qp_set(run=None, view=None, doc=None)
                    # Clear cache
                    if f"analysis_stats_{run_id}" in st.session_state:
                        del st.session_state[f"analysis_stats_{run_id}"]
                    if f"analysis_docs_{run_id}" in st.session_state:
                        del st.session_state[f"analysis_docs_{run_id}"]
                    # Refresh runs list
                    st.cache_data.clear()
                    # Don't sleep - just rerun immediately
                    st.rerun()  # Only rerun after delete to refresh list
                except Exception as e:
                    st.error(f"Failed to delete run: {e}")
                    st.session_state[f"{delete_key}_confirmed"] = False

    # Add preview mode toggle for faster loading
    preview_mode = st.checkbox(
        "⚡ Quick Preview Mode (load first 20 docs only)",
        value=True,
        key=f"preview_{run_id}",
        help="Enable for faster initial loading. Disable to load all documents."
    )

    try:
        if preview_mode:
            # Use cached optimized loader with document limit (fast!)
            run = _cached_load_run_preview(run_id, max_docs=20) or {}
            if run.get("truncated"):
                st.info(f"ℹ️ Showing first 20 of {run.get('total_docs', 0)} documents. Uncheck preview mode to load all documents.")
        else:
            run = cached_get_run(run_id) or {}
    except Exception as e:
        st.warning(f"Could not load run {run_id}: {e}")
        run = {}
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

    # ── Lazy load statistics - only calculate when needed --------
    # Store docs in session state for lazy loading
    cache_key = f"analysis_docs_{run_name}"
    st.session_state[cache_key] = docs
    
    # Store statistics computation state
    stats_cache_key = f"analysis_stats_{run_name}"
    
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
    
    # Helper function to get or compute stats lazily
    def get_stats():
        """Lazy load statistics - only compute when accessed."""
        if stats_cache_key not in st.session_state:
            with st.spinner("Computing statistics..."):
                st.session_state[stats_cache_key] = _aggregate_statistics_by_model_mode_folder(docs)
        return st.session_state[stats_cache_key]

    # ════════════════════════════════════════════════════════════════════════
    # Tab 1 – Detailed per-folder / model table + charts (+NEW wc histogram)
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        # Lazy load stats only when this tab is accessed
        detailed_stats = get_stats()
        st.subheader("🎯 Detailed Statistics by Folder, Model, and Mode")

        with st.expander("ℹ️ Understanding the metrics", expanded=False):
            st.markdown(
                """
                **Key Metrics:**  
                • **Δ GZ / Δ SP** – change in AI-detection score (negative = better)  
                • **Zero-shot** – % drafts ≤ 10 % on detector  
                • **Quality %** – average of all quality checks  
                • **Grammar Lv** – average grammatical correctness (0-10 scale)  
                • **Within 10 / 20 words** – word-count distance from original  
                • **% Longer / % Shorter** – drafts that grew / shrank  
                • **Mismatch %** – paragraph-count mismatches (document-level)
                • **Para Draft Mismatch %** – % of para-mode drafts with 1→N paragraph mismatches
                • **Para Para Mismatch %** – % of paragraphs where humanizer returned multiple paragraphs
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

                # Apply comprehensive styling for the new 33-column table
                styled_df = df.style
                
                # Delta columns (AI score changes)
                delta_cols = ["Δ GZ", "Δ SP"]
                styled_df = styled_df.applymap(_style_delta, subset=[c for c in delta_cols if c in df.columns])
                
                # Zero-shot columns
                zs_cols = ["Zero-shot GZ", "Zero-shot SP"]
                styled_df = styled_df.applymap(_style_zs, subset=[c for c in zs_cols if c in df.columns])
                
                # Quality percentage columns (boolean flags)
                qual_pct_cols = [c for c in df.columns if c.endswith(" %") and c not in 
                                ("Zero-shot GZ", "Zero-shot SP", "Avg Draft Δ %", "Avg Para Δ %", 
                                 "Len ±10 %", "Len ±15 %", "Len ±20 %", "Within 10 words %", 
                                 "Within 20 words %", "% Longer", "% Shorter", "Draft with Mismatches %",
                                 "Mismatched Paragraphs %",
                                 "Citation Preserved %", "Citation Exact %")]
                styled_df = styled_df.applymap(_style_quality, subset=qual_pct_cols)
                
                # NEW: Mismatch percentage columns
                mismatch_cols = ["Draft with Mismatches %", "Mismatched Paragraphs %"]
                styled_df = styled_df.applymap(_style_mismatch_percentages, subset=[c for c in mismatch_cols if c in df.columns])
                
                # NEW: Length deviation percentage columns
                length_dev_cols = ["Len ±10 %", "Len ±15 %", "Len ±20 %"]
                styled_df = styled_df.applymap(_style_length_deviation, subset=[c for c in length_dev_cols if c in df.columns])
                
                # NEW: Grammar level (0-10 scale)
                if "Grammar Lv" in df.columns:
                    styled_df = styled_df.applymap(_style_grammar, subset=["Grammar Lv"])
                
                # NEW: Numeric quality levels (0-10 scale)
                if "Same Meaning Lv" in df.columns:
                    styled_df = styled_df.applymap(_style_levels, subset=["Same Meaning Lv"])
                
                if "Missing Info Lv" in df.columns:
                    styled_df = styled_df.applymap(_style_missing_info_levels, subset=["Missing Info Lv"])
                
                # NEW: Citation preservation metrics
                citation_cols = ["Citation Preserved %", "Citation Exact %"]
                styled_df = styled_df.applymap(_style_citation_metrics, subset=[c for c in citation_cols if c in df.columns])
                
                # Format numeric columns
                format_dict = {}
                if "Δ GZ" in df.columns:
                    format_dict["Δ GZ"] = "{:+.3f}"
                if "Δ SP" in df.columns:
                    format_dict["Δ SP"] = "{:+.3f}"
                
                if format_dict:
                    styled_df = styled_df.format(format_dict)
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
                            
                            gram_val = mdf.iloc[0]["Grammar Lv"]
                            if gram_val != "—":
                                grammars.append(float(gram_val))
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
        # Lazy load stats only when this tab is accessed
        detailed_stats = get_stats()
        st.subheader("📈 Model Performance")
        with st.expander("ℹ️ About this view", expanded=False):
            st.markdown(
                """
                Compare humanizer models on different document sets.  
                Lower **Δ GZ / Δ SP** values and higher **Zero-shot** rates are better.
                Higher **Grammar Lv** indicates better grammatical quality (0-10 scale).
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
        # Lazy load stats only when this tab is accessed
        detailed_stats = get_stats()
        st.subheader("📐 Extended Statistics")
        st.caption("Min, 25-percentile, median, mean, 75-percentile and max for each metric.")
        ext_df = _build_extended_stats(detailed_stats)
        if ext_df.empty:
            st.info("No data available for extended statistics.")
        else:
            st.dataframe(ext_df, use_container_width=True, hide_index=True)

    with tab4:
        # Lazy load stats only when this tab is accessed
        detailed_stats = get_stats()
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
                    "Avg Grammar": f"{np.mean(all_grammar_scores):.1f}" if all_grammar_scores else "—",  # 0-10 scale
                })
        
        folder_df = pd.DataFrame(folder_summary)
        if not folder_df.empty:
            # Style the dataframe
            styled_folder = folder_df.style.applymap(
                lambda x: 'color: green; font-weight: bold' if isinstance(x, (int, float)) and x < 0 else ('color: red; font-weight: bold' if isinstance(x, (int, float)) and x > 0 else ''),
                subset=['Avg Δ GZ', 'Avg Δ SP']
            ).applymap(
                _style_levels,  # Changed from _style_grammar since we now use 0-10 scale
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
        # Lazy load stats only when this tab is accessed
        detailed_stats = get_stats()
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
            
            # Filter out None values for disabled detectors before calculating means
            gz_baselines = [x for x in folder_baselines[folder]["gz"] if x is not None]
            sp_baselines = [x for x in folder_baselines[folder]["sp"] if x is not None]
            
            base_gz = np.mean(gz_baselines) if gz_baselines else None
            base_sp = np.mean(sp_baselines) if sp_baselines else None

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
                                    # Use session state to trigger navigation without full reload
                                    st.session_state["nav_to_doc"] = True
                                    st.rerun()
                            elif doc.get("runs"):
                                st.caption("(View in single run mode)")
                            else:
                                st.button("View", key=f"view_{folder}_{i}", disabled=True)


# ──────────────────────────────────────────────────────────────────────────
def _page_document(run_id: str, docs: List[Dict], doc_name: str):
    """Completely redesigned document detail page with comprehensive paragraph-by-paragraph analysis"""
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
            # Clear navigation flag
            if "nav_to_doc" in st.session_state:
                del st.session_state["nav_to_doc"]
            st.rerun()  # Navigation requires rerun

    # Check if document has results
    if not doc.get("runs"):
        st.error("This document failed processing and has no results.")
        if doc.get("error"):
            st.error(f"Error: {doc['error']}")
        if doc.get("warning"):
            st.warning(f"Warning: {doc['warning']}")
        return

    # Document metadata and mismatch summary
    para_total = doc["paragraph_count"]
    baseline_wc = next((r.get('wordcount_before', 0) for r in doc['runs'] if r.get('mode')=='doc'), 0)
    
    # Handle None values for disabled detectors
    baseline_gz = None
    baseline_sp = None
    for r in doc['runs']:
        if 'scores_before' in r and 'group_doc' in r['scores_before']:
            if baseline_gz is None and r['scores_before']['group_doc']['gptzero'] is not None:
                baseline_gz = r['scores_before']['group_doc']['gptzero']
            if baseline_sp is None and r['scores_before']['group_doc']['sapling'] is not None:
                baseline_sp = r['scores_before']['group_doc']['sapling']
            if baseline_gz is not None and baseline_sp is not None:
                break
    
    # Set defaults for disabled detectors
    if baseline_gz is None:
        baseline_gz = 0.5  # Default for disabled GPTZero
    if baseline_sp is None:
        baseline_sp = 0.5  # Default for disabled Sapling
    
    # Count paragraph-level mismatches across all drafts
    total_para_mismatches = 0
    total_drafts_with_para_mismatches = 0
    for run in doc.get('runs', []):
        if run.get('mode') == 'para' and run.get('para_level_mismatches', 0) > 0:
            total_para_mismatches += run.get('para_level_mismatches', 0)
            total_drafts_with_para_mismatches += 1
    
    # Metadata cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📁 Folder", doc.get('folder', 'unknown'))
    with col2:
        st.metric("📝 Paragraphs", para_total)
    with col3:
        st.metric("📊 Word Count", baseline_wc)
    with col4:
        # Check if GPTZero was actually used (not just default)
        gz_was_used = any(r.get('scores_before', {}).get('group_doc', {}).get('gptzero') is not None 
                         for r in doc['runs'])
        if gz_was_used:
            st.metric("🎯 Baseline GZ", f"{baseline_gz:.3f}")
        else:
            st.metric("🎯 Baseline GZ", "N/A", help="GPTZero detector was disabled")
    with col5:
        # Check if Sapling was actually used (not just default)
        sp_was_used = any(r.get('scores_before', {}).get('group_doc', {}).get('sapling') is not None 
                         for r in doc['runs'])
        if sp_was_used:
            st.metric("🎯 Baseline SP", f"{baseline_sp:.3f}")
        else:
            st.metric("🎯 Baseline SP", "N/A", help="Sapling detector was disabled")
    
    # Show paragraph mismatch summary if any exist
    if total_para_mismatches > 0:
        st.warning(f"⚠️ **Paragraph-Level Mismatches Detected:** {total_para_mismatches} paragraph mismatches across {total_drafts_with_para_mismatches} para-mode drafts. These occur when the humanizer returns multiple paragraphs for a single input paragraph (1→N mapping).")

    # Organize drafts by model and mode
    by_model: DefaultDict[str, Dict[str, List[Dict]]] = defaultdict(lambda: {"doc": [], "para": []})
    for dr in doc["runs"]:
        model = dr.get("model", "unknown")
        mode = dr.get("mode", "unknown")
        by_model[model][mode].append(dr)

    # Create main comparison table first (existing functionality)
    comparison_data = []
    for model in sorted(by_model):
        for mode in ["doc", "para"]:
            drafts = by_model[model][mode]
            valid_drafts = [d for d in drafts if "scores_after" in d and "group_doc" in d["scores_after"]]
            
            if valid_drafts:
                # Filter out None values for disabled detectors
                gz_scores = [d["scores_after"]["group_doc"]["gptzero"] for d in valid_drafts if d["scores_after"]["group_doc"]["gptzero"] is not None]
                sp_scores = [d["scores_after"]["group_doc"]["sapling"] for d in valid_drafts if d["scores_after"]["group_doc"]["sapling"] is not None]
                
                avg_gz = np.mean(gz_scores) if gz_scores else None
                avg_sp = np.mean(sp_scores) if sp_scores else None
                avg_wc = np.mean([d.get("wordcount_after", 0) - d.get("wordcount_before", 0) for d in valid_drafts])
                
                # Count zero-shot successes (only for enabled detectors)
                zero_shot_gz = sum(1 for d in valid_drafts if d["scores_after"]["group_doc"]["gptzero"] is not None and d["scores_after"]["group_doc"]["gptzero"] <= ZERO_SHOT_THRESHOLD)
                zero_shot_sp = sum(1 for d in valid_drafts if d["scores_after"]["group_doc"]["sapling"] is not None and d["scores_after"]["group_doc"]["sapling"] <= ZERO_SHOT_THRESHOLD)
                
                # Calculate average quality
                quality_scores = []
                grammar_scores = []
                for d in valid_drafts:
                    if not d.get("para_mismatch", False) and d.get("flag_counts"):
                        content_paras = d["flag_counts"].get("content_paragraph_count", para_total)
                        score = sum(d["flag_counts"].get(f, 0) for f in GEMINI_FLAGS) / (len(GEMINI_FLAGS) * content_paras) * 100 if content_paras else 0
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
                    "Avg GPTZero": f"{avg_gz:.3f}" if avg_gz is not None else "N/A",
                    "Δ GZ": avg_gz - baseline_gz if avg_gz is not None else 0,
                    "Zero-shot GZ": f"{zero_shot_gz}/{len(valid_drafts)}",
                    "Avg Sapling": f"{avg_sp:.3f}" if avg_sp is not None else "N/A",
                    "Δ SP": avg_sp - baseline_sp if avg_sp is not None else 0,
                    "Zero-shot SP": f"{zero_shot_sp}/{len(valid_drafts)}",
                    "Avg WC Δ": f"{avg_wc:+.0f}",
                    "Avg Quality": f"{avg_quality:.1f}%",
                    "Avg Grammar": f"{avg_grammar:.1f}/10" if avg_grammar is not None else "—"
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if not comparison_df.empty:
        st.markdown("### 📊 Model Performance Summary")
        
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
            _style_levels, subset=['Avg Grammar']
        ).format({
            'Δ GZ': '{:+.3f}',
            'Δ SP': '{:+.3f}'
        })
        
        st.dataframe(styled_comparison, use_container_width=True, hide_index=True)

    # NEW: Comprehensive Document Analysis with Paragraph-by-Paragraph Comparison
    st.divider()
    st.markdown("## 🔍 Detailed Document Analysis")
    
    # Create main analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Document Mode Analysis", 
        "📝 Paragraph Mode Analysis", 
        "📋 Paragraph Comparison",
        "📖 Full Document View",
        "🎯 Quality Deep Dive"
    ])
    
    with tab1:
        _render_document_mode_analysis(by_model, doc_name, baseline_gz, baseline_sp, para_total)
    
    with tab2:
        _render_paragraph_mode_analysis(by_model, doc_name, baseline_gz, baseline_sp, para_total)
    
    with tab3:
        _render_paragraph_comparison_view(by_model, doc_name, para_total)
    
    with tab4:
        _render_full_document_view(by_model, doc_name, para_total)
    
    with tab5:
        _render_quality_deep_dive(by_model, doc_name, para_total)


def _render_document_mode_analysis(by_model: Dict, doc_name: str, baseline_gz: float, baseline_sp: float, para_total: int):
    """Render detailed document mode analysis with paragraph-by-paragraph breakdowns"""
    st.markdown("### 📄 Document-Level Humanization Analysis")
    st.info("Document mode rewrites the entire document as one unit. Each draft shows paragraph-level effects of the global rewrite.")
    
    for model in sorted(by_model):
        if not by_model[model]["doc"]:
            continue
            
        st.markdown(f"#### 🤖 Model: **{model}**")
        model_drafts = by_model[model]["doc"]
        
        # Model summary metrics
        valid_drafts = [d for d in model_drafts if "scores_after" in d and "group_doc" in d["scores_after"]]
        if valid_drafts:
            # Filter out None values for disabled detectors
            gz_scores = [d["scores_after"]["group_doc"]["gptzero"] for d in valid_drafts if d["scores_after"]["group_doc"]["gptzero"] is not None]
            sp_scores = [d["scores_after"]["group_doc"]["sapling"] for d in valid_drafts if d["scores_after"]["group_doc"]["sapling"] is not None]
            
            avg_gz = np.mean(gz_scores) if gz_scores else None
            avg_sp = np.mean(sp_scores) if sp_scores else None
            
            # Count zero-shot successes (only for enabled detectors)
            zero_shot_count = sum(1 for d in valid_drafts if 
                                 (d["scores_after"]["group_doc"]["gptzero"] is not None and d["scores_after"]["group_doc"]["gptzero"] <= ZERO_SHOT_THRESHOLD) or
                                 (d["scores_after"]["group_doc"]["sapling"] is not None and d["scores_after"]["group_doc"]["sapling"] <= ZERO_SHOT_THRESHOLD))
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if avg_gz is not None:
                    colored_metric("Avg GPTZero", f"{avg_gz:.3f}", avg_gz - baseline_gz)
                else:
                    st.metric("Avg GPTZero", "N/A", help="GPTZero detector was disabled")
            with col2:
                if avg_sp is not None:
                    colored_metric("Avg Sapling", f"{avg_sp:.3f}", avg_sp - baseline_sp)
                else:
                    st.metric("Avg Sapling", "N/A", help="Sapling detector was disabled")
            with col3:
                st.metric("Zero-shot Success", f"{zero_shot_count}/{len(valid_drafts)}")
            with col4:
                st.metric("Total Iterations", len(model_drafts))
        
        # Individual draft analysis
        for draft_idx, draft in enumerate(sorted(model_drafts, key=lambda x: x.get("iter", 0))):
            iter_num = draft.get("iter", 0) + 1
            draft_title = f"🔄 **Iteration {iter_num}** - Document Mode"
            
            # Check for zero-shot success (handle None values for disabled detectors)
            gz_after = draft.get("scores_after", {}).get("group_doc", {}).get("gptzero")
            sp_after = draft.get("scores_after", {}).get("group_doc", {}).get("sapling")
            
            # Check zero-shot success only for enabled detectors
            is_zero_shot = False
            if gz_after is not None and gz_after <= ZERO_SHOT_THRESHOLD:
                is_zero_shot = True
            elif sp_after is not None and sp_after <= ZERO_SHOT_THRESHOLD:
                is_zero_shot = True
            
            if is_zero_shot:
                draft_title += " ✨ **ZERO-SHOT SUCCESS!**"
            
            st.markdown(draft_title)
            
            # Draft-level metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                if gz_after is not None:
                    colored_metric("GPTZero", f"{gz_after:.3f}", gz_after - baseline_gz)
                else:
                    st.metric("GPTZero", "N/A", help="GPTZero detector was disabled")
            with col2:
                if sp_after is not None:
                    colored_metric("Sapling", f"{sp_after:.3f}", sp_after - baseline_sp)
                else:
                    st.metric("Sapling", "N/A", help="Sapling detector was disabled")
            with col3:
                wc_delta = draft.get("wordcount_after", 0) - draft.get("wordcount_before", 0)
                st.metric("WC Change", f"{wc_delta:+d}")
            with col4:
                draft_length_dev = draft.get("draft_length_deviation", 0)
                st.metric("Length Dev", f"{draft_length_dev:+.1f}%")
            with col5:
                para_mismatch = draft.get("para_mismatch", False)
                mismatch_reason = draft.get("mismatch_reason")
                if para_mismatch:
                    st.error("Para Mismatch")
                    if mismatch_reason:
                        st.caption(f"{mismatch_reason}")
                else:
                    st.success("Para Match")
            
            # Paragraph breakdown for this draft
            if draft.get("paragraph_details") and not para_mismatch:
                _render_paragraph_breakdown_table(draft, "Document Mode")
                
        st.divider()


def _render_paragraph_mode_analysis(by_model: Dict, doc_name: str, baseline_gz: float, baseline_sp: float, para_total: int):
    """Render detailed paragraph mode analysis"""
    st.markdown("### 📝 Paragraph-Level Humanization Analysis")
    st.info("Paragraph mode rewrites each paragraph independently. This shows how individual paragraph changes affect the overall document.")
    
    for model in sorted(by_model):
        if not by_model[model]["para"]:
            continue
            
        st.markdown(f"#### 🤖 Model: **{model}**")
        model_drafts = by_model[model]["para"]
        
        # Model summary metrics (same as document mode)
        valid_drafts = [d for d in model_drafts if "scores_after" in d and "group_doc" in d["scores_after"]]
        if valid_drafts:
            # Filter out None values for disabled detectors
            gz_scores = [d["scores_after"]["group_doc"]["gptzero"] for d in valid_drafts if d["scores_after"]["group_doc"]["gptzero"] is not None]
            sp_scores = [d["scores_after"]["group_doc"]["sapling"] for d in valid_drafts if d["scores_after"]["group_doc"]["sapling"] is not None]
            
            avg_gz = np.mean(gz_scores) if gz_scores else None
            avg_sp = np.mean(sp_scores) if sp_scores else None
            
            # Count zero-shot successes (only for enabled detectors)
            zero_shot_count = sum(1 for d in valid_drafts if 
                                 (d["scores_after"]["group_doc"]["gptzero"] is not None and d["scores_after"]["group_doc"]["gptzero"] <= ZERO_SHOT_THRESHOLD) or
                                 (d["scores_after"]["group_doc"]["sapling"] is not None and d["scores_after"]["group_doc"]["sapling"] <= ZERO_SHOT_THRESHOLD))
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if avg_gz is not None:
                    colored_metric("Avg GPTZero", f"{avg_gz:.3f}", avg_gz - baseline_gz)
                else:
                    st.metric("Avg GPTZero", "N/A", help="GPTZero detector was disabled")
            with col2:
                if avg_sp is not None:
                    colored_metric("Avg Sapling", f"{avg_sp:.3f}", avg_sp - baseline_sp)
                else:
                    st.metric("Avg Sapling", "N/A", help="Sapling detector was disabled")
            with col3:
                st.metric("Zero-shot Success", f"{zero_shot_count}/{len(valid_drafts)}")
            with col4:
                st.metric("Total Iterations", len(model_drafts))
        
        # Individual draft analysis
        for draft_idx, draft in enumerate(sorted(model_drafts, key=lambda x: x.get("iter", 0))):
            iter_num = draft.get("iter", 0) + 1
            draft_title = f"🔄 **Iteration {iter_num}** - Paragraph Mode"
            
            # Check for zero-shot success (handle None values for disabled detectors)
            gz_after = draft.get("scores_after", {}).get("group_doc", {}).get("gptzero")
            sp_after = draft.get("scores_after", {}).get("group_doc", {}).get("sapling")
            
            # Check zero-shot success only for enabled detectors
            is_zero_shot = False
            if gz_after is not None and gz_after <= ZERO_SHOT_THRESHOLD:
                is_zero_shot = True
            elif sp_after is not None and sp_after <= ZERO_SHOT_THRESHOLD:
                is_zero_shot = True
            
            if is_zero_shot:
                draft_title += " ✨ **ZERO-SHOT SUCCESS!**"
            
            st.markdown(draft_title)
            
            # Draft-level metrics (same layout as document mode)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                if gz_after is not None:
                    colored_metric("GPTZero", f"{gz_after:.3f}", gz_after - baseline_gz)
                else:
                    st.metric("GPTZero", "N/A", help="GPTZero detector was disabled")
            with col2:
                if sp_after is not None:
                    colored_metric("Sapling", f"{sp_after:.3f}", sp_after - baseline_sp)
                else:
                    st.metric("Sapling", "N/A", help="Sapling detector was disabled")
            with col3:
                wc_delta = draft.get("wordcount_after", 0) - draft.get("wordcount_before", 0)
                st.metric("WC Change", f"{wc_delta:+d}")
            with col4:
                draft_length_dev = draft.get("draft_length_deviation", 0)
                st.metric("Length Dev", f"{draft_length_dev:+.1f}%")
            with col5:
                para_mismatch = draft.get("para_mismatch", False)
                mismatch_reason = draft.get("mismatch_reason")
                para_level_mismatches = draft.get("para_level_mismatches", 0)
                
                if para_mismatch:
                    st.error("Para Mismatch")
                    if mismatch_reason:
                        st.caption(f"{mismatch_reason}")
                elif para_level_mismatches > 0:
                    st.warning(f"Para Mismatches: {para_level_mismatches}")
                    st.caption("Some paragraphs 1→N")
                else:
                    st.success("Para Match")
            
            # Paragraph breakdown for this draft - now show even with mismatches
            if draft.get("paragraph_details"):
                _render_paragraph_breakdown_table(draft, "Paragraph Mode")
                
        st.divider()


def _render_paragraph_breakdown_table(draft: Dict, mode_name: str):
    """Render a detailed breakdown table for paragraph-level analysis"""
    para_details = draft.get("paragraph_details", [])
    if not para_details:
        st.warning("No paragraph details available")
        return
        
    st.markdown(f"##### 📋 Paragraph Breakdown - {mode_name}")
    
    # Get stored pair information for accurate mismatch display
    para_pair_info = draft.get("para_pair_info", [])
    
    # Build data for the table
    table_data = []
    content_para_idx = 0  # Track content paragraph index for pair info
    
    for detail in para_details:
        para_type = detail.get("type", "content")
        para_num = detail["paragraph"]
        
        # Basic metrics
        wc_before = detail["wc_before"]
        wc_after = detail["wc_after"]
        wc_delta = wc_after - wc_before
        wc_delta_pct = (wc_delta / wc_before * 100) if wc_before > 0 else 0
        
        # AI scores
        gz_before = detail["ai_before"].get("gptzero")
        gz_after = detail["ai_after"].get("gptzero")
        sp_before = detail["ai_before"].get("sapling")
        sp_after = detail["ai_after"].get("sapling")
        
        # Get mismatch information from stored pairs for content paragraphs
        if para_type == "content" and content_para_idx < len(para_pair_info):
            pair_info = para_pair_info[content_para_idx]
            is_mismatch = pair_info["is_mismatch"] if pair_info else False
            sent_count = pair_info["sent_count"] if pair_info else 1
            received_count = pair_info["received_count"] if pair_info else 1
        else:
            is_mismatch = False
            sent_count = 1
            received_count = 1
        
        row = {
            "Para #": para_num,
            "Type": "📋" if para_type == "heading" else "📄",
            "Match": "❌ Mismatch" if is_mismatch else "✅ Match",
            "Sent/Recv": f"{sent_count}→{received_count}" if para_type == "content" else "—",
            "WC Before": wc_before,
            "WC After": wc_after,
            "WC Δ": f"{wc_delta:+d}",
            "WC Δ %": f"{wc_delta_pct:+.1f}%",
            "GZ Before": f"{gz_before:.3f}" if gz_before is not None else "—",
            "GZ After": f"{gz_after:.3f}" if gz_after is not None else "—",
            "SP Before": f"{sp_before:.3f}" if sp_before is not None else "—",
            "SP After": f"{sp_after:.3f}" if sp_after is not None else "—",
        }
        
        if para_type == "content":
            # Quality metrics for content paragraphs
            flags = detail.get("flags", {})
            meaning_level = detail.get("same_meaning_level")
            missing_level = detail.get("missing_info_level")
            grammar_level = detail.get("grammar_level")
            
            row.update({
                "Meaning": f"{meaning_level:.1f}" if meaning_level is not None else "—",
                "Missing": f"{missing_level:.1f}" if missing_level is not None else "—", 
                "Grammar": f"{grammar_level:.1f}" if grammar_level is not None else "—",
                "Length OK": "✅" if flags.get("length_ok") else "❌",
                "Same Meaning": "✅" if flags.get("same_meaning") else "❌",
                "Same Lang": "✅" if flags.get("same_lang") else "❌",
                "No Missing": "✅" if flags.get("no_missing_info") else "❌",
                "Citation Pres": "✅" if flags.get("citation_preserved") else "❌",
                "Citation OK": "✅" if flags.get("citation_content_ok") else "❌",
            })
            content_para_idx += 1  # Increment for next content paragraph
        else:
            # Empty metrics for headings
            row.update({
                "Meaning": "—", "Missing": "—", "Grammar": "—",
                "Length OK": "—", "Same Meaning": "—", "Same Lang": "—",
                "No Missing": "—", "Citation Pres": "—", "Citation OK": "—",
            })
        
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Style the dataframe
    def style_wc_delta(val):
        if isinstance(val, str) and ('+' in val or '-' in val):
            return 'color: red' if '+' in val else 'color: green'
        return ''
    
    def style_type(val):
        if val == "📋":
            return 'background-color: #e8f4f8; font-weight: bold'
        return 'background-color: #f8f8f8'
    
    def style_quality_flags(val):
        if val == "✅":
            return 'color: green; font-weight: bold'
        elif val == "❌":
            return 'color: red; font-weight: bold'
        return 'color: gray'
    
    def style_levels(val):
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
    
    def style_match_status(val):
        if "✅" in str(val):
            return 'color: green; font-weight: bold'
        elif "❌" in str(val):
            return 'color: red; font-weight: bold'
        return ''
    
    def style_sent_recv(val):
        if isinstance(val, str) and "→" in val:
            parts = val.split("→")
            if len(parts) == 2:
                try:
                    sent, recv = int(parts[0]), int(parts[1])
                    if recv != sent:
                        return 'color: orange; font-weight: bold'
                except:
                    pass
        return ''
    
    styled_df = (df.style
                 .applymap(style_type, subset=['Type'])
                 .applymap(style_match_status, subset=['Match'])
                 .applymap(style_sent_recv, subset=['Sent/Recv'])
                 .applymap(style_wc_delta, subset=['WC Δ', 'WC Δ %'])
                 .applymap(style_levels, subset=['Meaning', 'Missing', 'Grammar'])
                 .applymap(style_quality_flags, subset=['Length OK', 'Same Meaning', 'Same Lang', 'No Missing', 'Citation Pres', 'Citation OK']))
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def _render_paragraph_comparison_view(by_model: Dict, doc_name: str, para_total: int):
    """Render side-by-side paragraph comparison view with actual original text extraction"""
    st.markdown("### 📋 Paragraph-by-Paragraph Comparison")
    st.info("Compare original vs humanized text for each paragraph with detailed quality analysis")
    
    # Model and draft selection
    available_models = [m for m in sorted(by_model) if by_model[m]["doc"] or by_model[m]["para"]]
    if not available_models:
        st.warning("No drafts available for comparison")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_model = st.selectbox("Select Model", available_models, key="para_comp_model")
    
    available_modes = []
    if by_model[selected_model]["doc"]:
        available_modes.append("doc")
    if by_model[selected_model]["para"]:
        available_modes.append("para")
    
    with col2:
        selected_mode = st.selectbox("Select Mode", available_modes, key="para_comp_mode")
    
    available_drafts = by_model[selected_model][selected_mode]
    available_iters = [f"Iteration {d.get('iter', 0) + 1}" for d in available_drafts]
    
    with col3:
        selected_iter_label = st.selectbox("Select Draft", available_iters, key="para_comp_iter")
        selected_iter = int(selected_iter_label.split()[-1]) - 1
    
    # Get the selected draft
    selected_draft = next((d for d in available_drafts if d.get("iter", 0) == selected_iter), None)
    if not selected_draft:
        st.error("Selected draft not found")
        return
    
    # Draft overview
    st.markdown(f"#### 🔍 Analyzing: **{selected_model}** - **{selected_mode.title()} Mode** - **Iteration {selected_iter + 1}**")
    
    # Quick draft metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gz_after = selected_draft.get("scores_after", {}).get("group_doc", {}).get("gptzero")
        if gz_after is not None:
            st.metric("GPTZero", f"{gz_after:.3f}")
        else:
            st.metric("GPTZero", "N/A", help="GPTZero detector was disabled")
    with col2:
        sp_after = selected_draft.get("scores_after", {}).get("group_doc", {}).get("sapling")
        if sp_after is not None:
            st.metric("Sapling", f"{sp_after:.3f}")
        else:
            st.metric("Sapling", "N/A", help="Sapling detector was disabled")
    with col3:
        wc_delta = selected_draft.get("wordcount_after", 0) - selected_draft.get("wordcount_before", 0)
        st.metric("WC Change", f"{wc_delta:+d}")
    with col4:
        para_mismatch = selected_draft.get("para_mismatch", False)
        if para_mismatch:
            st.error("Para Mismatch")
        else:
            st.success("Para Match")
    
    # Extract original text from document
    original_paragraphs = []
    doc_folder = None
    
    # Try to determine document folder and extract original text
    try:
        from pathlib import Path
        from src.docx_utils import extract_paragraphs_with_type
        from src.paths import ROOT
        
        # First, try to determine the folder from any draft in this document
        all_drafts = []
        for model in by_model:
            all_drafts.extend(by_model[model]["doc"])
            all_drafts.extend(by_model[model]["para"])
        
        if all_drafts:
            # Get folder info from any available draft that has it
            sample_draft = next((d for d in all_drafts if d.get("scores_before")), None)
            if sample_draft:
                # Try to infer folder from document name patterns or check common folders
                for folder_name in ["ai_texts", "human_texts", "ai_paras", "human_paras"]:
                    doc_path = ROOT / "data" / folder_name / doc_name
                    if doc_path.exists():
                        doc_folder = folder_name
                        break
                
                if doc_folder:
                    doc_path = ROOT / "data" / doc_folder / doc_name
                    original_paragraphs = extract_paragraphs_with_type(doc_path)
                    st.success(f"✅ Original text loaded from {doc_folder}/{doc_name}")
                else:
                    st.warning(f"⚠️ Could not locate original document {doc_name} in data folders")
            else:
                st.warning("⚠️ No draft information available to determine document location")
        
    except Exception as e:
        st.error(f"❌ Error loading original document: {str(e)}")
        original_paragraphs = []
    
    # Get humanized text segments
    humanized_text = selected_draft.get("humanized_text", "")
    if humanized_text:
        # Split humanized text by double newlines (paragraph separator)
        humanized_segments = []
        for segment in humanized_text.split('\n\n'):
            segment = segment.strip()
            if segment:
                humanized_segments.append(segment)
    else:
        humanized_segments = []
    
    # Always show paragraph-by-paragraph comparison (NEW: removed para_mismatch check)
    st.divider()
    st.markdown("##### 📝 Paragraph-by-Paragraph Analysis")
    
    # Handle different scenarios
    if selected_draft.get("paragraph_details"):
        para_details = selected_draft["paragraph_details"]
        
        # Get the stored paragraph pairs for accurate comparison
        para_pair_info = selected_draft.get("para_pair_info", [])
        
        # Create comparison for each paragraph
        for idx, detail in enumerate(para_details):
            para_num = detail["paragraph"]
            para_type = detail.get("type", "content")
            
            # Header for this paragraph
            if para_type == "heading":
                st.markdown(f"### 📋 Paragraph {para_num} (Heading)")
            else:
                st.markdown(f"### 📄 Paragraph {para_num} (Content)")
            
            # Metrics row
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
            with col1:
                wc_delta = detail['wc_after'] - detail['wc_before']
                wc_delta_pct = (wc_delta / detail['wc_before'] * 100) if detail['wc_before'] > 0 else 0
                st.caption(f"WC: {detail['wc_before']} → {detail['wc_after']} ({wc_delta_pct:+.1f}%)")
            with col2:
                gz_before = detail["ai_before"].get("gptzero")
                gz_after = detail["ai_after"].get("gptzero")
                if gz_before is not None and gz_after is not None:
                    delta = gz_after - gz_before
                    color = "🟢" if delta < -0.1 else "🟡" if abs(delta) < 0.1 else "🔴"
                    st.caption(f"GZ: {gz_before:.3f} → {gz_after:.3f} {color}")
                else:
                    st.caption("GZ: —")
            with col3:
                sp_before = detail["ai_before"].get("sapling")
                sp_after = detail["ai_after"].get("sapling")
                if sp_before is not None and sp_after is not None:
                    delta = sp_after - sp_before
                    color = "🟢" if delta < -0.1 else "🟡" if abs(delta) < 0.1 else "🔴"
                    st.caption(f"SP: {sp_before:.3f} → {sp_after:.3f} {color}")
                else:
                    st.caption("SP: —")
            with col4:
                if para_type == "content":
                    meaning_level = detail.get("same_meaning_level")
                    if meaning_level is not None:
                        color = "🟢" if meaning_level >= 8 else "🟡" if meaning_level >= 6 else "🔴"
                        st.caption(f"Meaning: {color} {meaning_level:.1f}/10")
                    else:
                        st.caption("Meaning: —")
                else:
                    st.caption("Meaning: N/A")
            with col5:
                if para_type == "content":
                    missing_level = detail.get("missing_info_level")
                    if missing_level is not None:
                        color = "🟢" if missing_level <= 2 else "🟡" if missing_level <= 4 else "🔴"
                        st.caption(f"Missing: {color} {missing_level:.1f}/10")
                    else:
                        st.caption("Missing: —")
                else:
                    st.caption("Missing: N/A")
            with col6:
                if para_type == "content":
                    grammar_level = detail.get("grammar_level")
                    if grammar_level is not None:
                        color = "🟢" if grammar_level >= 8 else "🟡" if grammar_level >= 6 else "🔴"
                        st.caption(f"Grammar: {color} {grammar_level:.1f}/10")
                    else:
                        st.caption("Grammar: —")
                else:
                    st.caption("Grammar: N/A")
            with col7:
                if para_type == "content":
                    flags = detail.get("flags", {})
                    citation_preserved = flags.get("citation_preserved")
                    if citation_preserved is not None:
                        color = "🟢" if citation_preserved else "🔴"
                        status = "✓" if citation_preserved else "✗"
                        st.caption(f"Cit Pres: {color} {status}")
                    else:
                        st.caption("Cit Pres: —")
                else:
                    st.caption("Cit Pres: N/A")
            with col8:
                if para_type == "content":
                    flags = detail.get("flags", {})
                    citation_content_ok = flags.get("citation_content_ok")
                    if citation_content_ok is not None:
                        color = "🟢" if citation_content_ok else "🔴"
                        status = "✓" if citation_content_ok else "✗"
                        st.caption(f"Cit Content: {color} {status}")
                    else:
                        st.caption("Cit Content: —")
                else:
                    st.caption("Cit Content: N/A")
            
            # Text comparison (side by side) - using stored pairs
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📄 Original:**")
                
                # Get original text using consistent pairing system
                if para_type == "content":
                    # Find the corresponding pair info
                    content_para_idx = sum(1 for i in range(idx) if para_details[i].get("type") == "content")
                    
                    if content_para_idx < len(para_pair_info) and para_pair_info[content_para_idx]:
                        original_text = para_pair_info[content_para_idx]["original_paragraph"]
                    else:
                        # Fallback: try to get from original document if available
                        if original_paragraphs and idx < len(original_paragraphs):
                            original_text = original_paragraphs[idx]['text']
                        else:
                            original_text = "Original content paragraph not available"
                    
                    st.text_area(
                        f"Original paragraph {para_num}",
                        original_text,
                        height=max(150, len(original_text) // 5),  # Dynamic height based on text length
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"orig_{para_num}_{selected_model}_{selected_mode}_{selected_iter}"
                    )
                elif para_type == "heading":
                    # For headings, try to get from original document if available
                    if original_paragraphs and idx < len(original_paragraphs):
                        original_text = original_paragraphs[idx]['text']
                    else:
                        original_text = "Original heading text not available"
                    
                    st.text_area(
                        f"Original paragraph {para_num}",
                        original_text,
                        height=max(150, len(original_text) // 5),
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"orig_head_{para_num}_{selected_model}_{selected_mode}_{selected_iter}"
                    )
            
            with col2:
                # Display humanized text using consistent pairing system
                if para_type == "content":
                    # Use the stored paragraph pair information directly from detail
                    pair_original = detail.get("para_original_paragraph", "")
                    pair_quality_text = detail.get("para_quality_evaluation_text", "")
                    is_mismatch = detail.get("para_is_mismatch", False)
                    received_count = detail.get("para_received_count", 1)
                    
                    # Use the quality evaluation text which contains the full humanized output
                    full_humanized_text = pair_quality_text
                    
                    # If no pair info is stored in details, try to find it in para_pair_info
                    if not full_humanized_text and para_pair_info:
                        # Find the corresponding pair by matching original text
                        for pair_info in para_pair_info:
                            if pair_info and pair_info.get("original_paragraph") == original_text:
                                full_humanized_text = pair_info.get("document_assembly_text", "")
                                is_mismatch = pair_info.get("is_mismatch", False)
                                received_count = pair_info.get("received_count", 1)
                                break
                    
                    # If still no text found, use fallback
                    if not full_humanized_text:
                        if idx < len(humanized_segments):
                            full_humanized_text = humanized_segments[idx]
                        else:
                            full_humanized_text = "Humanized content not available"

                    if is_mismatch:
                        st.markdown(f"**✨ Humanized (❌ MISMATCH - Received {received_count} paragraphs):**")
                    else:
                        st.markdown(f"**✨ Humanized (✅ MATCH):**")
                    
                    st.text_area(
                        f"Humanized paragraph {para_num}",
                        full_humanized_text,
                        height=max(150, len(full_humanized_text) // 5),  # Dynamic height
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"hum_{para_num}_{selected_model}_{selected_mode}_{selected_iter}"
                    )

                    if is_mismatch:
                        st.error("⚠️ **Paragraph Mismatch:** Sent 1 paragraph, received multiple. Quality evaluation performed on the full combined text.")
                        
                else:
                    # For headings, they typically don't get humanized in para mode
                    st.markdown(f"**✨ Humanized (Heading - Usually Unchanged):**")
                    
                    # For headings, always use the original text since they don't get humanized
                    # Don't use humanized_segments indexing as it can be incorrect due to structure mismatches
                    if original_paragraphs and idx < len(original_paragraphs):
                        humanized_text_segment = original_paragraphs[idx]['text']
                    elif len(original_text.strip()) > 0:
                        # Use the original text from para_details if available
                        humanized_text_segment = original_text
                    else:
                        humanized_text_segment = "Heading text not available"
                    
                    st.text_area(
                        f"Humanized paragraph {para_num}",
                        humanized_text_segment,
                        height=max(150, len(humanized_text_segment) // 5),  # Dynamic height
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"hum_head_{para_num}_{selected_model}_{selected_mode}_{selected_iter}"
                    )
            
            # Quality details for content paragraphs
            if para_type == "content":
                # Create sub-tabs for different types of quality information
                detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs(["🔤 Grammar", "💭 Meaning", "📋 Information", "📖 Citations"])
                
                with detail_tab1:
                    # Grammar errors and analysis
                    grammar_errors = detail.get("grammar_errors", [])
                    grammar_level = detail.get("grammar_level")
                    
                    if grammar_level is not None:
                        if grammar_level >= 8:
                            st.success(f"✅ **Excellent Grammar** (Score: {grammar_level:.1f}/10)")
                        elif grammar_level >= 6:
                            st.warning(f"⚠️ **Good Grammar** (Score: {grammar_level:.1f}/10)")
                        else:
                            st.error(f"❌ **Poor Grammar** (Score: {grammar_level:.1f}/10)")
                    else:
                        st.info("Grammar level not evaluated")
                    
                    if grammar_errors:
                        st.markdown(f"**Grammar Issues Found:** {len(grammar_errors)}")
                        for i, error in enumerate(grammar_errors, 1):
                            st.markdown(f"{i}. {error}")
                    else:
                        st.success("✅ No grammar issues detected")
                
                with detail_tab2:
                    # Meaning preservation analysis
                    meaning_level = detail.get("same_meaning_level")
                    meaning_details = detail.get("same_meaning_details", "")
                    
                    if meaning_level is not None:
                        if meaning_level >= 8:
                            st.success(f"✅ **Excellent Meaning Preservation** (Score: {meaning_level:.1f}/10)")
                        elif meaning_level >= 6:
                            st.warning(f"⚠️ **Good Meaning Preservation** (Score: {meaning_level:.1f}/10)")
                        else:
                            st.error(f"❌ **Poor Meaning Preservation** (Score: {meaning_level:.1f}/10)")
                    else:
                        st.info("Meaning level not evaluated")
                    
                    if meaning_details:
                        st.markdown("**Analysis Details:**")
                        st.markdown(meaning_details)
                    else:
                        st.info("No detailed meaning analysis available")
                
                with detail_tab3:
                    # Information changes (missing/added)
                    missing_level = detail.get("missing_info_level")
                    missing_items = detail.get("missing_items", [])
                    added_items = detail.get("added_items", [])
                    
                    if missing_level is not None:
                        if missing_level <= 2:
                            st.success(f"✅ **Excellent Information Retention** (Missing Info Level: {missing_level:.1f}/10)")
                        elif missing_level <= 4:
                            st.warning(f"⚠️ **Good Information Retention** (Missing Info Level: {missing_level:.1f}/10)")
                        else:
                            st.error(f"❌ **Poor Information Retention** (Missing Info Level: {missing_level:.1f}/10)")
                    else:
                        st.info("Missing information level not evaluated")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if missing_items:
                            st.markdown("**⚠️ Missing Information:**")
                            for item in missing_items:
                                st.markdown(f"• {item}")
                        else:
                            st.success("✅ No missing information detected")
                    
                    with col2:
                        if added_items:
                            st.markdown("**➕ Added Information:**")
                            for item in added_items:
                                st.markdown(f"• {item}")
                        else:
                            st.info("ℹ️ No additional information added")
                
                with detail_tab4:
                    # Citation analysis and preservation
                    flags = detail.get("flags", {})
                    citation_preserved = flags.get("citation_preserved")
                    citation_content_ok = flags.get("citation_content_ok")
                    
                    # Extract original and humanized text to analyze citations
                    if original_paragraphs and idx < len(original_paragraphs):
                        original_text = original_paragraphs[idx]['text']
                    else:
                        original_text = ""
                    
                    if idx < len(humanized_segments):
                        humanized_text_seg = humanized_segments[idx]
                    else:
                        humanized_text_seg = ""
                    
                    # Use the same citation extraction as in quality.py
                    try:
                        from src.evaluation.quality import _citations
                        original_citations = _citations(original_text) if original_text else []
                        humanized_citations = _citations(humanized_text_seg) if humanized_text_seg else []
                    except:
                        # Fallback simple citation extraction
                        import re
                        citation_re = re.compile(r'\(([^()]{1,100}?)\)')
                        original_citations = citation_re.findall(original_text) if original_text else []
                        humanized_citations = citation_re.findall(humanized_text_seg) if humanized_text_seg else []
                    
                    # Citation preservation status
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📖 Citation Preservation:**")
                        if citation_preserved is not None:
                            if citation_preserved:
                                st.success("✅ **Citations Preserved**")
                            else:
                                st.error("❌ **Citations Not Preserved**")
                        else:
                            st.info("Citation preservation not evaluated")
                    
                    with col2:
                        st.markdown("**🔍 Citation Content Quality:**")
                        if citation_content_ok is not None:
                            if citation_content_ok:
                                st.success("✅ **Citation Content OK**")
                            else:
                                st.error("❌ **Citation Content Issues**")
                        else:
                            st.info("Citation content not evaluated")
                    
                    # Detailed citation analysis
                    st.markdown("**Citation Analysis:**")
                    
                    if original_citations or humanized_citations:
                        # Show citation comparison
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Original Citations ({len(original_citations)}):**")
                            if original_citations:
                                for i, cite in enumerate(original_citations, 1):
                                    st.markdown(f"{i}. ({cite})")
                            else:
                                st.info("No citations in original")
                        
                        with col2:
                            st.markdown(f"**Humanized Citations ({len(humanized_citations)}):**")
                            if humanized_citations:
                                for i, cite in enumerate(humanized_citations, 1):
                                    # Check if this citation was in the original
                                    if cite in original_citations:
                                        st.markdown(f"{i}. ✅ ({cite})")
                                    else:
                                        st.markdown(f"{i}. ⚠️ ({cite}) *[New/Modified]*")
                            else:
                                st.info("No citations in humanized")
                        
                        # Citation statistics
                        if original_citations:
                            preserved_count = sum(1 for cite in original_citations if cite in humanized_citations)
                            preservation_rate = (preserved_count / len(original_citations)) * 100
                            
                            st.markdown("**📊 Citation Statistics:**")
                            st.markdown(f"• **Preservation Rate:** {preservation_rate:.1f}% ({preserved_count}/{len(original_citations)})")
                            
                            if preserved_count < len(original_citations):
                                missing_citations = [cite for cite in original_citations if cite not in humanized_citations]
                                st.markdown("• **Missing Citations:**")
                                for cite in missing_citations:
                                    st.markdown(f"  - ({cite})")
                            
                            if len(humanized_citations) > preserved_count:
                                new_citations = [cite for cite in humanized_citations if cite not in original_citations]
                                st.markdown("• **New/Modified Citations:**")
                                for cite in new_citations:
                                    st.markdown(f"  - ({cite})")
                    else:
                        st.info("📝 No citations detected in this paragraph")
            
            else:
                # For headings, show a note that quality metrics don't apply
                st.info("📋 **Heading paragraph** - Quality metrics are not calculated for headings as they are not humanized.")
            
            st.divider()
    else:
        # NEW: Show basic comparison even without detailed paragraph analysis
        st.warning("⚠️ **Detailed paragraph analysis not available for this draft**")
        
        # Show basic document-level comparison if we have the texts
        if para_mismatch:
            mismatch_reason = selected_draft.get("mismatch_reason")
            st.info(f"**Document structure mismatch detected:** {mismatch_reason}")
            st.info("This occurs when the humanized document has a different paragraph structure than the original.")
        
        # Try to show basic original vs humanized comparison
        if original_paragraphs and humanized_segments:
            st.markdown("#### 📋 Basic Document Comparison")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📄 Original Document:**")
                orig_text = "\n\n".join([p['text'] for p in original_paragraphs])
                st.text_area(
                    "Original full text",
                    orig_text,
                    height=400,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"basic_orig_{selected_model}_{selected_mode}_{selected_iter}"
                )
            
            with col2:
                st.markdown("**✨ Humanized Document:**")
                hum_text = "\n\n".join(humanized_segments)
                st.text_area(
                    "Humanized full text",
                    hum_text,
                    height=400,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"basic_hum_{selected_model}_{selected_mode}_{selected_iter}"
                )


def _render_full_document_view(by_model: Dict, doc_name: str, para_total: int):
    """Render full original vs humanized document comparison"""
    st.markdown("### 📖 Complete Document Comparison")
    st.info("Compare the entire original document with humanized versions. Perfect for understanding overall changes and document flow.")
    
    # Model and draft selection (same as paragraph comparison)
    available_models = [m for m in sorted(by_model) if by_model[m]["doc"] or by_model[m]["para"]]
    if not available_models:
        st.warning("No drafts available for comparison")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_model = st.selectbox("Select Model", available_models, key="full_doc_model")
    
    available_modes = []
    if by_model[selected_model]["doc"]:
        available_modes.append("doc")
    if by_model[selected_model]["para"]:
        available_modes.append("para")
    
    with col2:
        selected_mode = st.selectbox("Select Mode", available_modes, key="full_doc_mode")
    
    available_drafts = by_model[selected_model][selected_mode]
    available_iters = [f"Iteration {d.get('iter', 0) + 1}" for d in available_drafts]
    
    with col3:
        selected_iter_label = st.selectbox("Select Draft", available_iters, key="full_doc_iter")
        selected_iter = int(selected_iter_label.split()[-1]) - 1
    
    # Get the selected draft
    selected_draft = next((d for d in available_drafts if d.get("iter", 0) == selected_iter), None)
    if not selected_draft:
        st.error("Selected draft not found")
        return
    
    # Draft overview metrics
    st.markdown(f"#### 📊 Document Analysis: **{selected_model}** - **{selected_mode.title()} Mode** - **Iteration {selected_iter + 1}**")
    
    # Key metrics in a row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        gz_after = selected_draft.get("scores_after", {}).get("group_doc", {}).get("gptzero")
        if gz_after is not None:
            st.metric("GPTZero", f"{gz_after:.3f}")
        else:
            st.metric("GPTZero", "N/A", help="GPTZero detector was disabled")
    with col2:
        sp_after = selected_draft.get("scores_after", {}).get("group_doc", {}).get("sapling")
        if sp_after is not None:
            st.metric("Sapling", f"{sp_after:.3f}")
        else:
            st.metric("Sapling", "N/A", help="Sapling detector was disabled")
    with col3:
        wc_before = selected_draft.get("wordcount_before", 0)
        wc_after = selected_draft.get("wordcount_after", 0)
        wc_delta = wc_after - wc_before
        st.metric("Word Count", f"{wc_after:,}", delta=f"{wc_delta:+d}")
    with col4:
        draft_length_dev = selected_draft.get("draft_length_deviation", 0)
        st.metric("Length Dev", f"{draft_length_dev:+.1f}%")
    with col5:
        para_count_before = selected_draft.get("para_count_before", para_total)
        para_count_after = selected_draft.get("para_count_after", para_total)
        st.metric("Paragraphs", f"{para_count_after}", delta=f"{para_count_after - para_count_before:+d}")
    with col6:
        para_mismatch = selected_draft.get("para_mismatch", False)
        mismatch_reason = selected_draft.get("mismatch_reason")
        if para_mismatch:
            st.error("❌ Mismatch")
            if mismatch_reason:
                st.caption(f"{mismatch_reason}")
        else:
            # Check for zero-shot success (only for enabled detectors)
            is_zero_shot = False
            if gz_after is not None and gz_after <= ZERO_SHOT_THRESHOLD:
                is_zero_shot = True
            elif sp_after is not None and sp_after <= ZERO_SHOT_THRESHOLD:
                is_zero_shot = True
            
            if is_zero_shot:
                st.success("✨ Zero-shot!")
            else:
                st.success("✅ Match")
    
    # Load original document
    original_text = ""
    try:
        from pathlib import Path
        from src.docx_utils import extract_paragraphs_with_type
        from src.paths import ROOT
        
        # Try to find the document in common folders
        doc_folder = None
        for folder_name in ["ai_texts", "human_texts", "ai_paras", "human_paras"]:
            doc_path = ROOT / "data" / folder_name / doc_name
            if doc_path.exists():
                doc_folder = folder_name
                break
        
        if doc_folder:
            doc_path = ROOT / "data" / doc_folder / doc_name
            original_paragraphs = extract_paragraphs_with_type(doc_path)
            original_text = "\n\n".join([p["text"] for p in original_paragraphs])
            st.success(f"✅ Original document loaded from **{doc_folder}**")
        else:
            st.warning(f"⚠️ Could not locate original document **{doc_name}** in data folders")
            original_text = "Original document not found in standard data folders."
            
    except Exception as e:
        st.error(f"❌ Error loading original document: {str(e)}")
        original_text = f"Error loading original document: {str(e)}"
    
    # Get humanized text
    humanized_text = selected_draft.get("humanized_text", "No humanized text available")
    
    # Document statistics comparison
    st.divider()
    st.markdown("##### 📈 Document Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📄 Original Document**")
        orig_word_count = len(original_text.split()) if original_text else 0
        orig_char_count = len(original_text) if original_text else 0
        orig_para_count = original_text.count('\n\n') + 1 if original_text else 0
        st.metric("Words", f"{orig_word_count:,}")
        st.metric("Characters", f"{orig_char_count:,}")
        st.metric("Paragraphs", orig_para_count)
        
    with col2:
        st.markdown("**✨ Humanized Document**")
        hum_word_count = len(humanized_text.split()) if humanized_text else 0
        hum_char_count = len(humanized_text) if humanized_text else 0
        hum_para_count = humanized_text.count('\n\n') + 1 if humanized_text else 0
        st.metric("Words", f"{hum_word_count:,}")
        st.metric("Characters", f"{hum_char_count:,}")
        st.metric("Paragraphs", hum_para_count)
        
    with col3:
        st.markdown("**📊 Changes**")
        word_change = hum_word_count - orig_word_count
        char_change = hum_char_count - orig_char_count
        para_change = hum_para_count - orig_para_count
        
        word_change_pct = (word_change / orig_word_count * 100) if orig_word_count > 0 else 0
        char_change_pct = (char_change / orig_char_count * 100) if orig_char_count > 0 else 0
        
        st.metric("Word Δ", f"{word_change:+d}", delta=f"{word_change_pct:+.1f}%")
        st.metric("Char Δ", f"{char_change:+d}", delta=f"{char_change_pct:+.1f}%")
        st.metric("Para Δ", f"{para_change:+d}")
    
    # Main document comparison
    st.divider()
    st.markdown("##### 📖 Side-by-Side Document Comparison")
    
    # Document display options
    col1, col2 = st.columns([3, 1])
    with col2:
        show_line_numbers = st.checkbox("Show line numbers", value=False, key="full_doc_line_nums")
        wrap_text = st.checkbox("Wrap long lines", value=True, key="full_doc_wrap")
        sync_scroll = st.checkbox("Synchronized scrolling", value=False, key="full_doc_sync")
        if sync_scroll:
            st.caption("⚠️ Sync scrolling requires manual coordination")
    
    # Calculate appropriate height based on document length
    max_lines = max(
        original_text.count('\n') + 5 if original_text else 10,
        humanized_text.count('\n') + 5 if humanized_text else 10
    )
    text_height = min(max(400, max_lines * 20), 1200)  # Between 400px and 1200px
    
    # Side-by-side text areas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📄 Original Document**")
        if show_line_numbers and original_text:
            # Add line numbers
            original_lines = original_text.split('\n')
            numbered_original = '\n'.join([f"{i+1:3d}: {line}" for i, line in enumerate(original_lines)])
        else:
            numbered_original = original_text
            
        st.text_area(
            f"Original - {doc_name}",
            numbered_original,
            height=text_height,
            disabled=True,
            label_visibility="collapsed",
            key=f"full_orig_{selected_model}_{selected_mode}_{selected_iter}",
            help=f"Original document from data folder. Word count: {orig_word_count:,}"
        )
    
    with col2:
        st.markdown("**✨ Humanized Document**")
        if show_line_numbers and humanized_text:
            # Add line numbers
            humanized_lines = humanized_text.split('\n')
            numbered_humanized = '\n'.join([f"{i+1:3d}: {line}" for i, line in enumerate(humanized_lines)])
        else:
            numbered_humanized = humanized_text
            
        st.text_area(
            f"Humanized - {selected_model} ({selected_mode.title()} Mode)",
            numbered_humanized,
            height=text_height,
            disabled=True,
            label_visibility="collapsed",
            key=f"full_hum_{selected_model}_{selected_mode}_{selected_iter}",
            help=f"Humanized by {selected_model} in {selected_mode} mode. Word count: {hum_word_count:,}"
        )
    
    # Quality summary (if available)
    if not para_mismatch and selected_draft.get("flag_counts"):
        st.divider()
        st.markdown("##### 🎯 Overall Quality Summary")
        
        flag_counts = selected_draft["flag_counts"]
        content_paras = flag_counts.get("content_paragraph_count", para_total)
        
        # Create quality metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**📊 Boolean Quality Metrics**")
            for flag in ["length_ok", "same_meaning", "same_lang", "no_missing_info"]:
                count = flag_counts.get(flag, 0)
                percentage = (count / content_paras * 100) if content_paras > 0 else 0
                flag_name = flag.replace('_', ' ').title()
                if flag == "no_missing_info":
                    flag_name = "No Missing Info"
                
                if percentage >= 80:
                    st.success(f"✅ {flag_name}: {percentage:.1f}%")
                elif percentage >= 60:
                    st.warning(f"⚠️ {flag_name}: {percentage:.1f}%")
                else:
                    st.error(f"❌ {flag_name}: {percentage:.1f}%")
        
        with col2:
            st.markdown("**📖 Citation Quality**")
            citation_preserved = flag_counts.get("citation_preserved", 0)
            citation_content_ok = flag_counts.get("citation_content_ok", 0)
            
            if content_paras > 0:
                cit_pres_pct = (citation_preserved / content_paras * 100)
                cit_content_pct = (citation_content_ok / content_paras * 100)
                
                if cit_pres_pct >= 95:
                    st.success(f"✅ Citations Preserved: {cit_pres_pct:.1f}%")
                elif cit_pres_pct >= 80:
                    st.warning(f"⚠️ Citations Preserved: {cit_pres_pct:.1f}%")
                else:
                    st.error(f"❌ Citations Preserved: {cit_pres_pct:.1f}%")
                
                if cit_content_pct >= 95:
                    st.success(f"✅ Citation Content OK: {cit_content_pct:.1f}%")
                elif cit_content_pct >= 80:
                    st.warning(f"⚠️ Citation Content OK: {cit_content_pct:.1f}%")
                else:
                    st.error(f"❌ Citation Content OK: {cit_content_pct:.1f}%")
            else:
                st.info("No content paragraphs for citation analysis")
        
        with col3:
            st.markdown("**🎭 Advanced Quality Levels**")
            same_meaning_level = flag_counts.get("same_meaning_level_avg")
            missing_info_level = flag_counts.get("missing_info_level_avg")
            grammar_score = flag_counts.get("grammar_score")
            
            if same_meaning_level is not None:
                if same_meaning_level >= 8:
                    st.success(f"✅ Meaning Level: {same_meaning_level:.1f}/10")
                elif same_meaning_level >= 6:
                    st.warning(f"⚠️ Meaning Level: {same_meaning_level:.1f}/10")
                else:
                    st.error(f"❌ Meaning Level: {same_meaning_level:.1f}/10")
            else:
                st.info("Meaning level not available")
            
            if missing_info_level is not None:
                if missing_info_level <= 2:
                    st.success(f"✅ Missing Info: {missing_info_level:.1f}/10")
                elif missing_info_level <= 4:
                    st.warning(f"⚠️ Missing Info: {missing_info_level:.1f}/10")
                else:
                    st.error(f"❌ Missing Info: {missing_info_level:.1f}/10")
            else:
                st.info("Missing info level not available")
            
            if grammar_score is not None:
                if grammar_score >= 8:
                    st.success(f"✅ Grammar: {grammar_score:.1f}/10")
                elif grammar_score >= 6:
                    st.warning(f"⚠️ Grammar: {grammar_score:.1f}/10")
                else:
                    st.error(f"❌ Grammar: {grammar_score:.1f}/10")
            else:
                st.info("Grammar score not available")
        
        with col4:
            st.markdown("**📏 Length & Structure**")
            para_length_dev = flag_counts.get("para_length_deviation_avg", 0)
            draft_length_dev = selected_draft.get("draft_length_deviation", 0)
            
            # Document length deviation
            if abs(draft_length_dev) <= 10:
                st.success(f"✅ Doc Length Δ: {draft_length_dev:+.1f}%")
            elif abs(draft_length_dev) <= 20:
                st.warning(f"⚠️ Doc Length Δ: {draft_length_dev:+.1f}%")
            else:
                st.error(f"❌ Doc Length Δ: {draft_length_dev:+.1f}%")
            
            # Paragraph length deviation
            if abs(para_length_dev) <= 10:
                st.success(f"✅ Para Length Δ: {para_length_dev:+.1f}%")
            elif abs(para_length_dev) <= 20:
                st.warning(f"⚠️ Para Length Δ: {para_length_dev:+.1f}%")
            else:
                st.error(f"❌ Para Length Δ: {para_length_dev:+.1f}%")
            
            # Paragraph count
            if para_change == 0:
                st.success(f"✅ Paragraph Count: {hum_para_count}")
            else:
                st.error(f"❌ Para Count: {hum_para_count} ({para_change:+d})")
    
    elif para_mismatch:
        st.warning("⚠️ **Quality metrics unavailable due to paragraph mismatch**")
        if selected_draft.get("mismatch_reason"):
            st.warning(f"**Mismatch details:** {selected_draft.get('mismatch_reason')}")
        st.info("This occurs when the humanized document has a different paragraph structure than the original.")
    else:
        st.info("ℹ️ **Quality analysis not available for this draft**")


def _render_quality_deep_dive(by_model: Dict, doc_name: str, para_total: int):
    """Render comprehensive quality analysis across all drafts"""
    st.markdown("### 🎯 Quality Analysis Deep Dive")
    st.info("Comprehensive quality analysis across all models and modes showing patterns in meaning preservation, grammar, and information retention")
    
    # Collect all quality data
    quality_data = []
    
    for model in sorted(by_model):
        for mode in ["doc", "para"]:
            if not by_model[model][mode]:
                continue
                
            for draft in by_model[model][mode]:
                if draft.get("para_mismatch", False) or not draft.get("paragraph_details"):
                    continue
                    
                iter_num = draft.get("iter", 0) + 1
                
                # Aggregate draft-level metrics
                para_details = draft["paragraph_details"]
                content_paras = [p for p in para_details if p.get("type") == "content"]
                
                if content_paras:
                    # Calculate averages
                    meaning_levels = [p.get("same_meaning_level") for p in content_paras if p.get("same_meaning_level") is not None]
                    missing_levels = [p.get("missing_info_level") for p in content_paras if p.get("missing_info_level") is not None]
                    grammar_levels = [p.get("grammar_level") for p in content_paras if p.get("grammar_level") is not None]
                    
                    # Count grammar errors
                    total_grammar_errors = sum(len(p.get("grammar_errors", [])) for p in content_paras)
                    
                    # Count boolean flag successes
                    flag_counts = draft.get("flag_counts", {})
                    boolean_success_rate = 0
                    if flag_counts:
                        total_checks = len(GEMINI_FLAGS) * len(content_paras)
                        successful_checks = sum(flag_counts.get(flag, 0) for flag in GEMINI_FLAGS)
                        boolean_success_rate = (successful_checks / total_checks * 100) if total_checks > 0 else 0
                    
                    quality_data.append({
                        "Model": model,
                        "Mode": mode.title(),
                        "Iteration": iter_num,
                        "Content Paragraphs": len(content_paras),
                        "Avg Meaning Level": np.mean(meaning_levels) if meaning_levels else None,
                        "Avg Missing Info Level": np.mean(missing_levels) if missing_levels else None,
                        "Avg Grammar Level": np.mean(grammar_levels) if grammar_levels else None,
                        "Total Grammar Errors": total_grammar_errors,
                        "Boolean Success Rate": boolean_success_rate,
                        "Grammar Errors per Para": total_grammar_errors / len(content_paras) if content_paras else 0,
                        "AI Score After": draft.get("scores_after", {}).get("group_doc", {}).get("gptzero", 0),
                        "Zero Shot Success": "✅" if (
                            (draft.get("scores_after", {}).get("group_doc", {}).get("gptzero") is not None and 
                             draft.get("scores_after", {}).get("group_doc", {}).get("gptzero") <= ZERO_SHOT_THRESHOLD) or
                            (draft.get("scores_after", {}).get("group_doc", {}).get("sapling") is not None and 
                             draft.get("scores_after", {}).get("group_doc", {}).get("sapling") <= ZERO_SHOT_THRESHOLD)
                        ) else "❌"
                    })
    
    if not quality_data:
        st.warning("No quality data available for analysis")
        return
    
    quality_df = pd.DataFrame(quality_data)
    
    # Display comprehensive quality table
    st.markdown("##### 📋 Quality Metrics Summary")
    
    def style_quality_levels(val):
        if isinstance(val, (int, float)):
            if val >= 8:
                return 'color: green; font-weight: bold'
            elif val >= 6:
                return 'color: orange'
            elif val < 4:
                return 'color: red; font-weight: bold'
        return ''
    
    def style_missing_levels(val):
        if isinstance(val, (int, float)):
            if val <= 2:
                return 'color: green; font-weight: bold'
            elif val <= 4:
                return 'color: orange'
            elif val > 6:
                return 'color: red; font-weight: bold'
        return ''
    
    def style_zero_shot(val):
        if val == "✅":
            return 'color: green; font-weight: bold; font-size: 16px'
        elif val == "❌":
            return 'color: red; font-weight: bold; font-size: 16px'
        return ''
    
    styled_quality = (quality_df.style
                      .applymap(style_quality_levels, subset=['Avg Meaning Level', 'Avg Grammar Level'])
                      .applymap(style_missing_levels, subset=['Avg Missing Info Level'])
                      .applymap(style_zero_shot, subset=['Zero Shot Success'])
                      .format({
                          'Avg Meaning Level': lambda x: f"{x:.2f}" if x is not None else "—",
                          'Avg Missing Info Level': lambda x: f"{x:.2f}" if x is not None else "—", 
                          'Avg Grammar Level': lambda x: f"{x:.2f}" if x is not None else "—",
                          'Boolean Success Rate': '{:.1f}%',
                          'Grammar Errors per Para': '{:.1f}',
                          'AI Score After': '{:.3f}'
                      }))
    
    st.dataframe(styled_quality, use_container_width=True, hide_index=True)
    
    # Quality analysis charts
    st.markdown("##### 📊 Quality Trends Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Quality Levels by Model & Mode**")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Group data for plotting
        models = quality_df['Model'].unique()
        modes = quality_df['Mode'].unique()
        
        x = np.arange(len(models))
        width = 0.35
        
        # Meaning levels
        for i, mode in enumerate(modes):
            mode_data = quality_df[quality_df['Mode'] == mode]
            meaning_avg = [mode_data[mode_data['Model'] == model]['Avg Meaning Level'].mean() for model in models]
            ax1.bar(x + (i - 0.5) * width, meaning_avg, width, label=mode, alpha=0.8)
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Average Meaning Level')
        ax1.set_title('Meaning Preservation by Model & Mode')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 10)
        
        # Missing info levels (lower is better)
        for i, mode in enumerate(modes):
            mode_data = quality_df[quality_df['Mode'] == mode]
            missing_avg = [mode_data[mode_data['Model'] == model]['Avg Missing Info Level'].mean() for model in models]
            ax2.bar(x + (i - 0.5) * width, missing_avg, width, label=mode, alpha=0.8)
        
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Average Missing Info Level')
        ax2.set_title('Information Loss by Model & Mode (Lower is Better)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 10)
        
        # Grammar levels
        for i, mode in enumerate(modes):
            mode_data = quality_df[quality_df['Mode'] == mode]
            grammar_avg = [mode_data[mode_data['Model'] == model]['Avg Grammar Level'].mean() for model in models]
            ax3.bar(x + (i - 0.5) * width, grammar_avg, width, label=mode, alpha=0.8)
        
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Average Grammar Level')
        ax3.set_title('Grammar Quality by Model & Mode')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 10)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Quality vs AI Detection Correlation**")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Quality vs AI Score scatter
        for model in models:
            model_data = quality_df[quality_df['Model'] == model]
            ax1.scatter(model_data['AI Score After'], model_data['Boolean Success Rate'], 
                       label=model, alpha=0.7, s=60)
        
        ax1.set_xlabel('AI Detection Score (After)')
        ax1.set_ylabel('Boolean Quality Success Rate (%)')
        ax1.set_title('Quality vs AI Detection Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Zero-shot success by quality
        zero_shot_success = quality_df[quality_df['Zero Shot Success'] == '✅']['Boolean Success Rate']
        zero_shot_fail = quality_df[quality_df['Zero Shot Success'] == '❌']['Boolean Success Rate']
        
        ax2.hist([zero_shot_success, zero_shot_fail], bins=10, alpha=0.7, 
                label=['Zero-shot Success', 'Zero-shot Fail'], color=['green', 'red'])
        ax2.set_xlabel('Boolean Quality Success Rate (%)')
        ax2.set_ylabel('Number of Drafts')
        ax2.set_title('Quality Distribution: Zero-shot Success vs Failure')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Top performing drafts analysis
    st.markdown("##### 🏆 Best Performing Drafts")
    
    # Find top drafts by different criteria
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Best Quality (Boolean Success)**")
        top_quality = quality_df.nlargest(5, 'Boolean Success Rate')[['Model', 'Mode', 'Iteration', 'Boolean Success Rate', 'Zero Shot Success']]
        st.dataframe(top_quality, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Best Meaning Preservation**")
        meaning_available = quality_df.dropna(subset=['Avg Meaning Level'])
        if not meaning_available.empty:
            top_meaning = meaning_available.nlargest(5, 'Avg Meaning Level')[['Model', 'Mode', 'Iteration', 'Avg Meaning Level', 'Zero Shot Success']]
            st.dataframe(top_meaning, hide_index=True, use_container_width=True)
        else:
            st.info("No meaning level data available")
    
    with col3:
        st.markdown("**Best Grammar Quality**")
        grammar_available = quality_df.dropna(subset=['Avg Grammar Level'])
        if not grammar_available.empty:
            top_grammar = grammar_available.nlargest(5, 'Avg Grammar Level')[['Model', 'Mode', 'Iteration', 'Avg Grammar Level', 'Zero Shot Success']]
            st.dataframe(top_grammar, hide_index=True, use_container_width=True)
        else:
            st.info("No grammar level data available")


# ──────────────────────────── Standalone Page Setup ────────────────────
# When this file is executed directly by Streamlit's multi-page system,
# set up the page config and sidebar, then call the page function
# Check if we're being run as a standalone page (not imported)
if __name__ == "__main__":
    # Page config
    st.set_page_config(page_title="Benchmark Analysis - Humanizer Test-Bench", layout="wide", initial_sidebar_state="expanded")
    
    # Setup shared sidebar
    from src.pages._shared_layout import setup_sidebar
    setup_sidebar()
    
    # Call the page function
    page_runs()
