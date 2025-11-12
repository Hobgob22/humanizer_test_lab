"""
Statistics aggregation for benchmark runs - compatible with Streamlit data format.
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

# Zero-shot threshold for AI detection
ZERO_SHOT_THRESHOLD = 0.1

# Expected quality flags
_EXPECTED_FLAGS = [
    "length_ok",
    "same_meaning",
    "same_lang",
    "no_missing_info",
    "citation_preserved",
    "citation_content_ok",
]


def _iter_drafts(docs: List[Dict]) -> Tuple[Dict, Dict]:
    """Yield (doc, draft) pairs for every draft in docs."""
    for doc in docs:
        if not doc.get("runs"):
            continue
        for draft in doc["runs"]:
            yield doc, draft


def merge_runs_data(run_names: List[str]) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Merge multiple runs into a single dataset.

    Args:
        run_names: List of run names to merge

    Returns:
        Tuple of (merged_docs, metadata)
    """
    from src.results_db import load_run

    all_docs = []
    metadata = {
        "run_names": run_names,
        "total_runs": len(run_names),
        "doc_count": 0,
        "model_set": set()
    }

    for run_name in run_names:
        run_data = load_run(run_name)
        if not run_data:
            continue

        docs = run_data.get("docs", [])

        # Add source run info to each document and draft
        for doc in docs:
            for draft in doc.get("runs", []):
                draft["_source_run"] = run_name
                metadata["model_set"].add(draft.get("model", "unknown"))

        all_docs.extend(docs)

    metadata["doc_count"] = len(all_docs)
    metadata["model_set"] = list(metadata["model_set"])

    return all_docs, metadata


def aggregate_statistics_by_model(docs: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate statistics by folder → model → mode.

    Returns structure compatible with Streamlit frontend:
    {
        "folder_name": {
            "model_name": {
                "mode_name": {
                    "baseline": {...},
                    "after": {"gptzero": ..., "sapling": ...},
                    "deltas": {...},
                    "quality": {...},
                    "zero_shot_success": {"gptzero": ..., "sapling": ...},
                    "grammar_score": ...,
                    "same_meaning_level_avg": ...,
                    "missing_info_level_avg": ...,
                    ...
                }
            }
        }
    }
    """
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # Track document-level paragraph counts for each model/mode combination
    doc_paragraphs_by_model_mode = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    # Collect baselines by folder
    folder_baselines = defaultdict(list)
    for doc in docs:
        if not doc.get("runs"):
            continue
        first = doc["runs"][0]
        if "scores_before" in first and "group_doc" in first["scores_before"]:
            gz = first["scores_before"]["group_doc"].get("gptzero")
            sp = first["scores_before"]["group_doc"].get("sapling")
            if gz is not None:
                folder_baselines[doc.get("folder", "unknown")].append({
                    "gptzero": gz,
                    "sapling": sp,
                    "wordcount": first.get("wordcount_before", 0),
                })

    # Calculate folder average baselines
    _valid = lambda v: v is not None and not (isinstance(v, float) and np.isnan(v))

    folder_avg_baselines = {}
    for f, bl in folder_baselines.items():
        gz_vals = [b["gptzero"] for b in bl if _valid(b.get("gptzero"))]
        sp_vals = [b["sapling"] for b in bl if _valid(b.get("sapling"))]
        wc_vals = [b["wordcount"] for b in bl if _valid(b.get("wordcount"))]

        folder_avg_baselines[f] = {
            "gptzero": np.nanmean(gz_vals) if gz_vals else 0.5,
            "sapling": np.nanmean(sp_vals) if sp_vals else 0.5,
            "wordcount": np.nanmean(wc_vals) if wc_vals else 0,
        }

    # Track documents for paragraph count calculation
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

            doc_paragraphs_by_model_mode[folder][model][mode].add((doc_name, doc_para_count))

    # Collect statistics per draft
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
                "grammar_scores": [],
                "draft_count": 0,
                "doc_mismatch_count": 0,
                "doc_total_drafts": 0,
                "para_level_mismatch_count": 0,
                "para_level_mismatched_paragraphs": 0,
                "total_content_paragraphs": 0,
                "zs_hits": {"gptzero": 0, "sapling": 0},
                "draft_length_deviations": [],
                "para_length_deviations": [],
                "same_meaning_levels": [],
                "missing_info_levels": [],
                "citation_preservation_rates": [],
                "citation_exact_match_rates": [],
                "content_paragraph_counts": [],
                "total_paragraphs": [],
                "series": defaultdict(list),
                "source_runs": set(),
            },
        )

        # Track source run if present
        if "_source_run" in dr:
            bucket["source_runs"].add(dr["_source_run"])

        # Detector scores
        gz = dr["scores_after"]["group_doc"].get("gptzero")
        sp = dr["scores_after"]["group_doc"].get("sapling")
        bucket["after_scores"].append({"gptzero": gz, "sapling": sp})

        # Zero-shot hits
        if gz is not None and gz <= ZERO_SHOT_THRESHOLD:
            bucket["zs_hits"]["gptzero"] += 1
        if sp is not None and sp <= ZERO_SHOT_THRESHOLD:
            bucket["zs_hits"]["sapling"] += 1

        # Word count delta
        delta_wc = dr.get("wordcount_after", 0) - dr.get("wordcount_before", 0)
        bucket["wc_deltas"].append(delta_wc)

        # Draft-level length deviation
        draft_length_deviation = dr.get("draft_length_deviation", 0)
        bucket["draft_length_deviations"].append(draft_length_deviation)

        # Store raw series
        bucket["series"]["after_gz"].append(gz if gz is not None else np.nan)
        bucket["series"]["after_sp"].append(sp if sp is not None else np.nan)
        bucket["series"]["wc"].append(delta_wc)
        bucket["series"]["draft_length_dev"].append(draft_length_deviation)

        # Quality metrics (skip drafts with paragraph mismatch)
        if not dr.get("para_mismatch", False):
            flag_counts = dr.get("flag_counts", {})
            total_segments = flag_counts.get("total_segments", dr.get("para_count_before", 1))
            content_paragraphs = flag_counts.get("content_paragraph_count", total_segments)

            # Track paragraph counts
            bucket["content_paragraph_counts"].append(content_paragraphs)
            bucket["total_paragraphs"].append(total_segments)

            # Boolean quality flags
            for flag in _EXPECTED_FLAGS:
                cnt = flag_counts.get(flag, 0)
                pct = (cnt / content_paragraphs) * 100 if content_paragraphs else 0
                bucket["quality_flags"][flag].append(pct)

            # Numeric quality levels (0-10 scale)
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

            # Length deviation metrics
            para_length_deviation = flag_counts.get("para_length_deviation_avg", 0)
            bucket["para_length_deviations"].append(para_length_deviation)
            bucket["series"]["para_length_dev"].append(para_length_deviation)

            # Citation preservation metrics
            citation_preservation_rate = flag_counts.get("paragraph_citation_preservation_rate", 100)
            citation_exact_match_rate = flag_counts.get("citation_exact_match_rate", 100)
            bucket["citation_preservation_rates"].append(citation_preservation_rate)
            bucket["citation_exact_match_rates"].append(citation_exact_match_rate)
            bucket["series"]["citation_preservation"].append(citation_preservation_rate)
            bucket["series"]["citation_exact_match"].append(citation_exact_match_rate)

            # Overall quality percentage
            qual_pct = (
                sum(flag_counts.get(f, 0) for f in _EXPECTED_FLAGS)
                / (content_paragraphs * len(_EXPECTED_FLAGS))
                * 100
                if content_paragraphs
                else 0
            )
            bucket["series"]["quality"].append(qual_pct)

        bucket["draft_count"] += 1

        # Document-structure mismatch
        if dr.get("para_mismatch", False):
            bucket["doc_mismatch_count"] += 1
        bucket["doc_total_drafts"] += 1

        # Paragraph-level mismatches in para mode
        if dr.get("mode") == "para":
            if dr.get("has_para_level_mismatches", False):
                bucket["para_level_mismatch_count"] += 1

            para_level_mismatches = dr.get("para_level_mismatches", 0)
            bucket["para_level_mismatched_paragraphs"] += para_level_mismatches

    # Aggregate bucket data
    result = {}
    for folder, models in stats.items():
        baseline = folder_avg_baselines.get(folder, {"gptzero": 0.5, "sapling": 0.5})
        result[folder] = {}

        for model, modes in models.items():
            result[folder][model] = {}

            for mode, data in modes.items():
                if not data["draft_count"]:
                    continue

                # Calculate averages
                after_scores = data["after_scores"]
                gz_scores = [s["gptzero"] for s in after_scores if _valid(s.get("gptzero"))]
                sp_scores = [s["sapling"] for s in after_scores if _valid(s.get("sapling"))]

                after_gz = np.nanmean(gz_scores) if gz_scores else np.nan
                after_sp = np.nanmean(sp_scores) if sp_scores else np.nan

                zs_gz_pct = (data["zs_hits"]["gptzero"] / data["draft_count"]) * 100
                zs_sp_pct = (data["zs_hits"]["sapling"] / data["draft_count"]) * 100

                # Word count diff metrics
                deltas = np.array(data["wc_deltas"])
                within10 = (np.abs(deltas) <= 10).mean() * 100 if deltas.size else 0
                within20 = (np.abs(deltas) <= 20).mean() * 100 if deltas.size else 0
                pct_longer = (deltas > 0).mean() * 100 if deltas.size else 0
                pct_shorter = (deltas < 0).mean() * 100 if deltas.size else 0

                # Average quality metrics
                avg_grammar = np.mean(data["grammar_scores"]) if data["grammar_scores"] else None
                avg_same_meaning_level = np.mean(data["same_meaning_levels"]) if data["same_meaning_levels"] else None
                avg_missing_info_level = np.mean(data["missing_info_levels"]) if data["missing_info_levels"] else None

                # Length deviation metrics
                draft_length_devs = np.array(data["draft_length_deviations"])
                para_length_devs = np.array(data["para_length_deviations"])

                len_within_10_pct = (np.abs(draft_length_devs) <= 10).mean() * 100 if draft_length_devs.size else 0
                len_within_15_pct = (np.abs(draft_length_devs) <= 15).mean() * 100 if draft_length_devs.size else 0
                len_within_20_pct = (np.abs(draft_length_devs) <= 20).mean() * 100 if draft_length_devs.size else 0

                # Citation preservation metrics
                avg_citation_preservation = np.mean(data["citation_preservation_rates"]) if data["citation_preservation_rates"] else 100
                avg_citation_exact_match = np.mean(data["citation_exact_match_rates"]) if data["citation_exact_match_rates"] else 100

                # Paragraph counts
                avg_content_paras = np.mean(data["content_paragraph_counts"]) if data["content_paragraph_counts"] else 0
                doc_level_total_paras = sum(para_count for _, para_count in doc_paragraphs_by_model_mode[folder][model][mode])

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
                    "grammar_score": avg_grammar,
                    "same_meaning_level_avg": avg_same_meaning_level,
                    "missing_info_level_avg": avg_missing_info_level,
                    "draft_length_deviation_avg": draft_length_devs.mean() if draft_length_devs.size else 0,
                    "para_length_deviation_avg": para_length_devs.mean() if para_length_devs.size else 0,
                    "length_within_10_pct": len_within_10_pct,
                    "length_within_15_pct": len_within_15_pct,
                    "length_within_20_pct": len_within_20_pct,
                    "citation_preservation_rate_avg": avg_citation_preservation,
                    "citation_exact_match_rate_avg": avg_citation_exact_match,
                    "avg_content_paragraphs": avg_content_paras,
                    "total_content_paragraphs": doc_level_total_paras,
                    "draft_count": data["draft_count"],
                    "mismatch_rate": 0,  # Legacy
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
                    "wc_deltas": data["wc_deltas"],
                    "series": {k: list(v) for k, v in data["series"].items()},
                    "source_runs": list(data["source_runs"]),
                }

    return result
