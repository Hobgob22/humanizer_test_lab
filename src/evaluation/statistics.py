"""
Statistics aggregation for benchmark runs.
"""

from typing import List, Dict, Any
from collections import defaultdict
import numpy as np
from src.evaluation.stats import describe_full


def aggregate_statistics_by_model(docs: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate statistics by model across all documents.

    Returns structure:
    {
        "models": {
            "model_name": {
                "mean_ai_score": float,
                "median_ai_score": float,
                "std_ai_score": float,
                "p25_ai_score": float,
                "p75_ai_score": float,
                "zeroshot_success": float,  # % with score <= 0.1
                "sample_count": int
            }
        }
    }
    """
    # Collect scores by model
    model_scores = defaultdict(list)

    for doc in docs:
        if not doc.get("models"):
            continue

        for model_name, model_data in doc["models"].items():
            if not model_data.get("iterations"):
                continue

            for iteration in model_data["iterations"]:
                # Try to get AI score from paragraph or document level
                score = iteration.get("para_ai_score") or iteration.get("doc_ai_score")
                if score is not None and not np.isnan(score):
                    model_scores[model_name].append(score)

    # Compute statistics for each model
    results = {"models": {}}

    for model_name, scores in model_scores.items():
        if not scores:
            continue

        scores_array = np.array(scores)

        # Compute descriptive statistics
        desc_stats = describe_full(scores)

        # Compute zero-shot success rate (score <= 0.1)
        zeroshot_count = np.sum(scores_array <= 0.1)
        zeroshot_rate = zeroshot_count / len(scores) if scores else 0.0

        results["models"][model_name] = {
            "mean_ai_score": desc_stats.get("mean"),
            "median_ai_score": desc_stats.get("median"),
            "std_ai_score": desc_stats.get("std"),
            "p25_ai_score": desc_stats.get("p25"),
            "p75_ai_score": desc_stats.get("p75"),
            "zeroshot_success": zeroshot_rate,
            "sample_count": len(scores)
        }

    return results


def merge_runs_data(run_names: List[str]) -> tuple[List[Dict], Dict[str, Any]]:
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

        # Add source run info to each document
        for doc in docs:
            doc["_source_run"] = run_name
            if doc.get("models"):
                metadata["model_set"].update(doc["models"].keys())

        all_docs.extend(docs)

    metadata["doc_count"] = len(all_docs)
    metadata["model_set"] = list(metadata["model_set"])

    return all_docs, metadata


def aggregate_statistics_by_folder_and_model(docs: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate statistics grouped by folder and model.

    Returns structure:
    {
        "folders": {
            "folder_name": {
                "models": {
                    "model_name": { stats... }
                }
            }
        }
    }
    """
    # Group docs by folder
    docs_by_folder = defaultdict(list)
    for doc in docs:
        folder = doc.get("folder", "unknown")
        docs_by_folder[folder].append(doc)

    # Compute statistics for each folder
    results = {"folders": {}}
    for folder, folder_docs in docs_by_folder.items():
        folder_stats = aggregate_statistics_by_model(folder_docs)
        results["folders"][folder] = folder_stats

    return results
