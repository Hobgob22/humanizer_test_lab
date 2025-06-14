"""
Descriptive-stats helper for detector scores & word-counts.

Returned dict keys
------------------
mean   – arithmetic mean
std    – sample standard deviation (ddof=1)
p25    – 25th percentile
median – 50th percentile
p75    – 75th percentile
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np


def describe_full(series: List[float]) -> Dict[str, float]:
    arr = np.array(series, dtype=float)
    if arr.size == 0:
        return {k: None for k in ("mean", "std", "p25", "median", "p75")}
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std(ddof=1)),
        "p25":    float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75":    float(np.percentile(arr, 75)),
    }
