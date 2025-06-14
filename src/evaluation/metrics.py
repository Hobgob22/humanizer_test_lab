import numpy as np
import pandas as pd
from typing import List, Dict

def describe(series: List[float]) -> Dict[str, float]:
    arr = np.array(series)
    return {
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "mean": float(arr.mean()) if arr.size else None,
        "median": float(np.median(arr)) if arr.size else None,
        "p25": float(np.percentile(arr, 25)) if arr.size else None,
        "p75": float(np.percentile(arr, 75)) if arr.size else None,
    }

def distribution_dataframe(scores: List[float]):
    return pd.DataFrame({"score": scores})
