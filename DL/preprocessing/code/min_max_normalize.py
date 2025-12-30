from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

def minmax_norm_scalar(df: pd.DataFrame, cols: List[Any]) -> Dict[str, float]:
    params: Dict[str, float] = {}
    n = len(df)

    for col in cols:
        col_name = str(col)
        if isinstance(col, pd.Series):
            raw = col
            if raw.name is not None:
                col_name = str(raw.name)
        else:
            raw = df.get(col, None)

        if raw is None:
            s = pd.Series(np.zeros(n, dtype=np.float32))
        elif isinstance(raw, pd.Series):
            s = raw
        else:
            s = pd.Series([raw] * n)
        s = pd.to_numeric(s, errors="coerce").fillna(0.0).astype("float32")

        if s.size == 0:
            params[f"{col_name}_min"] = 0.0
            params[f"{col_name}_max"] = 0.0
        else:
            params[f"{col_name}_min"] = float(s.min())
            params[f"{col_name}_max"] = float(s.max())

    return params

def minmax_cal(value: float, value_min: float, value_max: float) -> float:
    if value is None or value_min is None or value_max is None:
        return -1.0
    if value_max == value_min:
        return 0.0
    return float((value - value_min) / (value_max - value_min))