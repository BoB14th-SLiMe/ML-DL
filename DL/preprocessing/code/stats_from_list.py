
from typing import Any, Dict, List, Optional
import math

from change_value_type import _to_float

def stats_count_min_max_range(values: List[Any]) -> Dict[str, Optional[float]]:
    float_values: List[float] = []

    for value in values:
        float_value = _to_float(value)
        if float_value is None:
            continue
        if not math.isfinite(float_value):
            continue
        float_values.append(float_value)

    if not float_values:
        return {"count": 0, "min": None, "max": None, "range": None}

    count = len(float_values)
    min_value = min(float_values)
    max_value = max(float_values)
    value_range = max_value - min_value

    return {
        "count": count,
        "min": min_value,
        "max": max_value,
        "range": value_range,
    }


def stats_min_max_mean_std(values: List[Any], *, ddof: int = 0) -> Dict[str, Optional[float]]:
    float_values: List[float] = []

    for value in values:
        float_value = _to_float(value)
        if float_value is None:
            continue
        if not math.isfinite(float_value):
            continue
        float_values.append(float_value)

    if not float_values:
        return {"min": None, "max": None, "mean": None, "std": None}

    min_value = min(float_values)
    max_value = max(float_values)

    count = len(float_values)
    mean_value = sum(float_values) / count

    denominator = count - ddof
    if denominator <= 0:
        std_value = None
    else:
        variance = sum((x - mean_value) ** 2 for x in float_values) / denominator
        std_value = math.sqrt(variance)

    return {
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "std": std_value,
    }