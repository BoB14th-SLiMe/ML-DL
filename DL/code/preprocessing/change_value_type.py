#!/usr/bin/env python3
# -*- coding: utf-8 -*
from typing import Any
import math

def _to_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except Exception as e:
        print(f"{e} (change_vlaue_type.py)")
        return None
    
def _to_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception as e:
        print(f"{e} (change_vlaue_type.py)")
        return None

def _hex_to_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return None

        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return int(value)

        if isinstance(value, int):
            return value

        string = str(value).strip().lower()
        if not string or string == "nan":
            return None

        if string.startswith("0x"):
            return int(string, 16)

        value_float = float(string)
        if not math.isfinite(value_float):
            return None
        return int(value_float)
    except Exception as e:
        print(f"{e} (change_vlaue_type.py)")
        return None

def _hex_to_float(value: Any) -> float:
    try:
        value_int = _hex_to_int(value)
        return float(value_int) if value_int is not None else None
    except Exception as e:
        print(f"{e} (change_vlaue_type.py)")
        return None