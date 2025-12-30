#!/usr/bin/env python3
# -*- coding: utf-8 -*
from typing import Any

def _to_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except Exception:
        return None
    
def _to_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None

def _hex_to_int(hex_str: str) -> int:
    return int(hex_str, 16)

def _hex_to_float(hex_str: str) -> float:
    return float(int(hex_str, 16))