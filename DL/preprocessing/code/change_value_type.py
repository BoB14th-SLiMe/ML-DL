#!/usr/bin/env python3
# -*- coding: utf-8 -*

from typing import Any

# 형변환
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
