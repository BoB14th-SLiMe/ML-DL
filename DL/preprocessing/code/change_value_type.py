#!/usr/bin/env python3
# -*- coding: utf-8 -*
from typing import Any

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

def _hex_to_int(hex_str: str) -> int:
    try:
        if hex_str in (None, ""):
            return None
        return int(hex_str, 16)
    except Exception as e:
        print(f"{e} (change_vlaue_type.py)")
        return None

def _hex_to_float(hex_str: str) -> float:
    try:
        if hex_str in (None, ""):
            return None
        return float(int(hex_str, 16))
    except Exception as e:
        print(f"{e} (change_vlaue_type.py)")
        return None