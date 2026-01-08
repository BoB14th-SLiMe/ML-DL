#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import logging
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import orjson  # type: ignore
except Exception:
    orjson = None

JsonDict = Dict[str, Any]

_ORJSON_OPT = 0
if orjson is not None:
    _ORJSON_OPT = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY


def json_loads_fast(s: str) -> Any:
    if orjson is not None:
        try:
            return orjson.loads(s)
        except Exception:
            pass
    return json.loads(s)


def json_dumps_bytes(obj: Any) -> bytes:
    if orjson is not None:
        try:
            return orjson.dumps(obj, option=_ORJSON_OPT, default=str)
        except Exception:
            pass
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def _is_nan_inf(x: float) -> bool:
    try:
        return math.isnan(x) or math.isinf(x)
    except Exception:
        return False


def sanitize_and_drop_none(obj: Any) -> Any:
    t = type(obj)

    if t is dict:
        out = {}
        for k, v in obj.items():
            cleaned = sanitize_and_drop_none(v)
            if cleaned is not None:
                out[str(k)] = cleaned
        return out

    if t is list:
        return [x for x in (sanitize_and_drop_none(i) for i in obj) if x is not None]

    if t in (str, int, bool):
        return obj

    if obj is None:
        return None

    if t is float:
        return None if _is_nan_inf(obj) else obj

    if t in (tuple, set):
        return [x for x in (sanitize_and_drop_none(i) for i in obj) if x is not None]

    try:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if _is_nan_inf(v) else v
        if isinstance(obj, np.ndarray):
            return [x for x in (sanitize_and_drop_none(i) for i in obj.tolist()) if x is not None]
    except Exception:
        pass

    return str(obj)


def flush_logger(logger: logging.Logger, *, fsync: bool = False) -> None:
    for h in list(logger.handlers):
        try:
            h.flush()
            if fsync:
                stream = getattr(h, "stream", None)
                if stream is not None and hasattr(stream, "fileno"):
                    try:
                        os.fsync(stream.fileno())
                    except Exception:
                        pass
        except Exception:
            pass


def setup_data_logger(log_path: Path, *, mode: str = "w") -> logging.Logger:
    logger = logging.getLogger("data_flow")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_path), mode=mode, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)

    logger.info("=== data_flow logger initialized ===")
    return logger


def kv_line(tag: str, data_flow_max_json_chars: int, **kv: Any) -> str:
    parts = [f"[{tag}]"]
    for k, v in kv.items():
        if isinstance(v, (dict, list)):
            s = json.dumps(v, ensure_ascii=False, separators=(",", ":"), default=str)
            if data_flow_max_json_chars and len(s) > data_flow_max_json_chars:
                s = s[:data_flow_max_json_chars] + "...(truncated)"
            parts.append(f"{k}={s}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def log_event(logger: logging.Logger, tag: str, *, data_flow_max_json_chars: int, **kv: Any) -> None:
    try:
        logger.info(kv_line(tag, data_flow_max_json_chars, **kv))
    except Exception:
        pass


@contextmanager
def timed(stats: Dict[str, List[float]], key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        stats.setdefault(key, []).append(time.perf_counter() - t0)


def stable_hash(s: str, mod: int = 1000) -> int:
    import hashlib
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return int(h, 16) % int(mod)


def to_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            fv = float(v)
            return int(fv) if fv.is_integer() else None
        s = str(v).strip()
        if not s:
            return None
        if s.lower().startswith("0x"):
            return int(s, 16)
        f = float(s)
        return int(f) if f.is_integer() else None
    except Exception:
        return None
