#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_translated_addr_slot.py

translated_addr용 전처리 (AUTO protocol)

현재 적용 프로토콜:
  - modbus
      - 주소: modbus.translated_addr
      - 값  : modbus.regs.val
  - xgt_fen
      - 주소: xgt_fen.translated_addr
      - 값  : xgt_fen.word_value

출력(out_dir 기준, 프로토콜별로 각각 생성):
  modbus:
    - modbus_addr_slot_vocab.json
    - modbus_addr_slot_norm_params.json
    - modbus_addr_slot.npy
  xgt_fen:
    - xgt_addr_slot_vocab.json
    - xgt_addr_slot_norm_params.json
    - xgt_addr_slot.npy
"""

import json, sys
import argparse
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from min_max_normalize import minmax_cal
from change_value_type import _to_float

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.file_load import file_load


NAN_STRINGS = {"", "nan", "none", "null"}


def _is_nan_like(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float):
        return math.isnan(x)
    if isinstance(x, (np.floating,)):
        return bool(np.isnan(x))
    if isinstance(x, str):
        return x.strip().lower() in NAN_STRINGS
    return False


def _flatten_str_like(x: Any) -> List[str]:
    out: List[str] = []
    if x is None or _is_nan_like(x):
        return out

    if isinstance(x, (list, tuple)):
        for y in x:
            out.extend(_flatten_str_like(y))
        return out

    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() in NAN_STRINGS:
            return out

        if s.startswith("[") and s.endswith("]"):
            try:
                loaded = json.loads(s)
                out.extend(_flatten_str_like(loaded))
                return out
            except Exception:
                pass

        if "," in s:
            for p in (pp.strip() for pp in s.split(",")):
                if p and p.lower() not in NAN_STRINGS:
                    out.append(p)
            return out

        out.append(s)
        return out

    s = str(x).strip()
    if not s or s.lower() in NAN_STRINGS:
        return out
    out.append(s)
    return out


def _flatten_float_like(x: Any) -> List[float]:
    out: List[float] = []
    if x is None or _is_nan_like(x):
        return out

    if isinstance(x, (list, tuple)):
        for y in x:
            out.extend(_flatten_float_like(y))
        return out

    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() in NAN_STRINGS:
            return out

        if s.startswith("[") and s.endswith("]"):
            try:
                loaded = json.loads(s)
                out.extend(_flatten_float_like(loaded))
                return out
            except Exception:
                pass

        if "," in s:
            for p in (pp.strip() for pp in s.split(",")):
                try:
                    v = float(p)
                    if not math.isnan(v):
                        out.append(v)
                except Exception:
                    continue
            return out

        try:
            v = float(s)
            if not math.isnan(v):
                out.append(v)
        except Exception:
            pass
        return out

    try:
        v = float(x)
        if not math.isnan(v):
            out.append(v)
    except Exception:
        pass
    return out


def _sort_numeric_key(addr: str) -> int:
    try:
        return int(addr)
    except Exception:
        return 10**18


def _safe_addr_str_from_float(v: float) -> Optional[str]:
    try:
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        return str(int(v))
    except Exception:
        return None


def _safe_minmax(v: Any, vmin: Any, vmax: Any) -> float:
    vv = _to_float(v)
    vmin_f = _to_float(vmin)
    vmax_f = _to_float(vmax)
    if vv is None or vmin_f is None or vmax_f is None:
        return -1.0
    try:
        out = float(minmax_cal(float(vv), float(vmin_f), float(vmax_f)))
        if math.isnan(out):
            return -1.0
        return out
    except Exception:
        return -1.0


def _get_cfg(proto: str) -> Dict[str, Any]:
    proto = (proto or "").strip().lower()
    if proto == "modbus":
        return {
            "protocol": "modbus",
            "addr_field": "modbus.translated_addr",
            "val_field": "modbus.regs.val",
            "vocab_file": "modbus_addr_slot_vocab.json",
            "norm_file": "modbus_addr_slot_norm_params.json",
            "npy_file": "modbus_addr_slot.npy",
            "feature_prefix": "modbus_addr_",
            "sort_numeric": True,
        }
    if proto == "xgt_fen":
        return {
            "protocol": "xgt_fen",
            "addr_field": "xgt_fen.translated_addr",
            "val_field": "xgt_fen.word_value",
            "vocab_file": "xgt_addr_slot_vocab.json",
            "norm_file": "xgt_addr_slot_norm_params.json",
            "npy_file": "xgt_addr_slot.npy",
            "feature_prefix": "xgt_addr_",
            "sort_numeric": False,
        }
    return {}


def _extract_pairs(records: Dict[str, Any], cfg: Dict[str, Any]) -> List[Tuple[str, float]]:
    if not cfg or records.get("protocol") != cfg["protocol"]:
        return []

    raw_addr = records.get(cfg["addr_field"])
    raw_val = records.get(cfg["val_field"])

    if raw_addr is None or _is_nan_like(raw_addr):
        return []
    if raw_val is None or _is_nan_like(raw_val):
        return []

    if cfg["protocol"] == "modbus":
        addr_nums = _flatten_float_like(raw_addr)
        addr_list: List[str] = []
        for a in addr_nums:
            s = _safe_addr_str_from_float(a)
            if s is not None and s.lower() not in NAN_STRINGS:
                addr_list.append(s)
    else:
        addr_list = [a for a in _flatten_str_like(raw_addr) if a.strip().lower() not in NAN_STRINGS]

    val_list = _flatten_float_like(raw_val)

    if not addr_list or not val_list:
        return []

    n = min(len(addr_list), len(val_list))
    pairs: List[Tuple[str, float]] = []
    for i in range(n):
        a = addr_list[i]
        v = val_list[i]
        if _is_nan_like(a) or _is_nan_like(v):
            continue
        try:
            fv = float(v)
            if math.isnan(fv):
                continue
            pairs.append((a, fv))
        except Exception:
            continue

    return pairs


# fit
def fit_preprocess_translated_addr_slot(input_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(input_path, lines=True, encoding="utf-8-sig")

    protos = ["modbus", "xgt_fen"]

    if df.empty:
        for proto in protos:
            cfg = _get_cfg(proto)
            (out_dir / cfg["vocab_file"]).write_text("{}", encoding="utf-8-sig")
            (out_dir / cfg["norm_file"]).write_text("{}", encoding="utf-8-sig")
            data = np.zeros(0, dtype=np.dtype([]))
            np.save(out_dir / cfg["npy_file"], data)
        return

    records = df.to_dict(orient="records")

    for proto in protos:
        cfg = _get_cfg(proto)

        # ---------- 1) hex 데이터 float 변환 ----------
        packet_pairs: List[List[Tuple[str, float]]] = []
        addr_stats: Dict[str, Dict[str, float]] = {}

        for rec in records:
            if not isinstance(rec, dict):
                continue
            pairs = _extract_pairs(rec, cfg)
            if not pairs:
                continue

            packet_pairs.append(pairs)
            for a, v in pairs:
                if _is_nan_like(a) or _is_nan_like(v):
                    continue
                if a not in addr_stats:
                    addr_stats[a] = {"min": v, "max": v}
                else:
                    if v < addr_stats[a]["min"]:
                        addr_stats[a]["min"] = v
                    if v > addr_stats[a]["max"]:
                        addr_stats[a]["max"] = v

        # ---------- 2) min-max 파라미터 산출 ----------
        addrs = [a for a in addr_stats.keys() if not _is_nan_like(a) and str(a).strip().lower() not in NAN_STRINGS]
        if cfg["sort_numeric"]:
            addr_list_sorted = sorted(addrs, key=_sort_numeric_key)
        else:
            addr_list_sorted = sorted(addrs)

        vocab: Dict[str, int] = {addr: idx for idx, addr in enumerate(addr_list_sorted)}
        norm_params: Dict[str, Dict[str, float]] = {
            addr: {"min": float(addr_stats[addr]["min"]), "max": float(addr_stats[addr]["max"])}
            for addr in addr_list_sorted
            if addr in addr_stats
        }

        # ---------- 3) min-max 정규화 적용 ----------
        if not vocab:
            (out_dir / cfg["vocab_file"]).write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8-sig")
            (out_dir / cfg["norm_file"]).write_text(json.dumps(norm_params, indent=2, ensure_ascii=False), encoding="utf-8-sig")
            data = np.zeros(0, dtype=np.dtype([]))
            np.save(out_dir / cfg["npy_file"], data)
            continue

        addr_list_sorted = [addr for addr, _ in sorted(vocab.items(), key=lambda x: x[1])]
        dtype = np.dtype([(cfg["feature_prefix"] + addr, "f4") for addr in addr_list_sorted])
        data = np.zeros(len(packet_pairs), dtype=dtype)
        for name in data.dtype.names:
            data[name].fill(np.float32(-1.0))

        for i, pairs in enumerate(packet_pairs):
            for a, v in pairs:
                if a not in vocab:
                    continue
                p = norm_params.get(a) or {}
                data[cfg["feature_prefix"] + a][i] = np.float32(_safe_minmax(v, p.get("min"), p.get("max")))

        # ---------- 4) vocab + norm_params 저장 ----------
        (out_dir / cfg["vocab_file"]).write_text(
            json.dumps(vocab, indent=2, ensure_ascii=False),
            encoding="utf-8-sig",
        )
        (out_dir / cfg["norm_file"]).write_text(
            json.dumps(norm_params, indent=2, ensure_ascii=False),
            encoding="utf-8-sig",
        )

        # ---------- 5) translated_addr_slot.npy 저장 ----------
        np.save(out_dir / cfg["npy_file"], data)

        print(f"\n===== {proto} 앞 5개 전처리 샘플 =====")
        for i in range(min(5, len(data))):
            row = {}
            for name in data.dtype.names[: min(10, len(data.dtype.names))]:
                row[name] = float(data[name][i])
            print(row)


def preprocess_translated_addr_slot(records: Dict[str, Any], assets: Dict[str, Any]) -> Dict[str, Any]:
    proto = (records.get("protocol") or "").strip().lower()
    cfg = assets.get(proto, {}).get("cfg")
    vocab = assets.get(proto, {}).get("vocab") or {}
    norm_params = assets.get(proto, {}).get("norm") or {}

    if not cfg or not vocab:
        return {}

    addr_list_sorted = [addr for addr, _ in sorted(vocab.items(), key=lambda x: x[1])]
    feat: Dict[str, Any] = {cfg["feature_prefix"] + addr: -1.0 for addr in addr_list_sorted}

    pairs = _extract_pairs(records, cfg)
    if not pairs:
        return feat

    for a, v in pairs:
        if a not in vocab:
            continue
        p = norm_params.get(a) or {}
        feat[cfg["feature_prefix"] + a] = float(_safe_minmax(v, p.get("min"), p.get("max")))

    return feat


def transform_preprocess_translated_addr_slot(packet: Dict[str, Any], param_dir: Path) -> Dict[str, Any]:
    assets: Dict[str, Any] = {}

    for proto in ["modbus", "xgt_fen"]:
        cfg = _get_cfg(proto)
        vocab_path = param_dir / cfg["vocab_file"]
        norm_path = param_dir / cfg["norm_file"]

        vocab = file_load("json", str(vocab_path)) or {}
        norm = file_load("json", str(norm_path)) or {}

        assets[proto] = {"cfg": cfg, "vocab": vocab, "norm": norm}

    return preprocess_translated_addr_slot(packet, assets)


# ---------------------------------------------
# MAIN
# ---------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--transform", action="store_true")

    args = parser.parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output)

    if args.fit:
        fit_preprocess_translated_addr_slot(input_path, out_dir)
    elif args.transform:
        packets = file_load("jsonl", str(input_path)) or []
        for pkt in packets:
            if not isinstance(pkt, dict):
                continue
            feat = transform_preprocess_translated_addr_slot(pkt, out_dir)
            print(feat)
    else:
        raise ValueError("❌ 반드시 --fit 또는 --transform 중 하나를 선택하세요.")
