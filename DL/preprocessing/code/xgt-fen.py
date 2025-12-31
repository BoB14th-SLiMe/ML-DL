#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xgt.py

두 가지 모드 제공:
  --fit       : var_vocab + norm_params 생성 후 xgt.npy 저장
  --transform : 기존 var_vocab + norm_params 사용

입력 JSONL에서 사용하는 필드:
  - xgt_fen.source            : xgt 소스 정보
  - xgt_fen.fenetpos          : xgt FEnet 포지션 
  - xgt_fen.cmd               : xgt 명령어 코드 
  - xgt_fen.dtype             : xgt 데이터 타입
  - xgt_fen.blkcnt            : xgt 블록 개수
  - xgt_fen.errstat           : xgt 에러 상태 
  - xgt_fen.datasize          : xgt 데이터 크기
  - xgt_fen.vars              : xgt 변수 이름
  - xgt_fen.word_value        : xgt 레지스터 값 
#   - xgt_fen.translated_addr : xgt 레지스터 이름

출력 feature (xgt.npy, structured numpy):
  - xgt_var_id            : vars[0] → vocab ID (int32, 정규화 X)
  - xgt_var_cnt_norm      : vars 개수 min-max 정규화
  - xgt_source_norm       : source min-max 정규화
  - xgt_len_norm          : len min-max 정규화
  - xgt_fenet_base_norm   : fenet_base min-max 정규화
  - xgt_fenet_slot_norm   : fenet_slot min-max 정규화
  - xgt_cmd_norm          : cmd min-max 정규화
  - xgt_dtype_norm        : dtype min-max 정규화
  - xgt_blkcnt_norm       : blkcnt min-max 정규화
  - xgt_err_flag          : err_code != 0 이면 1.0 else 0.0 (정규화 X)
  - xgt_err_code_norm     : err_code min-max 정규화
  - xgt_datasize_norm     : datasize min-max 정규화
  - xgt_addr_count        : translated_addr 개수
  - xgt_addr_min          : translated_addr 최소 (없으면 -1.0)
  - xgt_addr_max          : translated_addr 최대 (없으면 -1.0)
  - xgt_addr_range        : translated_addr 범위 (없으면 -1.0)
  - xgt_word_min          : word_value 최소 (없으면 -1.0)
  - xgt_word_max          : word_value 최대 (없으면 -1.0)
  - xgt_word_mean         : word_value 평균 (없으면 -1.0)
  - xgt_word_std          : word_value 표준편차 (없으면 -1.0)
"""

import json, sys
import argparse
import math
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

from min_max_normalize import minmax_cal, minmax_norm_scalar
from change_value_type import _to_float, _hex_to_float, _hex_to_int
from stats_from_list import stats_count_min_max_range, stats_min_max_mean_std

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.file_load import file_load


def _clean_var_name(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if s.lower() in ("nan", "none", "null"):
        return ""
    return s


def _as_list(value: Any) -> List[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            loaded = json.loads(s)
            if isinstance(loaded, list):
                return loaded
        except Exception:
            pass
    return [value]


def _parse_first_var_and_cnt(vars_value: Any) -> (str, float):
    if vars_value in (None, ""):
        return ("", float("nan"))
    if isinstance(vars_value, float) and not math.isfinite(vars_value):
        return ("", float("nan"))

    names: List[str] = []

    if isinstance(vars_value, (list, tuple)):
        for x in vars_value:
            s = _clean_var_name(x)
            if s:
                names.append(s)
    else:
        s = _clean_var_name(vars_value)
        if not s:
            return ("", float("nan"))
        for part in s.split(","):
            p = _clean_var_name(part)
            if p:
                names.append(p)

    if not names:
        return ("", float("nan"))

    return (names[0], float(len(names)))


def _get_var_id(var_map: Dict[str, int], var_name: str) -> int:
    name = _clean_var_name(var_name)
    if not name:
        return 0

    if name in var_map:
        return int(var_map[name])

    next_id = max(var_map.values()) + 1 if var_map else 1
    var_map[name] = next_id
    return int(next_id)


def _parse_translated_addr(values: Any) -> List[float]:
    out: List[float] = []
    for x in _as_list(values):
        s = _clean_var_name(x)
        if not s:
            continue
        m = re.search(r"(\d+)", s)
        if not m:
            continue
        v = _to_float(m.group(1))
        if v is None:
            continue
        out.append(float(v))
    return out


def _parse_word_value(values: Any) -> List[float]:
    out: List[float] = []
    for x in _as_list(values):
        v = _hex_to_float(x)
        if v is None:
            continue
        out.append(float(v))
    return out


def _fenet_base(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, float):
        if not math.isfinite(value):
            return float("nan")
        value = int(value)
    elif not isinstance(value, int):
        value = _hex_to_int(value)
        if value is None:
            return float("nan")
    return float((int(value) >> 4) & 0x0F)


def _fenet_slot(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, float):
        if not math.isfinite(value):
            return float("nan")
        value = int(value)
    elif not isinstance(value, int):
        value = _hex_to_int(value)
        if value is None:
            return float("nan")
    return float(int(value) & 0x0F)


def _df_get_first(df: pd.DataFrame, keys: List[str], n: int) -> pd.Series:
    for k in keys:
        if k in df.columns:
            return df[k]
    return pd.Series([None] * n, index=df.index)


# fit
def fit_preprocess_xgt(input_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(input_path, lines=True, encoding="utf-8-sig")

    norm_cols = [
        "xgt.var_cnt",
        "xgt.source",
        "xgt.len",
        "xgt.fenet_base",
        "xgt.fenet_slot",
        "xgt.cmd",
        "xgt.dtype",
        "xgt.blkcnt",
        "xgt.err_code",
        "xgt.datasize",
    ]

    if df.empty:
        norm_params = {f"{c}_min": -1.0 for c in norm_cols} | {f"{c}_max": -1.0 for c in norm_cols}
        (out_dir / "xgt_norm_params.json").write_text(json.dumps(norm_params, indent=2, ensure_ascii=False), encoding="utf-8-sig")
        (out_dir / "xgt_var_vocab.json").write_text(json.dumps({}, indent=2, ensure_ascii=False), encoding="utf-8-sig")

        dtype = np.dtype([
            ("xgt_var_id", "i4"),
            ("xgt_var_cnt_norm", "f4"),
            ("xgt_source_norm", "f4"),
            ("xgt_len_norm", "f4"),
            ("xgt_fenet_base_norm", "f4"),
            ("xgt_fenet_slot_norm", "f4"),
            ("xgt_cmd_norm", "f4"),
            ("xgt_dtype_norm", "f4"),
            ("xgt_blkcnt_norm", "f4"),
            ("xgt_err_flag", "f4"),
            ("xgt_err_code_norm", "f4"),
            ("xgt_datasize_norm", "f4"),
            ("xgt_addr_count", "f4"),
            ("xgt_addr_min", "f4"),
            ("xgt_addr_max", "f4"),
            ("xgt_addr_range", "f4"),
            ("xgt_word_min", "f4"),
            ("xgt_word_max", "f4"),
            ("xgt_word_mean", "f4"),
            ("xgt_word_std", "f4"),
        ])
        data = np.zeros(0, dtype=dtype)
        np.save(out_dir / "xgt.npy", data)
        return

    n = len(df)

    # ---------- 0) var_vocab + xgt.var_cnt 생성 ----------
    var_map: Dict[str, int] = {}
    vars_series = _df_get_first(df, ["xgt_fen.regs.vars", "xgt_fen.vars", "xgt.vars"], n)

    var_ids: List[int] = []
    var_cnts: List[float] = []

    for v in vars_series.tolist():
        first_var, var_cnt = _parse_first_var_and_cnt(v)
        var_ids.append(_get_var_id(var_map, first_var))
        var_cnts.append(var_cnt)

    var_map = {k: v for k, v in var_map.items() if _clean_var_name(k)}
    (out_dir / "xgt_var_vocab.json").write_text(json.dumps(var_map, indent=2, ensure_ascii=False), encoding="utf-8-sig")

    df["xgt_var_id"] = np.array(var_ids, dtype=np.int32)
    df["xgt.var_cnt"] = pd.to_numeric(pd.Series(var_cnts, index=df.index), errors="coerce").astype("float32")

    # ---------- 1) scalar RAW 컬럼 생성 ----------
    df["xgt.source"] = pd.to_numeric(_df_get_first(df, ["xgt.source", "xgt_fen.source", "xgt_fen.regs.source"], n).apply(_hex_to_float), errors="coerce").astype("float32")
    df["xgt.len"] = pd.to_numeric(_df_get_first(df, ["xgt.len", "xgt_fen.regs.len", "xgt_fen.len"], n).apply(_hex_to_float), errors="coerce").astype("float32")
    df["xgt.cmd"] = pd.to_numeric(_df_get_first(df, ["xgt.cmd", "xgt_fen.regs.cmd", "xgt_fen.cmd"], n).apply(_hex_to_float), errors="coerce").astype("float32")
    df["xgt.dtype"] = pd.to_numeric(_df_get_first(df, ["xgt.dtype", "xgt_fen.regs.dtype", "xgt_fen.dtype", "xgt_fen.regs.dype", "xgt_fen.dype"], n).apply(_hex_to_float), errors="coerce").astype("float32")
    df["xgt.blkcnt"] = pd.to_numeric(_df_get_first(df, ["xgt.blkcnt", "xgt_fen.regs.blkcnt", "xgt_fen.blkcnt"], n).apply(_hex_to_float), errors="coerce").astype("float32")
    df["xgt.err_code"] = pd.to_numeric(_df_get_first(df, ["xgt.errstat", "xgt_fen.regs.errstat", "xgt_fen.errstat"], n).apply(_hex_to_float), errors="coerce").astype("float32")
    df["xgt.datasize"] = pd.to_numeric(_df_get_first(df, ["xgt.datasize", "xgt_fen.regs.datasize", "xgt_fen.datasize"], n).apply(_hex_to_float), errors="coerce").astype("float32")

    # ---------- 1-1) fenetpos -> base/slot (여기서 float로 바뀌는 문제 방지) ----------
    fenetpos_src = _df_get_first(df, ["xgt.fenetpos", "xgt_fen.regs.fenetpos", "xgt_fen.fenetpos"], n).tolist()
    fenetpos_list = [_hex_to_int(v) for v in fenetpos_src]
    fenetpos_series = pd.Series(fenetpos_list, index=df.index, dtype="object")

    df["xgt.fenet_base"] = fenetpos_series.apply(_fenet_base).astype("float32")
    df["xgt.fenet_slot"] = fenetpos_series.apply(_fenet_slot).astype("float32")

    df["xgt_err_flag"] = df["xgt.err_code"].apply(
        lambda v: 1.0 if (v is not None and math.isfinite(float(v)) and float(v) != 0.0) else 0.0
    ).astype("float32")

    # ---------- 2) min-max 파라미터 산출 ----------
    norm_params = minmax_norm_scalar(df, norm_cols)
    print(norm_params)

    # ---------- 3) min-max 정규화 적용 ----------
    vminmax = np.vectorize(minmax_cal, otypes=[np.float32])
    for col in norm_cols:
        series = pd.to_numeric(
            df.get(col, pd.Series([np.nan] * n, index=df.index)),
            errors="coerce"
        ).astype("float32")

        arr = series.to_numpy(copy=False)
        out = np.full(arr.shape, -1.0, dtype=np.float32)

        mask = ~np.isnan(arr)
        vmin = float(norm_params.get(f"{col}_min", 0.0))
        vmax = float(norm_params.get(f"{col}_max", 0.0))

        out[mask] = vminmax(arr[mask], vmin, vmax)
        safe_col = col.replace(".", "_")
        df[f"{safe_col}_norm"] = out

    # ---------- 4) vocab + norm_params 저장 ----------
    (out_dir / "xgt_norm_params.json").write_text(json.dumps(norm_params, indent=2, ensure_ascii=False), encoding="utf-8-sig")

    # ---------- 5) list stats 추출 ----------
    translated_series = _df_get_first(df, ["xgt_fen.regs.translated_addr", "xgt_fen.translated_addr", "xgt.translated_addr"], n)
    word_series = _df_get_first(df, ["xgt_fen.regs.word_value", "xgt_fen.word_value", "xgt.word_value"], n)

    addr_stats = translated_series.apply(lambda v: stats_count_min_max_range(_parse_translated_addr(v)))
    df["xgt_addr_count"] = addr_stats.map(lambda r: float(r.get("count", 0.0) or 0.0)).astype("float32")
    df["xgt_addr_min"] = addr_stats.map(lambda r: float(r["min"]) if r.get("min") is not None else -1.0).astype("float32")
    df["xgt_addr_max"] = addr_stats.map(lambda r: float(r["max"]) if r.get("max") is not None else -1.0).astype("float32")
    df["xgt_addr_range"] = addr_stats.map(lambda r: float(r["range"]) if r.get("range") is not None else -1.0).astype("float32")

    word_stats = word_series.apply(lambda v: stats_min_max_mean_std(_parse_word_value(v), ddof=0))
    df["xgt_word_min"] = word_stats.map(lambda r: float(r["min"]) if r.get("min") is not None else -1.0).astype("float32")
    df["xgt_word_max"] = word_stats.map(lambda r: float(r["max"]) if r.get("max") is not None else -1.0).astype("float32")
    df["xgt_word_mean"] = word_stats.map(lambda r: float(r["mean"]) if r.get("mean") is not None else -1.0).astype("float32")
    df["xgt_word_std"] = word_stats.map(lambda r: float(r["std"]) if r.get("std") is not None else -1.0).astype("float32")

    # ---------- 6) xgt.npy 저장 ----------
    dtype = np.dtype([
        ("xgt_var_id", "i4"),
        ("xgt_var_cnt_norm", "f4"),
        ("xgt_source_norm", "f4"),
        ("xgt_len_norm", "f4"),
        ("xgt_fenet_base_norm", "f4"),
        ("xgt_fenet_slot_norm", "f4"),
        ("xgt_cmd_norm", "f4"),
        ("xgt_dtype_norm", "f4"),
        ("xgt_blkcnt_norm", "f4"),
        ("xgt_err_flag", "f4"),
        ("xgt_err_code_norm", "f4"),
        ("xgt_datasize_norm", "f4"),
        ("xgt_addr_count", "f4"),
        ("xgt_addr_min", "f4"),
        ("xgt_addr_max", "f4"),
        ("xgt_addr_range", "f4"),
        ("xgt_word_min", "f4"),
        ("xgt_word_max", "f4"),
        ("xgt_word_mean", "f4"),
        ("xgt_word_std", "f4"),
    ])

    data = np.zeros(len(df), dtype=dtype)

    data["xgt_var_id"]            = df["xgt_var_id"].to_numpy(dtype=np.int32, copy=False)

    data["xgt_var_cnt_norm"]      = df["xgt_var_cnt_norm"].to_numpy(dtype=np.float32, copy=False)
    data["xgt_source_norm"]       = df["xgt_source_norm"].to_numpy(dtype=np.float32, copy=False)
    data["xgt_len_norm"]          = df["xgt_len_norm"].to_numpy(dtype=np.float32, copy=False)
    data["xgt_fenet_base_norm"]   = df["xgt_fenet_base_norm"].to_numpy(dtype=np.float32, copy=False)
    data["xgt_fenet_slot_norm"]   = df["xgt_fenet_slot_norm"].to_numpy(dtype=np.float32, copy=False)
    data["xgt_cmd_norm"]          = df["xgt_cmd_norm"].to_numpy(dtype=np.float32, copy=False)
    data["xgt_dtype_norm"]        = df["xgt_dtype_norm"].to_numpy(dtype=np.float32, copy=False)
    data["xgt_blkcnt_norm"]       = df["xgt_blkcnt_norm"].to_numpy(dtype=np.float32, copy=False)
    data["xgt_err_flag"]          = df["xgt_err_flag"].to_numpy(dtype=np.float32, copy=False)
    data["xgt_err_code_norm"]     = df["xgt_err_code_norm"].to_numpy(dtype=np.float32, copy=False)
    data["xgt_datasize_norm"]     = df["xgt_datasize_norm"].to_numpy(dtype=np.float32, copy=False)

    data["xgt_addr_count"]        = df["xgt_addr_count"].to_numpy(dtype=np.float32, copy=False)
    data["xgt_addr_min"]          = df["xgt_addr_min"].to_numpy(dtype=np.float32, copy=False)
    data["xgt_addr_max"]          = df["xgt_addr_max"].to_numpy(dtype=np.float32, copy=False)
    data["xgt_addr_range"]        = df["xgt_addr_range"].to_numpy(dtype=np.float32, copy=False)

    data["xgt_word_min"]          = df["xgt_word_min"].to_numpy(dtype=np.float32, copy=False)
    data["xgt_word_max"]          = df["xgt_word_max"].to_numpy(dtype=np.float32, copy=False)
    data["xgt_word_mean"]         = df["xgt_word_mean"].to_numpy(dtype=np.float32, copy=False)
    data["xgt_word_std"]          = df["xgt_word_std"].to_numpy(dtype=np.float32, copy=False)

    np.save(out_dir / "xgt.npy", data)

    print("\n===== 앞 5개 전처리 샘플 =====")
    for i in range(min(5, len(data))):
        print({
            "xgt_var_id"          : int(data["xgt_var_id"][i]),
            "xgt_var_cnt_norm"    : float(data["xgt_var_cnt_norm"][i]),
            "xgt_source_norm"     : float(data["xgt_source_norm"][i]),
            "xgt_len_norm"        : float(data["xgt_len_norm"][i]),
            "xgt_fenet_base_norm" : float(data["xgt_fenet_base_norm"][i]),
            "xgt_fenet_slot_norm" : float(data["xgt_fenet_slot_norm"][i]),
            "xgt_cmd_norm"        : float(data["xgt_cmd_norm"][i]),
            "xgt_dtype_norm"      : float(data["xgt_dtype_norm"][i]),
            "xgt_blkcnt_norm"     : float(data["xgt_blkcnt_norm"][i]),
            "xgt_err_flag"        : float(data["xgt_err_flag"][i]),
            "xgt_err_code_norm"   : float(data["xgt_err_code_norm"][i]),
            "xgt_datasize_norm"   : float(data["xgt_datasize_norm"][i]),
            "xgt_addr_count"      : float(data["xgt_addr_count"][i]),
            "xgt_addr_min"        : float(data["xgt_addr_min"][i]),
            "xgt_addr_max"        : float(data["xgt_addr_max"][i]),
            "xgt_addr_range"      : float(data["xgt_addr_range"][i]),
            "xgt_word_min"        : float(data["xgt_word_min"][i]),
            "xgt_word_max"        : float(data["xgt_word_max"][i]),
            "xgt_word_mean"       : float(data["xgt_word_mean"][i]),
            "xgt_word_std"        : float(data["xgt_word_std"][i]),
        })


# 단일 패킷 전처리 함수 (운영 단계에서 사용)
def preprocess_xgt(records: Dict[str, Any], norm_params: Dict[str, Any], var_map: Dict[str, int]) -> Dict[str, Any]:
    vars_value = records.get("xgt_fen.regs.vars")
    if vars_value in (None, ""):
        vars_value = records.get("xgt_fen.vars")
    if vars_value in (None, ""):
        vars_value = records.get("xgt.vars")

    first_var, var_cnt_float = _parse_first_var_and_cnt(vars_value)

    if var_cnt_float is None or (isinstance(var_cnt_float, float) and not math.isfinite(var_cnt_float)):
        var_cnt_float = -1.0

    xgt_var_id = int(var_map.get(first_var, 0)) if first_var else 0

    source = records.get("xgt_fen.source")
    length = records.get("xgt_fen.len")
    fenetpos = records.get("xgt_fen.fenetpos")
    cmd = records.get("xgt_fen.cmd")
    dtype = records.get("xgt_fen.dtype")
    blkcnt = records.get("xgt_fen.blkcnt")
    errstat = records.get("xgt_fen.errstat")
    datasize = records.get("xgt_fen.datasize")

    source_float = _hex_to_float(source)
    len_float = _hex_to_float(length)
    cmd_float = _hex_to_float(cmd)
    dtype_float = _hex_to_float(dtype)
    blkcnt_float = _hex_to_float(blkcnt)
    err_code_float = _hex_to_float(errstat)
    datasize_float = _hex_to_float(datasize)

    fenetpos_int = _hex_to_int(fenetpos)
    base_float = _fenet_base(fenetpos_int)
    slot_float = _fenet_slot(fenetpos_int)

    err_flag = 0.0
    if err_code_float is not None and math.isfinite(float(err_code_float)) and float(err_code_float) != 0.0:
        err_flag = 1.0

    var_cnt_min, var_cnt_max = _to_float(norm_params.get("xgt.var_cnt_min")), _to_float(norm_params.get("xgt.var_cnt_max"))
    source_min, source_max   = _to_float(norm_params.get("xgt.source_min")), _to_float(norm_params.get("xgt.source_max"))
    len_min, len_max         = _to_float(norm_params.get("xgt.len_min")), _to_float(norm_params.get("xgt.len_max"))
    base_min, base_max       = _to_float(norm_params.get("xgt.fenet_base_min")), _to_float(norm_params.get("xgt.fenet_base_max"))
    slot_min, slot_max       = _to_float(norm_params.get("xgt.fenet_slot_min")), _to_float(norm_params.get("xgt.fenet_slot_max"))
    cmd_min, cmd_max         = _to_float(norm_params.get("xgt.cmd_min")), _to_float(norm_params.get("xgt.cmd_max"))
    dtype_min, dtype_max     = _to_float(norm_params.get("xgt.dtype_min")), _to_float(norm_params.get("xgt.dtype_max"))
    blkcnt_min, blkcnt_max   = _to_float(norm_params.get("xgt.blkcnt_min")), _to_float(norm_params.get("xgt.blkcnt_max"))
    err_min, err_max         = _to_float(norm_params.get("xgt.err_code_min")), _to_float(norm_params.get("xgt.err_code_max"))
    datasize_min, datasize_max = _to_float(norm_params.get("xgt.datasize_min")), _to_float(norm_params.get("xgt.datasize_max"))

    translated_addr = records.get("xgt_fen.regs.translated_addr")
    if translated_addr in (None, ""):
        translated_addr = records.get("xgt_fen.translated_addr")

    word_value = records.get("xgt_fen.regs.word_value")
    if word_value in (None, ""):
        word_value = records.get("xgt_fen.word_value")

    addr_list = _parse_translated_addr(translated_addr)
    word_list = _parse_word_value(word_value)

    addr_stats = stats_count_min_max_range(addr_list)
    word_stats = stats_min_max_mean_std(word_list, ddof=0)

    def _safe_norm(v: Any, vmin: Any, vmax: Any) -> float:
        out = minmax_cal(v, vmin, vmax)
        if out is None:
            return -1.0
        try:
            f = float(out)
            return f if math.isfinite(f) else -1.0
        except Exception:
            return -1.0

    return {
        "xgt_var_id"            : int(xgt_var_id),
        "xgt_var_cnt_norm"      : float(_safe_norm(var_cnt_float, var_cnt_min, var_cnt_max)),
        "xgt_source_norm"       : float(_safe_norm(source_float, source_min, source_max)),
        "xgt_len_norm"          : float(_safe_norm(len_float, len_min, len_max)),
        "xgt_fenet_base_norm"   : float(_safe_norm(base_float, base_min, base_max)),
        "xgt_fenet_slot_norm"   : float(_safe_norm(slot_float, slot_min, slot_max)),
        "xgt_cmd_norm"          : float(_safe_norm(cmd_float, cmd_min, cmd_max)),
        "xgt_dtype_norm"        : float(_safe_norm(dtype_float, dtype_min, dtype_max)),
        "xgt_blkcnt_norm"       : float(_safe_norm(blkcnt_float, blkcnt_min, blkcnt_max)),
        "xgt_err_flag"          : float(err_flag),
        "xgt_err_code_norm"     : float(_safe_norm(err_code_float, err_min, err_max)),
        "xgt_datasize_norm"     : float(_safe_norm(datasize_float, datasize_min, datasize_max)),
        "xgt_addr_count"        : float(addr_stats.get("count", 0.0) or 0.0),
        "xgt_addr_min"          : float(addr_stats["min"]) if addr_stats.get("min") is not None else -1.0,
        "xgt_addr_max"          : float(addr_stats["max"]) if addr_stats.get("max") is not None else -1.0,
        "xgt_addr_range"        : float(addr_stats["range"]) if addr_stats.get("range") is not None else -1.0,
        "xgt_word_min"          : float(word_stats["min"]) if word_stats.get("min") is not None else -1.0,
        "xgt_word_max"          : float(word_stats["max"]) if word_stats.get("max") is not None else -1.0,
        "xgt_word_mean"         : float(word_stats["mean"]) if word_stats.get("mean") is not None else -1.0,
        "xgt_word_std"          : float(word_stats["std"]) if word_stats.get("std") is not None else -1.0,
    }


def transform_preprocess_xgt(packet: Dict[str, Any], param_dir: Path) -> Dict[str, Any]:
    norm_path = param_dir / "xgt_norm_params.json"
    vocab_path = param_dir / "xgt_var_vocab.json"

    norm_params = file_load("json", str(norm_path)) or {}
    var_map = file_load("json", str(vocab_path)) or {}

    var_map = {k: v for k, v in var_map.items() if _clean_var_name(k)}

    return preprocess_xgt(packet, norm_params, var_map)


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
        fit_preprocess_xgt(input_path, out_dir)
    elif args.transform:
        packets = file_load("jsonl", str(input_path)) or []
        for pkt in packets:
            if not isinstance(pkt, dict):
                continue
            feat = transform_preprocess_xgt(pkt, out_dir)
            print(feat)
    else:
        raise ValueError("❌ 반드시 --fit 또는 --transform 중 하나를 선택하세요.")
