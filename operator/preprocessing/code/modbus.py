#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
modbus.py
modbus 전용 feature 전처리

두 모드 제공:
  --fit        : min-max 정규화 파라미터 생성 + modbus.npy 저장
  --transform  : 기존 정규화 파라미터 사용하여 modbus.npy 생성

입력 JSONL에서 사용하는 필드:
  - protocol == "modbus" 또는 "modbus_tcp" 등 (필요시 수정)
  - modbus.addr       : 정수화 후 min-max 정규화
  - modbus.fc         : 정수화 후 min-max 정규화
  - modbus.qty        : 정수화 후 min-max 정규화
  - modbus.bc         : 정수화 후 min-max 정규화
  - modbus.regs.addr  : 레지스터 주소 리스트
  - modbus.regs.val   : 레지스터 값 리스트
  - modbus.translated_addr : 존재하면 regs.addr 대신 사용

출력 feature (modbus.npy, structured numpy):
  - modbus_addr_norm   (float32)  ← modbus.addr min-max 정규화
  - modbus_fc_norm     (float32)  ← modbus.fc min-max 정규화
  - modbus_qty_norm    (float32)  ← modbus.qty min-max 정규화
  - modbus_bc_norm     (float32)  ← modbus.bc min-max 정규화

  ⚠ 아래 regs_* 계열도 모두 **min-max 정규화된 값**으로 저장됨
  - regs_addr_count    (float32)  ← len(modbus.regs.addr)의 min-max 정규화
  - regs_addr_min      (float32)  ← min(modbus.regs.addr)의 min-max 정규화
  - regs_addr_max      (float32)  ← max(modbus.regs.addr)의 min-max 정규화
  - regs_addr_range    (float32)  ← (max-min)의 min-max 정규화

  - regs_val_min       (float32)  ← min(modbus.regs.val)의 min-max 정규화
  - regs_val_max       (float32)  ← max(modbus.regs.val)의 min-max 정규화
  - regs_val_mean      (float32)  ← mean(modbus.regs.val)의 min-max 정규화
  - regs_val_std       (float32)  ← std(modbus.regs.val)의 min-max 정규화

보조 파일:
  - modbus_norm_params.json
      {
        "modbus.addr": {"min": ..., "max": ...},
        "modbus.fc":   {"min": ..., "max": ...},
        "modbus.qty":  {"min": ..., "max": ...},
        "modbus.bc":   {"min": ..., "max": ...},

        "regs_addr.count": {"min": ..., "max": ...},
        "regs_addr.min":   {"min": ..., "max": ...},
        "regs_addr.max":   {"min": ..., "max": ...},
        "regs_addr.range": {"min": ..., "max": ...},

        "regs_val.min":    {"min": ..., "max": ...},
        "regs_val.max":    {"min": ..., "max": ...},
        "regs_val.mean":   {"min": ..., "max": ...},
        "regs_val.std":    {"min": ..., "max": ...}
      }

실시간 / 단일 패킷 처리:
  - modbus_norm_params.json 로드 후
    preprocess_modbus_with_norm(obj, norm_params) 호출
"""

import json
import argparse
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------
# 공용 유틸
# ---------------------------------------------
def _flatten_int_like(x: Any) -> List[int]:
    """
    x가
      - 숫자
      - 숫자 문자열 ("123")
      - 리스트 [1,"2",["3","4"], ...]
      - JSON 문자열 '["1","2","3"]'
    어떤 형태든 최종적으로 int 리스트로 평탄화
    """
    out: List[int] = []

    # 리스트/튜플이면 재귀 flatten
    if isinstance(x, (list, tuple)):
        for y in x:
            out.extend(_flatten_int_like(y))
        return out

    # 문자열 처리
    if isinstance(x, str):
        s = x.strip()
        # '[...]' 형태면 json.loads 시도
        if s.startswith("[") and s.endswith("]"):
            try:
                loaded = json.loads(s)
                out.extend(_flatten_int_like(loaded))
                return out
            except Exception:
                # json 파싱 실패하면 그냥 숫자 문자열로 보고 아래에서 처리
                pass
        try:
            out.append(int(s))
        except Exception:
            pass
        return out

    # 그 외 (int/float 등)
    try:
        out.append(int(x))
    except Exception:
        pass
    return out


def _flatten_float_like(x: Any) -> List[float]:
    """
    x가
      - 숫자
      - 숫자 문자열 ("123.4")
      - 리스트 [1.0,"2.5",[...]]
      - JSON 문자열 '["1","2","3.5"]'
    어떤 형태든 최종적으로 float 리스트로 평탄화
    """
    out: List[float] = []

    if isinstance(x, (list, tuple)):
        for y in x:
            out.extend(_flatten_float_like(y))
        return out

    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                loaded = json.loads(s)
                out.extend(_flatten_float_like(loaded))
                return out
            except Exception:
                pass
        try:
            out.append(float(s))
        except Exception:
            pass
        return out

    try:
        out.append(float(x))
    except Exception:
        pass
    return out


def parse_int_field(val: Any) -> Optional[int]:
    if isinstance(val, list) and val:
        val = val[0]
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def parse_float_field(val: Any) -> Optional[float]:
    if isinstance(val, list) and val:
        val = val[0]
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def parse_int_list_field(val: Any) -> List[int]:
    if val is None:
        return []
    return _flatten_int_like(val)


def parse_float_list_field(val: Any) -> List[float]:
    if val is None:
        return []
    return _flatten_float_like(val)


def minmax_norm(v: Optional[float], vmin: Optional[float], vmax: Optional[float]) -> float:
    """
    단순 min-max 정규화:
      (v - vmin) / (vmax - vmin)

    - v가 None 이거나,
    - vmin/vmax가 없거나,
    - vmin == vmax 인 경우 → 0.0 반환
    """
    if v is None or vmin is None or vmax is None:
        return 0.0
    if vmax == vmin:
        return 0.0
    return float(v - vmin) / float(vmax - vmin)


# ---------------------------------------------
# 한 레코드(modbus)에서 raw 값 추출
# ---------------------------------------------
def extract_modbus_raw(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    addr = parse_int_field(obj.get("modbus.addr"))
    fc   = parse_int_field(obj.get("modbus.fc"))
    qty  = parse_int_field(obj.get("modbus.qty"))
    bc   = parse_int_field(obj.get("modbus.bc"))

    # 🔸 주소는 modbus.regs.translated_addr 를 최우선 사용
    #     없으면 modbus.translated_addr → 없으면 modbus.regs.addr
    raw_addr_source = obj.get("modbus.regs.translated_addr")
    if not raw_addr_source:  # [], None, "" 등 모두 포함
        raw_addr_source = obj.get("modbus.translated_addr")
    if not raw_addr_source:
        raw_addr_source = obj.get("modbus.regs.addr")

    regs_addr = parse_int_list_field(raw_addr_source)
    regs_val  = parse_float_list_field(obj.get("modbus.regs.val"))

    # 아무 정보도 없는 패킷이면 스킵
    if all(v is None for v in (addr, fc, qty, bc)) and (not regs_addr) and (not regs_val):
        return None

    return {
        "addr": addr,
        "fc": fc,
        "qty": qty,
        "bc": bc,
        "regs_addr": regs_addr,   # ← 이제 여기 안에 modbus.regs.translated_addr 값이 들어감
        "regs_val": regs_val,
    }


# ---------------------------------------------
# regs.* 통계 계산
# ---------------------------------------------
def compute_regs_addr_stats(addrs: List[int]) -> Dict[str, float]:
    if not addrs:
        return {
            "count": 0.0,
            "min": 0.0,
            "max": 0.0,
            "range": 0.0,
        }
    count = float(len(addrs))
    amin = float(min(addrs))
    amax = float(max(addrs))
    arange = float(amax - amin)
    return {
        "count": count,
        "min": amin,
        "max": amax,
        "range": arange,
    }


def compute_regs_val_stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
        }
    vmin = float(min(vals))
    vmax = float(max(vals))
    mean = float(sum(vals) / len(vals))
    var = float(sum((v - mean) ** 2 for v in vals) / len(vals))
    std = float(math.sqrt(var))
    return {
        "min": vmin,
        "max": vmax,
        "mean": mean,
        "std": std,
    }


def _update_minmax(cur_min: Optional[float], cur_max: Optional[float], v: float) -> (Optional[float], Optional[float]):
    if cur_min is None or v < cur_min:
        cur_min = v
    if cur_max is None or v > cur_max:
        cur_max = v
    return cur_min, cur_max


# ---------------------------------------------
# 단일 패킷 + 정규화까지 처리 (실시간/운영 용)
# ---------------------------------------------
def preprocess_modbus_with_norm(obj: Dict[str, Any],
                                norm_params: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    단일 Modbus 패킷 obj에 대해
    - modbus_addr_norm, modbus_fc_norm, ...
    12개 feature를 모두 정규화된 값(0~1)으로 담은 dict 반환.

    norm_params 예시 (modbus_norm_params.json):
        {
          "modbus.addr": {"min": ..., "max": ...},
          ...
          "regs_val.std": {"min": ..., "max": ...}
        }
    """
    raw = extract_modbus_raw(obj)
    if raw is None:
        return None

    addr = raw["addr"]
    fc   = raw["fc"]
    qty  = raw["qty"]
    bc   = raw["bc"]
    regs_addr = raw["regs_addr"]
    regs_val  = raw["regs_val"]

    addr_stats = compute_regs_addr_stats(regs_addr)
    val_stats  = compute_regs_val_stats(regs_val)

    mp = norm_params

    addr_min = mp.get("modbus.addr", {}).get("min")
    addr_max = mp.get("modbus.addr", {}).get("max")
    fc_min   = mp.get("modbus.fc",   {}).get("min")
    fc_max   = mp.get("modbus.fc",   {}).get("max")
    qty_min  = mp.get("modbus.qty",  {}).get("min")
    qty_max  = mp.get("modbus.qty",  {}).get("max")
    bc_min   = mp.get("modbus.bc",   {}).get("min")
    bc_max   = mp.get("modbus.bc",   {}).get("max")

    ra_count_min = mp.get("regs_addr.count", {}).get("min")
    ra_count_max = mp.get("regs_addr.count", {}).get("max")
    ra_min_min   = mp.get("regs_addr.min",   {}).get("min")
    ra_min_max   = mp.get("regs_addr.min",   {}).get("max")
    ra_max_min   = mp.get("regs_addr.max",   {}).get("min")
    ra_max_max   = mp.get("regs_addr.max",   {}).get("max")
    ra_range_min = mp.get("regs_addr.range", {}).get("min")
    ra_range_max = mp.get("regs_addr.range", {}).get("max")

    rv_min_min   = mp.get("regs_val.min",    {}).get("min")
    rv_min_max   = mp.get("regs_val.min",    {}).get("max")
    rv_max_min   = mp.get("regs_val.max",    {}).get("min")
    rv_max_max   = mp.get("regs_val.max",    {}).get("max")
    rv_mean_min  = mp.get("regs_val.mean",   {}).get("min")
    rv_mean_max  = mp.get("regs_val.mean",   {}).get("max")
    rv_std_min   = mp.get("regs_val.std",    {}).get("min")
    rv_std_max   = mp.get("regs_val.std",    {}).get("max")

    feat: Dict[str, float] = {
        "modbus_addr_norm": minmax_norm(float(addr) if addr is not None else None, addr_min, addr_max),
        "modbus_fc_norm":   minmax_norm(float(fc)   if fc   is not None else None, fc_min,   fc_max),
        "modbus_qty_norm":  minmax_norm(float(qty)  if qty  is not None else None, qty_min,  qty_max),
        "modbus_bc_norm":   minmax_norm(float(bc)   if bc   is not None else None, bc_min,   bc_max),

        "regs_addr_count":  minmax_norm(addr_stats["count"], ra_count_min, ra_count_max),
        "regs_addr_min":    minmax_norm(addr_stats["min"],   ra_min_min,   ra_min_max),
        "regs_addr_max":    minmax_norm(addr_stats["max"],   ra_max_min,   ra_max_max),
        "regs_addr_range":  minmax_norm(addr_stats["range"], ra_range_min, ra_range_max),

        "regs_val_min":     minmax_norm(val_stats["min"],    rv_min_min,   rv_min_max),
        "regs_val_max":     minmax_norm(val_stats["max"],    rv_max_min,   rv_max_max),
        "regs_val_mean":    minmax_norm(val_stats["mean"],   rv_mean_min,  rv_mean_max),
        "regs_val_std":     minmax_norm(val_stats["std"],    rv_std_min,   rv_std_max),
    }

    return feat


# ---------------------------------------------
# FIT
# ---------------------------------------------
def fit_preprocess(input_path: Path, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    rows_raw: List[Dict[str, Any]] = []

    # min/max 추적용 (기본 modbus 필드)
    addr_min = addr_max = None
    fc_min = fc_max = None
    qty_min = qty_max = None
    bc_min = bc_max = None

    # min/max 추적용 (regs_addr / regs_val 통계)
    ra_count_min = ra_count_max = None
    ra_min_min = ra_min_max = None
    ra_max_min = ra_max_max = None
    ra_range_min = ra_range_max = None

    rv_min_min = rv_min_max = None
    rv_max_min = rv_max_max = None
    rv_mean_min = rv_mean_max = None
    rv_std_min = rv_std_max = None

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # 필요시 protocol 값 조건 수정 (예: "modbus_tcp")
            if obj.get("protocol") != "modbus":
                continue

            raw = extract_modbus_raw(obj)
            if raw is None:
                continue

            rows_raw.append(raw)

            # 기본 필드 min/max 업데이트
            if raw["addr"] is not None:
                addr_min, addr_max = _update_minmax(addr_min, addr_max, float(raw["addr"]))
            if raw["fc"] is not None:
                fc_min, fc_max = _update_minmax(fc_min, fc_max, float(raw["fc"]))
            if raw["qty"] is not None:
                qty_min, qty_max = _update_minmax(qty_min, qty_max, float(raw["qty"]))
            if raw["bc"] is not None:
                bc_min, bc_max = _update_minmax(bc_min, bc_max, float(raw["bc"]))

            # regs 통계 계산 후 min/max 업데이트
            addr_stats = compute_regs_addr_stats(raw["regs_addr"])
            val_stats  = compute_regs_val_stats(raw["regs_val"])

            ra_count_min, ra_count_max = _update_minmax(ra_count_min, ra_count_max, addr_stats["count"])
            ra_min_min,   ra_min_max   = _update_minmax(ra_min_min,   ra_min_max,   addr_stats["min"])
            ra_max_min,   ra_max_max   = _update_minmax(ra_max_min,   ra_max_max,   addr_stats["max"])
            ra_range_min, ra_range_max = _update_minmax(ra_range_min, ra_range_max, addr_stats["range"])

            rv_min_min,  rv_min_max  = _update_minmax(rv_min_min,  rv_min_max,  val_stats["min"])
            rv_max_min,  rv_max_max  = _update_minmax(rv_max_min,  rv_max_max,  val_stats["max"])
            rv_mean_min, rv_mean_max = _update_minmax(rv_mean_min, rv_mean_max, val_stats["mean"])
            rv_std_min,  rv_std_max  = _update_minmax(rv_std_min,  rv_std_max,  val_stats["std"])

    # 정규화 파라미터 저장
    norm_params = {
        "modbus.addr": {"min": addr_min, "max": addr_max},
        "modbus.fc":   {"min": fc_min,   "max": fc_max},
        "modbus.qty":  {"min": qty_min,  "max": qty_max},
        "modbus.bc":   {"min": bc_min,   "max": bc_max},

        "regs_addr.count": {"min": ra_count_min, "max": ra_count_max},
        "regs_addr.min":   {"min": ra_min_min,   "max": ra_min_max},
        "regs_addr.max":   {"min": ra_max_min,   "max": ra_max_max},
        "regs_addr.range": {"min": ra_range_min, "max": ra_range_max},

        "regs_val.min":    {"min": rv_min_min,   "max": rv_min_max},
        "regs_val.max":    {"min": rv_max_min,   "max": rv_max_max},
        "regs_val.mean":   {"min": rv_mean_min,  "max": rv_mean_max},
        "regs_val.std":    {"min": rv_std_min,   "max": rv_std_max},
    }

    norm_path = out_dir / "modbus_norm_params.json"
    norm_path.write_text(
        json.dumps(norm_params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("✅ FIT 완료")
    print(f"- modbus_norm_params.json 저장: {norm_path}")

    # numpy 구조화 배열 생성 (필드 이름은 그대로, 값만 0~1 범위)
    dtype = np.dtype([
        ("modbus_addr_norm", "f4"),
        ("modbus_fc_norm", "f4"),
        ("modbus_qty_norm", "f4"),
        ("modbus_bc_norm", "f4"),

        ("regs_addr_count", "f4"),
        ("regs_addr_min", "f4"),
        ("regs_addr_max", "f4"),
        ("regs_addr_range", "f4"),

        ("regs_val_min", "f4"),
        ("regs_val_max", "f4"),
        ("regs_val_mean", "f4"),
        ("regs_val_std", "f4"),
    ])

    data = np.zeros(len(rows_raw), dtype=dtype)

    for idx, raw in enumerate(rows_raw):
        addr = raw["addr"]
        fc   = raw["fc"]
        qty  = raw["qty"]
        bc   = raw["bc"]
        regs_addr = raw["regs_addr"]
        regs_val  = raw["regs_val"]

        addr_stats = compute_regs_addr_stats(regs_addr)
        val_stats  = compute_regs_val_stats(regs_val)

        # 1) min-max 정규화 필드 (기본 modbus 4개)
        data["modbus_addr_norm"][idx] = minmax_norm(
            float(addr) if addr is not None else None, addr_min, addr_max
        )
        data["modbus_fc_norm"][idx]   = minmax_norm(
            float(fc) if fc is not None else None, fc_min, fc_max
        )
        data["modbus_qty_norm"][idx]  = minmax_norm(
            float(qty) if qty is not None else None, qty_min, qty_max
        )
        data["modbus_bc_norm"][idx]   = minmax_norm(
            float(bc) if bc is not None else None, bc_min, bc_max
        )

        # 2) regs.addr 통계 → min-max 정규화
        data["regs_addr_count"][idx] = minmax_norm(
            addr_stats["count"], ra_count_min, ra_count_max
        )
        data["regs_addr_min"][idx]   = minmax_norm(
            addr_stats["min"],   ra_min_min,   ra_min_max
        )
        data["regs_addr_max"][idx]   = minmax_norm(
            addr_stats["max"],   ra_max_min,   ra_max_max
        )
        data["regs_addr_range"][idx] = minmax_norm(
            addr_stats["range"], ra_range_min, ra_range_max
        )

        # 3) regs.val 통계 → min-max 정규화
        data["regs_val_min"][idx]   = minmax_norm(
            val_stats["min"], rv_min_min, rv_min_max
        )
        data["regs_val_max"][idx]   = minmax_norm(
            val_stats["max"], rv_max_min, rv_max_max
        )
        data["regs_val_mean"][idx]  = minmax_norm(
            val_stats["mean"], rv_mean_min, rv_mean_max
        )
        data["regs_val_std"][idx]   = minmax_norm(
            val_stats["std"],  rv_std_min,  rv_std_max
        )

    npy_path = out_dir / "modbus.npy"
    np.save(npy_path, data)

    print(f"- modbus.npy 저장: {npy_path}")
    print(f"- shape: {data.shape}")

    print("\n===== 앞 5개 modbus 전처리 샘플 (정규화된 값) =====")
    for i in range(min(5, len(data))):
        sample = {name: data[name][i] for name in data.dtype.names}
        print(sample)


# ---------------------------------------------
# TRANSFORM
# ---------------------------------------------
def transform_preprocess(input_path: Path, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    norm_path = out_dir / "modbus_norm_params.json"
    if not norm_path.exists():
        raise FileNotFoundError(
            f"❌ {norm_path} 가 없습니다. 먼저 --fit 을 실행해서 정규화 파라미터를 생성하세요."
        )

    norm_params = json.loads(norm_path.read_text(encoding="utf-8"))

    rows_feat: List[Dict[str, float]] = []

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("protocol") != "modbus":
                continue

            feat = preprocess_modbus_with_norm(obj, norm_params)
            if feat is None:
                continue

            rows_feat.append(feat)

    if not rows_feat:
        print("⚠ 변환할 Modbus 레코드가 없습니다. 빈 modbus.npy를 생성합니다.")
        dtype_empty = np.dtype([
            ("modbus_addr_norm", "f4"),
            ("modbus_fc_norm", "f4"),
            ("modbus_qty_norm", "f4"),
            ("modbus_bc_norm", "f4"),

            ("regs_addr_count", "f4"),
            ("regs_addr_min", "f4"),
            ("regs_addr_max", "f4"),
            ("regs_addr_range", "f4"),

            ("regs_val_min", "f4"),
            ("regs_val_max", "f4"),
            ("regs_val_mean", "f4"),
            ("regs_val_std", "f4"),
        ])
        data_empty = np.zeros(0, dtype=dtype_empty)
        np.save(out_dir / "modbus.npy", data_empty)
        print(f"✅ TRANSFORM 완료 (empty). - modbus.npy 저장: {out_dir/'modbus.npy'} shape={data_empty.shape}")
        return

    dtype = np.dtype([
        ("modbus_addr_norm", "f4"),
        ("modbus_fc_norm", "f4"),
        ("modbus_qty_norm", "f4"),
        ("modbus_bc_norm", "f4"),

        ("regs_addr_count", "f4"),
        ("regs_addr_min", "f4"),
        ("regs_addr_max", "f4"),
        ("regs_addr_range", "f4"),

        ("regs_val_min", "f4"),
        ("regs_val_max", "f4"),
        ("regs_val_mean", "f4"),
        ("regs_val_std", "f4"),
    ])

    data = np.zeros(len(rows_feat), dtype=dtype)

    for idx, feat in enumerate(rows_feat):
        for name in data.dtype.names:
            data[name][idx] = float(feat.get(name, 0.0))

    npy_path = out_dir / "modbus.npy"
    np.save(npy_path, data)

    print("✅ TRANSFORM 완료")
    print(f"- modbus.npy 저장: {npy_path} shape={data.shape}")


# ---------------------------------------------
# MAIN
# ---------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="입력 JSONL 파일 경로")
    parser.add_argument("-o", "--output", required=True, help="출력 디렉토리 경로")
    parser.add_argument("--fit", action="store_true", help="정규화 파라미터 생성 + modbus.npy 생성")
    parser.add_argument("--transform", action="store_true", help="기존 파라미터로 modbus.npy 생성")

    args = parser.parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output)

    if args.fit and args.transform:
        raise ValueError("❌ --fit 과 --transform 는 동시에 사용할 수 없습니다.")
    if not args.fit and not args.transform:
        raise ValueError("❌ 반드시 --fit 또는 --transform 중 하나를 선택하세요.")

    if args.fit:
        fit_preprocess(input_path, out_dir)
    else:
        transform_preprocess(input_path, out_dir)


"""
최종 데이터 사용 예시 (modbus.npy)
    import numpy as np

    data = np.load("../result/output_modbus/modbus.npy")

    # shape: (N, )
    features = np.stack([
        data["modbus_addr_norm"],
        data["modbus_fc_norm"],
        data["modbus_qty_norm"],
        data["modbus_bc_norm"],
        data["regs_addr_count"],
        data["regs_addr_min"],
        data["regs_addr_max"],
        data["regs_addr_range"],
        data["regs_val_min"],
        data["regs_val_max"],
        data["regs_val_mean"],
        data["regs_val_std"],
    ], axis=1).astype("float32")


실시간 단일 패킷 예시:

    import json
    from pathlib import Path
    from modbus import preprocess_modbus_with_norm

    out_dir = Path("../result/output_modbus")
    norm_params = json.loads((out_dir / "modbus_norm_params.json").read_text(encoding="utf-8"))

    pkt = {
        "protocol": "modbus",
        "modbus.addr": "23",
        "modbus.fc": "4",
        "modbus.qty": "6",
        "modbus.bc": "12",
        "modbus.regs.addr": ["23", "24", "25", "26", "27", "28"],
        "modbus.regs.val":  ["23", "30", "242", "0", "28", "9"],
    }

    feat = preprocess_modbus_with_norm(pkt, norm_params)
    # feat = {
    #   "modbus_addr_norm": ...,
    #   "modbus_fc_norm": ...,
    #   ...
    #   "regs_val_std": ...
    # }

usage:
    # 1) 학습용 modbus 데이터에서 정규화 파라미터 + feature 생성
    python modbus.py --fit -i "../data/ML_DL 학습.jsonl" -o "../result/output_modbus"

    # 2) 이후 새 데이터에 대해 같은 파라미터로 전처리
    python modbus.py --transform -i "../data/ML_DL_새데이터.jsonl" -o "../result/output_modbus"
"""
