#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modbus.py  

두 가지 모드 제공:
  --fit       : norm_params 생성 후 modbus.npy 저장
  --transform : 기존 norm_params 사용

입력 JSONL에서 사용하는 필드:
  - modbus.tid             : modbus 구분자
  - modbus.fc              : modbus 명령어 코드
  - modbus.addr            : modbus 시작 주소
  - modbus.qty             : modbus 읽어올 레지스터 개수
  - modbus.bc              : modbus byte count
  - modbus.regs.addr       : modbus 레지스터 번호
  - modbus.regs.val        : modbus 레지스터 값
#   - modbus.translated_addr : modbus 레지스터 이름 

출력 feature (modbus.npy, structured numpy):
  - modbus_tid_norm        : modbus.tid min-max 정규화
  - modbus_fc_norm         : modbus.fc min-max 정규화
  - modbus_addr_norm       : modbus.ros min-max 정규화
  - modbus_qty_norm        : modbus.qty min-max 정규화
  - modbus_bc_norm         : modbus.bc min-max 정규화
  - modbus_regs_count      : modbus.regs.addr 개수
  - modbus_regs_addr_min   : modbus.regs.addr 최소
  - modbus_regs_addr_max   : modbus.regs.addr 최대
  - modbus_regs_addr_range : modbus.regs.addr 범위
  - modbus_regs_val_min    : modbus.regs.val 최소
  - modbus_regs_val_max    : modbus.regs.val 최대
  - modbus_regs_val_mean   : modbus.regs.val 평균
  - modbus_regs_val_std    : modbus.regs.val 표준편차
#   - modbus_slot_*_norm     : modbus.translated_addr과 regs.val의 값을 매핑

"""
import json, sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict

from min_max_normalize import minmax_cal, minmax_norm_scalar
from change_value_type import _to_float, _hex_to_float
from stats_from_list import stats_count_min_max_range, stats_min_max_mean_std

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.file_load import file_load



# fit 
def fit_preprocess_modbus(input_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(input_path, lines=True, encoding="utf-8-sig")

    norm_cols = ["modbus.tid", "modbus.fc", "modbus.addr", "modbus.qty", "modbus.bc"]

    if df.empty:
        norm_params = {f"{c}_min": -1.0 for c in norm_cols} | {f"{c}_max": -1.0 for c in norm_cols}

        (out_dir / "modbus_norm_params.json").write_text(json.dumps(norm_params, indent=2, ensure_ascii=False), encoding="utf-8-sig")

        dtype = np.dtype([
            ("modbus_tid_norm", "f4"),
            ("modbus_fc_norm", "f4"),
            ("modbus_addr_norm", "f4"),
            ("modbus_qty_norm", "f4"),
            ("modbus_bc_norm", "f4"),
            ("modbus_regs_count", "f4"),
            ("modbus_regs_addr_min", "f4"),
            ("modbus_regs_addr_max", "f4"),
            ("modbus_regs_addr_range", "f4"),
            ("modbus_regs_val_min", "f4"),
            ("modbus_regs_val_max", "f4"),
            ("modbus_regs_val_mean", "f4"),
            ("modbus_regs_val_std", "f4"),
        ])
        data = np.zeros(0, dtype=dtype)
        np.save(out_dir / "modbus.npy", data)
        return

    n = len(df)

    # ---------- 1) min-max 파라미터 산출 ----------
    norm_params = minmax_norm_scalar(df, norm_cols)
    print (norm_params)

    # ---------- 2) min-max 정규화 적용 ----------
    vminmax = np.vectorize(minmax_cal, otypes=[np.float32])
    for col in norm_cols:
        series = pd.to_numeric(
            df.get(col, pd.Series([np.nan]*n, index=df.index)),
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

    # ---------- 3) vocab + norm_params 저장 ----------
    (out_dir / "modbus_norm_params.json").write_text(json.dumps(norm_params, indent=2, ensure_ascii=False), encoding="utf-8-sig")


    # ---------- 4) list stats 추출 ----------
    regs_addr_stats = df.get("modbus.regs.addr", pd.Series([None] * n, index=df.index)).apply(
        lambda values: stats_count_min_max_range(
            values if isinstance(values, (list, tuple)) else ([] if values in (None, "") else [values])
        )
    )
    df["modbus_regs_count"] = regs_addr_stats.map(lambda result: float(result["count"])).astype("float32")
    df["modbus_regs_addr_min"] = regs_addr_stats.map(lambda result: result["min"] if result["min"] is not None else -1.0).astype("float32")
    df["modbus_regs_addr_max"] = regs_addr_stats.map(lambda result: result["max"] if result["max"] is not None else -1.0).astype("float32")
    df["modbus_regs_addr_range"] = regs_addr_stats.map(lambda result: result["range"] if result["range"] is not None else -1.0).astype("float32")


    regs_val_stats = df.get("modbus.regs.val", pd.Series([None] * n, index=df.index)).apply(
        lambda values: stats_min_max_mean_std(
            values if isinstance(values, (list, tuple)) else ([] if values in (None, "") else [values]),
            ddof=0
        )
    )
    df["modbus_regs_val_min"] = regs_val_stats.map(lambda result: result["min"] if result["min"] is not None else -1.0).astype("float32")
    df["modbus_regs_val_max"] = regs_val_stats.map(lambda result: result["max"] if result["max"] is not None else -1.0).astype("float32")
    df["modbus_regs_val_mean"] = regs_val_stats.map(lambda result: result["mean"] if result["mean"] is not None else -1.0).astype("float32")
    df["modbus_regs_val_std"] = regs_val_stats.map(lambda result: result["std"] if result["std"] is not None else -1.0).astype("float32")

    # ---------- 5) modbus.npy 저장 ----------
    dtype = np.dtype([
        ("modbus_tid_norm", "f4"),
        ("modbus_fc_norm", "f4"),
        ("modbus_addr_norm", "f4"),
        ("modbus_qty_norm", "f4"),
        ("modbus_bc_norm", "f4"),
        ("modbus_regs_count", "f4"),
        ("modbus_regs_addr_min", "f4"),
        ("modbus_regs_addr_max", "f4"),
        ("modbus_regs_addr_range", "f4"),
        ("modbus_regs_val_min", "f4"),
        ("modbus_regs_val_max", "f4"),
        ("modbus_regs_val_mean", "f4"),
        ("modbus_regs_val_std", "f4"),
    ])
    data = np.zeros(len(df), dtype=dtype)

    data["modbus_tid_norm"]        = df["modbus_tid_norm"].to_numpy(dtype=np.float32, copy=False)
    data["modbus_fc_norm"]         = df["modbus_fc_norm"].to_numpy(dtype=np.float32, copy=False)
    data["modbus_addr_norm"]       = df["modbus_addr_norm"].to_numpy(dtype=np.float32, copy=False)
    data["modbus_qty_norm"]        = df["modbus_qty_norm"].to_numpy(dtype=np.float32, copy=False)
    data["modbus_bc_norm"]         = df["modbus_bc_norm"].to_numpy(dtype=np.float32, copy=False)
    data["modbus_regs_count"]      = df["modbus_regs_count"].to_numpy(dtype=np.float32, copy=False)
    data["modbus_regs_addr_min"]   = df["modbus_regs_addr_min"].to_numpy(dtype=np.float32, copy=False)
    data["modbus_regs_addr_max"]   = df["modbus_regs_addr_max"].to_numpy(dtype=np.float32, copy=False)
    data["modbus_regs_addr_range"] = df["modbus_regs_addr_range"].to_numpy(dtype=np.float32, copy=False)
    data["modbus_regs_val_min"]    = df["modbus_regs_val_min"].to_numpy(dtype=np.float32, copy=False)
    data["modbus_regs_val_max"]    = df["modbus_regs_val_max"].to_numpy(dtype=np.float32, copy=False)
    data["modbus_regs_val_mean"]   = df["modbus_regs_val_mean"].to_numpy(dtype=np.float32, copy=False)
    data["modbus_regs_val_std"]    = df["modbus_regs_val_std"].to_numpy(dtype=np.float32, copy=False)

    np.save(out_dir / "modbus.npy", data)

    print("\n===== 앞 5개 전처리 샘플 =====")
    for i in range(min(5, len(data))):
        print({
            "modbus_tid_norm"        : float(data["modbus_tid_norm"][i]),
            "modbus_fc_norm"         : float(data["modbus_fc_norm"][i]),
            "modbus_addr_norm"       : float(data["modbus_addr_norm"][i]),
            "modbus_qty_norm"        : float(data["modbus_qty_norm"][i]),
            "modbus_bc_norm"         : float(data["modbus_bc_norm"][i]),
            "modbus_regs_count"      : float(data["modbus_regs_count"][i]),
            "modbus_regs_addr_min"   : float(data["modbus_regs_addr_min"][i]),
            "modbus_regs_addr_max"   : float(data["modbus_regs_addr_max"][i]),
            "modbus_regs_addr_range" : float(data["modbus_regs_addr_range"][i]),
            "modbus_regs_val_min"    : float(data["modbus_regs_val_min"][i]),
            "modbus_regs_val_max"    : float(data["modbus_regs_val_max"][i]),
            "modbus_regs_val_mean"   : float(data["modbus_regs_val_mean"][i]),
            "modbus_regs_val_std"    : float(data["modbus_regs_val_std"][i]),
        })

# 단일 패킷 전처리 함수 (운영 단계에서 사용)
def preprocess_modbus(records: Dict[str, Any], norm_params: Dict[str, Any]) -> Dict[str, Any]:
    tid = records.get("modbus.tid")
    fc = records.get("modbus.fc")
    addr = records.get("modbus.addr")
    qty = records.get("modbus.qty")
    bc = records.get("modbus.bc")
    regs_addr = records.get("modbus.regs.addr")
    regs_val = records.get("modbus.regs.val")

    tid_float = _to_float(tid)
    fc_float = _to_float(fc)
    addr_float = _to_float(addr)
    qty_float = _to_float(qty)
    bc_float = _to_float(bc)

    tid_min, tid_max = _to_float(norm_params.get("modbus.tid_min")), _to_float(norm_params.get("modbus.tid_max"))
    fc_min, fc_max = _to_float(norm_params.get("modbus.fc_min")), _to_float(norm_params.get("modbus.fc_max"))
    addr_min, addr_max = _to_float(norm_params.get("modbus.addr_min")), _to_float(norm_params.get("modbus.addr_max"))
    qty_min, qty_max = _to_float(norm_params.get("modbus.qty_min")), _to_float(norm_params.get("modbus.qty_max"))
    bc_min, bc_max = _to_float(norm_params.get("modbus.bc_min")), _to_float(norm_params.get("modbus.bc_max"))

    if regs_addr in (None, ""):
        regs_addr = []
    elif not isinstance(regs_addr, (list, tuple)):
        regs_addr = [regs_addr]

    if regs_val in (None, ""):
        regs_val = []
    elif not isinstance(regs_val, (list, tuple)):
        regs_val = [regs_val]

    regs_addr_stats = stats_count_min_max_range(regs_addr)
    regs_val_stats  = stats_min_max_mean_std(regs_val)   
        
    return {
        "modbus_tid_norm"        : float(minmax_cal(tid_float, tid_min, tid_max)),   
        "modbus_fc_norm"         : float(minmax_cal(fc_float, fc_min, fc_max)),   
        "modbus_addr_norm"       : float(minmax_cal(addr_float, addr_min, addr_max)),   
        "modbus_qty_norm"        : float(minmax_cal(qty_float, qty_min, qty_max)),   
        "modbus_bc_norm"         : float(minmax_cal(bc_float, bc_min, bc_max)),   
        "modbus_regs_count"      : regs_addr_stats["count"],
        "modbus_regs_addr_min"   : regs_addr_stats["min"],
        "modbus_regs_addr_max"   : regs_addr_stats["max"],
        "modbus_regs_addr_range" : regs_addr_stats["range"],
        "modbus_regs_val_min"    : regs_val_stats["min"],
        "modbus_regs_val_max"    : regs_val_stats["max"],
        "modbus_regs_val_mean"   : regs_val_stats["mean"],
        "modbus_regs_val_std"    : regs_val_stats["std"],
    }

def transform_preprocess_modbus(packet: Dict[str, Any], param_dir: Path) -> Dict[str, Any]:
    norm_path = param_dir / "modbus_norm_params.json"

    norm_params = file_load("json", str(norm_path)) or {}

    return preprocess_modbus(packet, norm_params)

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
        fit_preprocess_modbus(input_path, out_dir)
    elif args.transform:
        packets = file_load("jsonl", str(input_path)) or []
        for pkt in packets:
            if not isinstance(pkt, dict):
                continue
            feat = transform_preprocess_modbus(pkt, out_dir)
            print(feat)
    else:
        raise ValueError("❌ 반드시 --fit 또는 --transform 중 하나를 선택하세요.")
