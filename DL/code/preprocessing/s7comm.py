#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s7comm.py  

두 가지 모드 제공:
  --fit       : norm_params 생성 후 s7comm.npy 저장
  --transform : 기존 norm_params 사용

입력 JSONL에서 사용하는 필드:
  - s7comm.ros  : s7comm ROS Control field
  - s7comm.fn   : s7comm 명령어 코드
  - s7comm.ic   : s7comm 아이템 개수
  - s7comm.db   : s7comm DB 번호
  - s7comm.addr : s7comm 주소값

출력 feature (s7comm.npy, structured numpy):
  - s7comm_ros_norm  : s7comm.ros min-max 정규화
  - s7comm_fn        : s7comm.fn 원본
  - s7comm_fn_norm   : s7comm.fn min-max 정규화
  - s7comm_ic_norm   : s7comm.ic min-max 정규화
  - s7comm_db_norm   : s7comm.db min-max 정규화
  - s7comm_addr_norm : s7comm.addr min-max 정규화

"""
import json, sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict

from min_max_normalize import minmax_cal, minmax_norm_scalar
from change_value_type import _to_float, _hex_to_float

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.file_load import file_load

# fit 
def fit_preprocess_s7comm(input_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(input_path, lines=True, encoding="utf-8-sig")

    hex_cols = ["s7comm.fn"]
    norm_cols = ["s7comm.ros", "s7comm.fn", "s7comm.ic", "s7comm.db", "s7comm.addr"]

    if df.empty:
        norm_params = {f"{c}_min": -1.0 for c in norm_cols} | {f"{c}_max": -1.0 for c in norm_cols}

        (out_dir / "s7comm_norm_params.json").write_text(json.dumps(norm_params, indent=2, ensure_ascii=False), encoding="utf-8-sig")

        dtype = np.dtype([
            ("s7comm_ros_norm", "f4"),
            ("s7comm_fn",        "f4"),
            ("s7comm_fn_norm",   "f4"),
            ("s7comm_ic_norm",   "f4"),
            ("s7comm_db_norm",   "f4"),
            ("s7comm_addr_norm", "f4"),
        ])
        data = np.zeros(0, dtype=dtype)
        np.save(out_dir / "s7comm.npy", data)
        return

    n = len(df)

    # ---------- 1) hex 데이터 float 변환 ----------
    for col in hex_cols:
        if col in df.columns:
            df[col] = df[col].map(_hex_to_float)
    
    # ---------- 2) min-max 파라미터 산출 ----------
    norm_params = minmax_norm_scalar(df, norm_cols)
    print (norm_params)

    # ---------- 3) min-max 정규화 적용 ----------
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

    # ---------- 4) vocab + norm_params 저장 ----------
    (out_dir / "s7comm_norm_params.json").write_text(json.dumps(norm_params, indent=2, ensure_ascii=False), encoding="utf-8-sig")

    # ---------- 5) s7comm.npy 저장 ----------
    dtype = np.dtype([
        ("s7comm_ros_norm",  "f4"),
        ("s7comm_fn",        "f4"),
        ("s7comm_fn_norm",   "f4"),
        ("s7comm_ic_norm",   "f4"),
        ("s7comm_db_norm",   "f4"),
        ("s7comm_addr_norm", "f4"),
    ])
    data = np.zeros(len(df), dtype=dtype)

    data["s7comm_ros_norm"]  = df["s7comm_ros_norm"].to_numpy(dtype=np.float32, copy=False)
    data["s7comm_fn"]        = df["s7comm.fn"].to_numpy(dtype=np.float32, copy=False)
    data["s7comm_fn_norm"]   = df["s7comm_fn_norm"].to_numpy(dtype=np.float32, copy=False)
    data["s7comm_ic_norm"]   = df["s7comm_ic_norm"].to_numpy(dtype=np.float32, copy=False)
    data["s7comm_db_norm"]   = df["s7comm_db_norm"].to_numpy(dtype=np.float32, copy=False)
    data["s7comm_addr_norm"] = df["s7comm_addr_norm"].to_numpy(dtype=np.float32, copy=False)

    np.save(out_dir / "s7comm.npy", data)

    print("\n===== 앞 5개 전처리 샘플 =====")
    for i in range(min(5, len(data))):
        print({
            "s7comm_ros_norm"  : float(data["s7comm_ros_norm"][i]),
            "s7comm_fn"        : float(data["s7comm_fn"][i]),
            "s7comm_fn_norm"   : float(data["s7comm_fn_norm"][i]),
            "s7comm_ic_norm"   : float(data["s7comm_ic_norm"][i]),
            "s7comm_db_norm"   : float(data["s7comm_db_norm"][i]),
            "s7comm_addr_norm" : float(data["s7comm_addr_norm"][i]),
        })

# 단일 패킷 전처리 함수 (운영 단계에서 사용)
def preprocess_s7comm(records: Dict[str, Any], norm_params: Dict[str, Any]) -> Dict[str, Any]:
    ros = records.get("s7comm.ros")
    fn = records.get("s7comm.fn")
    ic = records.get("s7comm.ic")
    db = records.get("s7comm.db")
    addr = records.get("s7comm.addr")

    fn_float = _hex_to_float(fn)

    ros_float = _to_float(ros)
    ic_float = _to_float(ic)
    db_float = _to_float(db)
    addr_float = _to_float(addr)

    ros_min, ros_max   = _to_float(norm_params.get("s7comm.ros_min")), _to_float(norm_params.get("s7comm.ros_max"))
    fn_min, fn_max     = _to_float(norm_params.get("s7comm.fn_min")), _to_float(norm_params.get("s7comm.fn_max"))
    ic_min, ic_max     = _to_float(norm_params.get("s7comm.ic_min")), _to_float(norm_params.get("s7comm.ic_max"))
    db_min, db_max     = _to_float(norm_params.get("s7comm.db_min")), _to_float(norm_params.get("s7comm.db_max"))
    addr_min, addr_max = _to_float(norm_params.get("s7comm.addr_min")), _to_float(norm_params.get("s7comm.addr_max"))

    return {
        "s7comm_ros_norm"  : float(minmax_cal(ros_float, ros_min, ros_max)),   
        "s7comm_fn"        : fn_float,   
        "s7comm_fn_norm"   : float(minmax_cal(fn_float, fn_min, fn_max)),   
        "s7comm_ic_norm"   : float(minmax_cal(ic_float, ic_min, ic_max)),   
        "s7comm_db_norm"   : float(minmax_cal(db_float, db_min, db_max)),   
        "s7comm_addr_norm" : float(minmax_cal(addr_float, addr_min, addr_max)),   
    }

def transform_preprocess_s7comm(packet: Dict[str, Any], param_dir: Path) -> Dict[str, Any]:
    norm_path = param_dir / "s7comm_norm_params.json"

    norm_params = file_load("json", str(norm_path)) or {}

    return preprocess_s7comm(packet, norm_params)

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
        fit_preprocess_s7comm(input_path, out_dir)
    elif args.transform:
        packets = file_load("jsonl", str(input_path)) or []
        for pkt in packets:
            if not isinstance(pkt, dict):
                continue
            feat = transform_preprocess_s7comm(pkt, out_dir)
            print(feat)
    else:
        raise ValueError("❌ 반드시 --fit 또는 --transform 중 하나를 선택하세요.")
