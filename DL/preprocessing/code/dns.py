#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dns.py  

두 가지 모드 제공:
  --fit        : norm_params 생성 후 dns.npy 저장
  --transform  : 기존 norm_params 사용

입력 JSONL에서 사용하는 필드:
  - dns.tid : dns 구분자 (transaction id)
  - dns.fl  : dns flags
  - dns.qc  : dns question section에 들어있는 질문 개수 
  - dns.ac  : dns answer section의 리소스 레코드 개수

출력 feature (common.npy, structured numpy):
  - dns_tid_norm  (float32) : dns.tid min-max 정규화 (str -> float 변환)
  - dns_fl_norm (float32) : dns.fl min-max 정규화 (str -> float 변환)
  - dns_qc_norm   (float32) : dns.qc min-max 정규화
  - dns_ac_norm   (float32) : dns.ac min-max 정규화

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
def fit_preprocess_dns(input_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(input_path, lines=True, encoding="utf-8-sig")

    hex_cols = ["dns.tid", "dns.fl"]
    norm_cols = ["dns.tid", "dns.fl", "dns.qc", "dns.ac"]

    if df.empty:
        norm_params = {f"{c}_min": -1.0 for c in norm_cols} | {f"{c}_max": -1.0 for c in norm_cols}

        (out_dir / "dns_norm_params.json").write_text(json.dumps(norm_params, indent=2, ensure_ascii=False), encoding="utf-8-sig")

        dtype = np.dtype([
            ("dns_tid_norm", "f4"),
            ("dns_fl_norm",  "f4"),
            ("dns_qc_norm",  "f4"),
            ("dns_ac_norm",  "f4"),
        ])
        data = np.zeros(0, dtype=dtype)
        np.save(out_dir / "dns.npy", data)
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
    (out_dir / "dns_norm_params.json").write_text(json.dumps(norm_params, indent=2, ensure_ascii=False), encoding="utf-8-sig")

    # ---------- 5) common.npy 저장 ----------
    dtype = np.dtype([
        ("dns_tid_norm", "f4"),
        ("dns_fl_norm",  "f4"),
        ("dns_qc_norm",  "f4"),
        ("dns_ac_norm",  "f4"),
    ])
    data = np.zeros(len(df), dtype=dtype)

    data["dns_tid_norm"] = df["dns_tid_norm"].to_numpy(dtype=np.float32, copy=False)
    data["dns_fl_norm"]  = df["dns_fl_norm"].to_numpy(dtype=np.float32, copy=False)
    data["dns_qc_norm"]  = df["dns_qc_norm"].to_numpy(dtype=np.float32, copy=False)
    data["dns_ac_norm"]  = df["dns_ac_norm"].to_numpy(dtype=np.float32, copy=False)

    np.save(out_dir / "dns.npy", data)

    print("\n===== 앞 5개 전처리 샘플 =====")
    for i in range(min(5, len(data))):
        print({
            "dns_tid_norm"  : float(data["dns_tid_norm"][i]),
            "dns_fl_norm" : float(data["dns_fl_norm"][i]),
            "dns_qc_norm"   : float(data["dns_qc_norm"][i]),
            "dns_ac_norm"   : float(data["dns_ac_norm"][i]),
        })

# 단일 패킷 전처리 함수 (운영 단계에서 사용)
def preprocess_dns(records: Dict[str, Any], norm_params: Dict[str, Any]) -> Dict[str, Any]:
    tid = records.get("dns.tid")
    fl = records.get("dns.fl")
    qc = records.get("dns.qc")
    ac = records.get("dns.ac")

    tid_float = _hex_to_float(tid)
    fl_float = _hex_to_float(fl)

    qc_float = _to_float(qc)
    ac_float = _to_float(ac)

    tid_min, tid_max = _to_float(norm_params.get("dns.tid_min")), _to_float(norm_params.get("dns.tid_max"))
    fl_min, fl_max   = _to_float(norm_params.get("dns.fl_min")), _to_float(norm_params.get("dns.fl_max"))
    qc_min, qc_max   = _to_float(norm_params.get("dns.qc_min")), _to_float(norm_params.get("dns.qc_max"))
    ac_min, ac_max   = _to_float(norm_params.get("dns.ac_min")), _to_float(norm_params.get("dns.ac_max"))

    return {
        "dns_tid_norm" : float(minmax_cal(tid_float, tid_min, tid_max)),   
        "dns_fl_norm"  : float(minmax_cal(fl_float, fl_min, fl_max)),   
        "dns_qc_norm"  : float(minmax_cal(qc_float, qc_min, qc_max)),   
        "dns_ac_norm"  : float(minmax_cal(ac_float, ac_min, ac_max)),   
    }

def transform_preprocess_dns(packet: Dict[str, Any], param_dir: Path) -> Dict[str, Any]:
    norm_path     = param_dir / "dns_norm_params.json"

    norm_params = file_load("json", str(norm_path)) or {}

    return preprocess_dns(packet, norm_params)

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
        fit_preprocess_dns(input_path, out_dir)
    elif args.transform:
        packets = file_load("jsonl", str(input_path)) or []
        for pkt in packets:
            if not isinstance(pkt, dict):
                continue
            feat = transform_preprocess_dns(pkt, out_dir)
            print(feat)
    else:
        raise ValueError("❌ 반드시 --fit 또는 --transform 중 하나를 선택하세요.")
