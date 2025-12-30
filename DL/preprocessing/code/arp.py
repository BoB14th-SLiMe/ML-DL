#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arp.py  

두 가지 모드 제공:
  --fit        : host_map + norm_params 생성 후 arp.npy 저장
  --transform  : 기존 host_map + norm_params 사용

입력 JSONL에서 사용하는 필드:
  - arp.smac : 소스 MAC
  - arp.tmac : 목적지 MAC
  - arp.sip  : 소스 IP
  - arp.tip  : 목적지 IP
  - arp.op   : arp 패킷의 목적
  
출력 feature (arp.npy, structured numpy):
  - arp_src_host_id   (int32)  : (arp.smac, arp.sip) 조합 → ID, Embedding용
  - arp_tgt_host_id   (int32)  : (arp.tmac, arp.tip) 조합 → ID, Embedding용
  - arp_op_num       (float32) : arp.op min-max 정규화

"""
import json, sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

from min_max_normalize import minmax_cal, minmax_norm_scalar
from change_value_type import _to_float
from ip_mac import get_ip_mac_id_from_vocab

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.file_load import file_load

# fit 
def fit_preprocess_arp(input_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(input_path, lines=True, encoding="utf-8-sig")

    norm_cols = ["arp.op"]

    if df.empty:
        norm_params = {f"{c}_min": -1.0 for c in norm_cols} | {f"{c}_max": -1.0 for c in norm_cols}

        (out_dir / "arp_host_map.json").write_text(json.dumps({}, indent=2, ensure_ascii=False), encoding="utf-8-sig")
        (out_dir / "arp_norm_params.json").write_text(json.dumps(norm_params, indent=2, ensure_ascii=False), encoding="utf-8-sig")

        dtype = np.dtype([
            ("arp_src_host_id", "i4"),
            ("arp_tgt_host_id", "i4"),
            ("arp_op_num", "f4"),
        ])
        data = np.zeros(0, dtype=dtype)
        np.save(out_dir / "arp.npy", data)
        return

    n = len(df)

    # ---------- 1) ip|mac 토큰 (벡터화 + 컬럼 누락 방어 + 빈값 방어) ----------
    sip  = df.get("arp.sip",  pd.Series([pd.NA]*n, index=df.index, dtype="string")).astype("string").str.strip()
    smac = df.get("arp.smac", pd.Series([pd.NA]*n, index=df.index, dtype="string")).astype("string").str.strip()
    dip  = df.get("arp.tip",  pd.Series([pd.NA]*n, index=df.index, dtype="string")).astype("string").str.strip()
    dmac = df.get("arp.tmac", pd.Series([pd.NA]*n, index=df.index, dtype="string")).astype("string").str.strip()

    src_invalid = sip.isna() | smac.isna() | (sip.str.len() == 0) | (smac.str.len() == 0)
    dst_invalid = dip.isna() | dmac.isna() | (dip.str.len() == 0) | (dmac.str.len() == 0)

    src_token = (sip + "|" + smac).mask(src_invalid, pd.NA)
    dst_token = (dip + "|" + dmac).mask(dst_invalid, pd.NA)

    all_tok = pd.concat([src_token, dst_token], ignore_index=True)
    codes, uniques = pd.factorize(all_tok, sort=True)  # NA => -1

    token_to_id: Dict[str, int] = {tok: i + 1 for i, tok in enumerate(uniques.tolist())}

    src_codes = codes[:n]
    dst_codes = codes[n:]
    df["arp_src_host_id"] = np.where(src_codes == -1, -1, src_codes + 1).astype("int32")
    df["arp_tgt_host_id"] = np.where(dst_codes == -1, -1, dst_codes + 1).astype("int32")

    # ---------- 2) min-max 파라미터 산출 (당신 함수 사용) ----------
    norm_params = minmax_norm_scalar(df, norm_cols)

    # ---------- 3) min-max 정규화 적용 (당신 minmax_cal 사용) ----------
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
        df[f"arp_op_num"] = out

    # ---------- 5) vocab + norm_params 저장 ----------
    (out_dir / "arp_host_map.json").write_text(json.dumps(token_to_id, indent=2, ensure_ascii=False), encoding="utf-8-sig")
    (out_dir / "arp_norm_params.json").write_text(json.dumps(norm_params, indent=2, ensure_ascii=False), encoding="utf-8-sig")

    # ---------- 6) arp.npy 저장 ----------
    dtype = np.dtype([
        ("arp_src_host_id", "i4"),
        ("arp_tgt_host_id", "i4"),
        ("arp_op_num", "f4"),
    ])

    data = np.zeros(len(df), dtype=dtype)

    data["arp_src_host_id"]   = df["arp_src_host_id"].to_numpy(dtype=np.int32, copy=False)
    data["arp_tgt_host_id"]   = df["arp_tgt_host_id"].to_numpy(dtype=np.int32, copy=False)
    data["arp_op_num"]        = df["arp_op_num"].to_numpy(dtype=np.float32, copy=False)

    np.save(out_dir / "arp.npy", data)

    print("\n===== 앞 5개 전처리 샘플 =====")
    for i in range(min(5, len(data))):
        print({
            "arp_src_host_id": int(data["arp_src_host_id"][i]),
            "arp_tgt_host_id": int(data["arp_tgt_host_id"][i]),
            "arp_op_num":     float(data["arp_op_num"][i]),
        })

# 단일 패킷 전처리 함수 (운영 단계에서 사용)
def preprocess_arp(records: Dict[str, Any], vocab: Dict[str, int], norm_params: Dict[str, Any]) -> Dict[str, Any]:
    smac = records.get("arp.smac")
    sip  = records.get("arp.sip")
    tmac = records.get("arp.tmac")
    tip  = records.get("arp.tip")
    op   = records.get("arp.op")

    src_id = get_ip_mac_id_from_vocab(vocab, smac, sip)
    tgt_id = get_ip_mac_id_from_vocab(vocab, tmac, tip)

    op_float = _to_float(op)

    op_min, op_max = _to_float(norm_params.get("arp.op_min")), _to_float(norm_params.get("arp.op_max"))

    return {
        "arp_src_host_id" : int(src_id),
        "arp_tgt_host_id" : int(tgt_id),
        "arp_op_num"      : float(minmax_cal(op_float, op_min, op_max)),
    }

def transform_preprocess_arp(packet: Dict[str, Any], param_dir: Path) -> Dict[str, Any]:
    host_map_path = param_dir / "arp_host_map.json"
    norm_path     = param_dir / "arp_norm_params.json"

    vocab = file_load("json", str(host_map_path)) or {}
    norm_params = file_load("json", str(norm_path)) or {}

    return preprocess_arp(packet, vocab, norm_params)

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
        fit_preprocess_arp(input_path, out_dir)
    elif args.transform:
        packets = file_load("jsonl", str(input_path)) or []
        for pkt in packets:
            if not isinstance(pkt, dict):
                continue
            feat = transform_preprocess_arp(pkt, out_dir)
            print(feat)
    else:
        raise ValueError("❌ 반드시 --fit 또는 --transform 중 하나를 선택하세요.")

