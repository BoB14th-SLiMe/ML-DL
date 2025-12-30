#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
common.py  

두 가지 모드 제공:
  --fit        : host_map + norm_params 생성 후 common.npy 저장
  --transform  : 기존 host_map + norm_params 사용

입력 JSONL에서 사용하는 필드:
  - smac     : 소스 MAC
  - dmac     : 목적지 MAC
  - sip      : 소스 IP
  - dip      : 목적지 IP
  - sp       : 소스 포트
  - dp       : 목적지 포트
  - dir      : "request" / "response" / "unknown"
  - protocol : 프로토콜명
  - len      : TCP 이후 payload의 길이

출력 feature (common.npy, structured numpy):
  - src_host_id   (int32)   : (smac, sip) 조합 → ID, Embedding용
  - dst_host_id   (int32)   : (dmac, dip) 조합 → ID, Embedding용
  - sp_norm       (float32) : sp min-max 정규화
  - dp_norm       (float32) : dp min-max 정규화
  - dir_code      (float32) : request=1.0, 그 외=0.0
  - len_norm      (float32) : len min-max 정규화
  - protocol      (float32) : protocol 번호
  - protocol_norm (float32) : protocol min-max정규화

"""
import json, sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict

from min_max_normalize import minmax_cal, minmax_norm_scalar
from change_value_type import _to_float

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.file_load import file_load


# ip, mac merge
def merge_ip_mac(ip:str, mac:str) -> str:
    if ip is None or mac is None:
        return None
    ip_mac = f"{ip}|{mac}"
    return str(ip_mac)

def get_ip_mac_id_from_vocab(vocab: Dict[str, int], mac: Any, ip: Any) -> int:
    token = merge_ip_mac(ip, mac)
    if not token:
        return -1
    return int(vocab.get(token, -1))

# dir mapping
def dir_str_to_float(dir:str) -> float:
    dir_result = -1.0
    if dir == "request":
        dir_result = 1.0
    elif dir == "response":
        dir_result = 0.0
    return dir_result        

# protocol
PROTOCOL_MAP: Dict[str, int] = {
    "s7comm": 0,
    "tcp": 1,
    "xgt_fen": 2,
    "modbus": 3,
    "arp": 4,
    "udp": 5,
    "dns": 6,
}

PROTOCOL_MIN: int = 0
PROTOCOL_MAX: int = max(PROTOCOL_MAP.values())

def protocol_to_code(p: str) -> int:
    if not p:
        return -1
    return int(PROTOCOL_MAP.get(p, -1))

def protocol_to_norm(code: int) -> float:
    if code < 0 or PROTOCOL_MAX <= 0:
        return -1.0
    return float(code / PROTOCOL_MAX)


# fit 
def fit_preprocess_common(input_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(input_path, lines=True, encoding="utf-8-sig")

    norm_cols = ["sp", "dp", "len"]

    if df.empty:
        norm_params = {f"{c}_min": -1.0 for c in norm_cols} | {f"{c}_max": -1.0 for c in norm_cols}

        (out_dir / "common_host_map.json").write_text(json.dumps({}, indent=2, ensure_ascii=False), encoding="utf-8-sig")
        (out_dir / "common_norm_params.json").write_text(json.dumps(norm_params, indent=2, ensure_ascii=False), encoding="utf-8-sig")

        dtype = np.dtype([
            ("src_host_id", "i4"),
            ("dst_host_id", "i4"),
            ("sp_norm",    "f4"),
            ("dp_norm",    "f4"),
            ("dir_code",   "f4"),
            ("len_norm",   "f4"),
            ("protocol", "i4"),
            ("protocol_norm", "f4"),
        ])
        data = np.zeros(0, dtype=dtype)
        np.save(out_dir / "common.npy", data)
        return

    n = len(df)

    # ---------- 1) ip|mac 토큰 (벡터화 + 컬럼 누락 방어 + 빈값 방어) ----------
    sip  = df.get("sip",  pd.Series([pd.NA]*n, index=df.index, dtype="string")).astype("string").str.strip()
    smac = df.get("smac", pd.Series([pd.NA]*n, index=df.index, dtype="string")).astype("string").str.strip()
    dip  = df.get("dip",  pd.Series([pd.NA]*n, index=df.index, dtype="string")).astype("string").str.strip()
    dmac = df.get("dmac", pd.Series([pd.NA]*n, index=df.index, dtype="string")).astype("string").str.strip()

    src_invalid = sip.isna() | smac.isna() | (sip.str.len() == 0) | (smac.str.len() == 0)
    dst_invalid = dip.isna() | dmac.isna() | (dip.str.len() == 0) | (dmac.str.len() == 0)

    src_token = (sip + "|" + smac).mask(src_invalid, pd.NA)
    dst_token = (dip + "|" + dmac).mask(dst_invalid, pd.NA)

    all_tok = pd.concat([src_token, dst_token], ignore_index=True)
    codes, uniques = pd.factorize(all_tok, sort=True)  # NA => -1

    token_to_id: Dict[str, int] = {tok: i + 1 for i, tok in enumerate(uniques.tolist())}

    src_codes = codes[:n]
    dst_codes = codes[n:]
    df["src_host_id"] = np.where(src_codes == -1, -1, src_codes + 1).astype("int32")
    df["dst_host_id"] = np.where(dst_codes == -1, -1, dst_codes + 1).astype("int32")

    # ---------- 2) dir_code (벡터화) ----------
    dir_series = df.get("dir", pd.Series([pd.NA]*n, index=df.index, dtype="string")).astype("string").str.strip()
    df["dir_code"] = np.select(
        [dir_series == "request", dir_series == "response"],
        [1.0, 0.0],
        default=-1.0
    ).astype("float32")

    # ---------- 3) min-max 파라미터 산출 (당신 함수 사용) ----------
    norm_params = minmax_norm_scalar(df, norm_cols)

    # ---------- 4) min-max 정규화 적용 (당신 minmax_cal 사용) ----------
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
        df[f"{col}_norm"] = out

    # ---------- 5) protocol / protocol_norm ----------
    protocol_series = df.get("protocol", pd.Series([pd.NA]*n, index=df.index, dtype="string")).astype("string").str.strip()
    protocol = protocol_series.map(PROTOCOL_MAP).fillna(-1).astype("int32")
    df["protocol"] = protocol

    df["protocol_norm"] = np.where(
        protocol.to_numpy() < 0,
        -1.0,
        (protocol.to_numpy(dtype=np.float32) / float(PROTOCOL_MAX)) if PROTOCOL_MAX > 0 else -1.0
    ).astype("float32")

    # ---------- 6) vocab + norm_params 저장 ----------
    (out_dir / "common_host_map.json").write_text(json.dumps(token_to_id, indent=2, ensure_ascii=False), encoding="utf-8-sig")
    (out_dir / "common_norm_params.json").write_text(json.dumps(norm_params, indent=2, ensure_ascii=False), encoding="utf-8-sig")

    # ---------- 7) common.npy 저장 ----------
    dtype = np.dtype([
        ("src_host_id", "i4"),
        ("dst_host_id", "i4"),
        ("sp_norm",    "f4"),
        ("dp_norm",    "f4"),
        ("dir_code",   "f4"),
        ("len_norm",   "f4"),
        ("protocol", "i4"),
        ("protocol_norm", "f4"),
    ])
    data = np.zeros(len(df), dtype=dtype)

    data["src_host_id"] = df["src_host_id"].to_numpy(dtype=np.int32, copy=False)
    data["dst_host_id"] = df["dst_host_id"].to_numpy(dtype=np.int32, copy=False)
    data["sp_norm"]     = df["sp_norm"].to_numpy(dtype=np.float32, copy=False)
    data["dp_norm"]     = df["dp_norm"].to_numpy(dtype=np.float32, copy=False)
    data["dir_code"]    = df["dir_code"].to_numpy(dtype=np.float32, copy=False)
    data["len_norm"]    = df["len_norm"].to_numpy(dtype=np.float32, copy=False)
    data["protocol"] = df["protocol"].to_numpy(dtype=np.int32, copy=False)
    data["protocol_norm"] = df["protocol_norm"].to_numpy(dtype=np.float32, copy=False)

    np.save(out_dir / "common.npy", data)

    print("\n===== 앞 5개 전처리 샘플 =====")
    for i in range(min(5, len(data))):
        print({
            "src_host_id": int(data["src_host_id"][i]),
            "dst_host_id": int(data["dst_host_id"][i]),
            "sp_norm":     float(data["sp_norm"][i]),
            "dp_norm":     float(data["dp_norm"][i]),
            "dir_code":    float(data["dir_code"][i]),
            "len_norm":    float(data["len_norm"][i]),
            "protocol":    float(data["protocol"][i]),
            "protocol_norm":    float(data["protocol_norm"][i]),
        })

# 단일 패킷 전처리 함수 (운영 단계에서 사용)
def preprocess_common(records: Dict[str, Any], vocab: Dict[str, int], norm_params: Dict[str, Any]) -> Dict[str, Any]:
    smac = records.get("smac")
    sip  = records.get("sip")
    dmac = records.get("dmac")
    dip  = records.get("dip")
    sp   = records.get("sp")
    dp   = records.get("dp")
    dir = records.get("dir")
    len  = records.get("len")
    protocol = records.get("protocol")

    src_id = get_ip_mac_id_from_vocab(vocab, smac, sip)
    dst_id = get_ip_mac_id_from_vocab(vocab, dmac, dip)

    dir_float = dir_str_to_float(dir)

    sp_float = _to_float(sp)
    dp_float = _to_float(dp)
    len_float = _to_float(len)

    sp_min, sp_max = _to_float(norm_params.get("sp_min")), _to_float(norm_params.get("sp_max"))
    dp_min, dp_max = _to_float(norm_params.get("dp_min")), _to_float(norm_params.get("dp_max"))
    ln_min, ln_max = _to_float(norm_params.get("len_min")), _to_float(norm_params.get("len_max"))

    protocol = records.get("protocol")
    protocol = protocol_to_code(protocol)
    protocol_norm = protocol_to_norm(protocol)

    return {
        "src_host_id": int(src_id),
        "dst_host_id": int(dst_id),
        "sp_norm": float(minmax_cal(sp_float, sp_min, sp_max)),
        "dp_norm": float(minmax_cal(dp_float, dp_min, dp_max)),
        "dir_code": float(dir_float),
        "len_norm": float(minmax_cal(len_float, ln_min, ln_max)),
        "protocol": int(protocol),
        "protocol_norm": float(protocol_norm),       
    }

def transform_preprocess_common(packet: Dict[str, Any], param_dir: Path) -> Dict[str, Any]:
    host_map_path = param_dir / "common_host_map.json"
    norm_path     = param_dir / "common_norm_params.json"

    vocab = file_load("json", str(host_map_path)) or {}
    norm_params = file_load("json", str(norm_path)) or {}

    return preprocess_common(packet, vocab, norm_params)

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
        fit_preprocess_common(input_path, out_dir)
    elif args.transform:
        packets = file_load("jsonl", str(input_path)) or []
        for pkt in packets:
            if not isinstance(pkt, dict):
                continue
            feat = transform_preprocess_common(pkt, out_dir)
            print(feat)
    else:
        raise ValueError("❌ 반드시 --fit 또는 --transform 중 하나를 선택하세요.")
