#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocess_xgt_fen_embed.py
A버전: xgt_fen 전용 embedding/feature 전처리 + 간단 정규화

두 모드 제공:
  --fit        : xgt_var_vocab + 정규화 파라미터 생성 후 xgt_fen.npy 저장
  --transform  : 기존 xgt_var_vocab + 정규화 파라미터 사용

입력 JSONL에서 사용하는 필드:
  - protocol == "xgt_fen"
  - xgt_fen.vars      : list[str] 또는 "R17,R20" 같은 str
  - xgt_fen.source    : int 또는 "0x11" 같은 hex 문자열
  - xgt_fen.fenetpos  : int 또는 "0x01" 같은 hex 문자열 (상위 4bit = base, 하위 4bit = slot)
  - xgt_fen.cmd       : int 또는 "0x0054" 같은 hex 문자열
  - xgt_fen.dtype     : int 또는 "0x0014" (또는 xgt_fen.dype)
  - xgt_fen.blkcnt    : int
  - xgt_fen.datasize  : int
  - xgt_fen.errstat   : int 또는 "0x0000" 같은 hex 문자열
  - xgt_fen.data      : list[str] 또는 str (hex string)

출력:
  - xgt_fen.npy (structured numpy 배열)
  - xgt_var_vocab.json (변수 이름 → ID)
  - xgt_fen_norm_params.json (정규화용 min/max)

xgt_fen.npy dtype (각 필드는 이미 아래 규칙대로 스케일링됨):
  - xgt_var_id         (int32)   ← vars[0] → ID, Embedding용 (정규화 X)
  - xgt_var_cnt        (float32) ← Min-Max 정규화
  - xgt_source         (float32) ← Min-Max 정규화
  - xgt_fenet_base     (float32) ← Min-Max 정규화
  - xgt_fenet_slot     (float32) ← Min-Max 정규화
  - xgt_cmd            (float32) ← Min-Max 정규화
  - xgt_dtype          (float32) ← Min-Max 정규화
  - xgt_blkcnt         (float32) ← Min-Max 정규화
  - xgt_err_flag       (float32) (0.0 / 1.0, 정규화 X)
  - xgt_err_code       (float32) ← Min-Max 정규화
  - xgt_datasize       (float32) ← Min-Max 정규화
  - xgt_data_missing   (float32) (0.0 / 1.0, datasize>0 & data 없음이면 1.0)
  - xgt_data_len_chars (float32) ← Min-Max 정규화
  - xgt_data_num_spaces(float32) ← Min-Max 정규화
  - xgt_data_is_hex    (float32) (0.0 / 1.0, 정규화 X)
  - xgt_data_n_bytes   (float32) ← Min-Max 정규화
  - xgt_data_zero_ratio(float32) (0.0 ~ 1.0, 정규화 X)
  - xgt_data_first_byte(float32) (0~1, 원래 0~255를 /255.0)
  - xgt_data_last_byte (float32) (0~1, 원래 0~255를 /255.0)
  - xgt_data_mean_byte (float32) (0~1, 원래 0~255를 /255.0)
  - xgt_data_bucket    (float32) (hash bucket, 정규화 X)

실시간 / 단일 패킷 처리:
  - xgt_var_vocab.json, xgt_fen_norm_params.json 로드 후
    preprocess_xgt_fen_with_norm(obj, var_map, norm_params) 호출
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Any, Dict, List
import re

# ---------------------------------------------
# Feature 이름 (설명용, 코드 내부에서는 dtype이 source of truth)
# ---------------------------------------------
FEATURE_NAMES = [
    "xgt_var_id",
    "xgt_var_cnt",
    "xgt_source",
    "xgt_fenet_base",
    "xgt_fenet_slot",
    "xgt_cmd",
    "xgt_dtype",
    "xgt_blkcnt",
    "xgt_err_flag",
    "xgt_err_code",
    "xgt_datasize",
    # 👇 새로 추가되는 8개
    "xgt_addr_count",
    "xgt_addr_min",
    "xgt_addr_max",
    "xgt_addr_range",
    "xgt_word_min",
    "xgt_word_max",
    "xgt_word_mean",
    "xgt_word_std",
    # 기존
    "xgt_data_missing",
    "xgt_data_len_chars",
    "xgt_data_num_spaces",
    "xgt_data_is_hex",
    "xgt_data_n_bytes",
    "xgt_data_zero_ratio",
    "xgt_data_first_byte",
    "xgt_data_last_byte",
    "xgt_data_mean_byte",
    "xgt_data_bucket",
]

NORM_FIELDS = [
    "xgt_var_cnt",
    "xgt_source",
    "xgt_fenet_base",
    "xgt_fenet_slot",
    "xgt_cmd",
    "xgt_dtype",
    "xgt_blkcnt",
    "xgt_err_code",
    "xgt_datasize",
    "xgt_data_len_chars",
    "xgt_data_num_spaces",
    "xgt_data_n_bytes",
    # 👇 translated_addr / word_value 통계도 정규화 대상 추가
    "xgt_addr_count",
    "xgt_addr_min",
    "xgt_addr_max",
    "xgt_addr_range",
    "xgt_word_min",
    "xgt_word_max",
    "xgt_word_mean",
    "xgt_word_std",
]


NORM_PARAMS_FILE = "xgt_fen_norm_params.json"


# ---------------------------------------------
# Var ID 생성기 (vars[0] → ID)
# ---------------------------------------------
def get_var_id_factory(var_map: Dict[str, int]):
    """
    var_map: {"R17": 1, "R20": 2, ...}
    """
    next_id = max(var_map.values()) + 1 if var_map else 1

    def get_var_id(var_name: str) -> int:
        nonlocal next_id
        if not var_name:
            return 0  # UNK
        if var_name not in var_map:
            var_map[var_name] = next_id
            next_id += 1
        return var_map[var_name]

    return get_var_id


# ---------------------------------------------
# 안전한 int 변환 (10진수 + 16진수 "0x.." 모두 지원)
# ---------------------------------------------
def to_int(value: Any, default: int = 0) -> int:
    """
    "10", 10, "0x10" 같은 값들을 모두 int로 변환.
    - "0x.." 형태면 16진수로 인식
    - 그 외는 10진수 시도
    """
    if value is None:
        return default

    s = str(value).strip()
    if not s:
        return default

    try:
        # "0x10" 같이 base 자동 인식
        return int(s, 0)
    except ValueError:
        try:
            return int(s)
        except ValueError:
            return default

# ---------------------------------------------
# translated_addr / word_value 리스트 파싱 + 통계
# ---------------------------------------------
# ---------------------------------------------
# translated_addr / word_value 리스트 파싱 + 통계
# ---------------------------------------------
def parse_xgt_translated_addr_list(value: Any) -> List[int]:
    """
    xgt_fen.translated_addr: ["M1","M2", ...] 같은 값들을 숫자 리스트로 변환.
    - "M1" -> 1, "M2" -> 2 처럼 '숫자 부분'만 추출해서 사용
    """
    result: List[int] = []

    def _handle_one(x: Any):
        s = str(x).strip()
        if not s:
            return
        # 문자열 안에서 숫자 부분만 추출
        m = re.search(r"(\d+)", s)
        if m:
            try:
                result.append(int(m.group(1)))
            except ValueError:
                pass

    if value is None:
        return result

    if isinstance(value, list):
        for v in value:
            _handle_one(v)
        return result

    s = str(value).strip()
    # JSON 문자열 형태인 경우 예: '["M1","M2"]'
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            return parse_xgt_translated_addr_list(arr)
        except Exception:
            _handle_one(s)
            return result

    _handle_one(s)
    return result


def parse_xgt_word_value_list(value: Any) -> List[float]:
    """
    xgt_fen.word_value: ["0","1","2"] or [0,1,2] or "0" 등을 float 리스트로 변환.
    """
    result: List[float] = []

    def _handle_one(x: Any):
        try:
            v = float(to_int(x))
            result.append(v)
        except Exception:
            pass

    if value is None:
        return result

    if isinstance(value, list):
        for v in value:
            _handle_one(v)
        return result

    s = str(value).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            return parse_xgt_word_value_list(arr)
        except Exception:
            _handle_one(s)
            return result

    _handle_one(s)
    return result


def compute_xgt_addr_stats(addrs: List[int]) -> Dict[str, float]:
    """
    translated_addr(숫자화된 것) 리스트에 대해 count/min/max/range 계산.
    """
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


def compute_xgt_word_stats(vals: List[float]) -> Dict[str, float]:
    """
    word_value 리스트에 대해 min/max/mean/std 계산.
    """
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
    std = float(var ** 0.5)
    return {
        "min": vmin,
        "max": vmax,
        "mean": mean,
        "std": std,
    }


# ---------------------------------------------
# xgt_fen.data 요약 피처
# ---------------------------------------------
def extract_xgt_data_features(data: Any) -> Dict[str, float]:
    """
    xgt_fen.data (string 또는 string 리스트) -> 여러 개 numeric feature로 변환

    예시:
        "xgt_fen.data": [
            "0000",
            "000000000000000000000000",
            "05001e00f50000001c002700",
            "3e01"
        ]

    각 원소는 hex string 이라고 가정하고,
    공백 제거 후 2글자씩 잘라서 바이트 배열을 만든다.
    """

    # 1) data를 문자열 리스트로 통일
    if data is None:
        strings: List[str] = []
    elif isinstance(data, list):
        strings = [str(x).strip() for x in data if str(x).strip()]
    else:
        s = str(data).strip()
        strings = [s] if s else []

    # 전체 문자열 하나로 합치기 (공백으로 join)
    joined = " ".join(strings)
    s = joined
    s_no_space = s.replace(" ", "")

    # 공통 문자열 피처
    length_chars = len(s)
    num_spaces = s.count(" ")

    # hex 여부 판단
    hex_chars = sum(ch in "0123456789abcdefABCDEF" for ch in s_no_space)
    non_hex_chars = len(s_no_space) - hex_chars
    is_hex = int(len(s_no_space) > 0 and non_hex_chars == 0)

    # 2) hex로 해석해서 바이트 나열 만들기
    bytes_values: List[int] = []
    if is_hex:
        for elem in strings:
            hex_str = elem.replace(" ", "")
            if len(hex_str) < 2:
                continue
            # 2글자씩 잘라서 바이트로
            for i in range(0, len(hex_str) - 1, 2):
                chunk = hex_str[i:i+2]
                try:
                    v = int(chunk, 16)
                    bytes_values.append(v)
                except ValueError:
                    continue

    n_bytes = len(bytes_values)
    if n_bytes > 0:
        zero_count = sum(1 for v in bytes_values if v == 0)
        zero_ratio = zero_count / float(n_bytes)
        first_byte = bytes_values[0]
        last_byte = bytes_values[-1]
        mean_byte = float(sum(bytes_values) / float(n_bytes))
    else:
        zero_ratio = 0.0
        first_byte = 0
        last_byte = 0
        mean_byte = 0.0

    # 동일 문자열 패턴용 해시 버킷 (embedding으로 쓰고 싶으면 이 값 사용)
    bucket = hash(s) % 1024 if s else 0

    return {
        "xgt_data_len_chars": float(length_chars),
        "xgt_data_num_spaces": float(num_spaces),
        "xgt_data_is_hex": float(is_hex),
        "xgt_data_n_bytes": float(n_bytes),
        "xgt_data_zero_ratio": float(zero_ratio),
        "xgt_data_first_byte": float(first_byte),
        "xgt_data_last_byte": float(last_byte),
        "xgt_data_mean_byte": float(mean_byte),
        "xgt_data_bucket": float(bucket),
    }


# ---------------------------------------------
# 한 레코드(xgt_fen) 전처리 (정규화 전 RAW feature)
# ---------------------------------------------
def preprocess_xgt_record(obj: Dict[str, Any], get_var_id) -> Dict[str, float]:
    """
    protocol == "xgt_fen" 인 레코드를 RAW feature dict로 변환
    (정규화는 나중 단계에서 수행)
    """

    feat: Dict[str, float] = {}

    # 1) vars → var_id / var_cnt
    vars_field = obj.get("xgt_fen.vars")

    var_names: List[str] = []
    if isinstance(vars_field, list):
        var_names = [str(v).strip() for v in vars_field if str(v).strip()]
    elif isinstance(vars_field, str):
        # "R17,R20" 같은 경우
        for part in vars_field.split(","):
            p = part.strip()
            if p:
                var_names.append(p)

    if var_names:
        first_var = var_names[0]
        var_cnt = len(var_names)
    else:
        first_var = ""
        var_cnt = 0

    var_id = get_var_id(first_var) if first_var else 0
    feat["xgt_var_id"] = int(var_id)      # 저장도 int, dtype도 int32
    feat["xgt_var_cnt"] = float(var_cnt)

    # 2) 헤더/명령 필드 + errstat 처리
    source = to_int(obj.get("xgt_fen.source"))
    fenetpos = to_int(obj.get("xgt_fen.fenetpos"))
    cmd = to_int(obj.get("xgt_fen.cmd"))
    dtype = to_int(obj.get("xgt_fen.dtype") or obj.get("xgt_fen.dype"))
    blkcnt = to_int(obj.get("xgt_fen.blkcnt"))
    datasize = to_int(obj.get("xgt_fen.datasize"))

    # errstat → 에러 코드 / 플래그
    errstat_raw = obj.get("xgt_fen.errstat")
    err_code = to_int(errstat_raw)
    err_flag = 1.0 if err_code != 0 else 0.0

    base = (fenetpos >> 4) & 0x0F
    slot = fenetpos & 0x0F

    feat["xgt_source"] = float(source)
    feat["xgt_fenet_base"] = float(base)
    feat["xgt_fenet_slot"] = float(slot)
    feat["xgt_cmd"] = float(cmd)
    feat["xgt_dtype"] = float(dtype)
    feat["xgt_blkcnt"] = float(blkcnt)
    feat["xgt_datasize"] = float(datasize)

    feat["xgt_err_code"] = float(err_code)
    feat["xgt_err_flag"] = float(err_flag)

    # 3) data 요약
    data_field = obj.get("xgt_fen.data")
    data_feats = extract_xgt_data_features(data_field)
    feat.update(data_feats)

    # 4) 데이터 없음 플래그 추가
    length_chars = data_feats.get("xgt_data_len_chars", 0.0)
    xgt_data_missing = 1.0 if (datasize > 0 and length_chars == 0.0) else 0.0
    feat["xgt_data_missing"] = float(xgt_data_missing)

    # 5) translated_addr / word_value 통계 피처 추가 (M1, M2, ... 처리)
    translated_addr_raw = obj.get("xgt_fen.translated_addr")
    word_value_raw      = obj.get("xgt_fen.word_value")

    addr_list = parse_xgt_translated_addr_list(translated_addr_raw)
    word_list = parse_xgt_word_value_list(word_value_raw)

    addr_stats = compute_xgt_addr_stats(addr_list)
    word_stats = compute_xgt_word_stats(word_list)

    feat["xgt_addr_count"] = addr_stats["count"]
    feat["xgt_addr_min"]   = addr_stats["min"]
    feat["xgt_addr_max"]   = addr_stats["max"]
    feat["xgt_addr_range"] = addr_stats["range"]

    feat["xgt_word_min"]   = word_stats["min"]
    feat["xgt_word_max"]   = word_stats["max"]
    feat["xgt_word_mean"]  = word_stats["mean"]
    feat["xgt_word_std"]   = word_stats["std"]

    return feat


# ---------------------------------------------
# Min-Max 정규화 함수
# ---------------------------------------------
def minmax_norm(val: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.0
    return (val - vmin) / (vmax - vmin + 1e-9)


# ---------------------------------------------
# RAW feature → 정규화 feature로 변환 (공통 로직)
# ---------------------------------------------
def apply_norm_to_xgt_feat(raw_feat: Dict[str, float],
                           norm_params: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    feat: Dict[str, float] = {}
    feat["xgt_var_id"] = int(raw_feat.get("xgt_var_id", 0))

    for f in NORM_FIELDS:
        raw_v = float(raw_feat.get(f, 0.0))
        p = norm_params.get(f, {})
        vmin = float(p.get("min", 0.0))
        vmax = float(p.get("max", 1.0))

        if f == "xgt_cmd":
            # ✅ sentinel 로직 추가
            if raw_v < vmin or raw_v > vmax:
                feat[f] = -2.0
            elif vmax > vmin:
                feat[f] = (raw_v - vmin) / (vmax - vmin + 1e-9)
            else:
                feat[f] = 0.0
        else:
            feat[f] = float(minmax_norm(raw_v, vmin, vmax))

    # 나머지는 그대로
    feat["xgt_err_flag"]       = float(raw_feat.get("xgt_err_flag", 0.0))
    feat["xgt_data_missing"]   = float(raw_feat.get("xgt_data_missing", 0.0))
    feat["xgt_data_is_hex"]    = float(raw_feat.get("xgt_data_is_hex", 0.0))
    feat["xgt_data_zero_ratio"]= float(raw_feat.get("xgt_data_zero_ratio", 0.0))
    feat["xgt_data_bucket"]    = float(raw_feat.get("xgt_data_bucket", 0.0))

    fb = float(raw_feat.get("xgt_data_first_byte", 0.0))
    lb = float(raw_feat.get("xgt_data_last_byte", 0.0))
    mb = float(raw_feat.get("xgt_data_mean_byte", 0.0))
    feat["xgt_data_first_byte"] = fb / 255.0
    feat["xgt_data_last_byte"]  = lb / 255.0
    feat["xgt_data_mean_byte"]  = mb / 255.0

    return feat


# ---------------------------------------------
# 단일 패킷 + 정규화까지 처리 (실시간/운영 용)
# ---------------------------------------------
def preprocess_xgt_fen_with_norm(
    obj: Dict[str, Any],
    var_map: Dict[str, int],
    norm_params: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    단일 xgt_fen 패킷 obj에 대해
    - xgt_var_id (int)
    - 나머지 numeric feature (정규화 포함)
    를 모두 담은 dict 반환.

    사용 예:
        var_map = json.loads(open("xgt_var_vocab.json","r",encoding="utf-8").read())
        norm_params = json.loads(open("xgt_fen_norm_params.json","r",encoding="utf-8").read())

        feat = preprocess_xgt_fen_with_norm(obj, var_map, norm_params)
    """
    get_var_id = get_var_id_factory(var_map)
    raw_feat = preprocess_xgt_record(obj, get_var_id)
    norm_feat = apply_norm_to_xgt_feat(raw_feat, norm_params)
    return norm_feat


# ---------------------------------------------
# FIT
# ---------------------------------------------
def fit_preprocess(input_path: Path, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    var_map: Dict[str, int] = {}
    get_var_id = get_var_id_factory(var_map)

    rows_raw: List[Dict[str, float]] = []

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("protocol") != "xgt_fen":
                continue

            feat = preprocess_xgt_record(obj, get_var_id)
            rows_raw.append(feat)

    # vocab 저장
    (out_dir / "xgt_var_vocab.json").write_text(
        json.dumps(var_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("✅ FIT 완료")
    print(f"- xgt_var_vocab.json 저장: {out_dir/'xgt_var_vocab.json'}")

    # -----------------------------
    # 1) 정규화 파라미터 계산 (Min/Max)
    # -----------------------------
    norm_params: Dict[str, Dict[str, float]] = {
        f: {"min": None, "max": None} for f in NORM_FIELDS
    }

    for feat in rows_raw:
        for f in NORM_FIELDS:
            v = float(feat.get(f, 0.0))
            if norm_params[f]["min"] is None or v < norm_params[f]["min"]:
                norm_params[f]["min"] = v
            if norm_params[f]["max"] is None or v > norm_params[f]["max"]:
                norm_params[f]["max"] = v

    # 빈 경우 방어코드
    for f in NORM_FIELDS:
        if norm_params[f]["min"] is None:
            norm_params[f]["min"] = 0.0
            norm_params[f]["max"] = 1.0

    # JSON 저장
    (out_dir / NORM_PARAMS_FILE).write_text(
        json.dumps(norm_params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"- {NORM_PARAMS_FILE} 저장: {out_dir / NORM_PARAMS_FILE}")

    # -----------------------------
    # 2) numpy 구조화 배열 생성 (정규화 적용)
    # -----------------------------    
    dtype = np.dtype([
        ("xgt_var_id", "i4"),   # Embedding용 ID (int32)
        ("xgt_var_cnt", "f4"),
        ("xgt_source", "f4"),
        ("xgt_fenet_base", "f4"),
        ("xgt_fenet_slot", "f4"),
        ("xgt_cmd", "f4"),
        ("xgt_dtype", "f4"),
        ("xgt_blkcnt", "f4"),
        ("xgt_err_flag", "f4"),
        ("xgt_err_code", "f4"),
        ("xgt_datasize", "f4"),
        # 👇 여기 추가
        ("xgt_addr_count", "f4"),
        ("xgt_addr_min", "f4"),
        ("xgt_addr_max", "f4"),
        ("xgt_addr_range", "f4"),
        ("xgt_word_min", "f4"),
        ("xgt_word_max", "f4"),
        ("xgt_word_mean", "f4"),
        ("xgt_word_std", "f4"),
        # 기존
        ("xgt_data_missing", "f4"),
        ("xgt_data_len_chars", "f4"),
        ("xgt_data_num_spaces", "f4"),
        ("xgt_data_is_hex", "f4"),
        ("xgt_data_n_bytes", "f4"),
        ("xgt_data_zero_ratio", "f4"),
        ("xgt_data_first_byte", "f4"),
        ("xgt_data_last_byte", "f4"),
        ("xgt_data_mean_byte", "f4"),
        ("xgt_data_bucket", "f4"),
    ])



    data = np.zeros(len(rows_raw), dtype=dtype)

    for idx, raw_feat in enumerate(rows_raw):
        norm_feat = apply_norm_to_xgt_feat(raw_feat, norm_params)
        for name in data.dtype.names:
            data[name][idx] = norm_feat.get(name, 0.0)

    np.save(out_dir / "xgt_fen.npy", data)

    print(f"- xgt_fen.npy 저장: {out_dir/'xgt_fen.npy'}")
    print(f"- shape: {data.shape}")

    # 앞 5개 샘플 출력
    print("\n===== 앞 5개 xgt_fen 전처리 샘플 (정규화 적용 후) =====")
    for i in range(min(5, len(data))):
        sample = {name: data[name][i] for name in data.dtype.names}
        print(sample)


# ---------------------------------------------
# TRANSFORM
# ---------------------------------------------
def transform_preprocess(input_path: Path, out_dir: Path):

    vocab_path = out_dir / "xgt_var_vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"❌ {vocab_path} 가 없습니다. 먼저 --fit 을 실행하세요.")

    norm_path = out_dir / NORM_PARAMS_FILE
    if not norm_path.exists():
        raise FileNotFoundError(f"❌ {norm_path} 가 없습니다. 먼저 --fit 을 실행하세요.")

    var_map = json.loads(vocab_path.read_text(encoding="utf-8"))
    norm_params = json.loads(norm_path.read_text(encoding="utf-8"))

    get_var_id = get_var_id_factory(var_map)

    rows_norm: List[Dict[str, float]] = []

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("protocol") != "xgt_fen":
                continue

            raw_feat = preprocess_xgt_record(obj, get_var_id)
            norm_feat = apply_norm_to_xgt_feat(raw_feat, norm_params)
            rows_norm.append(norm_feat)

    dtype = np.dtype([
        ("xgt_var_id", "i4"),
        ("xgt_var_cnt", "f4"),
        ("xgt_source", "f4"),
        ("xgt_fenet_base", "f4"),
        ("xgt_fenet_slot", "f4"),
        ("xgt_cmd", "f4"),
        ("xgt_dtype", "f4"),
        ("xgt_blkcnt", "f4"),
        ("xgt_err_flag", "f4"),
        ("xgt_err_code", "f4"),
        ("xgt_datasize", "f4"),
        ("xgt_addr_count", "f4"),
        ("xgt_addr_min", "f4"),
        ("xgt_addr_max", "f4"),
        ("xgt_addr_range", "f4"),
        ("xgt_word_min", "f4"),
        ("xgt_word_max", "f4"),
        ("xgt_word_mean", "f4"),
        ("xgt_word_std", "f4"),
        ("xgt_data_missing", "f4"),
        ("xgt_data_len_chars", "f4"),
        ("xgt_data_num_spaces", "f4"),
        ("xgt_data_is_hex", "f4"),
        ("xgt_data_n_bytes", "f4"),
        ("xgt_data_zero_ratio", "f4"),
        ("xgt_data_first_byte", "f4"),
        ("xgt_data_last_byte", "f4"),
        ("xgt_data_mean_byte", "f4"),
        ("xgt_data_bucket", "f4"),
    ])


    data = np.zeros(len(rows_norm), dtype=dtype)

    for idx, feat in enumerate(rows_norm):
        for name in data.dtype.names:
            data[name][idx] = feat.get(name, 0.0)

    np.save(out_dir / "xgt_fen.npy", data)

    print("✅ TRANSFORM 완료")
    print(f"- xgt_fen.npy 저장: {out_dir/'xgt_fen.npy'} shape={data.shape}")


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

    if args.fit and args.transform:
        raise ValueError("❌ --fit 과 --transform 는 동시에 사용할 수 없습니다.")
    if not args.fit and not args.transform:
        raise ValueError("❌ 반드시 --fit 또는 --transform 중 하나를 선택하세요.")

    if args.fit:
        fit_preprocess(input_path, out_dir)
    else:
        transform_preprocess(input_path, out_dir)


"""
최종 데이터 사용 (xgt_fen.npy)

    import numpy as np

    data = np.load("output_xgt_fen/xgt_fen.npy")

    # 1) vars embedding 용 ID (이미 int32)
    xgt_var_id = data["xgt_var_id"].astype("int32")

    # 2) numeric feature (이미 이 스크립트에서 정규화 완료된 값들)
    xgt_numeric = np.stack([
        data["xgt_var_cnt"],        # 0~1
        data["xgt_source"],         # 0~1
        data["xgt_fenet_base"],     # 0~1
        data["xgt_fenet_slot"],     # 0~1
        data["xgt_cmd"],            # 0~1
        data["xgt_dtype"],          # 0~1
        data["xgt_blkcnt"],         # 0~1
        data["xgt_err_flag"],       # 0 or 1
        data["xgt_err_code"],       # 0~1
        data["xgt_datasize"],       # 0~1
        data["xgt_data_missing"],   # 0 or 1
        data["xgt_data_len_chars"], # 0~1
        data["xgt_data_num_spaces"],# 0~1
        data["xgt_data_is_hex"],    # 0 or 1
        data["xgt_data_n_bytes"],   # 0~1
        data["xgt_data_zero_ratio"],# 0~1
        data["xgt_data_first_byte"],# 0~1 (0~255 → /255)
        data["xgt_data_last_byte"], # 0~1
        data["xgt_data_mean_byte"], # 0~1
        data["xgt_data_bucket"],    # hash bucket (정규화 X, 필요하면 별도 embedding 사용)
    ], axis=1).astype("float32")


실시간 단일 패킷 예시:

    import json
    from pathlib import Path
    from preprocess_xgt_fen_embed import preprocess_xgt_fen_with_norm

    out_dir = Path("../result/output_xgt_fen")
    var_map = json.loads((out_dir / "xgt_var_vocab.json").read_text(encoding="utf-8"))
    norm_params = json.loads((out_dir / "xgt_fen_norm_params.json").read_text(encoding="utf-8"))

    pkt = {
        "protocol": "xgt_fen",
        "xgt_fen.source": "0x33",
        "xgt_fen.fenetpos": "0x00",
        "xgt_fen.cmd": "0x0054",
        "xgt_fen.dtype": "0x0014",
        "xgt_fen.blkcnt": "1",
        "xgt_fen.errstat": "0x0000",
        "xgt_fen.vars": "%DB001046",
        "xgt_fen.datasize": "12",
        "xgt_fen.data": "05001e00f50000001c002700",
    }

    feat = preprocess_xgt_fen_with_norm(pkt, var_map, norm_params)
    # feat dict를 그대로 모델 입력용 벡터로 변환해서 사용 가능

usage:
    # 학습 데이터 기준 vocab + feature 생성
    python preprocess_xgt_fen_embed.py --fit -i "../data/ML_DL 학습.jsonl" -o "../result/output_xgt_fen"

    # 새 데이터(테스트/운영) 전처리
    python preprocess_xgt_fen_embed.py --transform -i "../data/ML_DL 학습.jsonl" -o "../result/output_xgt_fen"
"""
