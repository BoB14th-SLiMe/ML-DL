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
  - xgt_fen.source    : int
  - xgt_fen.fenetpos  : int (상위 4bit = base, 하위 4bit = slot)
  - xgt_fen.cmd       : int
  - xgt_fen.dtype     : int (또는 xgt_fen.dype)
  - xgt_fen.blkcnt    : int
  - xgt_fen.datasize  : int
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
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Any, Dict, List


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

# Min-Max 정규화 대상 필드
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

    # ★ 여기서부터 int로 유지
    var_id = get_var_id(first_var) if first_var else 0
    feat["xgt_var_id"] = int(var_id)      # 저장도 int, dtype도 int32
    feat["xgt_var_cnt"] = float(var_cnt)

    # 2) 헤더/명령 필드
    def to_int(value: Any, default: int = 0) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    source = to_int(obj.get("xgt_fen.source"))
    fenetpos = to_int(obj.get("xgt_fen.fenetpos"))
    cmd = to_int(obj.get("xgt_fen.cmd"))
    dtype = to_int(obj.get("xgt_fen.dtype") or obj.get("xgt_fen.dype"))
    blkcnt = to_int(obj.get("xgt_fen.blkcnt"))
    datasize = to_int(obj.get("xgt_fen.datasize"))

    base = (fenetpos >> 4) & 0x0F
    slot = fenetpos & 0x0F

    feat["xgt_source"] = float(source)
    feat["xgt_fenet_base"] = float(base)
    feat["xgt_fenet_slot"] = float(slot)
    feat["xgt_cmd"] = float(cmd)
    feat["xgt_dtype"] = float(dtype)
    feat["xgt_blkcnt"] = float(blkcnt)
    feat["xgt_datasize"] = float(datasize)

    # 3) data 요약
    data_field = obj.get("xgt_fen.data")
    data_feats = extract_xgt_data_features(data_field)
    feat.update(data_feats)

    # 4) 데이터 없음 플래그 추가
    #    - datasize > 0 이고, data_len_chars == 0 인 경우 → 1.0
    length_chars = data_feats.get("xgt_data_len_chars", 0.0)
    xgt_data_missing = 1.0 if (datasize > 0 and length_chars == 0.0) else 0.0
    feat["xgt_data_missing"] = float(xgt_data_missing)

    return feat


# ---------------------------------------------
# Min-Max 정규화 함수
# ---------------------------------------------
def minmax_norm(val: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.0
    return (val - vmin) / (vmax - vmin + 1e-9)


# ---------------------------------------------
# FIT
# ---------------------------------------------
def fit_preprocess(input_path: Path, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    var_map: Dict[str, int] = {}
    get_var_id = get_var_id_factory(var_map)

    rows: List[Dict[str, float]] = []

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
            rows.append(feat)

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

    for feat in rows:
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

    data = np.zeros(len(rows), dtype=dtype)

    for idx, feat in enumerate(rows):
        # 1) ID (정규화 X)
        data["xgt_var_id"][idx] = int(feat.get("xgt_var_id", 0))

        # 2) Min-Max 정규화 대상
        for f in NORM_FIELDS:
            raw_v = float(feat.get(f, 0.0))
            vmin = norm_params[f]["min"]
            vmax = norm_params[f]["max"]
            norm_v = minmax_norm(raw_v, vmin, vmax)
            data[f][idx] = float(norm_v)

        # 3) 그대로 쓰는 값들 (flag / ratio / bucket 등)
        data["xgt_err_flag"][idx]       = float(feat.get("xgt_err_flag", 0.0))
        # xgt_err_code는 이미 위에서 Min-Max 정규화로 채워짐
        # xgt_datasize도 이미 위에서 Min-Max 정규화로 채워짐
        data["xgt_data_missing"][idx]   = float(feat.get("xgt_data_missing", 0.0))
        # xgt_data_len_chars, xgt_data_num_spaces, xgt_data_n_bytes
        # -> 위에서 Min-Max 정규화됨
        data["xgt_data_is_hex"][idx]    = float(feat.get("xgt_data_is_hex", 0.0))
        data["xgt_data_zero_ratio"][idx]= float(feat.get("xgt_data_zero_ratio", 0.0))

        # 4) 바이트 값들은 0~1 스케일 (/255)
        fb = float(feat.get("xgt_data_first_byte", 0.0))
        lb = float(feat.get("xgt_data_last_byte", 0.0))
        mb = float(feat.get("xgt_data_mean_byte", 0.0))

        data["xgt_data_first_byte"][idx] = fb / 255.0
        data["xgt_data_last_byte"][idx]  = lb / 255.0
        data["xgt_data_mean_byte"][idx]  = mb / 255.0

        # 5) bucket (정규화 X)
        data["xgt_data_bucket"][idx]    = float(feat.get("xgt_data_bucket", 0.0))

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

    rows: List[Dict[str, float]] = []

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
            rows.append(feat)

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

    data = np.zeros(len(rows), dtype=dtype)

    for idx, feat in enumerate(rows):
        data["xgt_var_id"][idx] = int(feat.get("xgt_var_id", 0))

        # Min-Max 정규화 대상
        for f in NORM_FIELDS:
            raw_v = float(feat.get(f, 0.0))
            vmin = norm_params[f]["min"]
            vmax = norm_params[f]["max"]
            norm_v = minmax_norm(raw_v, vmin, vmax)
            data[f][idx] = float(norm_v)

        # 그대로 쓰는 값
        data["xgt_err_flag"][idx]       = float(feat.get("xgt_err_flag", 0.0))
        data["xgt_data_missing"][idx]   = float(feat.get("xgt_data_missing", 0.0))
        data["xgt_data_is_hex"][idx]    = float(feat.get("xgt_data_is_hex", 0.0))
        data["xgt_data_zero_ratio"][idx]= float(feat.get("xgt_data_zero_ratio", 0.0))

        # 바이트 값 → /255
        fb = float(feat.get("xgt_data_first_byte", 0.0))
        lb = float(feat.get("xgt_data_last_byte", 0.0))
        mb = float(feat.get("xgt_data_mean_byte", 0.0))

        data["xgt_data_first_byte"][idx] = fb / 255.0
        data["xgt_data_last_byte"][idx]  = lb / 255.0
        data["xgt_data_mean_byte"][idx]  = mb / 255.0

        data["xgt_data_bucket"][idx]    = float(feat.get("xgt_data_bucket", 0.0))

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

    if args.fit:
        fit_preprocess(input_path, out_dir)
    elif args.transform:
        transform_preprocess(input_path, out_dir)
    else:
        raise ValueError("❌ 반드시 --fit 또는 --transform 중 하나를 선택하세요.")


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
"""


"""
usage:
    # 학습 데이터 기준 vocab + feature 생성
    python xgt-fen.py --fit -i "../data/ML_DL 학습.jsonl" -o "../result/output_xgt_fen"

    # 새 데이터(테스트/운영) 전처리
    python xgt-fen.py --transform -i "../data/ML_DL 학습.jsonl" -o "../result/output_xgt_fen"
"""