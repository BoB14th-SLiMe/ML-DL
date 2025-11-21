#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocess_arp_embed.py
A버전: ARP 전용 embedding/feature 전처리 (초미니멀 버전)

두 모드 제공:
  --fit        : arp_host_map 생성 후 arp.npy 저장
  --transform  : 기존 arp_host_map 사용

입력 JSONL에서 사용하는 필드:
  - protocol == "arp"
  - smac      : 소스 MAC
  - sip       : 소스 IP
  - arp.op    : "1" (request), "2" (reply)
  - arp.tmac  : 타겟 MAC
  - arp.tip   : 타겟 IP (예: 192.168.10.81)

출력 feature (arp.npy, structured numpy):
  - arp_src_host_id   (int32)   ← (smac, sip) 조합 → ID, Embedding용
  - arp_tgt_host_id   (int32)   ← (arp.tmac, arp.tip) 조합 → ID, Embedding용
  - arp_op_num       (float32) ← op 원본 값 (0, 1, 2)  # 1=request, 2=reply
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------
# Host ID 생성기 (MAC+IP 묶어서 host 기준)
# ---------------------------------------------
def get_host_id_factory(host_map: Dict[str, int]):
    """
    host_map: {"00:0b:29:74:0f:7b|192.168.10.15": 1, ...}
    """
    next_id = max(host_map.values()) + 1 if host_map else 1

    def get_host_id(mac: Any, ip: Any) -> int:
        nonlocal next_id
        mac_str = str(mac).strip() if mac else ""
        ip_str = str(ip).strip() if ip else ""
        if not mac_str or not ip_str:
            return 0  # UNK
        key = f"{mac_str}|{ip_str}"
        if key not in host_map:
            host_map[key] = next_id
            next_id += 1
        return host_map[key]

    return get_host_id


# ---------------------------------------------
# arp.op 파싱
# ---------------------------------------------
def parse_arp_op(op_val: Any) -> int:
    """
    op_val: "1", 1, "2", 2 등 → int 1 또는 2 (그 외는 0)
      - 1: request
      - 2: reply
      - 0: unknown / invalid
    """
    if isinstance(op_val, list) and op_val:
        op_val = op_val[0]
    try:
        op_int = int(op_val)
    except (TypeError, ValueError):
        return 0
    if op_int in (1, 2):
        return op_int
    return 0


# ---------------------------------------------
# 한 레코드(ARP) 전처리
# ---------------------------------------------
def preprocess_arp_record(obj: Dict[str, Any], get_host_id) -> Dict[str, float]:
    """
    protocol == "arp" 인 레코드를 feature dict로 변환
    """

    feat: Dict[str, float] = {}

    # 원시 필드 가져오기
    smac = obj.get("smac")
    sip = obj.get("sip")
    tmac = obj.get("arp.tmac")
    tip = obj.get("arp.tip")
    op_raw = obj.get("arp.op")

    # 1) host id (소스 / 타겟)
    src_host_id = get_host_id(smac, sip)
    tgt_host_id = get_host_id(tmac, tip)

    feat["arp_src_host_id"] = int(src_host_id)
    feat["arp_tgt_host_id"] = int(tgt_host_id)

    # 2) op 원본 코드 (1=request, 2=reply, 나머지 0)
    op_int = parse_arp_op(op_raw)  # 0,1,2
    feat["arp_op_num"] = float(op_int)  # 정규화 없이 원본 값 그대로

    return feat


# ---------------------------------------------
# FIT
# ---------------------------------------------
def fit_preprocess(input_path: Path, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    host_map: Dict[str, int] = {}
    get_host_id = get_host_id_factory(host_map)

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

            # ARP만 처리
            if obj.get("protocol") != "arp":
                continue

            feat = preprocess_arp_record(obj, get_host_id)
            rows.append(feat)

    # host_map 저장
    (out_dir / "arp_host_map.json").write_text(
        json.dumps(host_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("✅ FIT 완료")
    print(f"- arp_host_map.json 저장: {out_dir/'arp_host_map.json'}")

    # numpy 구조화 배열 생성
    dtype = np.dtype([
        ("arp_src_host_id", "i4"),
        ("arp_tgt_host_id", "i4"),
        ("arp_op_num", "f4"),  # 0/1/2 원본 코드
    ])

    data = np.zeros(len(rows), dtype=dtype)

    for idx, feat in enumerate(rows):
        data["arp_src_host_id"][idx] = int(feat.get("arp_src_host_id", 0))
        data["arp_tgt_host_id"][idx] = int(feat.get("arp_tgt_host_id", 0))
        data["arp_op_num"][idx]     = float(feat.get("arp_op_num", 0.0))

    np.save(out_dir / "arp.npy", data)

    print(f"- arp.npy 저장: {out_dir/'arp.npy'}")
    print(f"- shape: {data.shape}")

    # 앞 5개 샘플 출력
    print("\n===== 앞 5개 ARP 전처리 샘플 =====")
    for i in range(min(5, len(data))):
        sample = {name: data[name][i] for name in data.dtype.names}
        print(sample)


# ---------------------------------------------
# TRANSFORM
# ---------------------------------------------
def transform_preprocess(input_path: Path, out_dir: Path):

    host_map_path = out_dir / "arp_host_map.json"
    if not host_map_path.exists():
        raise FileNotFoundError(f"❌ {host_map_path} 가 없습니다. 먼저 --fit 을 실행하세요.")

    host_map = json.loads(host_map_path.read_text(encoding="utf-8"))
    get_host_id = get_host_id_factory(host_map)

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

            if obj.get("protocol") != "arp":
                continue

            feat = preprocess_arp_record(obj, get_host_id)
            rows.append(feat)

    dtype = np.dtype([
        ("arp_src_host_id", "i4"),
        ("arp_tgt_host_id", "i4"),
        ("arp_op_num", "f4"),
    ])

    data = np.zeros(len(rows), dtype=dtype)

    for idx, feat in enumerate(rows):
        data["arp_src_host_id"][idx] = int(feat.get("arp_src_host_id", 0))
        data["arp_tgt_host_id"][idx] = int(feat.get("arp_tgt_host_id", 0))
        data["arp_op_num"][idx]     = float(feat.get("arp_op_num", 0.0))

    np.save(out_dir / "arp.npy", data)

    print("✅ TRANSFORM 완료")
    print(f"- arp.npy 저장: {out_dir/'arp.npy'} shape={data.shape}")


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
최종 데이터 사용 (arp.npy)
    import numpy as np

    data = np.load("../result/output_arp/arp.npy")

    arp_src_host_id = data["arp_src_host_id"].astype("int32")
    arp_tgt_host_id = data["arp_tgt_host_id"].astype("int32")

    # 0=unknown, 1=request, 2=reply
    arp_numeric = np.stack([
        data["arp_op_num"],
    ], axis=1).astype("float32")
"""

"""
usage:
    # 학습용 ARP 데이터에서 host_map + feature 생성
    python arp.py --fit -i "../data/ML_DL 학습.jsonl" -o "../result/output_arp"
    
    # 이후 새 데이터에 대해 같은 host_map으로 전처리
    python arp.py --transform -i "../data/ML_DL 학습.jsonl" -o "../result/output_arp"
"""
