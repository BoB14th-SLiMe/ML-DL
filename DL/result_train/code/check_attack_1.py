#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from packet_feature_extractor import (
    load_preprocess_params,
    sequence_group_to_feature_matrix,
    PACKET_FEATURE_COLUMNS,
)


def load_top_n_modbus_packets(jsonl_path: Path, n: int = 3) -> List[Dict[str, Any]]:
    """
    JSONL에서 protocol == 'modbus' 인 패킷 상위 n개만 가져오기
    """
    result: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            proto = obj.get("protocol")
            # 필요하면 여기서 'modbus_tcp' 같은 것도 포함하도록 조건 확장 가능
            if proto == "modbus":
                result.append(obj)
                if len(result) >= n:
                    break
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True, help="attack.jsonl 경로")
    p.add_argument("-p", "--pre-dir", required=True, help="전처리 파라미터 디렉토리")
    p.add_argument(
        "-n", "--top-n",
        type=int,
        default=3,
        help="출력할 modbus 패킷 개수 (기본: 3)",
    )
    args = p.parse_args()

    input_path = Path(args.input)
    pre_dir = Path(args.pre_dir)

    # 1) modbus 패킷 상위 N개 로드
    modbus_pkts = load_top_n_modbus_packets(input_path, n=args.top_n)

    if not modbus_pkts:
        print("❌ protocol == 'modbus' 인 패킷을 찾지 못했습니다.")
        return

    print(f"✅ protocol == 'modbus' 패킷 {len(modbus_pkts)}개 로드 (요청: {args.top_n}개)\n")

    # 2) 전처리 파라미터 로드
    params = load_preprocess_params(pre_dir)

    # 3) 각 패킷에 대해 raw + 전처리 결과 출력
    for idx, pkt in enumerate(modbus_pkts, start=1):
        print("=" * 80)
        print(f"[MODBUS PACKET #{idx}] RAW")
        print("-" * 80)
        print(json.dumps(pkt, ensure_ascii=False, indent=2))

        # sequence_group처럼 1개만 넣어서 전처리
        seq_group: List[Dict[str, Any]] = [pkt]
        X_list = sequence_group_to_feature_matrix(seq_group, params)

        if not X_list:
            print("\n[WARN] 이 패킷은 전처리 결과가 비었습니다.")
            continue

        feat_vec = X_list[0]

        print("\n[MODBUS PACKET #{idx}] PREPROCESSED FEATURE")
        print("-" * 80)
        for name, val in zip(PACKET_FEATURE_COLUMNS, feat_vec):
            print(f"{name:25s} : {val}")
        print()  # 한 줄 띄우기


if __name__ == "__main__":
    main()

"""
사용 예시:
python check_attack_1.py -i "../data/attack.jsonl" -p "../../preprocessing/result" -n 3
"""
