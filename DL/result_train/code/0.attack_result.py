#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mark_fc6_attack_windows.py

attack.jsonl 처럼 "한 줄 = 1 패킷" 형식의 JSONL을 입력으로 받아서,

- window_size, step_size 로 슬라이딩 윈도우를 만들고
- 각 윈도우에 modbus 프로토콜 패킷 중 modbus.fc == 6 이 하나라도 포함되면
    → is_anomaly = 1
  그렇지 않으면
    → is_anomaly = 0
- 다음 컬럼만 가진 CSV를 생성한다.

출력 CSV 컬럼:
  window_index, start_packet_idx, end_packet_idx, valid_len, is_anomaly

※ packet index는 0-based (attack.jsonl에서의 라인 순서 기준)
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input", "-i", required=True,
        help="패킷 단위 JSONL 경로 (각 line = 1 packet, 예: attack.jsonl)",
    )
    p.add_argument(
        "--window-size", "-w", type=int, default=80,
        help="윈도우 길이(묶을 패킷 개수, 기본=80)",
    )
    p.add_argument(
        "--step-size", "-s", type=int, default=None,
        help="슬라이딩 stride (기본: window-size와 동일 → non-overlap)",
    )
    p.add_argument(
        "--output", "-o", required=True,
        help="출력 CSV 경로 (window_index,start_packet_idx,end_packet_idx,valid_len,is_anomaly)",
    )

    return p.parse_args()


def safe_int(val: Any, default: int = 0) -> int:
    """
    modbus.fc 처럼 [\"6\", null] 형태도 들어올 수 있으니,
    리스트면 첫 번째 유효 값을 int로 캐스팅하고,
    실패하면 default 반환.
    """
    try:
        if isinstance(val, list):
            # 리스트 안에서 None이 아닌 첫 값 사용
            for v in val:
                if v is None:
                    continue
                val = v
                break
        return int(val)
    except Exception:
        return default


def window_has_modbus_fc6(packets: List[Dict[str, Any]]) -> bool:
    """
    윈도우 내에 protocol == 'modbus' 이고 modbus.fc == 6 인 패킷이
    하나라도 있으면 True 반환.
    """
    for pkt in packets:
        if pkt.get("protocol") != "modbus":
            continue
        fc_raw = pkt.get("modbus.fc")
        fc = safe_int(fc_raw, default=-1)
        if fc == 6:
            return True
    return False


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    window_size = args.window_size
    step_size = args.step_size if args.step_size is not None else window_size

    print(f"[INFO] 입력 JSONL : {input_path}")
    print(f"[INFO] window_size = {window_size}, step_size = {step_size}")
    print(f"[INFO] 출력 CSV    : {output_path}")

    # 1) JSONL 전체 읽어서 패킷 리스트 생성
    packets: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] JSON 파싱 실패, 스킵: {e}")
                continue
            packets.append(obj)

    total_packets = len(packets)
    print(f"[INFO] 총 패킷 수 = {total_packets}")

    if total_packets == 0:
        print("[WARN] 유효한 패킷이 없습니다. 종료합니다.")
        return

    # 2) 슬라이딩 윈도우 만들면서 fc=6 포함 여부 체크
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow([
            "window_index",
            "start_packet_idx",
            "end_packet_idx",
            "valid_len",
            "is_anomaly",
        ])

        window_index = 0
        start_idx = 0

        while start_idx < total_packets:
            end_idx = start_idx + window_size
            window_packets = packets[start_idx:end_idx]

            if not window_packets:
                break

            valid_len = len(window_packets)
            end_packet_idx = start_idx + valid_len - 1

            # 규칙: 윈도우 안에 modbus.fc == 6 이 하나라도 있으면 공격
            is_attack = 1 if window_has_modbus_fc6(window_packets) else 0

            writer.writerow([
                window_index,
                start_idx,
                end_packet_idx,
                valid_len,
                is_attack,
            ])

            window_index += 1
            start_idx += step_size

    print(f"[INFO] 완료: 총 {window_index}개 윈도우 결과를 {output_path} 에 저장했습니다.")


if __name__ == "__main__":
    main()

"""
python 0.attack_result.py --input "../data/attack.jsonl" --window-size 80 --step-size 30 --output "../result/attack_result.csv"

"""