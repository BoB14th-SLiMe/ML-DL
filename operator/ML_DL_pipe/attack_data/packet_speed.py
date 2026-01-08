#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calc_avg_packet_rate.py

주어진 JSONL 파일(@timestamp 필드 포함)에서
패킷 간 시간 차이를 계산하여 평균 패킷 속도(pps)를 산출합니다.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

def parse_iso8601(ts_str: str) -> float:
    """ISO8601 형식(끝에 Z 포함)을 UTC epoch float 초 단위로 변환"""
    if not ts_str:
        raise ValueError("빈 timestamp 문자열입니다.")
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    return dt.timestamp()

def calc_avg_pps(jsonl_path: Path) -> None:
    """JSONL 파일을 읽어 평균 패킷 속도 계산"""
    timestamps = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ts = obj.get("@timestamp")
                if ts:
                    timestamps.append(parse_iso8601(ts))
            except json.JSONDecodeError:
                continue

    if len(timestamps) < 2:
        print("패킷이 2개 미만입니다. 속도를 계산할 수 없습니다.")
        return

    timestamps.sort()
    total_span = timestamps[-1] - timestamps[0]
    interval_count = len(timestamps) - 1
    avg_interval = total_span / interval_count if interval_count > 0 else 0
    avg_pps = (1.0 / avg_interval) if avg_interval > 0 else 0

    print("=== Packet Rate Summary ===")
    print(f"총 패킷 수        : {len(timestamps)}")
    print(f"첫 패킷 시각      : {datetime.fromtimestamp(timestamps[0], tz=timezone.utc)}")
    print(f"마지막 패킷 시각  : {datetime.fromtimestamp(timestamps[-1], tz=timezone.utc)}")
    print(f"총 시간 차이(sec) : {total_span:.6f}")
    print(f"평균 간격(sec)    : {avg_interval:.6f}")
    print(f"평균 패킷 속도    : {avg_pps:.2f} packets/sec")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python calc_avg_packet_rate.py <input.jsonl>")
        sys.exit(1)

    jsonl_file = Path(sys.argv[1])
    if not jsonl_file.exists():
        print(f"파일을 찾을 수 없습니다: {jsonl_file}")
        sys.exit(1)

    calc_avg_pps(jsonl_file)

"""
python packet_speed.py attack_ver2.jsonl
"""