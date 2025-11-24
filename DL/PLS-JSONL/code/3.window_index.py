#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3.window_index.py

역할:
  - 2_window_PLS_to_Raw_result.jsonl : window_id 별 RAW 패킷 전체
      {"window_id": 1, "RAW": [ {...}, {...}, ... ]}

  - 1-1_PLS_to_Raw_result.jsonl : 패턴 라벨 + 부분 패킷(window_group)
      {"window_id": 1, "label": "P_0021", "window_group": [ {...}, ... ]}

  → 각 1-1 라인에 "index": [...] 를 추가.
    - index[i] 는 해당 window_id 의 RAW 리스트에서
      window_group[i] 패킷이 위치한 인덱스(0-based).

추가 동작:
  - pattern(또는 label)이 "noise" (대소문자 무시) 인 윈도우는 출력에서 제외
  - window_group 중 하나라도 RAW에 매핑되지 않으면
    해당 윈도우 전체를 버림 (결과 파일에는 포함되지 않음)

사용 예:
  python 3.window_index.py \
      --raw-jsonl 2_window_PLS_to_Raw_result.jsonl \
      --pls-jsonl 1-1_PLS_to_Raw_result.jsonl \
      --output-jsonl 1-2_PLS_to_Raw_with_index.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


# -------------------------------
# 패킷 매칭에 사용할 키 생성
# (여기서는 timestamp + sq 사용)
# -------------------------------
KEY_FIELDS = [
    "@timestamp",
]


def build_packet_key(pkt: dict):
    """패킷 dict -> 매칭용 튜플 키

    없는 필드는 빈 문자열로 채우고,
    값은 모두 str로 변환해서 비교의 일관성을 유지.
    """
    return tuple(str(pkt.get(f, "")) for f in KEY_FIELDS)


# -------------------------------
# 2_window 파일로부터 index 맵 구성
# -------------------------------
def build_raw_index_map(raw_jsonl_path: Path):
    """
    return:
      raw_index_map: dict[
          window_id -> dict[
              packet_key(tuple) -> list[index(int)]
          ]
      ]
    """
    raw_index_map = defaultdict(lambda: defaultdict(list))

    with raw_jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"[ERROR] JSON 파싱 실패 (line {line_no}) in {raw_jsonl_path}: {e}")

            window_id = obj["window_id"]
            raw_list = obj.get("RAW", [])

            for idx, pkt in enumerate(raw_list):
                key = build_packet_key(pkt)
                raw_index_map[window_id][key].append(idx)

    return raw_index_map


# -------------------------------
# 1-1 파일에 index 추가
# -------------------------------
def add_index_to_pls_raw(
    pls_jsonl_path: Path,
    raw_index_map,
    output_path: Path,
):
    missing_count = 0          # 매핑 실패한 패킷 수 (스킵한 이유)
    total_pkt_considered = 0   # noise 제외하고, 매핑 시도한 패킷 수
    total_windows = 0          # noise 제외하고, 매핑 시도한 윈도우 수
    written_windows = 0        # 실제로 출력된 윈도우 수
    skipped_noise = 0          # pattern / label == "noise" 로 스킵된 윈도우 수
    skipped_unmatched = 0      # 매핑 실패로 스킵된 윈도우 수

    with pls_jsonl_path.open("r", encoding="utf-8") as fin, \
            output_path.open("w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"[ERROR] JSON 파싱 실패 (line {line_no}) in {pls_jsonl_path}: {e}")

            # 1) noise 패턴은 아예 버림 (대소문자 무시)
            pattern_raw = obj.get("pattern") or obj.get("label") or ""
            pattern = str(pattern_raw).strip().lower()
            if pattern == "noise":
                skipped_noise += 1
                continue

            window_id = obj["window_id"]
            window_group = obj.get("window_group", [])

            total_windows += 1
            total_pkt_considered += len(window_group)

            per_window_map = raw_index_map.get(window_id, {})

            index_list = []
            unmatched_flag = False

            for pkt in window_group:
                key = build_packet_key(pkt)
                idx_list = per_window_map.get(key)

                if idx_list:
                    # RAW에서 해당 키로 찾은 첫 번째 인덱스를 사용 (소모 X)
                    idx = idx_list[0]
                    index_list.append(idx)
                else:
                    # 매핑 실패 → 이 윈도우 전체를 버림
                    index_list.append(-1)
                    missing_count += 1
                    unmatched_flag = True

            # 하나라도 매핑 실패가 있으면 이 윈도우는 통째로 스킵
            if unmatched_flag:
                skipped_unmatched += 1
                continue

            # 여기까지 왔다는 건 모든 패킷이 RAW에 매핑 성공했다는 뜻
            obj["index"] = index_list

            # (옵션) 각 패킷 dict 안에 raw_index를 넣고 싶다면 아래 주석 해제
            # for pkt, idx in zip(window_group, index_list):
            #     pkt["raw_index"] = idx

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written_windows += 1

    print(f"[INFO] noise 패턴으로 스킵된 윈도우 수: {skipped_noise}")
    print(f"[INFO] 매핑 시도한 윈도우 수 (noise 제외): {total_windows}")
    print(f"[INFO] 완전히 매핑되어 출력된 윈도우 수: {written_windows}")
    print(f"[INFO] 매핑 실패로 스킵된 윈도우 수: {skipped_unmatched}")
    print(f"[INFO] 매핑 시도한 패킷 수 (noise 제외): {total_pkt_considered}")
    print(f"[INFO] 매칭 실패 패킷 수(스킵 이유): {missing_count}")


# -------------------------------
# main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="1-1_PLS_to_Raw_result.jsonl 에 RAW index 를 추가하는 스크립트 (3.window_index.py)"
    )
    parser.add_argument(
        "--raw-jsonl",
        type=Path,
        required=True,
        help="2_window_PLS_to_Raw_result.jsonl 경로",
    )
    parser.add_argument(
        "--pls-jsonl",
        type=Path,
        required=True,
        help="1-1_PLS_to_Raw_result.jsonl 경로",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="index 추가된 JSONL 출력 경로",
    )

    args = parser.parse_args()

    print(f"[INFO] RAW 인덱스 맵 생성 중: {args.raw_jsonl}")
    raw_index_map = build_raw_index_map(args.raw_jsonl)

    print(f"[INFO] 1-1 파일에 index 추가 중: {args.pls_jsonl}")
    add_index_to_pls_raw(args.pls_jsonl, raw_index_map, args.output_jsonl)

    print(f"[DONE] 결과 저장: {args.output_jsonl}")


if __name__ == "__main__":
    main()


"""
python 3.window_index.py --raw-jsonl ../result/2_window_PLS_to_Raw_result.jsonl --pls-jsonl ../result/1-1_PLS_to_Raw_result.jsonl --output-jsonl ../../preprocessing/data/pattern_windows.jsonl

"""