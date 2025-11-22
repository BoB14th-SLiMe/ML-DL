#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pad_pattern_features_by_index.py

입력:
  - pattern_features.jsonl
    (window_to_feature_csv.py 에서 만든 window 단위 feature JSONL)

역할:
  - 각 window에서 index 리스트를 시간 축으로 사용
  - window 전체 길이(window_size)를 기준으로 sequence_group을
    index 위치에 맞게 채우고, 나머지 위치는 0 혹은 -1로 padding
  - top-level 구조는 그대로:
      { "window_id", "pattern", "index", "sequence_group" }

출력:
  - 패딩이 적용된 JSONL (각 줄: window 단위)
    - index: [0, 1, ..., window_size-1]
    - sequence_group: 길이 window_size인 리스트
        - 실제 패킷 위치: 원래 feature dict (단, index/packet_idx 없음)
        - 나머지 위치: pad_value 로 채워진 feature dict (역시 index/packet_idx 없음)
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List

# 패킷 dict 안에서 버릴 메타 키 (sequence_group 내부에서는 사용 X)
META_KEYS_IN_PKT = {"index", "packet_idx"}


def load_windows(jsonl_path: Path) -> List[Dict[str, Any]]:
    """pattern_features.jsonl → window 리스트 로드"""
    windows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            windows.append(obj)
    return windows


def infer_window_size(windows: List[Dict[str, Any]]) -> int:
    """
    전체 window의 'index' 목록을 보고 window_size 추론.
    - global max(index) + 1 사용
    """
    max_idx = 0
    for win in windows:
        idx_list = win.get("index", [])
        for v in idx_list:
            try:
                iv = int(v)
            except Exception:
                continue
            if iv > max_idx:
                max_idx = iv
    return max_idx + 1


def build_pad_template(first_pkt: Dict[str, Any], pad_value: float) -> Dict[str, Any]:
    """
    첫 패킷 dict를 기준으로 padding용 템플릿 dict 생성.
    - META_KEYS_IN_PKT("index", "packet_idx")는 제외
    - 나머지 key는 모두 pad_value로 채움
    """
    tmpl: Dict[str, Any] = {}
    for k, v in first_pkt.items():
        if k in META_KEYS_IN_PKT:
            continue
        tmpl[k] = pad_value
    return tmpl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_jsonl",
        required=True,
        help="pattern_features.jsonl 경로"
    )
    parser.add_argument(
        "-o", "--output_jsonl",
        required=True,
        help="padding 적용 후 저장할 JSONL 경로"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="윈도우 길이 (None이면 전체 index 최대값+1을 사용)"
    )
    parser.add_argument(
        "--pad_value",
        type=float,
        default=-1.0,
        help="패딩에 사용할 값 (예: 0.0 또는 -1.0)"
    )
    parser.add_argument(
        "--drop_keys",
        nargs="+",
        default=[],
        help="sequence_group에서 제거할 feature key 리스트 (공백으로 구분)"
        # 예: --drop_keys sq ak fl
    )

    args = parser.parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) window 로딩
    windows = load_windows(input_path)
    if not windows:
        raise RuntimeError("❌ 입력 JSONL에서 window를 하나도 읽지 못했습니다.")

    print(f"[INFO] 총 window 개수: {len(windows)}")

    # 2) window_size 결정
    if args.window_size is not None:
        window_size = args.window_size
    else:
        window_size = infer_window_size(windows)

    print(f"[INFO] 사용할 window_size = {window_size}")

    # 3) padding 템플릿 생성 (첫 window의 첫 packet 기준)
    first_win = windows[0]
    first_seq = first_win.get("sequence_group", [])
    if not first_seq:
        raise RuntimeError("❌ 첫 window의 sequence_group이 비어 있습니다.")

    first_pkt = first_seq[0]
    pad_template = build_pad_template(first_pkt, args.pad_value)
    feature_keys = list(pad_template.keys())
    print(f"[INFO] feature key 수 (pad 대상): {len(feature_keys)}")

    # ★★★ feature 목록을 result(=output 디렉토리)에 저장 ★★★
    feature_list_path = output_path.parent / "feature_keys.txt"
    with feature_list_path.open("w", encoding="utf-8") as f_feat:
        for k in feature_keys:
            f_feat.write(k + "\n")
    print(f"[INFO] feature_keys.txt 저장 → {feature_list_path}")

    # 4) 각 window에 대해 index 기반 padding 적용
    with output_path.open("w", encoding="utf-8") as fout:
        for win in windows:
            window_id = win.get("window_id")
            pattern = win.get("pattern")
            idx_list = win.get("index", [])
            seq = win.get("sequence_group", [])

            # index → packet 매핑 (안전하게 길이 체크)
            pos_to_pkt: Dict[int, Dict[str, Any]] = {}
            T = min(len(idx_list), len(seq))
            for i in range(T):
                try:
                    pos = int(idx_list[i])
                except Exception:
                    pos = i  # fallback
                pos_to_pkt[pos] = seq[i]

            # 새 index / sequence_group 생성
            new_index = list(range(window_size))
            new_seq: List[Dict[str, Any]] = []

            for pos in range(window_size):
                if pos in pos_to_pkt:
                    # 실제 패킷이 존재하는 위치 → 원본 feature dict 사용
                    #  단, index/packet_idx는 제거
                    orig_pkt = pos_to_pkt[pos]
                    pkt = {k: v for k, v in orig_pkt.items() if k not in META_KEYS_IN_PKT}
                    new_seq.append(pkt)
                else:
                    # 패딩 위치 → pad_template 기반으로 채움
                    pad_pkt = {}
                    for k in feature_keys:
                        pad_pkt[k] = args.pad_value
                    # index / packet_idx는 아예 넣지 않음
                    new_seq.append(pad_pkt)

            out_obj = {
                "window_id": window_id,
                "pattern": pattern,
                "index": new_index,
                "sequence_group": new_seq,
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"[INFO] padding 적용 완료 → {output_path}")


if __name__ == "__main__":
    main()

"""
# drop key
python 1.padding.py -i "../data/pattern_features.jsonl" -o "../result/pattern_features_padded_0.jsonl" --pad_value 0 --window_size 76 --drop_keys deltat

# drop key
python 1.padding.py -i "../data/pattern_features.jsonl" -o "../result/pattern_features_padded_-1.jsonl" --pad_value 0 --window_size 76 --drop_keys deltat
"""
