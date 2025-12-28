#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3.window_index.py

SLM이 추출한 패턴(PLS/sequence_group) 내 각 패킷이,
해당 window의 sequence_group(=RAW 매핑된 윈도우 패킷 리스트)에서 몇 번째 위치(index)에 존재하는지 파악해
패턴 레코드에 "index" 필드를 추가한다.
"""

import sys
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.file_load import file_load
from utils.extract_feature import raw_extract
from utils.file_save import save_jsonl

Key = Tuple[str, str, str, str]


def load_window(window_records: List[Dict[str, Any]]) -> Dict[str, Dict[Key, List[int]]]:
    window_to_key_indices: Dict[str, Dict[Key, List[int]]] = {}

    total_obj = 0
    skipped_no_window_id = 0
    skipped_bad_group = 0
    total_pkt = 0
    dict_pkt = 0
    str_pkt = 0
    other_pkt = 0
    key_none = 0
    example_bad_pkt: Optional[str] = None

    for window in window_records:
        total_obj += 1

        window_id = window.get("window_id")
        if window_id is None:
            skipped_no_window_id += 1
            continue

        seq = window.get("sequence_group")
        if not isinstance(seq, list):
            skipped_bad_group += 1
            continue

        key_map = window_to_key_indices.setdefault(window_id, {})

        for i, pkt in enumerate(seq):
            total_pkt += 1
            if isinstance(pkt, dict):
                dict_pkt += 1
            elif isinstance(pkt, str):
                str_pkt += 1
            else:
                other_pkt += 1

            key = raw_extract(pkt, ["@timestamp", "sq", "ak", "fl"])

            if key is None:
                key_none += 1
                if example_bad_pkt is None:
                    example_bad_pkt = str(pkt)[:200]
                continue

            key_map.setdefault(key, []).append(i)

    print("[load_window] summary")
    print(f"  total_obj={total_obj}")
    print(f"  windows_built={len(window_to_key_indices)}")
    print(f"  skipped_no_window_id={skipped_no_window_id}")
    print(f"  skipped_bad_sequence_group={skipped_bad_group}")
    print(f"  total_pkt={total_pkt} (dict={dict_pkt}, str={str_pkt}, other={other_pkt}), key_none={key_none}")
    if example_bad_pkt is not None:
        print(f"  example_unparsable_pkt={example_bad_pkt}")

    return window_to_key_indices


def insert_index(
    pattern_records: List[Dict[str, Any]],
    window_index: Dict[str, Dict[Key, List[int]]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    total = 0
    written = 0
    skipped_noise = 0
    skipped_no_window_id = 0
    skipped_bad_group = 0
    skipped_no_window_map = 0
    skipped_unmatched = 0
    example_missing_window: Optional[str] = None

    for obj in pattern_records:
        total += 1

        label = obj.get("pattern") or obj.get("label") or ""
        if str(label).strip().lower() == "noise":
            skipped_noise += 1
            continue

        window_id = obj.get("window_id")
        if window_id is None:
            skipped_no_window_id += 1
            continue

        seq = obj.get("sequence_group")
        if not isinstance(seq, list):
            skipped_bad_group += 1
            continue

        per_window = window_index.get(window_id)
        if not per_window:
            skipped_no_window_map += 1
            if example_missing_window is None:
                example_missing_window = window_id
            continue

        used_cursor: Dict[Key, int] = {}
        idxs: List[int] = []
        ok = True

        for pkt in seq:
            key = raw_extract(pkt, ["@timestamp", "sq", "ak", "fl"])
            if key is None:
                ok = False
                break

            candidates = per_window.get(key)
            if not candidates:
                ok = False
                break

            k = used_cursor.get(key, 0)
            if k >= len(candidates):
                ok = False
                break

            idxs.append(candidates[k])
            used_cursor[key] = k + 1

        if not ok:
            skipped_unmatched += 1
            continue

        obj["index"] = idxs
        out.append(obj)
        written += 1

    print("[insert_index] summary")
    print(f"  total_records={total}")
    print(f"  written={written}")
    print(f"  skipped_noise={skipped_noise}")
    print(f"  skipped_no_window_id={skipped_no_window_id}")
    print(f"  skipped_bad_sequence_group={skipped_bad_group}")
    print(f"  skipped_no_window_map={skipped_no_window_map}")
    print(f"  skipped_unmatched={skipped_unmatched}")
    if example_missing_window is not None:
        print(f"  example_window_id_missing_in_window_index={example_missing_window}")

    return out

def window_index(pattern_jsonl: Path, window_jsonl: Path, out_jsonl: Path):
    window_records = file_load("jsonl", str(window_jsonl))
    pattern_records = file_load("jsonl", str(pattern_jsonl))

    window_index = load_window(window_records)
    results = insert_index(pattern_records, window_index)

    save_jsonl(results, out_jsonl)
    print(f"결과 : {len(results)} -> {out_jsonl}")

if __name__ == "__main__":
    pattern_path = ROOT / "results" / "PLS_to_RAW_mapped.jsonl"
    window_path = ROOT / "results" / "PLS_to_RAW_windows_mapped.jsonl"
    out_path = ROOT / "results" / "pattern.jsonl"

    window_index(pattern_path, window_path, out_path)

