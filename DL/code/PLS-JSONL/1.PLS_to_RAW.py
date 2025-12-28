#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1.PLS_to_RAW.py

PLS(JSONL) 결과와 RAW(JSONL) 패킷을 다음 기준으로 매핑한다.
  - @timestamp
  - sq
  - ak
  - fl
"""

from pathlib import Path
from tqdm import tqdm
import sys
from collections import defaultdict, deque
from typing import List, Dict, Any, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.file_load import file_load
from utils.extract_feature import pls_extract, raw_extract
from utils.file_save import save_jsonl

def PLS_to_RAW_mapping(PLS_jsonl: Path, RAW_jsonl: Path, out_jsonl: Path):
    file_type = "jsonl"
    match_required = ("@timestamp", "sq", "ak", "fl")

    RAW = file_load(file_type, RAW_jsonl)
    if RAW is None:
        print(f"RAW empty: {RAW_jsonl}")

    valid_records = raw_extract(RAW, list(match_required))
    skipped_raw = len(RAW) - len(valid_records)

    PLS = file_load(file_type, PLS_jsonl)
    if PLS is None:
        print(f"PLS empty: {PLS_jsonl}")

    packet_map: Dict[Tuple[str, str, str, str], deque] = defaultdict(deque)

    for pkt in RAW:
        if any(pkt.get(k) in (None, "") for k in match_required):
            continue
        key = (str(pkt["@timestamp"]), str(pkt["sq"]), str(pkt["ak"]), str(pkt["fl"]))
        packet_map[key].append(pkt)

    results: List[Dict[str, Any]] = []
    matched_windows = 0
    total_pls_lines = 0
    no_fields_lines = 0
    miss_match_lines = 0

    for pattern in tqdm(PLS, desc="PLS 매핑 중", leave=True, ncols=80):
        window_id_from_slm = pattern.get("window_id", 0)
        label_from_slm = pattern.get("label", "Unknown")
        description_from_slm = pattern.get("description", "Unknown")

        sequence_group = []
        for pls_line in pattern.get("sequence_group", []):
            total_pls_lines += 1
            fields = pls_extract(pls_line)
            if not fields:
                no_fields_lines += 1
                continue

            key = (str(fields["@timestamp"]), str(fields["sq"]), str(fields["ak"]), str(fields["fl"]))
            dq = packet_map.get(key)
            if dq:
                pkt = dq.popleft()
                pkt_copy = pkt.copy()
                pkt_copy["match"] = "O"
                sequence_group.append(pkt_copy)
            else:
                miss_match_lines += 1

        if sequence_group:
            matched_windows += 1
            results.append({
                "window_id": window_id_from_slm,
                "label": label_from_slm,
                "description": description_from_slm,
                "sequence_group": sequence_group
            })

    print("\n=== MAPPING SUMMARY ===")
    print(f"raw_total={len(RAW)}, raw_skipped_missing_required={skipped_raw}")
    print(f"pls_total_patterns={len(PLS)}")
    print(f"pls_total_lines={total_pls_lines}, pls_no_fields={no_fields_lines}, pls_miss_match={miss_match_lines}")
    print(f"matched_windows={matched_windows}, results={len(results)}")

    if out_jsonl is not None:
        save_jsonl(results, out_jsonl)
        print(f"[SAVE] results_jsonl -> {out_jsonl} (lines={len(results)})")
    return results


if __name__ == "__main__":
    pls_path = ROOT / "data" / "PLS.jsonl"
    raw_path = ROOT / "data" / "RAW.jsonl"
    out_path = ROOT / "results" / "PLS_to_RAW_mapped.jsonl"
    PLS_to_RAW_mapping(pls_path, raw_path, out_path)
