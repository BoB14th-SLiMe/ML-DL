# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""
2.PLS_to_RAW_windows.py

SLM이 패턴 분석에 사용한 전체 windows의 데이터는 PLS로 이루어져 있다. 이를 원본 데이터 RAW로 변경한다.
PLS(JSONL) 결과와 RAW(JSONL) 패킷을 다음 기준으로 매핑한다.
  - @timestamp
  - sq
  - ak
  - fl
"""
import re
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.file_load import file_load
from utils.extract_feature import raw_extract, timestamp_extract
from utils.file_save import save_jsonl


def load_window_pls(window_pls_file: Path) -> Dict[int, Dict[str, Any]]:
    data = file_load("jsonl", str(window_pls_file)) or []
    result: Dict[int, Dict[str, Any]] = {}

    for obj in tqdm(data, desc="window_pls 로딩", ncols=50):
        if not isinstance(obj, dict):
            continue

        win_id = obj.get("window_id")
        if win_id is None:
            continue

        flows_info: List[Dict[str, Any]] = []
        for flow in (obj.get("pls") or []):
            ts_str = timestamp_extract(flow)
            if not ts_str:
                continue

            flows_info.append({"ts": ts_str})

        result[int(win_id)] = {"flows": flows_info}

    print(f"총 window 개수: {len(result):,}")
    return result


def load_raw_packets(raw_file: Path) -> Dict[datetime, Dict[str, Any]]:
    raw_list = file_load("jsonl", str(raw_file)) or []

    valid_records = raw_extract(raw_list, ["@timestamp"])
    skipped_missing_ts = len(raw_list) - len(valid_records)

    ts_map: Dict[datetime, Dict[str, Any]] = {}
    skipped_parse = 0

    for obj in tqdm(raw_list, desc="RAW 패킷 인덱스 로딩", ncols=50):
        if not isinstance(obj, dict):
            continue
        ts_str = obj.get("@timestamp")
        if not ts_str:
            continue
        ts_map[ts_str] = obj

    print(f"RAW 패킷 개수: {len(ts_map):,} (raw_extract 누락 {skipped_missing_ts:,}개, ts 파싱 실패 {skipped_parse:,}개)")
    return ts_map


def PLS_to_RAW_mapping(PLS_jsonl: Path, RAW_jsonl: Path, out_jsonl: Path):
    window_pls = load_window_pls(PLS_jsonl)
    raw_ts_map = load_raw_packets(RAW_jsonl)

    results: List[Dict[str, Any]] = []
    total = len(window_pls)
    full_matched = 0
    dropped = 0

    for win_id, info in tqdm(window_pls.items(), desc="윈도우 매핑 중", ncols=90):
        flows = info.get("flows", [])
        if not flows:
            dropped += 1
            continue

        window_group: List[Dict[str, Any]] = []
        ok = True

        for f in flows:
            dt_pls = f["ts"]
            pkt = raw_ts_map.get(dt_pls)
            if not pkt:
                ok = False
                break
            pkt_copy = dict(pkt)
            pkt_copy["match"] = "O"
            window_group.append(pkt_copy)

        if ok and len(window_group) == len(flows):
            full_matched += 1
            results.append({"window_id": win_id, "sequence_group": window_group})
        else:
            dropped += 1

    print(f"총 윈도우 {total:,}개 중 완전 매핑 {full_matched:,}개, 드롭 {dropped:,}개")

    if out_jsonl is not None:
        save_jsonl(results, out_jsonl)
        print(f"결과 저장 : {out_jsonl} (lines={len(results)})")

    return results


if __name__ == "__main__":
    pls_path = ROOT / "data" / "window_pls_80.jsonl"
    raw_path = ROOT / "data" / "RAW.jsonl"
    out_path = ROOT / "results" / "PLS_to_RAW_windows_mapped.jsonl"
    PLS_to_RAW_mapping(pls_path, raw_path, out_path)
