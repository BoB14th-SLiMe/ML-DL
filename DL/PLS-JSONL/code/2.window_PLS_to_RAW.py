#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1-2.window_pls_to_Raw(timestamp).py

목표:
  window_pls_80.jsonl 의 각 window 에 대해,
  RAW 패킷 JSONL(예: attack.jsonl / normal.jsonl)의 timestamp 와 매칭하여

  {
    "window_id": 1,
    "window_group": [
      { "@timestamp": "...", ..., "match": "O" },
      ...
    ]
  }

형태의 JSONL(1-2_window_pls_to_Raw_result.jsonl 같은 것)을 생성한다.

제약:
  - label / pattern 은 결과 JSONL 에 포함하지 않는다.
  - 한 window 의 모든 flow 가 RAW 와 매핑되지 않으면
    → 해당 window 는 통째로 버린다(부분 매핑 허용 X).

※ 타임존 이슈 대응
- RAW 의 timestamp 가 한국 시간(+09:00)이고,
- window_pls_80.jsonl 의 timestamp 가 UTC 라고 가정하면,
  RAW timestamp 를 datetime 으로 파싱한 뒤 9시간을 빼서(−9h) window_pls 와 비교한다.
"""

import re
import orjson
import yaml
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


# ============================================================
# Logging 설정
# ============================================================
def setup_logging(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ],
        force=True,
    )


# ============================================================
# 공통: timestamp 파싱
# ============================================================
def parse_ts(ts: str) -> datetime:
    """
    ISO8601 style timestamp → datetime
    예:
      - "2025-09-22T02:52:00.727524Z"
      - "2025-09-22T02:52:00.727524+00:00"
      - "2025-09-22T02:52:00.727524"
    """
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


# ============================================================
# window_pls_80.jsonl 의 flow 문자열에서 timestamp 추출
# ============================================================
def extract_timestamp_from_flow(flow: str) -> Optional[str]:
    """
    window_pls_80.jsonl 의 "pls" 요소인 문자열에서 timestamp 추출.

    우선순위:
      1) <flow timestamp="...">
      2) "@timestamp: ..." 패턴
    """
    if not isinstance(flow, str):
        return None

    m = re.search(r'timestamp="([\w\-\:\.TZ\+]+)"', flow)
    if m:
        return m.group(1)

    m2 = re.search(r'@timestamp:\s*([\w\-\:\.TZ\+]+)', flow)
    if m2:
        return m2.group(1)

    return None


# ============================================================
# window_pls_80.jsonl 로드
# ============================================================
def load_window_pls(window_pls_file: Path) -> Dict[int, Dict[str, Any]]:
    """
    window_pls_80.jsonl → window_id 기준으로 정리

    입력 라인 예:
      {
        "window_id": 1,
        "label": "P_0021",
        "pls": [ "<flow timestamp=\"...\">@timestamp: ...</flow>", ... ]
      }

    반환 구조:
      {
        window_id: {
          "pattern": "P_0021" or None,
          "flows": [
             {"ts": datetime, "ts_str": "2025-09-22T02:52:00.742511Z", "raw": <원본문자열>},
             ...
          ]
        },
        ...
      }
    """
    result: Dict[int, Dict[str, Any]] = {}

    with window_pls_file.open("r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="window_pls_80 로딩", ncols=90):
            line = line.strip()
            if not line:
                continue
            obj = orjson.loads(line)

            win_id = obj.get("window_id")
            if win_id is None:
                continue

            pattern = obj.get("pattern") or obj.get("label")

            flows_info: List[Dict[str, Any]] = []
            for flow in obj.get("pls", []):
                ts_str = extract_timestamp_from_flow(flow)
                if not ts_str:
                    continue
                try:
                    dt = parse_ts(ts_str)
                except Exception:
                    continue
                flows_info.append({"ts": dt, "ts_str": ts_str, "raw": flow})

            result[win_id] = {
                "pattern": pattern,
                "flows": flows_info,
            }

    logging.info(f"window_pls_80 윈도우 수: {len(result):,}")
    return result


# ============================================================
# RAW(JSONL: attack.jsonl / normal.jsonl 등) 로드
# ============================================================
def load_raw_packets(raw_file: Path) -> Dict[datetime, Dict[str, Any]]:
    """
    RAW 패킷 JSONL 파일 로드 후 timestamp 인덱스 생성.

    입력 라인 예:
      {
        "@timestamp": "2025-09-22T11:52:00.742511Z",
        "protocol": "tcp",
        ...
      }

    가정:
      - RAW timestamp 는 KST(+09:00) 기준
      - window_pls_80 의 timestamp 는 UTC 기준

    처리:
      - ts_raw = RAW timestamp (KST)
      - ts_key = ts_raw - 9h  (UTC 로 보정)
      - 인덱스 key 에 ts_key 를 사용

    반환:
      {
        ts_key(datetime, UTC): pkt(dict),
        ...
      }
    """
    ts_map: Dict[datetime, Dict[str, Any]] = {}
    skipped = 0

    with raw_file.open("r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="RAW 패킷 로딩", ncols=90):
            line = line.strip()
            if not line:
                continue
            obj = orjson.loads(line)
            ts_str = obj.get("@timestamp") or obj.get("timestamp")
            if not ts_str:
                skipped += 1
                continue
            try:
                dt_raw = parse_ts(ts_str)
                dt_key = dt_raw - timedelta(hours=9)  # ★ 9시간 보정
            except Exception:
                skipped += 1
                continue

            # 동일한 ts_key 가 여러 개면 마지막 것으로 덮어쓴다
            ts_map[dt_key] = obj

    logging.info(f"RAW 패킷 인덱스 수: {len(ts_map):,} (스킵 {skipped:,}개)")
    return ts_map


# ============================================================
# window_pls_80 ↔ RAW 매핑
# ============================================================
def map_window_pls_to_raw(
    window_pls: Dict[int, Dict[str, Any]],
    raw_ts_map: Dict[datetime, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    window_id 단위로:
      - window_pls_80 의 flows[].ts 를 기준으로
      - RAW ts_map 에서 동일 timestamp 를 가진 패킷을 찾는다.
      - 매칭된 RAW 패킷들을 window_group 에 넣고, match="O" 를 붙인다.

    조건:
      - 해당 window 의 모든 flow 가 RAW 와 1:1 매칭되어야 한다.
      - 하나라도 매칭 실패 시 → 그 window 는 결과에서 제외.

    반환 리스트 요소 예:
      {
        "window_id": 1,
        "window_group": [ { "@timestamp": "...", ..., "match": "O" }, ... ]
      }
    """
    results: List[Dict[str, Any]] = []
    total = len(window_pls)
    full_matched = 0
    partial_or_empty = 0

    for win_id, info in tqdm(window_pls.items(), desc="윈도우 매핑 중", ncols=90):
        flows = info.get("flows", [])
        if not flows:
            partial_or_empty += 1
            continue

        window_group: List[Dict[str, Any]] = []

        for f in flows:
            dt_pls = f["ts"]
            pkt = raw_ts_map.get(dt_pls)
            if not pkt:
                # 하나라도 매칭 실패하면 전체 윈도우 drop 할 것이므로
                window_group = []
                break

            pkt_copy = dict(pkt)
            pkt_copy["match"] = "O"
            window_group.append(pkt_copy)

        # 전부 매핑된 경우에만 결과에 포함
        if window_group and len(window_group) == len(flows):
            full_matched += 1
            results.append({
                "window_id": win_id,
                "RAW": window_group,
            })
        else:
            partial_or_empty += 1

    logging.info(
        f"총 윈도우 {total:,}개 중 "
        f"완전 매핑 {full_matched:,}개, 부분/실패 {partial_or_empty:,}개 (버림)"
    )
    return results


# ============================================================
# 파이프라인 (YAML config 사용)
# ============================================================
def window_pls_to_raw_pipeline(config: Dict[str, Any]):
    paths_cfg = config["pipeline"]

    log_file = Path(paths_cfg["1-2_log_file"])
    window_pls_file = Path(paths_cfg["window_pls_file"])
    raw_file = Path(paths_cfg["RAW_file"])
    result_file = Path(paths_cfg["1-2_result_file"])

    setup_logging(log_file)
    logging.info("### 1-2.window_pls_to_Raw 시작 ###")

    logging.info(f"window_pls_80 파일: {window_pls_file}")
    logging.info(f"RAW 패킷 파일   : {raw_file}")
    logging.info(f"결과 저장 경로   : {result_file}")

    with logging_redirect_tqdm():
        window_pls = load_window_pls(window_pls_file)
        raw_ts_map = load_raw_packets(raw_file)
        mapped_windows = map_window_pls_to_raw(window_pls, raw_ts_map)

    # 저장
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("wb") as fout:
        for r in mapped_windows:
            fout.write(orjson.dumps(r) + b"\n")

    logging.info(f"저장 완료 → {result_file.resolve()}")
    logging.info("### 1-2.window_pls_to_Raw 종료 ###")


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="YAML 설정 파일 경로")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    window_pls_to_raw_pipeline(cfg)

"""
python "2.window_PLS_to_RAW.py" -c "../config/2.window_PLS_to_RAW.yaml"
"""
