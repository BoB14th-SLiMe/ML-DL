# import json
# import re
# import logging
# import orjson
# import argparse, yaml
# from tqdm.contrib.logging import logging_redirect_tqdm
# from pathlib import Path
# from tqdm import tqdm

# # ============================================================
# # Logging 설정
# # ============================================================
# def setup_logging(log_file: Path):
#     log_file.parent.mkdir(parents=True, exist_ok=True)

#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s [%(levelname)s] %(message)s",
#         handlers=[
#             logging.FileHandler(log_file, encoding="utf-8"),
#             logging.StreamHandler()  # 콘솔 출력용
#         ]
#     )


# # ============================================================
# # JSONL 파일 로드
# # ============================================================
# def load_jsonl(file_path: Path, desc: str):
#     if not file_path.exists():
#         raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
#     data = []
#     with open(file_path, "r", encoding="utf-8-sig") as fin:
#         for line in tqdm(fin, desc=desc, leave=False, ncols=90):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 data.append(orjson.loads(line))
#             except orjson.JSONDecodeError:
#                 logging.warning(f"JSON 파싱 오류: {line[:80]}...")
#     logging.info(f"{file_path.name}: {len(data):,}개 로드 완료")
#     return data


# # ============================================================
# # <flow ...> 문자열에서 timestamp 추출 (안정형)
# # ============================================================
# def extract_fields(flow):
#     if isinstance(flow, dict):
#         ts = flow.get("@timestamp") or flow.get("timestamp")
#         return {"timestamp": ts} if ts else None

#     if isinstance(flow, str) and flow:
#         ts_match = re.search(r'timestamp[=:]"?([\w\-\:\.TZ]+)"?', flow)
#         if ts_match:
#             return {"timestamp": ts_match.group(1)}

#     return None


# # ============================================================
# # SLM <-> RAW 매핑 (timestamp 기준)
# # ============================================================
# def slm_timestamp_mapping_pipeline(PLS_jsonl: Path, RAW_jsonl: Path):
#     PLS = load_jsonl(PLS_jsonl, "PLS 패턴 로딩 중")
#     RAW = load_jsonl(RAW_jsonl, "RAW 패킷 로딩 중")
#     results = []

#     packet_map = {}
#     for pkt in RAW:
#         ts = pkt.get("@timestamp") or pkt.get("timestamp")
#         if ts:
#             packet_map[ts] = pkt

#     logging.info(f"패킷 인덱스 구성 완료: {len(packet_map):,}개 키")

#     for pattern in tqdm(PLS, desc="Global 매핑 중", leave=False, ncols=90):
#         win_id = pattern.get("window_id", 0)
#         label_from_slm = pattern.get("label", "Unknown")

#         window_group = []

#         for flow in pattern.get("sequence_group", []):
#             fields = extract_fields(flow)
#             if not fields:
#                 continue

#             ts = fields["timestamp"] or fields["@timestamp"]
#             pkt = packet_map.get(ts)
#             if pkt:
#                 pkt_copy = pkt.copy()
#                 pkt_copy["match"] = "O"
#                 window_group.append(pkt_copy)

#         if window_group:
#             results.append({
#                 "window_id": win_id,
#                 "label": label_from_slm,
#                 "window_group": window_group
#             })

#     return results


# # ============================================================
# # 진입점 함수
# # ============================================================
# def PLS_to_Raw(config: dict):
#     paths_cfg = config["pipeline"]
#     log_file = Path(paths_cfg["1-1_log_file"])
#     setup_logging(log_file)
#     logging.info("### PLS_to_Raw 시작 ###")

#     with logging_redirect_tqdm():

#         PLS_file = Path(paths_cfg["PLS_file"])
#         RAW_file = Path(paths_cfg["RAW_file"])
#         result_file = Path(paths_cfg["1-1_result_file"])

#         logging.info(f"PLS 데이터 위치: {PLS_file}")
#         logging.info(f"RAW 데이터 위치: {RAW_file}")
#         logging.info(f"결과 위치: {result_file}")

#         results = slm_timestamp_mapping_pipeline(PLS_file, RAW_file)
#         total = len(results)

#     logging.info(f"### PLS_to_Raw 결과 ###")
#     logging.info(f"총 {total:,}개 윈도우 매핑 완료")

#     result_file.parent.mkdir(parents=True, exist_ok=True)
#     with open(result_file, "wb") as fout:
#         for r in results:
#             fout.write(orjson.dumps(r) + b"\n")

#     logging.info(f"저장 완료 → {result_file.resolve()}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-c", "--config", type=str, required=True)
#     args = parser.parse_args()

#     with open(args.config, "r", encoding="utf-8") as f:
#         config = yaml.safe_load(f)
#     PLS_to_Raw(config)

# """
# usage: 
# python "1-1.PLS_to_Raw(timestamp).py" -c "../config/1.PLS_to_Raw.yaml"
# """


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1-1.PLS_to_Raw(timestamp).py

PLS(JSONL) 윈도우 결과와 RAW(JSONL) 패킷을
timestamp 기준으로 매핑한다.

※ 변경 사항
- RAW에서 읽은 timestamp를 datetime으로 파싱한 뒤,
  **9시간을 빼서(UTC+9 → UTC 기준으로 보정)** PLS와 비교하도록 변경.
"""

import json
import re
import logging
import orjson
import argparse, yaml
from datetime import datetime, timedelta
from tqdm.contrib.logging import logging_redirect_tqdm
from pathlib import Path
from tqdm import tqdm

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
            logging.StreamHandler()  # 콘솔 출력용
        ]
    )


# ============================================================
# JSONL 파일 로드
# ============================================================
def load_jsonl(file_path: Path, desc: str):
    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    data = []
    with open(file_path, "r", encoding="utf-8-sig") as fin:
        for line in tqdm(fin, desc=desc, leave=False, ncols=90):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(orjson.loads(line))
            except orjson.JSONDecodeError:
                logging.warning(f"JSON 파싱 오류: {line[:80]}...")
    logging.info(f"{file_path.name}: {len(data):,}개 로드 완료")
    return data


# ============================================================
# timestamp 문자열 → datetime 변환 유틸
#   - "2025-09-22T02:52:00.727524Z"
#   - "2025-09-22T02:52:00.727524+00:00"
#   - "2025-09-22T02:52:00"
# ============================================================
def parse_ts_to_dt(ts: str) -> datetime:
    """
    ISO8601 스타일 timestamp 문자열을 datetime으로 변환.
    'Z'가 붙어 있으면 +00:00 으로 간주.
    """
    s = ts.strip()
    # Z(UTC) → +00:00 으로 치환
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        # 파싱 실패 시 None 대신 예외를 올리는 편이 디버깅에 유리
        raise


# ============================================================
# <flow ...> 문자열/객체에서 timestamp 추출 (안정형)
# ============================================================
def extract_fields(flow):
    # dict인 경우: @timestamp / timestamp 중 하나 사용
    if isinstance(flow, dict):
        ts = flow.get("@timestamp") or flow.get("timestamp")
        return {"timestamp": ts} if ts else None

    # 문자열인 경우: timestamp="..." 패턴에서 추출
    if isinstance(flow, str) and flow:
        ts_match = re.search(r'timestamp[=:]"?([\w\-\:\.TZ\+]+)"?', flow)
        if ts_match:
            return {"timestamp": ts_match.group(1)}

    return None


# ============================================================
# SLM <-> RAW 매핑 (timestamp 기준)
#   - RAW timestamp는 9시간 감소 후 비교
# ============================================================
def slm_timestamp_mapping_pipeline(PLS_jsonl: Path, RAW_jsonl: Path):
    PLS = load_jsonl(PLS_jsonl, "PLS 패턴 로딩 중")
    RAW = load_jsonl(RAW_jsonl, "RAW 패킷 로딩 중")
    results = []

    # 1) RAW 패킷 인덱스 구성
    #    RAW 쪽 timestamp를 datetime으로 파싱 후 9시간 빼서 key로 사용
    packet_map = {}
    skipped_raw = 0
    for pkt in RAW:
        ts = pkt.get("@timestamp") or pkt.get("timestamp")
        if not ts:
            skipped_raw += 1
            continue
        try:
            dt = parse_ts_to_dt(ts) - timedelta(hours=0)  # ★ 여기서 -9시간 보정
            packet_map[dt] = pkt
        except Exception:
            skipped_raw += 1
            continue

    logging.info(f"패킷 인덱스 구성 완료: {len(packet_map):,}개 키 (RAW 스킵 {skipped_raw:,}개)")

    # 2) PLS 윈도우별로 timestamp 매핑 시도
    matched_windows = 0
    for pattern in tqdm(PLS, desc="Global 매핑 중", leave=False, ncols=90):
        win_id = pattern.get("window_id", 0)
        label_from_slm = pattern.get("label", "Unknown")

        window_group = []

        for flow in pattern.get("sequence_group", []):
            fields = extract_fields(flow)
            if not fields:
                continue

            ts = fields["timestamp"]
            if not ts:
                continue

            try:
                dt_pls = parse_ts_to_dt(ts)  # PLS는 그대로 (보정 없이) 파싱
            except Exception:
                continue

            pkt = packet_map.get(dt_pls)
            if pkt:
                pkt_copy = pkt.copy()
                pkt_copy["match"] = "O"
                window_group.append(pkt_copy)

        if window_group:
            matched_windows += 1
            results.append({
                "window_id": win_id,
                "label": label_from_slm,
                "window_group": window_group
            })

    logging.info(f"윈도우 매핑 성공: {matched_windows:,}개 / 전체 {len(PLS):,}개")

    return results


# ============================================================
# 진입점 함수
# ============================================================
def PLS_to_Raw(config: dict):
    paths_cfg = config["pipeline"]
    log_file = Path(paths_cfg["1-1_log_file"])
    setup_logging(log_file)
    logging.info("### PLS_to_Raw 시작 ###")

    with logging_redirect_tqdm():

        PLS_file = Path(paths_cfg["PLS_file"])
        RAW_file = Path(paths_cfg["RAW_file"])
        result_file = Path(paths_cfg["1-1_result_file"])

        logging.info(f"PLS 데이터 위치: {PLS_file}")
        logging.info(f"RAW 데이터 위치: {RAW_file}")
        logging.info(f"결과 위치: {result_file}")

        results = slm_timestamp_mapping_pipeline(PLS_file, RAW_file)
        total = len(results)

    logging.info(f"### PLS_to_Raw 결과 ###")
    logging.info(f"총 {total:,}개 윈도우 매핑 완료")

    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "wb") as fout:
        for r in results:
            fout.write(orjson.dumps(r) + b"\n")

    logging.info(f"저장 완료 → {result_file.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    PLS_to_Raw(config)

"""
usage:
python "1-1.PLS_to_Raw(timestamp).py" -c "../config/1.PLS_to_Raw.yaml"

"""
