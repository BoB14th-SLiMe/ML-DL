import json
import re
import logging
import orjson
import argparse, yaml
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
# <flow ...> 문자열에서 timestamp, sq 추출 (안정형)
# ============================================================
def extract_fields(flow):
    if isinstance(flow, dict):
        ts = flow.get("@timestamp") or flow.get("timestamp")
        sq = flow.get("sq")
        return {"timestamp": ts, "sq": sq} if ts else None
    
    if isinstance(flow, str) and flow:
        ts_match = re.search(r'timestamp[=:]"?([\w\-\:\.TZ]+)"?', flow)
        sq_match = re.search(r'sq[=:]"?(\d+)"?', flow)

        ts = ts_match.group(1) if ts_match else None
        sq = int(sq_match.group(1)) if sq_match else None

        if ts:
            return {"timestamp": ts, "sq": sq}

    return None


# ============================================================
# SLM <-> RAW 매핑 (timestamp 기준)
# ============================================================
def slm_timestamp_mapping_pipeline(PLS_jsonl: Path, RAW_jsonl: Path):
    PLS = load_jsonl(PLS_jsonl, "PLS 패턴 로딩 중")
    RAW = load_jsonl(RAW_jsonl, "RAW 패킷 로딩 중")
    results = []

    packet_map = {}
    for pkt in RAW:
        ts = pkt.get("@timestamp")
        sq = pkt.get("sq")
        if ts is not None and sq is not None:
            packet_map[(ts, sq)] = pkt

    logging.info(f"패킷 인덱스 구성 완료: {len(packet_map):,}개 키")

    for pattern in tqdm(PLS, desc="[3/3] Global 매핑 중", leave=False, ncols=90):
        win_id = pattern.get("window_id", 0)
        label_from_slm = pattern.get("label", "Unknown")

        window_group = []

        for flow in pattern.get("sequence_group", []):
            fields = extract_fields(flow)
            if not fields:
                continue

            ts = fields["timestamp"]
            sq = fields.get("sq")
            pkt = packet_map.get((ts, sq))
            if pkt:
                pkt_copy = pkt.copy()
                pkt_copy["match"] = "O"
                window_group.append(pkt_copy)

        if window_group:
            results.append({
                "window_id": win_id,
                "label": label_from_slm,
                "window_group": window_group
            })

    return results


# ============================================================
# 진입점 함수
# ============================================================
def PLS_to_Raw(config: dict):
    paths_cfg = config["pipeline"]
    log_file = Path(paths_cfg["1-2_log_file"])
    setup_logging(log_file)
    logging.info("### PLS_to_Raw 시작 ###")

    with logging_redirect_tqdm():

        PLS_file = Path(paths_cfg["PLS_file"])
        RAW_file = Path(paths_cfg["RAW_file"])
        result_file = Path(paths_cfg["1-2_result_file"])

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
usage: python "1-2.PLS_to_Raw(timestamp_sq).py" -c "../config/1.PLS_to_Raw.yaml"
"""