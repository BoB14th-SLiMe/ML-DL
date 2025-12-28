#!/usr/bin/env python3
import os
import json
import orjson
import yaml
import logging
from pathlib import Path
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
        force=True
    )


# ============================================================
# JSONL 파일 확인
# ============================================================
def check_jsonl(path: Path):
    """
    • 파일 존재 확인
    • 비어있는지 확인
    • JSONL 파싱되는지 확인
    • 라인 수 카운트
    """
    if not path.exists():
        logging.error(f"❌ 파일 없음: {path}")
        return False

    if path.stat().st_size == 0:
        logging.error(f"❌ 파일이 비어 있음: {path}")
        return False

    valid_lines = 0
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in tqdm(f, desc=f"검증 중: {path.name}", ncols=90):
            line = line.strip()
            if not line:
                continue
            try:
                orjson.loads(line)
                valid_lines += 1
            except orjson.JSONDecodeError:
                logging.error(f"❌ JSON 파싱 오류 발생 → {path}")
                logging.error(f"문제 라인: {line[:100]}")
                return False

    logging.info(f"✔ JSONL 정상: {path} (총 {valid_lines:,} 라인)")
    return True


# ============================================================
# 디렉토리 확인
# ============================================================
def check_directory(path: Path):
    if not path.exists():
        logging.warning(f"📁 디렉토리가 없어 생성합니다 → {path}")
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"✔ 디렉토리 생성 완료: {path}")
        return True

    if not path.is_dir():
        logging.error(f"❌ 디렉토리 아님: {path}")
        return False

    logging.info(f"✔ 디렉토리 OK: {path}")
    return True



# ============================================================
# 0.prepare 메인 함수
# ============================================================
def prepare_check(config: dict):
    paths_cfg = config["pipeline"]

    # 로그 파일 설정
    log_file = Path(paths_cfg["prepare_log"])
    setup_logging(log_file)

    logging.info("### 0.prepare: 데이터 준비 상태 점검 시작 ###")

    all_ok = True

    # 🔥 1) 필수 경로 목록
    required_paths = {
        "PLS_file": Path(paths_cfg["PLS_file"]),
        "RAW_file": Path(paths_cfg["RAW_file"]),
    }

    # 🔥 2) 결과 저장 디렉토리 확인
    result_dir = Path(paths_cfg["result_dir"])
    required_dirs = {
        "result_dir": result_dir,
    }

    # 🔥 3) 디렉토리 검사
    for name, p in required_dirs.items():
        if not check_directory(p):
            all_ok = False

    # 🔥 4) 파일 검사
    for name, p in required_paths.items():
        logging.info(f"\n🔍 검사 대상: {name} = {p}")

        if name.endswith("_file"):
            if not check_jsonl(p):
                all_ok = False
        else:
            if not p.exists():
                logging.error(f"❌ 파일 없음: {p}")
                all_ok = False
            else:
                logging.info(f"✔ 파일 OK: {p}")

    if all_ok:
        logging.info("\n🎉 모든 준비 완료! 0.prepare PASS")
    else:
        logging.error("\n❌ 0.prepare 실패: 누락된 데이터가 있습니다.")
        raise SystemExit("0.prepare aborted due to missing files")


# ============================================================
# CLI Entry
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    prepare_check(config)


"""
usage :
python 0.prepare.py -c "../config/0.prepare.yaml"
"""
