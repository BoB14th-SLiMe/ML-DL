#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_pipeline_pattern.py

다음 세 단계를 한 번에 실행하는 파이프라인 스크립트.

1) 전처리 파라미터 생성 (common/arp/s7comm/xgt_fen 등)
   python 2.run_all_preprocess.py \
       --input "../data/ML_DL 학습.jsonl" \
       --output "../result" \
       --mode fit \
       --skip dns.py modbus.py

2) 패턴 윈도우 → 패킷 단위 feature CSV/JSONL
   python 2.window_to_feature_csv.py \
       --input "../data/pattern_windows.jsonl" \
       --pre_dir "../result" \
       --output "../result/pattern_features.csv"

3) 패턴 윈도우 → 고정 길이(T = max_index) 시퀀스 feature JSONL
   python 3.window_to_feature_csv_dynamic_index.py \
       --input "../data/pattern_windows.jsonl" \
       --pre_dir "../result" \
       --output "../../train/data/pattern_features.csv" \
       --max-index <T>

여기서 사용자가 건드릴 옵션은 --max-index 하나뿐이다.
나머지 경로/옵션은 위에 하드코딩해 둔다.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd=None):
    """단일 명령 실행 유틸"""
    print("\n[▶] 실행:", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"[❌] 실패 (exit code={result.returncode})")
        sys.exit(result.returncode)
    print("[✅] 완료")


def main():
    parser = argparse.ArgumentParser(
        description="ML/DL 학습용 패턴 전처리 파이프라인 실행기"
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=8,
        help="window_to_feature_csv_dynamic_index.py 의 --max-index (window_size T)",
    )
    args = parser.parse_args()

    # 이 파일 기준 디렉토리
    base_dir = Path(__file__).parent

    # --- 공통 경로 (질문에서 준 그대로 하드코딩) ---
    ml_dl_json = base_dir / "../data/ML_DL 학습.jsonl"
    pre_dir = base_dir / "../result"
    pattern_windows = base_dir / "../data/pattern_windows.jsonl"
    static_feature_csv = pre_dir / "pattern_features.csv"
    dynamic_feature_out = base_dir / "../../train/data/pattern_features.csv"

    # python 실행기 (현재 파이썬 그대로 사용)
    py = sys.executable

    # ---------------------------------------------------
    # 1단계: 모든 프로토콜 전처리 (dns.py, modbus.py만 skip)
    # ---------------------------------------------------
    cmd1 = [
        py,
        str(base_dir / "1.run_all_preprocess.py"),
        "--input",
        str(ml_dl_json),
        "--output",
        str(pre_dir),
        "--mode",
        "fit",
    ]
    run_cmd(cmd1)

    # ---------------------------------------------------
    # 3단계: 패턴 윈도우 → 고정 길이(T=max_index) 시퀀스 feature JSONL
    #      (여기서만 --max-index 를 사용자가 변경)
    # ---------------------------------------------------
    cmd3 = [
        py,
        str(base_dir / "3.window_to_feature_csv_dynamic_index.py"),
        "--input",
        str(pattern_windows),
        "--pre_dir",
        str(pre_dir),
        "--output",
        str(dynamic_feature_out),
        "--max-index",
        str(args.max_index),
    ]
    run_cmd(cmd3)

    print("\n🎉 전체 파이프라인 완료!")
    print(f"  - 전처리 파라미터 디렉토리 : {pre_dir}")
    print(f"  - 패킷 단위 feature CSV   : {static_feature_csv}")
    print(f"  - T={args.max_index} 시퀀스 JSONL 기준경로 : {dynamic_feature_out.with_suffix('.jsonl')}")


if __name__ == "__main__":
    main()

"""
python 0.run_pipeline_pattern.py --max-index 16
"""