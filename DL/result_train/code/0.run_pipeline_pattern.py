#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
00.run_pipeline_pattern.py

다음 4단계를 한 번에 실행하는 파이프라인 스크립트.

1) 0.attack_result.py
   - attack.jsonl → 슬라이딩 윈도우 → FC=6 포함 여부로 GT 라벨 생성
   - 출력: ../result/attack_result.csv

2) 1.benchmark.py
   - attack.jsonl → 전처리 + 윈도우 feature + LSTM-AE MSE 계산
   - 출력:
       ../result/benchmark/X_windows.npy
       ../result/benchmark/windows_meta.jsonl
       ../result/benchmark/window_scores.csv

3) 2.eval_detection_metrics.py
   - GT CSV + 예측 CSV → detection metric 계산
   - 출력: ../result/eval_detection_metrics.json

4) 3.analyze_mse_dist.py
   - GT CSV + 예측 CSV → MSE 분포/요약 통계
   - 출력: ../result/analyze_mse_dist.json

사용자가 지정하는 것:
  --window-size  (0,1단계 둘 다 동일하게 사용)
  --step-size    (0,1단계 둘 다 동일하게 사용)
  --threshold    (1단계 benchmark에서 MSE threshold로 사용; None이면 threshold.json 활용)

경로는 다음 기준으로 고정되어 있음 (이 스크립트가 있는 디렉토리 기준):
  attack_jsonl      : ../data/attack.jsonl
  attack_result_csv : ../result/attack_result.csv
  benchmark_out_dir : ../result/benchmark
  window_scores_csv : ../result/benchmark/window_scores.csv
  eval_metrics_json : ../result/eval_detection_metrics.json
  analyze_json      : ../result/analyze_mse_dist.json
  pre_dir           : ../../preprocessing/result
  model_dir         : ../data
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd: Path):
    """subprocess.run 래퍼 (실패 시 바로 종료)."""
    print("\n[RUN] ", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        print(f"[ERROR] 명령 실패 (exit code={result.returncode}): {' '.join(cmd)}")
        sys.exit(result.returncode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--window-size", "-w",
        type=int,
        required=True,
        help="슬라이딩 윈도우 크기 (예: 8, 76, 80 등)",
    )
    p.add_argument(
        "--step-size", "-s",
        type=int,
        default=None,
        help="슬라이딩 stride (기본: window-size와 동일 → non-overlap)",
    )
    p.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help=(
            "benchmark 단계에서 사용할 MSE threshold.\n"
            "지정하면 1.benchmark.py에 --threshold로 그대로 전달.\n"
            "지정하지 않으면 threshold.json(학습 시 저장한 값)을 사용하거나, "
            "threshold 없는 상태로 window_scores.csv 생성."
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    # 이 스크립트가 있는 디렉토리 (0,1,2,3번 스크립트도 여기 있다고 가정)
    here = Path(__file__).resolve().parent

    # 고정 경로 (여기 기준 상대 경로)
    attack_jsonl = here / "../data/attack.jsonl"
    attack_result_csv = here / "../result/attack_result.csv"
    pre_dir = here / "../../preprocessing/result"
    benchmark_out_dir = here / "../result/benchmark"
    window_scores_csv = benchmark_out_dir / "window_scores.csv"
    eval_metrics_json = here / "../result/eval_detection_metrics.json"
    analyze_json = here / "../result/analyze_mse_dist.json"
    model_dir = here / "../data"

    window_size = args.window_size
    step_size = args.step_size if args.step_size is not None else window_size
    threshold = args.threshold

    print("========== 00.run_pipeline_pattern.py ==========")
    print(f"[INFO] window_size = {window_size}")
    print(f"[INFO] step_size   = {step_size} (None이면 window_size와 동일)")
    print(f"[INFO] threshold   = {threshold if threshold is not None else 'None (threshold.json 또는 -1 사용)'}")
    print("================================================")

    # 1) 0.attack_result.py
    #    attack.jsonl → attack_result.csv (GT 라벨)
    cmd0 = [
        sys.executable,
        "0.attack_result.py",
        "--input", str(attack_jsonl),
        "--window-size", str(window_size),
        "--step-size", str(step_size),
        "--output", str(attack_result_csv),
    ]
    run_cmd(cmd0, cwd=here)

    # 2) 1.benchmark.py
    #    attack.jsonl → 윈도우 feature + LSTM-AE MSE 계산
    cmd1 = [
        sys.executable,
        "1.benchmark.py",
        "--input", str(attack_jsonl),
        "--pre-dir", str(pre_dir),
        "--window-size", str(window_size),
        "--step-size", str(step_size),
        "--output-dir", str(benchmark_out_dir),
        "--model-dir", str(model_dir),
        "--batch-size", "128",          # 고정 (요청대로 CLI로는 expose 안 함)
    ]
    if threshold is not None:
        cmd1.extend(["--threshold", str(threshold)])

    run_cmd(cmd1, cwd=here)

    # 3) 2.eval_detection_metrics.py
    #    attack_result.csv + window_scores.csv → detection metrics
    cmd2 = [
        sys.executable,
        "2.eval_detection_metrics.py",
        "--attack-csv", str(attack_result_csv),
        "--pred-csv", str(window_scores_csv),
        "--output-json", str(eval_metrics_json),
        # 필요하면 --ignore-pred-minus1 옵션을 여기 추가해서 고정할 수도 있음
        # "--ignore-pred-minus1",
    ]
    run_cmd(cmd2, cwd=here)

    # 4) 3.analyze_mse_dist.py
    #    attack_result.csv + window_scores.csv → MSE 통계
    cmd3 = [
        sys.executable,
        "3.analyze_mse_dist.py",
        "--attack-csv", str(attack_result_csv),
        "--pred-csv", str(window_scores_csv),
        "--output-json", str(analyze_json),
    ]
    run_cmd(cmd3, cwd=here)

    print("\n[INFO] 파이프라인 완료 🎉")
    print(f"  - GT CSV               : {attack_result_csv}")
    print(f"  - Benchmark dir        : {benchmark_out_dir}")
    print(f"  - window_scores.csv    : {window_scores_csv}")
    print(f"  - Detection metrics    : {eval_metrics_json}")
    print(f"  - MSE dist summary     : {analyze_json}")


if __name__ == "__main__":
    main()

"""
python 0.run_pipeline_pattern.py --window-size 8 --step-size 4 --threshold 100

"""