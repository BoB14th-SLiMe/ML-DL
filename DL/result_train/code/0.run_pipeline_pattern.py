#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
00.run_pipeline_pattern.py

여러 개의 공격/정상 JSONL 파일에 대해,
다음 4단계를 일괄 실행하는 파이프라인 스크립트.

각 입력 JSONL 파일마다 tag를 입력 파일명 stem(확장자 제거)으로 두고,
결과 파일들은 모두 *_<tag> 형식으로 저장한다.

예: input = ../data/attack_ver5_1.jsonl (tag = attack_ver5_1)
  1) 0.attack_result.py
     - attack_ver5_1.jsonl → 슬라이딩 윈도우 → GT 라벨 생성
     - 출력: ../result/attack_result_attack_ver5_1.csv

  2) 1.benchmark.py
     - attack_ver5_1.jsonl → 전처리 + 윈도우 feature + LSTM-AE MSE 계산
     - 출력:
         ../result/benchmark/X_windows_attack_ver5_1.npy
         ../result/benchmark/windows_meta_attack_ver5_1.jsonl
         ../result/benchmark/window_scores_attack_ver5_1.csv

  3) 2.eval_detection_metrics.py
     - GT CSV + 예측 CSV → detection metric 계산
     - 출력 (기본): ../result/benchmark/metrics_attack_ver5_1.json
       또는 2.eval_detection_metrics.py 내부 로직에 따라 metrics_*.json

  4) 3.analyze_mse_dist.py
     - GT CSV + 예측 CSV → MSE 분포/요약 통계
     - 출력 (기본): ../result/benchmark/analyze_mse_dist_attack_ver5_1.json

사용자가 지정하는 것:
  --inputs      : 처리할 JSONL 파일 리스트 (6개 등)
                  지정하지 않으면 기본으로 ../data/attack.jsonl 한 개만 처리
  --window-size : 0,1단계 둘 다 동일하게 사용
  --step-size   : 0,1단계 둘 다 동일하게 사용 (미지정 시 window-size와 동일)
  --threshold   : 1단계 benchmark에서 MSE threshold로 사용; None이면 threshold.json 활용

경로는 다음 기준으로 고정되어 있음 (이 스크립트가 있는 디렉토리 기준):
  기본 input (옵션 미지정 시):     ../data/attack.jsonl
  attack_result_csv(tag별)       : ../result/attack_result_<tag>.csv
  benchmark_out_dir              : ../result/benchmark
  window_scores_csv(tag별)       : ../result/benchmark/window_scores_<tag>.csv
  eval_metrics_json(tag별 기본값): ../result/benchmark/metrics_<tag>.json
  analyze_json(tag별 기본값)     : ../result/benchmark/analyze_mse_dist_<tag>.json
  pre_dir                        : ../../preprocessing/result
  model_dir                      : ../data
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
        "--inputs", "-i",
        nargs="+",
        help=(
            "처리할 패킷 JSONL 파일 경로 리스트 "
            "(예: ../data/attack_ver5.jsonl ../data/attack_ver5_1.jsonl ...). "
            "지정하지 않으면 ../data/attack.jsonl 한 개만 처리."
        ),
    )

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
            "threshold 없는 상태로 window_scores_*.csv 생성."
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    # 이 스크립트가 있는 디렉토리 (0,1,2,3번 스크립트도 여기 있다고 가정)
    here = Path(__file__).resolve().parent

    # 공통 경로 (여기 기준 상대 경로)
    default_attack_jsonl = here / "../data/attack.jsonl"
    pre_dir = here / "../../preprocessing/result"
    benchmark_out_dir = here / "../result/benchmark"
    model_dir = here / "../data"

    window_size = args.window_size
    step_size = args.step_size if args.step_size is not None else window_size
    threshold = args.threshold

    # 입력 파일 리스트 결정
    if args.inputs:
        input_paths = [Path(p) if not Path(p).is_absolute() else Path(p) for p in args.inputs]
    else:
        input_paths = [default_attack_jsonl]

    print("========== 00.run_pipeline_pattern.py ==========")
    print(f"[INFO] window_size = {window_size}")
    print(f"[INFO] step_size   = {step_size} (None이면 window_size와 동일)")
    print(f"[INFO] threshold   = {threshold if threshold is not None else 'None (threshold.json 또는 -1 사용)'}")
    print(f"[INFO] 입력 파일 수 = {len(input_paths)}")
    for idx, ip in enumerate(input_paths, start=1):
        print(f"  [{idx}] {ip}")
    print("================================================")

    # 각 입력 파일에 대해 파이프라인 전체 수행
    for idx, attack_jsonl in enumerate(input_paths, start=1):
        attack_jsonl = attack_jsonl.resolve()
        tag = attack_jsonl.stem  # 예: attack_ver5_1.jsonl -> attack_ver5_1

        print(f"\n\n===== [{idx}/{len(input_paths)}] tag={tag} 대상 파이프라인 시작 =====")
        print(f"[INFO] attack_jsonl = {attack_jsonl}")

        # 출력 경로(tag별)
        attack_result_csv = here / f"../result/attack_result_{tag}.csv"
        window_scores_csv = benchmark_out_dir / f"window_scores_{tag}.csv"
        # 2,3단계 스크립트에서 --tag 를 주면 자체 규칙에 따라 파일명 생성하므로
        # 여기서는 output-json을 명시하지 않아도 됨(원하면 직접 명시 가능).
        eval_metrics_json = benchmark_out_dir / f"metrics_{tag}.json"
        analyze_json = benchmark_out_dir / f"analyze_mse_dist_{tag}.json"

        print(f"[INFO] attack_result_csv : {attack_result_csv}")
        print(f"[INFO] benchmark_out_dir : {benchmark_out_dir}")
        print(f"[INFO] window_scores_csv : {window_scores_csv}")
        print(f"[INFO] eval_metrics_json : {eval_metrics_json}")
        print(f"[INFO] analyze_json      : {analyze_json}")

        # 1) 0.attack_result.py
        #    attack_jsonl → attack_result_<tag>.csv (GT 라벨)
        cmd0 = [
            sys.executable,
            "0.attack_result.py",
            "--input", str(attack_jsonl),
            "--window-size", str(window_size),
            "--step-size", str(step_size),
            "--output", str(attack_result_csv),
            # 0.attack_result.py 내부에 --mode auto 등이 있으면,
            # 파일명(tag)에 따라 자동으로 xgt / fc6 기준을 선택하도록 설계해둔 상태라고 가정.
        ]
        run_cmd(cmd0, cwd=here)

        # 2) 1.benchmark.py
        #    attack_jsonl → 윈도우 feature + LSTM-AE MSE 계산
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
            "--tag", tag,                   # 파일별로 다른 이름으로 저장
        ]
        if threshold is not None:
            cmd1.extend(["--threshold", str(threshold)])

        run_cmd(cmd1, cwd=here)

        # 3) 2.eval_detection_metrics.py
        #    attack_result_<tag>.csv + window_scores_<tag>.csv → detection metrics
        cmd2 = [
            sys.executable,
            "2.eval_detection_metrics.py",
            "--attack-csv", str(attack_result_csv),
            "--pred-csv", str(window_scores_csv),
            "--tag", tag,
            "--ignore-pred-minus1",
            # --output-json 을 직접 넘기고 싶으면 여기에 추가하면 됨:
            # "--output-json", str(eval_metrics_json),
        ]
        run_cmd(cmd2, cwd=here)

        # 4) 3.analyze_mse_dist.py
        #    attack_result_<tag>.csv + window_scores_<tag>.csv → MSE 통계
        cmd3 = [
            sys.executable,
            "3.analyze_mse_dist.py",
            "--attack-csv", str(attack_result_csv),
            "--pred-csv", str(window_scores_csv),
            "--tag", tag,
            # "--output-json", str(analyze_json),  # 원하면 명시적 지정 가능
        ]
        run_cmd(cmd3, cwd=here)

        print(f"[INFO] tag={tag} 파이프라인 완료 🎉")
        print(f"  - GT CSV               : {attack_result_csv}")
        print(f"  - Benchmark dir        : {benchmark_out_dir}")
        print(f"  - window_scores CSV    : {window_scores_csv}")
        print(f"  - Detection metrics    : {eval_metrics_json} (또는 2번 스크립트 내부 규칙대로)")
        print(f"  - MSE dist summary     : {analyze_json} (또는 3번 스크립트 내부 규칙대로)")

    print("\n[INFO] 모든 입력 파일에 대한 파이프라인 완료 ✅")


if __name__ == "__main__":
    main()

"""
예시 실행:

1) JSONL 6개를 한 번에 처리 (5번만 나중에 따로 threshold 조정해서 돌리고 싶으면,
   여기서는 공통 설정으로 먼저 한 번 돌리고, 5번 파일만 따로 다시 실행하면 됨)

python 0.run_pipeline_pattern.py --inputs ../data/attack.jsonl ../data/attack_ver2.jsonl ../data/attack_ver5.jsonl ../data/attack_ver5_1.jsonl ../data/attack_ver5_2.jsonl ../data/attack_ver11.jsonl --window-size 16 --step-size 4 --threshold 0.11

2) 5번 시나리오만 threshold 다르게 다시 돌리고 싶을 때:

python 00.run_pipeline_pattern.py --inputs ../data/attack_ver5.jsonl --window-size 80 --step-size 20 --threshold 100

"""

"""
python 0.run_pipeline_pattern.py --window-size 80 --step-size 20 --threshold 100

"""