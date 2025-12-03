#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline_lstm_ae_simple.py

1) 1.padding.py 실행해서 윈도우 패딩 + 일부 feature drop
2) 2.LSTM_AE.py 실행해서 LSTM Autoencoder 학습

우리가 바꿀 수 있는 옵션은 아래 5개만:
  --window-size
  --epochs
  --batch-size
  --hidden-dim
  --latent-dim

나머지 값들은 고정:
  - 입력 JSONL : ../data/pattern_features.jsonl
  - 패딩 JSONL : ../result/pattern_features_padded_0.jsonl
  - pad_value (padding) : 0
  - drop_keys (padding) : ["deltat"]
  - pad_value (train)   : 0.0
  - exclude-file        : ../data/exclude.txt
  - model-output        : ../../result_train/data
  - device              : "cuda"
  - seed                : 42
"""

import argparse
import subprocess
from pathlib import Path
import sys


def run_cmd(cmd, cwd=None):
    print("\n[PIPELINE] 실행할 명령:")
    print("  ", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"[ERROR] 명령 실패 (returncode={result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()

    # ✅ 우리가 바꿀 수 있는 5개 옵션만 받기
    parser.add_argument(
        "--window-size",
        type=int,
        required=True,
        help="padding 시 window_size (예: 76)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="LSTM-AE 학습 epochs (기본=400)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="LSTM-AE 학습 batch size (기본=64)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="LSTM hidden dim (기본=64)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=64,
        help="latent dim (기본=64)",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=5,
        help="Bayesian LSTM-AE에서 MC 샘플 수 (기본=5)"
    )

    args = parser.parse_args()

    # 🔧 고정 값들
    PAD_VALUE_PADDING = -1           # 1.padding.py --pad_value
    PAD_VALUE_TRAIN = -1          # 2.LSTM_AE.py --pad_value
    DROP_KEYS = []         # 1.padding.py --drop_keys
    DEVICE = "cuda"
    SEED = 42

    # 스크립트 위치 기준 상대 경로
    script_dir = Path(__file__).resolve().parent

    padding_script = script_dir / "1.padding.py"
    train_script = script_dir / "2.LSTM_AE.py"
    train_basian_script = script_dir / "3.LSTM_AE_basian.py"

    if not padding_script.exists():
        print(f"[ERROR] 1.padding.py 를 찾을 수 없습니다: {padding_script}")
        sys.exit(1)
    if not train_script.exists():
        print(f"[ERROR] 2.LSTM_AE.py 를 찾을 수 없습니다: {train_script}")
        sys.exit(1)
    if not train_basian_script.exists():
        print(f"[ERROR] 2.LSTM_AE.py 를 찾을 수 없습니다: {train_basian_script}")
        sys.exit(1)

    # 경로들 고정
    input_jsonl = (script_dir / ".." / "data" / "pattern_features.jsonl").resolve()
    padded_jsonl = (script_dir / ".." / "result" / "pattern_features_padded_0.jsonl").resolve()
    exclude_file = (script_dir / ".." / "data" / "exclude.txt").resolve()
    model_output_dir = (script_dir / ".." / ".." / "result_train" / "data").resolve()

    print("[PIPELINE] 고정 경로 설정")
    print(f"  input_jsonl   : {input_jsonl}")
    print(f"  padded_jsonl  : {padded_jsonl}")
    print(f"  exclude_file  : {exclude_file}")
    print(f"  model_output  : {model_output_dir}")

    # --------------------------
    # 1단계: padding 실행
    # --------------------------
    cmd_padding = [
        sys.executable,
        str(padding_script),
        "-i", str(input_jsonl),
        "-o", str(padded_jsonl),
        "--pad_value", str(PAD_VALUE_PADDING),
        "--window_size", str(args.window_size),
    ]

    if DROP_KEYS:
        cmd_padding += ["--drop_keys", *DROP_KEYS]

    print("\n==============================")
    print(" [STEP 1] 1.padding.py 실행")
    print("==============================")
    run_cmd(cmd_padding)

    # --------------------------
    # 2단계: LSTM-AE 학습 실행
    # --------------------------
    model_output_dir.mkdir(parents=True, exist_ok=True)

    cmd_train = [
        sys.executable,
        str(train_script),
        "-i", str(padded_jsonl),
        "-o", str(model_output_dir),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--hidden_dim", str(args.hidden_dim),
        "--latent_dim", str(args.latent_dim),
        "--pad_value", str(PAD_VALUE_TRAIN),
        "--device", DEVICE,
        "--seed", str(SEED),
    ]

    # exclude.txt 고정 사용
    cmd_train += ["--exclude-file", str(exclude_file)]

    print("\n==============================")
    print(" [STEP 2] 2.LSTM_AE.py 실행")
    print("==============================")
    run_cmd(cmd_train)

    print("\n[PIPELINE] 전체 파이프라인 완료 ✅")
    print(f"  ↳ 최종 모델 디렉토리: {model_output_dir}")

    # # --------------------------
    # # 3단계: LSTM-AE_bayesian 학습 실행
    # # --------------------------
    # model_output_dir.mkdir(parents=True, exist_ok=True)

    # cmd_train = [
    #     sys.executable,
    #     str(train_basian_script),
    #     "-i", str(padded_jsonl),
    #     "-o", str(model_output_dir),
    #     "--epochs", str(args.epochs),
    #     "--batch_size", str(args.batch_size),
    #     "--hidden_dim", str(args.hidden_dim),
    #     "--latent_dim", str(args.latent_dim),
    #     "--pad_value", str(PAD_VALUE_TRAIN),
    #     "--device", DEVICE,
    #     "--seed", str(SEED),
    # ]

    # # mc-samples 옵션 (wrapper에 없으면 기본값 5로)
    # mc_samples = getattr(args, "mc_samples", None)
    # if mc_samples is not None:
    #     cmd_train += ["--mc-samples", str(mc_samples)]

    # # exclude.txt 고정 사용
    # cmd_train += ["--exclude-file", str(exclude_file)]

    # print("\n==============================")
    # print(" [STEP 3] 3.LSTM_AE_bayesian.py 실행")
    # print("==============================")
    # run_cmd(cmd_train)

    # print("\n[PIPELINE] 전체 파이프라인 완료 ✅")
    # print(f"  ↳ 최종 모델 디렉토리: {model_output_dir}")



if __name__ == "__main__":
    main()

"""
python 0.run_pipeline_pattern.py --window-size 16 --epochs 300 --batch-size 128 --hidden-dim 128 --latent-dim 64 --mc-samples 10

| 인자             | 설명                     | 주요 영향              |
| --------------- | ------------------------ | -------------------- |
| `--window-size` | 시퀀스 길이 (패킷 묶음 단위) | 패턴 포착 범위         |
| `--epochs`      | 학습 반복 횟수             | under/overfitting     |
| `--batch-size`  | 병렬 학습 윈도우 수         | 메모리·속도·일반화      |
| `--hidden-dim`  | LSTM 내부 상태 크기        | 표현력 / 과적합         |
| `--latent-dim`  | 압축된 잠재공간 크기        | 정보 손실 / 분리도      |

"""