#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_lstm_ae_inference.py

학습된 LSTM-AE(Keras) 모델의 윈도우 단위 추론 시간을 측정하는 스크립트.

전제:
  - train 스크립트(train_lstm_ae_windows_keras.py)로 학습한 결과 디렉토리가 있음
    • model.h5
    • config.json
    • feature_keys.txt
    • (선택) threshold.json

  - 성능 테스트용 JSONL도 train 때와 같은 형식(패딩 완료된 window):
      {
        "window_id": ...,
        "pattern": "...",
        "index": [0, 1, ..., window_size-1],
        "sequence_group": [
          { feature_key1: float, feature_key2: float, ... },
          ...
        ]
      }

역할:
  - JSONL → (N, T, D) numpy array로 변환
  - model.h5 로드
  - 각 window를 하나씩 넣으면서 추론 시간 측정
  - 평균/중앙값/상위 퍼센타일(ms) 통계 출력 + JSON 저장
"""

import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# -------------------------------------------------------
# JSONL → (N, T, D) 변환 (train 때 쓰던 것과 동일)
# -------------------------------------------------------
def load_windows_to_array(
    jsonl_path: Path,
) -> Tuple[np.ndarray, List[str], List[int], List[str]]:
    X_list: List[np.ndarray] = []
    window_ids: List[int] = []
    patterns: List[str] = []
    feature_keys: List[str] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            seq = obj.get("sequence_group", [])
            if not seq:
                continue

            if not feature_keys:
                feature_keys = sorted(list(seq[0].keys()))

            T = len(seq)
            D = len(feature_keys)
            arr = np.zeros((T, D), dtype=np.float32)

            for t, pkt in enumerate(seq):
                for d, k in enumerate(feature_keys):
                    arr[t, d] = float(pkt.get(k, 0.0))

            X_list.append(arr)
            window_ids.append(int(obj.get("window_id", -1)))
            patterns.append(str(obj.get("pattern", "")))

    if not X_list:
        raise RuntimeError("❌ JSONL에서 유효한 window를 하나도 읽지 못했습니다.")

    X = np.stack(X_list, axis=0)
    return X, feature_keys, window_ids, patterns


# -------------------------------------------------------
# main
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_dir",
        required=True,
        help="train_lstm_ae_windows_keras.py 결과 디렉토리 (model.h5, config.json 등)"
    )
    parser.add_argument(
        "-i", "--input_jsonl",
        required=True,
        help="성능 테스트용 window JSONL (패딩 완료본)"
    )
    parser.add_argument(
        "-o", "--output_json",
        default="benchmark_result.json",
        help="벤치마크 결과를 저장할 JSON 파일 경로"
    )
    parser.add_argument(
        "--max_windows",
        type=int,
        default=None,
        help="앞에서부터 최대 몇 개의 window만 사용할지 (default: 전체)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="추론 전 워밍업용 윈도우 개수 (default: 10)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="predict 시 사용할 batch size (기본: 1, 순수 1-window latency 측정)"
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_json)

    # ------------------------
    # 1) config / 모델 로드
    # ------------------------
    config_path = model_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"[INFO] config 로드: {config_path}")
    print(f"[INFO] 학습 시 T={config['T']}, D={config['D']}")

    import tensorflow as tf
    from tensorflow.keras.models import load_model

    model_path = model_dir / "model.h5"
    print(f"[INFO] model 로드: {model_path}")
    # 추론만 할 거라 compile=False 로 로드 (커스텀 loss 필요 없음)
    model = load_model(model_path, compile=False)

    # 디바이스 정보 출력
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[INFO] 사용 가능한 GPU: {gpus}")
    else:
        print("[INFO] GPU 없음, CPU 사용")

    # ------------------------
    # 2) JSONL 로드
    # ------------------------
    print(f"[INFO] 벤치마크용 JSONL 로드: {input_path}")
    X, feature_keys, window_ids, patterns = load_windows_to_array(input_path)
    N, T, D = X.shape
    print(f"[INFO] 데이터 shape: N={N}, T={T}, D={D}")

    if args.max_windows is not None and args.max_windows < N:
        N = args.max_windows
        X = X[:N]
        window_ids = window_ids[:N]
        patterns = patterns[:N]
        print(f"[INFO] max_windows={args.max_windows} → 실제 사용 N={N}")

    # ------------------------
    # 3) warm-up (그래프 로딩/캐싱 비용 제외)
    # ------------------------
    if args.warmup > 0:
        warmup_n = min(args.warmup, N)
        print(f"[INFO] warmup {warmup_n} 윈도우로 실행...")
        _ = model.predict(X[:warmup_n], batch_size=args.batch_size, verbose=0)

    # ------------------------
    # 4) 1-window 추론 시간 측정
    # ------------------------
    print("[INFO] 1-window 추론 시간 측정 시작...")
    times_ms: List[float] = []

    for i in range(N):
        x = X[i:i+1]  # (1, T, D)
        t0 = time.perf_counter()
        _ = model.predict(x, batch_size=1, verbose=0)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        times_ms.append(elapsed_ms)

    times_arr = np.array(times_ms, dtype=np.float64)

    stats = {
        "num_windows": int(N),
        "mean_ms": float(times_arr.mean()),
        "median_ms": float(np.median(times_arr)),
        "p90_ms": float(np.percentile(times_arr, 90)),
        "p95_ms": float(np.percentile(times_arr, 95)),
        "p99_ms": float(np.percentile(times_arr, 99)),
        "min_ms": float(times_arr.min()),
        "max_ms": float(times_arr.max()),
    }

    print("===== Inference Latency (per window) =====")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # ------------------------
    # 5) 결과 저장
    # ------------------------
    result: Dict[str, Any] = {
        "model_dir": str(model_dir),
        "input_jsonl": str(input_path),
        "batch_size": args.batch_size,
        "stats_ms": stats,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[INFO] 벤치마크 결과 저장 → {output_path}")


if __name__ == "__main__":
    main()
