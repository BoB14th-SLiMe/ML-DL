#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark.py

패킷 단위 JSONL (각 line이 하나의 패킷) →
전처리 파라미터 + packet_feature_extractor 를 이용해서

1) 패킷 window_size개씩 순서대로 묶어서 하나의 window(sequence_group)로 보고
2) 해당 window를 feature matrix (shape = [window_size, feat_dim]) 로 변환
3) 모든 window를 모아서 X_windows.npy 로 저장
4) window 메타정보는 windows_meta.jsonl 로 저장
5) (선택) 학습된 LSTM-AE(Keras) 모델을 불러서 윈도우별 reconstruction MSE 계산 + anomaly 여부 기록

⚠️ 슬라이딩 윈도우:
   start = 0, step_size, 2*step_size, ...
   예: window_size=80, step_size=40 이면
       [0~79], [40~119], [80~159], ...
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from datetime import datetime

# 기존에 만들어둔 모듈 재사용
from packet_feature_extractor import (
    load_preprocess_params,
    sequence_group_to_feature_matrix,
    PACKET_FEATURE_COLUMNS,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input", "-i", required=True,
        help="패킷 단위 JSONL 경로 (각 line = 1 packet)",
    )
    p.add_argument(
        "--pre-dir", "-p", required=True,
        help="전처리 파라미터 JSON들이 모여있는 디렉토리",
    )
    p.add_argument(
        "--window-size", "-w", type=int, default=80,
        help="윈도우 길이(묶을 패킷 개수, 기본=80)",
    )
    p.add_argument(
        "--step-size", "-s", type=int, default=None,
        help="슬라이딩 stride (기본: window-size와 동일 → non-overlap)",
    )
    p.add_argument(
        "--output-dir", "-o", required=True,
        help="출력 디렉토리 (X_windows.npy, windows_meta.jsonl, window_scores.csv)",
    )
    p.add_argument(
        "--no-pad-last", action="store_true",
        help="마지막 윈도우가 window-size 보다 짧으면 0 패딩 없이 버림",
    )

    # ⬇️ DL 모델 inference 관련 옵션 (model_dir 방식)
    p.add_argument(
        "--model-dir", "-m", default=None,
        help="train_lstm_ae_windows_keras.py 결과 디렉토리 (model.h5, config.json 등)",
    )
    p.add_argument(
        "--batch-size", "-b", type=int, default=128,
        help="DL 모델 inference batch size (기본=128)",
    )
    p.add_argument(
        "--threshold", "-t", type=float, default=None,
        help=(
            "윈도우 MSE anomaly threshold "
            "(미지정 시 threshold.json이 있으면 그 값을 사용, 둘 다 없으면 점수만 기록)"
        ),
    )

    return p.parse_args()


# -----------------------------
# timestamp → delta_t 계산
# -----------------------------
def parse_timestamp(ts: Any) -> float:
    """
    "@timestamp": "2025-11-10T17:43:40.425Z" 형태를 float epoch 로 변환.
    실패하면 0.0 리턴.
    """
    if not ts:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    ts_str = str(ts)
    try:
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1]
        dt = datetime.fromisoformat(ts_str)
        return dt.timestamp()
    except Exception:
        return 0.0


# -----------------------------
# train 코드와 동일한 window MSE 계산
# -----------------------------
def compute_window_errors(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    pad_value: float,
) -> np.ndarray:
    """
    X_true, X_pred: shape (N, T, D)
    pad_value    : 패딩 값 (해당 timestep은 마스크)

    반환:
      errors: shape (N,), 윈도우별 재구성 오차
    """
    # 패딩이 아닌 timestep 마스크 (N, T)
    not_pad = np.any(np.not_equal(X_true, pad_value), axis=-1)
    mask = not_pad.astype(np.float32)

    # 타임스텝별 MSE (N, T)
    se = np.mean((X_pred - X_true) ** 2, axis=-1)
    se_masked = se * mask

    denom = np.sum(mask, axis=-1) + 1e-8
    errors = np.sum(se_masked, axis=-1) / denom
    return errors


# -----------------------------
# 모델 로딩 (benchmark_lstm_ae_inference 스타일)
# -----------------------------
def load_model_from_dir(model_dir: Path):
    """
    model_dir 내부:
      - config.json : {"T": ..., "D": ..., "pad_value": ...} 포함
      - model.h5    : Keras LSTM-AE 모델
      - threshold.json (선택)

    반환:
      model, config(dict), threshold(float 또는 None)
    """
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"❌ config.json 없음: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"[INFO] config 로드: {config_path}")
    T = config.get("T")
    D = config.get("D")
    print(f"[INFO] 학습 시 T={T}, D={D}")

    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.models import load_model

    model_path = model_dir / "model.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"❌ model.h5 없음: {model_path}")

    print(f"[INFO] model 로드: {model_path}")

    # 🔥 config.json 에서 T, D 읽기 (decoder에서 반복할 길이)
    T = config.get("T")
    D = config.get("D")

    def repeat_latent(x):
        """
        train_lstm_ae_windows_keras.py 에서 썼던 repeat_latent와 동일한 동작:
        (B, latent_dim) -> (B, T, latent_dim)
        """
        x = tf.expand_dims(x, axis=1)      # (B, 1, latent_dim)
        x = tf.tile(x, [1, T, 1])          # (B, T, latent_dim)
        return x

    custom_objects = {
        "repeat_latent": repeat_latent,
    }

    model = load_model(
        model_path,
        compile=False,
        custom_objects=custom_objects,
    )

    # ---------------------------
    # threshold.json 읽기 (구조 맞게)
    # ---------------------------
    thresh_path = model_dir / "threshold.json"
    threshold = None
    if thresh_path.exists():
        try:
            with thresh_path.open("r", encoding="utf-8") as f:
                th_cfg = json.load(f)

            # train 코드에서 저장한 구조:
            # {
            #   "threshold_p99": ...,
            #   "threshold_mu3": ...,
            #   "stats": {...}
            # }
            if "threshold" in th_cfg:
                # 혹시 나중에 단일 threshold로 저장해둔 버전
                threshold = float(th_cfg["threshold"])
                print(f"[INFO] threshold.json(threshold) 사용: {threshold}")
            elif "threshold_p99" in th_cfg:
                threshold = float(th_cfg["threshold_p99"])
                print(f"[INFO] threshold_p99 사용: {threshold}")
            elif "threshold_mu3" in th_cfg:
                threshold = float(th_cfg["threshold_mu3"])
                print(f"[INFO] threshold_mu3 사용: {threshold}")
            else:
                print(
                    "[INFO] threshold.json 안에 사용할 키가 없습니다. "
                    "(threshold / threshold_p99 / threshold_mu3 없음)"
                )
        except Exception as e:
            print(f"[WARN] threshold.json 로딩 실패: {e}")

    # feature_keys.txt 있으면 길이 확인 용도로만 사용
    feat_path = model_dir / "feature_keys.txt"
    if feat_path.exists():
        try:
            with feat_path.open("r", encoding="utf-8") as f:
                feature_keys = [line.strip() for line in f if line.strip()]
            print(f"[INFO] feature_keys.txt 길이 = {len(feature_keys)}")
        except Exception as e:
            print(f"[WARN] feature_keys.txt 로딩 실패: {e}")

    return model, config, threshold


def main():
    args = parse_args()

    input_path = Path(args.input)
    pre_dir = Path(args.pre_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window_size = args.window_size
    step_size = args.step_size if args.step_size is not None else window_size
    pad_last = not args.no_pad_last

    print(f"[INFO] 입력 JSONL : {input_path}")
    print(f"[INFO] 전처리 디렉토리 : {pre_dir}")
    print(f"[INFO] 출력 디렉토리 : {output_dir}")
    print(f"[INFO] window_size = {window_size}, step_size = {step_size}, pad_last = {pad_last}")

    # 1) 전처리 파라미터 로딩 (packet_feature_extractor 내부 정의에 맞게)
    print("[INFO] 전처리 파라미터 로딩 중...")
    params = load_preprocess_params(pre_dir)
    feat_dim = len(PACKET_FEATURE_COLUMNS)
    print(f"[INFO] feature dimension = {feat_dim} (PACKET_FEATURE_COLUMNS 길이)")

    # 2) JSONL 전체를 읽어서 delta_t 포함한 packet 리스트 만들기
    all_packets: List[Dict[str, Any]] = []
    prev_ts = None

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                pkt = json.loads(line)
            except Exception as e:
                print(f"[WARN] JSON 파싱 실패: {e}")
                continue

            # delta_t 계산
            ts_val = parse_timestamp(pkt.get("@timestamp"))
            if prev_ts is None:
                delta_t = 0.0
            else:
                dt = ts_val - prev_ts
                delta_t = dt if dt >= 0 else 0.0
            prev_ts = ts_val

            pkt["delta_t"] = delta_t  # feature extractor 에서 사용할 수 있게 넣어줌
            all_packets.append(pkt)

    total_packets = len(all_packets)
    print(f"[INFO] 총 패킷 수 = {total_packets}")

    if total_packets == 0:
        print("[WARN] 입력 JSONL에서 유효한 패킷이 없습니다.")
        return

    # 3) 슬라이딩 윈도우 생성
    all_windows: List[np.ndarray] = []
    meta_list: List[Dict[str, Any]] = []

    start_idx = 0
    window_index = 0

    while start_idx < total_packets:
        end_idx = start_idx + window_size
        group = all_packets[start_idx:end_idx]

        if len(group) == 0:
            break

        # 마지막 윈도우가 window_size보다 짧고 pad_last가 False면 중단
        if len(group) < window_size and not pad_last:
            print(f"[INFO] 마지막 window({start_idx}~{total_packets-1})는 길이 {len(group)} < {window_size}, 패딩 없이 버림")
            break

        X_list = sequence_group_to_feature_matrix(group, params)
        if not X_list:
            print(f"[WARN] window({start_idx}~{min(end_idx, total_packets)-1})에서 feature 추출 실패, 스킱")
            start_idx += step_size
            continue

        X = np.array(X_list, dtype="float32")
        valid_len = X.shape[0]

        if valid_len < window_size:
            if pad_last:
                pad_len = window_size - valid_len
                pad_block = np.zeros((pad_len, feat_dim), dtype=X.dtype)
                X = np.concatenate([X, pad_block], axis=0)
        elif valid_len > window_size:
            X = X[:window_size, :]
            valid_len = window_size

        all_windows.append(X)
        meta_list.append({
            "window_index": window_index,
            "start_packet_idx": start_idx,
            "end_packet_idx": min(end_idx, total_packets) - 1,
            "valid_len": int(valid_len),
        })

        window_index += 1
        start_idx += step_size

    num_windows = len(all_windows)
    print(f"[INFO] 생성된 윈도우 수 = {num_windows}")

    if num_windows == 0:
        print("[WARN] 생성된 윈도우가 없습니다. 입력/파라미터/step_size를 확인하세요.")
        return

    # 4) numpy 저장
    X_windows = np.stack(all_windows, axis=0)  # [num_windows, window_size, feat_dim]
    out_npy = output_dir / "X_windows.npy"
    np.save(out_npy, X_windows)
    print(f"[INFO] 저장 완료: {out_npy} (shape={X_windows.shape})")

    # 5) 메타정보 JSONL 저장
    out_meta = output_dir / "windows_meta.jsonl"
    with out_meta.open("w", encoding="utf-8") as fmeta:
        for m in meta_list:
            fmeta.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"[INFO] 메타 정보 저장 완료: {out_meta}")

    # ============================
    # 6) DL 모델 불러서 탐지 (선택)
    # ============================
    if args.model_dir:
        model_dir = Path(args.model_dir)
        print(f"[INFO] DL 모델 디렉토리: {model_dir}")

        model, config, threshold_from_file = load_model_from_dir(model_dir)

        # config와 현재 데이터 shape consistency 체크
        T_cfg = config.get("T")
        D_cfg = config.get("D")
        pad_value = float(config.get("pad_value", 0.0))
        _, T_cur, D_cur = X_windows.shape

        if T_cfg is not None and T_cfg != T_cur:
            print(f"[WARN] config.T({T_cfg}) != 현재 window_size({T_cur})")
        if D_cfg is not None and D_cfg != D_cur:
            print(f"[WARN] config.D({D_cfg}) != 현재 feature_dim({D_cur})")

        # threshold 결정: CLI > threshold.json > None
        threshold = args.threshold
        if threshold is None and threshold_from_file is not None:
            threshold = threshold_from_file
            print(f"[INFO] threshold.json의 값을 사용: threshold={threshold}")
        elif threshold is not None:
            print(f"[INFO] CLI로 전달된 threshold 사용: threshold={threshold}")
        else:
            print("[INFO] threshold 미지정 → anomaly 라벨링 없이 MSE만 기록")

        # 디바이스 정보 (옵션)
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                print(f"[INFO] 사용 가능한 GPU: {gpus}")
            else:
                print("[INFO] GPU 없음, CPU 사용")
        except Exception:
            pass

        print("[INFO] DL 모델로 윈도우별 reconstruction 예측 중...")
        recon = model.predict(X_windows, batch_size=args.batch_size, verbose=1)

        if recon.shape != X_windows.shape:
            print(f"[WARN] 재구성 결과 shape {recon.shape} != 입력 shape {X_windows.shape}")
            print("       브로드캐스트 가능한지 확인 후 MSE 계산을 진행합니다.")

        # 🔥 train 코드와 동일한 방식으로 윈도우별 MSE 계산 (pad_value 마스킹)
        # -------------------------
        # 윈도우별 MSE 계산 (학습 코드와 동일한 방식, pad mask 적용)
        # -------------------------
        pad_value = float(config.get("pad_value", 0.0))

        diff = X_windows - recon  # (N, T, D)

        # 각 timestep이 pad인지 아닌지: feature 중 하나라도 pad_value가 아니면 유효
        not_pad = np.any(np.not_equal(X_windows, pad_value), axis=-1)  # (N, T)
        mask = not_pad.astype(np.float32)  # (N, T)

        # timestep별 MSE
        se = np.mean(diff ** 2, axis=-1)         # (N, T)
        se_masked = se * mask                    # (N, T)

        denom = np.sum(mask, axis=-1) + 1e-8     # (N,)
        mse_per_window = np.sum(se_masked, axis=-1) / denom  # (N,)

        print(
            "[INFO] benchmark 윈도우 MSE 통계: "
            f"mean={mse_per_window.mean():.4f}, "
            f"std={mse_per_window.std():.4f}, "
            f"min={mse_per_window.min():.4f}, "
            f"max={mse_per_window.max():.4f}"
        )

        out_scores = output_dir / "window_scores.csv"
        with out_scores.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "window_index",
                "start_packet_idx",
                "end_packet_idx",
                "valid_len",
                "mse",
                "is_anomaly",  # threshold 없으면 -1
            ])
            for m, mse in zip(meta_list, mse_per_window):
                if threshold is not None:
                    is_anom = int(mse > threshold)
                else:
                    is_anom = -1  # 라벨 없음
                writer.writerow([
                    m["window_index"],
                    m["start_packet_idx"],
                    m["end_packet_idx"],
                    m["valid_len"],
                    float(mse),
                    is_anom,
                ])

        print(f"[INFO] 윈도우별 MSE / anomaly 여부 저장 완료: {out_scores}")

        if threshold is not None:
            num_anom = int(np.sum(mse_per_window > threshold))
            print(f"[INFO] threshold={threshold} 기준 anomaly 윈도우 수: {num_anom}/{num_windows}")
        else:
            print("[INFO] threshold가 없으므로 is_anomaly = -1 로만 기록되었습니다. CSV에서 MSE 분포를 먼저 확인하세요.")


if __name__ == "__main__":
    main()


"""
실행 예시:

# 1) non-overlap 윈도우 + feature만 만들고 싶을 때
python 1.benchmark.py --input "../data/attack.jsonl" --pre-dir "../../preprocessing/result" --window-size 80 --output-dir "../result/benchmark"

# 2) 슬라이딩 윈도우 (size=80, step=40) + LSTM-AE 탐지까지 수행할 때
python 1.benchmark.py --input "../data/attack.jsonl" --pre-dir "../../preprocessing/result" --window-size 80 --step-size 30 --output-dir "../result/benchmark" --model-dir "../data" --batch-size 128

python 1.benchmark.py --input "../data/attack.jsonl" --pre-dir "../../preprocessing/result" --window-size 80 --step-size 30 --output-dir "../result/benchmark" --model-dir "../data" --batch-size 128 --threshold 100

# threshold.json 안에 threshold_p99 / threshold_mu3가 있으면 자동으로 사용,
# CLI에서 --threshold 300 같이 주면 그 값이 우선 사용됨.
"""
