#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_lstm_ae_windows_keras.py

Keras/TensorFlow 버전의 LSTM Autoencoder 학습 스크립트.

입력:
  - window 단위 패턴 feature JSONL (pad_pattern_features_by_index.py 결과)
    각 라인:
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
  - LSTM Autoencoder 학습 (패딩 값에 대한 mask 지원)
  - 모델/설정/feature_key 리스트 저장

출력 (output_dir):
  - model.h5           : 학습된 모델 (전체 Keras 모델)
  - config.json        : 학습 설정 및 데이터 차원 정보
  - feature_keys.txt   : feature key 순서 (한 줄 하나)
  - train_log.json     : epoch별 train/val loss 기록
"""

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import random

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------------------------------
# 데이터 로더 (원본과 동일)
# -------------------------------------------------------
def compute_window_errors(X_true: np.ndarray,
                          X_pred: np.ndarray,
                          pad_value: float) -> np.ndarray:
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


def set_global_seed(seed: int):
    """Python, NumPy, TensorFlow 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)

    # TensorFlow seeding (지연 import)
    import tensorflow as tf
    tf.random.set_seed(seed)

def load_windows_to_array(
    jsonl_path: Path,
) -> Tuple[np.ndarray, List[str], List[int], List[str]]:
    """
    JSONL 파일 → (N, T, D) numpy array로 변환

    반환:
      X           : shape (N, T, D), float32
      feature_keys: feature 이름 리스트 (길이 D, 순서 고정)
      window_ids  : 각 윈도우의 window_id 리스트
      patterns    : 각 윈도우의 pattern 리스트
    """
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

            # feature_keys를 첫 window에서 한 번만 결정
            if not feature_keys:
                # 정렬해서 고정된 순서 사용 (원본과 동일)
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

    X = np.stack(X_list, axis=0)  # (N, T, D)
    return X, feature_keys, window_ids, patterns


# -------------------------------------------------------
# main
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_jsonl",
        required=True,
        help="pad_pattern_features_by_index.py 결과 JSONL 경로",
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="모델 및 로그를 저장할 디렉토리",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="학습 epoch 수 (default: 50)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size (default: 64)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="LSTM hidden dim (default: 128)",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
        help="latent dim (default: 64)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="LSTM layer 수 (encoder에만 적용, default: 1; 현재는 1만 권장)",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="encoder LSTM을 bidirectional로 사용할지 여부",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="validation 비율 (default: 0.2)",
    )
    parser.add_argument(
        "--pad_value",
        type=float,
        default=0.0,
        help="패딩 값 (loss 계산 시 mask용, default: 0.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu (Keras/TensorFlow는 자동 선택; 이 값은 로그용)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (default: 42)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    print(f"[INFO] Random seed = {args.seed}")

    # TensorFlow / Keras import
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    print(f"[INFO] TensorFlow version: {tf.__version__}")
    print(f"[INFO] device flag = {args.device} (실제 사용 디바이스는 TensorFlow가 자동 선택)")

    # 1) 데이터 로드
    print(f"[INFO] JSONL 로드: {input_path}")
    X, feature_keys, window_ids, patterns = load_windows_to_array(input_path)
    N, T, D = X.shape
    print(f"[INFO] 데이터 shape: N={N}, T={T}, D={D}")
    print(f"[INFO] feature 수: {len(feature_keys)}")

    # feature key 순서 저장
    feat_path = output_dir / "feature_keys.txt"
    with feat_path.open("w", encoding="utf-8") as f:
        for k in feature_keys:
            f.write(k + "\n")
    print(f"[INFO] feature_keys.txt 저장 → {feat_path}")

    # 2) Train/Val split
    val_ratio = args.val_ratio
    indices = np.arange(N)
    np.random.shuffle(indices)
    split = int(N * (1.0 - val_ratio))
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train = X[train_idx]
    X_val = X[val_idx]

    print(f"[INFO] train N = {X_train.shape[0]}, val N = {X_val.shape[0]}")

    # 3) LSTM Autoencoder 모델 정의 (Keras)
    input_dim = D
    hidden_dim = args.hidden_dim
    latent_dim = args.latent_dim
    bidirectional = args.bidirectional

    # Encoder
    encoder_inputs = layers.Input(shape=(T, input_dim), name="encoder_input")

    if bidirectional:
        # 양방향 LSTM: 출력 차원은 hidden_dim * 2
        lstm_layer = layers.Bidirectional(
            layers.LSTM(hidden_dim, return_sequences=False),
            name="encoder_bi_lstm",
        )
        encoder_output = lstm_layer(encoder_inputs)  # (B, hidden_dim * 2)
    else:
        lstm_layer = layers.LSTM(
            hidden_dim, return_sequences=False, name="encoder_lstm"
        )
        encoder_output = lstm_layer(encoder_inputs)  # (B, hidden_dim)

    latent = layers.Dense(latent_dim, name="latent_dense")(encoder_output)  # (B, latent_dim)

    # Decoder: latent를 시퀀스 길이만큼 반복
    def repeat_latent(x):
        # x: (B, latent_dim)
        x = tf.expand_dims(x, axis=1)  # (B, 1, latent_dim)
        x = tf.tile(x, [1, T, 1])      # (B, T, latent_dim)
        return x

    repeated_latent = layers.Lambda(repeat_latent, name="repeat_latent")(latent)
    decoder_lstm = layers.LSTM(
        hidden_dim,
        return_sequences=True,
        name="decoder_lstm",
    )
    decoder_output = decoder_lstm(repeated_latent)          # (B, T, hidden_dim)
    decoder_dense = layers.TimeDistributed(
        layers.Dense(input_dim), name="decoder_output_dense"
    )
    outputs = decoder_dense(decoder_output)                 # (B, T, D)

    model = models.Model(inputs=encoder_inputs, outputs=outputs, name="lstm_autoencoder")
    model.summary()

    # 4) 손실 함수 (pad_value 마스킹)
    pad_value = float(args.pad_value)

    def make_masked_mse(pad_val: float):
        def masked_mse(y_true, y_pred):
            # y_true, y_pred: (B, T, D)
            # 모든 feature가 pad_val인 timestep은 마스크 0
            # (원본 PyTorch 구현: (batch != pad_value).any(dim=-1))
            not_pad = tf.reduce_any(tf.not_equal(y_true, pad_val), axis=-1)  # (B, T) bool
            mask = tf.cast(not_pad, tf.float32)                              # (B, T)

            se = tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)        # (B, T)
            se_masked = se * mask

            # eps로 0 나누기 방지
            loss = tf.reduce_sum(se_masked) / (tf.reduce_sum(mask) + 1e-8)
            return loss
        return masked_mse

    loss_fn = make_masked_mse(pad_value)

    optimizer = optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss=loss_fn)



    # 5) 학습
    es = EarlyStopping(
        monitor="val_loss",
        patience=5,       # 5 epoch 동안 개선 없으면 멈춤
        restore_best_weights=True,
        verbose=1,
    )

    print("[INFO] Keras model.fit() 시작")
    history_obj = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=[es],
        verbose=1,
    )

    history = {
        "train_loss": list(map(float, history_obj.history.get("loss", []))),
        "val_loss": list(map(float, history_obj.history.get("val_loss", []))),
    }

    # 6) train set reconstruction error 기반 threshold 계산
    print("[INFO] train set reconstruction error 계산...")
    X_train_pred = model.predict(X_train,
                                 batch_size=args.batch_size,
                                 verbose=1)

    errors_train = compute_window_errors(X_train,
                                         X_train_pred,
                                         pad_value)

    print(f"[INFO] train error 통계: "
          f"mean={errors_train.mean():.4f}, "
          f"std={errors_train.std():.4f}, "
          f"min={errors_train.min():.4f}, "
          f"max={errors_train.max():.4f}")

    # 대표적인 두 종류 threshold
    threshold_p99 = float(np.percentile(errors_train, 99.0))
    threshold_mu3 = float(errors_train.mean() + 3.0 * errors_train.std())

    print(f"[INFO] 99th percentile threshold = {threshold_p99:.4f}")
    print(f"[INFO] mean + 3*std threshold    = {threshold_mu3:.4f}")

    threshold_info = {
        "threshold_p99": threshold_p99,
        "threshold_mu3": threshold_mu3,
        "stats": {
            "mean": float(errors_train.mean()),
            "std": float(errors_train.std()),
            "min": float(errors_train.min()),
            "max": float(errors_train.max()),
        }
    }

    thr_path = output_dir / "threshold.json"
    with thr_path.open("w", encoding="utf-8") as f:
        json.dump(threshold_info, f, indent=2, ensure_ascii=False)
    print(f"[INFO] threshold.json 저장 → {thr_path}")

    # 7) 모델/설정/로그 저장
    model_path = output_dir / "model.h5"
    model.save(model_path)
    print(f"[INFO] 모델 저장 → {model_path}")

    config = {
        "input_jsonl": str(input_path),
        "N": int(N),
        "T": int(T),
        "D": int(D),
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "num_layers": args.num_layers,
        "bidirectional": bidirectional,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_ratio": args.val_ratio,
        "pad_value": pad_value,
        "device_flag": args.device,
        "framework": "tensorflow.keras",
        "seed": args.seed,
    }
    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[INFO] config 저장 → {config_path}")

    log_path = output_dir / "train_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[INFO] train_log 저장 → {log_path}")

    # 7) loss / val_loss 곡선 그림 저장
    try:
        epochs_range = range(1, len(history["train_loss"]) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, history["train_loss"], label="train_loss")
        plt.plot(epochs_range, history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("LSTM-AE Training / Validation Loss")
        plt.legend()
        plt.grid(True)

        plot_path = output_dir / "loss_curve.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"[INFO] loss_curve.png 저장 → {plot_path}")
    except Exception as e:
        print(f"[WARN] loss 그래프 저장 중 오류 발생: {e}")



if __name__ == "__main__":
    main()


"""
python LSTM_AE.py -i "../result/pattern_features_padded_0.jsonl" -o "../result/LSTM_AE" --epochs 100 --batch_size 128 --hidden_dim 128 --latent_dim 64 --pad_value 0.0 --device cuda --seed 42

"""