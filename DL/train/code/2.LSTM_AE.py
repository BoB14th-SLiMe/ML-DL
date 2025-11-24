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
# 공통 유틸
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


# -------------------------------------------------------
# JSONL → (N, T, D) 변환 + feature 선택
# -------------------------------------------------------
def load_windows_to_array(
    jsonl_path: Path,
    exclude_features: List[str] | None = None,
) -> Tuple[np.ndarray, List[str], List[int], List[str]]:
    """
    JSONL 파일 → (N, T, D) numpy array로 변환

    exclude_features:
      - 학습에서 제외할 feature 이름 리스트
      - seq[0].keys() 중 해당 이름이 있으면 제거

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

    exclude_set = set(exclude_features) if exclude_features else set()

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
                all_keys = sorted(list(seq[0].keys()))

                if exclude_set:
                    actually_excluded = sorted(set(all_keys) & exclude_set)
                    if actually_excluded:
                        print(f"[INFO] load_windows_to_array: 실제로 제외되는 feature = {actually_excluded}")
                    not_found = sorted(exclude_set - set(all_keys))
                    if not_found:
                        print(f"[WARN] load_windows_to_array: JSONL에 존재하지 않는 feature (무시됨) = {not_found}")

                    feature_keys = [k for k in all_keys if k not in exclude_set]
                    if not feature_keys:
                        raise RuntimeError("❌ 모든 feature가 exclude되어 남는 feature가 없습니다.")
                else:
                    feature_keys = all_keys

                print(f"[INFO] 최종 사용 feature 수 = {len(feature_keys)}")
                print(f"[INFO] 예시 feature 목록 (앞 10개): {feature_keys[:10]}")

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
# 데이터 인스펙션 유틸
# -------------------------------------------------------
def inspect_data(
    X: np.ndarray,
    feature_keys: List[str],
    window_ids: List[int],
    patterns: List[str],
    pad_value: float = 0.0,
    n_samples: int = 3,
):
    """
    학습 전에 X / feature / window 몇 개를 눈으로 확인하는 디버그용 함수.
    """
    N, T, D = X.shape
    print("\n================= [INSPECT DATA] =================")
    print(f"N (windows) = {N}, T (time steps) = {T}, D (features) = {D}")
    print(f"pad_value = {pad_value}")
    print(f"feature_keys (앞 10개): {feature_keys[:10]}")
    print("===================================================\n")

    # 전체 데이터 flatten 해서 feature별 통계
    X_flat = X.reshape(-1, D)  # (N*T, D)

    print(">>> Feature-wise 통계 (pad_value 제외):")
    for i, k in enumerate(feature_keys):
        col = X_flat[:, i]
        # pad_value로만 가득한 feature면 제외
        mask = col != pad_value
        if not np.any(mask):
            print(f"  - {k}: (모든 값이 pad_value={pad_value})")
            continue
        vals = col[mask]
        print(
            f"  - {k:25s} | "
            f"min={vals.min():.4f}, max={vals.max():.4f}, "
            f"mean={vals.mean():.4f}, std={vals.std():.4f}, "
            f"non_pad_ratio={len(vals)/len(col):.3f}"
        )

    # 몇 개 윈도우 샘플 출력
    print("\n>>> 샘플 윈도우 몇 개 보기:")
    n_samples = min(n_samples, N)
    for idx in range(n_samples):
        print(f"\n--- Window #{idx} (global index) ---")
        print(f"window_id = {window_ids[idx]}, pattern = {patterns[idx]}")
        # 앞 5 timestep만
        steps = min(5, T)
        for t in range(steps):
            row = X[idx, t]
            # 이 timestep이 패딩만 있는지 여부
            if np.all(row == pad_value):
                print(f"  t={t:2d}: [PAD ROW]")
            else:
                # 앞 몇 feature만 보기
                feat_preview_cnt = min(8, D)
                preview = ", ".join(
                    f"{feature_keys[j]}={row[j]:.4f}"
                    for j in range(feat_preview_cnt)
                )
                print(f"  t={t:2d}: {preview}")
    print("\n===================================================\n")


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
    # 🔥 기존: CLI에서 직접 feature 나열
    parser.add_argument(
        "--exclude-features",
        nargs="+",
        default=None,
        help=(
            "학습에서 제외할 feature 이름 리스트 (공백으로 구분). "
            "예: --exclude-features protocol delta_t modbus_regs_val_std"
        ),
    )
    # 🔥 추가: TXT 파일로 feature 제외 리스트 관리
    parser.add_argument(
        "--exclude-file",
        type=str,
        default=None,
        help=(
            "학습에서 제외할 feature 이름을 줄 단위로 적어둔 txt 파일 경로. "
            "빈 줄 / #으로 시작하는 줄은 무시됨. "
            "예: --exclude-file ../config/exclude_features.txt"
        ),
    )
    # 👀 데이터만 보고 싶은 옵션
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="데이터를 로드/요약 출력만 하고 학습은 수행하지 않음",
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

    # -----------------------------
    # 제외 feature 리스트 구성 (CLI + TXT 합치기)
    # -----------------------------
    exclude_from_cli: List[str] = args.exclude_features or []
    exclude_from_file: List[str] = []

    if args.exclude_file:
        excl_path = Path(args.exclude_file)
        if not excl_path.exists():
            print(f"[WARN] exclude-file 경로에 파일이 없습니다: {excl_path}")
        else:
            print(f"[INFO] exclude-file 로드: {excl_path}")
            with excl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    name = line.strip()
                    if not name:
                        continue
                    if name.startswith("#"):
                        continue
                    exclude_from_file.append(name)

    # 두 리스트 합치고 순서 유지하면서 중복 제거
    merged_exclude: List[str] = []
    for name in exclude_from_cli + exclude_from_file:
        if name not in merged_exclude:
            merged_exclude.append(name)

    if merged_exclude:
        print(f"[INFO] 최종 제외 feature 목록 = {merged_exclude}")
    else:
        print("[INFO] 제외할 feature 없음 (전체 feature 사용)")

    # 1) 데이터 로드
    print(f"[INFO] JSONL 로드: {input_path}")
    X, feature_keys, window_ids, patterns = load_windows_to_array(
        input_path,
        exclude_features=merged_exclude,
    )
    N, T, D = X.shape
    print(f"[INFO] 데이터 shape: N={N}, T={T}, D={D}")
    print(f"[INFO] 최종 feature 수: {len(feature_keys)}")

    # 👀 inspect-only 모드면 여기서 데이터만 보고 종료
    if args.inspect_only:
        inspect_data(
            X,
            feature_keys,
            window_ids,
            patterns,
            pad_value=float(args.pad_value),
            n_samples=3,
        )
        print("[INFO] --inspect-only 플래그로 인해 학습 없이 종료합니다.")
        return

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
            not_pad = tf.reduceAny(tf.not_equal(y_true, pad_val), axis=-1)  # (B, T) bool
            mask = tf.cast(not_pad, tf.float32)                              # (B, T)

            se = tf.reduceMean(tf.square(y_pred - y_true), axis=-1)        # (B, T)
            se_masked = se * mask

            # eps로 0 나누기 방지
            loss = tf.reduceSum(se_masked) / (tf.reduceSum(mask) + 1e-8)
            return loss
        return masked_mse

    # 위 reduceAny / reduceMean / reduceSum 오타 주의:
    import tensorflow as tf  # 이미 위에서 했지만 안전하게
    def make_masked_mse(pad_val: float):
        def masked_mse(y_true, y_pred):
            not_pad = tf.reduce_any(tf.not_equal(y_true, pad_val), axis=-1)  # (B, T) bool
            mask = tf.cast(not_pad, tf.float32)                              # (B, T)

            se = tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)        # (B, T)
            se_masked = se * mask

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
        "exclude_features": merged_exclude,
    }
    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[INFO] config 저장 → {config_path}")

    log_path = output_dir / "train_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[INFO] train_log 저장 → {log_path}")

    # 8) loss / val_loss 곡선 그림 저장
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
python 2.LSTM_AE.py -i "../result/pattern_features_padded_0.jsonl" -o "../../result_train/data" --epochs 400 --batch_size 64 --hidden_dim 64 --latent_dim 64 --pad_value 0.0 --device cuda --seed 42 --exclude-file "../data/exclude.txt"

inspect 모드:
python 2.LSTM_AE.py -i "../result/pattern_features_padded_0.jsonl" -o "../../result_train/inspect" --pad_value 0.0 --exclude-file "../data/exclude.txt" --inspect-only
"""
