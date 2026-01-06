#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_lstm_ae_windows_keras.py

Keras/TensorFlow LSTM Autoencoder 학습 스크립트.

입력(JSONL):
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

특이값(sentinel):
  - pad_value     : 패딩(예: -1)
  - missing_value : 실데이터 없음(예: -2)
  => 학습(loss) 및 threshold(error) 계산에서 둘 다 제외(mask)

출력(output_dir):
  - model.h5
  - config.json
  - feature_keys.txt
  - train_log.json
  - threshold.json
  - training_loss_curve.png
  - (옵션) bayes_opt_trials.json
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
import random
import gc
import math
import shutil

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------------------------------------
# 공통 유틸
# -------------------------------------------------------
def compute_window_errors(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    pad_value: float,
    missing_value: float,
    feature_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    X_true, X_pred: (N, T, D)
    pad_value/missing_value: sentinel 값 → error 계산에서 제외 (원소 단위 마스킹)
    feature_weights: (D,) 또는 None

    반환:
      errors: (N,) 윈도우별 평균 재구성 오차 (valid 원소 기준)
    """
    valid = (X_true != pad_value) & (X_true != missing_value)  # (N,T,D)

    se = (X_pred - X_true) ** 2  # (N,T,D)
    if feature_weights is not None:
        se = se * feature_weights[np.newaxis, np.newaxis, :]  # (N,T,D)

    num = np.sum(se * valid, axis=(1, 2))
    den = np.sum(valid, axis=(1, 2)) + 1e-8
    return num / den


def save_training_loss_curve(history: Dict[str, List[float]], out_png: Path):
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if not train_loss:
        print("[WARN] save_training_loss_curve: train_loss가 비어 있어 그래프를 저장하지 않습니다.")
        return

    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    if val_loss:
        plt.plot(epochs, val_loss, label="val_loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=150)
    plt.close()

    print(f"[INFO] training loss curve 저장 → {out_png}")


def set_global_seed(seed: int):
    """Python, NumPy, TensorFlow 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


# -------------------------------------------------------
# JSONL → (N, T, D) 변환 + feature 선택
# -------------------------------------------------------
def load_windows_to_array(
    jsonl_path: Path,
    exclude_features: List[str] | None = None,
    pad_value: float = -1.0,
    missing_value: float = -2.0,
) -> Tuple[np.ndarray, List[str], List[int], List[str]]:
    """
    - JSONL 로드 후 (N,T,D)로 스택
    - key 누락 시 기본값은 missing_value로 채움 (안전장치)
    - "전체가 pad/missing"인 윈도우는 학습에 의미 없으므로 스킵
    """
    X_list: List[np.ndarray] = []
    window_ids: List[int] = []
    patterns: List[str] = []
    feature_keys: List[str] = []

    exclude_set = set(exclude_features) if exclude_features else set()

    skipped_all_invalid = 0
    total_read = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            total_read += 1

            seq = obj.get("sequence_group", [])
            if not seq:
                continue

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
                # key 누락 시 missing_value 사용
                arr[t, :] = np.array(
                    [float(pkt.get(k, missing_value)) for k in feature_keys],
                    dtype=np.float32
                )

            # 전체가 pad/missing이면 학습/threshold에 기여 0 → 제거
            valid = (arr != pad_value) & (arr != missing_value)
            if not np.any(valid):
                skipped_all_invalid += 1
                continue

            X_list.append(arr)
            window_ids.append(int(obj.get("window_id", -1)))
            patterns.append(str(obj.get("pattern", "")))

    if not X_list:
        raise RuntimeError("❌ JSONL에서 유효한 window를 하나도 읽지 못했습니다.")

    if skipped_all_invalid > 0:
        print(f"[WARN] 전체가 pad/missing인 윈도우 스킵 = {skipped_all_invalid} (total_read={total_read})")

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
    pad_value: float,
    missing_value: float,
    n_samples: int = 3,
):
    N, T, D = X.shape
    print("\n================= [INSPECT DATA] =================")
    print(f"N (windows) = {N}, T (time steps) = {T}, D (features) = {D}")
    print(f"pad_value = {pad_value}, missing_value = {missing_value}")
    print(f"feature_keys (앞 10개): {feature_keys[:10]}")
    print("===================================================\n")

    X_flat = X.reshape(-1, D)

    print(">>> Feature-wise 통계 (pad/missing 제외):")
    for i, k in enumerate(feature_keys):
        col = X_flat[:, i]
        mask = (col != pad_value) & (col != missing_value)
        if not np.any(mask):
            print(f"  - {k}: (모든 값이 pad/missing)")
            continue
        vals = col[mask]
        print(
            f"  - {k:25s} | "
            f"min={vals.min():.6f}, max={vals.max():.6f}, "
            f"mean={vals.mean():.6f}, std={vals.std():.6f}, "
            f"valid_ratio={len(vals)/len(col):.3f}"
        )

    print("\n>>> 샘플 윈도우 몇 개 보기:")
    n_samples = min(n_samples, N)
    for idx in range(n_samples):
        print(f"\n--- Window #{idx} (global index) ---")
        print(f"window_id = {window_ids[idx]}, pattern = {patterns[idx]}")
        steps = min(5, T)
        for t in range(steps):
            row = X[idx, t]
            if np.all(row == pad_value):
                print(f"  t={t:2d}: [PAD ROW]")
            elif np.all(row == missing_value):
                print(f"  t={t:2d}: [MISSING ROW]")
            else:
                feat_preview_cnt = min(8, D)
                preview = ", ".join(
                    f"{feature_keys[j]}={row[j]:.4f}"
                    for j in range(feat_preview_cnt)
                )
                print(f"  t={t:2d}: {preview}")

    print("\n===================================================\n")


# -------------------------------------------------------
# (추가) BO: Expected Improvement + GP (없으면 random search fallback)
# -------------------------------------------------------
def _normal_pdf(z: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_y: float, xi: float) -> np.ndarray:
    """Minimization 기준 EI."""
    sigma = np.maximum(sigma, 1e-12)
    imp = (best_y - mu) - xi
    z = imp / sigma
    cdf = np.vectorize(_normal_cdf)(z)
    pdf = np.vectorize(_normal_pdf)(z)
    ei = imp * cdf + sigma * pdf
    return np.maximum(ei, 0.0)


def build_lstm_ae_model(
    T: int,
    D: int,
    hidden_dim: int,
    latent_dim: int,
    bidirectional: bool,
):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    encoder_inputs = layers.Input(shape=(T, D), name="encoder_input")

    if bidirectional:
        encoder_output = layers.Bidirectional(
            layers.LSTM(hidden_dim, return_sequences=False),
            name="encoder_bi_lstm",
        )(encoder_inputs)
    else:
        encoder_output = layers.LSTM(
            hidden_dim, return_sequences=False, name="encoder_lstm"
        )(encoder_inputs)

    latent = layers.Dense(latent_dim, name="latent_dense")(encoder_output)

    def repeat_latent(x):
        x = tf.expand_dims(x, axis=1)  # (B, 1, latent_dim)
        x = tf.tile(x, [1, T, 1])      # (B, T, latent_dim)
        return x

    repeated_latent = layers.Lambda(repeat_latent, name="repeat_latent")(latent)

    decoder_output = layers.LSTM(
        hidden_dim, return_sequences=True, name="decoder_lstm"
    )(repeated_latent)

    outputs = layers.TimeDistributed(
        layers.Dense(D), name="decoder_output_dense"
    )(decoder_output)

    model = models.Model(inputs=encoder_inputs, outputs=outputs, name="lstm_autoencoder")
    return model


def make_masked_weighted_mse_tf(pad_val: float, missing_val: float, feat_w: np.ndarray):
    """
    pad(-1) + missing(-2) 둘 다 loss에서 제외 (원소 단위 마스킹)
    """
    import tensorflow as tf
    feat_w_tf = tf.constant(feat_w, dtype=tf.float32)  # (D,)

    def masked_weighted_mse(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        valid = tf.not_equal(y_true, pad_val)
        valid = tf.logical_and(valid, tf.not_equal(y_true, missing_val))
        valid_f = tf.cast(valid, tf.float32)  # (B,T,D)

        se = tf.square(y_pred - y_true) * feat_w_tf  # broadcast -> (B,T,D)
        num = tf.reduce_sum(se * valid_f)
        den = tf.reduce_sum(valid_f) + 1e-8
        return num / den

    return masked_weighted_mse


def train_one_trial(
    X_train: np.ndarray,
    X_val: np.ndarray,
    T: int,
    D: int,
    pad_value: float,
    missing_value: float,
    feature_weights: np.ndarray,
    hidden_dim: int,
    latent_dim: int,
    lr: float,
    batch_size: int,
    bidirectional: bool,
    max_epochs: int,
    patience: int,
    seed: int,
) -> Tuple[float, Dict[str, List[float]], List[np.ndarray]]:
    """
    1개 trial 학습 → best val_loss, history(dict), best_weights 반환
    """
    import tensorflow as tf
    from tensorflow.keras import optimizers
    from tensorflow.keras.callbacks import EarlyStopping

    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)

    model = build_lstm_ae_model(T, D, hidden_dim, latent_dim, bidirectional)
    loss_fn = make_masked_weighted_mse_tf(pad_value, missing_value, feature_weights)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss=loss_fn)

    es = EarlyStopping(
        monitor="val_loss",
        patience=int(patience),
        restore_best_weights=True,
        verbose=0,
    )

    hist = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=int(max_epochs),
        batch_size=int(batch_size),
        shuffle=True,
        callbacks=[es],
        verbose=0,
    )

    train_loss = list(map(float, hist.history.get("loss", [])))
    val_loss = list(map(float, hist.history.get("val_loss", [])))
    best_val = float(np.min(val_loss)) if len(val_loss) > 0 else float(train_loss[-1])

    history_dict = {"train_loss": train_loss, "val_loss": val_loss}
    best_weights = model.get_weights()

    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return best_val, history_dict, best_weights


def bayes_optimize_hparams(
    X_train: np.ndarray,
    X_val: np.ndarray,
    T: int,
    D: int,
    pad_value: float,
    missing_value: float,
    feature_weights: np.ndarray,
    bidirectional: bool,
    seed: int,
    n_trials: int,
    n_init: int,
    n_candidates: int,
    xi: float,
    max_epochs: int,
    patience: int,
) -> Tuple[Dict[str, Any], Dict[str, List[float]], List[np.ndarray], List[Dict[str, Any]]]:
    """
    GP + EI 베이지안 최적화.
    - sklearn 없으면 random search로 폴백.
    반환:
      best_params, best_history, best_weights, trials_log
    """
    rng = np.random.default_rng(seed)

    # 탐색 공간
    HIDDEN_CHOICES = [32, 64, 128, 256]
    LATENT_CHOICES = [16, 32, 64, 128]
    BATCH_CHOICES = [32, 64, 128]
    LR_MIN, LR_MAX = 1e-4, 5e-3  # log-uniform

    def sample_params() -> Dict[str, Any]:
        hd = int(rng.choice(HIDDEN_CHOICES))
        ld = int(rng.choice(LATENT_CHOICES))
        bs = int(rng.choice(BATCH_CHOICES))
        log_lr = rng.uniform(np.log10(LR_MIN), np.log10(LR_MAX))
        lr = float(10 ** log_lr)
        return {"hidden_dim": hd, "latent_dim": ld, "batch_size": bs, "lr": lr}

    def encode(p: Dict[str, Any]) -> np.ndarray:
        return np.array([p["hidden_dim"], p["latent_dim"], np.log10(p["lr"]), p["batch_size"]], dtype=float)

    trials_log: List[Dict[str, Any]] = []

    best_y = float("inf")
    best_params: Dict[str, Any] = {}
    best_history: Dict[str, List[float]] = {}
    best_weights: List[np.ndarray] = []

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
        gp_ok = True
    except Exception as e:
        print(f"[WARN] scikit-learn GP 사용 불가 → random search로 폴백합니다. ({e})")
        gp_ok = False

    X_obs: List[np.ndarray] = []
    y_obs: List[float] = []

    if gp_ok:
        kernel = (
            C(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(4), nu=2.5)
            + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
        )
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)

    for t in range(int(n_trials)):
        if (not gp_ok) or (t < int(n_init)) or (len(y_obs) < 2):
            params = sample_params()
            reason = "random_init" if gp_ok else "random_fallback"
        else:
            X_train_gp = np.stack(X_obs, axis=0)
            y_train_gp = np.array(y_obs, dtype=float)
            gpr.fit(X_train_gp, y_train_gp)

            cand_params = [sample_params() for _ in range(int(n_candidates))]
            X_cand = np.stack([encode(p) for p in cand_params], axis=0)
            mu, sigma = gpr.predict(X_cand, return_std=True)
            ei = expected_improvement(mu, sigma, best_y, xi=float(xi))
            best_idx = int(np.argmax(ei))
            params = cand_params[best_idx]
            reason = "ei_max"

        val_loss, hist_dict, weights = train_one_trial(
            X_train=X_train,
            X_val=X_val,
            T=T,
            D=D,
            pad_value=pad_value,
            missing_value=missing_value,
            feature_weights=feature_weights,
            hidden_dim=params["hidden_dim"],
            latent_dim=params["latent_dim"],
            lr=params["lr"],
            batch_size=params["batch_size"],
            bidirectional=bidirectional,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
        )

        X_obs.append(encode(params))
        y_obs.append(val_loss)

        rec = dict(params)
        rec.update({"trial": t + 1, "best_val_loss": float(val_loss), "pick": reason})
        trials_log.append(rec)

        print(
            f"[BO] trial={t+1}/{n_trials} pick={reason} "
            f"hidden={params['hidden_dim']} latent={params['latent_dim']} "
            f"bs={params['batch_size']} lr={params['lr']:.6g} "
            f"best_val_loss={val_loss:.6g}"
        )

        if val_loss < best_y:
            best_y = val_loss
            best_params = dict(params)
            best_history = hist_dict
            best_weights = weights

    return best_params, best_history, best_weights, trials_log


# -------------------------------------------------------
# main
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_jsonl", required=True, help="pad_pattern_features_by_index.py 결과 JSONL 경로")
    parser.add_argument("-o", "--output_dir", required=True, help="모델 및 로그를 저장할 디렉토리")

    parser.add_argument("--epochs", type=int, default=50, help="학습 epoch 수 (default: 50)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size (default: 64)")
    parser.add_argument("--hidden_dim", type=int, default=128, help="LSTM hidden dim (default: 128)")
    parser.add_argument("--latent_dim", type=int, default=64, help="latent dim (default: 64)")
    parser.add_argument("--num_layers", type=int, default=1, help="LSTM layer 수 (encoder에만 적용, default: 1)")
    parser.add_argument("--bidirectional", action="store_true", help="encoder LSTM을 bidirectional로 사용할지 여부")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="validation 비율 (default: 0.2)")

    parser.add_argument("--pad_value", type=float, default=-1.0, help="패딩 값 (loss/error 계산에서 제외, default: -1)")
    parser.add_argument("--missing_value", type=float, default=-2.0, help="실데이터 없음(missing) sentinel 값 (loss/error 계산에서 제외, default: -2)")

    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu (TensorFlow 자동 선택; 이 값은 로그용)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 (default: 42)")

    parser.add_argument(
        "--exclude-features",
        nargs="+",
        default=None,
        help="학습에서 제외할 feature 이름 리스트 (공백으로 구분). 예: --exclude-features protocol delta_t",
    )
    parser.add_argument(
        "--exclude-file",
        type=str,
        default=None,
        help="학습에서 제외할 feature 이름을 줄 단위로 적어둔 txt 파일 경로. 빈 줄/#줄 무시.",
    )
    parser.add_argument("--inspect-only", action="store_true", help="데이터 요약 출력만 하고 학습은 수행하지 않음")

    parser.add_argument(
        "--feature-weights-file",
        type=str,
        default=None,
        help="각 feature별 가중치를 정의한 TXT 파일 경로. 형식: 'feature_name weight'. 미정의는 1.0.",
    )

    # =========================
    # Bayesian Optimization 옵션
    # =========================
    parser.add_argument(
        "--bayes-opt",
        action="store_true",
        help="베이지안 최적화로 hidden_dim/latent_dim/lr/batch_size 탐색 후 best 1회 결과를 최종 산출물로 저장",
    )
    parser.add_argument("--bo-trials", type=int, default=15, help="BO 총 trial 수 (default: 15)")
    parser.add_argument("--bo-init-trials", type=int, default=5, help="BO 초기 랜덤 trial 수 (default: 5)")
    parser.add_argument("--bo-candidates", type=int, default=800, help="각 BO step에서 EI 후보 개수 (default: 800)")
    parser.add_argument("--bo-xi", type=float, default=0.01, help="Expected Improvement 파라미터 xi (default: 0.01)")

    # ✅ BO 속도 개선 옵션
    parser.add_argument("--bo-epochs", type=int, default=30, help="BO trial에서 사용할 최대 epoch (default: 30)")
    parser.add_argument("--bo-patience", type=int, default=3, help="BO trial early stop patience (default: 3)")
    parser.add_argument("--bo-train-subset", type=float, default=0.3, help="BO trial용 train subset 비율 (0~1, default: 0.3)")
    parser.add_argument("--bo-val-subset", type=float, default=1.0, help="BO trial용 val subset 비율 (0~1, default: 1.0)")
    parser.add_argument("--bo-final-refit", action="store_true",
                        help="BO best hyperparam으로 전체 데이터 최종 1회 재학습(권장)")

    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(int(args.seed))
    print(f"[INFO] Random seed = {args.seed}")

    import tensorflow as tf
    from tensorflow.keras import optimizers
    from tensorflow.keras.callbacks import EarlyStopping

    print(f"[INFO] TensorFlow version: {tf.__version__}")
    print(f"[INFO] device flag = {args.device} (실제 사용 디바이스는 TensorFlow가 자동 선택)")

    pad_value = float(args.pad_value)
    missing_value = float(args.missing_value)

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
                    if not name or name.startswith("#"):
                        continue
                    exclude_from_file.append(name)

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
        pad_value=pad_value,
        missing_value=missing_value,
    )
    N, T, D = X.shape
    print(f"[INFO] 데이터 shape: N={N}, T={T}, D={D}")
    print(f"[INFO] 최종 feature 수: {len(feature_keys)}")
    print(f"[INFO] pad_value={pad_value}, missing_value={missing_value}")

    # -----------------------------
    # feature별 가중치 설정 (기본 1.0) + 파일에서 override
    # -----------------------------
    feature_weights = np.ones(len(feature_keys), dtype=np.float32)

    if args.feature_weights_file:
        fw_path = Path(args.feature_weights_file)
        if not fw_path.exists():
            print(f"[WARN] feature-weights-file 이 존재하지 않습니다: {fw_path}")
        else:
            print(f"[INFO] feature-weights-file 로드: {fw_path}")
            weight_map: dict[str, float] = {}
            with fw_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        print(f"[WARN] 잘못된 weight 라인(무시): {line}")
                        continue
                    name = parts[0]
                    try:
                        w = float(parts[1])
                    except ValueError:
                        print(f"[WARN] weight 파싱 실패(무시): {line}")
                        continue
                    weight_map[name] = w

            for i, k in enumerate(feature_keys):
                if k in weight_map:
                    feature_weights[i] = weight_map[k]

    print("[INFO] feature-wise weights (앞 10개):")
    for k, w in list(zip(feature_keys, feature_weights))[:10]:
        print(f"  - {k:25s}: {w}")

    # inspect-only
    if args.inspect_only:
        inspect_data(
            X,
            feature_keys,
            window_ids,
            patterns,
            pad_value=pad_value,
            missing_value=missing_value,
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
    val_ratio = float(args.val_ratio)
    indices = np.arange(N)
    np.random.shuffle(indices)
    split = int(N * (1.0 - val_ratio))
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train = X[train_idx]
    X_val = X[val_idx]

    print(f"[INFO] train N = {X_train.shape[0]}, val N = {X_val.shape[0]}")

    # ✅ BO 속도 개선: subset 구성
    def _clip_ratio(r: float) -> float:
        try:
            r = float(r)
        except Exception:
            return 1.0
        return max(0.0, min(1.0, r))

    def take_subset(rng: np.random.Generator, X_in: np.ndarray, ratio: float) -> np.ndarray:
        ratio = _clip_ratio(ratio)
        if ratio >= 1.0:
            return X_in
        n = max(1, int(round(len(X_in) * ratio)))
        idx = rng.choice(len(X_in), size=n, replace=False)
        return X_in[idx]

    rng = np.random.default_rng(int(args.seed))
    if args.bayes_opt:
        X_train_bo = take_subset(rng, X_train, args.bo_train_subset)
        X_val_bo = take_subset(rng, X_val, args.bo_val_subset)
        print(f"[INFO] BO subset: train={len(X_train_bo)}/{len(X_train)}, val={len(X_val_bo)}/{len(X_val)}")
        print(f"[INFO] BO trial: epochs={args.bo_epochs}, patience={args.bo_patience}")
    else:
        X_train_bo = X_train
        X_val_bo = X_val

    # -------------------------------------------------------
    # 3~5) 학습
    # -------------------------------------------------------
    history: Dict[str, List[float]]
    model: Any

    if args.bayes_opt:
        print("[INFO] --bayes-opt 활성화: Bayesian Optimization을 진행합니다.")

        best_params, best_history, best_weights, trials_log = bayes_optimize_hparams(
            X_train=X_train_bo,
            X_val=X_val_bo,
            T=T,
            D=D,
            pad_value=pad_value,
            missing_value=missing_value,
            feature_weights=feature_weights,
            bidirectional=bool(args.bidirectional),
            seed=int(args.seed),
            n_trials=int(args.bo_trials),
            n_init=int(args.bo_init_trials),
            n_candidates=int(args.bo_candidates),
            xi=float(args.bo_xi),
            max_epochs=int(args.bo_epochs),
            patience=int(args.bo_patience),
        )

        bo_path = output_dir / "bayes_opt_trials.json"
        with bo_path.open("w", encoding="utf-8") as f:
            json.dump({"best_params": best_params, "trials": trials_log}, f, indent=2, ensure_ascii=False)
        print(f"[INFO] bayes_opt_trials.json 저장 → {bo_path}")

        # best hyperparams 반영
        args.hidden_dim = int(best_params["hidden_dim"])
        args.latent_dim = int(best_params["latent_dim"])
        args.batch_size = int(best_params["batch_size"])
        args.lr = float(best_params["lr"])

        print(
            "[INFO] BO best hyperparams:",
            f"hidden_dim={args.hidden_dim}, latent_dim={args.latent_dim}, "
            f"batch_size={args.batch_size}, lr={args.lr:.6g}"
        )

        if args.bo_final_refit:
            # best로 전체 데이터 재학습(권장)
            model = build_lstm_ae_model(
                T=T,
                D=D,
                hidden_dim=int(args.hidden_dim),
                latent_dim=int(args.latent_dim),
                bidirectional=bool(args.bidirectional),
            )
            loss_fn = make_masked_weighted_mse_tf(pad_value, missing_value, feature_weights)
            model.compile(optimizer=optimizers.Adam(learning_rate=args.lr), loss=loss_fn)

            es = EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            )

            print("[INFO] BO best로 full refit 시작")
            history_obj = model.fit(
                X_train, X_train,
                validation_data=(X_val, X_val),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                shuffle=True,
                callbacks=[es],
                verbose=1,
            )

            history = {
                "train_loss": list(map(float, history_obj.history.get("loss", []))),
                "val_loss": list(map(float, history_obj.history.get("val_loss", []))),
            }
        else:
            # BO trial best weights 그대로 사용(빠르지만, subset 기반일 수 있음)
            model = build_lstm_ae_model(
                T=T,
                D=D,
                hidden_dim=int(args.hidden_dim),
                latent_dim=int(args.latent_dim),
                bidirectional=bool(args.bidirectional),
            )
            loss_fn = make_masked_weighted_mse_tf(pad_value, missing_value, feature_weights)
            model.compile(optimizer=optimizers.Adam(learning_rate=args.lr), loss=loss_fn)
            model.set_weights(best_weights)
            history = best_history

        model.summary()

    else:
        # 단일 학습
        model = build_lstm_ae_model(
            T=T,
            D=D,
            hidden_dim=int(args.hidden_dim),
            latent_dim=int(args.latent_dim),
            bidirectional=bool(args.bidirectional),
        )
        model.summary()

        loss_fn = make_masked_weighted_mse_tf(pad_value, missing_value, feature_weights)
        model.compile(optimizer=optimizers.Adam(learning_rate=float(args.lr)), loss=loss_fn)

        es = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        )

        print("[INFO] Keras model.fit() 시작")
        history_obj = model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            shuffle=True,
            callbacks=[es],
            verbose=1,
        )

        history = {
            "train_loss": list(map(float, history_obj.history.get("loss", []))),
            "val_loss": list(map(float, history_obj.history.get("val_loss", []))),
        }

    # -------------------------------------------------------
    # 6) train set reconstruction error 기반 threshold 계산
    # -------------------------------------------------------
    print("[INFO] train set reconstruction error 계산...")
    X_train_pred = model.predict(
        X_train,
        batch_size=int(args.batch_size),
        verbose=1,
    )

    errors_train = compute_window_errors(
        X_train,
        X_train_pred,
        pad_value=pad_value,
        missing_value=missing_value,
        feature_weights=feature_weights,
    )

    print(
        f"[INFO] train error 통계: "
        f"mean={errors_train.mean():.6f}, "
        f"std={errors_train.std():.6f}, "
        f"min={errors_train.min():.6f}, "
        f"max={errors_train.max():.6f}"
    )

    threshold_p99 = float(np.percentile(errors_train, 99.0))
    threshold_mu3 = float(errors_train.mean() + 3.0 * errors_train.std())

    print(f"[INFO] 99th percentile threshold = {threshold_p99:.6f}")
    print(f"[INFO] mean + 3*std threshold    = {threshold_mu3:.6f}")

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

    # -------------------------------------------------------
    # 7) 모델/설정/로그 저장
    # -------------------------------------------------------
    model_path = output_dir / "model.h5"
    model.save(model_path)
    print(f"[INFO] 모델 저장 → {model_path}")

    # feature weight 파일 복사 + config에는 상대 파일명만 기록
    feature_weights_file_for_config = None
    if args.feature_weights_file:
        src = Path(args.feature_weights_file)
        if src.exists():
            dst = output_dir / src.name
            try:
                shutil.copy2(src, dst)
                print(f"[INFO] feature_weights 파일 복사 → {dst}")
                feature_weights_file_for_config = dst.name
            except Exception as e:
                print(f"[WARN] feature_weights 파일 복사 실패: {e}")
                feature_weights_file_for_config = str(src)
        else:
            print(f"[WARN] feature_weights 파일이 존재하지 않습니다: {src}")
            feature_weights_file_for_config = str(src)

    config = {
        "input_jsonl": str(input_path),
        "N": int(N),
        "T": int(T),
        "D": int(D),
        "hidden_dim": int(args.hidden_dim),
        "latent_dim": int(args.latent_dim),
        "num_layers": int(args.num_layers),
        "bidirectional": bool(args.bidirectional),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "val_ratio": float(args.val_ratio),
        "pad_value": float(pad_value),
        "missing_value": float(missing_value),
        "device_flag": str(args.device),
        "framework": "tensorflow.keras",
        "seed": int(args.seed),
        "exclude_features": merged_exclude,
        "feature_weights_file": feature_weights_file_for_config,

        # BO 메타
        "bayes_opt": bool(args.bayes_opt),
        "bo_trials": int(args.bo_trials) if args.bayes_opt else None,
        "bo_init_trials": int(args.bo_init_trials) if args.bayes_opt else None,
        "bo_candidates": int(args.bo_candidates) if args.bayes_opt else None,
        "bo_xi": float(args.bo_xi) if args.bayes_opt else None,

        # BO 속도 옵션 기록
        "bo_epochs": int(args.bo_epochs) if args.bayes_opt else None,
        "bo_patience": int(args.bo_patience) if args.bayes_opt else None,
        "bo_train_subset": float(args.bo_train_subset) if args.bayes_opt else None,
        "bo_val_subset": float(args.bo_val_subset) if args.bayes_opt else None,
        "bo_final_refit": bool(args.bo_final_refit) if args.bayes_opt else None,
    }

    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[INFO] config 저장 → {config_path}")

    log_path = output_dir / "train_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[INFO] train_log 저장 → {log_path}")

    loss_png_path = output_dir / "training_loss_curve.png"
    save_training_loss_curve(history, loss_png_path)


if __name__ == "__main__":
    main()


"""
예시 실행:

기본(기존과 동일):
python 2.LSTM_AE.py \
  -i "../result/pattern_features_padded_0.jsonl" \
  -o "../../result_train/data" \
  --epochs 400 --batch_size 64 \
  --hidden_dim 64 --latent_dim 64 \
  --pad_value -1 --device cuda --seed 42 \
  --exclude-file "../data/exclude.txt" \
  --feature-weights-file "../data/feature_weights.txt"

BO 빠르게 + 최종 full refit(권장):
python 3.LSTM_AE_basian.py \
  -i "../result/pattern_features_padded_0.jsonl" \
  -o "../../result_train/data_bayes" \
  --bayes-opt \
  --bo-trials 20 --bo-init-trials 6 --bo-candidates 300 --bo-xi 0.01 \
  --bo-epochs 30 --bo-patience 3 --bo-train-subset 0.3 --bo-val-subset 1.0 \
  --bo-final-refit \
  --epochs 300 --batch_size 64 \
  --pad_value -1 --seed 42 \
  --exclude-file "../data/exclude.txt" \
  --feature-weights-file "../data/feature_weights.txt"

inspect 모드:
python 2.LSTM_AE.py \
  -i "../result/pattern_features_padded_0.jsonl" \
  -o "../../result_train/inspect" \
  --pad_value -1 \
  --exclude-file "../data/exclude.txt" \
  --inspect-only


python 3.LSTM_AE_basian.py -i "../result/pattern_features_padded_0.jsonl" -o "../../result_train/data_bayes" --bayes-opt --bo-trials 8 --bo-init-trials 6 --bo-candidates 120 --bo-xi 0.01 --bo-epochs 20 --bo-patience 3 --bo-train-subset 0.15 --bo-final-refit --epochs 300 --batch_size 64 --hidden_dim 64 --latent_dim 64 --pad_value -1 --seed 42 --exclude-file "../data/exclude.txt" --feature-weights-file "../data/feature_weights.txt"


python 3.LSTM_AE_basian.py -i "../result/pattern_features_padded_0.jsonl" -o "../../result_train/data_bayes-normal" --bayes-opt --bo-trials 8 --bo-init-trials 6 --bo-candidates 120 --bo-xi 0.01 --bo-epochs 20 --bo-patience 3 --bo-train-subset 0.15 --bo-final-refit --epochs 300 --batch_size 64 --hidden_dim 64 --latent_dim 64 --pad_value -1 --seed 42 --exclude-file "../data/exclude-normal.txt" --feature-weights-file "../data/feature_weights-normal.txt"


python 3.LSTM_AE_basian.py -i "../result/pattern_features_padded_0.jsonl" -o "../../result_train/data_bayes-normal" --bayes-opt --bo-trials 8 --bo-init-trials 6 --bo-candidates 120 --bo-xi 0.01 --bo-epochs 20 --bo-patience 3 --bo-train-subset 0.15 --bo-final-refit --epochs 300 --batch_size 64 --hidden_dim 64 --latent_dim 64 --pad_value -1 --seed 42 --exclude-file "../data/exclude-normal.txt" --feature-weights-file "../data/feature_weights-normal.txt"

python 3.LSTM_AE_basian.py -i "../result/pattern_features_padded_0.jsonl" -o "../../result_train/data_bayes-normal" --bayes-opt --bo-trials 8 --bo-init-trials 6 --bo-candidates 120 --bo-xi 0.01 --bo-epochs 20 --bo-patience 3 --bo-train-subset 0.15 --bo-final-refit --epochs 300 --batch_size 64 --hidden_dim 32 --latent_dim 8 --pad_value -1 --seed 42 --exclude-file "../data/exclude-normal.txt" --feature-weights-file "../data/feature_weights-normal.txt"


python 3.LSTM_AE.py -i "../result/pattern_features_padded_0.jsonl" -o "../../result_train/data" --epochs 400 --batch_size 64 --hidden_dim 32 --latent_dim 8 --lr 5e-4 --pad_value -1 --seed 42 --exclude-file "../data/exclude.txt" --feature-weights-file "../data/feature_weights.txt"


"""
