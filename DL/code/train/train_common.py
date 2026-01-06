#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping
from padding import pad_window


IGNORE_VALUE = -1.0


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


def infer_global_window_size(jsonl_path: Path) -> int:
    mx = -1
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                win = json.loads(line)
            except Exception:
                continue
            if not isinstance(win, dict):
                continue
            for v in (win.get("index", []) or []):
                try:
                    iv = int(v)
                    if iv > mx:
                        mx = iv
                except Exception:
                    pass
    return (mx + 1) if mx >= 0 else 0


def load_feature_policy_ox(policy_file: Optional[str]) -> Tuple[Set[str], Dict[str, float]]:
    exclude_set: Set[str] = set()
    weight_map: Dict[str, float] = {}

    if not policy_file:
        return exclude_set, weight_map

    p = Path(policy_file)
    if not p.exists():
        return exclude_set, weight_map

    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0].upper() in ("O", "X"):
                flag = parts[0].upper()
                if len(parts) < 2:
                    continue
                name = parts[1]
                w = 1.0
                if len(parts) >= 3:
                    try:
                        w = float(parts[2])
                    except Exception:
                        w = 1.0
            else:
                flag = "O"
                name = parts[0]
                w = 1.0
                if len(parts) >= 2:
                    try:
                        w = float(parts[1])
                    except Exception:
                        w = 1.0

            if flag == "X":
                exclude_set.add(name)
                continue

            weight_map[name] = float(w)

    return exclude_set, weight_map


def build_feature_weights(feature_keys: List[str], weight_map: Dict[str, float]) -> np.ndarray:
    w = np.ones(len(feature_keys), dtype=np.float32)
    for i, k in enumerate(feature_keys):
        if k in weight_map:
            w[i] = float(weight_map[k])
    return w


def make_masked_weighted_mse(feature_weights: np.ndarray):
    import tensorflow as tf
    fw = tf.constant(feature_weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        valid = tf.not_equal(y_true, IGNORE_VALUE)
        se = tf.square(y_pred - y_true) * fw
        se = se * tf.cast(valid, tf.float32)
        denom = tf.reduce_sum(tf.cast(valid, tf.float32)) + 1e-8
        return tf.reduce_sum(se) / denom

    return loss_fn


def compute_window_errors(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    feature_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    valid = (X_true != IGNORE_VALUE)
    se = (X_pred - X_true) ** 2
    if feature_weights is not None:
        se = se * feature_weights[np.newaxis, np.newaxis, :]
    se = se * valid.astype(np.float32)
    denom = np.sum(valid, axis=(1, 2)) + 1e-8
    num = np.sum(se, axis=(1, 2))
    return num / denom


def save_training_loss_curve(history: Dict[str, List[float]], out_png: Path) -> None:
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    if not train_loss:
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


def load_and_pad_to_array(
    input_jsonl: Path,
    window_size: int,
    pad_value: float,
    exclude_set: Optional[Set[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    X_list: List[np.ndarray] = []
    feature_keys: List[str] = []
    ex = exclude_set or set()

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                win = json.loads(line)
            except Exception:
                continue
            if not isinstance(win, dict):
                continue

            padded = pad_window(win, window_size=window_size, pad_value=pad_value)
            if not padded:
                continue

            seq = padded.get("sequence_group", [])
            if not seq:
                continue

            if not feature_keys:
                all_keys = sorted(list(seq[0].keys()))
                feature_keys = [k for k in all_keys if k not in ex]
                if not feature_keys:
                    raise RuntimeError("exclude로 인해 남는 feature가 없습니다.")

            T = len(seq)
            D = len(feature_keys)
            arr = np.zeros((T, D), dtype=np.float32)

            for t, pkt in enumerate(seq):
                for d, k in enumerate(feature_keys):
                    arr[t, d] = float(pkt.get(k, pad_value))

            X_list.append(arr)

    if not X_list:
        raise RuntimeError("입력 JSONL에서 유효한 window를 읽지 못했습니다.")

    return np.stack(X_list, axis=0), feature_keys


def copy_policy_file_if_any(policy_file: Optional[str], out_dir: Path) -> Optional[str]:
    if not policy_file:
        return None
    src = Path(policy_file)
    if not src.exists():
        return str(src)

    dst = out_dir / src.name
    try:
        shutil.copy2(src, dst)
        return dst.name
    except Exception:
        return str(src)


def run_training_common(
    *,
    input_jsonl: str,
    output_dir: str,
    window_size: Optional[int],
    pad_value: float,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    latent_dim: int,
    bidirectional: bool,
    lr: float,
    val_ratio: float,
    seed: int,
    feature_policy_file: Optional[str],
    model_type: str,
    build_model_fn: Callable[[int, int, int, int, bool, int], Any],
    predict_fn: Callable[[Any, np.ndarray, int], np.ndarray],
    extra_config: Optional[Dict[str, Any]] = None,
) -> None:
    input_path = Path(input_jsonl)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(int(seed))

    if window_size is None:
        inferred = infer_global_window_size(input_path)
        if inferred <= 0:
            raise RuntimeError("window_size 추론 실패")
        window_size = inferred
    else:
        window_size = int(window_size)
        if window_size <= 0:
            raise RuntimeError("window_size는 1 이상이어야 합니다.")

    pad_value = float(pad_value)

    exclude_set, weight_map = load_feature_policy_ox(feature_policy_file)

    X, feature_keys = load_and_pad_to_array(
        input_path,
        window_size=window_size,
        pad_value=pad_value,
        exclude_set=exclude_set,
    )
    N, T, D = X.shape

    # --- feature weights (loss 가중치) 생성 ---
    feature_weights = build_feature_weights(feature_keys, weight_map)

    # --- feature keys/weights 저장 (재현성/추론 일관성 보장) ---
    (out_dir / "feature_keys.txt").write_text("\n".join(feature_keys) + "\n", encoding="utf-8")
    np.save(out_dir / "feature_weights.npy", feature_weights)

    resolved_weights = {k: float(feature_weights[i]) for i, k in enumerate(feature_keys)}
    (out_dir / "feature_weights.json").write_text(
        json.dumps(resolved_weights, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(N * (1.0 - float(val_ratio)))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train = X[train_idx]
    X_val = X[val_idx]

    from tensorflow.keras import optimizers

    model = build_model_fn(
        T,
        D,
        int(hidden_dim),
        int(latent_dim),
        bool(bidirectional),
        int(X_train.shape[0]),
    )

    # --- model_summary.txt 관련 로직 제거됨 ---

    model.compile(
        optimizer=optimizers.Adam(learning_rate=float(lr)),
        loss=make_masked_weighted_mse(feature_weights),
    )

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

    hist = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=int(epochs),
        batch_size=int(batch_size),
        shuffle=True,
        callbacks=[es],
        verbose=1,
    )

    history = {
        "train_loss": list(map(float, hist.history.get("loss", []))),
        "val_loss": list(map(float, hist.history.get("val_loss", []))),
    }

    X_train_pred = predict_fn(model, X_train, int(batch_size))
    errors_train = compute_window_errors(X_train, X_train_pred, feature_weights=feature_weights)

    threshold_info = {
        "threshold_p99": float(np.percentile(errors_train, 99.0)),
        "threshold_mu3": float(errors_train.mean() + 3.0 * errors_train.std()),
        "stats": {
            "mean": float(errors_train.mean()),
            "std": float(errors_train.std()),
            "min": float(errors_train.min()),
            "max": float(errors_train.max()),
        },
        "model_type": str(model_type),
    }
    (out_dir / "threshold.json").write_text(
        json.dumps(threshold_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 모델 파라미터(신경망 weights)는 여기 저장됨
    model.save(out_dir / "model.h5")

    policy_file_for_config = copy_policy_file_if_any(feature_policy_file, out_dir)

    config = {
        "input_jsonl": str(input_path),
        "N": int(N),
        "T": int(T),
        "D": int(D),
        "hidden_dim": int(hidden_dim),
        "latent_dim": int(latent_dim),
        "bidirectional": bool(bidirectional),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "val_ratio": float(val_ratio),
        "seed": int(seed),
        "window_size": int(window_size),
        "pad_value": float(pad_value),
        "framework": "tensorflow.keras",
        "loss_ignore_value": float(IGNORE_VALUE),
        "feature_policy_file": policy_file_for_config,
        "excluded_features": sorted(list(exclude_set)),
        "model_type": str(model_type),
        "feature_weights_npy": "feature_weights.npy",
        "feature_weights_json": "feature_weights.json",
    }
    if extra_config:
        config.update(extra_config)

    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "train_log.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    save_training_loss_curve(history, out_dir / "training_loss_curve.png")
