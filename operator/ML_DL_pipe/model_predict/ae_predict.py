#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)

USE_HARD_THRESHOLD: bool = True
HARD_THRESHOLD: float = 0.32

JsonDict = Dict[str, Any]


def _safe_float(x: Any, default: float) -> float:
    try:
        if x is None:
            return float(default)
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _load_threshold_from_dir(model_dir: Path) -> Optional[float]:
    p = model_dir / "threshold.json"
    if not p.exists():
        return None
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
        for k in ("threshold", "threshold_p99", "threshold_mu3"):
            if k in cfg:
                return float(cfg[k])
    except Exception:
        return None
    return None


def _load_feature_keys(model_dir: Path) -> List[str]:
    p = model_dir / "feature_keys.txt"
    if not p.exists():
        raise FileNotFoundError(f"feature_keys.txt 없음: {p}")
    raw = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [rk.split()[-1] for rk in raw]
    if not keys:
        raise ValueError(f"feature_keys.txt가 비어 있습니다: {p}")
    return keys


def load_feature_weights(
    config: Dict[str, Any],
    feature_keys: List[str],
    model_dir: Path,
    cli_path: Optional[str] = None,
) -> np.ndarray:
    weights = np.ones(len(feature_keys), dtype=np.float32)

    candidates: List[Path] = []
    if cli_path:
        candidates.append(Path(cli_path))

    fw_cfg = config.get("feature_weights_file")
    if fw_cfg:
        p_cfg = Path(str(fw_cfg))
        candidates.append(p_cfg)
        if not p_cfg.is_absolute():
            candidates.append((model_dir / p_cfg).resolve())

    fw_path: Optional[Path] = None
    for c in candidates:
        try:
            if c.exists():
                fw_path = c
                break
        except Exception:
            continue

    if fw_path is None:
        return weights

    weight_map: Dict[str, float] = {}
    try:
        for line in fw_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            try:
                w = float(parts[1])
            except Exception:
                continue
            weight_map[name] = w
    except Exception:
        return weights

    for i, k in enumerate(feature_keys):
        if k in weight_map:
            weights[i] = float(weight_map[k])

    return weights


def load_lstm_ae_bundle(
    model_dir: Union[str, Path],
    feature_weights_file: Optional[str] = None,
) -> Dict[str, Any]:
    model_dir = Path(model_dir).expanduser().resolve()

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json 없음: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    T = config.get("T")
    if T is None:
        raise ValueError("config.json에 'T'가 없습니다.")
    T = int(T)

    pad_value = float(config.get("pad_value", -1.0))
    missing_value = float(config.get("missing_value", -2.0))

    model_path = model_dir / "model.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"model.h5 없음: {model_path}")

    from tensorflow.keras.models import load_model  # local import

    def repeat_latent(x):
        x = tf.expand_dims(x, axis=1)
        return tf.tile(x, [1, T, 1])

    model = load_model(model_path, compile=False, custom_objects={"repeat_latent": repeat_latent})

    feature_keys = _load_feature_keys(model_dir)
    feature_index = {k: i for i, k in enumerate(feature_keys)}

    file_threshold = _load_threshold_from_dir(model_dir)
    threshold = float(HARD_THRESHOLD) if USE_HARD_THRESHOLD else (float(file_threshold) if file_threshold is not None else None)

    fw = load_feature_weights(
        config=config,
        feature_keys=feature_keys,
        model_dir=model_dir,
        cli_path=feature_weights_file,
    )

    bundle = {
        "model": model,
        "config": config,
        "feature_keys": feature_keys,
        "feature_index": feature_index,
        "pad_value": pad_value,
        "missing_value": missing_value,
        "threshold": threshold,
        "threshold_from_file": file_threshold,
        "feature_weights": fw,
        "model_dir": str(model_dir),
        "model_path": str(model_path),
        "_tf_pad": tf.constant(pad_value, dtype=tf.float32),
        "_tf_miss": tf.constant(missing_value, dtype=tf.float32),
        "_tf_fw": tf.constant(np.asarray(fw, dtype=np.float32), dtype=tf.float32) if fw is not None else None,
        "_ae_scorer_cache": {},
    }
    return bundle


def _get_ae_scorer(bundle: Dict[str, Any], k_eff: int):
    k_eff = int(max(0, k_eff))
    cache = bundle.setdefault("_ae_scorer_cache", {})
    if k_eff in cache:
        return cache[k_eff]

    model = bundle["model"]
    pad_t = bundle["_tf_pad"]
    miss_t = bundle["_tf_miss"]
    fw_t = bundle.get("_tf_fw", None)

    T = int(bundle["config"]["T"])
    D = int(len(bundle["feature_keys"]))

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, T, D], dtype=tf.float32)],
        reduce_retracing=True,
    )
    def scorer(Xb: tf.Tensor):
        recon = model(Xb, training=False)
        se = tf.square(recon - Xb)
        if fw_t is not None:
            se = se * tf.reshape(fw_t, [1, 1, -1])

        valid = tf.logical_and(tf.not_equal(Xb, pad_t), tf.not_equal(Xb, miss_t))
        mask = tf.cast(tf.reduce_any(valid, axis=-1), tf.float32)  # (B,T)

        se_t = tf.reduce_mean(se, axis=-1)                         # (B,T)
        denom = tf.reduce_sum(mask, axis=-1) + 1e-8                # (B,)
        mse = tf.reduce_sum(se_t * mask, axis=-1) / denom          # (B,)
        tmax = tf.reduce_max(se_t, axis=-1)                        # (B,)

        if k_eff > 0:
            se_f = tf.reduce_mean(se, axis=1)                      # (B,D)
            vals, idx = tf.math.top_k(se_f, k=k_eff, sorted=True)
            return mse, tmax, idx, vals
        return mse, tmax

    cache[k_eff] = scorer
    return scorer


def _ensure_btf_np(X: np.ndarray, model: Any) -> np.ndarray:
    """
    TF LSTM 계열 입력은 (B,T,F) 기대.
    일부 파이프라인에서 (B,F,T)로 넘어오는 케이스 방어.
    """
    if X.ndim != 3:
        return X

    exp_f: Optional[int] = None
    try:
        exp_f = int(model.input_shape[-1])
    except Exception:
        exp_f = None

    if exp_f is not None:
        if X.shape[1] == exp_f and X.shape[2] != exp_f:
            return np.transpose(X, (0, 2, 1)).copy()

    # 휴리스틱(자주 나오는 케이스): (B,36,80)->(B,80,36)
    if X.shape[1] < X.shape[2] and X.shape[1] <= 512:
        # feature_dim이 time보다 작을 확률이 높음
        return np.transpose(X, (0, 2, 1)).copy()

    return X


def predict_lstm_ae_windows_sliding(
    prepares: List[Dict[str, Any]],
    bundle: Dict[str, Any],
    *,
    window_size: int = 80,
    step: int = 5,
    batch_size: int = 128,
    threshold_override: Optional[float] = None,
    topk_feature_error: int = 7,
) -> Dict[str, Any]:
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size/step must be > 0")
    if len(prepares) < window_size:
        raise ValueError(f"need >= window_size. got={len(prepares)} window_size={window_size}")

    T = int(bundle["config"]["T"])
    if T != int(window_size):
        raise ValueError(f"DL model T={T} != requested window_size={int(window_size)}")

    feature_keys: List[str] = list(bundle["feature_keys"])
    feat2idx: Dict[str, int] = dict(bundle["feature_index"])

    pad_value = float(bundle["pad_value"])
    missing_value = float(bundle.get("missing_value", -2.0))
    feature_weights = bundle.get("feature_weights", None)

    N = len(prepares)
    D = len(feature_keys)

    num_windows = 1 + (N - window_size) // step
    if num_windows <= 0:
        raise ValueError("num_windows computed as <= 0")

    F = np.full((N, D), float(missing_value), dtype=np.float32)
    mv = float(missing_value)

    for i, pr in enumerate(prepares):
        feat = pr.get("features")
        if not isinstance(feat, dict):
            continue
        for k, v in feat.items():
            j = feat2idx.get(k)
            if j is None:
                continue
            F[i, j] = _safe_float(v, mv)

    fw = None
    if feature_weights is not None:
        fw = np.asarray(feature_weights, dtype=np.float32)
        if fw.shape[0] != D:
            fw = None

    if USE_HARD_THRESHOLD:
        th_f = float(HARD_THRESHOLD)
    else:
        th = threshold_override if threshold_override is not None else bundle.get("threshold", None)
        th_f = float(th) if th is not None else None

    topk = int(max(0, topk_feature_error))
    k_eff = min(topk, D) if topk > 0 else 0
    scorer = _get_ae_scorer(bundle, k_eff)

    mse_all = np.empty((num_windows,), dtype=np.float32)
    temporal_max_all = np.empty((num_windows,), dtype=np.float32)
    is_pred_all = np.full((num_windows,), -1, dtype=np.int32)

    topk_idx_all = np.empty((num_windows, k_eff), dtype=np.int32) if k_eff > 0 else None
    topk_val_all = np.empty((num_windows, k_eff), dtype=np.float32) if k_eff > 0 else None

    model = bundle["model"]

    if num_windows == 1:
        Xb_np = F[None, :, :]
        Xb_np = _ensure_btf_np(Xb_np, model)
        Xb = tf.convert_to_tensor(Xb_np, dtype=tf.float32)
        out = scorer(Xb)

        if k_eff > 0:
            mse_b, tmax_b, idx_b, val_b = out
            topk_idx_all[0, :] = idx_b.numpy()[0].astype(np.int32, copy=False)
            topk_val_all[0, :] = val_b.numpy()[0].astype(np.float32, copy=False)
        else:
            mse_b, tmax_b = out

        mse0 = float(mse_b.numpy()[0])
        tmax0 = float(tmax_b.numpy()[0])
        mse_all[0] = mse0
        temporal_max_all[0] = tmax0
        if th_f is not None:
            is_pred_all[0] = int(mse0 > float(th_f))

    else:
        try:
            W_all = np.lib.stride_tricks.sliding_window_view(F, window_size, axis=0)  # (N-T+1, T, D)
            W_view = W_all[::step]  # (W,T,D)
        except Exception:
            W_view = None

        bs = int(max(1, batch_size))
        for b0 in range(0, num_windows, bs):
            b1 = min(num_windows, b0 + bs)

            if W_view is not None:
                Xb_np = np.ascontiguousarray(W_view[b0:b1, :, :])
            else:
                t_idx = np.arange(window_size, dtype=np.int64)[None, :]
                starts = (np.arange(b0, b1, dtype=np.int64) * int(step))[:, None]
                idx = starts + t_idx
                Xb_np = np.take(F, idx, axis=0)

            Xb_np = _ensure_btf_np(Xb_np, model)
            Xb = tf.convert_to_tensor(Xb_np, dtype=tf.float32)
            out = scorer(Xb)

            if k_eff > 0:
                mse_b, tmax_b, idx_b, val_b = out
                topk_idx_all[b0:b1, :] = idx_b.numpy().astype(np.int32, copy=False)
                topk_val_all[b0:b1, :] = val_b.numpy().astype(np.float32, copy=False)
            else:
                mse_b, tmax_b = out

            mse_b_np = mse_b.numpy().astype(np.float32, copy=False)
            tmax_b_np = tmax_b.numpy().astype(np.float32, copy=False)

            mse_all[b0:b1] = mse_b_np
            temporal_max_all[b0:b1] = tmax_b_np
            if th_f is not None:
                is_pred_all[b0:b1] = (mse_b_np > float(th_f)).astype(np.int32)

    windows: List[Dict[str, Any]] = []
    for w in range(num_windows):
        s = int(w * step)
        e = int(s + window_size - 1)

        fe: Dict[str, float] = {}
        if k_eff > 0 and topk_idx_all is not None and topk_val_all is not None:
            for kk in range(k_eff):
                j = int(topk_idx_all[w, kk])
                fe[feature_keys[j]] = float(topk_val_all[w, kk])

        windows.append(
            {
                "seq_id": int(w + 1),
                "start": int(s),
                "end": int(e),
                "mse": float(mse_all[w]),
                "threshold": th_f,
                "is_anomaly_pred": int(is_pred_all[w]),
                "temporal_error_max": float(temporal_max_all[w]),
                "feature_error": fe,
            }
        )

    return {
        "window_size": int(window_size),
        "step": int(step),
        "threshold": th_f,
        "num_packets": int(N),
        "num_windows": int(num_windows),
        "mse_per_window": mse_all.astype(float).tolist(),
        "is_anomaly_pred": is_pred_all.astype(int).tolist(),
        "windows": windows,
    }
