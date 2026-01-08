#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model_predict/DL_predict.py (Optimized)

- LSTM-AE (TF/Keras) + DL-pattern (Torch) sliding inference
- 속도 최적화 반영:
  1) AE 입력 F 구성: O(N*D) (feature_keys 전체 루프) -> "존재하는 feature만" 채우기 (sparse fill)
  2) AE 슬라이딩 X 생성: Python for -> numpy sliding_window_view (가능 시)
  3) feature_error top-k: argsort -> argpartition
  4) pattern window 내부 compaction: per-t loop -> np.flatnonzero 기반
  5) always_run_pattern=False 이면 "anomaly 윈도우에만" pattern 추론 수행(큰 속도 개선 포인트)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# TensorFlow 최적화 설정
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)

# ============================================================
# HARD-CODED THRESHOLD (EDIT THIS)
# ============================================================
USE_HARD_THRESHOLD: bool = True
HARD_THRESHOLD: float = 0.32  # <- 여기만 원하는 값으로 고정
# ============================================================

# ============================================================
# RISK SCORING (Commercial-style: Impact × Likelihood × Confidence)
# ============================================================
RISK_GAMMA: float = 0.85
RISK_ALPHA: float = 6.0
RISK_ZERO_WHEN_NORMAL_DEFAULT: bool = True
RISK_INCLUDE_DETAIL: bool = False

LIKELIHOOD_W_SEVERITY: float = 1.00
LIKELIHOOD_W_TEMPORAL: float = 0.60
LIKELIHOOD_W_SEMANTIC: float = 0.35

SEVERITY_SIGMOID_K: float = 6.0

CONF_W_ANOMALY: float = 0.65
CONF_W_PATTERN: float = 0.35

IMPACT_DEFAULT: float = 0.60
# ============================================================

JsonDict = Dict[str, Any]


# ============================================================
# Utils
# ============================================================
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


def _clamp(x: Any, lo: float, hi: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = float(lo)
    if v < lo:
        return float(lo)
    if v > hi:
        return float(hi)
    return float(v)


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def _load_threshold_from_dir(model_dir: Path) -> Optional[float]:
    thresh_path = model_dir / "threshold.json"
    if not thresh_path.exists():
        return None

    try:
        with thresh_path.open("r", encoding="utf-8") as f:
            th_cfg = json.load(f)
        for k in ("threshold", "threshold_p99", "threshold_mu3"):
            if k in th_cfg:
                return float(th_cfg[k])
    except Exception:
        return None

    return None


def _load_feature_keys(model_dir: Path) -> List[str]:
    feat_path = model_dir / "feature_keys.txt"
    if not feat_path.exists():
        raise FileNotFoundError(f"feature_keys.txt 없음: {feat_path}")

    with feat_path.open("r", encoding="utf-8") as f:
        raw_keys = [line.strip() for line in f if line.strip()]

    # 원본이 "idx key" 형태면 마지막 토큰만
    feature_keys = [rk.split()[-1] for rk in raw_keys]
    if not feature_keys:
        raise ValueError(f"feature_keys.txt가 비어 있습니다: {feat_path}")
    return feature_keys


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
        p_cfg = Path(fw_cfg)
        candidates.append(p_cfg)
        if not p_cfg.is_absolute():
            candidates.append((model_dir / p_cfg).resolve())

            in_path = config.get("input_jsonl")
            if in_path:
                try:
                    in_path = Path(in_path)
                    train_root = in_path.parent.parent
                    candidates.append((train_root / "data" / p_cfg.name).resolve())
                except Exception:
                    pass

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
        with fw_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
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


# ============================================================
# DL-anomaly (Keras/TensorFlow) bundle
# ============================================================
def load_lstm_ae_bundle(
    model_dir: Union[str, Path],
    feature_weights_file: Optional[str] = None,
) -> Dict[str, Any]:
    model_dir = Path(model_dir)

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json 없음: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    T = config.get("T")
    if T is None:
        raise ValueError("config.json에 'T'가 없습니다. (repeat_latent에 필요)")
    T_for_repeat = int(T)

    pad_value = float(config.get("pad_value", -1.0))
    missing_value = float(config.get("missing_value", -2.0))

    model_path = model_dir / "model.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"model.h5 없음: {model_path}")

    import tensorflow as tf  # local import
    from tensorflow.keras.models import load_model

    def repeat_latent(x):
        x = tf.expand_dims(x, axis=1)
        x = tf.tile(x, [1, T_for_repeat, 1])
        return x

    model = load_model(
        model_path,
        compile=False,
        custom_objects={"repeat_latent": repeat_latent},
    )

    feature_keys = _load_feature_keys(model_dir)
    feature_index = {k: i for i, k in enumerate(feature_keys)}  # ✅ cache

    file_threshold = _load_threshold_from_dir(model_dir)
    threshold = float(HARD_THRESHOLD) if USE_HARD_THRESHOLD else file_threshold

    feature_weights = load_feature_weights(
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
        "feature_weights": feature_weights,
        "model_dir": str(model_dir),
        "model_path": str(model_path),
    }

    # ✅ TF 캐시(매 호출 tf.constant / tf.function 생성 방지)
    bundle["_tf_pad"] = tf.constant(pad_value, dtype=tf.float32)
    bundle["_tf_miss"] = tf.constant(missing_value, dtype=tf.float32)
    fw_np = np.asarray(feature_weights, dtype=np.float32) if feature_weights is not None else None
    bundle["_tf_fw"] = tf.constant(fw_np, dtype=tf.float32) if fw_np is not None else None
    bundle["_tf_t_idx"] = tf.range(T_for_repeat, dtype=tf.int64)[None, :]  # (1,T)
    bundle["_ae_scorer_cache"] = {}  # k_eff별 scorer 캐시

    return bundle

def _get_ae_scorer(bundle: Dict[str, Any], k_eff: int):
    """
    k_eff(top-k)별로 tf.function을 1회만 생성/trace 해서 캐시.
    (dl_window 느림의 큰 원인: 함수 내부에서 매번 @tf.function 생성)
    """
    k_eff = int(max(0, k_eff))
    cache = bundle.setdefault("_ae_scorer_cache", {})
    if k_eff in cache:
        return cache[k_eff]

    model = bundle["model"]
    pad_t = bundle["_tf_pad"]
    miss_t = bundle["_tf_miss"]
    fw_t = bundle.get("_tf_fw", None)

    T = int(bundle.get("config", {}).get("T") or bundle["_tf_t_idx"].shape[1])
    D = int(len(bundle.get("feature_keys") or []))

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
        mask = tf.cast(tf.reduce_any(valid, axis=-1), tf.float32)   # (B,T)

        se_t = tf.reduce_mean(se, axis=-1)                          # (B,T)
        denom = tf.reduce_sum(mask, axis=-1) + 1e-8                 # (B,)
        mse = tf.reduce_sum(se_t * mask, axis=-1) / denom           # (B,)

        tmax = tf.reduce_max(se_t, axis=-1)                         # (B,)

        if k_eff > 0:
            se_f = tf.reduce_mean(se, axis=1)                       # (B,D)
            vals, idx = tf.math.top_k(se_f, k=k_eff, sorted=True)
            return mse, tmax, idx, vals
        return mse, tmax

    cache[k_eff] = scorer
    return scorer


def _compute_window_errors_components(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    pad_value: float,
    missing_value: float,
    feature_weights: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # valid mask (T,D)
    valid_feat = (X_true != pad_value) & (X_true != missing_value)
    not_pad = np.any(valid_feat, axis=-1)  # (W,T)
    mask = not_pad.astype(np.float32)      # (W,T)

    se = (X_pred - X_true) ** 2            # (W,T,D)
    if feature_weights is not None:
        fw = np.asarray(feature_weights, dtype=np.float32)
        se = se * fw[np.newaxis, np.newaxis, :]

    # temporal mean over features
    se_t = np.mean(se, axis=-1)            # (W,T)
    # feature mean over time
    se_f = np.mean(se, axis=1)             # (W,D)

    se_t_masked = se_t * mask
    denom = np.sum(mask, axis=-1) + 1e-8   # (W,)
    mse_per_window = np.sum(se_t_masked, axis=-1) / denom

    return mse_per_window, se_t, se_f


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

    t_build0 = tf.timestamp()  # TF timestamp (ms 정밀도는 낮지만 OK)
    config = bundle.get("config") or {}
    T_cfg = config.get("T")
    if T_cfg is not None and int(T_cfg) != int(window_size):
        raise ValueError(f"DL model T={int(T_cfg)} != requested window_size={int(window_size)}")

    feature_keys: List[str] = list(bundle["feature_keys"])
    feat2idx: Dict[str, int] = dict(bundle.get("feature_index") or {k: i for i, k in enumerate(feature_keys)})

    pad_value = float(bundle["pad_value"])
    missing_value = float(bundle.get("missing_value", -2.0))
    feature_weights = bundle.get("feature_weights", None)

    N = len(prepares)
    D = len(feature_keys)
    num_windows = 1 + (N - window_size) // step
    if num_windows <= 0:
        raise ValueError("num_windows computed as <= 0")

    # -------------------------------
    # 1) Build per-packet matrix F (N,D)
    # -------------------------------
    F = np.full((N, D), float(missing_value), dtype=np.float32)
    _sf = _safe_float
    mv = float(missing_value)

    for i, pr in enumerate(prepares):
        feat = pr.get("features")
        if not isinstance(feat, dict):
            continue
        for k, v in feat.items():
            j = feat2idx.get(k)
            if j is None:
                continue
            F[i, j] = _sf(v, mv)

    fw = None
    if feature_weights is not None:
        fw = np.asarray(feature_weights, dtype=np.float32)
        if fw.shape[0] != D:
            fw = None

    # -------------------------------
    # 2) Threshold
    # -------------------------------
    if USE_HARD_THRESHOLD:
        th_f = float(HARD_THRESHOLD)
    else:
        th = threshold_override if threshold_override is not None else bundle.get("threshold", None)
        th_f = float(th) if th is not None else None

    # -------------------------------
    # 3) Cached scorer (NO retrace per call)
    # -------------------------------
    topk = int(max(0, topk_feature_error))
    k_eff = min(topk, D) if topk > 0 else 0
    scorer = _get_ae_scorer(bundle, k_eff)  # ✅ 핵심: 캐시된 tf.function

    t_build1 = tf.timestamp()

    mse_all = np.empty((num_windows,), dtype=np.float32)
    is_pred_all = np.full((num_windows,), -1, dtype=np.int32)
    temporal_max_all = np.empty((num_windows,), dtype=np.float32)

    topk_idx_all = np.empty((num_windows, k_eff), dtype=np.int32) if k_eff > 0 else None
    topk_val_all = np.empty((num_windows, k_eff), dtype=np.float32) if k_eff > 0 else None

    # -------------------------------
    # 3-a) W=1 fast path (online pipeline에서 매우 흔함)
    # -------------------------------
    t_inf0 = tf.timestamp()
    if num_windows == 1:
        Xb_np = F[None, :, :]  # (1,T,D)  ✅ np.take/idx 계산 제거
        Xb = tf.convert_to_tensor(Xb_np, dtype=tf.float32)

        # --- FIX: align axis order for TF LSTM-AE (batch, timesteps, features) ---
        # import numpy as np
        # import tensorflow as tf

        # def _ensure_btf(X, model=None):
        #     """
        #     Ensure shape is (B, T, F) for TF LSTM models.
        #     If incoming is (B, F, T), transpose to (B, T, F).
        #     """
        #     # expected feature dim from model if available
        #     exp_f = None
        #     if model is not None:
        #         try:
        #             exp_f = model.input_shape[-1]  # e.g., 36
        #         except Exception:
        #             exp_f = None

        #     if isinstance(X, tf.Tensor):
        #         s = X.shape
        #         # common case: (B, F, T) and F == exp_f
        #         if s.rank == 3:
        #             if exp_f is not None and s[1] == exp_f and s[2] != exp_f:
        #                 return tf.transpose(X, perm=[0, 2, 1])
        #             # fallback heuristic for your exact error: (B, 36, 80) -> (B, 80, 36)
        #             if s[1] == 36 and s[2] == 80:
        #                 return tf.transpose(X, perm=[0, 2, 1])
        #         return X

        #     X = np.asarray(X, dtype=np.float32)
        #     if X.ndim == 3:
        #         if exp_f is not None and X.shape[1] == exp_f and X.shape[2] != exp_f:
        #             return np.transpose(X, (0, 2, 1)).copy()
        #         if X.shape[1] == 36 and X.shape[2] == 80:
        #             return np.transpose(X, (0, 2, 1)).copy()
        #     return X

        # # ... inside predict_lstm_ae_windows_sliding just before scorer call:
        # Xb = _ensure_btf(Xb, model=lstm_ae_model)   # lstm_ae_model 변수명은 실제 코드에 맞게
        # out = scorer(Xb)


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
        # 일반 sliding 경로 (윈도우 여러 개)
        # sliding_window_view는 view라서 메모리 폭발을 줄임
        try:
            W_all = np.lib.stride_tricks.sliding_window_view(F, window_size, axis=0)  # (N-T+1, T, D) view
            W_view = W_all[::step]  # (W,T,D) view
        except Exception:
            # fallback: 기존 방식
            t_idx = np.arange(window_size, dtype=np.int64)[None, :]
            starts = (np.arange(num_windows, dtype=np.int64) * int(step))
            bs = int(max(1, batch_size))
            for b0 in range(0, num_windows, bs):
                b1 = min(num_windows, b0 + bs)
                b_starts = starts[b0:b1][:, None]
                idx = b_starts + t_idx
                Xb_np = np.take(F, idx, axis=0)
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
        else:
            bs = int(max(1, batch_size))
            for b0 in range(0, num_windows, bs):
                b1 = min(num_windows, b0 + bs)
                # tf.convert_to_tensor가 내부에서 copy할 수 있으므로, 여기서 한 번만 contiguous로 만들어줌
                Xb_np = np.ascontiguousarray(W_view[b0:b1, :, :])
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

    t_inf1 = tf.timestamp()

    # -------------------------------
    # 4) Pack result (top-k feature error는 tf top_k 결과 사용)
    # -------------------------------
    t_pack0 = tf.timestamp()
    windows: List[Dict[str, Any]] = []
    for w in range(num_windows):
        s = int(w * step)
        e = int(s + window_size - 1)

        fe: Dict[str, float] = {}
        if k_eff > 0:
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
    t_pack1 = tf.timestamp()

    timing = {
        "ae_build_frame_ms": float((t_build1 - t_build0) * 1000.0),
        "ae_infer_ms": float((t_inf1 - t_inf0) * 1000.0),
        "ae_pack_ms": float((t_pack1 - t_pack0) * 1000.0),
    }

    return {
        "window_size": int(window_size),
        "step": int(step),
        "threshold": th_f,
        "num_packets": int(N),
        "num_windows": int(num_windows),
        "mse_per_window": mse_all.astype(float).tolist(),
        "is_anomaly_pred": is_pred_all.astype(int).tolist(),
        "windows": windows,
        "_timing": timing,  # ✅ 상위에서 합산 가능
    }


# ============================================================
# DL-pattern (Torch)
# ============================================================
import torch

# PyTorch 최적화 설정
torch.set_num_threads(4)  # 스레드 수 제한으로 오버헤드 감소
torch.set_grad_enabled(False)  # 전역 grad 비활성화 (추론 모드)


def _import_from_file(mod_name: str, py_path: Path):
    import importlib.util

    py_path = Path(py_path)
    if not py_path.exists():
        raise FileNotFoundError(str(py_path))

    spec = importlib.util.spec_from_file_location(mod_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"spec load failed: {py_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def _confidence_distance(p: float, eps: float = 1e-12) -> float:
    p = float(np.clip(_safe_float(p, 0.0), eps, 1.0))
    return float(-np.log(p))


def load_dl_pattern_bundle(
    ckpt_dir_or_file: Union[str, Path],
    *,
    loader_py: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    loader_py = Path(loader_py)
    mod = _import_from_file("_dl_pattern_loader_runtime", loader_py)
    if not hasattr(mod, "load_dl_torch_bundle"):
        raise AttributeError(f"loader_py has no load_dl_torch_bundle(): {loader_py}")

    bundle = mod.load_dl_torch_bundle(ckpt_dir_or_file, device=device)
    if not isinstance(bundle, dict):
        raise ValueError("DL-pattern loader returned non-dict bundle")
    return bundle


def _pattern_logits_to_pack(logits: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs_t = torch.softmax(logits, dim=-1)
    probs = probs_t.detach().cpu().numpy()  # (B, C)
    pred_idx = np.argmax(probs, axis=1).astype(np.int64)
    pred_prob = probs[np.arange(probs.shape[0]), pred_idx].astype(np.float64)

    # Vectorized entropy
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    ent = -(p * np.log(p)).sum(axis=1).astype(np.float64)
    return pred_idx, pred_prob, ent


def _build_pattern_frame(
    prepares: List[Dict[str, Any]],
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    prepares -> (Fp, valid_step, feat2idx)
    - Fp: (N, Dp)
    - valid_step: (N,) bool (해당 step에 유효 feature 1개라도 있으면 True)
    """
    N = len(prepares)
    Dp = len(feature_names)
    feat2idx = {k: i for i, k in enumerate(feature_names)}

    Fp = np.zeros((N, Dp), dtype=np.float32)
    valid_step = np.zeros((N,), dtype=np.bool_)

    pad_set = {float(-1.0), float(-2.0)}

    for i, pr in enumerate(prepares):
        feat = pr.get("features")
        if not isinstance(feat, dict):
            continue
        step_valid = False
        for k, v in feat.items():
            j = feat2idx.get(k)
            if j is None:
                continue
            fv = _safe_float(v, 0.0)
            if float(fv) in pad_set:
                continue
            Fp[i, j] = float(fv)
            step_valid = True
        valid_step[i] = step_valid

    return Fp, valid_step, feat2idx


def _predict_pattern_for_window_indices(
    prepares: List[Dict[str, Any]],
    pat_bundle: Dict[str, Any],
    window_indices: List[int],
    *,
    window_size: int,
    step: int,
    batch_size: int,
) -> Dict[int, Dict[str, Any]]:
    """
    특정 window index들만 pattern 추론.
    return: {win_idx: {...pattern pack...}}
    """
    if not window_indices:
        return {}

    model = pat_bundle.get("model")
    feature_names: List[str] = pat_bundle.get("feature_names") or []
    label_classes: List[str] = pat_bundle.get("label_classes") or []
    if model is None or not feature_names or not label_classes:
        raise ValueError("pattern bundle missing fields (model/feature_names/label_classes)")

    N = len(prepares)
    if N < window_size:
        raise ValueError(f"need >= window_size packets. got={N}, window_size={window_size}")

    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    Fp, valid_step, _ = _build_pattern_frame(prepares, feature_names)
    Dp = int(Fp.shape[1])

    # 안정성: out dict
    out: Dict[int, Dict[str, Any]] = {}

    # batch
    wpos = 0
    W = len(window_indices)

    model.eval()
    with torch.inference_mode():
        while wpos < W:
            b_end = min(W, wpos + int(batch_size))
            B = b_end - wpos

            Xb = np.zeros((B, window_size, Dp), dtype=np.float32)
            lengths = np.ones((B,), dtype=np.int64)

            for bi in range(B):
                win_idx = int(window_indices[wpos + bi])
                s = win_idx * step

                win = Fp[s : s + window_size, :]
                win_valid = valid_step[s : s + window_size]

                # ------------------------------------------------------------
                # ✅ OPT-4: per-t loop 제거 (flatnonzero로 한 번에 복사)
                # ------------------------------------------------------------
                idx_valid = np.flatnonzero(win_valid)
                t_out = int(idx_valid.size)
                if t_out > 0:
                    Xb[bi, :t_out, :] = win[idx_valid, :]
                lengths[bi] = int(max(1, t_out))

            x = torch.from_numpy(Xb).float().to(device)
            lens = torch.from_numpy(lengths).long().to(device)

            logits = model(x, lens)
            pred_idx, pred_prob, ent = _pattern_logits_to_pack(logits)

            for bi in range(B):
                win_idx = int(window_indices[wpos + bi])
                idx = int(pred_idx[bi])
                prob = float(pred_prob[bi])
                label = label_classes[idx] if idx < len(label_classes) else f"class_{idx}"

                out[win_idx] = {
                    "pattern": str(label),
                    "pattern_index": int(idx),
                    "pattern_prob": float(prob),
                    "similarity": float(prob * 100.0),
                    "probs_entropy": float(ent[bi]),
                    "num_classes": int(len(label_classes)),
                    "latent_distance": float(_confidence_distance(prob)),
                }

            wpos = b_end

    return out


def predict_dl_pattern_windows_sliding(
    prepares: List[Dict[str, Any]],
    pat_bundle: Dict[str, Any],
    *,
    window_size: int = 80,
    step: int = 5,
    batch_size: int = 256,
) -> List[Dict[str, Any]]:
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size/step must be > 0")
    if len(prepares) < window_size:
        raise ValueError(f"need >= window_size packets. got={len(prepares)}, window_size={window_size}")

    model = pat_bundle.get("model")
    feature_names: List[str] = pat_bundle.get("feature_names") or []
    label_classes: List[str] = pat_bundle.get("label_classes") or []
    if model is None or not feature_names or not label_classes:
        raise ValueError("pattern bundle missing fields (model/feature_names/label_classes)")

    N = len(prepares)
    num_windows = 1 + (N - window_size) // step
    results: List[Dict[str, Any]] = [{} for _ in range(num_windows)]

    # 전체 window index로 selected 경로 재사용
    all_idx = list(range(num_windows))
    out_map = _predict_pattern_for_window_indices(
        prepares,
        pat_bundle,
        all_idx,
        window_size=int(window_size),
        step=int(step),
        batch_size=int(batch_size),
    )
    for i in range(num_windows):
        results[i] = out_map.get(i, {})

    return results


# ============================================================
# RISK (Commercial-style)
# ============================================================
def _infer_asset_value01_from_name(name: Optional[str]) -> float:
    if not name:
        return float(IMPACT_DEFAULT)

    s = str(name).lower()
    if "plc" in s or "rtu" in s or "dcs" in s:
        return 1.00
    if "hmi" in s or "scada" in s:
        return 0.85
    if "hist" in s or "historian" in s:
        return 0.80
    if "eng" in s or "engineering" in s:
        return 0.75
    if "server" in s:
        return 0.70
    if "switch" in s or "router" in s or "fw" in s or "firewall" in s:
        return 0.65
    return float(IMPACT_DEFAULT)


def _infer_impact01(window_last_origin: Dict[str, Any]) -> float:
    dst_asset = window_last_origin.get("dst_asset")
    src_asset = window_last_origin.get("src_asset")

    impact = _infer_asset_value01_from_name(dst_asset)
    if impact <= 0.0:
        impact = _infer_asset_value01_from_name(src_asset)

    proto = str(window_last_origin.get("protocol") or "").lower()
    if proto in ("modbus", "xgt_fen", "xgt-fen", "s7comm", "dnp3", "bacnet"):
        impact = _clamp(impact + 0.05, 0.0, 1.0)

    return float(_clamp(impact, 0.0, 1.0))


def _severity_sigmoid01(ratio: float) -> float:
    k = float(max(SEVERITY_SIGMOID_K, 1e-6))
    s = _sigmoid((float(ratio) - 1.0) * k)  # ratio=1 -> 0.5
    sev = (s - 0.5) * 2.0                   # ratio=1 -> 0.0
    return float(_clamp(sev, 0.0, 1.0))


def _compute_likelihood01(
    *,
    anomaly_score: float,
    threshold: float,
    temporal_error_max: float,
    pattern_prob: Optional[float],
) -> float:
    thr = max(float(threshold), 1e-12)
    ratio = float(anomaly_score) / thr

    severity01 = _severity_sigmoid01(ratio)
    temporal01 = _clamp(np.log10(max(float(temporal_error_max), 0.0) + 1.0) / 6.0, 0.0, 1.0)

    semantic01 = 0.0
    if pattern_prob is not None:
        semantic01 = _clamp(1.0 - float(pattern_prob), 0.0, 1.0)

    a = _clamp(float(severity01) * LIKELIHOOD_W_SEVERITY, 0.0, 1.0)
    b = _clamp(float(temporal01) * LIKELIHOOD_W_TEMPORAL, 0.0, 1.0)
    c = _clamp(float(semantic01) * LIKELIHOOD_W_SEMANTIC, 0.0, 1.0)

    likelihood = 1.0 - (1.0 - a) * (1.0 - b) * (1.0 - c)
    return float(_clamp(likelihood, 0.0, 1.0))


def _compute_confidence01(
    *,
    anomaly_score: float,
    threshold: float,
    pattern_prob: Optional[float],
    pattern_entropy: Optional[float],
    num_classes: Optional[int],
) -> float:
    thr = max(float(threshold), 1e-12)
    ratio = float(anomaly_score) / thr

    conf_anom = _sigmoid((ratio - 1.0) * 3.0)

    conf_pat = 0.5
    if pattern_prob is not None and pattern_entropy is not None:
        ncls = int(num_classes) if num_classes else 0
        max_ent = float(np.log(max(ncls, 2))) if ncls >= 2 else float(np.log(10.0))
        ent_norm = _clamp(float(pattern_entropy) / max(max_ent, 1e-9), 0.0, 1.0)
        conf_pat = 0.5 * _clamp(float(pattern_prob), 0.0, 1.0) + 0.5 * (1.0 - ent_norm)

    confidence = CONF_W_ANOMALY * float(conf_anom) + CONF_W_PATTERN * float(conf_pat)
    return float(_clamp(confidence, 0.0, 1.0))


def compute_risk_score_commercial(
    *,
    anomaly_type: str,
    anomaly_score: float,
    threshold: float,
    temporal_error_max: float,
    feature_error: Dict[str, Any],
    pattern_prob: Optional[float],
    pattern_entropy: Optional[float],
    num_classes: Optional[int],
    impact01: float,
    risk_zero_when_normal: bool = True,
) -> Tuple[float, Dict[str, float]]:
    anomaly_type = str(anomaly_type or "normal")

    if risk_zero_when_normal and anomaly_type != "anomalous":
        return 0.0, {"impact": float(impact01), "likelihood": 0.0, "confidence": 0.0, "raw01": 0.0, "cal01": 0.0}

    likelihood01 = _compute_likelihood01(
        anomaly_score=float(anomaly_score),
        threshold=float(threshold),
        temporal_error_max=float(temporal_error_max),
        pattern_prob=pattern_prob,
    )
    confidence01 = _compute_confidence01(
        anomaly_score=float(anomaly_score),
        threshold=float(threshold),
        pattern_prob=pattern_prob,
        pattern_entropy=pattern_entropy,
        num_classes=num_classes,
    )

    raw01 = float(_clamp(float(impact01) * float(likelihood01) * float(confidence01), 0.0, 1.0))

    alpha = float(max(RISK_ALPHA, 1e-6))
    cal01 = float(_clamp(1.0 - float(np.exp(-alpha * raw01)), 0.0, 1.0))

    score01 = float(_clamp(cal01 ** float(max(RISK_GAMMA, 1e-6)), 0.0, 1.0))
    score = float(round(score01 * 100.0, 2))

    detail = {
        "impact": float(round(float(impact01), 4)),
        "likelihood": float(round(float(likelihood01), 4)),
        "confidence": float(round(float(confidence01), 4)),
        "raw01": float(round(float(raw01), 6)),
        "cal01": float(round(float(cal01), 6)),
    }
    return score, detail


# ============================================================
# Combined DL (AE + Pattern + Risk)
# ============================================================
def predict_dl_models_windows_sliding(
    prepares: List[Dict[str, Any]],
    anom_bundle: Dict[str, Any],
    pat_bundle: Optional[Dict[str, Any]] = None,
    *,
    window_size: int = 80,
    step: int = 5,
    batch_size_anom: int = 128,
    batch_size_pat: int = 256,
    always_run_pattern: bool = True,
    risk_zero_when_normal: bool = RISK_ZERO_WHEN_NORMAL_DEFAULT,
) -> Dict[str, Any]:
    # 1) AE
    anom_pack = predict_lstm_ae_windows_sliding(
        prepares,
        anom_bundle,
        window_size=int(window_size),
        step=int(step),
        batch_size=int(batch_size_anom),
        threshold_override=None,
        topk_feature_error=7,
    )
    anom_windows = anom_pack["windows"]
    num_windows = int(anom_pack["num_windows"])

    # 2) Pattern (최적화: always_run_pattern=False면 anomaly 윈도우에만 수행)
    pat_windows: List[Dict[str, Any]] = [{} for _ in range(num_windows)]
    if pat_bundle is not None:
        if bool(always_run_pattern):
            pat_windows = predict_dl_pattern_windows_sliding(
                prepares,
                pat_bundle,
                window_size=int(window_size),
                step=int(step),
                batch_size=int(batch_size_pat),
            )
        else:
            anom_idx: List[int] = [
                i for i in range(num_windows) if int(anom_windows[i].get("is_anomaly_pred", 0)) == 1
            ]
            out_map = _predict_pattern_for_window_indices(
                prepares,
                pat_bundle,
                anom_idx,
                window_size=int(window_size),
                step=int(step),
                batch_size=int(batch_size_pat),
            )
            for i in range(num_windows):
                pat_windows[i] = out_map.get(i, {})

    # 3) Merge + risk
    merged: List[Dict[str, Any]] = []
    for i in range(num_windows):
        aw = anom_windows[i]
        pw = pat_windows[i] if i < len(pat_windows) else {}

        anomaly_type = "anomalous" if int(aw.get("is_anomaly_pred", 0)) == 1 else "normal"

        # 항상 패턴을 쓰거나 / anomalous일 때만 패턴 사용
        use_pw = pw if (bool(always_run_pattern) or anomaly_type == "anomalous") else {}

        end_idx = int(aw.get("end", aw.get("start", 0) + int(window_size) - 1))
        end_idx = int(_clamp(end_idx, 0, len(prepares) - 1))
        last_origin = prepares[end_idx].get("origin") or {}
        if not isinstance(last_origin, dict):
            last_origin = {}

        impact01 = _infer_impact01(last_origin)

        anomaly_score = float(_safe_float(aw.get("mse", aw.get("anomaly_score", 0.0)), 0.0))
        threshold = float(_safe_float(aw.get("threshold", HARD_THRESHOLD), HARD_THRESHOLD))
        temporal_error_max = float(_safe_float(aw.get("temporal_error_max", 0.0), 0.0))

        pattern_prob = use_pw.get("pattern_prob") if use_pw else None
        pattern_entropy = use_pw.get("probs_entropy") if use_pw else None
        num_classes = use_pw.get("num_classes") if use_pw else None

        risk_score, risk_detail = compute_risk_score_commercial(
            anomaly_type=anomaly_type,
            anomaly_score=anomaly_score,
            threshold=threshold,
            temporal_error_max=temporal_error_max,
            feature_error=aw.get("feature_error") or {},
            pattern_prob=pattern_prob if pattern_prob is not None else None,
            pattern_entropy=pattern_entropy if pattern_entropy is not None else None,
            num_classes=int(num_classes) if num_classes is not None else None,
            impact01=impact01,
            risk_zero_when_normal=bool(risk_zero_when_normal),
        )

        risk_obj: Dict[str, Any] = {
            "score": float(risk_score),
            "detected_time": last_origin.get("@timestamp"),
            "src_ip": last_origin.get("sip"),
            "src_asset": last_origin.get("src_asset"),
            "dst_ip": last_origin.get("dip"),
            "dst_asset": last_origin.get("dst_asset"),
        }
        if bool(RISK_INCLUDE_DETAIL):
            risk_obj.update(
                {
                    k: float(v)
                    for k, v in risk_detail.items()
                    if k in ("impact", "likelihood", "confidence", "raw01", "cal01")
                }
            )

        summary = {
            "semantic_score": pattern_prob if use_pw else None,
            "anomaly_type": anomaly_type,
            "anomaly_score": anomaly_score,
            "threshold": threshold,
            "similarity": use_pw.get("similarity") if use_pw else None,
            "similarity_entropy": pattern_entropy if use_pw else None,
            "latent_distance": use_pw.get("latent_distance") if use_pw else None,
            "feature_error": aw.get("feature_error") or {},
            "temporal_error_max": temporal_error_max,
            "risk": risk_obj,
        }

        merged.append(
            {
                "seq_id": int(aw["seq_id"]),
                "start": int(aw["start"]),
                "end": int(aw["end"]),
                "pattern": use_pw.get("pattern") if use_pw else None,
                "summary": summary,
            }
        )

    return {
        "window_size": int(window_size),
        "step": int(step),
        "threshold": float(anom_pack.get("threshold", HARD_THRESHOLD)),
        "num_packets": int(anom_pack["num_packets"]),
        "num_windows": int(num_windows),
        "windows": merged,
    }

def predict_dl_models(prepares, models, seq_id=None, **kwargs):
    """
    Backward-compatible wrapper for legacy main.py.
    main.py expects: predict_dl_models(prepares=..., models=..., seq_id=...)
    """
    # models dict에서 anomaly/pattern bundle 추출(프로젝트마다 키가 다를 수 있어 방어적으로 처리)
    dl_anom = models.get("dl_anomaly") or models.get("dl_anom") or models.get("dl_anom_bundle")
    dl_pat  = models.get("dl_pattern") or models.get("dl_pat")  or models.get("dl_pat_bundle")

    # dl_pattern이 {"bundle": {...}} 구조일 수도 있음
    if isinstance(dl_pat, dict) and "bundle" in dl_pat and isinstance(dl_pat["bundle"], dict):
        dl_pat = dl_pat["bundle"]

    # 최신 sliding API가 있는 경우 거기로 위임 (있으면 사용)
    if "predict_dl_models_windows_sliding" in globals():
        pack = predict_dl_models_windows_sliding(
            prepares=prepares,
            anom_bundle=dl_anom,
            pat_bundle=dl_pat,
            window_size=len(prepares),
            step=len(prepares),
            **kwargs
        )
        win = (pack.get("windows") or [{}])[0]
        # legacy가 기대할 법한 형태로 리턴
        return {
            "seq_id": seq_id,
            "pattern": win.get("pattern"),
            "summary": win.get("summary") or {},
            "alert": win.get("alert", False),
            "timing": win.get("timing", {}),
        }

    # sliding API가 없다면, 기존 DL_predict 내부 구현(있다면) 사용하거나 최소 구조 반환
    return {"seq_id": seq_id, "pattern": "UNKNOWN", "summary": {}, "alert": False}
