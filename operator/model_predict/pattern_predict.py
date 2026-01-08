#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_num_threads(4)

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


def _import_from_file(mod_name: str, py_path: Path):
    import importlib.util

    py_path = Path(py_path).expanduser().resolve()
    if not py_path.exists():
        raise FileNotFoundError(str(py_path))

    spec = importlib.util.spec_from_file_location(mod_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"spec load failed: {py_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _confidence_distance(p: float, eps: float = 1e-12) -> float:
    p = float(np.clip(_safe_float(p, 0.0), eps, 1.0))
    return float(-np.log(p))


def load_dl_pattern_bundle(
    ckpt_dir_or_file: Union[str, Path],
    *,
    loader_py: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    loader_py = Path(loader_py).expanduser().resolve()
    mod = _import_from_file("_dl_pattern_loader_runtime", loader_py)
    if not hasattr(mod, "load_dl_torch_bundle"):
        raise AttributeError(f"loader_py has no load_dl_torch_bundle(): {loader_py}")

    bundle = mod.load_dl_torch_bundle(ckpt_dir_or_file, device=device)
    if not isinstance(bundle, dict):
        raise ValueError("DL-pattern loader returned non-dict bundle")
    return bundle


def _pattern_logits_to_pack(logits: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs_t = torch.softmax(logits, dim=-1)
    probs = probs_t.detach().cpu().numpy()  # (B,C)

    pred_idx = np.argmax(probs, axis=1).astype(np.int64)
    pred_prob = probs[np.arange(probs.shape[0]), pred_idx].astype(np.float64)

    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    ent = -(p * np.log(p)).sum(axis=1).astype(np.float64)
    return pred_idx, pred_prob, ent


def _build_pattern_frame(
    prepares: List[Dict[str, Any]],
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    N = len(prepares)
    D = len(feature_names)
    feat2idx = {k: i for i, k in enumerate(feature_names)}

    Fp = np.zeros((N, D), dtype=np.float32)
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

    return Fp, valid_step


def _predict_pattern_for_window_indices(
    prepares: List[Dict[str, Any]],
    pat_bundle: Dict[str, Any],
    window_indices: List[int],
    *,
    window_size: int,
    step: int,
    batch_size: int,
) -> Dict[int, Dict[str, Any]]:
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

    Fp, valid_step = _build_pattern_frame(prepares, feature_names)
    Dp = int(Fp.shape[1])

    out: Dict[int, Dict[str, Any]] = {}

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

    N = len(prepares)
    num_windows = 1 + (N - window_size) // step
    all_idx = list(range(num_windows))

    out_map = _predict_pattern_for_window_indices(
        prepares,
        pat_bundle,
        all_idx,
        window_size=int(window_size),
        step=int(step),
        batch_size=int(batch_size),
    )

    return [out_map.get(i, {}) for i in range(num_windows)]


def predict_dl_pattern_windows_selected(
    prepares: List[Dict[str, Any]],
    pat_bundle: Dict[str, Any],
    window_indices: List[int],
    *,
    window_size: int = 80,
    step: int = 5,
    batch_size: int = 256,
) -> Dict[int, Dict[str, Any]]:
    return _predict_pattern_for_window_indices(
        prepares,
        pat_bundle,
        window_indices,
        window_size=int(window_size),
        step=int(step),
        batch_size=int(batch_size),
    )
