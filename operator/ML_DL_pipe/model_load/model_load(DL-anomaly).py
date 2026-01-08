#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


JsonDict = Dict[str, Any]


def _read_json_dict(path: Path) -> JsonDict:
    if not path.exists():
        raise FileNotFoundError(str(path))
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be dict: {path}")
    return obj


def _load_threshold(model_dir: Path) -> Optional[float]:
    path = model_dir / "threshold.json"
    if not path.exists():
        return None

    try:
        cfg = _read_json_dict(path)
    except Exception:
        return None

    for key in ("threshold", "threshold_p99", "threshold_mu3"):
        if key in cfg:
            try:
                return float(cfg[key])
            except Exception:
                return None
    return None


def _load_feature_keys(model_dir: Path) -> List[str]:
    path = model_dir / "feature_keys.txt"
    if not path.exists():
        raise FileNotFoundError(f"feature_keys.txt not found: {path}")

    raw = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [rk.split()[-1] for rk in raw]
    if not keys:
        raise ValueError(f"feature_keys.txt is empty: {path}")
    return keys


def _candidate_feature_weights_paths(
    config: JsonDict,
    model_dir: Path,
    cli_path: Optional[str],
) -> List[Path]:
    cands: List[Path] = []

    if cli_path:
        cands.append(Path(cli_path).expanduser())

    cfg_path = config.get("feature_weights_file")
    if cfg_path:
        p = Path(str(cfg_path)).expanduser()
        cands.append(p)
        if not p.is_absolute():
            cands.append((model_dir / p).resolve())

        in_path = config.get("input_jsonl")
        if in_path:
            try:
                ip = Path(str(in_path)).expanduser()
                train_root = ip.parent.parent
                cands.append((train_root / "data" / p.name).resolve())
            except Exception:
                pass

    # 관례적으로 model_dir 바로 아래에 두는 경우도 고려
    cands.append((model_dir / "feature_weights.txt").resolve())

    # 중복 제거(순서 유지)
    seen = set()
    uniq: List[Path] = []
    for x in cands:
        try:
            k = str(x.resolve())
        except Exception:
            k = str(x)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)
    return uniq


def _find_existing_path(cands: List[Path]) -> Optional[Path]:
    for p in cands:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None


def _parse_feature_weights_file(path: Path) -> Dict[str, float]:
    weight_map: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue

        # 지원 포맷:
        # 1) "feature_name weight"
        # 2) "O feature_name weight"
        if parts[0] in ("O", "X") and len(parts) >= 3:
            name = parts[1]
            w_raw = parts[2]
        else:
            name = parts[0]
            w_raw = parts[1]

        try:
            weight_map[name] = float(w_raw)
        except Exception:
            continue

    return weight_map


def load_feature_weights(
    config: JsonDict,
    feature_keys: List[str],
    model_dir: Path,
    feature_weights_file: Optional[str] = None,
) -> np.ndarray:
    weights = np.ones(len(feature_keys), dtype=np.float32)

    cands = _candidate_feature_weights_paths(config, model_dir, feature_weights_file)
    fw_path = _find_existing_path(cands)
    if fw_path is None:
        return weights

    try:
        weight_map = _parse_feature_weights_file(fw_path)
    except Exception:
        return weights

    for i, k in enumerate(feature_keys):
        if k in weight_map:
            weights[i] = float(weight_map[k])

    return weights


def _load_keras_model(model_path: Path, T: int):
    try:
        import tensorflow as tf  # type: ignore
        from tensorflow.keras.models import load_model  # type: ignore
    except Exception as e:
        raise ImportError("TensorFlow/Keras가 필요합니다. pip install tensorflow") from e

    T_for_repeat = int(T)

    def repeat_latent(x):
        x = tf.expand_dims(x, axis=1)         # (B, 1, latent_dim)
        x = tf.tile(x, [1, T_for_repeat, 1])  # (B, T, latent_dim)
        return x

    return load_model(
        str(model_path),
        compile=False,
        custom_objects={"repeat_latent": repeat_latent},
    )


def load_lstm_ae_bundle(
    model_dir: Union[str, Path],
    feature_weights_file: Optional[str] = None,
) -> Dict[str, Any]:
    model_dir = Path(model_dir).expanduser().resolve()

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found: {config_path}")
    config = _read_json_dict(config_path)

    if "T" not in config:
        raise ValueError("config.json missing key: 'T'")
    try:
        T = int(config["T"])
    except Exception:
        raise ValueError("config.json key 'T' must be int")

    pad_value = float(config.get("pad_value", 0.0))

    model_path = model_dir / "model.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"model.h5 not found: {model_path}")

    model = _load_keras_model(model_path, T=T)

    feature_keys = _load_feature_keys(model_dir)
    threshold = _load_threshold(model_dir)
    feature_weights = load_feature_weights(
        config=config,
        feature_keys=feature_keys,
        model_dir=model_dir,
        feature_weights_file=feature_weights_file,
    )

    return {
        "model": model,
        "config": config,
        "feature_keys": feature_keys,
        "pad_value": pad_value,
        "threshold": threshold,
        "feature_weights": feature_weights,
        "model_dir": str(model_dir),
        "model_path": str(model_path),
    }
