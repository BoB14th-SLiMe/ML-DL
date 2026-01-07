# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rt_common import to_float, to_int


def load_model_bundle(model_dir: Path) -> Tuple[Any, Dict[str, Any], List[str], Optional[float], float]:
    """
    Returns:
      model, config, feature_keys, threshold_from_file, pad_value
    """
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json 없음: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    T_cfg = config.get("T", None)
    if T_cfg is None:
        raise ValueError("config.json에 T가 없습니다. (repeat_latent에 필요)")

    model_path = model_dir / "model.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"model.h5 없음: {model_path}")

    feat_path = model_dir / "feature_keys.txt"
    if not feat_path.exists():
        raise FileNotFoundError(f"feature_keys.txt 없음: {feat_path}")

    with feat_path.open("r", encoding="utf-8") as f:
        raw = [line.strip() for line in f if line.strip()]
    feature_keys = [rk.split()[-1] for rk in raw]

    pad_value = float(config.get("pad_value", 0.0))

    threshold_from_file: Optional[float] = None
    th_path = model_dir / "threshold.json"
    if th_path.exists():
        try:
            with th_path.open("r", encoding="utf-8") as f:
                th = json.load(f)
            for k in ("threshold", "threshold_p99", "threshold_mu3"):
                if k in th:
                    threshold_from_file = float(th[k])
                    break
        except Exception:
            pass

    import tensorflow as tf
    from tensorflow.keras.models import load_model

    T_for_repeat = to_int(T_cfg)
    if T_for_repeat is None:
        raise ValueError("config.json T 파싱 실패")

    def repeat_latent(x):
        x = tf.expand_dims(x, axis=1)
        x = tf.tile(x, [1, int(T_for_repeat), 1])
        return x

    class CompatLSTM(tf.keras.layers.LSTM):
        def __init__(self, *args, **kwargs):
            kwargs.pop("time_major", None)
            super().__init__(*args, **kwargs)

        @classmethod
        def from_config(cls, cfg):
            if isinstance(cfg, dict):
                cfg.pop("time_major", None)
            return super().from_config(cfg)

    class CompatGRU(tf.keras.layers.GRU):
        def __init__(self, *args, **kwargs):
            kwargs.pop("time_major", None)
            super().__init__(*args, **kwargs)

        @classmethod
        def from_config(cls, cfg):
            if isinstance(cfg, dict):
                cfg.pop("time_major", None)
            return super().from_config(cfg)

    class CompatSimpleRNN(tf.keras.layers.SimpleRNN):
        def __init__(self, *args, **kwargs):
            kwargs.pop("time_major", None)
            super().__init__(*args, **kwargs)

        @classmethod
        def from_config(cls, cfg):
            if isinstance(cfg, dict):
                cfg.pop("time_major", None)
            return super().from_config(cfg)

    custom_objects = {
        "repeat_latent": repeat_latent,
        "LSTM": CompatLSTM,
        "GRU": CompatGRU,
        "SimpleRNN": CompatSimpleRNN,
        "keras.layers.LSTM": CompatLSTM,
        "keras.layers.GRU": CompatGRU,
        "keras.layers.SimpleRNN": CompatSimpleRNN,
        "tf.keras.layers.LSTM": CompatLSTM,
        "tf.keras.layers.GRU": CompatGRU,
        "tf.keras.layers.SimpleRNN": CompatSimpleRNN,
    }

    try:
        model = load_model(model_path, compile=False, custom_objects=custom_objects, safe_mode=False)
    except TypeError:
        model = load_model(model_path, compile=False, custom_objects=custom_objects)

    return model, config, feature_keys, threshold_from_file, pad_value


def load_feature_weights(feature_keys: List[str], fw_path: Optional[Path]) -> np.ndarray:
    """
    fw_path 없거나 파일 없으면 all-ones
    파일 형식: "feature_name weight"
    """
    w = np.ones(len(feature_keys), dtype=np.float32)
    if fw_path is None or not fw_path.exists():
        return w

    idx = {k: i for i, k in enumerate(feature_keys)}
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
                val = to_float(parts[1])
                if val is None:
                    continue
                j = idx.get(name, None)
                if j is not None:
                    w[j] = float(val)
    except Exception:
        return w

    return w
