# from __future__ import annotations

# from pathlib import Path
# from typing import Any, Dict, Optional


# def _safe_import_keras():
#     try:
#         import tensorflow as tf  # type: ignore
#         return tf.keras
#     except Exception as e:
#         raise ImportError(
#             "TensorFlow/Keras가 필요합니다. pip install tensorflow 로 설치 후 다시 시도하세요."
#         ) from e


# def load_dl_keras_model_only(
#     model_path: Path | str,
#     *,
#     config: Optional[Dict[str, Any]] = None,
# ):
#     """
#     DL 모델 '호출(로드)' 부분만 분리한 함수.

#     입력:
#       - model_path: model.h5 경로
#       - config: (선택) config.json dict
#           * config["T"]가 있으면 repeat_latent custom_objects를 구성해 로드 시도

#     반환:
#       - model: tf.keras 모델 객체
#     """
#     model_path = Path(model_path)
#     if not model_path.exists():
#         raise FileNotFoundError(f"model.h5 not found at {model_path}")

#     keras = _safe_import_keras()

#     # ---- custom_objects: repeat_latent 대응 (선택) ----
#     custom_objects: Dict[str, Any] = {}
#     T_val = None
#     if isinstance(config, dict) and "T" in config:
#         try:
#             T_val = int(config.get("T"))
#         except Exception:
#             T_val = None

#     if T_val is not None:
#         from tensorflow.keras import backend as K  # type: ignore

#         def repeat_latent(x, T=T_val):
#             return K.repeat(x, T)

#         custom_objects["repeat_latent"] = repeat_latent

#     # ---- 모델 로드(핵심) ----
#     try:
#         model = keras.models.load_model(
#             model_path, custom_objects=custom_objects or None, compile=False
#         )
#         return model
#     except Exception as e:
#         # repeat_latent 관련 fallback (T=1)
#         if "repeat_latent" in str(e) and "repeat_latent" not in custom_objects:
#             from tensorflow.keras import backend as K  # type: ignore

#             def repeat_latent(x, T=1):
#                 return K.repeat(x, T)

#             model = keras.models.load_model(
#                 model_path, custom_objects={"repeat_latent": repeat_latent}, compile=False
#             )
#             return model
#         raise


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def _load_threshold_from_dir(model_dir: Path) -> Optional[float]:
    """
    threshold.json 우선순위:
      threshold > threshold_p99 > threshold_mu3
    """
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
        raise FileNotFoundError(f"❌ feature_keys.txt 없음: {feat_path}")

    with feat_path.open("r", encoding="utf-8") as f:
        raw_keys = [line.strip() for line in f if line.strip()]

    # "0 protocol_norm" 같은 형식일 수도 있으니 마지막 토큰만 사용
    feature_keys = [rk.split()[-1] for rk in raw_keys]
    if not feature_keys:
        raise ValueError(f"❌ feature_keys.txt가 비어 있습니다: {feat_path}")
    return feature_keys


def load_feature_weights(
    config: Dict[str, Any],
    feature_keys: List[str],
    model_dir: Path,
    cli_path: Optional[str] = None,
) -> np.ndarray:
    """
    feature_weights.txt 로딩:
      1) cli_path(=--feature-weights-file)
      2) config["feature_weights_file"]
      3) 못 찾으면 all-ones
    파일 포맷: "feature_name weight"
    """
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
        # all-ones
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


def load_lstm_ae_bundle(
    model_dir: Union[str, Path],
    feature_weights_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    model_dir 구조:
      - config.json
      - model.h5
      - feature_keys.txt
      - (optional) threshold.json
      - (optional) feature_weights.txt (또는 config.feature_weights_file)

    반환:
      {
        "model": keras_model,
        "config": dict,
        "feature_keys": [..],
        "pad_value": float,
        "threshold": Optional[float],
        "feature_weights": np.ndarray shape (D,)
      }
    """
    model_dir = Path(model_dir)

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"❌ config.json 없음: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    T = config.get("T")
    if T is None:
        raise ValueError("❌ config.json에 'T'가 없습니다. (repeat_latent에 필요)")

    pad_value = float(config.get("pad_value", 0.0))

    model_path = model_dir / "model.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"❌ model.h5 없음: {model_path}")

    # Keras load (repeat_latent 커스텀)
    import tensorflow as tf  # local import
    from tensorflow.keras.models import load_model

    T_for_repeat = int(T)

    def repeat_latent(x):
        x = tf.expand_dims(x, axis=1)         # (B, 1, latent_dim)
        x = tf.tile(x, [1, T_for_repeat, 1])  # (B, T, latent_dim)
        return x

    model = load_model(
        model_path,
        compile=False,
        custom_objects={"repeat_latent": repeat_latent},
    )

    feature_keys = _load_feature_keys(model_dir)
    threshold = _load_threshold_from_dir(model_dir)
    feature_weights = load_feature_weights(
        config=config,
        feature_keys=feature_keys,
        model_dir=model_dir,
        cli_path=feature_weights_file,
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
