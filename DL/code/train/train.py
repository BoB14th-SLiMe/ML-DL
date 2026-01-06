#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# 중요: tensorflow import(=train_common 내부 포함)보다 먼저 설정
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from train_common import run_training_common


# ===== Bayesian(KL) divergence를 "모듈 레벨"로 둬서 H5 저장 시 pickle 문제 방지 =====
_KLD_SCALE: float = 1.0


def _kld_divergence(q, p, _):
    # 모듈을 클로저로 캡처하지 않도록 함수 내부에서 import
    import tensorflow_probability as tfp

    return tfp.distributions.kl_divergence(q, p) / float(_KLD_SCALE)


def _load_cfg(yaml_path: Path) -> Dict[str, Any]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"train.yaml이 없습니다: {yaml_path}")

    # 우선 PyYAML 사용
    try:
        import yaml  # type: ignore

        obj = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if obj is None:
            return {}
        if not isinstance(obj, dict):
            raise RuntimeError("train.yaml 최상위는 map(dict) 이어야 합니다.")
        return obj

    except ModuleNotFoundError:
        # PyYAML 없으면 key: value만 단순 파싱 (중첩/리스트 불가)
        def _parse_scalar(s: str) -> Any:
            s = s.strip()
            if not s:
                return ""
            low = s.lower()
            if low in ("null", "none", "~"):
                return None
            if low in ("true", "yes", "on"):
                return True
            if low in ("false", "no", "off"):
                return False
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                return s[1:-1]
            try:
                if "." in s or "e" in low:
                    return float(s)
                return int(s)
            except Exception:
                return s

        cfg: Dict[str, Any] = {}
        for raw in yaml_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            cfg[k.strip()] = _parse_scalar(v)
        return cfg


def _resolve(base_dir: Path, p: Any) -> Any:
    if p is None or not isinstance(p, str):
        return p
    s = p.strip()
    if not s:
        return s
    pp = Path(s)
    return str(pp if pp.is_absolute() else (base_dir / pp).resolve())


def build_normal_model(T: int, D: int, hidden_dim: int, latent_dim: int, bidirectional: bool, _n_train: int):
    import tf_keras as keras

    layers = keras.layers
    models = keras.models

    x = layers.Input(shape=(T, D), name="x")

    if bidirectional:
        h = layers.Bidirectional(layers.LSTM(hidden_dim, return_sequences=False), name="enc_bi")(x)
    else:
        h = layers.LSTM(hidden_dim, return_sequences=False, name="enc")(x)

    if isinstance(h, (tuple, list)):
        h = h[0]

    z = layers.Dense(latent_dim, name="z")(h)
    rep = layers.RepeatVector(T, name="repeat")(z)

    dec = layers.LSTM(hidden_dim, return_sequences=True, name="dec")(rep)
    y = layers.TimeDistributed(layers.Dense(D), name="y")(dec)

    return models.Model(x, y, name="lstm_ae")


def predict_normal(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    return model.predict(X, batch_size=batch_size, verbose=1)


def build_bayesian_model(T: int, D: int, hidden_dim: int, latent_dim: int, bidirectional: bool, n_train: int):
    """
    - DenseFlipout에 들어가는 kernel_divergence_fn을 모듈레벨 함수(_kld_divergence)로 고정
    - 스케일은 전역(_KLD_SCALE)에 넣어서 클로저/모듈 캡처로 인한 H5 저장 pickle 오류 방지
    """
    global _KLD_SCALE
    _KLD_SCALE = float(max(int(n_train), 1))

    try:
        import tensorflow_probability as tfp
    except Exception as e:
        raise RuntimeError(
            "bayesian 모드는 tensorflow_probability가 필요합니다. (pip install tensorflow-probability)"
        ) from e

    import tf_keras as keras

    layers = keras.layers
    models = keras.models

    x = layers.Input(shape=(T, D), name="x")

    if bidirectional:
        h = layers.Bidirectional(layers.LSTM(hidden_dim, return_sequences=False), name="enc_bi")(x)
    else:
        h = layers.LSTM(hidden_dim, return_sequences=False, name="enc")(x)

    if isinstance(h, (tuple, list)):
        h = h[0]

    z = tfp.layers.DenseFlipout(
        latent_dim,
        kernel_divergence_fn=_kld_divergence,
        name="z_flipout",
    )(h)
    if isinstance(z, (tuple, list)):
        z = z[0]

    rep = layers.RepeatVector(T, name="repeat")(z)

    dec = layers.LSTM(hidden_dim, return_sequences=True, name="dec")(rep)

    out_layer = tfp.layers.DenseFlipout(
        D,
        kernel_divergence_fn=_kld_divergence,
        name="y_flipout",
    )
    y = layers.TimeDistributed(out_layer, name="y")(dec)

    return models.Model(x, y, name="bayesian_lstm_ae")


def make_predict_bayesian(mc_samples: int):
    mc_samples = max(int(mc_samples), 1)

    def predict_fn(model, X: np.ndarray, batch_size: int) -> np.ndarray:
        n = int(X.shape[0])
        acc: Optional[np.ndarray] = None

        for _ in range(mc_samples):
            outs = []
            for i in range(0, n, batch_size):
                xb = X[i : i + batch_size]
                yb = model(xb, training=True)
                outs.append(yb.numpy())
            pred = np.concatenate(outs, axis=0)
            acc = pred if acc is None else (acc + pred)

        return acc / float(mc_samples)

    return predict_fn


def main():
    here = Path(__file__).resolve().parent
    yaml_path = here / "train.yaml"
    cfg = _load_cfg(yaml_path)
    base_dir = yaml_path.parent

    mode = str(cfg.get("mode", "normal")).strip().lower()

    input_jsonl = _resolve(base_dir, cfg.get("input_jsonl"))
    output_dir = _resolve(base_dir, cfg.get("output_dir"))
    feature_policy_file = _resolve(base_dir, cfg.get("feature_policy_file"))

    if not isinstance(input_jsonl, str) or not input_jsonl:
        raise RuntimeError("train.yaml에 input_jsonl이 필요합니다.")
    if not isinstance(output_dir, str) or not output_dir:
        raise RuntimeError("train.yaml에 output_dir이 필요합니다.")

    window_size = cfg.get("window_size", None)
    pad_value = float(cfg.get("pad_value", -1.0))

    epochs = int(cfg.get("epochs", 50))
    batch_size = int(cfg.get("batch_size", 64))
    hidden_dim = int(cfg.get("hidden_dim", 128))
    latent_dim = int(cfg.get("latent_dim", 64))
    bidirectional = bool(cfg.get("bidirectional", False))
    lr = float(cfg.get("lr", 1e-3))
    val_ratio = float(cfg.get("val_ratio", 0.2))
    seed = int(cfg.get("seed", 42))

    if mode == "bayesian":
        mc_samples = int(cfg.get("mc_samples", 20))
        build_fn = build_bayesian_model
        pred_fn = make_predict_bayesian(mc_samples)
        model_type = "bayesian"
        extra = {"mc_samples": mc_samples}
    else:
        build_fn = build_normal_model
        pred_fn = predict_normal
        model_type = "normal"
        extra = {"mc_samples": 0}

    run_training_common(
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        window_size=window_size,
        pad_value=pad_value,
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        bidirectional=bidirectional,
        lr=lr,
        val_ratio=val_ratio,
        seed=seed,
        feature_policy_file=feature_policy_file if isinstance(feature_policy_file, str) else None,
        model_type=model_type,
        build_model_fn=build_fn,
        predict_fn=pred_fn,
        extra_config=extra,
    )


if __name__ == "__main__":
    main()
