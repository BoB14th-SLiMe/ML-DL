#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from model_predict import ML_predict
from utils import json_loads_fast, stable_hash

from model_load import loader_dl_anomaly, loader_dl_pattern, loader_ml


def load_and_cache_3_models(
    model_load_dir: Path,
    dl_threshold_fixed: Optional[float] = None,
) -> Dict[str, Any]:
    """모델 3종(ML, DL-anomaly, DL-pattern)을 로드하고 캐시합니다."""
    t_start = time.perf_counter()
    timings = {}

    # 1. ML 모델 로드
    t0 = time.perf_counter()
    ml_model_dir = model_load_dir / "ML_model"
    ml_model, ml_scaler, ml_feats, ml_meta = loader_ml.load_model_bundle(ml_model_dir)
    timings["ml_load_s"] = time.perf_counter() - t0

    # 2. DL-anomaly 모델 로드
    t0 = time.perf_counter()
    dl_anom_dir = model_load_dir / "DL_model_anomaly"
    dl_anom_bundle = loader_dl_anomaly.load_lstm_ae_bundle(dl_anom_dir)
    if dl_threshold_fixed is not None:
        dl_anom_bundle["threshold"] = float(dl_threshold_fixed)
    timings["dl_anom_load_s"] = time.perf_counter() - t0

    # 3. DL-pattern 모델 로드
    t0 = time.perf_counter()
    dl_pat_dir = model_load_dir / "DL_model_pattern"
    dl_pat_bundle = loader_dl_pattern.load_dl_torch_bundle(dl_pat_dir)
    timings["dl_pat_load_s"] = time.perf_counter() - t0

    total_load_s = time.perf_counter() - t_start
    timings["total_load_s"] = total_load_s

    # 최종 반환 객체 구성
    ml_bundle_out = {
        "model": ml_model,
        "scaler": ml_scaler,
        "features": ml_feats,
        "metadata": ml_meta,
        "enabled": ml_model is not None,
    }
    
    dl_anom_bundle["enabled"] = dl_anom_bundle.get("model") is not None
    dl_pat_bundle["enabled"] = dl_pat_bundle.get("model") is not None

    return {
        "ml": ml_bundle_out,
        "dl_anomaly": dl_anom_bundle,
        "dl_pattern": dl_pat_bundle,
        "_timing": timings,
    }


def ensure_lstm_ae_tf_cache(dl_anomaly_bundle: Dict[str, Any]) -> None:
    """TensorFlow/Keras 모델의 그래프 캐시를 위해 dummy predict를 수행합니다."""
    try:
        model = dl_anomaly_bundle.get("model")
        if model is None:
            return

        T = int(dl_anomaly_bundle.get("config", {}).get("T", 0))
        D = len(dl_anomaly_bundle.get("feature_keys", []))
        if T <= 0 or D <= 0:
            return

        dummy_input = np.zeros((1, T, D), dtype=np.float32)
        _ = model.predict(dummy_input)
        print("✓ DL-anomaly model cache warmed up.")
    except Exception as e:
        print(f"⚠️ DL-anomaly model cache warmup failed (ignored): {e}")


def ml_batch_predict_and_contribs(
    prepares: List[Dict[str, Any]],
    ml_bundle: Dict[str, Any],
    *,
    topk: int,
    hash_fallback: bool,
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, float]]]]:
    """ML 모델로 배치 예측 및 기여도 계산을 수행합니다."""
    if not prepares:
        return [], []

    model = ml_bundle.get("model")
    scaler = ml_bundle.get("scaler")
    features = ml_bundle.get("features")
    metadata = ml_bundle.get("metadata", {})

    if model is None or features is None:
        # ML 모델이 비활성화된 경우
        n = len(prepares)
        empty_contribs = [[] for _ in range(n)]
        empty_infos = [{"score": 0.0, "match": "U"} for _ in range(n)]
        return empty_infos, empty_contribs

    # ML_predict.py의 로직을 사용하여 예측 및 기여도 계산
    engine = ML_predict.MLAnomalyProbEngine(
        selected_features=features,
        scaler=scaler,
        topk=topk,
    )
    
    # 기여도 계산
    contribs = engine.compute_probs(prepares)

    # 예측
    Xs = engine.transform_scaled(prepares)
    
    normal_label = metadata.get("normal_label", 1)
    threshold = metadata.get("threshold", 0.5)

    match_ints, anomaly_probs = ML_predict._predict_match_from_model_or_threshold(
        model, Xs, threshold=threshold, normal_label=normal_label
    )

    infos: List[Dict[str, Any]] = []
    for i, p in enumerate(prepares):
        match_char = "O" if match_ints[i] == 1 else "X"
        score = float(anomaly_probs[i])
        
        info = {"score": score, "match": match_char}

        if hash_fallback and match_char == "X":
            origin = p.get("origin", {})
            sip = origin.get("sip")
            dip = origin.get("dip")
            if sip and dip:
                h = stable_hash(f"{sip}-{dip}", mod=100)
                if h < 5:
                    info["match"] = "O"
                    info["hash_fallback"] = True

        infos.append(info)

    return infos, contribs