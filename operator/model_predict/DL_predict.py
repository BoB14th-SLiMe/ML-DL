#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .ae_predict import HARD_THRESHOLD, predict_lstm_ae_windows_sliding
from .pattern_predict import predict_dl_pattern_windows_selected, predict_dl_pattern_windows_sliding

JsonDict = Dict[str, Any]

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
    s = _sigmoid((float(ratio) - 1.0) * k)
    sev = (s - 0.5) * 2.0
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
            anom_idx = [i for i in range(num_windows) if int(anom_windows[i].get("is_anomaly_pred", 0)) == 1]
            out_map = predict_dl_pattern_windows_selected(
                prepares,
                pat_bundle,
                anom_idx,
                window_size=int(window_size),
                step=int(step),
                batch_size=int(batch_size_pat),
            )
            for i in range(num_windows):
                pat_windows[i] = out_map.get(i, {})

    merged: List[Dict[str, Any]] = []
    for i in range(num_windows):
        aw = anom_windows[i]
        pw = pat_windows[i] if i < len(pat_windows) else {}

        anomaly_type = "anomalous" if int(aw.get("is_anomaly_pred", 0)) == 1 else "normal"
        use_pw = pw if (bool(always_run_pattern) or anomaly_type == "anomalous") else {}

        end_idx = int(aw.get("end", aw.get("start", 0) + int(window_size) - 1))
        end_idx = int(_clamp(end_idx, 0, len(prepares) - 1))
        last_origin = prepares[end_idx].get("origin") or {}
        if not isinstance(last_origin, dict):
            last_origin = {}

        impact01 = _infer_impact01(last_origin)

        anomaly_score = float(_safe_float(aw.get("mse", 0.0), 0.0))
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
            pattern_prob=float(pattern_prob) if pattern_prob is not None else None,
            pattern_entropy=float(pattern_entropy) if pattern_entropy is not None else None,
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
            risk_obj.update({k: float(v) for k, v in risk_detail.items()})

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
    dl_anom = models.get("dl_anomaly") or models.get("dl_anom") or models.get("dl_anom_bundle")
    dl_pat = models.get("dl_pattern") or models.get("dl_pat") or models.get("dl_pat_bundle")

    if isinstance(dl_pat, dict) and "bundle" in dl_pat and isinstance(dl_pat["bundle"], dict):
        dl_pat = dl_pat["bundle"]

    pack = predict_dl_models_windows_sliding(
        prepares=prepares,
        anom_bundle=dl_anom,
        pat_bundle=dl_pat,
        window_size=len(prepares),
        step=len(prepares),
        **kwargs
    )
    win = (pack.get("windows") or [{}])[0]
    return {
        "seq_id": seq_id,
        "pattern": win.get("pattern"),
        "summary": win.get("summary") or {},
        "alert": win.get("alert", False),
        "timing": win.get("timing", {}),
    }
