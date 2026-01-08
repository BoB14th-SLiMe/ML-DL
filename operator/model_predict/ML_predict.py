#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class MLPredictConfig:
    hard_threshold: float = -126.0
    normal_label: int = 1
    feature_source: str = "features_then_origin"  # features_only | origin_only | features_then_origin
    origin_prob_key: str = "ml_anomaly_prob"
    origin_match_key: str = "match"
    origin_score_key: str = "ml_anomaly_score"
    topk: int = 2
    default: float = 0.0
    max_batch: int = 256
    round_digits: int = 2


def _to_float(v: Any, default: float) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float, np.number)):
        return float(v)
    try:
        return float(v)
    except Exception:
        return default


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def _get_feature_map(rec: JsonDict, mode: str) -> JsonDict:
    mode = str(mode or "features_then_origin")

    if mode == "features_only":
        f = rec.get("features")
        if isinstance(f, dict):
            return f
        rec["features"] = {}
        return rec["features"]

    if mode == "origin_only":
        o = rec.get("origin")
        if isinstance(o, dict):
            return o
        rec["origin"] = {}
        return rec["origin"]

    f = rec.get("features")
    if isinstance(f, dict):
        return f
    o = rec.get("origin")
    if isinstance(o, dict):
        return o

    rec["features"] = {}
    return rec["features"]


def _predict_anomaly_prob_from_model(model: Any, Xs: np.ndarray, *, normal_label: int) -> np.ndarray:
    n = int(Xs.shape[0])
    out = np.full((n,), 0.5, dtype=float)

    try:
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(Xs))
            if proba.ndim == 2 and proba.shape[0] == n:
                classes = getattr(model, "classes_", None)
                if classes is not None:
                    classes = np.asarray(classes)
                    idxn = None
                    for i, c in enumerate(classes.tolist()):
                        if c == normal_label or str(c) == str(normal_label):
                            idxn = i
                            break
                    if idxn is not None and idxn < proba.shape[1]:
                        p_normal = np.clip(proba[:, idxn].astype(float), 0.0, 1.0)
                        return (1.0 - p_normal).astype(float, copy=False)

                pmax = np.clip(np.max(proba, axis=1).astype(float), 0.0, 1.0)
                return (1.0 - pmax).astype(float, copy=False)

        if hasattr(model, "decision_function"):
            z = np.asarray(model.decision_function(Xs))

            if z.ndim == 2 and z.shape[0] == n:
                classes = getattr(model, "classes_", None)
                if classes is not None:
                    classes = np.asarray(classes)
                    idxn = None
                    for i, c in enumerate(classes.tolist()):
                        if c == normal_label or str(c) == str(normal_label):
                            idxn = i
                            break
                    if idxn is not None and idxn < z.shape[1]:
                        p_normal = _sigmoid(z[:, idxn].astype(float))
                        return (1.0 - p_normal).astype(float, copy=False)

                pmax = np.max(_sigmoid(z.astype(float)), axis=1)
                return (1.0 - pmax).astype(float, copy=False)

            z1 = np.ravel(z).astype(float)
            classes = getattr(model, "classes_", None)

            if classes is not None:
                classes = np.asarray(classes)
                pos_is_normal = (
                    classes.size == 2
                    and ((classes[1] == normal_label) or (str(classes[1]) == str(normal_label)))
                )
            else:
                pos_is_normal = (normal_label == 1)

            p_pos = _sigmoid(z1)
            return ((1.0 - p_pos) if pos_is_normal else p_pos).astype(float, copy=False)

    except Exception:
        return out

    return out


def _predict_match_from_model_or_threshold(
    model: Any,
    Xs: np.ndarray,
    *,
    threshold: float,
    normal_label: int,
) -> Tuple[List[int], np.ndarray]:
    n = int(Xs.shape[0])
    a_prob = _predict_anomaly_prob_from_model(model, Xs, normal_label=normal_label)

    pred = None
    try:
        if hasattr(model, "predict"):
            pred = np.ravel(model.predict(Xs))
    except Exception:
        pred = None

    if pred is not None and len(pred) == n:
        match = [1] * n
        for i in range(n):
            p = pred[i]
            try:
                if hasattr(p, "item"):
                    p = p.item()
            except Exception:
                pass
            match[i] = 1 if (p == normal_label or str(p) == str(normal_label)) else 0
        return match, a_prob

    thr = float(threshold)

    if 0.0 <= thr <= 1.0:
        match = [0 if float(a_prob[i]) >= thr else 1 for i in range(n)]
        return match, a_prob

    if hasattr(model, "decision_function"):
        try:
            z = np.asarray(model.decision_function(Xs))
            if z.ndim == 1 and z.shape[0] == n:
                z1 = np.ravel(z).astype(float)

                classes = getattr(model, "classes_", None)
                if classes is not None:
                    classes = np.asarray(classes)
                    pos_is_normal = (
                        classes.size == 2
                        and ((classes[1] == normal_label) or (str(classes[1]) == str(normal_label)))
                    )
                else:
                    pos_is_normal = (normal_label == 1)

                normal_raw = z1 if pos_is_normal else (-z1)
                match = [1 if float(normal_raw[i]) >= thr else 0 for i in range(n)]
                return match, a_prob
        except Exception:
            pass

    match = [0 if float(a_prob[i]) >= 0.5 else 1 for i in range(n)]
    return match, a_prob


class MLAnomalyProbEngine:
    def __init__(
        self,
        selected_features: Sequence[str],
        scaler=None,
        *,
        default: float = 0.0,
        topk: int = 5,
        max_batch: int = 256,
        round_digits: int = 2,
        feature_source: str = "features_then_origin",
    ):
        feats = [str(x) for x in list(selected_features)]
        if not feats:
            raise ValueError("selected_features is empty/None")

        self.feats: List[str] = feats
        self.names: List[str] = feats
        self.D: int = len(feats)

        self.default_f = float(default)
        self.default = np.float32(default)
        self.topk = int(topk)
        self.max_batch = int(max(1, max_batch))
        self.round_digits = int(round_digits)

        self.feature_source = str(feature_source or "features_then_origin")

        self.X = np.empty((self.max_batch, self.D), dtype=np.float32)
        self.absbuf = np.empty_like(self.X)
        self.percbuf = np.empty_like(self.X)
        self.sumabs = np.empty((self.max_batch,), dtype=np.float32)

        self.scaler = scaler
        self._scale_mode = "none"
        self._mean = None
        self._inv_scale = None
        self._mm_min = None
        self._mm_scale = None
        self._init_scaler_cache(scaler)

    def _init_scaler_cache(self, scaler) -> None:
        if scaler is None:
            self._scale_mode = "none"
            return

        mean_ = getattr(scaler, "mean_", None)
        scale_ = getattr(scaler, "scale_", None)
        if mean_ is not None and scale_ is not None:
            with_mean = getattr(scaler, "with_mean", True)
            with_std = getattr(scaler, "with_std", True)
            self._scale_mode = "standard"
            self._mean = np.asarray(mean_, dtype=np.float32) if with_mean else None
            if with_std:
                s = np.asarray(scale_, dtype=np.float32)
                s = np.where(s == 0, 1.0, s)
                self._inv_scale = (1.0 / s).astype(np.float32, copy=False)
            else:
                self._inv_scale = None
            return

        mm_min = getattr(scaler, "min_", None)
        mm_scale = getattr(scaler, "scale_", None)
        if mm_min is not None and mm_scale is not None:
            self._scale_mode = "minmax"
            self._mm_min = np.asarray(mm_min, dtype=np.float32)
            self._mm_scale = np.asarray(mm_scale, dtype=np.float32)
            return

        self._scale_mode = "sklearn"

    def _ensure_capacity(self, N: int) -> None:
        if N <= self.X.shape[0]:
            return
        newN = int(max(N, self.X.shape[0] * 2))
        self.X = np.empty((newN, self.D), dtype=np.float32)
        self.absbuf = np.empty_like(self.X)
        self.percbuf = np.empty_like(self.X)
        self.sumabs = np.empty((newN,), dtype=np.float32)

    def _extract_X_inplace(self, records: Sequence[JsonDict]) -> np.ndarray:
        N = len(records)
        self._ensure_capacity(N)

        X = self.X[:N, :]
        X.fill(self.default)

        feats = self.feats
        default_f = self.default_f
        mode = self.feature_source

        for i, rec in enumerate(records):
            fmap = _get_feature_map(rec, mode)
            get = fmap.get
            row = X[i]
            for j, c in enumerate(feats):
                row[j] = _to_float(get(c, None), default_f)

        return X

    def _scale_inplace(self, X: np.ndarray) -> np.ndarray:
        if self._scale_mode == "none":
            return X

        if self._scale_mode == "standard":
            if self._mean is not None:
                X -= self._mean
            if self._inv_scale is not None:
                X *= self._inv_scale
            return X

        if self._scale_mode == "minmax":
            X *= self._mm_scale
            X += self._mm_min
            return X

        return self.scaler.transform(X)

    def _topk_one(self, x1d: np.ndarray) -> List[Dict[str, float]]:
        absx = np.abs(x1d)
        denom = float(absx.sum()) + 1e-12
        if denom <= 0.0:
            return []

        perc = absx * (100.0 / denom)
        k = int(self.topk)
        D = int(perc.shape[0])
        if k <= 0:
            return []

        if k >= D:
            idx = np.argsort(-perc)
        else:
            idx = np.argpartition(-perc, kth=k - 1)[:k]
            idx = idx[np.argsort(-perc[idx])]

        out: List[Dict[str, float]] = []
        rd = int(self.round_digits)
        names = self.names
        for j in idx:
            p = float(perc[int(j)])
            if p <= 0.0:
                continue
            out.append({"name": names[int(j)], "percent": float(round(p, rd))})
        return out

    def _topk_batch(self, X: np.ndarray) -> List[List[Dict[str, float]]]:
        N, D = X.shape
        k = int(self.topk)
        if k <= 0:
            return [[] for _ in range(N)]

        absbuf = self.absbuf[:N, :]
        percbuf = self.percbuf[:N, :]
        sumabs = self.sumabs[:N]

        np.abs(X, out=absbuf)
        sumabs[:] = absbuf.sum(axis=1)
        inv = 100.0 / (sumabs + 1e-12)
        percbuf[:] = absbuf * inv[:, None]

        if k >= D:
            top_idx = np.argsort(-percbuf, axis=1)
        else:
            cand = np.argpartition(-percbuf, kth=k - 1, axis=1)[:, :k]
            row = np.arange(N)[:, None]
            cand = cand[row, np.argsort(-percbuf[row, cand], axis=1)]
            top_idx = cand

        out: List[List[Dict[str, float]]] = []
        rd = int(self.round_digits)
        names = self.names

        for i in range(N):
            idxs = top_idx[i]
            lst: List[Dict[str, float]] = []
            for j in idxs:
                p = float(percbuf[i, int(j)])
                if p <= 0.0:
                    continue
                lst.append({"name": names[int(j)], "percent": float(round(p, rd))})
            out.append(lst)
        return out

    def compute_probs(self, records: Sequence[JsonDict]) -> List[List[Dict[str, float]]]:
        N = len(records)
        if N == 0:
            return []
        X = self._extract_X_inplace(records)
        Xs = self._scale_inplace(X)
        return [self._topk_one(Xs[0])] if N == 1 else self._topk_batch(Xs)

    def transform_scaled(self, records: Sequence[JsonDict]) -> np.ndarray:
        N = len(records)
        if N == 0:
            return np.empty((0, self.D), dtype=np.float32)
        X = self._extract_X_inplace(records)
        return self._scale_inplace(X)


def predict_enrich_origin_records_with_bundle_fast(
    records: List[JsonDict],
    *,
    scaler,
    selected_features: Sequence[str],
    model: Any = None,
    config: Optional[MLPredictConfig] = None,
    engine: Optional[MLAnomalyProbEngine] = None,
) -> List[JsonDict]:
    cfg = config or MLPredictConfig()

    N = len(records)
    if N == 0:
        return []

    if engine is None:
        engine = MLAnomalyProbEngine(
            selected_features=selected_features,
            scaler=scaler,
            default=float(cfg.default),
            topk=int(cfg.topk),
            max_batch=max(int(cfg.max_batch), N),
            round_digits=int(cfg.round_digits),
            feature_source=str(cfg.feature_source),
        )
    else:
        engine.topk = int(cfg.topk)
        engine.feature_source = str(cfg.feature_source)

    probs = engine.compute_probs(records)
    out: List[JsonDict] = [{cfg.origin_prob_key: probs[i]} for i in range(N)]

    if model is None:
        return out

    Xs = engine.transform_scaled(records)
    match_ints, a_prob = _predict_match_from_model_or_threshold(
        model,
        Xs,
        threshold=float(cfg.hard_threshold),
        normal_label=int(cfg.normal_label),
    )

    for i in range(N):
        out[i][cfg.origin_score_key] = float(a_prob[i])
        out[i][cfg.origin_match_key] = int(match_ints[i])

    return out


def verify_model_direction(
    model: Any,
    Xs: np.ndarray,
    *,
    normal_label: int = 1,
    sample: int = 5,
) -> None:
    print("======== [ML Model Verification] ========")
    print(f"[INFO] model type             : {type(model)}")
    if hasattr(model, "classes_"):
        print(f"[INFO] model.classes_         : {getattr(model, 'classes_')}")
    print(f"[INFO] has decision_function  : {bool(hasattr(model, 'decision_function'))}")
    print(f"[INFO] normal_label           : {int(normal_label)}")

    try:
        y_pred = model.predict(Xs)
        print(f"[INFO] predict() shape        : {np.shape(y_pred)}")
        print(f"[INFO] predict() sample       : {np.ravel(y_pred)[:sample]}")
    except Exception as e:
        print(f"[WARN] predict() failed       : {e}")

    try:
        if hasattr(model, "decision_function"):
            z = model.decision_function(Xs)
            z = np.asarray(z)
            print(f"[INFO] decision_function shape: {z.shape}")
            print(f"[INFO] decision_function samp : {np.ravel(z)[:sample]}")
    except Exception as e:
        print(f"[WARN] decision_function fail : {e}")

    try:
        a_prob = _predict_anomaly_prob_from_model(model, Xs, normal_label=int(normal_label))
        print(f"[INFO] anomaly_prob sample    : {a_prob[:sample]}")
        print(f"[INFO] mean anomaly_prob      : {float(np.mean(a_prob)):.4f}")
        print(f"[INFO] max anomaly_prob       : {float(np.max(a_prob)):.4f}")
    except Exception as e:
        print(f"[WARN] anomaly_prob calc fail : {e}")

    print("=========================================")
