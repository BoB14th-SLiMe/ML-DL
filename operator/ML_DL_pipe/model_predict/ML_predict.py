#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

JsonDict = Dict[str, Any]

# ============================================================
# Hard-coded threshold here
# - fallback(모델 predict 불가 시) match 생성에만 사용됩니다.
# ============================================================
HARD_THRESHOLD: float = -126  # <- 여기만 수정해서 사용


# ============================================================
# Feature source helper
# ============================================================
def _ensure_feature_map(rec: JsonDict) -> JsonDict:
    f = rec.get("features")
    if isinstance(f, dict):
        return f
    o = rec.get("origin")
    if isinstance(o, dict):
        return o
    rec["features"] = {}
    return rec["features"]


def _to_float_or_default(v: Any, default: float) -> float:
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


def _predict_anomaly_prob_from_model(
    model: Any,
    Xs: np.ndarray,
    *,
    normal_label: int = 1,
) -> np.ndarray:
    """
    반환: anomaly probability (클수록 이상)
    - predict_proba 가능하면: 1 - P(normal_label)
    - decision_function이면: classes_의 양/음 방향을 normal_label 기준으로 보정
    """
    n = Xs.shape[0]
    a_prob = np.full((n,), 0.5, dtype=float)

    try:
        # 1) predict_proba (가장 신뢰)
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(Xs))
            if proba.ndim == 2 and proba.shape[0] == n:
                classes = getattr(model, "classes_", None)
                if classes is not None:
                    classes = np.asarray(classes)
                    # normal_label 컬럼을 찾으면: anomaly = 1 - p_normal
                    idxn = None
                    for i, c in enumerate(classes.tolist()):
                        if c == normal_label or str(c) == str(normal_label):
                            idxn = i
                            break
                    if idxn is not None and idxn < proba.shape[1]:
                        p_normal = np.clip(proba[:, idxn].astype(float), 0.0, 1.0)
                        a_prob = 1.0 - p_normal
                    else:
                        # normal_label 못 찾으면: 확신도 낮을수록 이상(대체)
                        pmax = np.clip(np.max(proba, axis=1).astype(float), 0.0, 1.0)
                        a_prob = 1.0 - pmax
                else:
                    pmax = np.clip(np.max(proba, axis=1).astype(float), 0.0, 1.0)
                    a_prob = 1.0 - pmax

        # 2) decision_function (부호 방향 보정이 핵심)
        elif hasattr(model, "decision_function"):
            z = np.asarray(model.decision_function(Xs))

            # multi-class OVR: (N, C) 형태일 수 있음
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
                        a_prob = 1.0 - p_normal
                    else:
                        pmax = np.max(_sigmoid(z.astype(float)), axis=1)
                        a_prob = 1.0 - pmax
                else:
                    pmax = np.max(_sigmoid(z.astype(float)), axis=1)
                    a_prob = 1.0 - pmax

            # binary: (N,) 형태
            else:
                z1 = np.ravel(z).astype(float)

                # decision_function의 +방향은 classes_[1]
                classes = getattr(model, "classes_", None)
                pos_is_normal = False
                if classes is not None:
                    classes = np.asarray(classes)
                    if classes.size == 2:
                        pos = classes[1]
                        pos_is_normal = (pos == normal_label) or (str(pos) == str(normal_label))
                else:
                    # classes_가 없으면 관례적으로 1이 positive라 가정
                    pos_is_normal = (normal_label == 1)

                p_pos = _sigmoid(z1)  # P(positive class ~ classes_[1])
                # positive가 "정상"이면 anomaly = 1 - p_pos
                a_prob = (1.0 - p_pos) if pos_is_normal else p_pos

    except Exception:
        a_prob = np.full((n,), 0.5, dtype=float)

    return a_prob


def _predict_match_from_model_or_threshold(
    model: Any,
    Xs: np.ndarray,
    *,
    threshold: float = HARD_THRESHOLD,
    normal_label: int = 1,
) -> Tuple[List[int], np.ndarray]:
    """
    match_ints: 1=정상, 0=이상
    a_prob    : anomaly probability (클수록 이상)

    우선순위:
      1) model.predict 가능하면 predict 결과를 그대로 match로 사용 (정상=1, 이상=0)
      2) predict 불가하면 threshold로 match 생성:
         - threshold가 [0,1]이면 anomaly_prob 기준
         - threshold가 그 밖(예:-12)이면 decision_function의 "정상 점수" 기준
    """
    n = Xs.shape[0]
    a_prob = _predict_anomaly_prob_from_model(model, Xs, normal_label=normal_label)

    # 1) predict로 match
    pred = None
    try:
        if hasattr(model, "predict"):
            pred = np.ravel(model.predict(Xs))
    except Exception:
        pred = None

    match_ints: List[int] = [1] * n

    if pred is not None and len(pred) == n:
        for i in range(n):
            p = pred[i]
            try:
                if hasattr(p, "item"):
                    p = p.item()
            except Exception:
                pass
            match_ints[i] = 1 if (p == 1 or str(p) == "1") else 0
        return match_ints, a_prob

    # 2) fallback threshold
    thr = float(threshold)

    # 2-A) threshold가 확률형이면 anomaly_prob 기준
    if 0.0 <= thr <= 1.0:
        for i in range(n):
            match_ints[i] = 0 if float(a_prob[i]) >= thr else 1
        return match_ints, a_prob

    # 2-B) threshold가 -12 같은 raw-score면 decision_function의 "정상 점수" 기준
    if hasattr(model, "decision_function"):
        try:
            z = np.asarray(model.decision_function(Xs))
            z1 = np.ravel(z).astype(float) if z.ndim == 1 else None

            if z1 is not None and z1.shape[0] == n:
                classes = getattr(model, "classes_", None)
                pos_is_normal = False
                if classes is not None:
                    classes = np.asarray(classes)
                    if classes.size == 2:
                        pos = classes[1]
                        pos_is_normal = (pos == normal_label) or (str(pos) == str(normal_label))
                else:
                    pos_is_normal = (normal_label == 1)

                # +방향이 정상(pos_is_normal=True)이면 normal_raw=z
                # +방향이 이상이면 normal_raw=-z
                normal_raw = z1 if pos_is_normal else (-z1)

                for i in range(n):
                    match_ints[i] = 1 if float(normal_raw[i]) >= thr else 0
                return match_ints, a_prob
        except Exception:
            pass

    # 2-C) 마지막 fallback: anomaly_prob 0.5 기준
    for i in range(n):
        match_ints[i] = 0 if float(a_prob[i]) >= 0.5 else 1
    return match_ints, a_prob


class MLAnomalyProbEngine:
    # (사용자 코드 그대로)
    def __init__(
        self,
        selected_features: Sequence[str],
        scaler=None,
        *,
        default: float = 0.0,
        topk: int = 5,
        max_batch: int = 256,
        round_digits: int = 2,
    ):
        feats = list(selected_features)
        if not feats:
            raise ValueError("selected_features is empty/None")

        self.feats: List[str] = feats
        self.names: List[str] = [str(x) for x in feats]
        self.D: int = len(feats)

        self.default_f = float(default)
        self.default = np.float32(default)
        self.topk = int(topk)
        self.max_batch = int(max(1, max_batch))
        self.round_digits = int(round_digits)

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

        for i, rec in enumerate(records):
            fmap = _ensure_feature_map(rec)
            get = fmap.get
            row = X[i]
            for j, c in enumerate(feats):
                row[j] = _to_float_or_default(get(c, None), default_f)

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
        if denom <= 0:
            return []
        perc = absx * (100.0 / denom)

        k = self.topk
        D = perc.shape[0]
        if k <= 0:
            return []
        if k >= D:
            idx = np.argsort(-perc)
        else:
            idx = np.argpartition(-perc, kth=k - 1)[:k]
            idx = idx[np.argsort(-perc[idx])]

        out: List[Dict[str, float]] = []
        rd = self.round_digits
        names = self.names
        for j in idx:
            p = float(perc[int(j)])
            if p <= 0.0:
                continue
            out.append({"name": names[int(j)], "percent": float(round(p, rd))})
        return out

    def _topk_batch(self, X: np.ndarray) -> List[List[Dict[str, float]]]:
        N, D = X.shape
        if self.topk <= 0:
            return [[] for _ in range(N)]

        absbuf = self.absbuf[:N, :]
        percbuf = self.percbuf[:N, :]
        sumabs = self.sumabs[:N]

        np.abs(X, out=absbuf)
        sumabs[:] = absbuf.sum(axis=1)
        inv = 100.0 / (sumabs + 1e-12)
        percbuf[:] = absbuf * inv[:, None]

        k = int(self.topk)
        if k >= D:
            top_idx = np.argsort(-percbuf, axis=1)
        else:
            cand = np.argpartition(-percbuf, kth=k - 1, axis=1)[:, :k]
            row = np.arange(N)[:, None]
            cand = cand[row, np.argsort(-percbuf[row, cand], axis=1)]
            top_idx = cand

        out: List[List[Dict[str, float]]] = []
        rd = self.round_digits
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
        if N == 1:
            return [self._topk_one(Xs[0])]
        return self._topk_batch(Xs)

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
    topk: int = 2,
    origin_prob_key: str = "ml_anomaly_prob",
    # ✅ 모델만 받음 (threshold는 HARD_THRESHOLD로 하드코딩)
    model: Any = None,
    normal_label: int = 1,
    origin_match_key: str = "match",
    origin_score_key: str = "ml_anomaly_score",
    engine: Optional[MLAnomalyProbEngine] = None,
) -> List[JsonDict]:
    """
    반환(기본):
      [{ "ml_anomaly_prob": [...] }, ...]

    model을 주면 추가로:
      - ml_anomaly_score (anomaly_prob)
      - match (1=정상, 0=이상)
        * model.predict 사용 가능하면 predict 결과를 그대로 사용 (기존 의미 유지)
        * predict 불가한 경우에만 HARD_THRESHOLD로 fallback 매칭
    """
    N = len(records)
    if N == 0:
        return []

    if engine is None:
        engine = MLAnomalyProbEngine(
            selected_features=selected_features,
            scaler=scaler,
            topk=int(topk),
            max_batch=max(256, N),
        )
    else:
        engine.topk = int(topk)

    probs = engine.compute_probs(records)
    out: List[JsonDict] = [{origin_prob_key: probs[i]} for i in range(N)]

    if model is not None:
        Xs = engine.transform_scaled(records)

        match_ints, a_prob = _predict_match_from_model_or_threshold(
            model,
            Xs,
            threshold=HARD_THRESHOLD,          # ✅ 하드코딩 사용
            normal_label=int(normal_label),
        )

        for i in range(N):
            out[i][origin_score_key] = float(a_prob[i])
            out[i][origin_match_key] = int(match_ints[i])

    return out



# ============================================================
# Debug / Verification Utility
# ============================================================

def verify_model_direction(
    model: Any,
    Xs: np.ndarray,
    *,
    normal_label: int = 1,
    sample: int = 5,
) -> None:
    """
    모델이 정상(1)/이상(0) 기준으로 제대로 동작하는지 간단 점검.

    출력:
      - model type
      - classes_
      - decision_function 여부
      - sample prediction / anomaly_prob 비교
    """
    print("======== [ML Model Verification] ========")
    print(f"[INFO] model type        : {type(model)}")
    if hasattr(model, "classes_"):
        print(f"[INFO] model.classes_    : {getattr(model, 'classes_')}")
    if hasattr(model, "decision_function"):
        print("[INFO] has decision_function() : True")
    else:
        print("[INFO] has decision_function() : False")
    print(f"[INFO] normal_label      : {normal_label}")

    try:
        y_pred = model.predict(Xs)
        print(f"[INFO] predict() shape   : {np.shape(y_pred)}")
        print(f"[INFO] predict() sample  : {y_pred[:sample]}")
    except Exception as e:
        print(f"[WARN] predict() failed: {e}")
        y_pred = None

    try:
        if hasattr(model, "decision_function"):
            z = model.decision_function(Xs)
            if isinstance(z, np.ndarray):
                print(f"[INFO] decision_function shape : {z.shape}")
                print(f"[INFO] decision_function sample: {z[:sample]}")
    except Exception as e:
        print(f"[WARN] decision_function() failed: {e}")

    try:
        a_prob = _predict_anomaly_prob_from_model(model, Xs, normal_label=normal_label)
        print(f"[INFO] anomaly_prob sample: {a_prob[:sample]}")
        print(f"[INFO] mean anomaly_prob  : {np.mean(a_prob):.4f}")
        print(f"[INFO] max anomaly_prob   : {np.max(a_prob):.4f}")
    except Exception as e:
        print(f"[WARN] anomaly_prob calc failed: {e}")

    print("=========================================")
