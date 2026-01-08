#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_pipeline.py

- data_flow.log 1개 로그 파일 유지
- 추가 저장(요청 반영):
  1) incoming_packets.jsonl      : Redis에서 pop된 "들어온 패킷" 전체
  2) reassembly_before.jsonl     : 재조립(merge) 전(flush 단위)
  3) reassembly_after.jsonl      : 재조립(merge) 후(flush 단위)
  4) final_results.json          : "최종 반환 결과"(window 단위 DL 결과) ✅ JSON(Array)로 저장

- final_results.json 포맷(요구사항):
  {
    "seq_id": 1,
    "pattern": "...",
    "summary": {...},
    "window_raw": [ {packet1}, {packet2}, ... ]
  }

- server처럼 동작 확인/모드:
  --server 옵션을 주면 stop-after 없이 계속 돌면서 Redis stream을 소비하는 서버 모드로 동작.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading
import queue
import signal

import redis
import numpy as np

try:
    import requests  # Alarm 전송용
except Exception:
    requests = None


# ============================================================
# ✅ script dir -> sys.path (상대 import 이슈 방지)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ✅ DL predict (윈도우 단위 호출)
from model_predict.DL_predict import predict_dl_models  # noqa: E402


# ============================================================
# ✅ 기본 설정
# ============================================================
DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_DB = 0
DEFAULT_REDIS_PASSWORD = None

DEFAULT_PROTOCOLS = ["modbus", "s7comm", "xgt_fen", "tcp", "udp", "dns", "arp"]
# DEFAULT_PROTOCOLS = ["modbus", "s7comm", "xgt_fen"]

WINDOW_SIZE = 80
WINDOW_STEP = 20
ALLOW_PARTIAL_WINDOW = False

DEFAULT_STOP_AFTER_WINDOWS = 1  # server 모드면 None
MAX_RAW_PACKETS = 10_000_000
MAX_PREPARES = 10_000_000

DEFAULT_INTERVAL_SEC = 0.0
DEFAULT_REPLAY = False

DEFAULT_PRE_DIR = (SCRIPT_DIR / ".." / "preprocessing" / "result").resolve()
DEFAULT_MODEL_LOAD_DIR = (SCRIPT_DIR / "model_load").resolve()

# run 루트
DEFAULT_RUN_ROOT = (SCRIPT_DIR / "final_results").resolve()

# ----------------------------
# data_flow.log 기록 옵션
# ----------------------------
LOG_MERGE = True
LOG_FINAL = True
LOG_FSYNC = False  # 기본 OFF
DATA_FLOW_MAX_JSON_CHARS = 4000

# ----------------------------
# Async writer tuning
# ----------------------------
ASYNC_BATCH = 200
ASYNC_FLUSH_SEC = 0.5

# ----------------------------
# ML knobs
# ----------------------------
DEFAULT_ML_TOPK = 2
DEFAULT_ML_WARMUP = 5
DEFAULT_ML_SKIP_STATS = 0
DEFAULT_ML_TRIM_PCT = 0.0
DEFAULT_ML_HASH_FALLBACK = True

# ----------------------------
# DL knobs
# ----------------------------
DEFAULT_DL_WARMUP = 3
DEFAULT_DL_SKIP_STATS = 0
DEFAULT_DL_TRIM_PCT = 0.0

# ----------------------------
# Alarm
# ----------------------------
DEFAULT_ALARM_BASE_URL = "http://192.168.4.140:8080"
DEFAULT_ALARM_ENGINE = "dl"
DEFAULT_ALARM_TIMEOUT = 3.0
DEFAULT_ALARM_ENABLED = False

# ----------------------------
# Merge behavior
# ----------------------------
MERGE_PROTOCOLS = {"modbus", "xgt_fen", "xgt-fen"}
MERGE_BUCKET_MS = 3

# ----------------------------
# Server mode
# ----------------------------
DEFAULT_IDLE_SLEEP_SEC = 0.01
HEARTBEAT_SEC = 30.0


# ============================================================
# ✅ import (패키지/스크립트 실행 모두 대응)
# ============================================================
PacketFeaturePreprocessor = None  # type: ignore
_PFP_IMPORT_ERR = None
try:
    from preprocessing.packet_feature_preprocessor import PacketFeaturePreprocessor  # type: ignore
except Exception as e1:
    try:
        from .preprocessing.packet_feature_preprocessor import PacketFeaturePreprocessor  # type: ignore
    except Exception as e2:
        PacketFeaturePreprocessor = None  # type: ignore
        _PFP_IMPORT_ERR = f"{e1} / {e2}"


# ============================================================
# 로깅 / 타이밍 유틸
# ============================================================
def _flush_logger(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        try:
            h.flush()
            if LOG_FSYNC:
                stream = getattr(h, "stream", None)
                if stream is not None and hasattr(stream, "fileno"):
                    try:
                        os.fsync(stream.fileno())
                    except Exception:
                        pass
        except Exception:
            pass


def setup_data_logger(log_path: Path, *, mode: str = "w") -> logging.Logger:
    logger = logging.getLogger("data_flow")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_path), mode=mode, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)

    logger.info("=== data_flow logger initialized ===")
    _flush_logger(logger)
    print(f"✓ data_flow log file: {log_path}")
    return logger


def _kv_line(tag: str, **kv: Any) -> str:
    parts = [f"[{tag}]"]
    for k, v in kv.items():
        if isinstance(v, (dict, list)):
            s = json.dumps(v, ensure_ascii=False, separators=(",", ":"), default=str)
            if DATA_FLOW_MAX_JSON_CHARS and len(s) > DATA_FLOW_MAX_JSON_CHARS:
                s = s[:DATA_FLOW_MAX_JSON_CHARS] + "...(truncated)"
            parts.append(f"{k}={s}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def log_event(logger: logging.Logger, tag: str, **kv: Any) -> None:
    try:
        logger.info(_kv_line(tag, **kv))
    except Exception:
        pass


@contextmanager
def timed(stats: Dict[str, Any], key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        stats[key].append(time.perf_counter() - t0)


# ============================================================
# JSON-safe / None 처리 유틸
# ============================================================
def _is_nan_inf(x: float) -> bool:
    try:
        return math.isnan(x) or math.isinf(x)
    except Exception:
        return False


def sanitize_for_json(obj: Any) -> Any:
    try:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if _is_nan_inf(v) else v
        if isinstance(obj, (np.ndarray,)):
            return [sanitize_for_json(x) for x in obj.tolist()]
    except Exception:
        pass

    if isinstance(obj, float):
        return None if _is_nan_inf(obj) else obj
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(x) for x in obj]
    return str(obj)


def drop_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if v is None:
                continue
            out[k] = drop_none(v)
        return out
    if isinstance(obj, list):
        return [drop_none(x) for x in obj if x is not None]
    return obj


def _json_dumps(obj: Any, *, compact: bool = True) -> str:
    if compact:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


# ============================================================
# Async JSONL writer (배치 write)
# ============================================================
class AsyncJsonlWriter:
    def __init__(self, path: Path, *, batch: int = 200, flush_sec: float = 0.5):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.batch = int(max(1, batch))
        self.flush_sec = float(max(0.05, flush_sec))

        self.q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        buf: List[Dict[str, Any]] = []
        last_flush = time.monotonic()

        # buffering 크게 잡으면 flush 부담이 줄어드는 경우가 많음
        with self.path.open("a", encoding="utf-8", buffering=1024 * 1024) as f:
            while not self.stop_event.is_set() or not self.q.empty():
                timeout = max(0.05, self.flush_sec - (time.monotonic() - last_flush))
                try:
                    obj = self.q.get(timeout=timeout)
                    buf.append(obj)
                    self.q.task_done()
                except queue.Empty:
                    pass

                now = time.monotonic()
                if buf and (len(buf) >= self.batch or (now - last_flush) >= self.flush_sec):
                    try:
                        lines = []
                        for o in buf:
                            try:
                                lines.append(_json_dumps(o, compact=True) + "\n")
                            except Exception:
                                lines.append('{"_error":"json_dumps_failed"}\n')
                        f.write("".join(lines))
                        f.flush()
                    except Exception:
                        pass
                    buf.clear()
                    last_flush = now

            if buf:
                try:
                    lines = []
                    for o in buf:
                        try:
                            lines.append(_json_dumps(o, compact=True) + "\n")
                        except Exception:
                            lines.append('{"_error":"json_dumps_failed"}\n')
                    f.write("".join(lines))
                    f.flush()
                except Exception:
                    pass

    def write_obj(self, obj: Dict[str, Any]) -> None:
        # ✅ dumps는 worker에서
        self.q.put(obj)

    def close(self) -> None:
        self.stop_event.set()
        self.q.join()
        self.thread.join()

# ============================================================
# Async JSON Array writer (final_results.json)
# - 파일 구조: [ {..}, {..}, ... ]
# - 종료 시 "]"를 붙여 유효한 JSON으로 완성
# ============================================================
class AsyncJsonArrayWriter:
    def __init__(self, path: Path, *, batch: int = 200, flush_sec: float = 0.5):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.batch = int(max(1, batch))
        self.flush_sec = float(max(0.05, flush_sec))

        self.q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)

        self._first = True
        self.thread.start()

    def _worker(self):
        buf: List[Dict[str, Any]] = []
        last_flush = time.monotonic()

        with self.path.open("w", encoding="utf-8", buffering=1024 * 1024) as f:
            f.write("[\n")
            f.flush()

            while not self.stop_event.is_set() or not self.q.empty():
                timeout = max(0.05, self.flush_sec - (time.monotonic() - last_flush))
                try:
                    obj = self.q.get(timeout=timeout)
                    buf.append(obj)
                    self.q.task_done()
                except queue.Empty:
                    pass

                now = time.monotonic()
                if buf and (len(buf) >= self.batch or (now - last_flush) >= self.flush_sec):
                    try:
                        for o in buf:
                            try:
                                s = _json_dumps(o, compact=True)
                            except Exception:
                                s = _json_dumps({"_error": "json_dumps_failed"}, compact=True)

                            if self._first:
                                f.write(s)
                                self._first = False
                            else:
                                f.write(",\n" + s)
                        f.flush()
                    except Exception:
                        pass
                    buf.clear()
                    last_flush = now

            if buf:
                try:
                    for o in buf:
                        try:
                            s = _json_dumps(o, compact=True)
                        except Exception:
                            s = _json_dumps({"_error": "json_dumps_failed"}, compact=True)

                        if self._first:
                            f.write(s)
                            self._first = False
                        else:
                            f.write(",\n" + s)
                    f.flush()
                except Exception:
                    pass

            f.write("\n]\n")
            f.flush()

    def write_obj(self, obj: Dict[str, Any]) -> None:
        self.q.put(obj)

    def close(self) -> None:
        self.stop_event.set()
        self.q.join()
        self.thread.join()


class RunStores:
    """
    run 단위 저장:
    - incoming_packets.jsonl    : 들어온 패킷 전체(ingest_id)
    - reassembly_before.jsonl   : merge 전(merge_event_id)
    - reassembly_after.jsonl    : merge 후(merge_event_id)
    - final_results.json        : 최종 반환 결과(window/DL 결과) ✅ JSON(Array)
    """
    def __init__(
        self,
        incoming_path: Path,
        before_path: Path,
        after_path: Path,
        final_path: Path,
    ):
        self.incoming = AsyncJsonlWriter(incoming_path, batch=ASYNC_BATCH, flush_sec=ASYNC_FLUSH_SEC)
        self.before = AsyncJsonlWriter(before_path, batch=ASYNC_BATCH, flush_sec=ASYNC_FLUSH_SEC)
        self.after = AsyncJsonlWriter(after_path, batch=ASYNC_BATCH, flush_sec=ASYNC_FLUSH_SEC)

        # ✅ final만 JSON(Array)로
        self.final = AsyncJsonArrayWriter(final_path, batch=ASYNC_BATCH, flush_sec=ASYNC_FLUSH_SEC)

        print(f"✓ incoming_packets : {incoming_path}")
        print(f"✓ reassembly_before: {before_path}")
        print(f"✓ reassembly_after : {after_path}")
        print(f"✓ final_results    : {final_path}")

    def close(self) -> None:
        self.incoming.close()
        self.before.close()
        self.after.close()
        self.final.close()


# ============================================================
# Alarm sender (비동기 POST)  ✅ 유지
# ============================================================
class AlarmSender:
    def __init__(self, base_url: str, engine: str = "dl", timeout: float = 3.0, logger: Optional[logging.Logger] = None):
        self.base_url = str(base_url).rstrip("/")
        self.engine = str(engine).strip("/") or "dl"
        self.timeout = float(timeout)
        self.logger = logger

        self.q: "queue.Queue[dict]" = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while not self.stop_event.is_set() or not self.q.empty():
            try:
                payload = self.q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if requests is None:
                    if self.logger:
                        log_event(self.logger, "ALARM", status="skip", reason="requests_not_available")
                    continue

                url = f"{self.base_url}/api/alarms/{self.engine}"
                resp = requests.post(url, json=payload, timeout=self.timeout)
                code = getattr(resp, "status_code", None)
                if hasattr(resp, "raise_for_status"):
                    resp.raise_for_status()
                if self.logger:
                    log_event(self.logger, "ALARM", status="sent", http_status=code, url=url)
            except Exception as e:
                if self.logger:
                    log_event(self.logger, "ALARM", status="fail", error=str(e))
            finally:
                self.q.task_done()

    def send_risk(self, risk: Dict[str, Any], *, extra: Optional[Dict[str, Any]] = None) -> None:
        if not isinstance(risk, dict):
            return
        payload = {"risk": risk}
        if extra:
            payload.update(extra)
        try:
            if isinstance(payload.get("risk"), dict):
                payload["risk"].setdefault("detected_time", datetime.utcnow().isoformat(timespec="seconds") + "Z")
        except Exception:
            pass
        self.q.put(payload)

    def close(self) -> None:
        self.stop_event.set()
        self.q.join()
        self.thread.join()


# ============================================================
# model loader
# ============================================================
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


def _try_read_text(path: Path) -> Optional[str]:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
    except Exception:
        return None
    return None


def _try_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        txt = _try_read_text(path)
        if txt:
            return json.loads(txt)
    except Exception:
        return None
    return None


def load_and_cache_3_models(model_load_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ml": {"enabled": False},
        "dl_anomaly": {"enabled": False},
        "dl_pattern": {"enabled": False},
    }

    model_load_dir = Path(model_load_dir)
    ml_dir = (model_load_dir / "ML_model").resolve()
    dl_anom_dir = (model_load_dir / "DL_model_anomaly").resolve()
    dl_pat_dir = (model_load_dir / "DL_model_pattern").resolve()

    # -------- ML --------
    try:
        ml_loader_py = model_load_dir / "model_load(ML).py"
        ml_mod = _import_from_file("_ml_loader", ml_loader_py)
        model, scaler, selected_features, metadata = ml_mod.load_model_bundle(ml_dir)  # type: ignore
        out["ml"] = {
            "enabled": True,
            "model": model,
            "scaler": scaler,
            "selected_features": list(selected_features) if selected_features else [],
            "metadata": metadata or {},
        }
        print(f"✓ ML bundle loaded: {ml_dir}")
    except Exception as e:
        print(f"⚠️ ML load failed: {e}")

    # -------- DL-anomaly --------
    try:
        dl_anom_loader_py = model_load_dir / "model_load(DL-anomaly).py"
        dl_anom_mod = _import_from_file("_dl_anom_loader", dl_anom_loader_py)

        h5 = dl_anom_dir / "model.h5"
        if not h5.exists():
            raise FileNotFoundError(f"DL-anomaly model.h5 not found: {h5}")

        model = dl_anom_mod.load_dl_keras_model_only(h5)  # type: ignore

        feature_keys = None
        fk = dl_anom_dir / "feature_keys.txt"
        if fk.exists():
            feature_keys = [line.strip() for line in fk.read_text(encoding="utf-8").splitlines() if line.strip()]

        cfg = _try_read_json(dl_anom_dir / "config.json") or {}
        pad_value = float(cfg.get("pad_value", -1.0))
        missing_value = float(cfg.get("missing_value", -2.0))

        # thj = _try_read_json(dl_anom_dir / "threshold.json") or {}
        # threshold = thj.get("threshold") or thj.get("threshold_p99") or thj.get("p99")
        # try:
        #     threshold = float(threshold) if threshold is not None else None
        # except Exception:
        #     threshold = None
        
        threshold = 0.32
        out["dl_anomaly"] = {
            "enabled": True,
            "model": model,
            "path": str(h5),
            "feature_keys": feature_keys,
            "pad_value": pad_value,
            "missing_value": missing_value,
            "threshold": threshold,
        }
        print(f"✓ DL-anomaly loaded: {h5}")
    except Exception as e:
        print(f"⚠️ DL-anomaly load failed: {e}")

    # -------- DL-pattern --------
    try:
        dl_pat_loader_py = model_load_dir / "model_load(DL-pattern).py"
        dl_pat_mod = _import_from_file("_dl_pat_loader", dl_pat_loader_py)

        ckpt = None
        for name in [
            "best_model.h5",
            "best_model.pt",
            "best_model.pth",
            "model.h5",
            "model.pt",
            "model.pth",
            "checkpoint.h5",
            "checkpoint.pt",
            "checkpoint.pth",
        ]:
            p = dl_pat_dir / name
            if p.exists():
                ckpt = p
                break

        if ckpt is None:
            cand = list(dl_pat_dir.rglob("*.h5")) + list(dl_pat_dir.rglob("*.pt")) + list(dl_pat_dir.rglob("*.pth"))
            ckpt = cand[0] if cand else None

        if ckpt is None:
            raise FileNotFoundError(f"DL-pattern ckpt not found under: {dl_pat_dir}")

        bundle = dl_pat_mod.load_dl_torch_bundle(ckpt)  # type: ignore
        out["dl_pattern"] = {"enabled": True, "bundle": bundle, "path": str(ckpt)}
        print(f"✓ DL-pattern loaded: {ckpt}")
    except Exception as e:
        print(f"⚠️ DL-pattern load failed: {e}")

    return out


# ============================================================
# ML 예측
# ============================================================
def _stable_hash(s: str, mod: int = 1000) -> int:
    import hashlib
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return int(h, 16) % mod


def ml_predict_from_bundle(
    features: Dict[str, Any],
    origin: Dict[str, Any],
    ml_bundle: Dict[str, Any],
    topk: int = 2,
    *,
    hash_fallback: bool = True,
) -> Dict[str, Any]:
    if not ml_bundle.get("enabled"):
        return {}

    model = ml_bundle.get("model")
    scaler = ml_bundle.get("scaler")
    selected_features: List[str] = list(ml_bundle.get("selected_features") or [])

    if model is None:
        return {"error": "ml model is None"}

    sample = dict(features or {})
    if "protocol" not in sample:
        sample["protocol"] = origin.get("protocol", "unknown")

    if not selected_features:
        selected_features = sorted([k for k, v in sample.items() if isinstance(v, (int, float, bool))])

    vals: List[float] = []
    _get = sample.get
    _hash = _stable_hash

    for k in selected_features:
        v = _get(k, 0.0)
        try:
            if isinstance(v, bool):
                vals.append(float(int(v)))
            elif v is None:
                vals.append(0.0)
            else:
                vals.append(float(v))
        except Exception:
            vals.append(float(_hash(str(v)))) if hash_fallback else vals.append(0.0)

    X = np.asarray([vals], dtype=float)

    try:
        Xs = scaler.transform(X) if scaler is not None else X
    except Exception:
        Xs = X

    def _sigmoid(z: float) -> float:
        z = float(np.clip(z, -20.0, 20.0))
        return 1.0 / (1.0 + np.exp(-z))

    a_prob = 0.5
    try:
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(Xs))
            if proba.ndim == 2 and proba.shape[1] >= 2 and hasattr(model, "classes_"):
                classes = np.asarray(model.classes_)
                if 1 in classes:
                    idx = int(np.where(classes == 1)[0][0])
                    p1 = float(np.clip(proba[0, idx], 0.0, 1.0))
                    a_prob = 1.0 - p1
                else:
                    pmax = float(np.clip(np.max(proba[0]), 0.0, 1.0))
                    a_prob = 1.0 - pmax
            else:
                pmax = float(np.clip(np.max(proba[0]), 0.0, 1.0))
                a_prob = 1.0 - pmax
        elif hasattr(model, "decision_function"):
            z = float(np.ravel(model.decision_function(Xs))[0])
            a_prob = _sigmoid(z)
    except Exception:
        a_prob = 0.5

    pred = None
    try:
        pred = model.predict(Xs)[0]
        pred = pred.item() if hasattr(pred, "item") else pred
    except Exception:
        pred = None

    if pred is None:
        match = "X" if a_prob >= 0.5 else "O"
        match_prob = a_prob if match == "X" else (1.0 - a_prob)
    else:
        match = "O" if pred == 1 else "X"
        match_prob = (1.0 - a_prob) if match == "O" else a_prob

    return {
        "match": match,
        "match_확률": float(round(match_prob * 100.0, 3)),
        "anomaly_prob": float(round(a_prob, 6)),
    }


def ml_top_feature_contribs(
    features: Dict[str, Any],
    origin: Dict[str, Any],
    ml_bundle: Dict[str, Any],
    *,
    topk: int = 2,
    hash_fallback: bool = True,
) -> List[Dict[str, Any]]:
    """
    packet 1개 기준으로 '어떤 feature가 ML 판단에 크게 기여했는지' top-k를 percent로 반환.
    - 모델이 coef_ 또는 feature_importances_를 제공하면 그 기반
    - 아니면 |standardized value| 기반 fallback
    """
    if topk <= 0 or (not ml_bundle.get("enabled")):
        return []

    model = ml_bundle.get("model")
    scaler = ml_bundle.get("scaler")
    selected_features: List[str] = list(ml_bundle.get("selected_features") or [])
    if model is None:
        return []

    sample = dict(features or {})
    if "protocol" not in sample:
        sample["protocol"] = origin.get("protocol", "unknown")

    if not selected_features:
        selected_features = sorted([k for k, v in sample.items() if isinstance(v, (int, float, bool))])

    vals: List[float] = []
    _get = sample.get
    _hash = _stable_hash
    for k in selected_features:
        v = _get(k, 0.0)
        try:
            if isinstance(v, bool):
                vals.append(float(int(v)))
            elif v is None:
                vals.append(0.0)
            else:
                vals.append(float(v))
        except Exception:
            vals.append(float(_hash(str(v)))) if hash_fallback else vals.append(0.0)

    X = np.asarray([vals], dtype=float)
    try:
        Xs = scaler.transform(X) if scaler is not None else X
    except Exception:
        Xs = X

    x = np.ravel(Xs).astype(float)

    contrib = None
    try:
        if hasattr(model, "coef_"):
            w = np.ravel(getattr(model, "coef_"))
            if w.shape[0] == x.shape[0]:
                contrib = np.abs(w * x)
        if contrib is None and hasattr(model, "feature_importances_"):
            w = np.ravel(getattr(model, "feature_importances_"))
            if w.shape[0] == x.shape[0]:
                contrib = np.abs(w * x)
    except Exception:
        contrib = None

    if contrib is None:
        contrib = np.abs(x)

    s = float(np.sum(contrib))
    if not np.isfinite(s) or s <= 0:
        return []

    idxs = np.argsort(-contrib)[: int(topk)]
    out: List[Dict[str, Any]] = []
    for i in idxs:
        pct = float(contrib[i] / s * 100.0)
        out.append({"name": selected_features[int(i)], "percent": round(pct, 2)})
    return out

def ml_batch_predict_and_contribs(
    prepares: List[Dict[str, Any]],
    ml_bundle: Dict[str, Any],
    *,
    topk: int = 2,
    hash_fallback: bool = True,
) -> Tuple[List[Optional[int]], List[List[Dict[str, Any]]]]:
    """
    ✅ FLUSH 단위 배치 ML:
      - X 구성 1회
      - scaler.transform 1회
      - model.predict_proba/decision_function 1회
      - (가능하면) model.predict 1회
      - contrib(topk)도 Xs(표준화 벡터) 기반으로 1회 계산

    반환:
      - match_ints: [0/1/None]  (1=O, 0=X, None=ML 비활성)
      - contribs  : [[{"name":..., "percent":...}, ...], ...]
    """
    n = len(prepares)
    if n == 0 or (not ml_bundle.get("enabled")):
        return [None] * n, [[] for _ in range(n)]

    model = ml_bundle.get("model")
    scaler = ml_bundle.get("scaler")
    selected_features: List[str] = list(ml_bundle.get("selected_features") or [])

    if model is None:
        return [None] * n, [[] for _ in range(n)]

    # selected_features 없으면(비권장) 첫 샘플 기준 숫자형 키로 대체
    if not selected_features:
        first = prepares[0]
        sample0 = dict((first.get("features") or {}))
        origin0 = (first.get("origin") or {})
        if "protocol" not in sample0:
            sample0["protocol"] = origin0.get("protocol", "unknown")
        selected_features = sorted([k for k, v in sample0.items() if isinstance(v, (int, float, bool))])

    d = len(selected_features)
    if d == 0:
        return [None] * n, [[] for _ in range(n)]

    _hash = _stable_hash

    def _to_float(v: Any) -> float:
        try:
            if isinstance(v, bool):
                return float(int(v))
            if v is None:
                return 0.0
            return float(v)
        except Exception:
            return float(_hash(str(v))) if hash_fallback else 0.0

    # 1) X 벡터화 (n,d) 1회
    X = np.empty((n, d), dtype=float)
    for i, pr in enumerate(prepares):
        feat = dict(pr.get("features") or {})
        origin = pr.get("origin") or {}
        if "protocol" not in feat:
            feat["protocol"] = origin.get("protocol", "unknown")
        for j, k in enumerate(selected_features):
            X[i, j] = _to_float(feat.get(k, 0.0))

    # 2) 스케일링 1회
    try:
        Xs = scaler.transform(X) if scaler is not None else X
    except Exception:
        Xs = X

    # 3) anomaly_prob 배치 계산
    a_prob = np.full((n,), 0.5, dtype=float)

    try:
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(Xs))
            if proba.ndim == 2 and proba.shape[0] == n:
                classes = getattr(model, "classes_", None)
                if classes is not None:
                    classes = np.asarray(classes)
                    if proba.shape[1] == classes.shape[0] and np.any(classes == 1):
                        idx1 = int(np.where(classes == 1)[0][0])
                        p1 = np.clip(proba[:, idx1].astype(float), 0.0, 1.0)  # p(normal=1)
                        a_prob = 1.0 - p1
                    else:
                        pmax = np.clip(np.max(proba, axis=1).astype(float), 0.0, 1.0)
                        a_prob = 1.0 - pmax
                else:
                    pmax = np.clip(np.max(proba, axis=1).astype(float), 0.0, 1.0)
                    a_prob = 1.0 - pmax

        elif hasattr(model, "decision_function"):
            z = np.ravel(model.decision_function(Xs)).astype(float)
            z = np.clip(z, -20.0, 20.0)
            a_prob = 1.0 / (1.0 + np.exp(-z))
    except Exception:
        a_prob = np.full((n,), 0.5, dtype=float)

    # 4) pred 배치(가능하면) → match 결정
    match_ints: List[Optional[int]] = [None] * n
    pred = None
    try:
        if hasattr(model, "predict"):
            pred = np.ravel(model.predict(Xs))
    except Exception:
        pred = None

    if pred is not None and len(pred) == n:
        for i in range(n):
            p = pred[i]
            try:
                if hasattr(p, "item"):
                    p = p.item()
            except Exception:
                pass
            # 기존 로직: pred==1 => match 'O'
            match_ints[i] = 1 if str(p) == "1" or p == 1 else 0
    else:
        # pred 실패 시 기존 로직: anomaly_prob>=0.5 => 'X'
        for i in range(n):
            match_ints[i] = 0 if float(a_prob[i]) >= 0.5 else 1

    # 5) contrib(topk) 배치 계산
    if topk <= 0:
        contribs = [[] for _ in range(n)]
        return match_ints, contribs

    topk_eff = int(min(topk, d))

    # weight 추출(가능하면) → 기존과 동일하게 |w*x|, 아니면 |x|
    W = None
    try:
        if hasattr(model, "coef_"):
            w = np.ravel(getattr(model, "coef_"))
            if w.shape[0] == d:
                W = w.astype(float)
        if W is None and hasattr(model, "feature_importances_"):
            w = np.ravel(getattr(model, "feature_importances_"))
            if w.shape[0] == d:
                W = w.astype(float)
    except Exception:
        W = None

    Xabs = np.abs(Xs.astype(float))
    if W is not None:
        contrib_mat = np.abs(Xs.astype(float) * W.reshape(1, -1))
    else:
        contrib_mat = Xabs

    sums = np.sum(contrib_mat, axis=1).astype(float)

    contribs: List[List[Dict[str, Any]]] = []
    for i in range(n):
        s = float(sums[i])
        if not np.isfinite(s) or s <= 0:
            contribs.append([])
            continue

        row = contrib_mat[i]
        if topk_eff == 1:
            idxs = np.array([int(np.argmax(row))], dtype=int)
        else:
            idxs = np.argpartition(-row, topk_eff - 1)[:topk_eff]
            idxs = idxs[np.argsort(-row[idxs])]

        out_i: List[Dict[str, Any]] = []
        for j in idxs:
            pct = float(row[int(j)] / s * 100.0)
            out_i.append({"name": selected_features[int(j)], "percent": round(pct, 2)})
        contribs.append(out_i)

    return match_ints, contribs


# ============================================================
# Redis Pop Server (payload @timestamp oldest-first)
# ============================================================
class RedisPopServer:
    def __init__(self, host: str, port: int, db: int, password: Optional[str], protocols: List[str]):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.protocols = list(protocols)
        self.redis_client: Optional[redis.Redis] = None
        print(f"✓ 감시 대상 프로토콜: {self.protocols}")
        self.connect()

    def connect(self) -> None:
        while True:
            try:
                self.redis_client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                )
                self.redis_client.ping()

                run_id = None
                try:
                    info = self.redis_client.info(section="server")
                    run_id = info.get("run_id")
                except Exception:
                    pass

                print(f"✓ Redis 연결 성공: {self.host}:{self.port} (db={self.db}, run_id={run_id})")

                try:
                    for p in self.protocols:
                        sname = f"stream:protocol:{p}"
                        ln = self.redis_client.xlen(sname)
                        if ln:
                            xinfo = self.redis_client.xinfo_stream(sname)
                            first = xinfo.get("first-entry", [None])[0]
                            last = xinfo.get("last-entry", [None])[0]
                            print(f"  - {sname:<24} XLEN={ln:<8} first={first} last={last}")
                except Exception:
                    pass

                break
            except Exception as e:
                print(f"🚫 Redis 연결 실패: {e}. 3초 후 재시도합니다.")
                time.sleep(3)

    @staticmethod
    def parse_id(sid: str) -> Tuple[int, int]:
        try:
            ts_ms, seq = sid.split("-")
            return int(ts_ms), int(seq)
        except Exception:
            return (0, 0)

    @staticmethod
    def _fast_extract_timestamp(raw: str) -> Optional[str]:
        if not raw or '"@timestamp"' not in raw:
            return None
        try:
            key = '"@timestamp"'
            i = raw.find(key)
            if i < 0:
                return None
            j = raw.find(":", i + len(key))
            if j < 0:
                return None
            q1 = raw.find('"', j)
            if q1 < 0:
                return None
            q2 = raw.find('"', q1 + 1)
            if q2 < 0:
                return None
            return raw[q1 + 1 : q2]
        except Exception:
            return None

    @staticmethod
    def parse_packet_ts_ms(ts_str: Any) -> Optional[int]:
        if not ts_str:
            return None
        if not isinstance(ts_str, str):
            ts_str = str(ts_str)

        s = ts_str.strip()
        if not s:
            return None

        if s.endswith("Z"):
            s = s[:-1] + "+00:00"

        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            return None

    def pop_oldest(self) -> Optional[Dict[str, Any]]:
        try:
            if self.redis_client is None:
                self.connect()
                if self.redis_client is None:
                    return None

            pipe = self.redis_client.pipeline(transaction=False)

            stream_infos = []  # (proto, stream_name)
            for p in self.protocols:
                sname = f"stream:protocol:{p}"
                stream_infos.append((p, sname))
                pipe.xrange(sname, "-", "+", count=1)

            results = pipe.execute()

            best = None  # (proto, sname, msg_id, raw, id_ts, id_seq, pkt_ts_ms)

            for (proto, sname), msgs in zip(stream_infos, results):
                if not msgs:
                    continue

                msg_id, fields = msgs[0]
                raw = fields.get("data")
                if raw is None:
                    continue

                id_ts, id_seq = self.parse_id(msg_id)

                # ✅ tail spike 줄이려면 payload 파싱(@timestamp) 제거가 효과적
                pkt_ts_ms = int(id_ts)

                # --- payload timestamp 옵션(정확한 시간 정렬이 필요하면 활성화) ---
                # ts_iso = self._fast_extract_timestamp(raw)
                # pkt_ts_ms2 = self.parse_packet_ts_ms(ts_iso) if ts_iso else None
                # if pkt_ts_ms2 is not None:
                #     pkt_ts_ms = int(pkt_ts_ms2)

                cand = (proto, sname, msg_id, raw, int(id_ts), int(id_seq), int(pkt_ts_ms))

                if best is None:
                    best = cand
                else:
                    _, _, _, _, best_id_ts, best_id_seq, best_pkt_ts_ms = best
                    if (pkt_ts_ms < best_pkt_ts_ms) or (
                        pkt_ts_ms == best_pkt_ts_ms and (id_ts < best_id_ts or (id_ts == best_id_ts and id_seq < best_id_seq))
                    ):
                        best = cand

            if best is None:
                return None

            proto, sname, msg_id, raw, id_ts, id_seq, pkt_ts_ms = best

            # 삭제는 선택된 1개만
            self.redis_client.xdel(sname, msg_id)

            meta = {
                "redis_id": msg_id,
                "redis_timestamp_ms": int(id_ts or 0),
                "packet_timestamp_ms": int(pkt_ts_ms or 0),
                "pop_time": datetime.now().isoformat(),
                "protocol": proto,
            }

            # ✅ JSON 파싱은 밖(run loop)으로 넘김
            return {"origin_raw": raw, "protocol": proto, "_meta": meta}

        except Exception as e:
            print(f"❌ pop_oldest() 오류: {e}")
            self.connect()
            return None


# ============================================================
# merge util
# ============================================================
def _normalize_protocol_for_features(proto: str) -> str:
    return "xgt_fen" if proto == "xgt-fen" else proto


def merge_packets_by_modbus_diff(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not logs:
        return []
    proto = logs[0].get("protocol", "")
    if proto not in MERGE_PROTOCOLS or len(logs) == 1:
        return [logs[0]] if logs else []

    base_packet = dict(logs[0])  # deepcopy 대신 shallow copy

    prefixes = ["xgt_fen.", "xgt-fen."] if proto in ("xgt_fen", "xgt-fen") else [str(proto) + "."]

    # ✅ prefix key만 수집
    keys = set()
    for p in logs:
        for k in p.keys():
            if any(k.startswith(pref) for pref in prefixes):
                keys.add(k)

    for key in keys:  # 정렬이 꼭 필요 없으면 sorted 제거
        values = [p.get(key) for p in logs]
        first_val = next((v for v in values if v is not None), None)

        same = True
        for v in values:
            if v is None:
                continue
            if v != first_val:
                same = False
                break

        if same:
            base_packet[key] = first_val
        else:
            merged_vals = [v for v in values if v is not None]
            base_packet[key] = merged_vals[0] if len(merged_vals) == 1 else merged_vals

    return [base_packet]


# ============================================================
# DL output + alert
# ============================================================
def parse_dl_output(dl_out: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    pattern = dl_out.get("pattern")
    summary = dl_out.get("summary")

    if pattern is None and isinstance(dl_out.get("dl_pattern"), dict):
        pattern = dl_out["dl_pattern"].get("pattern")

    if summary is None:
        da = dl_out.get("dl_anomaly")
        if isinstance(da, dict) and "summary" in da and isinstance(da["summary"], dict):
            summary = da["summary"]
        elif isinstance(da, dict):
            summary = da

    if pattern is None:
        pattern = "UNKNOWN"
    if not isinstance(summary, dict):
        summary = {}

    return str(pattern), summary


def derive_alert(summary: Dict[str, Any]) -> bool:
    try:
        score = summary.get("anomaly_score", None)
        th = summary.get("threshold", None)
        if score is not None and th is not None:
            return float(score) >= float(th)
    except Exception:
        pass

    try:
        at = str(summary.get("anomaly_type", "")).strip().lower()
        if at in {"anomalous", "abnormal", "attack", "anomaly"}:
            return True
    except Exception:
        pass

    try:
        risk = summary.get("risk", {}) or {}
        return float(risk.get("score", 0.0)) > 0.0
    except Exception:
        return False


def normalize_risk_fields(summary: Dict[str, Any], base_origin: Dict[str, Any]) -> Dict[str, Any]:
    """
    - DL summary의 risk가 있으면 최대한 그대로 유지
    - 다만 risk의 src/dst ip/asset, detected_time 등이 '없거나(None)'이면 window의 base_origin으로 보정
    """
    s = summary if isinstance(summary, dict) else {}
    risk = s.get("risk")
    if not isinstance(risk, dict):
        risk = {}

    # detected_time: 없거나 None/빈값이면 base_origin timestamp로
    detected_at = base_origin.get("@timestamp") or base_origin.get("timestamp") or datetime.now().isoformat()
    if risk.get("detected_time") in (None, "", "null"):
        risk["detected_time"] = detected_at

    # score: 숫자형 보장
    try:
        risk_score = risk.get("score", 0.0)
        risk["score"] = float(risk_score) if risk_score is not None else 0.0
    except Exception:
        risk["score"] = 0.0

    # src/dst: 값이 None이면 base_origin에서 채움
    if risk.get("src_ip") is None:
        risk["src_ip"] = base_origin.get("sip")
    if risk.get("dst_ip") is None:
        risk["dst_ip"] = base_origin.get("dip")
    if risk.get("src_asset") is None:
        risk["src_asset"] = base_origin.get("src_asset")
    if risk.get("dst_asset") is None:
        risk["dst_asset"] = base_origin.get("dst_asset")

    s["risk"] = risk

    # 숫자 필드 None -> 0.0 (JSON 안정화)
    for k in [
        "temporal_error_max",
        "latent_distance",
        "similarity_entropy",
        "similarity",
        "semantic_score",
        "anomaly_score",
        "threshold",
    ]:
        if k in s and s.get(k) is None:
            s[k] = 0.0

    if "feature_error" in s and s.get("feature_error") is None:
        s["feature_error"] = {}

    return s


# ============================================================
# final_results.json window_raw 포맷 생성 유틸
# ============================================================
def _to_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            fv = float(v)
            return int(fv) if fv.is_integer() else None
        s = str(v).strip()
        if not s:
            return None
        if s.lower().startswith("0x"):
            return int(s, 16)
        f = float(s)
        return int(f) if f.is_integer() else None
    except Exception:
        return None


def _coerce_scalar(v: Any) -> Any:
    """JSON에 넣을 값: 숫자 문자열은 int/float로 최대한 변환."""
    if v is None:
        return None
    if isinstance(v, (int, float, bool, np.integer, np.floating)):
        return sanitize_for_json(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return v
        try:
            if s.lower().startswith("0x"):
                return int(s, 16)
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                return int(s)
        except Exception:
            pass
        try:
            if any(c in s for c in [".", "e", "E"]):
                return float(s)
        except Exception:
            pass
        return v
    return sanitize_for_json(v)


def _origin_to_nested(origin: Dict[str, Any]) -> Dict[str, Any]:
    """
    origin(원본 packet)을 최대한 유지하되,
    - dotted key("arp.op") => nested dict {"arp": {"op": ...}}
    - 숫자 문자열 => int/float 변환
    """
    if not isinstance(origin, dict):
        return {}

    out: Dict[str, Any] = {}

    # 1) dotted 아닌 top-level 복사
    for k, v in origin.items():
        if "." in k:
            continue
        out[k] = _coerce_scalar(v)

    # 2) dotted -> nested
    for k, v in origin.items():
        if "." not in k:
            continue
        prefix, sub = k.split(".", 1)
        if prefix not in out or not isinstance(out.get(prefix), dict):
            # 충돌 시 dict로 강제
            if prefix in out and not isinstance(out[prefix], dict):
                out[prefix] = {"_value": out[prefix]}
            else:
                out[prefix] = {}
        out[prefix][sub] = _coerce_scalar(v)

    # 3) 핵심 필드 타입 보정
    for nk in ("len", "sp", "dp"):
        if nk in out:
            iv = _to_int(out.get(nk))
            if iv is not None:
                out[nk] = iv

    # 4) 통일 키 보장(없으면 null로)
    out.setdefault("sip", None)
    out.setdefault("dip", None)
    out.setdefault("src_asset", None)
    out.setdefault("dst_asset", None)

    return sanitize_for_json(out)


def build_window_raw_entry(prepare: Dict[str, Any]) -> Dict[str, Any]:
    """
    window_raw의 원소는:
      - origin(원본 packet; nested 변환 포함)
      - + redis_id
      - + ml(예: match/match_확률/anomaly_prob)
      - + ml_anomaly_prob(top feature percent)
    """
    origin = prepare.get("origin", {}) or {}
    # meta = prepare.get("_meta", {}) or {}

    pkt = _origin_to_nested(origin)

    # redis_id 부착
    # redis_id = meta.get("redis_id")
    # if redis_id is None and isinstance(meta.get("redis_ids"), list) and meta["redis_ids"]:
    #     redis_id = meta["redis_ids"][0]
    # pkt["redis_id"] = redis_id

    # ✅ ML 결과 부착(요구사항)
    if "ml" in prepare:
        pkt["ml"] = sanitize_for_json(prepare.get("ml"))

    # ✅ ML feature percent 부착(요구사항 예시와 동일 키)
    pkt["ml_anomaly_prob"] = sanitize_for_json(prepare.get("ml_anomaly_prob") or [])

    return sanitize_for_json(pkt)


# ============================================================
# main pipeline
# ============================================================
class OperationalPipeLine:
    MERGE_PROTOCOLS = MERGE_PROTOCOLS

    def __init__(
        self,
        logger: logging.Logger,
        stores: RunStores,
        redis_host: str = DEFAULT_REDIS_HOST,
        redis_port: int = DEFAULT_REDIS_PORT,
        redis_db: int = DEFAULT_REDIS_DB,
        redis_password: Optional[str] = DEFAULT_REDIS_PASSWORD,
        protocols: Optional[List[str]] = None,
        interval_sec: float = DEFAULT_INTERVAL_SEC,
        replay: bool = DEFAULT_REPLAY,
        stop_after_windows: Optional[int] = DEFAULT_STOP_AFTER_WINDOWS,
        *,
        # ML
        ml_topk: int = DEFAULT_ML_TOPK,
        ml_warmup: int = DEFAULT_ML_WARMUP,
        ml_skip_stats: int = DEFAULT_ML_SKIP_STATS,
        ml_trim_pct: float = DEFAULT_ML_TRIM_PCT,
        ml_hash_fallback: bool = DEFAULT_ML_HASH_FALLBACK,
        # DL
        dl_warmup: int = DEFAULT_DL_WARMUP,
        dl_skip_stats: int = DEFAULT_DL_SKIP_STATS,
        dl_trim_pct: float = DEFAULT_DL_TRIM_PCT,
        # Alarm
        alarm_enabled: bool = DEFAULT_ALARM_ENABLED,
        alarm_base_url: str = DEFAULT_ALARM_BASE_URL,
        alarm_engine: str = DEFAULT_ALARM_ENGINE,
        alarm_timeout: float = DEFAULT_ALARM_TIMEOUT,
        # Server
        server_mode: bool = False,
        idle_sleep_sec: float = DEFAULT_IDLE_SLEEP_SEC,
    ):
        self.logger = logger
        self.stores = stores

        self.redis_host = redis_host
        self.redis_port = int(redis_port)
        self.redis_db = int(redis_db)
        self.redis_password = redis_password
        self.protocols = list(protocols) if protocols else list(DEFAULT_PROTOCOLS)

        if "xgt_fen" in self.protocols and "xgt-fen" not in self.protocols:
            self.protocols.append("xgt-fen")
        if "xgt-fen" in self.protocols and "xgt_fen" not in self.protocols:
            self.protocols.append("xgt_fen")

        self.interval_sec = float(interval_sec)
        self.replay = bool(replay)
        self.stop_after_windows = stop_after_windows
        self.server_mode = bool(server_mode)
        self.idle_sleep_sec = float(max(0.0, idle_sleep_sec))

        # ML
        self.ml_topk = int(ml_topk)
        self.ml_warmup = int(ml_warmup)
        self.ml_skip_stats = int(max(0, ml_skip_stats))
        self.ml_trim_pct = float(max(0.0, min(0.49, ml_trim_pct)))
        self.ml_hash_fallback = bool(ml_hash_fallback)
        self._ml_call_count = 0

        # DL
        self.dl_warmup = int(max(0, dl_warmup))
        self.dl_skip_stats = int(max(0, dl_skip_stats))
        self.dl_trim_pct = float(max(0.0, min(0.49, dl_trim_pct)))
        self._dl_call_count = 0

        # Alarm
        self.alarm_enabled = bool(alarm_enabled)
        self.alarm_sender: Optional[AlarmSender] = None
        if self.alarm_enabled:
            if requests is None:
                print("⚠️ alarm enabled but 'requests' not available → alarm disabled", flush=True)
                self.alarm_enabled = False
            else:
                self.alarm_sender = AlarmSender(
                    base_url=str(alarm_base_url),
                    engine=str(alarm_engine),
                    timeout=float(alarm_timeout),
                    logger=self.logger,
                )

        self.server = RedisPopServer(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=self.redis_password,
            protocols=self.protocols,
        )

        self.featurizer = self._init_featurizer()
        self.models: Dict[str, Any] = load_and_cache_3_models(model_load_dir=DEFAULT_MODEL_LOAD_DIR)

        if self.ml_warmup > 0:
            self._warmup_ml(self.ml_warmup)
        if self.dl_warmup > 0:
            self._warmup_dl(self.dl_warmup)

        self.preprocessing_buffer: List[Dict[str, Any]] = []
        self.window_buffer: List[Dict[str, Any]] = []

        self.seq_id = 0
        self.total_raw_packets = 0
        self.total_prepares = 0
        self.total_windows = 0

        self.stats: Dict[str, Any] = defaultdict(list)
        self.last_packet_ts_ms: Optional[int] = None

        self._run_t0: Optional[float] = None
        self._run_started_at: Optional[str] = None
        self._run_ended_at: Optional[str] = None
        self._run_elapsed_sec: Optional[float] = None

        # merge 이벤트 id (before/after 매칭)
        self.merge_event_id = 0

        # incoming ingest id (모든 pop 패킷 순서)
        self.ingest_id = 0

        # 종료 플래그(서버 모드)
        self._stop_flag = threading.Event()

    def request_stop(self) -> None:
        self._stop_flag.set()

    def close(self) -> None:
        try:
            if self.alarm_sender is not None:
                self.alarm_sender.close()
        except Exception:
            pass
        try:
            self.stores.close()
        except Exception:
            pass
        try:
            _flush_logger(self.logger)
            for h in list(self.logger.handlers):
                try:
                    h.flush()
                except Exception:
                    pass
                try:
                    h.close()
                except Exception:
                    pass
        except Exception:
            pass

    def _init_featurizer(self):
        if PacketFeaturePreprocessor is None:
            print(f"⚠️ PacketFeaturePreprocessor import 실패: {_PFP_IMPORT_ERR}")
            print("⚠️ 피처 전처리 없이 features={}로만 진행됩니다.")
            return None

        if not DEFAULT_PRE_DIR.exists():
            print(f"⚠️ pre_dir 없음: {DEFAULT_PRE_DIR}")
            print("⚠️ 피처 전처리 없이 features={}로만 진행됩니다.")
            return None

        try:
            featurizer = PacketFeaturePreprocessor(
                Path(DEFAULT_PRE_DIR),
                allow_new_ids=False,
                index_source="redis_id",
                include_index=False,
            )
            print(f"✓ PacketFeaturePreprocessor 로드 완료: {DEFAULT_PRE_DIR}")
            return featurizer
        except Exception as e:
            print(f"⚠️ PacketFeaturePreprocessor init 실패: {e}")
            return None

    def _warmup_ml(self, n: int) -> None:
        mlb = self.models.get("ml", {})
        if not mlb.get("enabled"):
            print("⚠️ ML warmup skip: ml bundle disabled")
            return

        selected = list(mlb.get("selected_features") or [])
        dummy_features = {k: 0.0 for k in selected} if selected else {"protocol": 0.0}
        dummy_origin = {"protocol": "warmup"}

        for _ in range(max(1, int(n))):
            _ = ml_predict_from_bundle(
                dummy_features,
                dummy_origin,
                mlb,
                topk=0,
                hash_fallback=self.ml_hash_fallback,
            )
        print(f"✓ ML warmup done: n={n}")

    def _warmup_dl(self, n: int) -> None:
        try:
            dummy_prepare = {"origin": {"protocol": "warmup"}, "features": {}, "_meta": {"redis_id": "warmup"}}
            dummy_window = [deepcopy(dummy_prepare) for _ in range(WINDOW_SIZE)]
            for i in range(max(1, int(n))):
                _ = predict_dl_models(prepares=dummy_window, models=self.models, seq_id=-(i + 1)) or {}
            print(f"✓ DL warmup done: n={n}")
        except Exception as e:
            print(f"⚠️ DL warmup failed (ignored): {e}")

    def _merge_key_from_wrapper(self, wrapped: Dict[str, Any]) -> Optional[Tuple[str, Any, Any]]:
        origin = wrapped.get("origin", {}) or {}
        meta = wrapped.get("_meta", {}) or {}
        proto = origin.get("protocol", "")
        if proto not in self.MERGE_PROTOCOLS:
            return None

        proto_norm = "xgt_fen" if proto == "xgt-fen" else str(proto)

        sq = None
        for k in [
            "sq",
            "modbus.sq",
            "xgt_fen.sq",
            "xgt-fen.sq",
            "transaction_id",
            "trans_id",
            "tid",
            "modbus.tid",
            "modbus.transaction_id",
            "xgt_fen.invoke_id",
            "xgt-fen.invoke_id",
        ]:
            v = origin.get(k)
            if v is not None and str(v) != "":
                sq = str(v)
                break

        ts = origin.get("@timestamp") or origin.get("timestamp")
        if ts is not None and str(ts) != "" and sq is not None:
            return (proto_norm, str(ts), sq)

        pkt_ts_ms = meta.get("packet_timestamp_ms") or meta.get("redis_timestamp_ms")
        try:
            bucket = int(pkt_ts_ms) // int(MERGE_BUCKET_MS)
        except Exception:
            bucket = 0
        return (proto_norm, bucket, "bucket")

    def _build_group_meta(self, wrappers: List[Dict[str, Any]]) -> Dict[str, Any]:
        metas = [w.get("_meta") or {} for w in wrappers]
        ids = [m.get("redis_id") for m in metas if m.get("redis_id") is not None]
        tss = [int(m.get("redis_timestamp_ms", 0)) for m in metas if m.get("redis_timestamp_ms") is not None]
        pops = [m.get("pop_time") for m in metas if m.get("pop_time") is not None]
        proto = (metas[0].get("protocol") if metas else None)

        ingests = []
        for m in metas:
            v = m.get("ingest_id")
            try:
                if v is not None:
                    ingests.append(int(v))
            except Exception:
                pass

        return {
            "protocol": proto,
            "redis_id": ids[0] if ids else None,
            "redis_ids": ids,
            "redis_timestamp_ms_min": min(tss) if tss else 0,
            "redis_timestamp_ms_max": max(tss) if tss else 0,
            "pop_time_first": pops[0] if pops else None,
            "pop_time_last": pops[-1] if pops else None,
            "raw_count": len(wrappers),
            "ingest_id_min": min(ingests) if ingests else None,
            "ingest_id_max": max(ingests) if ingests else None,
        }

    def _call_featurizer(self, origin: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        if self.featurizer is None:
            return {}

        origin_for_feat = dict(origin)
        origin_for_feat["protocol"] = _normalize_protocol_for_features(str(origin_for_feat.get("protocol", "")))
        wrapped = {"origin": origin_for_feat, "_meta": meta}

        if hasattr(self.featurizer, "preprocess"):
            try:
                out = self.featurizer.preprocess(wrapped)  # type: ignore
            except TypeError:
                try:
                    out = self.featurizer.preprocess(origin_for_feat, meta=meta)  # type: ignore
                except TypeError:
                    out = self.featurizer.preprocess(origin_for_feat)  # type: ignore

            if isinstance(out, dict) and "features" in out and isinstance(out["features"], dict):
                return out["features"]
            return out if isinstance(out, dict) else {"value": out}

        if hasattr(self.featurizer, "process"):
            try:
                out = self.featurizer.process(wrapped)  # type: ignore
            except TypeError:
                out = self.featurizer.process(origin_for_feat)  # type: ignore

            if isinstance(out, dict) and "features" in out and isinstance(out["features"], dict):
                return out["features"]
            return out if isinstance(out, dict) else {"value": out}

        return {"features_error": "no preprocess/process method"}

    def _append_prepare(self, origin: Dict[str, Any], group_meta: Dict[str, Any]) -> Dict[str, Any]:
        features: Dict[str, Any] = {}
        if self.featurizer is not None:
            with timed(self.stats, "feature"):
                try:
                    features = self._call_featurizer(origin, group_meta)
                except Exception as e:
                    features = {"features_error": str(e)}

        prepare: Dict[str, Any] = {"origin": origin, "features": features, "_meta": group_meta}

        # ✅ 기존 출력 유지: match / ml_anomaly_prob 키는 유지하되, 값은 배치 ML에서 채움
        prepare["match"] = None
        prepare["ml_anomaly_prob"] = []

        self.window_buffer.append(prepare)
        self.total_prepares += 1
        return prepare

    def _apply_ml_batch(self, prepares: List[Dict[str, Any]]) -> None:
        if not prepares:
            return

        t0 = time.perf_counter()
        match_ints, contribs = ml_batch_predict_and_contribs(
            prepares=prepares,
            ml_bundle=self.models.get("ml", {}),
            topk=int(self.ml_topk),
            hash_fallback=bool(self.ml_hash_fallback),
        )
        dt = time.perf_counter() - t0

        # 기존처럼 "packet 단위" stats 유지(배치 시간을 per-packet으로 분배)
        per_pkt = dt / max(1, len(prepares))
        for _ in prepares:
            self._ml_call_count += 1
            if self._ml_call_count > self.ml_skip_stats:
                self.stats["ml"].append(per_pkt)

        for pr, m, c in zip(prepares, match_ints, contribs):
            pr["match"] = m
            pr["ml_anomaly_prob"] = c

    def _emit_one_window(self, window_prepares: List[Dict[str, Any]]) -> None:
        self.seq_id += 1

        t0 = time.perf_counter()
        dl_out = predict_dl_models(prepares=window_prepares, models=self.models, seq_id=self.seq_id) or {}
        dt = time.perf_counter() - t0
        dl_ms = dt * 1000.0
        
        self._dl_call_count += 1
        if self._dl_call_count > self.dl_skip_stats:
            self.stats["dl_window"].append(dt)

        pattern, summary = parse_dl_output(dl_out)
        base_origin = (window_prepares[-1].get("origin") or {}) if window_prepares else {}
        summary = normalize_risk_fields(summary, base_origin)

        # alert 판단은 유지(로그/알람에 사용)
        alert_flag = bool(dl_out.get("alert")) if "alert" in dl_out else derive_alert(summary)
        alert = "o" if alert_flag else "x"

        # ✅ 최종 저장 포맷: seq_id/pattern/summary/window_raw
        window_raw = [build_window_raw_entry(pr) for pr in window_prepares]  # ✅ 80개
        final_record = {
            "seq_id": self.seq_id,
            "pattern": pattern,
            "summary": summary,     # ✅ DL summary (normalize_risk_fields 적용)
            "window_raw": window_raw,
            # ✅ DL 결과마다 걸린 시간 저장
            "timing": {
                "dl_infer_ms": round(dl_ms, 3),
                "dl_infer_sec": round(dt, 6),
            },            
        }
        self.stores.final.write_obj(final_record)

        if LOG_FINAL:
            rs = 0.0
            try:
                rs = float((summary.get("risk") or {}).get("score", 0.0))
            except Exception:
                rs = 0.0
            log_event(
                self.logger,
                "FINAL",
                seq_id=int(self.seq_id),
                pattern=str(pattern),
                alert=str(alert),
                risk_score=rs,
                anomaly_score=summary.get("anomaly_score"),
                threshold=summary.get("threshold"),
                dl_ms=round(dl_ms, 3),
            )
            _flush_logger(self.logger)

        if alert_flag and self.alarm_enabled and self.alarm_sender is not None:
            risk = (summary.get("risk") or {}) if isinstance(summary, dict) else {}
            self.alarm_sender.send_risk(
                risk if isinstance(risk, dict) else {},
                extra={"seq_id": int(self.seq_id), "pattern": pattern, "engine": "dl"},
            )

        self.total_windows += 1

    def _emit_windows_if_ready(self) -> int:
        emitted = 0
        while len(self.window_buffer) >= WINDOW_SIZE:
            window_prepares = self.window_buffer[:WINDOW_SIZE]
            self.window_buffer = self.window_buffer[WINDOW_STEP:]
            self._emit_one_window(window_prepares)
            emitted += 1
        return emitted

    def _store_incoming_packet(self, wrapped: Dict[str, Any]) -> None:
        """
        들어온 패킷 전체 저장 + wrapper meta에 ingest_id 심어서 추적 가능하게.
        """
        self.ingest_id += 1
        origin = wrapped.get("origin", {}) or {}
        meta = wrapped.get("_meta", {}) or {}
        meta["ingest_id"] = int(self.ingest_id)  # ✅ 추적용
        wrapped["_meta"] = meta

        rec = {
            "ingest_id": int(self.ingest_id),
            "protocol": origin.get("protocol") or meta.get("protocol"),
            "_meta": drop_none(sanitize_for_json(meta)),
            "origin": drop_none(sanitize_for_json(origin)),
        }
        self.stores.incoming.write_obj(rec)

    def flush_buffer(self) -> Tuple[int, int]:
        if not self.preprocessing_buffer:
            return (0, 0)

        created_prepares = 0
        emitted_windows = 0

        with timed(self.stats, "merge_total"):
            self.merge_event_id += 1
            merge_id = int(self.merge_event_id)

            wrappers = self.preprocessing_buffer
            origins = [w.get("origin", {}) for w in wrappers]
            group_meta = self._build_group_meta(wrappers)
            group_meta["merge_event_id"] = merge_id  # 추적용 (final 저장에는 포함 안 함)

            # reassembly_before 저장
            for i, w in enumerate(wrappers):
                meta = w.get("_meta") or {}
                rec = {
                    "merge_event_id": merge_id,
                    "idx": int(i),
                    "ingest_id": meta.get("ingest_id"),
                    "protocol": group_meta.get("protocol"),
                    "redis_id": meta.get("redis_id"),
                    "redis_timestamp_ms": meta.get("redis_timestamp_ms"),
                    "packet_timestamp_ms": meta.get("packet_timestamp_ms"),
                    "origin": drop_none(sanitize_for_json(w.get("origin") or {})),
                }
                self.stores.before.write_obj(rec)

            if LOG_MERGE:
                log_event(
                    self.logger,
                    "MERGE",
                    merge_event_id=merge_id,
                    protocol=str(group_meta.get("protocol")),
                    raw_count=int(len(origins)),
                    redis_ids=group_meta.get("redis_ids", []),
                    ingest_min=group_meta.get("ingest_id_min"),
                    ingest_max=group_meta.get("ingest_id_max"),
                    ts_min=int(group_meta.get("redis_timestamp_ms_min", 0)),
                    ts_max=int(group_meta.get("redis_timestamp_ms_max", 0)),
                )

            merged = merge_packets_by_modbus_diff(origins)

            # reassembly_after 저장
            for j, mp in enumerate(merged):
                rec = {
                    "merge_event_id": merge_id,
                    "idx": int(j),
                    "protocol": group_meta.get("protocol"),
                    "raw_count": int(len(origins)),
                    "merged_count": int(len(merged)),
                    "merged": drop_none(sanitize_for_json(mp)),
                }
                self.stores.after.write_obj(rec)

            new_prepares: List[Dict[str, Any]] = []

            for origin in merged:
                pr = self._append_prepare(origin, group_meta)
                new_prepares.append(pr)
                created_prepares += 1

            # ✅ FLUSH 단위 배치 ML
            self._apply_ml_batch(new_prepares)

            emitted_windows = self._emit_windows_if_ready()
            self.preprocessing_buffer = []

            if LOG_MERGE:
                _flush_logger(self.logger)

        return (created_prepares, emitted_windows)

    def run(self) -> None:
        self._run_t0 = time.perf_counter()
        self._run_started_at = datetime.now().isoformat()
        print("\n🚀 main_pipeline start")
        print(f"   - redis        : {self.redis_host}:{self.redis_port} (db={self.redis_db})")
        print(f"   - protocols    : {self.protocols}")
        print(f"   - pre_dir      : {DEFAULT_PRE_DIR}")
        print(f"   - model_load   : {DEFAULT_MODEL_LOAD_DIR}")
        print(f"   - window       : size={WINDOW_SIZE}, step={WINDOW_STEP}, partial={ALLOW_PARTIAL_WINDOW}")
        print(f"   - interval     : {self.interval_sec}s  (pps={(1.0/self.interval_sec) if self.interval_sec>0 else 'inf'})")
        print(f"   - replay       : {self.replay}")
        print(f"   - stop_after   : windows={self.stop_after_windows}")
        print(f"   - server_mode  : {self.server_mode}")
        print("   - pop ordering : payload @timestamp oldest-first")
        print(f"   - merge        : protocols={sorted(list(MERGE_PROTOCOLS))}, bucket_ms={MERGE_BUCKET_MS}")
        print(f"   - Alarm        : {'ON' if self.alarm_enabled else 'OFF'}")
        print("   - Files        : data_flow.log + incoming + reassembly(before/after) + final_results.json")
        print("=" * 80)

        log_event(
            self.logger,
            "RUN",
            created_at=datetime.now().isoformat(),
            argv=sys.argv,
            protocols=self.protocols,
            window={"size": WINDOW_SIZE, "step": WINDOW_STEP, "partial": ALLOW_PARTIAL_WINDOW},
            merge={"protocols": sorted(list(MERGE_PROTOCOLS)), "bucket_ms": MERGE_BUCKET_MS},
            alarm={"enabled": self.alarm_enabled, "base": DEFAULT_ALARM_BASE_URL, "engine": DEFAULT_ALARM_ENGINE},
            server_mode=self.server_mode,
        )
        _flush_logger(self.logger)

        last_hb = time.monotonic()

        while True:
            if self._stop_flag.is_set():
                print("\n🛑 stop requested (signal).")
                break

            if self.stop_after_windows is not None and self.total_windows >= int(self.stop_after_windows):
                print(f"\n✅ stop_after_windows={self.stop_after_windows} 만족하여 종료합니다.")
                break

            if self.total_raw_packets >= MAX_RAW_PACKETS:
                self.flush_buffer()
                break
            if self.total_prepares >= MAX_PREPARES:
                self.flush_buffer()
                break

            wrapped = None
            with timed(self.stats, "redis"):
                wrapped = self.server.pop_oldest()

            if not wrapped:
                self.flush_buffer()

                now = time.monotonic()
                if (now - last_hb) >= HEARTBEAT_SEC:
                    log_event(self.logger, "HEARTBEAT", windows=self.total_windows, raw_packets=self.total_raw_packets, prepares=self.total_prepares)
                    _flush_logger(self.logger)
                    last_hb = now

                if self.server_mode:
                    time.sleep(max(self.idle_sleep_sec, 0.001))
                    continue

                print("⚠️ Redis에 데이터 없음... 대기 중.")
                time.sleep(max(self.interval_sec, 0.01))
                continue

            if "origin_raw" in wrapped:
                raw = wrapped.pop("origin_raw")
                proto = wrapped.get("protocol")

                with timed(self.stats, "json_parse"):
                    try:
                        data = json.loads(raw)
                    except Exception:
                        data = {"raw": raw}

                if isinstance(data, dict):
                    data["protocol"] = proto
                else:
                    data = {"value": data, "protocol": proto}

                wrapped["origin"] = data

            self.total_raw_packets += 1
            self._store_incoming_packet(wrapped)

            if self.replay:
                meta = wrapped.get("_meta") or {}
                current_ts_ms = int(meta.get("redis_timestamp_ms", 0))
                if self.last_packet_ts_ms is not None:
                    delta_ms = current_ts_ms - self.last_packet_ts_ms
                    if delta_ms > 0:
                        time.sleep(delta_ms / 1000.0)
                self.last_packet_ts_ms = current_ts_ms

            origin = wrapped.get("origin", {}) or {}
            proto = origin.get("protocol", "")

            # non-merge protocol
            if proto not in self.MERGE_PROTOCOLS:
                if self.preprocessing_buffer:
                    self.flush_buffer()
                    group_meta = self._build_group_meta([wrapped])
                    pr = self._append_prepare(origin, group_meta)
                    self._apply_ml_batch([pr])  # ✅ 1개 배치
                    self._emit_windows_if_ready()


                if (not self.replay) and self.interval_sec > 0:
                    time.sleep(self.interval_sec)
                continue

            # merge protocol: group by relaxed key
            new_key = self._merge_key_from_wrapper(wrapped)
            last_key = None
            if self.preprocessing_buffer:
                last_key = self._merge_key_from_wrapper(self.preprocessing_buffer[-1])

            if self.preprocessing_buffer and new_key != last_key:
                self.flush_buffer()

            self.preprocessing_buffer.append(wrapped)

            if (not self.replay) and self.interval_sec > 0:
                time.sleep(self.interval_sec)

        # 종료 처리
        self.flush_buffer()
        if not ALLOW_PARTIAL_WINDOW and self.window_buffer:
            print(f"⚠️ window_buffer에 {len(self.window_buffer)}개가 남았지만 (partial off) 실행/저장하지 않습니다.")
        
        self._run_ended_at = datetime.now().isoformat()
        if self._run_t0 is not None:
            self._run_elapsed_sec = time.perf_counter() - self._run_t0
        self.print_statistics()

    def print_statistics(self) -> None:
        print("\n" + "=" * 80)
        print("📊 성능 요약")
        print("=" * 80)
        
        def _line_full(name: str, arr: List[float]) -> str:
            if not arr:
                return f"- {name:<14}: (no data)"
            ms = [x * 1000 for x in arr]
            p50 = np.percentile(ms, 50)
            p95 = np.percentile(ms, 95)
            return (
                f"- {name:<14}: avg={statistics.mean(ms):.3f}ms  "
                f"p50={p50:.3f}ms  p95={p95:.3f}ms  "
                f"min={min(ms):.3f}ms  max={max(ms):.3f}ms  n={len(ms)}"
            )

        def _trim_ms(ms: List[float], trim_pct: float) -> List[float]:
            if not ms or trim_pct <= 0:
                return ms
            ms_sorted = sorted(ms)
            n = len(ms_sorted)
            k = int(n * trim_pct)
            if n - 2 * k <= 2:
                return ms_sorted
            return ms_sorted[k : n - k]

        def _line_compact_with_trim(name: str, arr: List[float], trim_pct: float, skip_first: int) -> str:
            if not arr:
                return f"- {name:<14}: (no data)"
            ms_raw = [x * 1000 for x in arr]
            ms_used = _trim_ms(ms_raw, trim_pct)
            p50 = float(np.percentile(ms_used, 50))
            p95 = float(np.percentile(ms_used, 95))
            avg = float(statistics.mean(ms_used))
            min_raw = float(min(ms_raw))
            max_raw = float(max(ms_raw))
            suffix = f"  n={len(ms_used)}/{len(ms_raw)}"
            if trim_pct > 0:
                suffix += f"  (trim={trim_pct})"
            if skip_first > 0:
                suffix += f"  (skip_first={skip_first})"
            return (
                f"- {name:<14}: avg={avg:.3f}ms  p50={p50:.3f}ms  p95={p95:.3f}ms  "
                f"min={min_raw:.3f}ms  max={max_raw:.3f}ms"
                + suffix
            )

        print(f"- total_raw_packets : {self.total_raw_packets}")
        print(f"- total_prepares    : {self.total_prepares}")
        print(f"- total_windows     : {self.total_windows}")

        if self._run_elapsed_sec is not None:
            sec = float(self._run_elapsed_sec)
            pps_raw = (self.total_raw_packets / sec) if sec > 0 else 0.0
            pps_prep = (self.total_prepares / sec) if sec > 0 else 0.0
            wps = (self.total_windows / sec) if sec > 0 else 0.0

            print(f"- run_started_at    : {self._run_started_at}")
            print(f"- run_ended_at      : {self._run_ended_at}")
            print(f"- total_elapsed     : {sec:.3f}s")
            print(f"- throughput        : raw={pps_raw:.2f} pkt/s, prepares={pps_prep:.2f} prep/s, windows={wps:.2f} win/s")

        print(_line_full("redis", self.stats.get("redis", [])))

        mt = self.stats.get("merge_total", [])
        if not mt:
            print(
                f"- {'merge_total':<14}: "
                f"avg=0.000ms  p50=0.000ms  p95=0.000ms  "
                f"min=0.000ms  max=0.000ms  n=0  (no merge events)"
            )
        else:
            print(_line_full("merge_total", mt))

        print(_line_full("feature", self.stats.get("feature", [])))
        print(_line_compact_with_trim("ml", self.stats.get("ml", []), self.ml_trim_pct, self.ml_skip_stats))
        print(_line_compact_with_trim("dl_window", self.stats.get("dl_window", []), self.dl_trim_pct, self.dl_skip_stats))

        dlw = self.stats.get("dl_window", [])
        if dlw:
            ms = [x * 1000 for x in dlw]
            avg_ms = statistics.mean(ms)
            win_per_sec = 1000.0 / avg_ms if avg_ms > 0 else 0.0
            eff_pkt_per_sec_by_step = win_per_sec * WINDOW_STEP
            pkt_per_sec_by_full = win_per_sec * WINDOW_SIZE
            # print(f"  · DL throughput (by window calls): {win_per_sec:.3f} windows/sec")
            # print(f"  · DL effective throughput (new packets): ~{eff_pkt_per_sec_by_step:.3f} pkt/sec  (step={WINDOW_STEP})")
            # print(f"  · DL raw throughput (window-size basis): ~{pkt_per_sec_by_full:.3f} pkt/sec  (size={WINDOW_SIZE})")

        print("=" * 80)


# ============================================================
# main()
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="main_pipeline (data_flow.log + incoming + reassembly + final_results.json)")

    parser.add_argument("--host", default=DEFAULT_REDIS_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_REDIS_PORT)
    parser.add_argument("--db", type=int, default=DEFAULT_REDIS_DB)
    parser.add_argument("--password", default=DEFAULT_REDIS_PASSWORD)

    parser.add_argument("--protocols", nargs="+", default=None)

    speed = parser.add_mutually_exclusive_group()
    speed.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SEC)
    speed.add_argument("--pps", type=int, default=None)
    speed.add_argument("--replay", action="store_true", default=DEFAULT_REPLAY)

    parser.add_argument("--stop-after-windows", type=int, default=DEFAULT_STOP_AFTER_WINDOWS)
    parser.add_argument("--server", action="store_true", default=False, help="서버 모드(무한 루프, stop-after 무시)")
    parser.add_argument("--idle-sleep", type=float, default=DEFAULT_IDLE_SLEEP_SEC, help="server 모드 idle sleep (sec)")

    # ML
    parser.add_argument("--ml-topk", type=int, default=DEFAULT_ML_TOPK)
    parser.add_argument("--ml-warmup", type=int, default=DEFAULT_ML_WARMUP)
    parser.add_argument("--ml-skip-stats", type=int, default=DEFAULT_ML_SKIP_STATS)
    parser.add_argument("--ml-trim-pct", type=float, default=DEFAULT_ML_TRIM_PCT)
    parser.add_argument("--ml-no-hash-fallback", action="store_true", default=False)

    # DL
    parser.add_argument("--dl-warmup", type=int, default=DEFAULT_DL_WARMUP)
    parser.add_argument("--dl-skip-stats", type=int, default=DEFAULT_DL_SKIP_STATS)
    parser.add_argument("--dl-trim-pct", type=float, default=DEFAULT_DL_TRIM_PCT)

    # Alarm
    parser.add_argument("--alarm", action="store_true", default=DEFAULT_ALARM_ENABLED)
    parser.add_argument("--alarm-base-url", default=DEFAULT_ALARM_BASE_URL)
    parser.add_argument("--alarm-engine", default=DEFAULT_ALARM_ENGINE)
    parser.add_argument("--alarm-timeout", type=float, default=DEFAULT_ALARM_TIMEOUT)

    # run root
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))

    args = parser.parse_args()

    interval = float(args.interval)
    replay = bool(args.replay)
    if args.pps is not None:
        if args.pps <= 0:
            raise SystemExit("--pps must be > 0")
        interval = 1.0 / float(args.pps)
        replay = False

    # run dir
    run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_root = Path(args.run_root).resolve()
    run_dir = (run_root / run_tag).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # files
    log_path = run_dir / "data_flow.log"
    incoming_path = run_dir / "incoming_packets.jsonl"
    reasm_before_path = run_dir / "reassembly_before.jsonl"
    reasm_after_path = run_dir / "reassembly_after.jsonl"
    final_path = run_dir / "final_results.json"  # ✅ JSON(Array)

    logger = setup_data_logger(log_path, mode="w")
    stores = RunStores(incoming_path, reasm_before_path, reasm_after_path, final_path)

    stop_after = None if args.server else int(args.stop_after_windows)

    pipeline = OperationalPipeLine(
        logger=logger,
        stores=stores,
        redis_host=args.host,
        redis_port=args.port,
        redis_db=args.db,
        redis_password=args.password,
        protocols=args.protocols,
        interval_sec=interval,
        replay=replay,
        stop_after_windows=stop_after,
        # ML
        ml_topk=int(args.ml_topk),
        ml_warmup=int(args.ml_warmup),
        ml_skip_stats=int(args.ml_skip_stats),
        ml_trim_pct=float(args.ml_trim_pct),
        ml_hash_fallback=(not bool(args.ml_no_hash_fallback)),
        # DL
        dl_warmup=int(args.dl_warmup),
        dl_skip_stats=int(args.dl_skip_stats),
        dl_trim_pct=float(args.dl_trim_pct),
        # Alarm
        alarm_enabled=bool(args.alarm),
        alarm_base_url=str(args.alarm_base_url),
        alarm_engine=str(args.alarm_engine),
        alarm_timeout=float(args.alarm_timeout),
        # Server
        server_mode=bool(args.server),
        idle_sleep_sec=float(args.idle_sleep),
    )

    def _sig_handler(signum, frame):
        pipeline.request_stop()

    try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
    except Exception:
        pass

    try:
        pipeline.run()
    finally:
        pipeline.close()

    print(f"\n✅ run saved to: {run_dir}")
    print("   - data_flow.log")
    print("   - incoming_packets.jsonl")
    print("   - reassembly_before.jsonl")
    print("   - reassembly_after.jsonl")
    print("   - final_results.json")


if __name__ == "__main__":
    main()


"""
python main.py

# 알람 전달 O
python main.py --alarm

# reassembly before/after 저장 확인:
tail -f final_results/run_YYYYMMDD_HHMMSS/data_flow.log
wc -l final_results/run_YYYYMMDD_HHMMSS/reassembly_before.jsonl
wc -l final_results/run_YYYYMMDD_HHMMSS/reassembly_after.jsonl


python main.py --server --idle-sleep 0.01

python main.py --server --alarm

"""
