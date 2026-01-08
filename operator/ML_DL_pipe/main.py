#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py (Operational main_pipeline, optimized + robust)

핵심 요구 반영
- data_flow.log 1개 로그 파일 유지
- 추가 저장:
  1) incoming_packets.jsonl      : Redis에서 pop된 들어온 패킷 전체
  2) reassembly_before.jsonl     : 재조립(merge) 전(flush 단위)
  3) reassembly_after.jsonl      : 재조립(merge) 후(flush 단위)
  4) final_results.json          : 최종 반환 결과(window 단위 DL 결과) ✅ JSON(Array)

final_results.json 포맷:
  {
    "seq_id": 1,
    "pattern": "...",
    "summary": {...},
    "window_raw": [ {packet1}, {packet2}, ... ],
    "timing": {...}   # (옵션) window 단위 timing
  }

서버 모드:
  --server 옵션이면 stop-after 없이 무한 루프(pop→처리)로 동작

성능/안정성 포인트
- Redis stream 여러 개에서 XRANGE(count=1) 후 stream-id(ts_ms, seq) oldest-first pop
- merge 프로토콜은 flush 단위로 before/after 저장 + merge 후 prepares 생성
- ML: flush 단위 배치 예측(+topk contrib)
- DL: window 단위 예측 (DL_predict.predict_dl_models 호출)
- 파일 저장은 Async writer로 I/O amortize
- window_buffer는 ring-buffer(오프셋)로 list 슬라이싱 비용 최소화
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import statistics
import sys
import threading
import time
import queue
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import redis
import numpy as np

try:
    import requests  # Alarm 전송용
except Exception:
    requests = None

# --- orjson (optional) ---
try:
    import orjson  # type: ignore
except Exception:
    orjson = None


# ============================================================
# ✅ script dir -> sys.path (상대 import 이슈 방지)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ✅ DL predict (윈도우 단위 호출)
from model_predict.DL_predict import predict_dl_models  # noqa: E402


# ============================================================
# ✅ 기본 설정(기본값은 CLI로 override)
# ============================================================
DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_DB = 0
DEFAULT_REDIS_PASSWORD = None

DEFAULT_PROTOCOLS = ["modbus", "s7comm", "xgt_fen", "tcp", "udp", "dns", "arp"]

DEFAULT_WINDOW_SIZE = 80
DEFAULT_WINDOW_STEP = 20
DEFAULT_ALLOW_PARTIAL_WINDOW = False

DEFAULT_STOP_AFTER_WINDOWS = 1  # server 모드면 None
MAX_RAW_PACKETS = 10_000_000
MAX_PREPARES = 10_000_000

DEFAULT_INTERVAL_SEC = 0.0
DEFAULT_REPLAY = False

DEFAULT_PRE_DIR = (SCRIPT_DIR / ".." / "preprocessing" / "result").resolve()
DEFAULT_MODEL_LOAD_DIR = (SCRIPT_DIR / "model_load").resolve()

DEFAULT_RUN_ROOT = (SCRIPT_DIR / "final_results").resolve()

# ----------------------------
# data_flow.log 기록 옵션
# ----------------------------
LOG_MERGE = True
LOG_FINAL = True
LOG_FSYNC = False
DATA_FLOW_MAX_JSON_CHARS = 4000

# ----------------------------
# Async writer tuning
# ----------------------------
ASYNC_BATCH = 300
ASYNC_FLUSH_SEC = 0.7

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
DEFAULT_MERGE_PROTOCOLS = {"modbus", "xgt_fen", "xgt-fen"}
DEFAULT_MERGE_BUCKET_MS = 3

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
# JSON (fast path)
# ============================================================
_ORJSON_OPT = 0
if orjson is not None:
    _ORJSON_OPT = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY


def _json_loads_fast(s: str) -> Any:
    if orjson is not None:
        try:
            return orjson.loads(s)
        except Exception:
            pass
    return json.loads(s)


def _json_dumps_bytes(obj: Any) -> bytes:
    if orjson is not None:
        try:
            return orjson.dumps(obj, option=_ORJSON_OPT, default=str)
        except Exception:
            pass
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


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


def sanitize_and_drop_none(obj: Any) -> Any:
    """sanitize_for_json + drop_none 단일 패스."""
    obj_type = type(obj)

    if obj_type is dict:
        result = {}
        for k, v in obj.items():
            cleaned = sanitize_and_drop_none(v)
            if cleaned is not None:
                result[str(k)] = cleaned
        return result

    if obj_type is list:
        return [x for x in (sanitize_and_drop_none(item) for item in obj) if x is not None]

    if obj_type is str or obj_type is int or obj_type is bool:
        return obj

    if obj is None:
        return None

    if obj_type is float:
        return None if _is_nan_inf(obj) else obj

    if obj_type is tuple or obj_type is set:
        return [x for x in (sanitize_and_drop_none(item) for item in obj) if x is not None]

    try:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if _is_nan_inf(v) else v
        if isinstance(obj, np.ndarray):
            return [x for x in (sanitize_and_drop_none(item) for item in obj.tolist()) if x is not None]
    except Exception:
        pass

    return str(obj)


# ============================================================
# Async Writers (binary, bytes-dump)
# ============================================================
class AsyncJsonlWriter:
    __slots__ = ("path", "batch", "flush_sec", "q", "stop_event", "thread")

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

        with self.path.open("ab", buffering=1024 * 1024) as f:
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
                        out = bytearray()
                        for o in buf:
                            try:
                                out += _json_dumps_bytes(o)
                            except Exception:
                                out += b'{"_error":"json_dumps_failed"}'
                            out += b"\n"
                        f.write(out)
                        f.flush()
                    except Exception:
                        pass
                    buf.clear()
                    last_flush = now

            if buf:
                try:
                    out = bytearray()
                    for o in buf:
                        try:
                            out += _json_dumps_bytes(o)
                        except Exception:
                            out += b'{"_error":"json_dumps_failed"}'
                        out += b"\n"
                    f.write(out)
                    f.flush()
                except Exception:
                    pass

    def write_obj(self, obj: Dict[str, Any]) -> None:
        self.q.put(obj)

    def close(self) -> None:
        self.stop_event.set()
        self.q.join()
        self.thread.join()


class AsyncJsonArrayWriter:
    __slots__ = ("path", "batch", "flush_sec", "q", "stop_event", "thread", "_first")

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

        with self.path.open("wb", buffering=1024 * 1024) as f:
            f.write(b"[\n")
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
                    self._flush_buf(f, buf)
                    buf.clear()
                    last_flush = now

            if buf:
                self._flush_buf(f, buf)
                buf.clear()

            f.write(b"\n]\n")
            f.flush()

    def _flush_buf(self, f, buf: List[Dict[str, Any]]) -> None:
        try:
            out = bytearray()
            for o in buf:
                try:
                    b = _json_dumps_bytes(o)
                except Exception:
                    b = b'{"_error":"json_dumps_failed"}'
                if self._first:
                    out += b
                    self._first = False
                else:
                    out += b",\n" + b
            f.write(out)
            f.flush()
        except Exception:
            pass

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
    - reassembly_before.jsonl   : merge 전(flush 단위)
    - reassembly_after.jsonl    : merge 후(flush 단위)
    - final_results.json        : 최종 반환(window 단위) ✅ JSON(Array)
    """
    __slots__ = ("incoming", "before", "after", "final")

    def __init__(self, incoming_path: Path, before_path: Path, after_path: Path, final_path: Path):
        self.incoming = AsyncJsonlWriter(incoming_path, batch=ASYNC_BATCH, flush_sec=ASYNC_FLUSH_SEC)
        self.before = AsyncJsonlWriter(before_path, batch=ASYNC_BATCH, flush_sec=ASYNC_FLUSH_SEC)
        self.after = AsyncJsonlWriter(after_path, batch=ASYNC_BATCH, flush_sec=ASYNC_FLUSH_SEC)
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
# Alarm sender (비동기 POST)
# ============================================================
class AlarmSender:
    __slots__ = ("base_url", "engine", "timeout", "logger", "q", "stop_event", "thread")

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
# model loader (중요: DL-anomaly는 "bundle 통째로" 유지)
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

def ensure_lstm_ae_tf_cache(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    DL_predict.py가 기대하는 TF cache 키들을 번들에 보장:
      - _tf_pad, _tf_miss, _tf_fw, _tf_t_idx, _ae_scorer_cache
    """
    if not isinstance(bundle, dict):
        return bundle

    try:
        import tensorflow as tf  # lazy import
        import numpy as np
    except Exception:
        return bundle

    # values
    try:
        pad_v = float(bundle.get("pad_value", -1.0))
    except Exception:
        pad_v = -1.0
    try:
        miss_v = float(bundle.get("missing_value", -2.0))
    except Exception:
        miss_v = -2.0

    # required: pad/miss
    if "_tf_pad" not in bundle:
        bundle["_tf_pad"] = tf.constant(pad_v, dtype=tf.float32)
    if "_tf_miss" not in bundle:
        bundle["_tf_miss"] = tf.constant(miss_v, dtype=tf.float32)

    # optional: feature weights
    fw = bundle.get("feature_weights", None)
    if fw is not None and "_tf_fw" not in bundle:
        try:
            fw_np = np.asarray(fw, dtype=np.float32)
            bundle["_tf_fw"] = tf.constant(fw_np, dtype=tf.float32)
        except Exception:
            pass

    # optional but used as fallback in DL_predict
    if "_tf_t_idx" not in bundle:
        try:
            T = int((bundle.get("config") or {}).get("T") or 0)
        except Exception:
            T = 0
        if T > 0:
            bundle["_tf_t_idx"] = tf.range(T, dtype=tf.int64)[None, :]  # (1,T)

    # scorer cache
    if "_ae_scorer_cache" not in bundle or not isinstance(bundle.get("_ae_scorer_cache"), dict):
        bundle["_ae_scorer_cache"] = {}

    return bundle


def load_and_cache_3_models(model_load_dir: Path, *, dl_threshold_fixed: Optional[float] = 0.32) -> Dict[str, Any]:
    """
    out 구조:
      out["ml"]         = {"enabled": True, ...}
      out["dl_anomaly"] = <DL-anomaly bundle dict>   ✅ 통째로 유지
      out["dl_pattern"] = <DL-pattern bundle dict>   ✅ 통째로 유지
      out["_timing"]    = {...}
    """
    out: Dict[str, Any] = {
        "ml": {"enabled": False},
        "dl_anomaly": None,
        "dl_pattern": None,
    }

    model_load_dir = Path(model_load_dir)
    ml_dir = (model_load_dir / "ML_model").resolve()
    dl_anom_dir = (model_load_dir / "DL_model_anomaly").resolve()
    dl_pat_dir = (model_load_dir / "DL_model_pattern").resolve()

    timing: Dict[str, float] = {}
    t_total0 = time.perf_counter()

    # -------- ML --------
    t0 = time.perf_counter()
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
    timing["ml_load_s"] = time.perf_counter() - t0

    # -------- DL-anomaly --------
    t0 = time.perf_counter()
    try:
        dl_anom_loader_py = model_load_dir / "model_load(DL-anomaly).py"
        dl_anom_mod = _import_from_file("_dl_anom_loader", dl_anom_loader_py)

        # load_lstm_ae_bundle 사용 (전체 bundle 로드)
        bundle = dl_anom_mod.load_lstm_ae_bundle(dl_anom_dir)  # type: ignore
        if not isinstance(bundle, dict):
            raise TypeError("load_lstm_ae_bundle() must return dict")

        # 사용자 고정 threshold (bundle의 threshold를 덮어씀)
        bundle["threshold"] = 0.32

        # enabled 플래그 유지
        bundle["enabled"] = True

        # DL_predict가 요구하는 TF 캐시 키 보장
        ensure_lstm_ae_tf_cache(bundle)

        out["dl_anomaly"] = bundle
        print(f"✓ DL-anomaly loaded: {dl_anom_dir}")
    except Exception as e:
        print(f"⚠️ DL-anomaly load failed: {e}")
    timing["dl_anom_load_s"] = time.perf_counter() - t0


    # -------- DL-pattern --------
    t0 = time.perf_counter()
    try:
        dl_pat_loader_py = model_load_dir / "model_load(DL-pattern).py"
        dl_pat_mod = _import_from_file("_dl_pat_loader", dl_pat_loader_py)

        ckpt = None
        for name in [
            "best_model.h5", "best_model.pt", "best_model.pth",
            "model.h5", "model.pt", "model.pth",
            "checkpoint.h5", "checkpoint.pt", "checkpoint.pth",
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

        pat_bundle = dl_pat_mod.load_dl_torch_bundle(ckpt)  # type: ignore
        if not isinstance(pat_bundle, dict) or "model" not in pat_bundle:
            raise ValueError("DL-pattern loader returned invalid bundle")

        out["dl_pattern"] = pat_bundle
        print(f"✓ DL-pattern loaded: {ckpt}")
    except Exception as e:
        print(f"⚠️ DL-pattern load failed: {e}")
        out["dl_pattern"] = None
    timing["dl_pat_load_s"] = time.perf_counter() - t0

    timing["total_load_s"] = time.perf_counter() - t_total0
    out["_timing"] = timing
    return out


# ============================================================
# ML (batch)
# ============================================================
def _stable_hash(s: str, mod: int = 1000) -> int:
    import hashlib
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return int(h, 16) % mod


def ml_batch_predict_and_contribs(
    prepares: List[Dict[str, Any]],
    ml_bundle: Dict[str, Any],
    *,
    topk: int = 2,
    hash_fallback: bool = True,
) -> Tuple[List[Optional[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
    n = len(prepares)
    if n == 0 or (not ml_bundle.get("enabled")):
        return [None] * n, [[] for _ in range(n)]

    model = ml_bundle.get("model")
    scaler = ml_bundle.get("scaler")
    selected_features: List[str] = list(ml_bundle.get("selected_features") or [])

    if model is None:
        return [None] * n, [[] for _ in range(n)]

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

    X = np.zeros((n, d), dtype=np.float32)
    feat2idx = {k: j for j, k in enumerate(selected_features)}

    for i, pr in enumerate(prepares):
        feat = pr.get("features")
        if not isinstance(feat, dict):
            origin = pr.get("origin") or {}
            feat = {"protocol": origin.get("protocol", "unknown")}

        for k, v in feat.items():
            j = feat2idx.get(k)
            if j is None:
                continue
            try:
                if isinstance(v, bool):
                    X[i, j] = float(int(v))
                elif v is not None:
                    X[i, j] = float(v)
            except (ValueError, TypeError):
                if hash_fallback:
                    X[i, j] = float(_stable_hash(str(v)))

    try:
        Xs = scaler.transform(X) if scaler is not None else X
    except Exception:
        Xs = X

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
                        p1 = np.clip(proba[:, idx1].astype(float), 0.0, 1.0)
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

    pred = None
    try:
        if hasattr(model, "predict"):
            pred = np.ravel(model.predict(Xs))
    except Exception:
        pred = None

    match_chars = ["X"] * n
    if pred is not None and len(pred) == n:
        for i in range(n):
            p = pred[i]
            try:
                if hasattr(p, "item"):
                    p = p.item()
            except Exception:
                pass
            match_chars[i] = "O" if (str(p) == "1" or p == 1) else "X"
    else:
        for i in range(n):
            match_chars[i] = "X" if float(a_prob[i]) >= 0.5 else "O"

    ml_infos: List[Optional[Dict[str, Any]]] = [None] * n
    for i in range(n):
        ap = float(a_prob[i])
        mch = match_chars[i]
        mp = (1.0 - ap) if mch == "O" else ap
        ml_infos[i] = {
            "match": mch,
            "match_확률": float(round(mp * 100.0, 3)),
            "anomaly_prob": float(round(ap, 6)),
        }

    if topk <= 0:
        return ml_infos, [[] for _ in range(n)]

    topk_eff = int(min(topk, d))

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

    Xsf = Xs.astype(np.float32, copy=False)
    if W is not None:
        contrib_mat = np.abs(Xsf * W.astype(np.float32).reshape(1, -1))
    else:
        contrib_mat = np.abs(Xsf)

    sums = contrib_mat.sum(axis=1)
    sums_safe = np.where(sums > 0, sums, 1.0)

    if topk_eff >= d:
        top_idx = np.argsort(-contrib_mat, axis=1)
    else:
        top_idx = np.argpartition(-contrib_mat, topk_eff - 1, axis=1)[:, :topk_eff]
        row_idx = np.arange(n)[:, None]
        sorted_order = np.argsort(-contrib_mat[row_idx, top_idx], axis=1)
        top_idx = top_idx[row_idx, sorted_order]

    contribs: List[List[Dict[str, Any]]] = []
    for i in range(n):
        s = float(sums_safe[i])
        if s <= 0 or not np.isfinite(s):
            contribs.append([])
            continue

        out_i = []
        for j in top_idx[i, :topk_eff]:
            pct = float(contrib_mat[i, j] / s * 100.0)
            out_i.append({"name": selected_features[int(j)], "percent": round(pct, 2)})
        contribs.append(out_i)

    return ml_infos, contribs


# ============================================================
# Redis Pop Server (stream-id oldest-first)
# ============================================================
class RedisPopServer:
    __slots__ = ("host", "port", "db", "password", "protocols", "redis_client")

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

    def pop_oldest(self) -> Optional[Dict[str, Any]]:
        try:
            rc = self.redis_client
            if rc is None:
                self.connect()
                rc = self.redis_client
                if rc is None:
                    return None

            pipe = rc.pipeline(transaction=False)

            stream_infos = []
            for p in self.protocols:
                sname = f"stream:protocol:{p}"
                stream_infos.append((p, sname))
                pipe.xrange(sname, "-", "+", count=1)

            results = pipe.execute()

            best = None  # (proto, sname, msg_id, raw, id_ts, id_seq)
            for (proto, sname), msgs in zip(stream_infos, results):
                if not msgs:
                    continue
                msg_id, fields = msgs[0]
                raw = fields.get("data")
                if raw is None:
                    continue

                id_ts, id_seq = self.parse_id(msg_id)
                cand = (proto, sname, msg_id, raw, int(id_ts), int(id_seq))

                if best is None:
                    best = cand
                else:
                    _, _, _, _, best_id_ts, best_id_seq = best
                    if (id_ts < best_id_ts) or (id_ts == best_id_ts and id_seq < best_id_seq):
                        best = cand

            if best is None:
                return None

            proto, sname, msg_id, raw, id_ts, _ = best
            rc.xdel(sname, msg_id)

            meta = {
                "redis_id": msg_id,
                "redis_timestamp_ms": int(id_ts or 0),
                "packet_timestamp_ms": int(id_ts or 0),
                "pop_time": datetime.now().isoformat(),
                "protocol": proto,
            }
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


def merge_packets_by_modbus_diff(logs: List[Dict[str, Any]], merge_protocols: set[str]) -> List[Dict[str, Any]]:
    """프로토콜별 패킷 병합 (단일 패스)."""
    if not logs:
        return []
    proto = logs[0].get("protocol", "")
    if proto not in merge_protocols or len(logs) == 1:
        return [logs[0]] if logs else []

    base_packet = dict(logs[0])

    if proto in ("xgt_fen", "xgt-fen"):
        prefixes = ("xgt_fen.", "xgt-fen.")
    else:
        prefixes = (proto + ".",)

    key_values: Dict[str, List[Any]] = {}
    for p in logs:
        for k, v in p.items():
            if v is None:
                continue
            for pref in prefixes:
                if k.startswith(pref):
                    key_values.setdefault(k, []).append(v)
                    break

    for key, vals in key_values.items():
        if not vals:
            continue
        first_val = vals[0]
        if all(v == first_val for v in vals[1:]):
            base_packet[key] = first_val
        else:
            base_packet[key] = vals[0] if len(vals) == 1 else vals

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
    s = summary if isinstance(summary, dict) else {}
    risk = s.get("risk")
    if not isinstance(risk, dict):
        risk = {}

    detected_at = base_origin.get("@timestamp") or base_origin.get("timestamp") or datetime.now().isoformat()
    if risk.get("detected_time") in (None, "", "null"):
        risk["detected_time"] = detected_at

    try:
        risk_score = risk.get("score", 0.0)
        risk["score"] = float(risk_score) if risk_score is not None else 0.0
    except Exception:
        risk["score"] = 0.0

    if risk.get("src_ip") is None:
        risk["src_ip"] = base_origin.get("sip")
    if risk.get("dst_ip") is None:
        risk["dst_ip"] = base_origin.get("dip")
    if risk.get("src_asset") is None:
        risk["src_asset"] = base_origin.get("src_asset")
    if risk.get("dst_asset") is None:
        risk["dst_asset"] = base_origin.get("dst_asset")

    s["risk"] = risk

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
    if v is None:
        return None
    if isinstance(v, (int, float, bool, np.integer, np.floating)):
        return sanitize_and_drop_none(v)
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
            if any(c in s for c in (".", "e", "E")):
                return float(s)
        except Exception:
            pass
        return v
    return sanitize_and_drop_none(v)


def _origin_to_nested(origin: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(origin, dict):
        return {}

    out: Dict[str, Any] = {}
    coerce = _coerce_scalar
    for k, v in origin.items():
        if "." in k:
            prefix, sub = k.split(".", 1)
            cur = out.get(prefix)
            if not isinstance(cur, dict):
                if prefix in out and cur is not None and not isinstance(cur, dict):
                    out[prefix] = {"_value": cur}
                else:
                    out[prefix] = {}
            out[prefix][sub] = coerce(v)
        else:
            out[k] = coerce(v)

    for nk in ("len", "sp", "dp"):
        if nk in out:
            iv = _to_int(out.get(nk))
            if iv is not None:
                out[nk] = iv

    out.setdefault("sip", None)
    out.setdefault("dip", None)
    out.setdefault("src_asset", None)
    out.setdefault("dst_asset", None)

    return out


def build_window_raw_entry(prepare: Dict[str, Any]) -> Dict[str, Any]:
    origin = prepare.get("origin", {}) or {}
    pkt = _origin_to_nested(origin)

    mlv = prepare.get("ml")
    if isinstance(mlv, dict) and mlv:
        pkt["ml"] = mlv

    pkt["ml_anomaly_prob"] = prepare.get("ml_anomaly_prob") or []
    return sanitize_and_drop_none(pkt)


# ============================================================
# main pipeline
# ============================================================
class OperationalPipeLine:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        stores: RunStores,
        models: Dict[str, Any],
        redis_host: str,
        redis_port: int,
        redis_db: int,
        redis_password: Optional[str],
        protocols: List[str],
        interval_sec: float,
        replay: bool,
        stop_after_windows: Optional[int],
        server_mode: bool,
        idle_sleep_sec: float,
        # window
        window_size: int,
        window_step: int,
        allow_partial_window: bool,
        # merge
        merge_protocols: set[str],
        merge_bucket_ms: int,
        # ML
        ml_topk: int,
        ml_warmup: int,
        ml_skip_stats: int,
        ml_trim_pct: float,
        ml_hash_fallback: bool,
        # DL
        dl_warmup: int,
        dl_skip_stats: int,
        dl_trim_pct: float,
        # Alarm
        alarm_enabled: bool,
        alarm_base_url: str,
        alarm_engine: str,
        alarm_timeout: float,
        pre_dir: Path,
    ):
        self.logger = logger
        self.stores = stores
        
        # ✅ main에서 models를 넘겨줬으면 그대로 사용 (재로드 금지)
        if models is not None:
            self.models = models
        else:
            self.models = load_and_cache_3_models(model_load_dir=self.model_load_dir)

        # ✅ dl_anomaly 번들에 TF cache 키 보장 (self.models 세팅 이후에만 가능)
        if isinstance(self.models.get("dl_anomaly"), dict):
            ensure_lstm_ae_tf_cache(self.models["dl_anomaly"])


        self.redis_host = redis_host
        self.redis_port = int(redis_port)
        self.redis_db = int(redis_db)
        self.redis_password = redis_password

        self.protocols = list(protocols)
        if "xgt_fen" in self.protocols and "xgt-fen" not in self.protocols:
            self.protocols.append("xgt-fen")
        if "xgt-fen" in self.protocols and "xgt_fen" not in self.protocols:
            self.protocols.append("xgt_fen")

        self.interval_sec = float(interval_sec)
        self.replay = bool(replay)
        self.stop_after_windows = stop_after_windows
        self.server_mode = bool(server_mode)
        self.idle_sleep_sec = float(max(0.0, idle_sleep_sec))

        # window
        self.window_size = int(window_size)
        self.window_step = int(window_step)
        self.allow_partial_window = bool(allow_partial_window)

        if self.window_size <= 0 or self.window_step <= 0:
            raise ValueError("window_size/window_step must be > 0")
        if self.window_step > self.window_size:
            raise ValueError("window_step must be <= window_size")

        # merge
        self.merge_protocols = set(merge_protocols)
        self.merge_bucket_ms = int(max(1, merge_bucket_ms))

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

        self.pre_dir = Path(pre_dir).resolve()
        self.featurizer = self._init_featurizer(self.pre_dir)

        # buffers
        self.preprocessing_buffer: List[Dict[str, Any]] = []

        # window ring-buffer
        self.window_buffer: List[Dict[str, Any]] = []
        self._wb_head = 0  # head offset

        # counters
        self.seq_id = 0
        self.total_raw_packets = 0
        self.total_prepares = 0
        self.total_windows = 0

        # meta
        self.stats: Dict[str, Any] = defaultdict(list)
        self.last_packet_ts_ms: Optional[int] = None
        self.merge_event_id = 0
        self.ingest_id = 0

        self._stop_flag = threading.Event()

        if self.ml_warmup > 0:
            self._warmup_ml(self.ml_warmup)
        if self.dl_warmup > 0:
            self._warmup_dl(self.dl_warmup)

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

    def _init_featurizer(self, pre_dir: Path):
        if PacketFeaturePreprocessor is None:
            print(f"⚠️ PacketFeaturePreprocessor import 실패: {_PFP_IMPORT_ERR}")
            print("⚠️ 피처 전처리 없이 features={}로만 진행됩니다.")
            return None

        if not pre_dir.exists():
            print(f"⚠️ pre_dir 없음: {pre_dir}")
            print("⚠️ 피처 전처리 없이 features={}로만 진행됩니다.")
            return None

        try:
            featurizer = PacketFeaturePreprocessor(
                Path(pre_dir),
                allow_new_ids=False,
                index_source="redis_id",
                include_index=False,
            )
            print(f"✓ PacketFeaturePreprocessor 로드 완료: {pre_dir}")
            return featurizer
        except Exception as e:
            print(f"⚠️ PacketFeaturePreprocessor init 실패: {e}")
            return None

    def _warmup_ml(self, n: int) -> None:
        mlb = self.models.get("ml", {})
        if not mlb.get("enabled"):
            print("⚠️ ML warmup skip: ml bundle disabled")
            return
        dummy = {"origin": {"protocol": "warmup"}, "features": {"protocol": 0.0}}
        dummies = [dummy] * max(1, int(n))
        _ = ml_batch_predict_and_contribs(dummies, mlb, topk=0, hash_fallback=self.ml_hash_fallback)
        print(f"✓ ML warmup done: n={n}")

    def _warmup_dl(self, n: int) -> None:
        try:
            dummy_prepare = {"origin": {"protocol": "warmup"}, "features": {}, "_meta": {"redis_id": "warmup"}}
            dummy_window = [dummy_prepare] * int(self.window_size)
            for i in range(max(1, int(n))):
                _ = predict_dl_models(prepares=dummy_window, models=self.models, seq_id=-(i + 1)) or {}
            print(f"✓ DL warmup done: n={n}")
        except Exception as e:
            print(f"⚠️ DL warmup failed (ignored): {e}")

    def _merge_key_from_wrapper(self, wrapped: Dict[str, Any]) -> Optional[Tuple[str, Any, Any]]:
        origin = wrapped.get("origin", {}) or {}
        meta = wrapped.get("_meta", {}) or {}
        proto = origin.get("protocol", "")
        if proto not in self.merge_protocols:
            return None

        proto_norm = "xgt_fen" if proto == "xgt-fen" else str(proto)

        for k in (
            "sq", "modbus.sq", "xgt_fen.sq", "xgt-fen.sq",
            "transaction_id", "trans_id", "tid", "modbus.tid", "modbus.transaction_id",
            "xgt_fen.invoke_id", "xgt-fen.invoke_id",
        ):
            v = origin.get(k)
            if v is not None and str(v) != "":
                sq = str(v)
                ts = origin.get("@timestamp") or origin.get("timestamp")
                if ts is not None and str(ts) != "":
                    return (proto_norm, str(ts), sq)
                break

        pkt_ts_ms = meta.get("packet_timestamp_ms") or meta.get("redis_timestamp_ms")
        try:
            bucket = int(pkt_ts_ms) // int(self.merge_bucket_ms)
        except Exception:
            bucket = 0
        return (proto_norm, bucket, "bucket")

    def _build_group_meta(self, wrappers: List[Dict[str, Any]]) -> Dict[str, Any]:
        ids: List[Any] = []
        tmin = None
        tmax = None
        pops_first = None
        pops_last = None
        ingest_min = None
        ingest_max = None
        proto = None

        for w in wrappers:
            m = w.get("_meta") or {}
            if proto is None:
                proto = m.get("protocol")
            rid = m.get("redis_id")
            if rid is not None:
                ids.append(rid)

            ts = m.get("redis_timestamp_ms")
            if ts is not None:
                try:
                    ts_i = int(ts)
                    tmin = ts_i if tmin is None else min(tmin, ts_i)
                    tmax = ts_i if tmax is None else max(tmax, ts_i)
                except Exception:
                    pass

            pt = m.get("pop_time")
            if pt is not None and pops_first is None:
                pops_first = pt
            if pt is not None:
                pops_last = pt

            ig = m.get("ingest_id")
            if ig is not None:
                try:
                    ig = int(ig)
                    ingest_min = ig if ingest_min is None else min(ingest_min, ig)
                    ingest_max = ig if ingest_max is None else max(ingest_max, ig)
                except Exception:
                    pass

        return {
            "protocol": proto,
            "redis_id": ids[0] if ids else None,
            "redis_ids": ids,
            "redis_timestamp_ms_min": int(tmin or 0),
            "redis_timestamp_ms_max": int(tmax or 0),
            "pop_time_first": pops_first,
            "pop_time_last": pops_last,
            "raw_count": len(wrappers),
            "ingest_id_min": ingest_min,
            "ingest_id_max": ingest_max,
        }

    def _call_featurizer(self, origin: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        if self.featurizer is None:
            return {}

        proto_norm = _normalize_protocol_for_features(str(origin.get("protocol", "")))
        if origin.get("protocol") == proto_norm:
            origin_for_feat = origin
        else:
            origin_for_feat = dict(origin)
            origin_for_feat["protocol"] = proto_norm

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

        # ML 결과 필드(아래 build_window_raw_entry가 보는 키)
        prepare["ml"] = {}
        prepare["match"] = None
        prepare["ml_anomaly_prob"] = []

        self.window_buffer.append(prepare)
        self.total_prepares += 1
        return prepare

    def _apply_ml_batch(self, prepares: List[Dict[str, Any]]) -> None:
        if not prepares:
            return

        t0 = time.perf_counter()
        ml_infos, contribs = ml_batch_predict_and_contribs(
            prepares=prepares,
            ml_bundle=self.models.get("ml", {}),
            topk=int(self.ml_topk),
            hash_fallback=bool(self.ml_hash_fallback),
        )
        dt = time.perf_counter() - t0

        per_pkt = dt / max(1, len(prepares))
        for _ in prepares:
            self._ml_call_count += 1
            if self._ml_call_count > self.ml_skip_stats:
                self.stats["ml"].append(per_pkt)

        for pr, mi, c in zip(prepares, ml_infos, contribs):
            if isinstance(mi, dict):
                pr["ml"] = mi
                pr["match"] = 1 if mi.get("match") == "O" else 0
            else:
                pr["ml"] = {}
                pr["match"] = None
            pr["ml_anomaly_prob"] = c

    def _emit_one_window(self, window_prepares: List[Dict[str, Any]]) -> None:
        self.seq_id += 1

        timing: Dict[str, Any] = {}
        t0 = time.perf_counter()
        dl_out = predict_dl_models(prepares=window_prepares, models=self.models, seq_id=self.seq_id) or {}
        dt = time.perf_counter() - t0
        timing["dl_s"] = float(round(dt, 6))
        dl_ms = dt * 1000.0

        self._dl_call_count += 1
        if self._dl_call_count > self.dl_skip_stats:
            self.stats["dl_window"].append(dt)

        pattern, summary = parse_dl_output(dl_out)
        base_origin = (window_prepares[-1].get("origin") or {}) if window_prepares else {}
        summary = normalize_risk_fields(summary, base_origin)

        alert_flag = bool(dl_out.get("alert")) if "alert" in dl_out else derive_alert(summary)
        alert = "o" if alert_flag else "x"

        window_raw = [build_window_raw_entry(pr) for pr in window_prepares]
        final_record = {
            "seq_id": int(self.seq_id),
            "pattern": str(pattern),
            "summary": summary,
            "window_raw": window_raw,
            # "timing": timing,
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
        # ring-buffer head 기준으로 검사
        while (len(self.window_buffer) - self._wb_head) >= self.window_size:
            s = self._wb_head
            e = s + self.window_size
            window_prepares = self.window_buffer[s:e]
            self._wb_head += self.window_step
            self._emit_one_window(window_prepares)
            emitted += 1

            # head가 많이 커지면 compact
            if self._wb_head > 4096 and self._wb_head > (len(self.window_buffer) // 2):
                self.window_buffer = self.window_buffer[self._wb_head :]
                self._wb_head = 0

        return emitted

    def _store_incoming_packet(self, wrapped: Dict[str, Any]) -> None:
        self.ingest_id += 1
        origin = wrapped.get("origin", {}) or {}
        meta = wrapped.get("_meta", {}) or {}
        meta["ingest_id"] = int(self.ingest_id)
        wrapped["_meta"] = meta

        rec = {
            "ingest_id": int(self.ingest_id),
            "protocol": origin.get("protocol") or meta.get("protocol"),
            "_meta": sanitize_and_drop_none(meta),
            "origin": sanitize_and_drop_none(origin),
        }
        self.stores.incoming.write_obj(rec)

    def flush_buffer(self) -> Tuple[int, int]:
        if not self.preprocessing_buffer:
            return (0, 0)

        created_prepares = 0
        emitted_windows = 0

        t0 = time.perf_counter()
        with timed(self.stats, "merge_only"):
            self.merge_event_id += 1
            merge_id = int(self.merge_event_id)

            wrappers = self.preprocessing_buffer
            origins = [w.get("origin", {}) for w in wrappers]
            group_meta = self._build_group_meta(wrappers)
            group_meta["merge_event_id"] = merge_id

            # before
            with timed(self.stats, "merge_before_enqueue"):
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
                        "origin": sanitize_and_drop_none(w.get("origin") or {}),
                    }
                    self.stores.before.write_obj(rec)

            with timed(self.stats, "merge_compute"):
                merged = merge_packets_by_modbus_diff(origins, self.merge_protocols)

            # after
            with timed(self.stats, "merge_after_enqueue"):
                for j, mp in enumerate(merged):
                    rec = {
                        "merge_event_id": merge_id,
                        "idx": int(j),
                        "protocol": group_meta.get("protocol"),
                        "raw_count": int(len(origins)),
                        "merged_count": int(len(merged)),
                        "merged": sanitize_and_drop_none(mp),
                    }
                    self.stores.after.write_obj(rec)

            new_prepares: List[Dict[str, Any]] = []
            with timed(self.stats, "merge_prepare_build"):
                for origin in merged:
                    pr = self._append_prepare(origin, group_meta)
                    new_prepares.append(pr)
                    created_prepares += 1

            self.preprocessing_buffer = []

        with timed(self.stats, "ml_batch_total"):
            self._apply_ml_batch(new_prepares)

        t_emit0 = time.perf_counter()
        emitted_windows = self._emit_windows_if_ready()
        emit_dt = time.perf_counter() - t_emit0
        self.stats["emit_windows_total"].append(emit_dt)

        total_dt = time.perf_counter() - t0
        if total_dt > 0.02:
            log_event(
                self.logger,
                "MERGE_OUTLIER",
                merge_event_id=merge_id,
                raw_count=len(origins),
                merged_count=len(merged),
                emitted_windows=emitted_windows,
                merge_only_ms=round((self.stats["merge_only"][-1]) * 1000, 3),
                emit_windows_ms=round(emit_dt * 1000, 3),
                total_flush_ms=round(total_dt * 1000, 3),
            )

        return (created_prepares, emitted_windows)

    def _finalize_partial_window_if_needed(self) -> None:
        remain = len(self.window_buffer) - self._wb_head
        if remain <= 0:
            return
        if not self.allow_partial_window:
            print(f"⚠️ window_buffer에 {remain}개가 남았지만 (partial off) 실행/저장하지 않습니다.")
            return

        # partial on: 남은 것 전부 1개 window로 처리
        window_prepares = self.window_buffer[self._wb_head :]
        if window_prepares:
            self._emit_one_window(window_prepares)

    def run(self) -> None:
        run_t0 = time.perf_counter()
        run_started_at = datetime.now().isoformat()

        print("\n🚀 main_pipeline start")
        print(f"   - redis        : {self.redis_host}:{self.redis_port} (db={self.redis_db})")
        print(f"   - protocols    : {self.protocols}")
        print(f"   - pre_dir      : {self.pre_dir}")
        print(f"   - window       : size={self.window_size}, step={self.window_step}, partial={self.allow_partial_window}")
        print(f"   - interval     : {self.interval_sec}s  (pps={(1.0/self.interval_sec) if self.interval_sec>0 else 'inf'})")
        print(f"   - replay       : {self.replay}")
        print(f"   - stop_after   : windows={self.stop_after_windows}")
        print(f"   - server_mode  : {self.server_mode}")
        print("   - pop ordering : stream-id(ts_ms, seq) oldest-first")
        print(f"   - merge        : protocols={sorted(list(self.merge_protocols))}, bucket_ms={self.merge_bucket_ms}")
        print(f"   - Alarm        : {'ON' if self.alarm_enabled else 'OFF'}")
        print("   - Files        : data_flow.log + incoming + reassembly(before/after) + final_results.json")
        print("=" * 80)

        log_event(
            self.logger,
            "RUN",
            created_at=run_started_at,
            argv=sys.argv,
            protocols=self.protocols,
            window={"size": self.window_size, "step": self.window_step, "partial": self.allow_partial_window},
            merge={"protocols": sorted(list(self.merge_protocols)), "bucket_ms": self.merge_bucket_ms},
            alarm={"enabled": self.alarm_enabled, "base": DEFAULT_ALARM_BASE_URL, "engine": DEFAULT_ALARM_ENGINE},
            server_mode=self.server_mode,
        )
        _flush_logger(self.logger)

        last_hb = time.monotonic()
        last_empty_notice = 0.0

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

                if now - last_empty_notice >= 5.0:
                    print("⚠️ Redis에 데이터 없음... 대기 중.")
                    last_empty_notice = now
                time.sleep(max(self.interval_sec, 0.01))
                continue

            # raw json parse
            if "origin_raw" in wrapped:
                raw = wrapped.pop("origin_raw")
                proto = wrapped.get("protocol")

                with timed(self.stats, "json_parse"):
                    try:
                        data = _json_loads_fast(raw)
                    except Exception:
                        data = {"raw": raw}

                if isinstance(data, dict):
                    data["protocol"] = proto
                else:
                    data = {"value": data, "protocol": proto}

                wrapped["origin"] = data

            self.total_raw_packets += 1
            self._store_incoming_packet(wrapped)

            # replay
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

            # non-merge protocol: merge buffer flush 후 단독 처리
            if proto not in self.merge_protocols:
                if self.preprocessing_buffer:
                    self.flush_buffer()

                group_meta = self._build_group_meta([wrapped])
                pr = self._append_prepare(origin, group_meta)
                self._apply_ml_batch([pr])
                self._emit_windows_if_ready()

                if (not self.replay) and self.interval_sec > 0:
                    time.sleep(self.interval_sec)
                continue

            # merge protocol: relaxed key로 group
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
        self._finalize_partial_window_if_needed()

        run_ended_at = datetime.now().isoformat()
        elapsed = time.perf_counter() - run_t0
        self.print_statistics(run_started_at, run_ended_at, elapsed)

    def print_statistics(self, started_at: str, ended_at: str, elapsed_sec: float) -> None:
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

        sec = float(elapsed_sec)
        pps_raw = (self.total_raw_packets / sec) if sec > 0 else 0.0
        pps_prep = (self.total_prepares / sec) if sec > 0 else 0.0
        wps = (self.total_windows / sec) if sec > 0 else 0.0

        print(f"- run_started_at    : {started_at}")
        print(f"- run_ended_at      : {ended_at}")
        print(f"- total_elapsed     : {sec:.3f}s")
        print(f"- throughput        : raw={pps_raw:.2f} pkt/s, prepares={pps_prep:.2f} prep/s, windows={wps:.2f} win/s")

        print(_line_full("redis", self.stats.get("redis", [])))
        print(_line_full("json_parse", self.stats.get("json_parse", [])))
        print(_line_full("merge_only", self.stats.get("merge_only", [])))
        print(_line_full("feature", self.stats.get("feature", [])))
        print(_line_compact_with_trim("ml", self.stats.get("ml", []), self.ml_trim_pct, self.ml_skip_stats))
        print(_line_compact_with_trim("dl_window", self.stats.get("dl_window", []), self.dl_trim_pct, self.dl_skip_stats))
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

    # window
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--window-step", type=int, default=DEFAULT_WINDOW_STEP)
    parser.add_argument("--partial-window", action="store_true", default=DEFAULT_ALLOW_PARTIAL_WINDOW)

    # merge
    parser.add_argument("--merge-bucket-ms", type=int, default=DEFAULT_MERGE_BUCKET_MS)

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

    # model load
    parser.add_argument("--model-load-dir", default=str(DEFAULT_MODEL_LOAD_DIR))
    parser.add_argument("--preload-only", action="store_true", default=False, help="모델만 로드하고 종료")
    parser.add_argument("--pre-dir", default=str(DEFAULT_PRE_DIR), help="PacketFeaturePreprocessor 결과 디렉토리")
    parser.add_argument("--dl-threshold", type=float, default=0.32, help="DL-anomaly threshold 고정값(기본 0.32)")

    args = parser.parse_args()

    interval = float(args.interval)
    replay = bool(args.replay)
    if args.pps is not None:
        if args.pps <= 0:
            raise SystemExit("--pps must be > 0")
        interval = 1.0 / float(args.pps)
        replay = False

    protocols = list(args.protocols) if args.protocols else list(DEFAULT_PROTOCOLS)

    run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_root = Path(args.run_root).resolve()
    run_dir = (run_root / run_tag).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "data_flow.log"
    incoming_path = run_dir / "incoming_packets.jsonl"
    reasm_before_path = run_dir / "reassembly_before.jsonl"
    reasm_after_path = run_dir / "reassembly_after.jsonl"
    final_path = run_dir / "final_results.json"

    logger = setup_data_logger(log_path, mode="w")
    stores = RunStores(incoming_path, reasm_before_path, reasm_after_path, final_path)

    stop_after = None if args.server else int(args.stop_after_windows)

    model_load_dir = Path(args.model_load_dir).resolve()
    pre_dir = Path(args.pre_dir).resolve()

    t0 = time.perf_counter()
    models = load_and_cache_3_models(model_load_dir=model_load_dir, dl_threshold_fixed=float(args.dl_threshold))
    t1 = time.perf_counter()

    tim = models.get("_timing", {})
    print("\n=== Model preload timings ===")
    print(f"  ML        : {tim.get('ml_load_s', 0.0):.3f}s")
    print(f"  DL-anomaly: {tim.get('dl_anom_load_s', 0.0):.3f}s")
    print(f"  DL-pattern: {tim.get('dl_pat_load_s', 0.0):.3f}s")
    print(f"  TOTAL     : {tim.get('total_load_s', (t1 - t0)):.3f}s")
    print("=============================\n")

    if args.preload_only:
        print("preload-only: exit.")
        return

    pipeline = OperationalPipeLine(
        logger=logger,
        stores=stores,
        models=models,
        redis_host=args.host,
        redis_port=args.port,
        redis_db=args.db,
        redis_password=args.password,
        protocols=protocols,
        interval_sec=interval,
        replay=replay,
        stop_after_windows=stop_after,
        server_mode=bool(args.server),
        idle_sleep_sec=float(args.idle_sleep),
        # window
        window_size=int(args.window_size),
        window_step=int(args.window_step),
        allow_partial_window=bool(args.partial_window),
        # merge
        merge_protocols=set(DEFAULT_MERGE_PROTOCOLS),
        merge_bucket_ms=int(args.merge_bucket_ms),
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
        # pre_dir
        pre_dir=pre_dir,
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
예시
- 기본(윈도우 1개만 뽑고 종료):
  python main.py --stop-after-windows 100

- extract_result처럼 step=5 (80/5):
  python main.py --stop-after-windows 1 --window-size 80 --window-step 100

- 서버 모드:
  python main.py --server --window-size 80 --window-step 20
"""
