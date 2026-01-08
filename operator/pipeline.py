#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import statistics
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from io_writers import RunStores
from models import ensure_lstm_ae_tf_cache, ml_batch_predict_and_contribs
from sources import JsonlPopServer, RedisPopServer
from utils import (
    flush_logger,
    log_event,
    sanitize_and_drop_none,
    timed,
    to_int,
)

try:
    import requests  # type: ignore
except Exception:
    requests = None

import sys
from model_predict.DL_predict import predict_dl_models  # type: ignore  # noqa: E402

_PIPELINE_DIR = Path(__file__).resolve().parent
_PREPROCESSING_CODE_DIR = (_PIPELINE_DIR / "../ML/code").resolve()
if str(_PREPROCESSING_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_PREPROCESSING_CODE_DIR))

_PFP_IMPORT_ERR = None
try:
    from preprocessing.preprocessing import (
        BASE_FEATURE_COLUMNS,
        MISSING_DEFAULT,
        _get,
        _load_yaml,
        _resolve_path,
        build_arp_features,
        build_common_features,
        build_dns_features,
        build_modbus_features,
        build_slot_config,
        build_s7comm_features,
        build_xgt_fen_features,
        extract_one_packet,
        load_json_or_default,
        protocol_to_code,
        safe_float,
        match_to_float,
    )
except Exception as e:
    _PFP_IMPORT_ERR = str(e)
    BASE_FEATURE_COLUMNS = []
    MISSING_DEFAULT = -1.0
    _get = lambda *a, **kw: None
    _load_yaml = lambda *a, **kw: {}
    _resolve_path = lambda *a, **kw: None
    build_arp_features = lambda *a, **kw: {}
    build_common_features = lambda *a, **kw: {}
    build_dns_features = lambda *a, **kw: {}
    build_modbus_features = lambda *a, **kw: {}
    build_slot_config = lambda *a, **kw: None
    build_s7comm_features = lambda *a, **kw: {}
    build_xgt_fen_features = lambda *a, **kw: {}
    extract_one_packet = lambda *a, **kw: {}
    load_json_or_default = lambda *a, **kw: {}
    protocol_to_code = lambda *a, **kw: 0
    safe_float = lambda *a, **kw: -1.0
    match_to_float = lambda *a, **kw: -1.0

PacketFeaturePreprocessor = None 

PROTOCOL_FLATTEN_KEYS = {"modbus", "s7comm", "xgt_fen", "xgt-fen", "tcp", "udp", "dns", "arp"}


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


def origin_to_dotted(origin: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(origin, dict):
        return {}

    out: Dict[str, Any] = {}
    for k, v in origin.items():
        k = str(k)
        if k in PROTOCOL_FLATTEN_KEYS and isinstance(v, dict):
            for subk, subv in v.items():
                out[f"{k}.{subk}"] = _coerce_scalar(subv)
            continue
        out[k] = _coerce_scalar(v)

    for nk in ("len", "sp", "dp"):
        if nk in out:
            iv = to_int(out.get(nk))
            if iv is not None:
                out[nk] = iv

    out.setdefault("sip", None)
    out.setdefault("dip", None)
    out.setdefault("src_asset", None)
    out.setdefault("dst_asset", None)
    return out


def build_window_raw_entry(prepare: Dict[str, Any]) -> Dict[str, Any]:
    origin = prepare.get("origin", {}) or {}
    pkt = origin_to_dotted(origin)
    pkt.pop("ml", None)
    return sanitize_and_drop_none(pkt)


def normalize_protocol_for_features(proto: str) -> str:
    return "xgt_fen" if proto == "xgt-fen" else proto


def merge_packets_by_modbus_diff(origins: List[Dict[str, Any]], merge_protocols: set[str]) -> List[Dict[str, Any]]:
    if not origins:
        return []
    proto = origins[0].get("protocol", "")
    if proto not in merge_protocols or len(origins) == 1:
        return [origins[0]]

    base_packet = dict(origins[0])

    if proto in ("xgt_fen", "xgt-fen"):
        prefixes = ("xgt_fen.", "xgt-fen.")
    else:
        prefixes = (proto + ".",)

    key_values: Dict[str, List[Any]] = {}
    for p in origins:
        for k, v in p.items():
            if v is None:
                continue
            for pref in prefixes:
                if str(k).startswith(pref):
                    key_values.setdefault(str(k), []).append(v)
                    break

    for key, vals in key_values.items():
        if not vals:
            continue
        first_val = vals[0]
        base_packet[key] = first_val if all(v == first_val for v in vals[1:]) else (vals[0] if len(vals) == 1 else vals)

    return [base_packet]


def parse_dl_output(dl_out: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    pattern = dl_out.get("pattern")
    summary = dl_out.get("summary")

    if pattern is None and isinstance(dl_out.get("dl_pattern"), dict):
        pattern = dl_out["dl_pattern"].get("pattern")

    if summary is None:
        da = dl_out.get("dl_anomaly")
        if isinstance(da, dict) and isinstance(da.get("summary"), dict):
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


class AlarmSender:
    __slots__ = ("base_url", "engine", "timeout", "logger", "q", "stop_event", "thread", "data_flow_max_json_chars")

    def __init__(
        self,
        base_url: str,
        engine: str,
        timeout: float,
        logger,
        *,
        data_flow_max_json_chars: int,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.engine = str(engine).strip("/") or "dl"
        self.timeout = float(timeout)
        self.logger = logger
        self.data_flow_max_json_chars = int(data_flow_max_json_chars)

        import queue
        self.q: "queue.Queue[dict]" = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        import queue
        while not self.stop_event.is_set() or not self.q.empty():
            try:
                payload = self.q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if requests is None:
                    log_event(self.logger, "ALARM", data_flow_max_json_chars=self.data_flow_max_json_chars,
                              status="skip", reason="requests_not_available")
                    continue

                url = f"{self.base_url}/api/alarms/{self.engine}"
                resp = requests.post(url, json=payload, timeout=self.timeout)
                code = getattr(resp, "status_code", None)
                if hasattr(resp, "raise_for_status"):
                    resp.raise_for_status()
                log_event(self.logger, "ALARM", data_flow_max_json_chars=self.data_flow_max_json_chars,
                          status="sent", http_status=code, url=url)
            except Exception as e:
                log_event(self.logger, "ALARM", data_flow_max_json_chars=self.data_flow_max_json_chars,
                          status="fail", error=str(e))
            finally:
                self.q.task_done()

    def send_risk(self, risk: Dict[str, Any], *, extra: Optional[Dict[str, Any]] = None) -> None:
        if not isinstance(risk, dict):
            return
        payload = {"risk": risk}
        if extra:
            payload.update(extra)
        try:
            payload["risk"].setdefault("detected_time", datetime.utcnow().isoformat(timespec="seconds") + "Z")
        except Exception:
            pass
        self.q.put(payload)

    def close(self) -> None:
        self.stop_event.set()
        self.q.join()
        self.thread.join()


class OperationalPipeline:
    def __init__(
        self,
        *,
        logger,
        stores: RunStores,
        models: Dict[str, Any],
        cfg: Dict[str, Any],
        initial_alert_seq_id: int,
    ):
        self.logger = logger
        self.stores = stores
        self.models = models
        if isinstance(self.models.get("dl_anomaly"), dict):
            ensure_lstm_ae_tf_cache(self.models["dl_anomaly"])

        self.cfg = cfg
        self.stats: Dict[str, List[float]] = defaultdict(list)

        tuning = cfg["tuning"]
        self.log_fsync = bool(tuning.get("log_fsync", False))
        self.data_flow_max_json_chars = int(tuning.get("data_flow_max_json_chars", 4000))
        self.heartbeat_sec = float(tuning.get("heartbeat_sec", 30.0))
        self.max_raw_packets = int(tuning.get("max_raw_packets", 10_000_000))
        self.max_prepares = int(tuning.get("max_prepares", 10_000_000))

        run = cfg["run"]
        self.server_mode = bool(run.get("server", False))
        self.idle_sleep_sec = float(run.get("idle_sleep_sec", 0.01))
        self.interval_sec = float(run.get("interval_sec", 0.0))
        self.replay = bool(run.get("replay", False))

        self.stop_after_windows = None if self.server_mode else int(run.get("stop_after_windows", 1))

        window = cfg["window"]
        self.window_size = int(window.get("size", 80))
        self.window_step = int(window.get("step", 20))
        self.allow_partial_window = bool(window.get("allow_partial", False))

        if self.window_size <= 0 or self.window_step <= 0:
            raise ValueError("window.size/window.step must be > 0")
        if self.window_step > self.window_size:
            raise ValueError("window.step must be <= window.size")

        merge = cfg["merge"]
        self.merge_protocols = set(merge.get("protocols", []))
        self.merge_bucket_ms = int(merge.get("bucket_ms", 3))

        ml = cfg["ml"]
        self.ml_topk = int(ml.get("topk", 2))
        self.ml_warmup = int(ml.get("warmup", 0))
        self.ml_skip_stats = int(ml.get("skip_stats", 0))
        self.ml_trim_pct = float(ml.get("trim_pct", 0.0))
        self.ml_hash_fallback = bool(ml.get("hash_fallback", True))
        self._ml_call_count = 0

        dl = cfg["dl"]
        self.dl_warmup = int(dl.get("warmup", 0))
        self.dl_skip_stats = int(dl.get("skip_stats", 0))
        self.dl_trim_pct = float(dl.get("trim_pct", 0.0))
        self._dl_call_count = 0

        alarm = cfg["alarm"]
        self.alarm_enabled = bool(alarm.get("enabled", False)) and (requests is not None)
        self.alarm_sender: Optional[AlarmSender] = None
        if self.alarm_enabled:
            self.alarm_sender = AlarmSender(
                base_url=str(alarm.get("base_url")),
                engine=str(alarm.get("engine", "dl")),
                timeout=float(alarm.get("timeout", 3.0)),
                logger=self.logger,
                data_flow_max_json_chars=self.data_flow_max_json_chars,
            )

        self.seq_id = 0
        self.alert_seq_id = int(max(0, initial_alert_seq_id))
        self.total_raw_packets = 0
        self.total_prepares = 0
        self.total_windows = 0

        self.preprocessing_buffer: List[Dict[str, Any]] = []
        self.window_buffer: List[Dict[str, Any]] = []
        self._wb_head = 0

        self.merge_event_id = 0
        self.ingest_id = 0
        self.last_packet_ts_ms: Optional[int] = None

        self._stop_flag = threading.Event()

        # source server
        source = cfg["source"]
        mode = str(source.get("mode", "file"))
        if mode == "file":
            input_jsonl = source.get("input_jsonl")
            if not input_jsonl:
                raise ValueError("source.input_jsonl is required for file mode.")
            self.server = JsonlPopServer(Path(input_jsonl).resolve())
            self.source_mode = "file"
        else:
            r = cfg["redis"]
            protocols = list(cfg.get("protocols") or [])
            if "xgt_fen" in protocols and "xgt-fen" not in protocols:
                protocols.append("xgt-fen")
            if "xgt-fen" in protocols and "xgt_fen" not in protocols:
                protocols.append("xgt_fen")
            self.server = RedisPopServer(
                host=str(r.get("host", "localhost")),
                port=int(r.get("port", 6379)),
                db=int(r.get("db", 0)),
                password=r.get("password", None),
                protocols=protocols,
            )
            self.source_mode = "redis"

        # featurizer
        self.pre_dir = Path(cfg["paths"]["pre_dir"]).resolve()
        self.featurizer = self._init_featurizer(self.pre_dir)

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
            if self.source_mode == "file" and hasattr(self.server, "close"):
                self.server.close()
        except Exception:
            pass

    def _init_featurizer(self, pre_dir: Path):
        if _PFP_IMPORT_ERR is not None:
            print(f"⚠️ Preprocessing functions import 실패: {_PFP_IMPORT_ERR}")
            print("⚠️ 피처 전처리 없이 features={}로만 진행됩니다.")
            return None
        if not pre_dir.exists():
            print(f"⚠️ pre_dir 없음: {pre_dir}")
            print("⚠️ 피처 전처리 없이 features={}로만 진행됩니다.")
            return None
        try:
            common_host_map = load_json_or_default(pre_dir / "common_host_map.json")
            common_norm = load_json_or_default(pre_dir / "common_norm_params.json")
            s7_norm = load_json_or_default(pre_dir / "s7comm_norm_params.json")
            modbus_norm = load_json_or_default(pre_dir / "modbus_norm_params.json")
            xgt_var_vocab = load_json_or_default(pre_dir / "xgt_var_vocab.json")
            xgt_norm = load_json_or_default(pre_dir / "xgt_norm_params.json")
            arp_host_map = load_json_or_default(pre_dir / "arp_host_map.json")
            dns_norm = load_json_or_default(pre_dir / "dns_norm_params.json")

            modbus_slot_vocab = load_json_or_default(pre_dir / "modbus_addr_slot_vocab.json")
            modbus_slot_norm = load_json_or_default(pre_dir / "modbus_addr_slot_norm_params.json")
            xgt_slot_vocab = load_json_or_default(pre_dir / "xgt_addr_slot_vocab.json")
            xgt_slot_norm = load_json_or_default(pre_dir / "xgt_addr_slot_norm_params.json")

            feature_columns = list(BASE_FEATURE_COLUMNS)
            modbus_slot_config = build_slot_config(modbus_slot_vocab, modbus_slot_norm, "modbus", feature_columns)
            xgt_slot_config = build_slot_config(xgt_slot_vocab, xgt_slot_norm, "xgt", feature_columns)

            base_feat_template = {k: float(MISSING_DEFAULT) for k in feature_columns}

            featurizer_bundle = {
                "feature_columns": feature_columns,
                "base_feat_template": base_feat_template,
                "common_host_map": common_host_map,
                "common_norm": common_norm,
                "s7_norm": s7_norm,
                "modbus_norm": modbus_norm,
                "xgt_var_vocab": xgt_var_vocab,
                "xgt_norm": xgt_norm,
                "arp_host_map": arp_host_map,
                "dns_norm": dns_norm,
                "modbus_slot_config": modbus_slot_config,
                "xgt_slot_config": xgt_slot_config,
                "extract_one_packet": extract_one_packet,
            }
            print(f"✓ Preprocessing functions 로드 완료: {pre_dir}")
            return featurizer_bundle
        except Exception as e:
            print(f"⚠️ Preprocessing init 실패: {e}")
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

    def _call_featurizer(self, origin: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        if self.featurizer is None:
            return {}
        
        extract_func = self.featurizer.get("extract_one_packet")
        if extract_func is None:
            return {"features_error": "extract_one_packet function not found in featurizer"}

        try:
            features = extract_func(
                pkt=origin,
                feature_columns=self.featurizer["feature_columns"],
                base_feat_template=self.featurizer["base_feat_template"],
                common_host_map=self.featurizer["common_host_map"],
                common_norm=self.featurizer["common_norm"],
                s7_norm=self.featurizer["s7_norm"],
                modbus_norm=self.featurizer["modbus_norm"],
                xgt_var_vocab=self.featurizer["xgt_var_vocab"],
                xgt_norm=self.featurizer["xgt_norm"],
                arp_host_map=self.featurizer["arp_host_map"],
                dns_norm=self.featurizer["dns_norm"],
                modbus_slot_config=self.featurizer["modbus_slot_config"],
                xgt_slot_config=self.featurizer["xgt_slot_config"],
            )
            return {"features": features}
        except Exception as e:
            return {"features_error": str(e)}

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
            bucket = int(pkt_ts_ms) // int(max(1, self.merge_bucket_ms))
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
                    igi = int(ig)
                    ingest_min = igi if ingest_min is None else min(ingest_min, igi)
                    ingest_max = igi if ingest_max is None else max(ingest_max, igi)
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

    def _append_prepare(self, origin: Dict[str, Any], group_meta: Dict[str, Any]) -> Dict[str, Any]:
        features: Dict[str, Any] = {}
        if self.featurizer is not None:
            with timed(self.stats, "feature"):
                try:
                    features = self._call_featurizer(origin, group_meta)
                except Exception as e:
                    features = {"features_error": str(e)}

        prepare: Dict[str, Any] = {"origin": origin, "features": features, "_meta": group_meta}
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

            origin = pr.get("origin")
            if isinstance(origin, dict):
                origin["ml_anomaly_prob"] = pr["ml_anomaly_prob"]
                origin["match"] = pr["match"]

    def _emit_windows_if_ready(self) -> int:
        emitted = 0
        while (len(self.window_buffer) - self._wb_head) >= self.window_size:
            s = self._wb_head
            e = s + self.window_size
            window_prepares = self.window_buffer[s:e]
            self._wb_head += self.window_step
            self._emit_one_window(window_prepares)
            emitted += 1

            if self._wb_head > 4096 and self._wb_head > (len(self.window_buffer) // 2):
                self.window_buffer = self.window_buffer[self._wb_head:]
                self._wb_head = 0
        return emitted

    def _emit_one_window(self, window_prepares: List[Dict[str, Any]]) -> None:
        self.seq_id += 1

        t0 = time.perf_counter()
        dl_out = predict_dl_models(prepares=window_prepares, models=self.models, seq_id=self.seq_id) or {}
        dt = time.perf_counter() - t0

        self._dl_call_count += 1
        if self._dl_call_count > self.dl_skip_stats:
            self.stats["dl_window"].append(dt)

        pattern, summary = parse_dl_output(dl_out)
        base_origin = (window_prepares[-1].get("origin") or {}) if window_prepares else {}
        summary = normalize_risk_fields(summary, base_origin)

        alert_flag = derive_alert(summary)
        if isinstance(dl_out, dict) and "alert" in dl_out:
            try:
                alert_flag = bool(dl_out.get("alert")) or alert_flag
            except Exception:
                pass

        window_raw = [build_window_raw_entry(pr) for pr in window_prepares]
        data_obj = sanitize_and_drop_none(
            {"seq_id": int(self.seq_id), "pattern": str(pattern), "summary": summary, "window_raw": window_raw}
        )

        self.stores.final.write_obj(data_obj)

        if alert_flag:
            self.alert_seq_id += 1
            alert_obj = dict(data_obj)
            alert_obj["seq_id"] = int(self.alert_seq_id)
            self.stores.dl_out.write_obj(alert_obj)

        # log + alarm
        try:
            rs = float((summary.get("risk") or {}).get("score", 0.0))
        except Exception:
            rs = 0.0

        log_event(
            self.logger,
            "FINAL",
            data_flow_max_json_chars=self.data_flow_max_json_chars,
            seq_id=int(self.seq_id),
            pattern=str(pattern),
            alert=("o" if alert_flag else "x"),
            risk_score=rs,
            anomaly_score=summary.get("anomaly_score"),
            threshold=summary.get("threshold"),
            dl_ms=round(dt * 1000.0, 3),
        )
        flush_logger(self.logger, fsync=self.log_fsync)

        if alert_flag and self.alarm_enabled and self.alarm_sender is not None:
            risk = (summary.get("risk") or {}) if isinstance(summary, dict) else {}
            self.alarm_sender.send_risk(
                risk if isinstance(risk, dict) else {},
                extra={"seq_id": int(self.seq_id), "pattern": pattern, "engine": "dl"},
            )

        self.total_windows += 1

    def flush_buffer(self) -> Tuple[int, int]:
        if not self.preprocessing_buffer:
            return (0, 0)

        created_prepares = 0
        t0 = time.perf_counter()

        with timed(self.stats, "merge_only"):
            self.merge_event_id += 1
            merge_id = int(self.merge_event_id)

            wrappers = self.preprocessing_buffer
            origins = [w.get("origin", {}) for w in wrappers]
            group_meta = self._build_group_meta(wrappers)
            group_meta["merge_event_id"] = merge_id

            # before
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

            merged = merge_packets_by_modbus_diff(origins, self.merge_protocols)

            # after
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
            for origin in merged:
                pr = self._append_prepare(origin, group_meta)
                new_prepares.append(pr)
                created_prepares += 1

            self.preprocessing_buffer = []

        self._apply_ml_batch(new_prepares)
        emitted_windows = self._emit_windows_if_ready()

        total_dt = time.perf_counter() - t0
        if total_dt > 0.02:
            log_event(
                self.logger,
                "MERGE_OUTLIER",
                data_flow_max_json_chars=self.data_flow_max_json_chars,
                merge_event_id=merge_id,
                raw_count=len(origins),
                merged_count=len(merged),
                emitted_windows=emitted_windows,
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
        window_prepares = self.window_buffer[self._wb_head:]
        if window_prepares:
            self._emit_one_window(window_prepares)

    def run(self) -> None:
        run_t0 = time.perf_counter()
        run_started_at = datetime.now().isoformat()

        log_event(
            self.logger,
            "RUN",
            data_flow_max_json_chars=self.data_flow_max_json_chars,
            created_at=run_started_at,
            window={"size": self.window_size, "step": self.window_step, "partial": self.allow_partial_window},
            merge={"protocols": sorted(list(self.merge_protocols)), "bucket_ms": self.merge_bucket_ms},
            alarm=self.cfg.get("alarm", {}),
            server_mode=self.server_mode,
        )
        flush_logger(self.logger, fsync=self.log_fsync)

        last_hb = time.monotonic()
        last_empty_notice = 0.0

        while True:
            if self._stop_flag.is_set():
                print("\n🛑 stop requested (signal).")
                break

            if self.stop_after_windows is not None and self.total_windows >= int(self.stop_after_windows):
                print(f"\n✅ stop_after_windows={self.stop_after_windows} 만족하여 종료합니다.")
                break

            if self.total_raw_packets >= self.max_raw_packets:
                self.flush_buffer()
                break
            if self.total_prepares >= self.max_prepares:
                self.flush_buffer()
                break

            with timed(self.stats, "pop"):
                wrapped = self.server.pop_oldest()

            if not wrapped:
                self.flush_buffer()

                if self.source_mode == "file":
                    break

                now = time.monotonic()
                if (now - last_hb) >= self.heartbeat_sec:
                    log_event(
                        self.logger,
                        "HEARTBEAT",
                        data_flow_max_json_chars=self.data_flow_max_json_chars,
                        windows=self.total_windows,
                        raw_packets=self.total_raw_packets,
                        prepares=self.total_prepares,
                    )
                    flush_logger(self.logger, fsync=self.log_fsync)
                    last_hb = now

                if self.server_mode:
                    time.sleep(max(self.idle_sleep_sec, 0.001))
                    continue

                if now - last_empty_notice >= 5.0:
                    print("⚠️ Redis에 데이터 없음... 대기 중.")
                    last_empty_notice = now
                time.sleep(max(self.interval_sec, 0.01))
                continue

            self.total_raw_packets += 1

            # store incoming
            self._store_incoming_packet(wrapped)

            origin = wrapped.get("origin", {}) or {}
            proto = origin.get("protocol", "")

            # non-merge protocol: flush then single
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

        print(f"- total_raw_packets : {self.total_raw_packets}")
        print(f"- total_prepares    : {self.total_prepares}")
        print(f"- total_windows     : {self.total_windows}")
        print(f"- run_started_at    : {started_at}")
        print(f"- run_ended_at      : {ended_at}")
        print(f"- total_elapsed     : {elapsed_sec:.3f}s")

        print(_line_full("pop", self.stats.get("pop", [])))
        print(_line_full("merge_only", self.stats.get("merge_only", [])))
        print(_line_full("feature", self.stats.get("feature", [])))
        print(_line_full("ml", self.stats.get("ml", [])))
        print(_line_full("dl_window", self.stats.get("dl_window", [])))
        print("=" * 80)
