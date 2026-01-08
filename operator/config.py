#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

JsonDict = Dict[str, Any]


DEFAULT_CONFIG: JsonDict = {
    "source": {"mode": "file", "input_jsonl": None},
    "redis": {"host": "localhost", "port": 6379, "db": 0, "password": None},
    "protocols": ["modbus", "s7comm", "xgt_fen", "tcp", "udp", "dns", "arp"],
    "run": {
        "run_root": "./final_results",
        "stop_after_windows": 1,
        "server": False,
        "idle_sleep_sec": 0.01,
        "interval_sec": 0.0,
        "pps": None,
        "replay": False,
        "preload_only": False,
    },
    "paths": {
        "pre_dir": "../preprocessing/result",
        "model_load_dir": "./model_load",
        "dl_output_path": "/home/slime/SLM/DL/output/dl_anomaly_detect.jsonl",
    },
    "window": {"size": 80, "step": 20, "allow_partial": False},
    "merge": {"protocols": ["modbus", "xgt_fen", "xgt-fen"], "bucket_ms": 3},
    "ml": {"topk": 2, "warmup": 5, "skip_stats": 0, "trim_pct": 0.0, "hash_fallback": True},
    "dl": {"warmup": 3, "skip_stats": 0, "trim_pct": 0.0, "threshold": 0.32},
    "alarm": {"enabled": False, "base_url": "http://192.168.4.140:8080", "engine": "dl", "timeout": 3.0},
    "tuning": {
        "async_batch": 300,
        "async_flush_sec": 0.7,
        "log_fsync": False,
        "data_flow_max_json_chars": 4000,
        "heartbeat_sec": 30.0,
        "max_raw_packets": 10_000_000,
        "max_prepares": 10_000_000,
    },
}


def _deep_merge(dst: JsonDict, src: JsonDict) -> JsonDict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def load_yaml_config(path: Optional[str | Path]) -> JsonDict:
    cfg = _deep_merge(dict(DEFAULT_CONFIG), {})
    if not path:
        return cfg

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))

    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required. Install: pip install pyyaml") from e

    with p.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    if not isinstance(y, dict):
        raise ValueError("YAML root must be a mapping/dict.")

    return _deep_merge(cfg, y)


def apply_cli_overrides(cfg: JsonDict, args) -> JsonDict:
    # config path already applied outside
    if getattr(args, "input_jsonl", None):
        cfg["source"]["input_jsonl"] = str(args.input_jsonl)
        cfg["source"]["mode"] = "file"

    if getattr(args, "dl_output_path", None):
        cfg["paths"]["dl_output_path"] = str(args.dl_output_path)

    if getattr(args, "run_root", None):
        cfg["run"]["run_root"] = str(args.run_root)

    if getattr(args, "model_load_dir", None):
        cfg["paths"]["model_load_dir"] = str(args.model_load_dir)

    if getattr(args, "pre_dir", None):
        cfg["paths"]["pre_dir"] = str(args.pre_dir)

    if getattr(args, "protocols", None):
        cfg["protocols"] = list(args.protocols)

    if getattr(args, "interval", None) is not None:
        cfg["run"]["interval_sec"] = float(args.interval)

    if getattr(args, "pps", None) is not None:
        cfg["run"]["pps"] = int(args.pps)

    if getattr(args, "replay", None) is not None:
        cfg["run"]["replay"] = bool(args.replay)

    if getattr(args, "stop_after_windows", None) is not None:
        cfg["run"]["stop_after_windows"] = int(args.stop_after_windows)

    if getattr(args, "server", None) is not None:
        cfg["run"]["server"] = bool(args.server)

    if getattr(args, "idle_sleep", None) is not None:
        cfg["run"]["idle_sleep_sec"] = float(args.idle_sleep)

    if getattr(args, "window_size", None) is not None:
        cfg["window"]["size"] = int(args.window_size)

    if getattr(args, "window_step", None) is not None:
        cfg["window"]["step"] = int(args.window_step)

    if getattr(args, "partial_window", None) is not None:
        cfg["window"]["allow_partial"] = bool(args.partial_window)

    if getattr(args, "merge_bucket_ms", None) is not None:
        cfg["merge"]["bucket_ms"] = int(args.merge_bucket_ms)

    if getattr(args, "ml_topk", None) is not None:
        cfg["ml"]["topk"] = int(args.ml_topk)

    if getattr(args, "ml_warmup", None) is not None:
        cfg["ml"]["warmup"] = int(args.ml_warmup)

    if getattr(args, "ml_skip_stats", None) is not None:
        cfg["ml"]["skip_stats"] = int(args.ml_skip_stats)

    if getattr(args, "ml_trim_pct", None) is not None:
        cfg["ml"]["trim_pct"] = float(args.ml_trim_pct)

    if getattr(args, "ml_no_hash_fallback", None):
        cfg["ml"]["hash_fallback"] = False

    if getattr(args, "dl_warmup", None) is not None:
        cfg["dl"]["warmup"] = int(args.dl_warmup)

    if getattr(args, "dl_skip_stats", None) is not None:
        cfg["dl"]["skip_stats"] = int(args.dl_skip_stats)

    if getattr(args, "dl_trim_pct", None) is not None:
        cfg["dl"]["trim_pct"] = float(args.dl_trim_pct)

    if getattr(args, "dl_threshold", None) is not None:
        cfg["dl"]["threshold"] = float(args.dl_threshold)

    if getattr(args, "alarm", None) is not None:
        cfg["alarm"]["enabled"] = bool(args.alarm)

    if getattr(args, "alarm_base_url", None):
        cfg["alarm"]["base_url"] = str(args.alarm_base_url)

    if getattr(args, "alarm_engine", None):
        cfg["alarm"]["engine"] = str(args.alarm_engine)

    if getattr(args, "alarm_timeout", None) is not None:
        cfg["alarm"]["timeout"] = float(args.alarm_timeout)

    if getattr(args, "preload_only", None):
        cfg["run"]["preload_only"] = True

    return cfg
