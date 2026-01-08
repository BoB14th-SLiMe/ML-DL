#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import apply_cli_overrides, load_yaml_config
from io_writers import RunStores, count_jsonl_records
from models import load_and_cache_3_models
from pipeline import OperationalPipeline
from utils import setup_data_logger


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="main_pipeline (run with: python main.py)")

    p.add_argument("--config", default=None, help="YAML config path (optional). Default: <script_dir>/main.yaml")

    p.add_argument("--input-jsonl", default=None, help="override: source.input_jsonl")
    p.add_argument("--dl-output-path", default=None, help="override: paths.dl_output_path")

    p.add_argument("--protocols", nargs="+", default=None)

    speed = p.add_mutually_exclusive_group()
    speed.add_argument("--interval", type=float, default=None)
    speed.add_argument("--pps", type=int, default=None)
    speed.add_argument("--replay", action="store_true", default=None)

    p.add_argument("--stop-after-windows", type=int, default=None)
    p.add_argument("--server", action="store_true", default=None)
    p.add_argument("--idle-sleep", type=float, default=None)

    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--window-step", type=int, default=None)
    p.add_argument("--partial-window", action="store_true", default=None)

    p.add_argument("--merge-bucket-ms", type=int, default=None)

    p.add_argument("--ml-topk", type=int, default=None)
    p.add_argument("--ml-warmup", type=int, default=None)
    p.add_argument("--ml-skip-stats", type=int, default=None)
    p.add_argument("--ml-trim-pct", type=float, default=None)
    p.add_argument("--ml-no-hash-fallback", action="store_true", default=False)

    p.add_argument("--dl-warmup", type=int, default=None)
    p.add_argument("--dl-skip-stats", type=int, default=None)
    p.add_argument("--dl-trim-pct", type=float, default=None)
    p.add_argument("--dl-threshold", type=float, default=None)

    p.add_argument("--alarm", action="store_true", default=None)
    p.add_argument("--alarm-base-url", default=None)
    p.add_argument("--alarm-engine", default=None)
    p.add_argument("--alarm-timeout", type=float, default=None)

    p.add_argument("--run-root", default=None)
    p.add_argument("--model-load-dir", default=None)
    p.add_argument("--pre-dir", default=None)

    p.add_argument("--preload-only", action="store_true", default=False)

    return p


def _resolve_cfg_path(arg: Optional[str]) -> Path:
    if arg is None or str(arg).strip() == "":
        return (SCRIPT_DIR / "main.yaml").resolve()

    p = Path(arg).expanduser()
    if p.is_absolute():
        return p.resolve()

    cand = (SCRIPT_DIR / p).resolve()
    if cand.exists():
        return cand
    return (Path.cwd() / p).resolve()


def _resolve_relpath(base_dir: Path, v: Any) -> Any:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return v
    p = Path(s).expanduser()
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _apply_path_base(cfg: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    cfg = dict(cfg)

    run = cfg.get("run") or {}
    paths = cfg.get("paths") or {}
    source = cfg.get("source") or {}

    if isinstance(run, dict) and "run_root" in run:
        run["run_root"] = _resolve_relpath(base_dir, run.get("run_root"))
    cfg["run"] = run

    if isinstance(source, dict) and "input_jsonl" in source:
        source["input_jsonl"] = _resolve_relpath(base_dir, source.get("input_jsonl"))
    cfg["source"] = source

    if isinstance(paths, dict):
        if "pre_dir" in paths:
            paths["pre_dir"] = _resolve_relpath(base_dir, paths.get("pre_dir"))
        if "model_load_dir" in paths:
            paths["model_load_dir"] = _resolve_relpath(base_dir, paths.get("model_load_dir"))
        if "dl_output_path" in paths:
            paths["dl_output_path"] = _resolve_relpath(base_dir, paths.get("dl_output_path"))
    cfg["paths"] = paths

    return cfg


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg_path = _resolve_cfg_path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}  (put main.yaml next to main.py)")

    cfg = load_yaml_config(str(cfg_path))
    cfg = _apply_path_base(cfg, cfg_path.parent)
    cfg = apply_cli_overrides(cfg, args)

    run = cfg["run"]

    interval = float(run.get("interval_sec", 0.0) or 0.0)
    pps = run.get("pps", None)
    replay = bool(run.get("replay", False))

    if pps is not None:
        pps = int(pps)
        if pps <= 0:
            raise SystemExit("--pps must be > 0")
        interval = 1.0 / float(pps)
        replay = False

    cfg["run"]["interval_sec"] = interval
    cfg["run"]["replay"] = replay

    run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_root = Path(cfg["run"]["run_root"]).expanduser().resolve()
    run_dir = (run_root / run_tag).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "data_flow.log"
    incoming_path = run_dir / "incoming_packets.jsonl"
    reasm_before_path = run_dir / "reassembly_before.jsonl"
    reasm_after_path = run_dir / "reassembly_after.jsonl"
    final_path = run_dir / "final_results.json"

    logger = setup_data_logger(log_path, mode="w")

    dl_output_path = Path(cfg["paths"]["dl_output_path"]).expanduser().resolve()
    existing_alert_count = count_jsonl_records(dl_output_path)
    print(f"✓ DL output existing records: {existing_alert_count} -> next alert seq_id={existing_alert_count + 1}")

    tuning = cfg.get("tuning") or {}
    stores = RunStores(
        incoming_path,
        reasm_before_path,
        reasm_after_path,
        final_path,
        dl_output_path,
        async_batch=int(tuning.get("async_batch", 300)),
        async_flush_sec=float(tuning.get("async_flush_sec", 0.7)),
    )

    model_load_dir = Path(cfg["paths"]["model_load_dir"]).expanduser().resolve()
    dl_threshold = float((cfg.get("dl") or {}).get("threshold", 0.32))

    t0 = time.perf_counter()
    models = load_and_cache_3_models(model_load_dir=model_load_dir, dl_threshold_fixed=dl_threshold)
    t1 = time.perf_counter()

    tim = models.get("_timing", {})
    print("\n=== Model preload timings ===")
    print(f"  ML        : {tim.get('ml_load_s', 0.0):.3f}s")
    print(f"  DL-anomaly: {tim.get('dl_anom_load_s', 0.0):.3f}s")
    print(f"  DL-pattern: {tim.get('dl_pat_load_s', 0.0):.3f}s")
    print(f"  TOTAL     : {tim.get('total_load_s', (t1 - t0)):.3f}s")
    print("=============================\n")

    if args.preload_only:
        cfg["run"]["preload_only"] = True

    if bool(cfg["run"].get("preload_only", False)):
        print("preload_only: exit.")
        stores.close()
        return

    pipeline = OperationalPipeline(
        logger=logger,
        stores=stores,
        models=models,
        cfg=cfg,
        initial_alert_seq_id=int(existing_alert_count),
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
