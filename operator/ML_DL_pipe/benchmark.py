#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark.py

모델을 미리 로드하고, 파이프라인을 실행하여
ml, dl, redis, 재조립, total의 mean, min, max 성능을 측정합니다.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ============================================================
# 경로 설정
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from main import (
    load_and_cache_3_models,
    OperationalPipeLine,
    RunStores,
    setup_data_logger,
    DEFAULT_MODEL_LOAD_DIR,
    DEFAULT_RUN_ROOT,
    DEFAULT_REDIS_HOST,
    DEFAULT_REDIS_PORT,
    DEFAULT_REDIS_DB,
    DEFAULT_REDIS_PASSWORD,
    DEFAULT_PROTOCOLS,
    WINDOW_SIZE,
    WINDOW_STEP,
)


def format_stats(name: str, times_sec: List[float], width: int = 16) -> str:
    """mean, min, max 형식으로 통계 출력"""
    if not times_sec:
        return f"- {name:<{width}}: (no data)"

    ms = [t * 1000 for t in times_sec]
    mean_val = statistics.mean(ms)
    min_val = min(ms)
    max_val = max(ms)

    return (
        f"- {name:<{width}}: mean={mean_val:>8.3f}ms  "
        f"min={min_val:>8.3f}ms  max={max_val:>8.3f}ms  n={len(ms)}"
    )


def format_stats_detailed(name: str, times_sec: List[float], width: int = 16) -> str:
    """mean, min, max, p50, p95, p99 형식으로 상세 통계 출력"""
    if not times_sec:
        return f"- {name:<{width}}: (no data)"

    ms = [t * 1000 for t in times_sec]
    mean_val = statistics.mean(ms)
    min_val = min(ms)
    max_val = max(ms)
    p50 = float(np.percentile(ms, 50))
    p95 = float(np.percentile(ms, 95))
    p99 = float(np.percentile(ms, 99))

    return (
        f"- {name:<{width}}: mean={mean_val:>8.3f}ms  "
        f"min={min_val:>8.3f}ms  max={max_val:>8.3f}ms  "
        f"p50={p50:>8.3f}ms  p95={p95:>8.3f}ms  p99={p99:>8.3f}ms  n={len(ms)}"
    )


def run_benchmark(
    model_load_dir: Path,
    stop_after_windows: int = 10,
    redis_host: str = DEFAULT_REDIS_HOST,
    redis_port: int = DEFAULT_REDIS_PORT,
    redis_db: int = DEFAULT_REDIS_DB,
    redis_password: Optional[str] = DEFAULT_REDIS_PASSWORD,
    protocols: Optional[List[str]] = None,
    detailed: bool = False,
) -> Dict[str, Any]:
    """벤치마크 실행"""

    print("=" * 80)
    print("🔧 Model Pre-loading Phase")
    print("=" * 80)

    # 1. 모델 미리 로드
    t_model_start = time.perf_counter()
    models = load_and_cache_3_models(model_load_dir=model_load_dir)
    t_model_end = time.perf_counter()

    model_load_time = t_model_end - t_model_start
    tim = models.get("_timing", {})

    print(f"\n✅ Model loading completed in {model_load_time:.3f}s")
    print(f"   - ML          : {tim.get('ml_load_s', 0):.3f}s (enabled: {models['ml']['enabled']})")
    print(f"   - DL-anomaly  : {tim.get('dl_anom_load_s', 0):.3f}s (enabled: {models['dl_anomaly']['enabled']})")
    print(f"   - DL-pattern  : {tim.get('dl_pat_load_s', 0):.3f}s (enabled: {models['dl_pattern']['enabled']})")

    # 2. 파이프라인 실행을 위한 준비
    print("\n" + "=" * 80)
    print("🚀 Pipeline Benchmark Phase")
    print("=" * 80)

    from datetime import datetime

    run_tag = datetime.now().strftime("bench_%Y%m%d_%H%M%S")
    run_root = DEFAULT_RUN_ROOT
    run_dir = (run_root / run_tag).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "data_flow.log"
    incoming_path = run_dir / "incoming_packets.jsonl"
    reasm_before_path = run_dir / "reassembly_before.jsonl"
    reasm_after_path = run_dir / "reassembly_after.jsonl"
    final_path = run_dir / "final_results.json"

    logger = setup_data_logger(log_path, mode="w")
    stores = RunStores(incoming_path, reasm_before_path, reasm_after_path, final_path)

    # 3. 파이프라인 생성 (모델은 이미 로드됨)
    pipeline = OperationalPipeLine(
        logger=logger,
        stores=stores,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        redis_password=redis_password,
        protocols=protocols or list(DEFAULT_PROTOCOLS),
        interval_sec=0.0,  # 최대 속도
        replay=False,
        stop_after_windows=stop_after_windows,
        ml_warmup=0,  # 이미 warmup 완료
        dl_warmup=0,  # 이미 warmup 완료
        ml_skip_stats=0,
        dl_skip_stats=0,
        server_mode=False,
        model_load_dir=model_load_dir,
        models=models,  # 미리 로드한 모델 전달
    )

    # 4. 파이프라인 실행
    print(f"\n🔄 Running pipeline for {stop_after_windows} windows...")

    t_pipeline_start = time.perf_counter()
    try:
        pipeline.run()
    finally:
        pipeline.close()
    t_pipeline_end = time.perf_counter()

    pipeline_time = t_pipeline_end - t_pipeline_start

    # 5. 결과 수집
    stats = pipeline.stats

    results = {
        "model_load_time": model_load_time,
        "model_timing": tim,
        "pipeline_time": pipeline_time,
        "total_raw_packets": pipeline.total_raw_packets,
        "total_prepares": pipeline.total_prepares,
        "total_windows": pipeline.total_windows,
        "stats": {
            "redis": list(stats.get("redis", [])),
            "json_parse": list(stats.get("json_parse", [])),
            "merge_total": list(stats.get("merge_total", [])),
            "feature": list(stats.get("feature", [])),
            "ml": list(stats.get("ml", [])),
            "dl_window": list(stats.get("dl_window", [])),
        },
        "run_dir": str(run_dir),
    }

    # 6. 결과 출력
    print("\n" + "=" * 80)
    print("📊 Performance Results (mean, min, max)")
    print("=" * 80)

    print(f"\n📁 Run directory: {run_dir}")
    print(f"📦 Total raw packets: {pipeline.total_raw_packets}")
    print(f"📦 Total prepares: {pipeline.total_prepares}")
    print(f"📦 Total windows: {pipeline.total_windows}")
    print(f"⏱️  Pipeline time: {pipeline_time:.3f}s")

    if pipeline_time > 0:
        pps_raw = pipeline.total_raw_packets / pipeline_time
        pps_prep = pipeline.total_prepares / pipeline_time
        wps = pipeline.total_windows / pipeline_time
        print(f"📈 Throughput: {pps_raw:.2f} pkt/s, {pps_prep:.2f} prep/s, {wps:.2f} win/s")

    print("\n" + "-" * 80)
    print("Timing breakdown (per operation):")
    print("-" * 80)

    formatter = format_stats_detailed if detailed else format_stats

    print(formatter("redis", stats.get("redis", [])))
    print(formatter("json_parse", stats.get("json_parse", [])))
    print(formatter("merge_total", stats.get("merge_total", [])))
    print(formatter("feature", stats.get("feature", [])))
    print(formatter("ml", stats.get("ml", [])))
    print(formatter("dl_window", stats.get("dl_window", [])))

    # total 계산 (redis + json_parse + merge_total + feature + ml + dl_window의 평균 합)
    total_times = []
    n_windows = pipeline.total_windows
    if n_windows > 0:
        # 각 윈도우당 총 시간 추정
        redis_avg = statistics.mean(stats.get("redis", [0])) if stats.get("redis") else 0
        json_avg = statistics.mean(stats.get("json_parse", [0])) if stats.get("json_parse") else 0
        merge_avg = statistics.mean(stats.get("merge_total", [0])) if stats.get("merge_total") else 0
        feature_avg = statistics.mean(stats.get("feature", [0])) if stats.get("feature") else 0
        ml_avg = statistics.mean(stats.get("ml", [0])) if stats.get("ml") else 0
        dl_avg = statistics.mean(stats.get("dl_window", [0])) if stats.get("dl_window") else 0

        # 윈도우당 평균 처리 시간
        per_window_avg = (
            redis_avg * WINDOW_SIZE +  # 윈도우당 패킷 수만큼 redis 호출
            json_avg * WINDOW_SIZE +
            merge_avg +
            feature_avg * WINDOW_SIZE +
            ml_avg * WINDOW_SIZE +
            dl_avg
        )

        print("\n" + "-" * 80)
        print("Summary (per window estimated):")
        print("-" * 80)
        print(f"- {'total_per_window':<16}: mean={per_window_avg * 1000:>8.3f}ms")
        print(f"- {'actual_per_window':<16}: mean={pipeline_time / n_windows * 1000:>8.3f}ms")

    print("\n" + "=" * 80)
    print("✅ Benchmark completed!")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="ML/DL Pipeline Benchmark")

    parser.add_argument("--model-load-dir", type=Path, default=DEFAULT_MODEL_LOAD_DIR)
    parser.add_argument("--stop-after-windows", type=int, default=10)
    parser.add_argument("--host", default=DEFAULT_REDIS_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_REDIS_PORT)
    parser.add_argument("--db", type=int, default=DEFAULT_REDIS_DB)
    parser.add_argument("--password", default=DEFAULT_REDIS_PASSWORD)
    parser.add_argument("--protocols", nargs="+", default=None)
    parser.add_argument("--detailed", action="store_true", help="Show detailed stats with percentiles")

    args = parser.parse_args()

    run_benchmark(
        model_load_dir=args.model_load_dir.resolve(),
        stop_after_windows=args.stop_after_windows,
        redis_host=args.host,
        redis_port=args.port,
        redis_db=args.db,
        redis_password=args.password,
        protocols=args.protocols,
        detailed=args.detailed,
    )


if __name__ == "__main__":
    main()
