#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class Stats:
    mean: float
    std: float
    min_: float
    max_: float


def load_json(path: Path) -> JsonDict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_stats(data: JsonDict) -> Stats:
    stats = data.get("stats")
    if not isinstance(stats, dict):
        raise ValueError('Invalid threshold.json: missing object key "stats".')

    missing = [k for k in ("mean", "std", "min", "max") if k not in stats]
    if missing:
        raise ValueError(f"Invalid threshold.json: stats missing keys: {missing}")

    try:
        return Stats(
            mean=float(stats["mean"]),
            std=float(stats["std"]),
            min_=float(stats["min"]),
            max_=float(stats["max"]),
        )
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid threshold.json: stats values must be numeric. ({e})") from e


def compute_threshold(stats: Stats, k: float) -> JsonDict:
    tau = stats.mean - (k * stats.std)
    tau_clip = min(max(tau, stats.min_), stats.max_)
    return {
        "k": float(k),
        "tau": float(tau),
        "tau_clip": float(tau_clip),
        "mean": float(stats.mean),
        "std": float(stats.std),
        "clip_min": float(stats.min_),
        "clip_max": float(stats.max_),
    }


def build_output_path(src: Path, k: float) -> Path:
    k_tag = f"{k}".replace(".", "_")
    return src.parent / f"threshold_mode_minus_k{k_tag}.json"


def save_payload(
    out_path: Path,
    src_path: Path,
    result: JsonDict,
    original: JsonDict | None = None,
) -> None:
    payload: JsonDict = {
        "source_threshold_json": str(src_path),
        "assumed_rule": "anomaly if score >= threshold(tau_clip)",
        "formula": "tau = mean - k*std; tau_clip = min(max(tau, min), max)",
        "result": result,
    }
    if original is not None:
        payload["original"] = original

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute threshold(tau_clip) from threshold.json stats.")
    p.add_argument(
        "--threshold-json",
        default=str(Path(r"../data/threshold.json")),
        help="Path to threshold.json containing { stats: { mean, std, min, max } }",
    )
    p.add_argument("--k", type=float, default=0.3, help="k (higher => lower threshold => more sensitive)")
    p.add_argument("--include-original", action="store_true", help="Include original threshold.json in output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src_path = Path(args.threshold_json)

    data = load_json(src_path)
    stats = parse_stats(data)
    result = compute_threshold(stats, float(args.k))

    out_path = build_output_path(src_path, float(args.k))
    save_payload(out_path, src_path, result, original=data if args.include_original else None)

    print(f"[OK] k={result['k']}")
    print(f"tau      = {result['tau']}")
    print(f"tau_clip = {result['tau_clip']}")
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
