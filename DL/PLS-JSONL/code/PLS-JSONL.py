#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
usage: python PLS-JSONL.py -c PLS-JSONL.yaml

PLS-JSONL의 전체 코드를 실행하는 파일

"""
from __future__ import annotations

from pathlib import Path
import sys
import importlib.util
import argparse
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_module(script_path: Path):
    module_name = f"dyn_{script_path.stem}".replace(".", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _resolve(root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp)


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _load_cfg(cfg_path: Path):
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a YAML mapping(dict).")
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c", "--config",
        default=str(ROOT / "config" / "pipeline.yaml"),
        help="YAML config path (default: ROOT/config/pipeline.yaml)",
    )
    args = ap.parse_args()

    cfg = _load_cfg(Path(args.config))

    mod_cfg = cfg["modules"]
    script_path1 = _resolve(ROOT, mod_cfg["step1_pls_to_raw"])
    script_path2 = _resolve(ROOT, mod_cfg["step2_pls_to_raw_windows"])
    script_path3 = _resolve(ROOT, mod_cfg["step3_window_index"])

    PLS_to_RAW_mod1 = load_module(script_path1).PLS_to_RAW_mapping
    PLS_to_RAW_mod2 = load_module(script_path2).PLS_to_RAW_mapping
    window_index = load_module(script_path3).window_index

    paths = cfg["paths"]

    # step1
    step1 = paths["step1"]
    pls_path1 = _resolve(ROOT, step1["pls"])
    raw_path1 = _resolve(ROOT, step1["raw"])
    out_path1 = _resolve(ROOT, step1["out"])
    _ensure_parent(out_path1)
    PLS_to_RAW_mod1(pls_path1, raw_path1, out_path1)

    # step2
    step2 = paths["step2"]
    pls_path2 = _resolve(ROOT, step2["pls"])
    raw_path2 = _resolve(ROOT, step2["raw"])
    out_path2 = _resolve(ROOT, step2["out"])
    _ensure_parent(out_path2)
    PLS_to_RAW_mod2(pls_path2, raw_path2, out_path2)

    # step3
    step3 = paths["step3"]
    pattern_path = _resolve(ROOT, step3["pattern"])
    window_path = _resolve(ROOT, step3["window"])
    out_path3 = _resolve(ROOT, step3["out"])
    _ensure_parent(out_path3)
    window_index(pattern_path, window_path, out_path3)