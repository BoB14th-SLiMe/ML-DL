#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocessing.py
전체 전처리 파이프라인 (경로는 YAML로 관리)
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a dict")
    return data


def rpath(base: Path, p: str) -> str:
    return str((base / p).resolve())


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"\n[RUN] (cwd={cwd})")
    print("      " + " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    here = Path(__file__).resolve().parent

    cfg_path = here / "preprocessing.yaml"
    cfg = load_yaml(cfg_path)

    paths = cfg.get("paths", {})
    scripts = cfg.get("scripts", {})

    script1 = scripts.get("run_all_preprocess", "1.run_all_preprocess.py")
    script2 = scripts.get("extract_feature", "2.extract_feature.py")

    raw_jsonl = rpath(here, paths["raw_jsonl"])
    preprocess_out_dir = rpath(here, paths["preprocess_out_dir"])
    pattern_jsonl = rpath(here, paths["pattern_jsonl"])
    feature_out_jsonl = rpath(here, paths["feature_out_jsonl"])

    # cmd1 = [
    #     sys.executable,
    #     str((here / script1).resolve()),
    #     "-i", raw_jsonl,
    #     "-o", preprocess_out_dir,
    # ]

    cmd2 = [
        sys.executable,
        str((here / script2).resolve()),
        "-i", pattern_jsonl,
        "-p", preprocess_out_dir,
        "-o", feature_out_jsonl,
    ]

    # run_cmd(cmd1, cwd=here)
    run_cmd(cmd2, cwd=here)

    print("\n[DONE] preprocess + extract_feature completed successfully.")


if __name__ == "__main__":
    main()
