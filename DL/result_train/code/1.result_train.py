#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1.result_train.py
LSTM + AE 학습 코드

- 1_1_common.py : utils/io/gt/core/eval/plot/roc
- 1_2_model.py  : keras3 호환 model loader + feature weights
- 1_3_runner.py : run_one 파이프라인

"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict


def _load_part(here: Path, filename: str, module_name: str):
    path = (here / filename).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing part file: {path}")

    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec: {module_name} from {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bootstrap(here: Path) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")

    parts: Dict[str, Any] = {}
    parts["rt_common"] = _load_part(here, "1_1.common.py", "rt_common")
    parts["rt_model"] = _load_part(here, "1_2.model.py", "rt_model")
    parts["rt_runner"] = _load_part(here, "1_3.runner.py", "rt_runner")
    return parts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default=None, help="YAML 경로(미지정 시 1.result_train.yaml)")
    return p.parse_args()


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    parts = _bootstrap(here)

    common = parts["rt_common"]
    runner = parts["rt_runner"]

    cfg_path = Path(args.config).resolve() if args.config else (here / "1.result_train.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"YAML 없음: {cfg_path}")

    cfg = common.load_yaml(cfg_path)
    base_dir = cfg_path.parent

    defaults = cfg.get("defaults", {})
    runs = cfg.get("runs", [])

    if not isinstance(defaults, dict):
        defaults = {}
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs가 비어있습니다.")

    fail_fast = bool(defaults.get("fail_fast", True))

    ok = 0
    fail = 0

    for i, r in enumerate(runs, start=1):
        if not isinstance(r, dict):
            continue

        merged = common.merge_cfg(defaults, r)
        tag = str(merged.get("tag", f"run{i}"))

        input_jsonl = common.resolve_path(merged.get("input"), base_dir)
        if input_jsonl is None or not input_jsonl.exists():
            msg = f"[ERROR] input 경로 invalid: tag={tag}, input={merged.get('input')}"
            if fail_fast:
                raise FileNotFoundError(msg)
            print(msg)
            fail += 1
            continue

        model_dir = common.resolve_path(merged.get("model_dir"), base_dir)
        if model_dir is None or not model_dir.exists():
            msg = f"[ERROR] model_dir invalid: tag={tag}, model_dir={merged.get('model_dir')}"
            if fail_fast:
                raise FileNotFoundError(msg)
            print(msg)
            fail += 1
            continue

        output_dir = common.resolve_path(merged.get("output_dir"), base_dir)
        if output_dir is None:
            msg = f"[ERROR] output_dir invalid: tag={tag}, output_dir={merged.get('output_dir')}"
            if fail_fast:
                raise ValueError(msg)
            print(msg)
            fail += 1
            continue

        window_size = common.to_int(merged.get("window_size", 80)) or 80
        batch_size = common.to_int(merged.get("batch_size", 128)) or 128
        threshold = common.to_float(merged.get("threshold", None))
        ignore_pred_minus1 = bool(merged.get("ignore_pred_minus1", False))

        gt_path = common.resolve_path(merged.get("gt_path", None), base_dir)
        feature_weights_file = common.resolve_path(merged.get("feature_weights_file", None), base_dir)

        smooth_window = common.to_int(merged.get("smooth_window", 31)) or 31
        plot_points = str(merged.get("plot_points", "exceed"))  # none|exceed|all
        point_stride = common.to_int(merged.get("point_stride", 10)) or 10

        try:
            print("\n" + "=" * 80)
            print(f"[RUN {i}/{len(runs)}] tag={tag}")
            print(f"  input      : {input_jsonl}")
            print(f"  model_dir  : {model_dir}")
            print(f"  output_dir : {output_dir}")
            print(f"  window_size={window_size}, batch_size={batch_size}, threshold={threshold}")
            print(f"  gt_path    : {gt_path if gt_path else '(default=input)'}")
            print(f"  fw_file    : {feature_weights_file if feature_weights_file else '(none -> ones)'}")
            print("=" * 80)

            runner.run_one(
                tag=tag,
                input_jsonl=input_jsonl,
                model_dir=model_dir,
                output_dir=output_dir,
                window_size=window_size,
                batch_size=batch_size,
                threshold=threshold,
                gt_path=gt_path,
                ignore_pred_minus1=ignore_pred_minus1,
                feature_weights_file=feature_weights_file,
                smooth_window=smooth_window,
                plot_points=plot_points,
                point_stride=point_stride,
            )
            ok += 1
        except Exception as e:
            print(f"[FAIL] tag={tag}: {e}")
            fail += 1
            if fail_fast:
                raise

    print("\n========== DONE ==========")
    print(f"OK   : {ok}")
    print(f"FAIL : {fail}")


if __name__ == "__main__":
    main()
