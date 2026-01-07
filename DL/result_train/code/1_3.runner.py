# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from rt_common import (
    build_windows,
    compute_window_mse,
    evaluate_predictions,
    load_gt_map,
    save_recon_plot,
    save_roc_curve,
    write_csv,
    write_json,
)
from rt_model import load_feature_weights, load_model_bundle


def run_one(
    *,
    tag: str,
    input_jsonl: Path,
    model_dir: Path,
    output_dir: Path,
    window_size: int,
    batch_size: int,
    threshold: Optional[float],
    gt_path: Optional[Path],
    ignore_pred_minus1: bool,
    feature_weights_file: Optional[Path],
    smooth_window: int,
    plot_points: str,
    point_stride: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config, feature_keys, threshold_from_file, pad_value = load_model_bundle(model_dir)

    T_cfg = int(config.get("T", window_size))
    if int(T_cfg) != int(window_size):
        raise ValueError(f"window_size({window_size}) != model config.T({T_cfg}) : tag={tag}")

    if threshold is None:
        threshold = threshold_from_file

    fw = load_feature_weights(feature_keys, feature_weights_file)

    X_windows, meta_list = build_windows(input_jsonl, feature_keys, window_size, pad_value)
    if X_windows.shape[0] == 0:
        raise RuntimeError(f"윈도우 0개: {input_jsonl}")

    recon = model.predict(X_windows, batch_size=batch_size, verbose=1)
    mse = compute_window_mse(X_windows, recon, pad_value, fw)

    pred_map: Dict[int, Tuple[float, int]] = {}
    mse_rows: list[Dict[str, Any]] = []
    for m, score in zip(meta_list, mse):
        wi = int(m["window_index"])
        ypred = (int(score > threshold) if threshold is not None else -1)
        pred_map[wi] = (float(score), int(ypred))
        mse_rows.append(
            {
                "window_index": wi,
                "mse": float(score),
                "pattern": m.get("pattern", ""),
                "valid_len": int(m.get("valid_len", 0)),
                "is_anomaly_pred": int(ypred),
            }
        )

    mse_csv = output_dir / f"mse_per_window_{tag}.csv"
    write_csv(mse_csv, ["window_index", "mse", "pattern", "valid_len", "is_anomaly_pred"], mse_rows)

    if gt_path is None:
        gt_path = input_jsonl
    gt_map = load_gt_map(gt_path)

    recon_png = output_dir / f"recon_error_{tag}.png"
    save_recon_plot(
        meta_list=meta_list,
        mse=mse,
        out_png=recon_png,
        threshold=threshold,
        gt_map=gt_map,
        smooth_window=smooth_window,
        plot_points=plot_points,
        point_stride=point_stride,
    )

    metrics, mse_stats, y_true, scores = evaluate_predictions(
        pred_map=pred_map,
        gt_map=gt_map,
        threshold=threshold,
        ignore_pred_minus1=ignore_pred_minus1,
    )
    metrics["tag"] = tag
    metrics["input"] = str(input_jsonl)
    metrics["gt_path"] = str(gt_path)

    write_json(output_dir / f"metrics_{tag}.json", metrics)
    write_json(output_dir / f"analyze_mse_dist_{tag}.json", mse_stats)

    if y_true is not None and scores is not None:
        save_roc_curve(
            y_true,
            scores,
            output_dir / f"roc_curve_{tag}.png",
            output_dir / f"roc_points_{tag}.csv",
        )
