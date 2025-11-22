#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path


def pick_label_col(df):
    """
    is_anomaly / is_anomaly_gt / is_anomaly_pred 중에서
    GT 라벨로 쓸 컬럼을 선택
    """
    if "is_anomaly_gt" in df.columns:
        return "is_anomaly_gt"
    if "is_anomaly" in df.columns:
        return "is_anomaly"
    # 최악의 경우: 유일하게 하나만 있으면 그걸 쓴다
    cand = [c for c in df.columns if c.startswith("is_anomaly")]
    if len(cand) == 1:
        return cand[0]
    raise KeyError(f"라벨 컬럼(is_anomaly*)을 찾을 수 없음. 현재 컬럼들: {list(df.columns)}")


def pick_score_col(df):
    """
    mse / mse_pred / score 등에서 점수 컬럼 선택
    (지금 파일 기준으로는 mse 하나만 있으니까 mse 먼저 본다)
    """
    if "mse" in df.columns:
        return "mse"
    cand = [c for c in df.columns if c in ("score", "recon_error", "mse_pred")]
    if len(cand) == 1:
        return cand[0]
    raise KeyError(f"점수 컬럼(mse / score 등)을 찾을 수 없음. 현재 컬럼들: {list(df.columns)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attack-csv", required=True)
    p.add_argument("--pred-csv", required=True)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    df_attack = pd.read_csv(args.attack_csv)
    df_pred = pd.read_csv(args.pred_csv)

    # window_index 기준으로 join
    df = pd.merge(
        df_attack,
        df_pred,
        on="window_index",
        suffixes=("_gt", "_pred")
    )

    # 어떤 컬럼이 라벨/점수인지 자동으로 선택
    label_col = pick_label_col(df)
    score_col = pick_score_col(df)

    # 디버깅용: 컬럼 한번 찍어볼 수도 있음
    # print("Columns:", list(df.columns))
    # print("label_col:", label_col, "score_col:", score_col)

    # attack / normal 분리
    attack = df[df[label_col] == 1][score_col].values
    normal = df[df[label_col] == 0][score_col].values

    def make_stats(arr):
        if len(arr) == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "p50": None,
                "p90": None,
                "p95": None,
                "p99": None,
            }
        return {
            "count": int(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    stats = {
        "meta": {
            "label_col": label_col,
            "score_col": score_col,
        },
        "attack": make_stats(attack),
        "normal": make_stats(normal),
    }

    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()


"""
python 3.analyze_mse_dist.py --attack-csv ../result/attack_result.csv --pred-csv ../result/benchmark/window_scores.csv --output-json ../result/analyze_mse_dist.json

"""