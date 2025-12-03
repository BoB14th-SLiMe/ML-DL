#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_mse_dist.py

attack_result (GT) CSV와 window_scores (pred) CSV를 기준으로
MSE / score 분포를 분석하고, 패턴/프로토콜/자산별 그룹 통계를 계산하는 스크립트.

입력:
  --attack-csv : 윈도우 단위 GT 라벨 CSV (예: attack_result_XXX.csv)
  --pred-csv   : 윈도우 단위 점수/예측 CSV (예: window_scores_XXX.csv)

출력:
  --output-json 을 지정하면 그 경로에 저장.
  지정하지 않으면, pred CSV 디렉토리에
    analyze_mse_dist_{tag}.json
  으로 자동 저장.
  (tag 기본값: pred CSV 파일명 stem)
"""

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
    """
    if "mse" in df.columns:
        return "mse"
    cand = [c for c in df.columns if c in ("score", "recon_error", "mse_pred")]
    if len(cand) == 1:
        return cand[0]
    raise KeyError(f"점수 컬럼(mse / score 등)을 찾을 수 없음. 현재 컬럼들: {list(df.columns)}")


def make_stats(arr):
    """
    단일 배열에 대한 기본 통계
    """
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


def make_group_stats(df, group_col, label_col, score_col):
    """
    group_col 기준으로 all / attack / normal 별 통계
    예: group_col='pattern_gt' 면 각 패턴별 통계
    """
    result = {}
    grouped = df.groupby(group_col)

    for g, gdf in grouped:
        arr_all = gdf[score_col].values
        arr_attack = gdf[gdf[label_col] == 1][score_col].values
        arr_normal = gdf[gdf[label_col] == 0][score_col].values

        result[str(g)] = {
            "all": make_stats(arr_all),
            "attack": make_stats(arr_attack),
            "normal": make_stats(arr_normal),
            "n_all": int(len(gdf)),
            "n_attack": int((gdf[label_col] == 1).sum()),
            "n_normal": int((gdf[label_col] == 0).sum()),
            "attack_ratio": float((gdf[label_col] == 1).mean()) if len(gdf) > 0 else None,
        }

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attack-csv", required=True, help="GT 윈도우 라벨 CSV (attack_result_XXX.csv 등)")
    p.add_argument("--pred-csv", required=True, help="모델 점수/예측 CSV (window_scores_XXX.csv 등)")
    p.add_argument(
        "--output-json",
        default=None,
        help=(
            "분석 결과 JSON 파일 경로. "
            "미지정 시 pred CSV 디렉토리에 analyze_mse_dist_{tag}.json 으로 저장"
        ),
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="오차가 큰 윈도우를 상위 몇 개까지 저장할지 (기본 20)",
    )
    p.add_argument(
        "--tag",
        default=None,
        help=(
            "출력 JSON 이름에 사용할 태그 "
            "(기본: pred CSV 파일명 stem, 예: window_scores_attack_ver5_1)"
        ),
    )
    args = p.parse_args()

    attack_path = Path(args.attack_csv)
    pred_path = Path(args.pred_csv)

    # 🔥 tag 결정 (기본: pred CSV stem)
    tag = args.tag if args.tag is not None else pred_path.stem

    # 🔥 output JSON 경로 결정
    if args.output_json is not None:
        out_path = Path(args.output_json)
    else:
        out_path = pred_path.parent / f"analyze_mse_dist_{tag}.json"

    print(f"[INFO] GT CSV (attack)      : {attack_path}")
    print(f"[INFO] Pred CSV (scores)    : {pred_path}")
    print(f"[INFO] 사용 태그(tag)       : {tag}")
    print(f"[INFO] 출력 JSON (result)   : {out_path}")

    df_attack = pd.read_csv(attack_path)
    df_pred = pd.read_csv(pred_path)

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

    # 타입 정리 (라벨은 int, score는 float)
    df[label_col] = df[label_col].astype(int)
    df[score_col] = df[score_col].astype(float)

    # attack / normal 분리
    attack = df[df[label_col] == 1][score_col].values
    normal = df[df[label_col] == 0][score_col].values

    stats = {
        "meta": {
            "label_col": label_col,
            "score_col": score_col,
            "n_total": int(len(df)),
            "n_attack": int((df[label_col] == 1).sum()),
            "n_normal": int((df[label_col] == 0).sum()),
        },
        "attack": make_stats(attack),
        "normal": make_stats(normal),
    }

    # --------------------------------------------------
    # 1) 패턴/그룹 단위로 왜 오차가 큰지 보고 싶을 때
    #    가능한 group 컬럼들을 자동으로 찾아본다
    # --------------------------------------------------
    group_candidates = [
        "pattern", "pattern_gt", "pattern_pred",
        "protocol", "protocol_gt", "protocol_pred",
        "src_asset", "dst_asset",
        "modbus.fc", "s7comm.fn", "xgt_fen.cmd",
    ]
    group_cols = [c for c in group_candidates if c in df.columns]

    group_stats = {}
    for gc in group_cols:
        group_stats[gc] = make_group_stats(df, gc, label_col, score_col)

    if group_stats:
        stats["group_stats"] = group_stats

    # --------------------------------------------------
    # 2) 오차가 큰 윈도우 TOP-K를 저장해서
    #    "어떤 윈도우/패턴 때문에 오차 범위가 커졌는지"를 직접 추적
    # --------------------------------------------------
    top_k = args.top_k

    # 디버깅에 도움이 될 만한 컬럼들만 추려서 같이 저장
    extra_cols = []
    for c in ["pattern", "pattern_gt", "pattern_pred",
              "protocol", "protocol_gt", "protocol_pred",
              "src_asset", "dst_asset"]:
        if c in df.columns:
            extra_cols.append(c)

    # 공격 중에서 오차 큰 윈도우
    top_attack_df = (
        df[df[label_col] == 1]
        .nlargest(top_k, score_col)[["window_index", score_col, label_col] + extra_cols]
    )
    # 정상 중에서 오차 큰 윈도우
    top_normal_df = (
        df[df[label_col] == 0]
        .nlargest(top_k, score_col)[["window_index", score_col, label_col] + extra_cols]
    )

    stats["top_attack_windows"] = top_attack_df.to_dict(orient="records")
    stats["top_normal_windows"] = top_normal_df.to_dict(orient="records")

    # 콘솔에도 요약만 찍기
    print(json.dumps({
        "meta": stats["meta"],
        "attack": stats["attack"],
        "normal": stats["normal"],
    }, indent=2, ensure_ascii=False))

    # 전체를 JSON 파일로 저장
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] 분석 결과 JSON 저장 완료 → {out_path}")


if __name__ == "__main__":
    main()

"""
예시:

python 3.analyze_mse_dist.py \
  --attack-csv ../result/attack_result_attack_ver5_1.csv \
  --pred-csv ../result/benchmark/window_scores_attack_ver5_1.csv

# → ../result/benchmark/analyze_mse_dist_window_scores_attack_ver5_1.json 생성

또는 태그를 직접 주고 싶으면:

python 3.analyze_mse_dist.py \
  --attack-csv ../result/attack_result_attack_ver5_2.csv \
  --pred-csv ../result/benchmark/window_scores_attack_ver5_2.csv \
  --tag attack_ver5_2

# → ../result/benchmark/analyze_mse_dist_attack_ver5_2.json 생성

명시적으로 경로를 주고 싶으면:

python 3.analyze_mse_dist.py \
  --attack-csv ../result/attack_result_attack_ver5_1.csv \
  --pred-csv ../result/benchmark/window_scores_attack_ver5_1.csv \
  --output-json ../result/analyze_mse_dist_attack_ver5_1.json
"""
