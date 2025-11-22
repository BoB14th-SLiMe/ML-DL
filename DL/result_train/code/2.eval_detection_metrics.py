#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_detection_metrics.py

두 개의 윈도우 레벨 CSV를 비교해서 성능 지표를 계산하는 스크립트.

입력 CSV 형식 (예시):

1) attack_result.csv  (실제 라벨, GT)
   window_index,start_packet_idx,end_packet_idx,valid_len,is_anomaly
   - is_anomaly: 1 = 공격(FC6 포함), 0 = 정상

2) window_scores.csv  (모델 예측 결과)
   window_index,start_packet_idx,end_packet_idx,valid_len,mse,is_anomaly
   - is_anomaly: 1 = 모델이 이상으로 판단, 0 = 정상
   - (threshold가 없어서 -1일 수도 있음 → 이 경우는 평가에서 제외하거나 별도 처리 가능)

동작:
  - window_index 기준 inner join
  - (필요시) 예측 is_anomaly == -1 인 행은 평가에서 제외
  - TP/TN/FP/FN, accuracy, precision, recall, F1, TPR, FPR 계산
  - 콘솔에 출력 + JSON 파일로 저장
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--attack-csv", "-a", required=True,
        help="실제 공격 여부가 들어있는 CSV (예: attack_result.csv, is_anomaly=GT)"
    )
    p.add_argument(
        "--pred-csv", "-p", required=True,
        help="모델 예측 결과 CSV (예: window_scores.csv, is_anomaly=예측)"
    )
    p.add_argument(
        "--output-json", "-o", default="metrics.json",
        help="성능 지표를 저장할 JSON 파일 경로 (default: metrics.json)"
    )
    p.add_argument(
        "--ignore-pred-minus1", action="store_true",
        help="예측 CSV에서 is_anomaly == -1 인 행은 평가에서 제외 (threshold 안 쓴 경우 등)"
    )

    return p.parse_args()


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def main():
    args = parse_args()

    attack_path = Path(args.attack_csv)
    pred_path = Path(args.pred_csv)
    out_path = Path(args.output_json)

    print(f"[INFO] GT CSV (attack)  : {attack_path}")
    print(f"[INFO] Pred CSV (model): {pred_path}")

    df_gt = pd.read_csv(attack_path)
    df_pred = pd.read_csv(pred_path)

    # 필수 컬럼 체크
    for col in ["window_index", "is_anomaly"]:
        if col not in df_gt.columns:
            raise ValueError(f"GT CSV에 '{col}' 컬럼이 없습니다.")
        if col not in df_pred.columns:
            raise ValueError(f"Pred CSV에 '{col}' 컬럼이 없습니다.")

    # window_index 기준 inner join
    merged = pd.merge(
        df_gt[["window_index", "is_anomaly"]],
        df_pred[["window_index", "is_anomaly"]],
        on="window_index",
        how="inner",
        suffixes=("_true", "_pred"),
    )

    print(f"[INFO] join 후 행 수: {len(merged)}")

    # 타입 정리
    merged["is_anomaly_true"] = merged["is_anomaly_true"].astype(int)
    merged["is_anomaly_pred"] = merged["is_anomaly_pred"].astype(int)

    # is_anomaly_pred == -1 (미라벨) 제거 옵션
    if args.ignore_pred_minus1:
        before = len(merged)
        merged = merged[merged["is_anomaly_pred"] != -1].copy()
        after = len(merged)
        print(f"[INFO] is_anomaly_pred == -1 제거: {before} -> {after}")

    y_true = merged["is_anomaly_true"].to_numpy()
    y_pred = merged["is_anomaly_pred"].to_numpy()

    # 혼동행렬 요소 계산
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = tp + tn + fp + fn

    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)  # TPR
    f1 = safe_div(2 * precision * recall, precision + recall)

    tpr = recall  # same
    fpr = safe_div(fp, fp + tn)
    tnr = safe_div(tn, tn + fp)
    fnr = safe_div(fn, fn + tp)

    metrics: Dict[str, Any] = {
        "num_samples": total,
        "confusion_matrix": {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        },
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": tpr,
        "fpr": fpr,
        "tnr": tnr,
        "fnr": fnr,
    }

    print("===== Detection Metrics =====")
    print(f"Samples (windows) : {total}")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall (TPR)      : {recall:.4f}")
    print(f"F1-score          : {f1:.4f}")
    print(f"FPR               : {fpr:.4f}")
    print(f"TNR               : {tnr:.4f}")
    print(f"FNR               : {fnr:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[INFO] 성능 지표 JSON 저장 완료 → {out_path}")


if __name__ == "__main__":
    main()


"""
# attack_result.csv vs window_scores.csv 비교
python 2.eval_detection_metrics.py --attack-csv ../result/attack_result.csv --pred-csv ../result/benchmark/window_scores.csv --output-json ../result/eval_detection_metrics.json

"""