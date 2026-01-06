#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_detection_metrics.py

두 개의 윈도우 레벨 파일을 비교해서 성능 지표를 계산하는 스크립트.

입력 형식:

1) GT (실제 라벨)
   - CSV (attack_result_XXX.csv)
     window_index,start_packet_idx,end_packet_idx,valid_len,is_anomaly
   - 또는 JSONL (attack_ver2_window.jsonl 등)
     {"window_index": ..., "start_packet_idx": ..., "end_packet_idx": ..., "valid_len": ..., "is_anomaly": ...}
     또는 pattern 컬럼(NORMAL/ATTACK)만 있을 경우 → 내부에서 is_anomaly 생성

2) Pred (모델 예측 결과, 현재는 CSV)
   window_scores_XXX.csv:
     window_index,start_packet_idx,end_packet_idx,valid_len,mse,is_anomaly
     - is_anomaly: 1 = 모델이 이상으로 판단, 0 = 정상
     - (threshold가 없어서 -1일 수도 있음 → 옵션으로 평가에서 제외 가능)

동작:
  - window_index 기준 inner join
  - (옵션) 예측 is_anomaly == -1 인 행은 평가에서 제외
  - TP/TN/FP/FN, accuracy, precision, recall, F1, TPR, FPR, Balanced Accuracy 등 계산
  - (mse 컬럼이 있으면)
      - ROC-AUC, PR-AUC 계산 시도
      - 공격/정상 별 mse 통계 (mean, std, min, max, p50, p90, p95, p99) 계산
      - 별도 JSON 파일(mse_stats_*.json)로 저장
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--attack-csv", "-a", required=True,
        help="실제 공격 여부가 들어있는 파일 (CSV 또는 JSONL; 예: attack_result_XXX.csv, attack_ver2_window.jsonl)"
    )
    p.add_argument(
        "--pred-csv", "-p", required=True,
        help="모델 예측 결과 CSV (예: window_scores_XXX.csv, is_anomaly=예측)"
    )
    p.add_argument(
        "--output-json", "-o", default=None,
        help=(
            "성능 지표를 저장할 JSON 파일 경로 "
            "(미지정 시 pred CSV 디렉토리에 metrics_{tag}.json 으로 저장)"
        ),
    )
    p.add_argument(
        "--mse-stats-json", default=None,
        help=(
            "MSE 통계를 별도로 저장할 JSON 경로 "
            "(미지정 시 pred CSV 디렉토리에 mse_stats_{tag}.json 으로 저장)"
        ),
    )
    p.add_argument(
        "--ignore-pred-minus1", action="store_true",
        help="예측 CSV에서 is_anomaly == -1 인 행은 평가에서 제외 (threshold 안 쓴 경우 등)"
    )
    p.add_argument(
        "--tag", default=None,
        help=(
            "출력 JSON 이름에 사용할 태그 "
            "(기본: pred CSV 파일명 stem, 예: window_scores_attack_ver5_1)"
        ),
    )

    return p.parse_args()


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def try_compute_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, Optional[float]]:
    """
    mse 기반 ROC-AUC / PR-AUC 계산 (가능하면).
    - y_true: 0/1
    - scores: 클수록 공격일 가능성이 높은 score (예: mse)
    """
    result = {"roc_auc": None, "pr_auc": None}
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        print("[WARN] scikit-le른이 설치되어 있지 않아 ROC-AUC / PR-AUC 계산을 생략합니다.")
        return result

    # 클래스가 하나뿐이면 AUC 계산 불가
    if len(np.unique(y_true)) < 2:
        print("[WARN] y_true 에 클래스가 하나뿐이라 AUC 계산 불가 (양성/음성 모두 포함되어야 함).")
        return result

    try:
        roc_auc = float(roc_auc_score(y_true, scores))
        pr_auc = float(average_precision_score(y_true, scores))
        result["roc_auc"] = roc_auc
        result["pr_auc"] = pr_auc
        print(f"[INFO] ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
    except Exception as e:
        print(f"[WARN] AUC 계산 중 오류 발생: {e}")

    return result


def compute_mse_stats(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, Any]:
    """
    라벨(0=정상, 1=공격)별로 score(mse)의 통계를 계산한다.
    반환 형식 예:

    {
      "meta": {
        "label_col": "is_anomaly_true",
        "score_col": "mse",
        "n_total": 4684,
        "n_attack": 107,
        "n_normal": 4577
      },
      "attack": {
        "count": 107,
        "mean": ...,
        "std": ...,
        "min": ...,
        "max": ...,
        "p50": ...,
        "p90": ...,
        "p95": ...,
        "p99": ...
      },
      "normal": {
        "count": 4577,
        "mean": ...,
        "std": ...,
        "min": ...,
        "max": ...,
        "p50": ...,
        "p90": ...,
        "p95": ...,
        "p99": ...
      }
    }
    """

    def _summary(arr: np.ndarray) -> Dict[str, Any]:
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
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
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    attack_mask = (y_true == 1)
    normal_mask = (y_true == 0)

    attack_scores = scores[attack_mask]
    normal_scores = scores[normal_mask]

    attack_stats = _summary(attack_scores)
    normal_stats = _summary(normal_scores)

    mse_stats = {
        "meta": {
            "label_col": "is_anomaly_true",
            "score_col": "mse",
            "n_total": int(len(scores)),
            "n_attack": int(attack_mask.sum()),
            "n_normal": int(normal_mask.sum()),
        },
        "attack": attack_stats,
        "normal": normal_stats,
    }

    return mse_stats


def load_gt_table(path: Path) -> pd.DataFrame:
    """
    GT 파일을 로드해서 DataFrame 으로 반환.
    - .csv  → pandas.read_csv
    - .jsonl / .json → JSON Lines 로드 후 DataFrame
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        print(f"[INFO] GT 파일 형식: CSV ({path.name})")
        return pd.read_csv(path)
    elif suffix in [".jsonl", ".json"]:
        print(f"[INFO] GT 파일 형식: JSONL/JSON ({path.name})")
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rows.append(obj)
                except Exception as e:
                    print(f"[WARN] JSONL 파싱 실패: {e} (line 일부) → 스킵")
                    continue
        if not rows:
            raise ValueError(f"GT JSONL 에서 유효한 레코드를 하나도 못 읽었습니다: {path}")
        df = pd.DataFrame(rows)
        return df
    else:
        raise ValueError(f"지원하지 않는 GT 파일 확장자입니다: {path} (csv/jsonl/json 만 지원)")


def main():
    args = parse_args()

    attack_path = Path(args.attack_csv)
    pred_path = Path(args.pred_csv)

    # 🔥 tag 결정 (기본: pred CSV stem)
    tag = args.tag if args.tag is not None else pred_path.stem

    # # 🔥 output JSON 경로 결정 (성능 지표용)
    # if args.output_json is not None:
    #     out_path = Path(args.output_json)
    # else:
    #     out_path = pred_path.parent / f"metrics_{tag}.json"

    # 🔥 MSE 통계 JSON 경로 결정
    if args.mse_stats_json is not None:
        mse_out_path: Optional[Path] = Path(args.mse_stats_json)
    else:
        mse_out_path = pred_path.parent / f"analyze_mse_dist_{tag}.json"

    print(f"[INFO] GT 파일 (attack)          : {attack_path}")
    print(f"[INFO] Pred CSV (model)          : {pred_path}")
    print(f"[INFO] 사용 태그(tag)            : {tag}")
    # print(f"[INFO] 출력 JSON (metrics)       : {out_path}")
    print(f"[INFO] 출력 JSON (MSE stats)     : {mse_out_path}")

    # # ✅ GT: CSV 또는 JSONL 처리
    df_gt = load_gt_table(attack_path)

    # # ✅ pattern 기반 is_anomaly 생성 (없을 때 자동 처리)
    # if "is_anomaly" not in df_gt.columns and "pattern" in df_gt.columns:
    #     df_gt["is_anomaly"] = df_gt["pattern"].apply(
    #         lambda x: 1 if str(x).strip().upper() == "ATTACK" else 0
    #     )
    #     print(f"[INFO] 'pattern' 컬럼에서 is_anomaly 자동 생성 완료 ({df_gt['is_anomaly'].sum()}개 공격 윈도우)")

    # # ✅ Pred: 지금은 CSV만 사용
    df_pred = pd.read_csv(pred_path)

    # # 필수 컬럼 체크
    # for col in ["window_index", "is_anomaly"]:
    #     if col not in df_gt.columns:
    #         raise ValueError(f"GT 파일에 '{col}' 컬럼이 없습니다. ({attack_path})")
    #     if col not in df_pred.columns:
    #         raise ValueError(f"Pred CSV에 '{col}' 컬럼이 없습니다. ({pred_path})")

    # pred 쪽에서 mse 컬럼도 같이 가져올 수 있으면 AUC/MSE 통계용으로 사용
    pred_cols = ["window_index", "is_anomaly"]
    has_mse = "mse" in df_pred.columns
    if has_mse:
        pred_cols.append("mse")

    # # window_index 기준 inner join
    merged = pd.merge(
        df_gt[["window_index", "is_anomaly"]],
        df_pred[pred_cols],
        on="window_index",
        how="inner",
        suffixes=("_true", "_pred"),
    )

    # print(f"[INFO] join 후 행 수: {len(merged)}")

    # # 타입 정리
    merged["is_anomaly_true"] = merged["is_anomaly_true"].astype(int)
    merged["is_anomaly_pred"] = merged["is_anomaly_pred"].astype(int)

    # # is_anomaly_pred == -1 (미라벨) 제거 옵션
    # if args.ignore_pred_minus1:
    #     before = len(merged)
    #     merged = merged[merged["is_anomaly_pred"] != -1].copy()
    #     after = len(merged)
    #     print(f"[INFO] is_anomaly_pred == -1 제거: {before} -> {after}")

    y_true = merged["is_anomaly_true"].to_numpy()
    y_pred = merged["is_anomaly_pred"].to_numpy()

    # # 혼동행렬 요소 계산
    # tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    # tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    # fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    # fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    # total = tp + tn + fp + fn

    # accuracy = safe_div(tp + tn, total)
    # precision = safe_div(tp, tp + fp)
    # recall = safe_div(tp, tp + fn)  # TPR
    # f1 = safe_div(2 * precision * recall, precision + recall)

    # tpr = recall  # same
    # fpr = safe_div(fp, fp + tn)
    # tnr = safe_div(tn, tn + fp)
    # fnr = safe_div(fn, fn + tp)

    # # 추가 지표: prevalence, predicted positive rate, balanced accuracy
    # prevalence = safe_div(tp + fn, total)          # 실제 공격 비율
    # pred_positive_rate = safe_div(tp + fp, total)  # 모델이 공격이라 때린 비율
    # balanced_accuracy = 0.5 * (tpr + tnr)

    # # AUC + MSE 통계 계산 (mse가 있을 때만 시도)
    # auc_dict = {"roc_auc": None, "pr_auc": None}
    mse_stats: Optional[Dict[str, Any]] = None

    if has_mse:
        scores = merged["mse"].to_numpy(dtype=float)
    #     auc_dict = try_compute_auc(y_true, scores)
        mse_stats = compute_mse_stats(y_true, scores)

    #     # 콘솔에도 간단 요약 찍어주기
    #     print("===== MSE Stats by Class =====")
    #     atk = mse_stats["attack"]
    #     nor = mse_stats["normal"]
    #     if atk["count"] > 0:
    #         print(
    #             f"[ATTACK] count={atk['count']}, "
    #             f"mean={atk['mean']:.6f}, std={atk['std']:.6f}, "
    #             f"min={atk['min']:.6f}, max={atk['max']:.6f}"
    #         )
    #     else:
    #         print("[ATTACK] count=0")

    #     if nor["count"] > 0:
    #         print(
    #             f"[NORMAL] count={nor['count']}, "
    #             f"mean={nor['mean']:.6f}, std={nor['std']:.6f}, "
    #             f"min={nor['min']:.6f}, max={nor['max']:.6f}"
    #         )
    #     else:
    #         print("[NORMAL] count=0")
    # else:
    #     print("[INFO] pred CSV에 'mse' 컬럼이 없어 ROC-AUC / PR-AUC / MSE 통계 계산을 생략합니다.")

    # metrics: Dict[str, Any] = {
    #     "num_samples": int(total),
    #     "confusion_matrix": {
    #         "TP": tp,
    #         "TN": tn,
    #         "FP": fp,
    #         "FN": fn,
    #     },
    #     "accuracy": accuracy,
    #     "precision": precision,
    #     "recall": recall,
    #     "f1": f1,
    #     "tpr": tpr,
    #     "fpr": fpr,
    #     "tnr": tnr,
    #     "fnr": fnr,
    #     "prevalence": prevalence,
    #     "pred_positive_rate": pred_positive_rate,
    #     "balanced_accuracy": balanced_accuracy,
    #     "roc_auc": auc_dict["roc_auc"],
    #     "pr_auc": auc_dict["pr_auc"],
    # }

    # print("===== Detection Metrics =====")
    # print(f"Samples (windows)       : {total}")
    # print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    # print(f"Accuracy                : {accuracy:.4f}")
    # print(f"Precision               : {precision:.4f}")
    # print(f"Recall (TPR)            : {recall:.4f}")
    # print(f"F1-score                : {f1:.4f}")
    # print(f"FPR                     : {fpr:.6f}")
    # print(f"TNR (Specificity)       : {tnr:.4f}")
    # print(f"FNR                     : {fnr:.4f}")
    # print(f"Prevalence (Attack rate): {prevalence:.6f}")
    # print(f"Pred Positive Rate      : {pred_positive_rate:.6f}")
    # print(f"Balanced Accuracy       : {balanced_accuracy:.4f}")
    # if metrics["roc_auc"] is not None:
    #     print(f"ROC-AUC                 : {metrics['roc_auc']:.4f}")
    # if metrics["pr_auc"] is not None:
    #     print(f"PR-AUC                  : {metrics['pr_auc']:.4f}")

    # # 메인 metrics JSON 저장
    # out_path.parent.mkdir(parents=True, exist_ok=True)
    # with out_path.open("w", encoding="utf-8") as f:
    #     json.dump(metrics, f, indent=2, ensure_ascii=False)
    # print(f"[INFO] 성능 지표 JSON 저장 완료 → {out_path}")

    # MSE 통계 JSON 별도 저장
    if mse_stats is not None and mse_out_path is not None:
        mse_out_path.parent.mkdir(parents=True, exist_ok=True)
        with mse_out_path.open("w", encoding="utf-8") as f:
            json.dump(mse_stats, f, indent=2, ensure_ascii=False)
        print(f"[INFO] MSE 통계 JSON 저장 완료 → {mse_out_path}")


if __name__ == "__main__":
    main()

"""
python eval_detection_metrics.py \
  --attack-csv ../result/attack_result.csv \
  --pred-csv ../result/benchmark/window_scores.csv \
  --ignore-pred-minus1
"""