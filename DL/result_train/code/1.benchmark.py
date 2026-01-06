#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_and_eval.py

윈도우 단위 JSONL + LSTM-AE 모델 → 윈도우별 MSE 계산 + (선택) GT와 비교한
탐지 성능 지표 / MSE 분포까지 한 번에 계산해 주는 통합 스크립트.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
from matplotlib.patches import Patch
matplotlib.use("Agg")  # GUI 없이 저장용
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="윈도우 JSONL + LSTM-AE 모델 → MSE 계산 + 탐지 성능 / MSE 통계까지 한 번에 수행"
    )

    # 윈도우 JSONL + 전처리 (전처리 디렉토리는 현재 버전에서는 이름만 받음)
    p.add_argument(
        "--input",
        "-i",
        required=True,
        help="윈도우 단위 JSONL 경로 (각 line = 1 window, sequence_group 포함)",
    )
    p.add_argument(
        "--pre-dir",
        "-p",
        required=False,
        default=None,
        help="(호환용) 전처리 파라미터 디렉토리. 현재 버전에서는 사용하지 않음.",
    )
    p.add_argument(
        "--window-size",
        "-w",
        type=int,
        default=80,
        help="윈도우 길이(T, time steps). sequence_group 길이가 다르면 pad/truncate",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="출력 디렉토리 (metrics_*.json, analyze_mse_dist_*.json 저장 위치)",
    )
    p.add_argument(
        "--no-pad-last",
        action="store_true",
        help="(이전 버전 호환용) 현재는 사용하지 않음. 항상 pad 해서 고정 길이 윈도우 생성.",
    )

    # DL 모델 / inference 관련 옵션
    p.add_argument(
        "--model-dir",
        "-m",
        required=True,
        help="train_lstm_ae_windows_keras.py 결과 디렉토리 (model.h5, config.json, feature_keys.txt 등)",
    )
    p.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=128,
        help="DL 모델 inference batch size (기본=128)",
    )
    p.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help=(
            "윈도우 MSE anomaly threshold "
            "(미지정 시 model_dir/threshold.json 값을 사용, 둘 다 없으면 is_anomaly=-1)"
        ),
    )
    p.add_argument(
        "--tag",
        default=None,
        help="출력 파일 이름에 붙일 태그 (기본: 입력 JSONL 파일명 stem)",
    )

    # 평가용 GT / 출력 경로
    p.add_argument(
        "--attack-csv",
        "-a",
        default=None,
        help="(선택) 실제 공격 여부가 들어있는 파일 (CSV 또는 JSONL; 예: attack_result_XXX.csv, attack_ver2_window.jsonl)",
    )
    p.add_argument(
        "--metrics-json",
        default=None,
        help=(
            "(선택) 탐지 성능 지표를 저장할 JSON 경로 "
            "(미지정 시 output-dir/metrics_{tag}.json 으로 저장)"
        ),
    )
    p.add_argument(
        "--mse-stats-json",
        default=None,
        help=(
            "(선택) MSE 통계를 별도로 저장할 JSON 경로 "
            "(미지정 시 output-dir/analyze_mse_dist_{tag}.json 으로 저장)"
        ),
    )
    p.add_argument(
        "--ignore-pred-minus1",
        action="store_true",
        help="is_anomaly_pred == -1 인 윈도우는 평가/통계에서 제외 (threshold 안 쓴 경우 등)",
    )

    # ⭐ 추가: feature weight 파일 직접 지정 (옵션)
    p.add_argument(
        "--feature-weights-file",
        type=str,
        default=None,
        help=(
            "train 때 사용한 feature_weights.txt 경로. "
            "지정 안 하면 config.feature_weights_file 기반으로 자동 탐색. "
            "둘 다 없으면 균일 가중치(1.0) 사용."
        ),
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
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
    result: Dict[str, Optional[float]] = {"roc_auc": None, "pr_auc": None}
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
    - 기본 통계: count, mean, std, min, max
    - 퍼센타일: p5, p10, ..., p95 + p99
    """

    def _summary(arr: np.ndarray) -> Dict[str, Any]:
        arr = np.asarray(arr, dtype=float)
        # 빈 배열이면 None으로 채우기
        if arr.size == 0:
            base = {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
            for p in range(5, 100, 5):  # p5, p10, ..., p95
                base[f"p{p}"] = None
            base["p99"] = None
            return base

        base = {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
        # 5% 단위 percentile (5, 10, ..., 95)
        for p in range(5, 100, 5):
            base[f"p{p}"] = float(np.percentile(arr, p))
        # 추가로 p99 유지
        base["p99"] = float(np.percentile(arr, 99))
        return base

    attack_mask = y_true == 1
    normal_mask = y_true == 0

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

def build_window_label_map(df_gt: pd.DataFrame) -> Dict[int, int]:
    tmp = df_gt.copy()

    tmp["window_index"] = pd.to_numeric(tmp["window_index"], errors="coerce")
    tmp = tmp.dropna(subset=["window_index"])
    tmp["window_index"] = tmp["window_index"].astype(int)

    tmp["is_anomaly"] = pd.to_numeric(tmp["is_anomaly"], errors="coerce").fillna(0).astype(int)

    # window_index 중복이 있으면 공격(1)을 우선(최대값)
    s = tmp.groupby("window_index")["is_anomaly"].max()
    return s.to_dict()

# ---------------------------------------------------------------------
# DL model loading (LSTM-AE with repeat_latent)
# ---------------------------------------------------------------------
def load_model_from_dir(model_dir: Path):
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"❌ config.json 없음: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"[INFO] config 로드: {config_path}")
    T = config.get("T")
    D = config.get("D")
    print(f"[INFO] 학습 시 T={T}, D={D}")

    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.models import load_model

    model_path = model_dir / "model.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"❌ model.h5 없음: {model_path}")

    print(f"[INFO] model 로드: {model_path}")

    # decoder 에서 latent 벡터를 T 길이로 반복하는 커스텀 레이어 함수
    T_for_repeat = T

    def repeat_latent(x):
        x = tf.expand_dims(x, axis=1)         # (B, 1, latent_dim)
        x = tf.tile(x, [1, T_for_repeat, 1])  # (B, T, latent_dim)
        return x

    custom_objects = {
        "repeat_latent": repeat_latent,
    }

    model = load_model(
        model_path,
        compile=False,
        custom_objects=custom_objects,
    )

    # threshold.json 읽기
    thresh_path = model_dir / "threshold.json"
    threshold = None
    if thresh_path.exists():
        try:
            with thresh_path.open("r", encoding="utf-8") as f:
                th_cfg = json.load(f)

            if "threshold" in th_cfg:
                threshold = float(th_cfg["threshold"])
                print(f"[INFO] threshold.json(threshold) 사용: {threshold}")
            elif "threshold_p99" in th_cfg:
                threshold = float(th_cfg["threshold_p99"])
                print(f"[INFO] threshold_p99 사용: {threshold}")
            elif "threshold_mu3" in th_cfg:
                threshold = float(th_cfg["threshold_mu3"])
                print(f"[INFO] threshold_mu3 사용: {threshold}")
            else:
                print(
                    "[INFO] threshold.json 안에 사용할 키가 없습니다. "
                    "(threshold / threshold_p99 / threshold_mu3 없음)"
                )
        except Exception as e:
            print(f"[WARN] threshold.json 로딩 실패: {e}")

    # feature_keys.txt (학습에 사용했던 feature 순서)
    feat_path = model_dir / "feature_keys.txt"
    if not feat_path.exists():
        raise FileNotFoundError(f"❌ feature_keys.txt 없음: {feat_path}")

    with feat_path.open("r", encoding="utf-8") as f:
        raw_keys = [line.strip() for line in f if line.strip()]

    # 혹시 "0 protocol_norm" 같은 형식일 수도 있으니, 마지막 토큰만 feature 이름으로 사용
    feature_keys = [rk.split()[-1] for rk in raw_keys]

    print(f"[INFO] feature_keys.txt 로드, 길이 = {len(feature_keys)}")
    print("[DEBUG] feature_keys[:20] =", feature_keys[:20])

    pad_value = float(config.get("pad_value", 0.0))
    print(f"[INFO] config pad_value = {pad_value}")

    return model, config, feature_keys, threshold, pad_value


# ---------------------------------------------------------------------
# Feature weights 로딩 (train 스크립트와 일관되게)
# ---------------------------------------------------------------------
def load_feature_weights(
    config: Dict[str, Any],
    feature_keys: List[str],
    model_dir: Path,
    cli_path: Optional[str] = None,
) -> np.ndarray:
    """
    feature_weights.txt 를 찾아서 feature 순서에 맞는 weight 벡터(D,) 생성.
    우선순위:
      1) --feature-weights-file (CLI에서 직접 지정)
      2) config["feature_weights_file"] 경로 (여러 기준으로 추정)
      3) 아무것도 못 찾으면 전체 1.0
    """
    feature_weights = np.ones(len(feature_keys), dtype=np.float32)

    # 1) 후보 경로들 모으기
    candidates: List[Path] = []

    # (1) CLI로 직접 지정된 경로
    if cli_path:
        candidates.append(Path(cli_path))

    # (2) config 에 저장된 경로
    fw_cfg = config.get("feature_weights_file")
    if fw_cfg:
        p_cfg = Path(fw_cfg)
        candidates.append(p_cfg)

        # 상대경로일 경우 몇 가지 heuristic 시도
        if not p_cfg.is_absolute():
            # 2-1) model_dir 기준
            candidates.append((model_dir / p_cfg).resolve())

            # 2-2) input_jsonl 기준으로 train root 추정 → train/data/feature_weights.txt
            in_path = config.get("input_jsonl")
            if in_path:
                in_path = Path(in_path)
                train_root = in_path.parent.parent  # .../train/result → .. → .../train
                candidates.append((train_root / "data" / p_cfg.name).resolve())

    # 2) 실제 존재하는 첫 번째 경로 선택
    fw_path: Optional[Path] = None
    for c in candidates:
        try:
            if c.exists():
                fw_path = c
                break
        except Exception:
            continue

    if fw_path is None:
        print("[INFO] feature_weights 파일을 찾지 못했습니다. → 모든 feature 가중치 1.0 사용")
        return feature_weights

    print(f"[INFO] feature_weights 파일 사용: {fw_path}")

    # 3) 파일 파싱 (train 스크립트와 동일한 형식: "feature_name weight")
    weight_map: Dict[str, float] = {}
    try:
        with fw_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    print(f"[WARN] 잘못된 weight 라인(무시): {line}")
                    continue
                name = parts[0]
                try:
                    w = float(parts[1])
                except ValueError:
                    print(f"[WARN] weight 파싱 실패(무시): {line}")
                    continue
                weight_map[name] = w
    except Exception as e:
        print(f"[WARN] feature_weights 파일 로딩 중 오류: {e}")
        return feature_weights

    # 4) feature_keys 순서에 맞춰 벡터 채우기
    for i, k in enumerate(feature_keys):
        if k in weight_map:
            feature_weights[i] = weight_map[k]

    print("[INFO] feature-wise weights (앞 10개):")
    for k, w in list(zip(feature_keys, feature_weights))[:10]:
        print(f"  - {k:25s}: {w}")

    return feature_weights


# ---------------------------------------------------------------------
# Window reconstruction error (train 코드와 동일)
# ---------------------------------------------------------------------
def compute_window_errors(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    pad_value: float,
    feature_weights: np.ndarray | None = None,
) -> np.ndarray:
    # 패딩이 아닌 timestep 마스크 (N, T)
    not_pad = np.any(np.not_equal(X_true, pad_value), axis=-1)
    mask = not_pad.astype(np.float32)

    # 타임스텝별 SE (N, T, D)
    se = (X_pred - X_true) ** 2  # (N, T, D)

    if feature_weights is not None:
        se = se * feature_weights[np.newaxis, np.newaxis, :]  # (N, T, D)

    # feature 평균 → (N, T)
    se = np.mean(se, axis=-1)  # (N, T)

    se_masked = se * mask

    denom = np.sum(mask, axis=-1) + 1e-8
    errors = np.sum(se_masked, axis=-1) / denom
    return errors

def _contiguous_segments(xs: np.ndarray, mask: np.ndarray):
    """
    xs는 정렬된 x축 값(정수), mask는 동일 길이 bool.
    True가 연속되는 구간을 [(x_start, x_end)]로 반환 (x_end 포함).
    """
    segs = []
    start = None
    for i in range(len(xs)):
        if mask[i] and start is None:
            start = xs[i]
        if (not mask[i]) and start is not None:
            segs.append((start, xs[i - 1]))
            start = None
    if start is not None:
        segs.append((start, xs[-1]))
    return segs


def save_recon_error_plot(
    meta_list: List[Dict[str, Any]],
    mse_per_window: np.ndarray,
    out_png: Path,
    threshold: Optional[float] = None,
    y_true_for_plot: Optional[np.ndarray] = None,
    smooth_window: int = 31,          # ✅ 스무딩 윈도우 (눈 피로 줄임)
    plot_points: str = "exceed",      # "none" | "exceed" | "all"
    point_stride: int = 10,           # ✅ 점을 찍더라도 듬성듬성
):
    # x축: window_index가 있으면 사용, 없으면 0..N-1
    xs = []
    for i, m in enumerate(meta_list):
        try:
            xs.append(int(m.get("window_index", i)))
        except Exception:
            xs.append(i)

    xs_arr = np.asarray(xs, dtype=int)
    ys = np.asarray(mse_per_window, dtype=float)

    # window_index가 섞였을 수 있으니 정렬
    order = np.argsort(xs_arr)
    xs_arr = xs_arr[order]
    ys = ys[order]

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)

    # 1) GT가 있으면 공격 구간을 "배경 음영"으로 표시 (점 색칠보다 훨씬 덜 피곤)
    if y_true_for_plot is not None:
        y_true_sorted = np.asarray(y_true_for_plot, dtype=int)[order]
        attack_mask = (y_true_sorted == 1)

        # 공격 구간 음영
        for (x0, x1) in _contiguous_segments(xs_arr, attack_mask):
            ax.axvspan(x0, x1, alpha=0.10)

        # 범례용 패치
        attack_patch = Patch(alpha=0.10, label="Attack region (GT=1)")
    else:
        attack_patch = None

    # 2) MSE 원본 선(얇게, 회색)
    ax.plot(xs_arr, ys, linewidth=0.8, color="0.5", label="MSE")

    # 3) 스무딩 선(굵게) - 추세가 훨씬 잘 보임
    if smooth_window and smooth_window > 1:
        ys_smooth = pd.Series(ys).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()
        ax.plot(xs_arr, ys_smooth, linewidth=2.0, color="0.15", label=f"Smoothed (w={smooth_window})")

    # 4) threshold
    if threshold is not None:
        ax.axhline(float(threshold), linestyle="--", linewidth=1.2, color="0.2",
                   label=f"threshold={threshold:.6g}")

    # 5) 점은 최소화: (기본) 임계값 초과점만 + stride로 듬성듬성
    if plot_points != "none":
        idx = np.arange(len(xs_arr))

        if plot_points == "exceed" and threshold is not None:
            idx = idx[ys > float(threshold)]

        if point_stride and point_stride > 1:
            idx = idx[::point_stride]

        if idx.size > 0:
            if y_true_for_plot is None:
                ax.scatter(xs_arr[idx], ys[idx], s=10, alpha=0.6, label="points")
            else:
                y_true_sorted = np.asarray(y_true_for_plot, dtype=int)[order]
                idx_n = idx[y_true_sorted[idx] == 0]
                idx_a = idx[y_true_sorted[idx] == 1]
                if idx_n.size > 0:
                    ax.scatter(xs_arr[idx_n], ys[idx_n], s=10, alpha=0.7, label="Normal points")
                if idx_a.size > 0:
                    ax.scatter(xs_arr[idx_a], ys[idx_a], s=10, alpha=0.7, label="Attack points")

    ax.set_xlabel("window_index")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction Error (Attack region shaded)")
    ax.grid(True, alpha=0.15)

    handles, labels = ax.get_legend_handles_labels()
    if attack_patch is not None:
        handles = [attack_patch] + handles
        labels = [attack_patch.get_label()] + labels
    ax.legend(handles, labels, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png))
    plt.close(fig)

    print(f"[INFO] recon error plot 저장 → {out_png}")


def save_roc_curve_and_points(
    y_true: np.ndarray,
    scores: np.ndarray,
    out_png: Path,
    out_csv: Optional[Path] = None,
) -> Optional[float]:
    """
    y_true: 0/1
    scores: 클수록 공격일 가능성 높은 score (여기서는 mse)
    """
    try:
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        print("[WARN] scikit-learn 미설치 → ROC curve 저장 생략")
        return None

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    if len(np.unique(y_true)) < 2:
        print("[WARN] y_true 클래스가 하나뿐 → ROC curve 계산/저장 불가")
        return None

    fpr, tpr, thr = roc_curve(y_true, scores)
    roc_auc = float(auc(fpr, tpr))

    # PNG
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=150)
    plt.close()
    print(f"[INFO] ROC curve 저장 → {out_png}")

    # 점 데이터 CSV (옵션)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[INFO] ROC points CSV 저장 → {out_csv}")

    return roc_auc

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag if args.tag is not None else input_path.stem
    window_size = args.window_size

    print(f"[INFO] 입력 JSONL : {input_path}")
    print(f"[INFO] 출력 디렉토리 : {output_dir}")
    print(f"[INFO] 사용 태그(tag) = {tag}")
    print(f"[INFO] window_size = {window_size}")
    if args.pre_dir:
        print(f"[INFO] (참고용) pre-dir = {args.pre_dir}")

    # -----------------------------
    # 1) DL 모델 / feature_keys 로드
    # -----------------------------
    model_dir = Path(args.model_dir)
    print(f"[INFO] DL 모델 디렉토리: {model_dir}")

    model, config, feature_keys, threshold_from_file, pad_value = load_model_from_dir(model_dir)

    # 🔥 feature_weights 로드 (config + CLI 기반)
    feature_weights = load_feature_weights(
        config=config,
        feature_keys=feature_keys,
        model_dir=model_dir,
        cli_path=args.feature_weights_file,
    )

    # threshold 결정: CLI > threshold.json > None
    threshold = args.threshold
    if threshold is None and threshold_from_file is not None:
        threshold = threshold_from_file
        print(f"[INFO] threshold.json의 값을 사용: threshold={threshold}")
    elif threshold is not None:
        print(f"[INFO] CLI로 전달된 threshold 사용: threshold={threshold}")
    else:
        print("[INFO] threshold 미지정 → is_anomaly_pred = -1 로만 기록 (ROC-AUC, MSE 통계는 가능)")

    # -----------------------------
    # 2) 입력 JSONL → [N, T, D] 윈도우 행렬 생성
    # -----------------------------
    all_windows: List[np.ndarray] = []
    meta_list: List[Dict[str, Any]] = []

    D_model: int = len(feature_keys)
    print(f"[INFO] feature dimension (D_model) = {D_model}")

    with input_path.open("r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] JSON 파싱 실패 (line {line_idx}): {e}")
                continue

            seq_group = obj.get("sequence_group", [])
            if not seq_group:
                continue

            # 윈도우 행렬 초기화
            X_win = np.full((window_size, D_model), pad_value, dtype=np.float32)

            for t, pkt in enumerate(seq_group):
                if t >= window_size:
                    break
                for j, feat_name in enumerate(feature_keys):
                    v = pkt.get(feat_name, pad_value)
                    try:
                        X_win[t, j] = float(v)
                    except Exception:
                        X_win[t, j] = pad_value

            all_windows.append(X_win)
            meta_list.append(
                {
                    "window_index": int(obj.get("window_index", len(meta_list))),
                    "pattern": obj.get("pattern", None),
                    "valid_len": int(min(len(seq_group), window_size)),
                }
            )

    num_windows = len(all_windows)
    print(f"[INFO] 생성된 윈도우 수 = {num_windows}")

    if num_windows == 0:
        print("[WARN] 생성된 윈도우가 없습니다. 입력/파라미터를 확인하세요.")
        return

    X_windows = np.stack(all_windows, axis=0)  # [N, T, D]
    print("[DEBUG] X_windows shape:", X_windows.shape)
    print(
        "[DEBUG] X_windows 전체 통계:",
        "min=",
        float(X_windows.min()),
        "max=",
        float(X_windows.max()),
        "mean=",
        float(X_windows.mean()),
    )

    # -----------------------------
    # 3) DL 모델 inference
    # -----------------------------
    N, T_cur, D_cur = X_windows.shape
    print(f"[INFO] 모델 입력용 X_windows shape: (N={N}, T={T_cur}, D={D_cur})")

    T_cfg = config.get("T")
    D_cfg = config.get("D")
    if T_cfg is not None and T_cfg != T_cur:
        print(f"[WARN] config.T({T_cfg}) != 현재 window_size({T_cur})")
    if D_cfg is not None and D_cfg != D_cur:
        print(f"[WARN] config.D({D_cfg}) != 현재 D({D_cur})")

    # 디바이스 정보 (옵션)
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"[INFO] 사용 가능한 GPU: {gpus}")
        else:
            print("[INFO] GPU 없음, CPU 사용")
    except Exception:
        pass

    print("[INFO] DL 모델로 윈도우별 reconstruction 예측 중...")
    recon = model.predict(X_windows, batch_size=args.batch_size, verbose=1)

    print(
        "[DEBUG] recon 전체 통계:",
        "min=",
        float(recon.min()),
        "max=",
        float(recon.max()),
        "mean=",
        float(recon.mean()),
    )

    # 🔥 가중치까지 포함한 window-level MSE 계산
    mse_per_window = compute_window_errors(
        X_windows,
        recon,
        pad_value,
        feature_weights=feature_weights,
    )
    print("[DEBUG] mse_per_window[:20] =", mse_per_window[:20])
    print(
        "[DEBUG] mse_per_window 통계: "
        f"min={float(mse_per_window.min())}, "
        f"max={float(mse_per_window.max())}, "
        f"mean={float(mse_per_window.mean())}"
    )

    print(
        "[INFO] benchmark 윈도우 MSE 통계: "
        f"mean={mse_per_window.mean():.4f}, "
        f"std={mse_per_window.std():.4f}, "
        f"min={mse_per_window.min():.4f}, "
        f"max={mse_per_window.max():.4f}"
    )

    # -----------------------------
    # 3.5) 재구성 오차 시계열 그래프 (정상=파랑, 공격=빨강)
    # -----------------------------
    y_true_for_plot = None

    if args.attack_csv is not None:
        attack_path = Path(args.attack_csv)
        df_gt = load_gt_table(attack_path)

        # pattern 기반 is_anomaly 생성 (없을 때 자동 처리)
        if "is_anomaly" not in df_gt.columns and "pattern" in df_gt.columns:
            df_gt["is_anomaly"] = df_gt["pattern"].apply(
                lambda x: 1 if "ATTACK" in str(x).strip().upper() else 0
            )

        # 필수 컬럼 체크
        if "window_index" in df_gt.columns and "is_anomaly" in df_gt.columns:
            label_map = build_window_label_map(df_gt)
            y_true_for_plot = np.array(
                [label_map.get(int(m.get("window_index", -1)), -1) for m in meta_list],
                dtype=int
            )
        else:
            print("[WARN] GT에 window_index/is_anomaly가 없어 색칠용 라벨 생성 불가")

    recon_png = output_dir / f"recon_error_{tag}.png"
    save_recon_error_plot(
        meta_list,
        mse_per_window,
        recon_png,
        threshold=threshold,
        y_true_for_plot=y_true_for_plot,   # ✅ 추가
    )

    # 원하면 per-window MSE도 파일로 저장(분석/디버깅용)
    mse_csv = output_dir / f"mse_per_window_{tag}.csv"
    rows = []
    for m, mse in zip(meta_list, mse_per_window):
        wi = int(m.get("window_index", -1))
        rows.append({
            "window_index": wi,
            "mse": float(mse),
            "pattern": m.get("pattern", None),
            "valid_len": int(m.get("valid_len", 0)),
            "is_anomaly_pred": (int(float(mse) > float(threshold)) if threshold is not None else -1),
        })
    pd.DataFrame(rows).to_csv(mse_csv, index=False, encoding="utf-8")
    print(f"[INFO] mse_per_window CSV 저장 → {mse_csv}")


    # -----------------------------
    # 4) (선택) GT와 비교하여 성능 계산
    # -----------------------------
    if args.attack_csv is None:
        print("[INFO] --attack-csv 가 지정되지 않았습니다. "
              "→ GT 기반 탐지 성능 / MSE 통계는 계산하지 않습니다.")
        return

    attack_path = Path(args.attack_csv)
    print(f"[INFO] GT 파일 (attack)          : {attack_path}")

    # metrics / mse-stats 출력 경로
    metrics_out_path = (
        Path(args.metrics_json)
        if args.metrics_json is not None
        else output_dir / f"metrics_{tag}.json"
    )
    mse_out_path = (
        Path(args.mse_stats_json)
        if args.mse_stats_json is not None
        else output_dir / f"analyze_mse_dist_{tag}.json"
    )
    print(f"[INFO] 출력 JSON (metrics)       : {metrics_out_path}")
    print(f"[INFO] 출력 JSON (MSE stats)    : {mse_out_path}")

    # GT 로드
    df_gt = load_gt_table(attack_path)

    # pattern 기반 is_anomaly 생성 (없을 때 자동 처리)
    if "is_anomaly" not in df_gt.columns and "pattern" in df_gt.columns:
        df_gt["is_anomaly"] = df_gt["pattern"].apply(
            lambda x: 1 if "ATTACK" in str(x).strip().upper() else 0
        )
        print(
            f"[INFO] 'pattern' 컬럼에서 is_anomaly 자동 생성 완료 "
            f"({df_gt['is_anomaly'].sum()}개 공격 윈도우)"
        )

    # 필수 컬럼 체크
    if "window_index" not in df_gt.columns or "is_anomaly" not in df_gt.columns:
        raise ValueError(
            f"GT 파일에 'window_index' 또는 'is_anomaly' 컬럼이 없습니다. ({attack_path})"
        )

    # Pred DataFrame 생성 (이 스크립트에서 바로 만든 MSE / is_anomaly_pred 사용)
    pred_rows: List[Dict[str, Any]] = []
    for m, mse in zip(meta_list, mse_per_window):
        if threshold is not None:
            is_anom = int(mse > threshold)
        else:
            is_anom = -1  # threshold 없으면 라벨 없음
        pred_rows.append(
            {
                "window_index": m["window_index"],
                "mse": float(mse),
                "is_anomaly": is_anom,
            }
        )
    df_pred = pd.DataFrame(pred_rows)

    # window_index 기준 inner join
    has_mse = "mse" in df_pred.columns
    pred_cols = ["window_index", "is_anomaly"]
    if has_mse:
        pred_cols.append("mse")

    merged = pd.merge(
        df_gt[["window_index", "is_anomaly"]],
        df_pred[pred_cols],
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

    if len(merged) == 0:
        print("[WARN] 평가에 사용할 윈도우가 0개입니다. (join 결과 또는 ignore-pred-minus1 영향)")
        return

    y_true = merged["is_anomaly_true"].to_numpy()
    y_pred = merged["is_anomaly_pred"].to_numpy()

    # 혼동행렬 요소 계산 (y_pred 에 -1 이 남아 있으면 0으로 취급)
    y_pred_bin = np.where(y_pred <= 0, 0, 1)

    tp = int(np.sum((y_true == 1) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_bin == 0)))

    total = tp + tn + fp + fn

    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)  # TPR
    f1 = safe_div(2 * precision * recall, precision + recall)

    tpr = recall  # same
    fpr = safe_div(fp, fp + tn)
    tnr = safe_div(tn, tn + fp)
    fnr = safe_div(fn, fn + tp)

    # 추가 지표: prevalence, predicted positive rate, balanced accuracy
    prevalence = safe_div(tp + fn, total)  # 실제 공격 비율
    pred_positive_rate = safe_div(tp + fp, total)  # 모델이 공격이라 때린 비율
    balanced_accuracy = 0.5 * (tpr + tnr)

    # AUC + MSE 통계 계산 (mse가 있을 때만 시도)
    auc_dict: Dict[str, Optional[float]] = {"roc_auc": None, "pr_auc": None}
    mse_stats: Optional[Dict[str, Any]] = None

    if has_mse:
        scores = merged["mse"].to_numpy(dtype=float)
        auc_dict = try_compute_auc(y_true, scores)
        mse_stats = compute_mse_stats(y_true, scores)
        roc_png = output_dir / f"roc_curve_{tag}.png"
        roc_csv = output_dir / f"roc_points_{tag}.csv"
        _ = save_roc_curve_and_points(y_true, scores, roc_png, out_csv=roc_csv)

    else:
        print("[INFO] 'mse' 컬럼이 없어 ROC-AUC / PR-AUC / MSE 통계 계산을 생략합니다.")

    metrics: Dict[str, Any] = {
        "num_samples": int(total),
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
        "prevalence": prevalence,
        "pred_positive_rate": pred_positive_rate,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": auc_dict["roc_auc"],
        "pr_auc": auc_dict["pr_auc"],
    }

    print("===== Detection Metrics =====")
    print(f"Samples (windows)       : {total}")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"Accuracy                : {accuracy:.4f}")
    print(f"Precision               : {precision:.4f}")
    print(f"Recall (TPR)            : {recall:.4f}")
    print(f"F1-score                : {f1:.4f}")
    print(f"FPR                     : {fpr:.6f}")
    print(f"TNR (Specificity)       : {tnr:.4f}")
    print(f"FNR                     : {fnr:.4f}")
    print(f"Prevalence (Attack rate): {prevalence:.6f}")
    print(f"Pred Positive Rate      : {pred_positive_rate:.6f}")
    print(f"Balanced Accuracy       : {balanced_accuracy:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC                 : {metrics['roc_auc']:.4f}")
    if metrics["pr_auc"] is not None:
        print(f"PR-AUC                  : {metrics['pr_auc']:.4f}")

    # metrics JSON 저장
    metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 성능 지표 JSON 저장 완료 → {metrics_out_path}")

    # MSE 통계 JSON 별도 저장
    if mse_stats is not None:
        mse_out_path.parent.mkdir(parents=True, exist_ok=True)
        with mse_out_path.open("w", encoding="utf-8") as f:
            json.dump(mse_stats, f, indent=2, ensure_ascii=False)
        print(f"[INFO] MSE 통계 JSON 저장 완료 → {mse_out_path}")


if __name__ == "__main__":
    main()

"""
python benchmark_and_eval.py \
  --input ../data/attack_windows.jsonl \
  --pre-dir ../preprocessing/result \
  --window-size 80 \
  --output-dir ../result/benchmark \
  --model-dir ../DL/result_train/data \
  --batch-size 128 \
  --attack-csv ../result/attack_result.csv \
  --ignore-pred-minus1
"""

"""
python benchmark_and_eval.py \
  --input ../data/attack_windows.jsonl \
  --window-size 16 \
  --output-dir ../result/benchmark \
  --model-dir ../../result_train/data \
  --batch-size 128 \
  --attack-csv ../result/attack_result.csv \
  --ignore-pred-minus1 \
  --feature-weights-file "../train/data/feature_weights.txt"
"""
