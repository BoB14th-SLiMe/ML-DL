# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Basic utils
def to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return None
            return int(float(s))
        return int(x)
    except Exception:
        return None


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return None
            return float(s)
        return float(x)
    except Exception:
        return None


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError("PyYAML이 필요합니다. 설치: pip install pyyaml") from e

    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj if isinstance(obj, dict) else {}


def resolve_path(p: Optional[str], base_dir: Path) -> Optional[Path]:
    if not p:
        return None
    pp = Path(str(p))
    return pp if pp.is_absolute() else (base_dir / pp).resolve()


def merge_cfg(defaults: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(defaults) if isinstance(defaults, dict) else {}
    if isinstance(run, dict):
        out.update(run)
    return out


# IO helpers
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    return rows


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# GT handling
def load_gt_map(gt_path: Path) -> Dict[int, int]:
    suffix = gt_path.suffix.lower()
    if suffix == ".csv":
        rows = read_csv_rows(gt_path)
    else:
        rows = list(iter_jsonl(gt_path))

    gt_map: Dict[int, int] = {}

    for r in rows:
        wi = r.get("window_index", None)
        if wi is None:
            wi = r.get("window_id", None)

        wi_i = to_int(wi)
        if wi_i is None:
            continue

        is_anom = r.get("is_anomaly", None)
        if is_anom is None and "pattern" in r:
            is_anom = 1 if "ATTACK" in str(r.get("pattern", "")).strip().upper() else 0

        ia = to_int(is_anom)
        ia = 1 if (ia is not None and ia > 0) else 0

        prev = gt_map.get(wi_i, 0)
        gt_map[wi_i] = max(prev, ia)

    return gt_map


# Core math
def build_windows(
    input_jsonl: Path,
    feature_keys: List[str],
    window_size: int,
    pad_value: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    feature_index = {k: i for i, k in enumerate(feature_keys)}
    D = len(feature_keys)

    windows: List[np.ndarray] = []
    meta_list: List[Dict[str, Any]] = []

    for obj in iter_jsonl(input_jsonl):
        seq = obj.get("sequence_group", [])
        if not isinstance(seq, list) or not seq:
            continue

        X = np.full((window_size, D), pad_value, dtype=np.float32)

        tmax = min(window_size, len(seq))
        for t in range(tmax):
            pkt = seq[t]
            if not isinstance(pkt, dict):
                continue

            for k, v in pkt.items():
                j = feature_index.get(k, None)
                if j is None:
                    continue
                fv = to_float(v)
                if fv is None:
                    continue
                X[t, j] = float(fv)

        wi = obj.get("window_index", obj.get("window_id", len(meta_list)))
        wi_i = to_int(wi)
        if wi_i is None:
            wi_i = len(meta_list)

        meta_list.append(
            {
                "window_index": int(wi_i),
                "pattern": obj.get("pattern", None),
                "valid_len": int(tmax),
            }
        )
        windows.append(X)

    if not windows:
        return np.zeros((0, window_size, D), dtype=np.float32), []

    return np.stack(windows, axis=0), meta_list


def compute_window_mse(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    pad_value: float,
    feature_weights: np.ndarray,
) -> np.ndarray:
    not_pad = np.any(X_true != pad_value, axis=-1)  # (N,T)
    mask = not_pad.astype(np.float32)

    se = (X_pred - X_true) ** 2  # (N,T,D)
    se = se * feature_weights[np.newaxis, np.newaxis, :]
    se = np.mean(se, axis=-1)  # (N,T)

    se = se * mask
    denom = np.sum(mask, axis=-1) + 1e-8
    return np.sum(se, axis=-1) / denom


def mse_stats_by_label(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    def summary(arr: np.ndarray) -> Dict[str, Any]:
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            out = {"count": 0, "mean": None, "std": None, "min": None, "max": None}
            for p in range(5, 100, 5):
                out[f"p{p}"] = None
            out["p99"] = None
            return out

        out = {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
        for p in range(5, 100, 5):
            out[f"p{p}"] = float(np.percentile(arr, p))
        out["p99"] = float(np.percentile(arr, 99))
        return out

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    a = scores[y_true == 1]
    n = scores[y_true == 0]

    return {
        "meta": {
            "label_col": "is_anomaly_true",
            "score_col": "mse",
            "n_total": int(len(scores)),
            "n_attack": int((y_true == 1).sum()),
            "n_normal": int((y_true == 0).sum()),
        },
        "attack": summary(a),
        "normal": summary(n),
    }


# ROC helpers (optional sklearn)
def try_sklearn_auc(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"roc_auc": None, "pr_auc": None}
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        return out

    if len(np.unique(y_true)) < 2:
        return out

    try:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    except Exception:
        pass
    return out


def save_roc_curve(y_true: np.ndarray, scores: np.ndarray, out_png: Path, out_csv: Path) -> None:
    try:
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        return

    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, thr = roc_curve(y_true, scores)
    roc_auc = float(auc(fpr, tpr))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=150)
    plt.close()

    rows = [{"fpr": float(a), "tpr": float(b), "threshold": float(c)} for a, b, c in zip(fpr, tpr, thr)]
    write_csv(out_csv, ["fpr", "tpr", "threshold"], rows)


# Plot
def moving_average_center(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return y.astype(float)
    y = np.asarray(y, dtype=float)
    k = int(w)
    left = k // 2
    right = k - 1 - left
    yp = np.pad(y, (left, right), mode="edge")
    kernel = np.ones(k, dtype=float) / float(k)
    return np.convolve(yp, kernel, mode="valid")


def contiguous_segments(xs: np.ndarray, mask: np.ndarray) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    start = None
    for i in range(len(xs)):
        if mask[i] and start is None:
            start = int(xs[i])
        if (not mask[i]) and start is not None:
            segs.append((start, int(xs[i - 1])))
            start = None
    if start is not None:
        segs.append((start, int(xs[-1])))
    return segs


def save_recon_plot(
    meta_list: List[Dict[str, Any]],
    mse: np.ndarray,
    out_png: Path,
    threshold: Optional[float],
    gt_map: Optional[Dict[int, int]],
    smooth_window: int,
    plot_points: str,
    point_stride: int,
) -> None:
    xs = np.array([int(m.get("window_index", i)) for i, m in enumerate(meta_list)], dtype=int)
    ys = np.asarray(mse, dtype=float)

    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)

    attack_patch = None
    y_true_sorted = None
    if gt_map is not None:
        y_true_sorted = np.array([gt_map.get(int(x), -1) for x in xs], dtype=int)
        attack_mask = y_true_sorted == 1
        for x0, x1 in contiguous_segments(xs, attack_mask):
            ax.axvspan(x0, x1, alpha=0.10)
        attack_patch = Patch(alpha=0.10, label="Attack region (GT=1)")

    ax.plot(xs, ys, linewidth=0.8, color="0.5", label="MSE")

    if smooth_window and smooth_window > 1:
        ys_s = moving_average_center(ys, smooth_window)
        ax.plot(xs, ys_s, linewidth=2.0, color="0.15", label=f"Smoothed (w={smooth_window})")

    if threshold is not None:
        ax.axhline(float(threshold), linestyle="--", linewidth=1.2, color="0.2",
                   label=f"threshold={threshold:.6g}")

    if plot_points != "none":
        idx = np.arange(len(xs))
        if plot_points == "exceed" and threshold is not None:
            idx = idx[ys > float(threshold)]
        if point_stride and point_stride > 1:
            idx = idx[::point_stride]

        if idx.size > 0:
            if y_true_sorted is None:
                ax.scatter(xs[idx], ys[idx], s=10, alpha=0.6, label="points")
            else:
                idx_n = idx[y_true_sorted[idx] == 0]
                idx_a = idx[y_true_sorted[idx] == 1]
                if idx_n.size > 0:
                    ax.scatter(xs[idx_n], ys[idx_n], s=10, alpha=0.7, label="Normal points")
                if idx_a.size > 0:
                    ax.scatter(xs[idx_a], ys[idx_a], s=10, alpha=0.7, label="Attack points")

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


# Evaluation
def evaluate_predictions(
    pred_map: Dict[int, Tuple[float, int]],
    gt_map: Dict[int, int],
    threshold: Optional[float],
    ignore_pred_minus1: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    y_true_list: List[int] = []
    y_pred_list: List[int] = []
    scores_list: List[float] = []

    for wi, gt in gt_map.items():
        if wi not in pred_map:
            continue
        mse, ypred = pred_map[wi]
        if ignore_pred_minus1 and int(ypred) == -1:
            continue
        y_true_list.append(int(gt))
        y_pred_list.append(int(ypred))
        scores_list.append(float(mse))

    if not y_true_list:
        metrics = {
            "num_samples": 0,
            "threshold_used": threshold,
            "warning": "No joined samples after filtering",
        }
        mse_stats = {
            "meta": {"label_col": "is_anomaly_true", "score_col": "mse", "n_total": 0, "n_attack": 0, "n_normal": 0},
            "attack": {},
            "normal": {},
        }
        return metrics, mse_stats, None, None

    y_true = np.asarray(y_true_list, dtype=int)
    y_pred = np.asarray(y_pred_list, dtype=int)
    scores = np.asarray(scores_list, dtype=float)

    y_pred_bin = np.where(y_pred <= 0, 0, 1)

    tp = int(np.sum((y_true == 1) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_bin == 0)))
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    prevalence = (tp + fn) / total if total else 0.0
    pred_positive_rate = (tp + fp) / total if total else 0.0
    balanced_accuracy = 0.5 * (recall + tnr)

    aucs = try_sklearn_auc(y_true, scores)
    mse_stats = mse_stats_by_label(y_true, scores)

    metrics = {
        "num_samples": int(total),
        "threshold_used": threshold,
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tpr": float(recall),
        "fpr": float(fpr),
        "tnr": float(tnr),
        "fnr": float(fnr),
        "prevalence": float(prevalence),
        "pred_positive_rate": float(pred_positive_rate),
        "balanced_accuracy": float(balanced_accuracy),
        "roc_auc": aucs["roc_auc"],
        "pr_auc": aucs["pr_auc"],
    }

    return metrics, mse_stats, y_true, scores
