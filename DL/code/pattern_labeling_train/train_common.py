#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


JsonDict = Dict[str, Any]


# Basic utils
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def utcnow_z() -> str:
    return datetime.utcnow().isoformat() + "Z"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        raise RuntimeError("PyYAML이 필요합니다. 설치: pip install pyyaml")
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("YAML root는 dict여야 합니다.")
    return obj


def get(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    return cfg.get(key, default)


def resolve_path(base_dir: Path, v: Any) -> Path:
    if v is None:
        raise ValueError("path is required")
    s = str(v).strip()
    if not s:
        raise ValueError("path is empty")
    p = Path(s).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()


# Data I/O
def load_features(feature_file: Path) -> List[str]:
    feats: List[str] = []
    with feature_file.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                feats.append(s)
    if not feats:
        raise ValueError(f"feature file empty: {feature_file}")
    return feats


def load_sequences_and_labels(jsonl_path: Path, drop_labels: Sequence[str]) -> Tuple[List[Any], List[str]]:
    drop_set = set(map(str, drop_labels))
    sequences: List[Any] = []
    labels: List[str] = []

    total = 0
    dropped = 0
    missing_seq = 0
    missing_label = 0
    key_usage = Counter()

    def _get_label(obj: Dict[str, Any]) -> Any:
        if "pattern" in obj:
            return obj.get("pattern")
        if "label" in obj:
            return obj.get("label")
        return None

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            total += 1
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            if "sequence_group" not in obj:
                missing_seq += 1
                continue

            lb = _get_label(obj)
            if lb is None:
                missing_label += 1
                continue

            used_key = "pattern" if "pattern" in obj else ("label" if "label" in obj else "none")
            key_usage[used_key] += 1

            if lb in drop_set:
                dropped += 1
                continue

            sequences.append(obj.get("sequence_group"))
            labels.append(str(lb))

    print(
        f"[load_data] total={total}, kept={len(labels)}, dropped={dropped}, "
        f"missing_seq={missing_seq}, missing_label={missing_label}, "
        f"drop_labels={sorted(drop_set)}, key_usage={dict(key_usage)}"
    )
    if labels:
        c = Counter(labels)
        print(f"[load_data] label top10: {c.most_common(10)}")
    return sequences, labels


def preprocess_sequences(
    sequences: List[Any],
    max_seq_length: int,
    feature_names: List[str],
    pad_values: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    feat2idx = {k: i for i, k in enumerate(feature_names)}
    D = len(feature_names)
    N = len(sequences)

    out = np.zeros((N, max_seq_length, D), dtype=np.float32)
    lengths = np.ones((N,), dtype=np.int64)

    pad_set = set(float(x) for x in pad_values)

    for i, seq in enumerate(tqdm(sequences, desc="Preprocess", leave=True)):
        if not seq:
            continue

        t_out = 0
        for step in seq:
            if t_out >= max_seq_length:
                break
            if not isinstance(step, dict):
                continue

            has_valid = False
            for k, v in step.items():
                j = feat2idx.get(k)
                if j is None or v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if fv in pad_set:
                    continue

                out[i, t_out, j] = fv
                has_valid = True

            if has_valid:
                t_out += 1

        lengths[i] = max(1, t_out)

    return out, lengths


class SequenceDataset(Dataset):
    def __init__(self, sequences_np: np.ndarray, lengths_np: np.ndarray, labels_np: np.ndarray):
        self.sequences = torch.from_numpy(sequences_np).float()
        self.lengths = torch.from_numpy(lengths_np).long()
        self.labels = torch.from_numpy(labels_np).long()

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.lengths[idx], self.labels[idx]


def can_stratify(y: np.ndarray, n_splits: int) -> bool:
    c = Counter(y.tolist())
    return len(c) >= 2 and min(c.values()) >= n_splits


# Checkpoint
def save_best_checkpoint(
    ckpt_path: Path,
    meta_path: Path,
    model: torch.nn.Module,
    feature_names: List[str],
    label_classes: List[str],
    config: Dict[str, Any],
    model_kwargs: Dict[str, Any],
    best_info: Dict[str, Any],
) -> None:
    ensure_dir(ckpt_path.parent)

    state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ckpt = {
        "version": 1,
        "saved_at": utcnow_z(),
        "model_state_dict": state_dict_cpu,
        "model_kwargs": model_kwargs,
        "feature_names": feature_names,
        "label_classes": list(map(str, label_classes)),
        "config": config,
        "best_info": best_info,
    }
    torch.save(ckpt, str(ckpt_path))

    meta = {
        "saved_at": ckpt["saved_at"],
        "best_info": best_info,
        "model_kwargs": model_kwargs,
        "config": config,
        "num_features": len(feature_names),
        "num_classes": len(label_classes),
        "label_classes_head": list(map(str, label_classes[:30])),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[SAVE] best checkpoint: {ckpt_path}")
    print(f"[SAVE] best meta      : {meta_path}")


def load_checkpoint(ckpt_path: Path, device: torch.device, model_ctor) -> Tuple[torch.nn.Module, List[str], List[str], Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model_kwargs = ckpt["model_kwargs"]
    feature_names = ckpt["feature_names"]
    label_classes = ckpt["label_classes"]
    config = ckpt["config"]
    best_info = ckpt.get("best_info", {})

    model = model_ctor(**model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, feature_names, label_classes, config, best_info


# Plot / ROC
def plot_learning_curve(history: Dict[str, List[float]], out_png: Path) -> None:
    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    val_f1 = history["val_f1"]

    ensure_dir(out_png.parent)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_f1, label="val_f1")
    plt.xlabel("epoch")
    plt.ylabel("f1")
    plt.title("Val F1")
    plt.legend()

    plt.tight_layout()
    plt.savefig(str(out_png), dpi=150)
    plt.close()
    print(f"[PLOT] saved: {out_png}")


def save_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    out_png: Path,
    out_csv: Optional[Path],
    top_n_classes: int,
) -> Dict[str, Any]:
    from sklearn.metrics import auc, roc_auc_score, roc_curve
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    n_classes = int(y_prob.shape[1])
    uniq = np.unique(y_true)

    if len(uniq) < 2:
        print("[ROC] y_true에 클래스가 1개뿐이라 ROC 계산 불가")
        return {"roc_auc": None}

    ensure_dir(out_png.parent)

    if n_classes == 2:
        scores = y_prob[:, 1]
        fpr, tpr, thr = roc_curve(y_true, scores)
        roc_auc = float(auc(fpr, tpr))

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Test)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(out_png), dpi=150)
        plt.close()
        print(f"[ROC] saved: {out_png} (AUC={roc_auc:.4f})")

        if out_csv is not None:
            ensure_dir(out_csv.parent)
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["fpr", "tpr", "threshold"])
                for a, b, c in zip(fpr, tpr, thr):
                    w.writerow([float(a), float(b), float(c)])
            print(f"[ROC] points saved: {out_csv}")

        return {"roc_auc": roc_auc}

    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    fpr_micro, tpr_micro, thr_micro = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    auc_micro = float(auc(fpr_micro, tpr_micro))

    try:
        auc_macro = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception as e:
        print(f"[ROC] macro AUC 계산 실패: {e}")
        auc_macro = None

    supports = y_true_bin.sum(axis=0)
    top_idx = np.argsort(-supports)[: min(int(top_n_classes), n_classes)]

    plt.figure()
    plt.plot(fpr_micro, tpr_micro, label=f"micro-average (AUC={auc_micro:.4f})")
    if auc_macro is not None:
        plt.plot([], [], label=f"macro-average (AUC={auc_macro:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

    per_class_top: Dict[int, Dict[str, Any]] = {}
    for c in top_idx:
        if supports[c] == 0 or supports[c] == len(y_true):
            continue
        fpr_c, tpr_c, _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
        auc_c = float(auc(fpr_c, tpr_c))
        name = class_names[c] if c < len(class_names) else f"class_{c}"
        plt.plot(fpr_c, tpr_c, label=f"{name} (AUC={auc_c:.3f}, n={int(supports[c])})")
        per_class_top[int(c)] = {"name": str(name), "auc": auc_c, "support": int(supports[c])}

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test, OVR)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=150)
    plt.close()
    print(f"[ROC] saved: {out_png} (micro={auc_micro:.4f}, macro={auc_macro})")

    if out_csv is not None:
        ensure_dir(out_csv.parent)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["fpr", "tpr", "threshold"])
            for a, b, c in zip(fpr_micro, tpr_micro, thr_micro):
                w.writerow([float(a), float(b), float(c)])
        print(f"[ROC] points saved (micro): {out_csv}")

    return {"roc_auc_micro": auc_micro, "roc_auc_macro": auc_macro, "per_class_top": per_class_top}
