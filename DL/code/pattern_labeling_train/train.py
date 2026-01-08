#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Subset

from train_common import (
    can_stratify,
    ensure_dir,
    get,
    load_checkpoint,
    load_features,
    load_sequences_and_labels,
    load_yaml,
    plot_learning_curve,
    preprocess_sequences,
    resolve_path,
    save_best_checkpoint,
    save_roc_curve,
    set_seed,
    utcnow_z,
    SequenceDataset,
)
from train_model import LSTMClassifier, eval_epoch, train_one_epoch


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="train.yaml")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        print(f"[ERROR] config not found: {cfg_path}", file=sys.stderr)
        return 1

    cfg = load_yaml(cfg_path)
    base_dir = cfg_path.parent.resolve()

    paths = get(cfg, "paths", {})
    data_cfg = get(cfg, "data", {})
    train_cfg = get(cfg, "training", {})
    model_cfg = get(cfg, "model", {})
    early_cfg = get(cfg, "early_stopping", {})
    roc_cfg = get(cfg, "roc", {})

    data_jsonl = resolve_path(base_dir, paths.get("data_jsonl"))
    feature_file = resolve_path(base_dir, paths.get("feature_file"))
    result_dir = resolve_path(base_dir, paths.get("result_dir", "./result"))

    best_ckpt = resolve_path(base_dir, paths.get("best_ckpt", str(result_dir / "best_model.h5")))
    best_meta = resolve_path(base_dir, paths.get("best_meta", str(result_dir / "best_model_meta.json")))
    test_metrics_path = resolve_path(base_dir, paths.get("test_metrics", str(result_dir / "test_metrics.json")))
    cv_summary_path = resolve_path(base_dir, paths.get("cv_summary", str(result_dir / "cv_summary.json")))
    curve_best_png = resolve_path(base_dir, paths.get("curve_best_png", str(result_dir / "learning_curve_best.png")))
    hist_dir = resolve_path(base_dir, paths.get("hist_dir", str(result_dir / "history")))
    roc_png = resolve_path(base_dir, paths.get("roc_png", str(result_dir / "roc_curve_test.png")))
    roc_csv = resolve_path(base_dir, paths.get("roc_csv", str(result_dir / "roc_points_test.csv")))

    max_seq_length = int(get(data_cfg, "max_seq_length", 128))
    pad_values = tuple(float(x) for x in get(data_cfg, "pad_values", [-1.0, -2.0]))
    drop_labels = list(get(data_cfg, "drop_labels", ["Noise"]))
    test_size = float(get(data_cfg, "test_size", 0.2))
    seed = int(get(data_cfg, "random_state", 42))

    num_folds = int(get(train_cfg, "num_folds", 3))
    num_epochs = int(get(train_cfg, "num_epochs", 50))
    batch_size = int(get(train_cfg, "batch_size", 32))
    lr = float(get(train_cfg, "lr", 0.01))
    weight_decay = float(get(train_cfg, "weight_decay", 0.0))
    num_workers = int(get(train_cfg, "num_workers", 0))
    pin_memory = bool(get(train_cfg, "pin_memory", True))
    use_amp_cfg = bool(get(train_cfg, "use_amp", True))

    hidden_dim = int(get(model_cfg, "hidden_dim", 64))
    n_layers = int(get(model_cfg, "n_layers", 1))
    dropout = float(get(model_cfg, "dropout", 0.0))

    patience = int(get(early_cfg, "patience", 5))
    min_delta = float(get(early_cfg, "min_delta", 1e-4))
    early_metric = str(get(early_cfg, "metric", "val_f1")).strip()

    roc_top_n = int(get(roc_cfg, "top_n_classes", 5))

    ensure_dir(result_dir)
    ensure_dir(hist_dir)

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_amp = bool(device.type == "cuda" and use_amp_cfg)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print(f"[ENV] device={device}, amp={bool(scaler)}")
    print(f"[PATH] data_jsonl={data_jsonl}")
    print(f"[PATH] feature_file={feature_file}")
    print(f"[PATH] result_dir={result_dir}")

    if args.dry_run:
        print("[DRY] config loaded successfully")
        return 0

    sequences, labels = load_sequences_and_labels(data_jsonl, drop_labels=drop_labels)
    if not sequences or not labels:
        print("[ERROR] no data after filtering", file=sys.stderr)
        return 1

    feature_names = load_features(feature_file)

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(labels).astype(np.int64)
    num_classes = int(len(label_encoder.classes_))
    print(f"[LABEL] num_classes={num_classes}")

    X_np, lengths_np = preprocess_sequences(sequences, max_seq_length, feature_names, pad_values=pad_values)
    N = int(len(y_all))
    if X_np.shape[0] != N:
        print("[ERROR] size mismatch", file=sys.stderr)
        return 1

    ds = SequenceDataset(X_np, lengths_np, y_all)

    loader_kwargs = dict(
        batch_size=batch_size,
        pin_memory=bool(device.type == "cuda" and pin_memory),
        num_workers=num_workers,
        persistent_workers=bool(num_workers > 0),
    )

    all_idx = np.arange(N)
    try:
        trainval_idx, test_idx = train_test_split(all_idx, test_size=test_size, random_state=seed, stratify=y_all)
    except ValueError:
        trainval_idx, test_idx = train_test_split(all_idx, test_size=test_size, random_state=seed, stratify=None)

    test_loader = DataLoader(Subset(ds, test_idx), shuffle=False, **loader_kwargs)
    print(f"[SPLIT] trainval={len(trainval_idx)} test={len(test_idx)}")

    y_trainval = y_all[trainval_idx]
    if num_folds > 1 and can_stratify(y_trainval, num_folds):
        splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(trainval_idx, y_trainval))
        print("[CV] StratifiedKFold")
    else:
        splitter = KFold(n_splits=max(2, num_folds), shuffle=True, random_state=seed)
        splits = list(splitter.split(trainval_idx))
        print("[CV] KFold")

    def score_for_best(val_loss: float, val_f1: float) -> float:
        if early_metric == "val_loss":
            return -val_loss
        return val_f1

    global_best: Dict[str, Any] = {
        "metric": early_metric,
        "score": -1e18,
        "val_f1": -1.0,
        "val_loss": None,
        "fold": None,
        "epoch": None,
        "val_acc": None,
        "val_recall": None,
    }
    best_history_for_plot: Optional[Dict[str, List[float]]] = None

    config_to_save = {
        "paths": {"data_jsonl": str(data_jsonl), "feature_file": str(feature_file), "result_dir": str(result_dir)},
        "data": {
            "drop_labels": drop_labels,
            "pad_values": list(pad_values),
            "max_seq_length": max_seq_length,
            "test_size": test_size,
            "random_state": seed,
        },
        "training": {
            "num_folds": num_folds,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "use_amp": use_amp,
        },
        "model": {"hidden_dim": hidden_dim, "n_layers": n_layers, "dropout": dropout},
        "early_stopping": {"patience": patience, "min_delta": min_delta, "metric": early_metric},
        "roc": {"top_n_classes": roc_top_n},
    }

    fold_results: List[Dict[str, Any]] = []

    for fold, (tr_sub, va_sub) in enumerate(splits, start=1):
        train_index = trainval_idx[tr_sub]
        val_index = trainval_idx[va_sub]

        print(f"\n[FOLD] {fold}/{len(splits)} train={len(train_index)} val={len(val_index)}")

        train_loader = DataLoader(Subset(ds, train_index), shuffle=True, **loader_kwargs)
        val_loader = DataLoader(Subset(ds, val_index), shuffle=False, **loader_kwargs)

        model_kwargs = dict(
            input_dim=len(feature_names),
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            n_layers=n_layers,
            dropout=dropout,
        )
        model = LSTMClassifier(**model_kwargs).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_score: Optional[float] = None
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_epoch: Optional[int] = None
        patience_count = 0

        history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "val_recall": [],
        }

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
            val_loss, val_acc, val_f1, val_recall, _, _ = eval_epoch(
                model, val_loader, criterion, device, name="VAL", scaler=scaler, return_probs=False
            )

            history["epoch"].append(float(epoch))
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(float(val_acc))
            history["val_f1"].append(float(val_f1))
            history["val_recall"].append(float(val_recall))

            print(
                f"[Fold {fold}] Epoch {epoch}/{num_epochs} | "
                f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
                f"ValAcc={val_acc:.4f} ValF1={val_f1:.4f} ValRecall={val_recall:.4f}"
            )

            current_score = float(score_for_best(val_loss, val_f1))
            improved = True if best_score is None else (current_score > (best_score + min_delta))

            if improved:
                best_score = current_score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    break

            if current_score > float(global_best["score"]):
                global_best.update(
                    {
                        "score": float(current_score),
                        "val_f1": float(val_f1),
                        "val_loss": float(val_loss),
                        "fold": int(fold),
                        "epoch": int(epoch),
                        "val_acc": float(val_acc),
                        "val_recall": float(val_recall),
                    }
                )
                save_best_checkpoint(
                    ckpt_path=best_ckpt,
                    meta_path=best_meta,
                    model=model,
                    feature_names=feature_names,
                    label_classes=label_encoder.classes_.tolist(),
                    config=config_to_save,
                    model_kwargs=model_kwargs,
                    best_info=dict(global_best),
                )
                best_history_for_plot = dict(history)

        if best_state is not None:
            model.load_state_dict(best_state)

        if best_epoch is None:
            best_epoch = int(history["epoch"][-1]) if history["epoch"] else 0

        idx_best = int(best_epoch) - 1 if best_epoch > 0 and best_epoch <= len(history["epoch"]) else (len(history["epoch"]) - 1)
        idx_best = max(0, idx_best)

        fold_best = {
            "fold": int(fold),
            "best_epoch": int(best_epoch),
            "train_loss": float(history["train_loss"][idx_best]) if history["train_loss"] else None,
            "val_loss": float(history["val_loss"][idx_best]) if history["val_loss"] else None,
            "val_acc": float(history["val_acc"][idx_best]) if history["val_acc"] else None,
            "val_f1": float(history["val_f1"][idx_best]) if history["val_f1"] else None,
            "val_recall": float(history["val_recall"][idx_best]) if history["val_recall"] else None,
        }
        fold_results.append(fold_best)

        fold_hist_path = hist_dir / f"history_fold{fold}.json"
        fold_hist_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SAVE] fold history: {fold_hist_path}")

    avg_val_acc = float(np.mean([r["val_acc"] for r in fold_results if r["val_acc"] is not None])) if fold_results else 0.0
    avg_val_f1 = float(np.mean([r["val_f1"] for r in fold_results if r["val_f1"] is not None])) if fold_results else 0.0
    avg_val_recall = float(np.mean([r["val_recall"] for r in fold_results if r["val_recall"] is not None])) if fold_results else 0.0

    cv_summary = {
        "fold_results": fold_results,
        "avg_val_acc": avg_val_acc,
        "avg_val_f1": avg_val_f1,
        "avg_val_recall": avg_val_recall,
        "global_best": global_best,
        "saved_at": utcnow_z(),
    }
    cv_summary_path.write_text(json.dumps(cv_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVE] cv summary: {cv_summary_path}")

    best_model, _, _, _, best_info = load_checkpoint(best_ckpt, device, model_ctor=LSTMClassifier)
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, test_f1, test_recall, test_probs, test_ytrue = eval_epoch(
        best_model,
        test_loader,
        criterion,
        device,
        name="TEST",
        scaler=(scaler if use_amp else None),
        return_probs=True,
    )

    roc_info = None
    if test_probs is not None:
        roc_info = save_roc_curve(
            y_true=test_ytrue,
            y_prob=test_probs,
            class_names=label_encoder.classes_.tolist(),
            out_png=roc_png,
            out_csv=roc_csv,
            top_n_classes=roc_top_n,
        )

    test_metrics = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_f1": float(test_f1),
        "test_recall": float(test_recall),
        "test_roc": roc_info,
        "evaluated_at": utcnow_z(),
        "best_info": best_info,
    }
    test_metrics_path.write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVE] test metrics: {test_metrics_path}")

    if best_history_for_plot is not None:
        plot_learning_curve(best_history_for_plot, curve_best_png)
        (hist_dir / "history_best.json").write_text(json.dumps(best_history_for_plot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f} recall={test_recall:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
