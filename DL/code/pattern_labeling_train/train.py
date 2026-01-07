import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from collections import Counter
from datetime import datetime

# ✅ plot
import matplotlib.pyplot as plt


# -------------------------------
# Configuration
# -------------------------------
FILE_PATH = r"C:/Users/USER/Desktop/bob 프로젝트/AI/ML-DL/DL/pattern_labeling/train/new_code/cluster_result.jsonl"
FEATURE_FILE = r"C:/Users/USER/Desktop/bob 프로젝트/AI/ML-DL/DL/pattern_labeling/train/new_code/feature.txt"

NUM_FOLDS = 3
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MAX_SEQ_LENGTH = 128

TEST_SIZE = 0.2
RANDOM_STATE = 42

PAD_VALUES = (-1.0, -2.0)
DROP_LABELS = ("Noise",)

# Early Stopping
EARLY_STOP_PATIENCE = 5        # 몇 epoch 연속 개선 없으면 stop
EARLY_STOP_MIN_DELTA = 1e-4    # 개선으로 인정할 최소 변화량 (F1 기준)
EARLY_STOP_METRIC = "val_f1"   # "val_f1" (권장) or "val_loss"

# DataLoader
NUM_WORKERS = 0  # Windows는 0이 빠른 경우 많음 (필요 시 2로 비교)
PIN_MEMORY = True

# AMP (Mixed Precision)
USE_AMP = True

# 저장 경로
RESULT_DIR = "./result"

# ✅ best 모델 확장자를 h5로 통일 (내용은 torch checkpoint)
BEST_CKPT_PATH = os.path.join(RESULT_DIR, "best_model.h5")

BEST_META_PATH = os.path.join(RESULT_DIR, "best_model_meta.json")
TEST_METRICS_PATH = os.path.join(RESULT_DIR, "test_metrics.json")
CURVE_BEST_PNG = os.path.join(RESULT_DIR, "learning_curve_best.png")
HIST_DIR = os.path.join(RESULT_DIR, "history")

# ✅ ROC 저장 경로
TEST_ROC_PNG = os.path.join(RESULT_DIR, "roc_curve_test.png")
TEST_ROC_POINTS_CSV = os.path.join(RESULT_DIR, "roc_points_test.csv")

# ✅ CV 요약 저장
CV_SUMMARY_PATH = os.path.join(RESULT_DIR, "cv_summary.json")


# -------------------------------
# Utils
# -------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_features(feature_file):
    features = []
    try:
        with open(feature_file, "r", encoding="utf-8") as f:
            for line in f:
                feat = line.strip()
                if feat:
                    features.append(feat)
    except FileNotFoundError:
        print(f"Error: Feature file not found at {feature_file}")
        return None
    return features


def load_data(file_path, drop_labels=DROP_LABELS):
    sequences = []
    labels = []
    dropped = 0
    total = 0
    drop_set = set(drop_labels)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "sequence_group" not in data or "label" not in data:
                continue

            lb = data["label"]
            if lb in drop_set:
                dropped += 1
                continue

            sequences.append(data["sequence_group"])
            labels.append(lb)

    print(f"[load_data] total={total}, kept={len(labels)}, dropped={dropped} (drop_labels={list(drop_set)})")
    if labels:
        c = Counter(labels)
        print(f"[load_data] label top10: {c.most_common(10)}")
    return sequences, labels


# ✅ -1/-2 timestep 제거 + lengths 생성
def preprocess_sequences(sequences, max_seq_length, feature_names, pad_values=PAD_VALUES):
    feat2idx = {k: i for i, k in enumerate(feature_names)}
    D = len(feature_names)
    N = len(sequences)

    out = np.zeros((N, max_seq_length, D), dtype=np.float32)
    lengths = np.ones((N,), dtype=np.int64)  # pack은 length=0 불가 → 최소 1

    pad_set = set(float(x) for x in pad_values)

    for i, seq in enumerate(tqdm(sequences, desc="Preprocess", leave=False)):
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

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx], self.labels[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        last = h_n[-1]
        return self.fc(last)


def save_best_checkpoint(
    path_pt: str,
    path_json: str,
    model: nn.Module,
    feature_names: list,
    label_classes: list,
    config: dict,
    model_kwargs: dict,
    best_info: dict,
):
    ensure_dir(os.path.dirname(path_pt))

    state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    ckpt = {
        "version": 1,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "model_state_dict": state_dict_cpu,
        "model_kwargs": model_kwargs,
        "feature_names": feature_names,
        "label_classes": list(map(str, label_classes)),
        "config": config,
        "best_info": best_info,
    }

    # ✅ 확장자가 .h5여도 torch checkpoint로 저장됨
    torch.save(ckpt, path_pt)

    meta = {
        "saved_at": ckpt["saved_at"],
        "best_info": best_info,
        "model_kwargs": model_kwargs,
        "config": config,
        "num_features": len(feature_names),
        "num_classes": len(label_classes),
        "label_classes_head": list(map(str, label_classes[:30])),
    }
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] best checkpoint saved (torch checkpoint): {path_pt}")
    print(f"[SAVE] best meta saved                         : {path_json}")


def load_checkpoint(path_pt: str, device: torch.device):
    ckpt = torch.load(path_pt, map_location="cpu")
    model_kwargs = ckpt["model_kwargs"]
    feature_names = ckpt["feature_names"]
    label_classes = ckpt["label_classes"]
    config = ckpt["config"]
    best_info = ckpt.get("best_info", {})

    model = LSTMClassifier(**model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, feature_names, label_classes, config, best_info


def plot_learning_curve(history, out_png):
    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    val_f1 = history["val_f1"]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_f1, label="val_f1")
    plt.xlabel("epoch")
    plt.ylabel("f1")
    plt.title("Val F1 Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[PLOT] saved: {out_png}")


# -------------------------------
# ROC Utils
# -------------------------------
def save_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list,
    out_png: str,
    out_csv: str | None = None,
    top_n_classes: int = 5,
):
    """
    - binary: positive(class=1) 확률로 ROC
    - multiclass: OVR 기준 micro ROC curve + 상위 N 클래스 곡선 (가독성)
    """
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    n_classes = y_prob.shape[1]
    uniq = np.unique(y_true)

    if len(uniq) < 2:
        print("[ROC] y_true에 클래스가 1개뿐이라 ROC-AUC/curve 계산 불가")
        return {"roc_auc": None}

    ensure_dir(os.path.dirname(out_png))

    # Binary
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
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[ROC] saved: {out_png} (AUC={roc_auc:.4f})")

        if out_csv:
            df = np.stack([fpr, tpr, thr], axis=1)
            import pandas as pd
            pd.DataFrame(df, columns=["fpr", "tpr", "threshold"]).to_csv(out_csv, index=False, encoding="utf-8")
            print(f"[ROC] points saved: {out_csv}")

        return {"roc_auc": roc_auc}

    # Multiclass (OVR)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))  # (N, C)

    # micro-average curve
    fpr_micro, tpr_micro, thr_micro = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    auc_micro = float(auc(fpr_micro, tpr_micro))

    try:
        auc_macro = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception as e:
        print(f"[ROC] macro AUC 계산 실패: {e}")
        auc_macro = None

    supports = y_true_bin.sum(axis=0)
    top_idx = np.argsort(-supports)[: min(top_n_classes, n_classes)]

    plt.figure()
    plt.plot(fpr_micro, tpr_micro, label=f"micro-average (AUC={auc_micro:.4f})")
    if auc_macro is not None:
        plt.plot([], [], label=f"macro-average (AUC={auc_macro:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

    per_class_top = {}
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
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[ROC] saved: {out_png} (micro={auc_micro:.4f}, macro={auc_macro})")

    if out_csv:
        import pandas as pd
        pd.DataFrame({"fpr": fpr_micro, "tpr": tpr_micro, "threshold": thr_micro}).to_csv(
            out_csv, index=False, encoding="utf-8"
        )
        print(f"[ROC] points saved (micro): {out_csv}")

    return {"roc_auc_micro": auc_micro, "roc_auc_macro": auc_macro, "per_class_top": per_class_top}


# -------------------------------
# Train / Eval
# -------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for sequences, lengths, labels in pbar:
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(sequences, lengths)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{total_loss/max(1,n_batches):.4f}")

    return total_loss / max(1, n_batches)


def evaluate_epoch(model, loader, criterion, device, name="VAL", scaler=None, return_probs=False):
    """
    return_probs=True 이면:
      (avg_loss, acc, f1, rec, probs[N,C], y_true[N])
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(loader, desc=f"{name}", leave=False)
    with torch.inference_mode():
        for sequences, lengths, labels in pbar:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(sequences, lengths)  # logits
                    loss = criterion(outputs, labels)
            else:
                outputs = model(sequences, lengths)
                loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item()
            n_batches += 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if return_probs:
                all_probs.append(probs.detach().cpu().numpy())

    avg_loss = total_loss / max(1, n_batches)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    rec = recall_score(all_labels, all_preds, average="weighted")

    if return_probs:
        probs_np = np.concatenate(all_probs, axis=0) if len(all_probs) else None
        labels_np = np.asarray(all_labels, dtype=np.int64)
        return avg_loss, acc, f1, rec, probs_np, labels_np

    return avg_loss, acc, f1, rec


# -------------------------------
# Main
# -------------------------------
def main():
    ensure_dir(RESULT_DIR)
    ensure_dir(HIST_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # AMP 준비
    use_amp = (device.type == "cuda" and USE_AMP)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"[AMP] enabled={bool(scaler)}")

    # 1) load
    print("Loading data (drop Noise)...")
    sequences, labels = load_data(FILE_PATH, drop_labels=DROP_LABELS)
    if not sequences or not labels:
        print("No data found after filtering. Exiting.")
        return

    feature_names = load_features(FEATURE_FILE)
    if feature_names is None:
        print("Failed to load feature names. Exiting.")
        return

    # 2) label encode
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels).astype(np.int64)
    num_classes = len(label_encoder.classes_)
    print(f"[labels] num_classes={num_classes}")

    # 3) preprocess
    print("Preprocessing sequences (pack-ready)...")
    processed_sequences, lengths = preprocess_sequences(
        sequences, MAX_SEQ_LENGTH, feature_names, pad_values=PAD_VALUES
    )
    N = len(encoded_labels)
    assert processed_sequences.shape[0] == N

    full_dataset = SequenceDataset(processed_sequences, lengths, encoded_labels)

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        pin_memory=(device.type == "cuda" and PIN_MEMORY),
        num_workers=NUM_WORKERS,
        persistent_workers=(NUM_WORKERS > 0),
    )

    # 4) Train/Val/Test split
    all_idx = np.arange(N)
    try:
        trainval_idx, test_idx = train_test_split(
            all_idx,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=encoded_labels
        )
    except ValueError:
        trainval_idx, test_idx = train_test_split(
            all_idx,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=None
        )

    test_subset = Subset(full_dataset, test_idx)
    test_loader = DataLoader(test_subset, shuffle=False, **loader_kwargs)
    print(f"[split] trainval={len(trainval_idx)}, test={len(test_idx)}")

    # 5) CV split
    y_trainval = encoded_labels[trainval_idx]

    def can_stratify(y, n_splits):
        c = Counter(y.tolist())
        return min(c.values()) >= n_splits

    if NUM_FOLDS > 1 and can_stratify(y_trainval, NUM_FOLDS):
        splitter = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(splitter.split(trainval_idx, y_trainval))
        print("[CV] using StratifiedKFold")
    else:
        splitter = KFold(n_splits=max(2, NUM_FOLDS), shuffle=True, random_state=RANDOM_STATE)
        splits = list(splitter.split(trainval_idx))
        print("[CV] using KFold (fallback)")

    # 6) best tracking
    global_best = {
        "metric": EARLY_STOP_METRIC,
        "f1": -1.0,
        "val_loss": None,
        "fold": None,
        "epoch": None,
        "val_acc": None,
        "val_recall": None,
    }
    best_history_for_plot = None

    config_to_save = {
        "PAD_VALUES": PAD_VALUES,
        "DROP_LABELS": DROP_LABELS,
        "MAX_SEQ_LENGTH": MAX_SEQ_LENGTH,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "NUM_EPOCHS": NUM_EPOCHS,
        "NUM_FOLDS": NUM_FOLDS,
        "TEST_SIZE": TEST_SIZE,
        "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE,
        "EARLY_STOP_MIN_DELTA": EARLY_STOP_MIN_DELTA,
        "EARLY_STOP_METRIC": EARLY_STOP_METRIC,
        "USE_AMP": bool(use_amp),
        "NUM_WORKERS": NUM_WORKERS,
    }

    fold_results = []

    # 7) Train folds
    for fold, (tr_sub, va_sub) in enumerate(splits, start=1):
        train_index = trainval_idx[tr_sub]
        val_index = trainval_idx[va_sub]

        print(f"\n--- Fold {fold}/{len(splits)} --- train={len(train_index)} val={len(val_index)}")

        train_subset = Subset(full_dataset, train_index)
        val_subset = Subset(full_dataset, val_index)

        train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)

        model_kwargs = dict(
            input_dim=len(feature_names),
            hidden_dim=64,
            output_dim=num_classes,
            n_layers=1,
        )
        model = LSTMClassifier(**model_kwargs).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_metric_value = None
        best_state = None
        best_epoch = None
        patience_count = 0

        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "val_recall": [],
        }

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
            val_loss, val_acc, val_f1, val_recall = evaluate_epoch(
                model, val_loader, criterion, device, name="VAL", scaler=scaler, return_probs=False
            )

            history["epoch"].append(epoch)
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(float(val_acc))
            history["val_f1"].append(float(val_f1))
            history["val_recall"].append(float(val_recall))

            print(
                f"[Fold {fold}] Epoch {epoch}/{NUM_EPOCHS} | "
                f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
                f"ValAcc={val_acc:.4f} ValF1={val_f1:.4f} ValRecall={val_recall:.4f}"
            )

            # --- Early stopping 기준 ---
            if EARLY_STOP_METRIC == "val_loss":
                current = -val_loss
                improve = (best_metric_value is None) or (current > best_metric_value + EARLY_STOP_MIN_DELTA)
            else:
                current = val_f1
                improve = (best_metric_value is None) or (current > best_metric_value + EARLY_STOP_MIN_DELTA)

            if improve:
                best_metric_value = current
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                patience_count = 0
                print(
                    f"  -> Fold {fold} best updated (epoch={epoch}, "
                    f"{EARLY_STOP_METRIC}={'{:.4f}'.format(val_f1) if EARLY_STOP_METRIC!='val_loss' else '{:.4f}'.format(val_loss)})"
                )
            else:
                patience_count += 1
                if patience_count >= EARLY_STOP_PATIENCE:
                    print(f"  -> EarlyStopping triggered on Fold {fold} (best_epoch={best_epoch})")
                    break

            # --- Global best 저장 (val_f1 기준) ---
            if val_f1 > global_best["f1"]:
                global_best.update({
                    "f1": float(val_f1),
                    "val_loss": float(val_loss),
                    "fold": int(fold),
                    "epoch": int(epoch),
                    "val_acc": float(val_acc),
                    "val_recall": float(val_recall),
                })

                save_best_checkpoint(
                    path_pt=BEST_CKPT_PATH,
                    path_json=BEST_META_PATH,
                    model=model,
                    feature_names=feature_names,
                    label_classes=label_encoder.classes_.tolist(),
                    config=config_to_save,
                    model_kwargs=model_kwargs,
                    best_info=global_best,
                )

                best_history_for_plot = dict(history)

        # fold 종료: fold-best state로 복구 후 fold_best 기록
        if best_state is not None:
            model.load_state_dict(best_state)
            if best_epoch is None:
                best_epoch = history["epoch"][-1]
            idx_best = history["epoch"].index(best_epoch)
            fold_best = {
                "fold": fold,
                "best_epoch": int(best_epoch),
                "train_loss": float(history["train_loss"][idx_best]),
                "val_loss": float(history["val_loss"][idx_best]),
                "val_acc": float(history["val_acc"][idx_best]),
                "val_f1": float(history["val_f1"][idx_best]),
                "val_recall": float(history["val_recall"][idx_best]),
            }
        else:
            fold_best = {
                "fold": fold,
                "best_epoch": int(history["epoch"][-1]),
                "train_loss": float(history["train_loss"][-1]),
                "val_loss": float(history["val_loss"][-1]),
                "val_acc": float(history["val_acc"][-1]),
                "val_f1": float(history["val_f1"][-1]),
                "val_recall": float(history["val_recall"][-1]),
            }

        fold_results.append(fold_best)

        # fold history 저장
        fold_hist_path = os.path.join(HIST_DIR, f"history_fold{fold}.json")
        with open(fold_hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] fold history: {fold_hist_path}")

    # 8) Summary
    print("\n--- CV Summary (best per fold) ---")
    avg_val_acc = float(np.mean([r["val_acc"] for r in fold_results])) if fold_results else 0.0
    avg_val_f1 = float(np.mean([r["val_f1"] for r in fold_results])) if fold_results else 0.0
    avg_val_recall = float(np.mean([r["val_recall"] for r in fold_results])) if fold_results else 0.0
    print(f"Avg ValAcc   : {avg_val_acc:.4f}")
    print(f"Avg ValF1    : {avg_val_f1:.4f}")
    print(f"Avg ValRecall: {avg_val_recall:.4f}")

    print("\n--- Best Model (Global, by ValF1) ---")
    print(global_best)
    print(f"Saved to: {BEST_CKPT_PATH}")
    print(f"Meta  to: {BEST_META_PATH}")

    # ✅ CV 요약 저장
    cv_summary = {
        "fold_results": fold_results,
        "avg_val_acc": avg_val_acc,
        "avg_val_f1": avg_val_f1,
        "avg_val_recall": avg_val_recall,
        "global_best": global_best,
    }
    with open(CV_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] cv summary: {CV_SUMMARY_PATH}")

    # 9) Test 평가 + ROC-AUC/curve
    best_model, _, _, _, best_info = load_checkpoint(BEST_CKPT_PATH, device)
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, test_f1, test_recall, test_probs, test_ytrue = evaluate_epoch(
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
            out_png=TEST_ROC_PNG,
            out_csv=TEST_ROC_POINTS_CSV,
            top_n_classes=5,  # 가독성 위해 상위 5개 클래스만 곡선
        )

    test_metrics = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_f1": float(test_f1),
        "test_recall": float(test_recall),
        "test_roc": roc_info,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "best_info": best_info,
    }

    with open(TEST_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] test metrics: {TEST_METRICS_PATH}")

    # meta에도 test_metrics 반영
    if os.path.exists(BEST_META_PATH):
        with open(BEST_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["test_metrics"] = test_metrics
        with open(BEST_META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] appended test_metrics to meta: {BEST_META_PATH}")

    print("\n--- Test (Best Model) ---")
    print(f"TestLoss={test_loss:.4f} TestAcc={test_acc:.4f} TestF1={test_f1:.4f} TestRecall={test_recall:.4f}")
    if roc_info is not None:
        if "roc_auc" in roc_info and roc_info["roc_auc"] is not None:
            print(f"Test ROC-AUC (binary)   : {roc_info['roc_auc']:.4f}")
        if "roc_auc_micro" in roc_info and roc_info["roc_auc_micro"] is not None:
            print(f"Test ROC-AUC (micro)    : {roc_info['roc_auc_micro']:.4f}")
        if "roc_auc_macro" in roc_info and roc_info["roc_auc_macro"] is not None:
            print(f"Test ROC-AUC (macro)    : {roc_info['roc_auc_macro']:.4f}")

    # 10) 학습 곡선 plot (global best 기준)
    if best_history_for_plot is not None:
        plot_learning_curve(best_history_for_plot, CURVE_BEST_PNG)

        best_hist_path = os.path.join(HIST_DIR, "history_best.json")
        with open(best_hist_path, "w", encoding="utf-8") as f:
            json.dump(best_history_for_plot, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] best history: {best_hist_path}")
    else:
        print("[WARN] best_history_for_plot is None (plot skipped)")


if __name__ == "__main__":
    main()
