#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        dr = float(dropout) if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dr)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        last = h_n[-1]
        return self.fc(last)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    *,
    show_pbar: bool = True,
    postfix_every: int = 20,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(
        loader,
        desc="Train",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5,
        disable=not show_pbar,
    )

    for step, (x, lengths, y) in enumerate(pbar, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x, lengths)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

        if postfix_every > 0 and (step % postfix_every == 0 or step == len(loader)):
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss/max(1,n_batches):.4f}")

    return total_loss / max(1, n_batches)


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    name: str,
    scaler: Optional[torch.cuda.amp.GradScaler],
    return_probs: bool,
    *,
    show_pbar: bool = True,
    postfix_every: int = 0,
) -> Tuple[float, float, float, float, Optional[np.ndarray], np.ndarray]:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    preds: List[int] = []
    labels: List[int] = []
    probs_list: List[np.ndarray] = []

    pbar = tqdm(
        loader,
        desc=name,
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5,
        disable=not show_pbar,
    )

    with torch.inference_mode():
        for step, (x, lengths, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(x, lengths)
                    loss = criterion(logits, y)
            else:
                logits = model(x, lengths)
                loss = criterion(logits, y)

            total_loss += float(loss.item())
            n_batches += 1

            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.detach().cpu().numpy().tolist())
            labels.extend(y.detach().cpu().numpy().tolist())

            if return_probs:
                probs_list.append(prob.detach().cpu().numpy())

            if postfix_every > 0 and (step % postfix_every == 0 or step == len(loader)):
                pbar.set_postfix(avg=f"{total_loss/max(1,n_batches):.4f}")

    avg_loss = total_loss / max(1, n_batches)
    acc = float(accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds, average="weighted"))
    rec = float(recall_score(labels, preds, average="weighted"))

    probs_np = np.concatenate(probs_list, axis=0) if return_probs and probs_list else None
    labels_np = np.asarray(labels, dtype=np.int64)

    return avg_loss, acc, f1, rec, probs_np, labels_np
