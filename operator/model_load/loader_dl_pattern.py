#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        return self.fc(h_n[-1])


_PREFERRED_NAMES = (
    "best_model.h5",
    "best_model.pt",
    "best_model.pth",
    "model.h5",
    "model.pt",
    "model.pth",
    "checkpoint.h5",
    "checkpoint.pt",
    "checkpoint.pth",
)

_EXTS = (".pt", ".pth", ".h5")


def _find_ckpt(path: Union[str, Path]) -> Path:
    p = Path(path).expanduser().resolve()

    if p.is_file():
        return p

    if p.is_dir():
        for name in _PREFERRED_NAMES:
            cand = p / name
            if cand.is_file():
                return cand

        cands: List[Path] = []
        for ext in _EXTS:
            cands.extend(p.rglob(f"*{ext}"))

        cands = [c for c in cands if c.is_file()]
        if cands:
            cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return cands[0]

    raise FileNotFoundError(f"DL-pattern checkpoint not found: {p}")


def _require_keys(ckpt: Dict[str, Any], keys: List[str]) -> None:
    for k in keys:
        if k not in ckpt:
            raise KeyError(f"Missing key in checkpoint: {k}")


def load_dl_torch_bundle(
    ckpt_path_or_dir: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    ckpt_path = _find_ckpt(ckpt_path_or_dir)
    device_t = torch.device(device)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Invalid checkpoint format (expected dict).")

    _require_keys(ckpt, ["model_state_dict", "model_kwargs", "feature_names", "label_classes"])

    model_kwargs = ckpt["model_kwargs"]
    if not isinstance(model_kwargs, dict):
        raise ValueError("Invalid checkpoint: model_kwargs must be dict.")

    state_dict = ckpt["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("Invalid checkpoint: model_state_dict must be dict.")

    model = LSTMClassifier(**model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device_t)
    model.eval()

    feature_names = list(ckpt["feature_names"]) if ckpt.get("feature_names") is not None else []
    label_classes = list(ckpt["label_classes"]) if ckpt.get("label_classes") is not None else []

    bundle: Dict[str, Any] = {
        "model": model,
        "feature_names": feature_names,
        "label_classes": label_classes,
        "model_kwargs": dict(model_kwargs),
        "config": dict(ckpt.get("config") or {}),
        "best_info": dict(ckpt.get("best_info") or {}),
        "ckpt_path": str(ckpt_path),
    }
    return bundle
