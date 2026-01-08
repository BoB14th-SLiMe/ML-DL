#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_load(DL-pattern).py

- 학습 스크립트에서 저장한 torch checkpoint(best_model.h5/.pt/.pth)를 로드해서
  운영에서 바로 쓸 수 있는 bundle(dict)을 반환.

주의:
- best_model.h5 확장자여도 내용은 torch.save() 체크포인트입니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


# -------------------------
# Model Definition (학습 코드와 동일해야 함)
# -------------------------
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
        last = h_n[-1]
        return self.fc(last)


# -------------------------
# Finder
# -------------------------
_PREFERRED_NAMES = [
    "best_model.h5",
    "best_model.pt",
    "best_model.pth",
    "model.h5",
    "model.pt",
    "model.pth",
    "checkpoint.h5",
    "checkpoint.pt",
    "checkpoint.pth",
]


def _find_ckpt(path: Union[str, Path]) -> Path:
    p = Path(path)

    # 1) path가 파일이면 그대로
    if p.is_file():
        return p

    # 2) 디렉토리면 우선순위 이름으로 탐색
    if p.is_dir():
        for name in _PREFERRED_NAMES:
            cand = p / name
            if cand.exists() and cand.is_file():
                return cand

        # 3) fallback: 확장자 후보
        cand = list(p.rglob("*.pt")) + list(p.rglob("*.pth")) + list(p.rglob("*.h5"))
        if cand:
            # 가장 최신 파일을 선택(원하면 정책 바꿔도 됨)
            cand.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return cand[0]

    raise FileNotFoundError(f"DL-pattern checkpoint not found: {p}")


# -------------------------
# Public API (main_pipeline에서 호출)
# -------------------------
def load_dl_torch_bundle(
    ckpt_path_or_dir: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """
    Returns:
      {
        "model": torch.nn.Module,
        "feature_names": [...],
        "label_classes": [...],
        "model_kwargs": {...},
        "config": {...},
        "best_info": {...},
        "ckpt_path": "..."
      }
    """
    ckpt_path = _find_ckpt(ckpt_path_or_dir)
    device = torch.device(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Invalid checkpoint format (expected dict).")

    # 필수 필드 확인
    for k in ["model_state_dict", "model_kwargs", "feature_names", "label_classes"]:
        if k not in ckpt:
            raise KeyError(f"Missing key in ckpt: {k}")

    model_kwargs = ckpt["model_kwargs"]
    model = LSTMClassifier(**model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    bundle = {
        "model": model,
        "feature_names": list(ckpt["feature_names"]),
        "label_classes": list(ckpt["label_classes"]),
        "model_kwargs": dict(model_kwargs),
        "config": dict(ckpt.get("config") or {}),
        "best_info": dict(ckpt.get("best_info") or {}),
        "ckpt_path": str(ckpt_path),
    }
    return bundle
