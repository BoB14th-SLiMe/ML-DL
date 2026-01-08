#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_bundle(model_dir: Union[str, Path]) -> Tuple[Any, Optional[Any], Optional[List[str]], Dict[str, Any]]:
    model_dir = Path(model_dir).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"model_dir is not a directory: {model_dir}")

    model_path = model_dir / "model.pkl"
    scaler_path = model_dir / "scaler.pkl"
    feat_path = model_dir / "selected_features.json"
    meta_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"model.pkl not found: {model_path}")

    model = joblib.load(str(model_path))
    scaler = joblib.load(str(scaler_path)) if scaler_path.exists() else None

    selected_features: Optional[List[str]] = None
    if feat_path.exists():
        obj = _read_json(feat_path)
        if isinstance(obj, list):
            selected_features = [str(x) for x in obj]
        else:
            raise ValueError(f"selected_features.json must be a list: {feat_path}")

    metadata: Dict[str, Any] = {}
    if meta_path.exists():
        obj = _read_json(meta_path)
        if isinstance(obj, dict):
            metadata = obj
        else:
            raise ValueError(f"metadata.json must be a dict: {meta_path}")

    return model, scaler, selected_features, metadata


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, required=True)
    args = ap.parse_args(argv)

    model, scaler, selected_features, metadata = load_model_bundle(args.model_dir)

    print("=== Model bundle info ===")
    print(f"model type      : {type(model)}")
    print(f"scaler type     : {type(scaler) if scaler is not None else None}")
    print(f"#features       : {len(selected_features) if selected_features is not None else None}")
    print(f"metadata keys   : {sorted(list(metadata.keys()))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
