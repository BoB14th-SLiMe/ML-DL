#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
모델 번들 로더

- model.pkl
- (선택) scaler.pkl
- selected_features.json
- metadata.json

을 한 번에 로드해주는 유틸 함수.
"""

from pathlib import Path
import json
import joblib  # scikit-learn 설치되어 있으면 같이 있음


def load_model_bundle(model_dir: Path | str):
    """
    model_dir 경로 아래에서 모델 번들을 로드해서 반환.

    반환값:
      model, scaler, selected_features, metadata
        - model: 학습된 sklearn 기반 모델 객체
        - scaler: 전처리 스케일러 (없으면 None)
        - selected_features: 학습에 사용한 피처 이름 리스트 (없으면 None)
        - metadata: dict (model 이름, threshold 등)
    """
    model_dir = Path(model_dir)

    model_path = model_dir / "model.pkl"
    scaler_path = model_dir / "scaler.pkl"
    feat_path = model_dir / "selected_features.json"
    meta_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"model.pkl not found at {model_path}")

    # 1) 모델 로드 (pickle / joblib 둘 다 joblib.load 로 가능)
    model = joblib.load(model_path)

    # 2) 스케일러 로드 (선택)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    # 3) 피처 리스트 로드 (선택)
    if feat_path.exists():
        with feat_path.open("r", encoding="utf-8") as f:
            selected_features = json.load(f)
    else:
        selected_features = None

    # 4) 메타데이터 로드 (선택)
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return model, scaler, selected_features, metadata


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Load a model bundle and print basic info")
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="Directory containing model.pkl, selected_features.json, metadata.json")
    args = ap.parse_args()

    model, scaler, selected_features, metadata = load_model_bundle(args.model_dir)

    print("=== Model bundle info ===")
    print(f"model type      : {type(model)}")
    print(f"scaler type     : {type(scaler) if scaler is not None else None}")
    print(f"#features       : {len(selected_features) if selected_features is not None else 'None'}")
    print(f"metadata        : {metadata}")


"""
python model_load.py --model-dir ../model
"""
