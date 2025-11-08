"""
build_pattern_centroids.py
---------------------------------
SLM 라벨링된 JSONL에서 패턴별 Latent Vector 평균 추출
LSTM AE Encoder 기반 centroid 저장
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# ------------------------------------------------------------
# 1️⃣ 기존 Feature Extractor 불러오기
# ------------------------------------------------------------
from detect_lstm_ae_from_jsonl import extract_features, pad_with_mask

# ------------------------------------------------------------
# 2️⃣ 설정
# ------------------------------------------------------------
MODEL_PATH = "outputs/models/LSTM_AE_Flexible_v2.keras"
JSONL_PATH = "dataset/PLS-JSONL/final.jsonl"
OUTPUT_PATH = "outputs/pattern_centroids.npy"

# ------------------------------------------------------------
# 3️⃣ 모델 로드 및 Encoder 추출
# ------------------------------------------------------------
logger.info(f"🚀 Loading LSTM Autoencoder: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
encoder = tf.keras.Model(inputs=model.input, outputs=model.get_layer("lstm_1").output)


def pad_with_fixed_length(X_list, fixed_len=14):
    """모델 입력 길이에 맞게 강제로 패딩"""
    feat_dim = X_list[0].shape[1]
    X_padded = np.zeros((len(X_list), fixed_len, feat_dim), dtype="float32")
    for i, x in enumerate(X_list):
        length = min(x.shape[0], fixed_len)
        X_padded[i, :length, :] = x[:length]
    return X_padded


# ------------------------------------------------------------
# 4️⃣ JSONL 로드
# ------------------------------------------------------------
def load_labeled_sequences(path):
    data_by_label = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="📥 Loading labeled SLM windows"):
            try:
                obj = json.loads(line)
                label = obj.get("label")
                seq = obj.get("window_group")
                if not label or not seq:
                    continue
                feats = [extract_features(pkt) for pkt in seq]
                if len(feats) < 2:
                    continue
                data_by_label.setdefault(label, []).append(np.array(feats, dtype="float32"))
            except Exception as e:
                logger.warning(f"⚠️ JSON decode error: {e}")
    logger.info(f"✅ Loaded {sum(len(v) for v in data_by_label.values())} windows across {len(data_by_label)} labels")
    return data_by_label

# ------------------------------------------------------------
# 5️⃣ 패턴별 Latent 평균 계산
# ------------------------------------------------------------
def compute_pattern_centroids(data_by_label):
    centroids = {}
    for label, seqs in data_by_label.items():
        # X_padded = pad_with_mask(seqs)
        X_padded = pad_with_fixed_length(seqs, fixed_len=14)
        latent = encoder.predict(X_padded, verbose=0)
        centroids[label] = np.mean(latent, axis=0)
        logger.info(f"📊 {label}: {len(seqs)} seqs → centroid shape={centroids[label].shape}")
    return centroids

# ------------------------------------------------------------
# 6️⃣ 저장
# ------------------------------------------------------------
if __name__ == "__main__":
    data_by_label = load_labeled_sequences(JSONL_PATH)
    centroids = compute_pattern_centroids(data_by_label)
    np.save(OUTPUT_PATH, centroids)
    logger.success(f"💾 Saved pattern centroids → {OUTPUT_PATH}")
