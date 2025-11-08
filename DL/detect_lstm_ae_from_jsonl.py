"""
detect_lstm_ae_with_similarity_XAI.py
---------------------------------
LSTM Autoencoder 기반 이상탐지 + 패턴 유사도 + XAI 확장 버전
"""
import json
import numpy as np
import tensorflow as tf
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm

# ============================================================
# ⚙️ Global Detection Parameters (중앙 설정 관리)
# ============================================================

# CONFIG = { # 갱신 전
#     # -----------------------------
#     # 📁 데이터 로드 관련
#     # -----------------------------
#     "limit": 10000,             # 최대 패킷 로드 수
#     "window_step": 1,           # 슬라이딩 윈도우 step (기본 1)
#     "adaptive_threshold": True, # IQR 기반 자동 임계값 계산 사용 여부

#     # -----------------------------
#     # 📈 Reconstruction Error Threshold
#     # -----------------------------
#     "threshold_factor": 3.0,    # mean + n * std 방식에서 n
#     "iqr_factor": 1.5,          # IQR 기반 threshold multiplier

#     # -----------------------------
#     # 🧠 Semantic Score Weights
#     # -----------------------------
#     "mse_weight": 0.30,
#     "latent_weight": 0.35,
#     "entropy_weight": 0.25,
#     "temporal_weight": 0.10,

#     # -----------------------------
#     # 🚨 Semantic 판단 기준
#     # -----------------------------
#     "semantic_threshold": 0.75,  # 의미론적 이상 탐지 기준
#     "similarity_cutoff": 85.0,   # 유사도 % 기준
#     "mse_high": 1000.0,          # 구조적 이상으로 판단할 MSE 기준
# }

CONFIG = { # 갱신 후
    "limit": 100000,
    "window_step": 1,
    "adaptive_threshold": False,
    "threshold_factor": 2.0,
    "iqr_factor": 1.5,

    "mse_weight": 0.46,
    "latent_weight": 0.29,
    "entropy_weight": 0.25,
    "temporal_weight": 0.0,

    "semantic_threshold": 0.75,
    "similarity_cutoff": 85.0,
    "mse_high": 1000.0,
}


def auto_config_tuning(mse_scores):
    """데이터 통계 기반으로 CONFIG 값을 자동 조정"""
    mean_mse, std_mse = mse_scores.mean(), mse_scores.std()
    q1, q3 = np.percentile(mse_scores, [25, 75])
    iqr = q3 - q1

    # 1️⃣ threshold 관련 자동화
    if std_mse > mean_mse * 0.5:      # 분산이 큰 경우 adaptive threshold
        CONFIG["adaptive_threshold"] = True
        CONFIG["iqr_factor"] = np.clip(iqr / mean_mse, 1.2, 3.0)
    else:
        CONFIG["adaptive_threshold"] = False
        CONFIG["threshold_factor"] = np.clip(std_mse / mean_mse * 5, 2.0, 4.0)

    # 2️⃣ Semantic Score 가중치 자동 조정
    entropy_level = np.mean(np.log1p(mse_scores)) / 5
    CONFIG["mse_weight"] = round(0.25 + entropy_level * 0.1, 2)
    CONFIG["latent_weight"] = round(0.35 + (1 - entropy_level) * 0.05, 2)
    CONFIG["entropy_weight"] = 0.25
    CONFIG["temporal_weight"] = round(
        1.0 - (CONFIG["mse_weight"] + CONFIG["latent_weight"] + CONFIG["entropy_weight"]), 2
    )

    logger.info(f"🧩 Auto-Tuned CONFIG: {CONFIG}")
    return CONFIG

# ============================================================
# 1️⃣ 프로토콜 매핑
# ============================================================
PROTO_MAP = {
    "unknown": 0, "arp": 1, "bacnet": 2, "dhcp": 3, "dnp3": 4, "dns": 5,
    "ethernet_ip": 6, "iec104": 7, "mms": 8, "modbus_tcp": 9,
    "opc_ua": 10, "s7comm": 11, "tcp_session": 12, "xgt-fen": 13
}
DIR_MAP = {"request": 0, "response": 1, "unknown": 2}


# ============================================================
# 2️⃣ Feature Extractor
# ============================================================
def safe_float(x, default=0.0):
    try:
        if isinstance(x, (list, dict)):
            return default
        return float(x)
    except Exception:
        return default


def extract_features(pkt):
    proto = pkt.get("protocol", "unknown")
    proto_id = PROTO_MAP.get(proto, 0)
    dir_flag = DIR_MAP.get(pkt.get("dir"), 2)
    fc, addr, val, flen = 0.0, 0.0, 0.0, 0.0

    d = pkt.get("d", {})
    if isinstance(d, dict) and "len" in d:
        flen = safe_float(d.get("len"))

    if proto == "modbus_tcp":
        pdu = d.get("pdu", {})
        fc = safe_float(pdu.get("fc"))
        addr = safe_float(pdu.get("addr"))
        if isinstance(pdu.get("regs"), dict):
            vals = [safe_float(v) for v in pdu["regs"].values()]
            val = np.mean(vals) if vals else 0.0
    elif proto == "xgt-fen":
        inst = d.get("inst", {})
        fc = safe_float(inst.get("cmd"))
        val = safe_float(inst.get("dataSize"))
        varNm = inst.get("varNm", "")
        digits = "".join(ch for ch in varNm if ch.isdigit())
        addr = safe_float(digits)
    elif proto == "s7comm":
        pdu = d.get("pdu", {})
        prm = pdu.get("prm", {})
        fc = safe_float(prm.get("fn"))
        itms = prm.get("itms", [])
        if itms:
            addr = safe_float(itms[0].get("addr"))
            val = safe_float(itms[0].get("amt"))
    elif proto == "mms":
        val = safe_float(d.get("len"))

    delta_t = safe_float(pkt.get("_delta_t", 0.0))
    return [proto_id, dir_flag, fc, addr, val, flen, delta_t]


# ============================================================
# 3️⃣ JSONL 로드
# ============================================================
def get_model_window_size(model):
    input_shape = model.input_shape
    return input_shape[1]


def load_sequences_from_jsonl(path, window_size=14, overlap=1, limit=10000):
    packets, raw_packets = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(tqdm(f, desc=f"Parsing JSONL (limit={limit})")):
            if line_idx >= limit:
                break
            if not line.strip():
                continue
            try:
                pkt = json.loads(line)
                packets.append(extract_features(pkt))
                # 🔹 원본 전체 JSON 보존
                raw_packets.append(pkt)
            except Exception:
                continue

    total_packets = len(packets)
    logger.info(f"📦 Loaded {total_packets} packets (limited to {limit}) from {path}")

    sequences, seq_raw = [], []
    step = CONFIG["window_step"]
    for i in range(0, total_packets - window_size + 1, step):
        window = packets[i: i + window_size]
        raw_window = raw_packets[i: i + window_size]
        sequences.append(np.array(window, dtype="float32"))  # ✅ 반드시 numpy 배열로 변환해야 함
        seq_raw.append(raw_window)


    logger.info(f"📂 Generated {len(sequences)} windows (size={window_size}, step={step})")
    return sequences, seq_raw



# ============================================================
# 4️⃣ Padding
# ============================================================
def pad_with_mask(X_list):
    if not X_list:
        raise ValueError("❌ No valid sequences found for padding.")
    X_list = [np.array(x, dtype="float32") if not isinstance(x, np.ndarray) else x for x in X_list]
    max_len = max(x.shape[0] for x in X_list)
    feat_dim = X_list[0].shape[1]
    X_padded = np.zeros((len(X_list), max_len, feat_dim), dtype="float32")
    for i, x in enumerate(X_list):
        X_padded[i, :x.shape[0], :] = x
    logger.info(f"✅ Padded shape: {X_padded.shape}")
    return X_padded


# ============================================================
# 5️⃣ DL - Reconstruction & Latent
# ============================================================
def get_latent_vectors(model, X):
    encoder = tf.keras.Model(inputs=model.input, outputs=model.get_layer("lstm_1").output)
    return encoder.predict(X, verbose=0)


def reconstruction_error(model, X):
    recon = model.predict(X, verbose=0)
    mse = np.mean(np.square(X - recon), axis=(1, 2))
    return mse, recon


# ============================================================
# 6️⃣ DL 기반 XAI 지표 계산
# ============================================================
def compute_xai_metrics(model, X, recon, latent_vecs, sim_matrix, best_indices, pattern_centroids):
    feature_err = np.mean(np.square(X - recon), axis=1)
    feature_names = ["proto_id", "dir_flag", "fc", "addr", "val", "flen", "delta_t"]

    temporal_err = np.mean(np.square(X - recon), axis=2)
    pattern_vectors = np.stack(list(pattern_centroids.values()))

    latent_dist = np.array([
        norm(latent_vecs[i] - pattern_vectors[best_indices[i]])
        for i in range(len(latent_vecs))
    ])

    sim_norm = sim_matrix / (np.sum(sim_matrix, axis=1, keepdims=True) + 1e-9)
    entropy = -np.sum(sim_norm * np.log(sim_norm + 1e-9), axis=1)

    return feature_err, temporal_err, latent_dist, entropy, feature_names


# ============================================================
# 7️⃣ Main Detection Function
# ============================================================
# ============================================================
# 7️⃣ Main Detection Function
# ============================================================
import time  # ⏱️ 추가

def detect_anomalies_with_similarity_XAI(model_path, jsonl_path, pattern_centroids, threshold=None):
    global CONFIG
    logger.info(f"🚀 Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # ------------------------------------------------------------
    # 🕒 전체 변환 + 추론 시간 측정 시작
    # ------------------------------------------------------------
    pipeline_start = time.time()

    # ------------------------
    # 1️⃣ 데이터 변환 (로드 + Feature 추출 + Padding)
    # ------------------------
    window_size = get_model_window_size(model)
    X_list, meta_windows = load_sequences_from_jsonl(
        jsonl_path, window_size=window_size, overlap=window_size // 2, limit=CONFIG["limit"]
    )
    X_padded = pad_with_mask(X_list)

    # ------------------------
    # 2️⃣ DL 추론 (Reconstruction + Latent Vector)
    # ------------------------
    infer_start = time.time()
    mse_scores, recon = reconstruction_error(model, X_padded)
    latent_vecs = get_latent_vectors(model, X_padded)
    infer_end = time.time()

    # ------------------------
    # 3️⃣ 전체 파이프라인 종료
    # ------------------------
    pipeline_end = time.time()

    # ------------------------
    # ⏱️ 시간 계산
    # ------------------------
    total_inference_time = infer_end - infer_start
    total_pipeline_time = pipeline_end - pipeline_start
    avg_inference_time = total_inference_time / len(X_padded)
    avg_pipeline_time = total_pipeline_time / len(X_padded)

    logger.info(f"🧠 Total DL inference time: {total_inference_time:.3f} sec")
    logger.info(f"⚡ Avg inference time per window: {avg_inference_time:.6f} sec")
    logger.info(f"🧩 Total pipeline time (load+transform+inference): {total_pipeline_time:.3f} sec")
    logger.info(f"🚀 Avg pipeline time per window: {avg_pipeline_time:.6f} sec")

    # ------------------------------------------------------------
    # 4️⃣ 임계값 계산 및 예측
    # ------------------------------------------------------------
    mean_mse, std_mse = mse_scores.mean(), mse_scores.std()
    if CONFIG["adaptive_threshold"]:
        q1, q3 = np.percentile(mse_scores, [25, 75])
        iqr = q3 - q1
        threshold = q3 + CONFIG["iqr_factor"] * iqr
    else:
        threshold = mean_mse + CONFIG["threshold_factor"] * std_mse

    logger.info(f"📊 mean={mean_mse:.4f}, std={std_mse:.4f}, threshold={threshold:.4f}")
    preds = (mse_scores > threshold).astype(int)

    # ------------------------------------------------------------
    # 5️⃣ Latent Similarity + XAI Metric 계산
    # ------------------------------------------------------------
    pattern_names = list(pattern_centroids.keys())
    pattern_vectors = np.stack(list(pattern_centroids.values()))
    sim_matrix = cosine_similarity(latent_vecs, pattern_vectors)
    best_indices = np.argmax(sim_matrix, axis=1)
    best_patterns = [pattern_names[i] for i in best_indices]
    best_scores = [sim_matrix[j, i] * 100 for j, i in enumerate(best_indices)]

    feat_err, time_err, latent_dist, entropy, feat_names = compute_xai_metrics(
        model, X_padded, recon, latent_vecs, sim_matrix, best_indices, pattern_centroids
    )

    # ------------------------------------------------------------
    # 6️⃣ 결과 저장
    # ------------------------------------------------------------
    result_path = Path(jsonl_path).with_name("reconstruction_detect_with_XAI.json")
    results = []

    for idx, m in enumerate(mse_scores):
        results.append({
            "seq_id": int(idx),
            "mse": float(m),
            "is_anomaly": bool(preds[idx]),
            "closest_pattern": best_patterns[idx],
            "similarity": round(best_scores[idx], 2),
            "latent_distance": float(latent_dist[idx]),
            "similarity_entropy": float(entropy[idx]),
            "feature_error": {feat_names[k]: float(feat_err[idx, k]) for k in range(len(feat_names))},
            "temporal_error_mean": float(np.mean(time_err[idx])),
            "temporal_error_max": float(np.max(time_err[idx])),
            "window_raw": meta_windows[idx],
        })

    # ------------------------------------------------------------
    # 7️⃣ 요약 정보 (추론 + 전체 파이프라인 시간)
    # ------------------------------------------------------------
    summary_info = {
        "inference_summary": {
            "num_windows": len(X_padded),
            "total_inference_time_sec": round(total_inference_time, 4),
            "avg_inference_time_per_window_sec": round(avg_inference_time, 6),
            "total_pipeline_time_sec": round(total_pipeline_time, 4),
            "avg_pipeline_time_per_window_sec": round(avg_pipeline_time, 6)
        }
    }

    import copy
    with open(result_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(copy.deepcopy(r), ensure_ascii=False, default=lambda o: float(o)) + "\n")
        f.write(json.dumps(summary_info, ensure_ascii=False) + "\n")

    logger.success(f"✅ XAI Detection done → {result_path.resolve()}")
    logger.info(f"📈 Avg similarity={np.mean(best_scores):.2f}%, Avg entropy={np.mean(entropy):.4f}")
    logger.info(f"🧾 Inference Summary → {summary_info}")
    return results



# ============================================================
# 8️⃣ Semantic Score & SLM Input Generator
# ============================================================
def compute_semantic_score(mse, latent_distance, entropy, temporal_mean, temporal_max):
    mse_norm = np.log1p(mse) / 10
    ld_norm = latent_distance / (latent_distance + 5)
    ent_norm = entropy / (entropy + 2)
    temp_norm = np.log1p(temporal_max) / 10

    score = (
        CONFIG["mse_weight"] * mse_norm +
        CONFIG["latent_weight"] * ld_norm +
        CONFIG["entropy_weight"] * ent_norm +
        CONFIG["temporal_weight"] * temp_norm
    )
    return float(np.clip(score, 0.0, 1.0))


def generate_SLM_input(results, save_path):
    out_path = Path(save_path).with_name("for_XAI_SLM.jsonl")
    slm_ready = []

    for r in results:
        score = compute_semantic_score(
            r["mse"], r["latent_distance"], r["similarity_entropy"],
            r["temporal_error_mean"], r["temporal_error_max"]
        )

        if score > CONFIG["semantic_threshold"] and r["similarity"] < CONFIG["similarity_cutoff"]:
            anomaly_type = "semantic_deformation"
        elif r["is_anomaly"] and r["mse"] > CONFIG["mse_high"]:
            anomaly_type = "structural_deviation"
        else:
            anomaly_type = "normal"

        top_feat = max(r["feature_error"], key=r["feature_error"].get)
        top_val = r["feature_error"][top_feat]

        context = (
            f"이 시퀀스는 {r['closest_pattern']} 패턴에 속하지만 "
            f"{top_feat} 필드에서 높은 복원 오차({top_val:.3f})가 발생했습니다. "
            f"유사도 {r['similarity']:.2f}%, 엔트로피 {r['similarity_entropy']:.3f}, "
            f"latent distance {r['latent_distance']:.3f}로 의미적 불안정성이 감지됩니다. "
            f"Semantic score={score:.3f}."
        )

        slm_ready.append({
            "seq_id": r["seq_id"],
            "pattern": r["closest_pattern"],
            "summary": {
                "semantic_score": score,
                "anomaly_type": anomaly_type,
                "similarity": r["similarity"],
                "similarity_entropy": r["similarity_entropy"],
                "latent_distance": r["latent_distance"],
                "feature_error": r["feature_error"],
                "temporal_error_max": r["temporal_error_max"],
                "context": context
            },
            "window_raw": r.get("window_raw", []),
            "prompt": (
                "이 시퀀스는 의미적으로 어떤 이상을 나타내는가? "
                "정상 패턴과 비교하여 어떤 필드가 변형되었는지 설명하고, "
                "해당 행위가 밸브나 액추에이터를 손상시킬 가능성이 있는지 평가하라."
            )
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for row in slm_ready:
            f.write(json.dumps(row, ensure_ascii=False, default=lambda o: float(o)) + "\n")

    logger.success(f"🧠 SLM Input file generated → {out_path.resolve()}")
    logger.info(f"💡 평균 Semantic Score: {np.mean([r['summary']['semantic_score'] for r in slm_ready]):.3f}")


# ============================================================
# 🔚 Entry Point
# ============================================================
if __name__ == "__main__":
    MODEL_PATH = "outputs/models/LSTM_AE_Flexible_v2.keras"
    JSONL_PATH = "dataset/PLS-JSONL/merged.jsonl"
    pattern_centroids = np.load("outputs/pattern_centroids.npy", allow_pickle=True).item()

    results = detect_anomalies_with_similarity_XAI(
        model_path=MODEL_PATH,
        jsonl_path=JSONL_PATH,
        pattern_centroids=pattern_centroids
    )

    generate_SLM_input(results, JSONL_PATH)



"""
{
  "seq_id": 9576,                # 시퀀스 고유 ID (DL 입력 윈도우 또는 슬라이딩 시퀀스의 식별자)
  "pattern": "P_0002",           # SLM 또는 DL이 분류한 패턴명 (예: 공정 제어 시퀀스 유형)
  "summary": {
    "semantic_score": 0.7091595467880214,  # SLM이 계산한 의미적 일관성 점수 (1.0에 가까울수록 정상)
    "anomaly_type": "normal",              # DL/SLM 판단 결과: "normal" 또는 "anomaly"
    "similarity": 74.56999969482422,       # 학습된 정상 패턴(P_0002)과의 유사도 (%)
    "similarity_entropy": 1.7913528680801392,  # 유사도 분포의 엔트로피 (높을수록 불확실성이 큼)
    "latent_distance": 3.7522637844085693,     # 잠재공간(latent space) 거리 (값이 클수록 비정상 가능성)
    
    # 🔹 재구성(복원) 단계에서 필드별 오차(Feature Reconstruction Error)
    "feature_error": {
      "proto_id": 122.9727554321289,  # 프로토콜 ID 필드 복원 오차
      "dir_flag": 1.0189133882522583, # 방향 플래그(요청/응답) 복원 오차
      "fc": 3558.677978515625,        # Modbus/S7 함수 코드(Function Code) 복원 오차
      "addr": 174670.53125,           # 주소(Address) 필드 복원 오차 (가장 큰 이상 징후 발생)
      "val": 33.698707580566406,      # 값(Value) 필드 복원 오차
      "flen": 5.429335594177246,      # 프레임 길이(Field Length) 복원 오차
      "delta_t": 0.07732956856489182  # 연속 패킷 간 시간 간격 복원 오차
    },

    "temporal_error_max": 205339.265625,  # 시퀀스 내에서 발생한 최대 시계열 오차값
    "context": "이 시퀀스는 P_0002 패턴에 속하지만 addr 필드에서 높은 복원 오차(174670.531)가 발생했습니다. ..."
              # SLM이 생성한 자연어 설명: 어떤 필드에서 오차가 컸는지, 의미론적 안정성 수준 설명
  },
  "window_raw": 원본 패킷에 관한 json 데이터
  "prompt": "이 시퀀스는 의미적으로 어떤 이상을 나타내는가? ..."  
            # LLM(XAI) 질의 프롬프트: SLM이 DL의 결과를 바탕으로 해석적 설명을 생성하도록 유도
}

"""