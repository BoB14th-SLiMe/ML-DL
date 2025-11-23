# ICS 이상 탐지 시스템 (ML/DL Pipeline)

## 개요

산업 제어 시스템(ICS) 네트워크 트래픽에서 이상을 탐지하는 파이프라인입니다.

```
Redis → ML (단일패킷) → Buffer(80개) → DL (시퀀스) → 이상 알람
```

---

## 출력 데이터 필드 설명

| 순서 | 필드 | 타입 | 설명 | 좋음 | 보통 | 나쁨 |
|:----:|------|------|------|------|------|------|
| 1 | `seq_id` | int | 윈도우 시퀀스 번호 | - | - | - |
| 2 | `pattern` | string | 탐지된 이상 패턴 (P_0001~P_0038) | - | - | - |
| 3 | **summary** | object | 분석 결과 요약 | - | - | - |
| 3.1 | `semantic_score` | float | 종합 탐지 신뢰도 [0~1] | 1에 가까울수록 좋음 | - | 0에 가까울수록 나쁨 |
| 3.2 | `anomaly_type` | string | 이상 여부 판정 | "normal" | - | "anomalous" |
| 3.3 | `anomaly_score` | float | LSTM-AE 재구성 오류 (MSE) | ≤ threshold (정상) | - | > threshold (이상) |
| 3.4 | `threshold` | float | 이상 판단 임계값 (변경 가능) | - | - | - |
| 3.5 | `similarity` | float | 패턴 매칭 확률 (%) | 100에 가까울수록 좋음 | - | 0에 가까울수록 나쁨 |
| 3.6 | `similarity_entropy` | float | 패턴 확률 엔트로피 (패턴의 개수에 따라 값이 변동됨) | 낮을수록 확실 | - | 높을수록 불확실 |
| 3.7 | `latent_distance` | float | 잠재 벡터 L2 노름 | - | - | - |
| 3.8 | **feature_error** (내부값 조정 알 수 없음) | object | 피처별 재구성 오류 | - | - | - |
| 3.8.1 | `proto_id` | float | 프로토콜 ID 오류 | - | - | - |
| 3.8.2 | `dir_flag` | float | 방향(req/res) 오류 | - | - | - |
| 3.8.3 | `fc` | float | Function Code 오류 | - | - | - |
| 3.8.4 | `addr` | float | 주소 오류 | - | - | - |
| 3.8.5 | `val` | float | 레지스터 값 오류 | - | - | - |
| 3.8.6 | `flen` | float | 길이 오류 | - | - | - |
| 3.8.7 | `delta_t` | float | 시간 간격 오류 | - | - | - |
| 3.9 | `temporal_error_max` | float | 80개 패킷 중 최대 재구성 오류 | - | - | - |
| 3.10 | **risk** | object | 위험도 평가 | - | - | - |
| 3.10.1 | `score` | float | 위험 점수 [0~100] | 0 ~ 33 (LOW) | 33 ~ 66 (MEDIUM) | 66 ~ 100 (HIGH) |
| 3.10.2 | `detected_time` | string | 탐지 시간 (ISO 8601) | - | - | - |
| 3.10.3 | `src_ip` | string | 출발지 IP | - | - | - |
| 3.10.4 | `src_asset` | string | 출발지 자산명 | - | - | - |
| 3.10.5 | `dst_ip` | string | 목적지 IP | - | - | - |
| 3.10.6 | `dst_asset` | string | 목적지 자산명 | - | - | - |
| 4 | **window_raw** | array | 원본 패킷 배열 (80개) | - | - | - |
| 4.1 | `@timestamp` | string | 패킷 타임스탬프 (ISO 8601) | - | - | - |
| 4.2 | `protocol` | string | 프로토콜 (modbus, xgt_fen, s7comm 등) | - | - | - |
| 4.3 | `smac` | string | 출발지 MAC 주소 | - | - | - |
| 4.4 | `dmac` | string | 목적지 MAC 주소 | - | - | - |
| 4.5 | `sip` | string | 출발지 IP | - | - | - |
| 4.6 | `dip` | string | 목적지 IP | - | - | - |
| 4.7 | `sp` | int | 출발지 포트 | - | - | - |
| 4.8 | `dp` | int | 목적지 포트 | - | - | - |
| 4.9 | `sq` | int | TCP 시퀀스 번호 | - | - | - |
| 4.10 | `ak` | int | TCP ACK 번호 | - | - | - |
| 4.11 | `fl` | int | TCP 플래그 | - | - | - |
| 4.12 | `dir` | string | 방향 (request/response) | - | - | - |
| 4.13 | `len` | int | 패킷 길이 | - | - | - |
| 4.14 | `src_asset` | string | 출발지 자산명 | - | - | - |
| 4.15 | `dst_asset` | string | 목적지 자산명 | - | - | - |
| 4.16 | `xgt_fen.prid` | int | XGT-FEN 프로토콜 ID | - | - | - |
| 4.17 | `xgt_fen.cmd` | int | XGT-FEN 명령어 | - | - | - |
| 4.18 | `xgt_fen.data` | string | XGT-FEN 데이터 (hex) | - | - | - |
| 4.19 | `xgt_fen.vars` | string | XGT-FEN 변수명 | - | - | - |
| 4.20 | `xgt_fen.description` | string | XGT-FEN 설명 | - | - | - |
| 4.21 | `modbus.fc` | int | Modbus Function Code | - | - | - |
| 4.22 | `modbus.addr` | int | Modbus 시작 주소 | - | - | - |
| 4.23 | `modbus.qty` | int | Modbus 레지스터 수량 | - | - | - |
| 4.24 | `redis_id` | string | Redis 원본 ID | - | - | - |
| 4.25 | `ml_anomaly_prob` | array | ML 피처별 이상 기여도 | - | - | - |
| 4.25.1 | `ml_anomaly_prob[].name` | string | 피처 이름 | - | - | - |
| 4.25.2 | `ml_anomaly_prob[].percent` | float | 이상 기여도 (%) | 0 ~ 30 | 30 ~ 70 | 70 ~ 100 |

---

## 출력 예시

### 정상 트래픽

```json
{
  "seq_id": 5,
  "pattern": "P_0012",
  "summary": {
    "semantic_score": 0.8921,
    "anomaly_type": "normal",
    "anomaly_score": 17.52,
    "threshold": 20.288,
    "similarity": 89.32,
    "similarity_entropy": 1.24,
    "latent_distance": 1.12,
    "feature_error": {
      "proto_id": 0.42,
      "dir_flag": 0.28,
      "fc": 1.85,
      "addr": 125.6,
      "val": 1850.3,
      "flen": 2.1e-05,
      "delta_t": 3.5e-05
    },
    "temporal_error_max": 45200.0,
    "risk": {
      "score": 28.5,
      "detected_time": "2025-11-10T08:44:16.150008Z",
      "src_ip": "192.168.10.80",
      "src_asset": "HMI 1(Blending)",
      "dst_ip": "192.168.10.45",
      "dst_asset": "Mitsubishi PLC (Wall)"
    }
  },
  "window_raw": []
}
```

### 이상 트래픽 (공격 의심)

```json
{
  "seq_id": 1,
  "pattern": "P_0001",
  "summary": {
    "semantic_score": 0.7449,
    "anomaly_type": "anomalous",
    "anomaly_score": 21.608,
    "threshold": 20.288,
    "similarity": 77.0,
    "similarity_entropy": 3.552,
    "latent_distance": 1.657,
    "feature_error": {
      "proto_id": 1.548,
      "dir_flag": 0.520,
      "fc": 3.293,
      "addr": 698.79,
      "val": 9188.65,
      "flen": 4.56e-06,
      "delta_t": 7.16e-05
    },
    "temporal_error_max": 2832480768.0,
    "risk": {
      "score": 47.13,
      "detected_time": "2025-11-10T08:44:16.150008Z",
      "src_ip": "192.168.10.80",
      "src_asset": "HMI 1(Blending)",
      "dst_ip": "192.168.10.45",
      "dst_asset": "Mitsubishi PLC (Wall)"
    }
  },
  "window_raw": [
    {
      "@timestamp": "2025-11-10T08:44:16.088754Z",
      "protocol": "xgt_fen",
      "smac": "00:05:14:07:4b:59",
      "dmac": "00:0b:29:74:0f:7b",
      "sip": "192.168.10.81",
      "dip": "192.168.10.15",
      "sp": 49157,
      "dp": 2004,
      "sq": 489841609,
      "ak": 1008750441,
      "fl": 24,
      "dir": "request",
      "len": 21,
      "src_asset": "HMI 2(Labeling)",
      "dst_asset": "LS Electric PLC",
      "xgt_fen.prid": 256,
      "xgt_fen.cmd": 84,
      "xgt_fen.vars": "%DB001000",
      "xgt_fen.description": "통신방법 스위치 신호",
      "redis_id": "line-1",
      "ml_anomaly_prob": [
        {"name": "sp", "percent": 95.93},
        {"name": "dp", "percent": 3.91}
      ]
    }
  ]
}
```

---

## 핵심 판단 기준 요약

| 판단 | 조건 |
|------|------|
| **정상** | `anomaly_score ≤ threshold` |
| **이상** | `anomaly_score > threshold` |
| **위험 LOW** | `risk.score < 33` |
| **위험 MEDIUM** | `33 ≤ risk.score < 66` |
| **위험 HIGH** | `risk.score ≥ 66` |

※ `threshold`는 `dl_data/inference_meta.json`에서 설정되며, 모델 학습 시 변경될 수 있습니다.

---

## 파일 구조

| 파일 | 설명 |
|------|------|
| `ML_start.py` | ML 단일 패킷 추론 |
| `DL_start.py` | DL 시퀀스 추론 |
| `lstm_ae_mtl_infer.py` | LSTM-AE 핵심 로직 |
| `S_PipeLine.py` | 메인 파이프라인 |
| `S_Pipeformodbus.py` | Modbus 전용 파이프라인 |

---

## 실행 방법

```bash
python3 S_PipeLine.py --host localhost --port 6379 --db 0 --loop --interval 0.5
```
