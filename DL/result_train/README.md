# Result Train

학습된 LSTM Autoencoder 모델을 사용한 이상 탐지 및 평가 모듈입니다.

## 개요

이 모듈은 학습된 모델을 사용하여 새로운 데이터에 대한 이상 탐지를 수행하고, 탐지 성능을 평가합니다.

## 폴더 구조

```
result_train/
├── code/
│   ├── 0.attack_result.py          # 공격 라벨 생성 (Ground Truth)
│   ├── 1.benchmark.py               # 모델 추론 및 이상 탐지
│   ├── 2.eval_detection_metrics.py  # 탐지 성능 평가
│   ├── 3.analyze_mse_dist.py        # MSE 분포 분석
│   ├── packet_feature_extractor.py  # 패킷 → Feature 변환 유틸
│   └── machine.py                   # (추가 유틸리티)
└── data/                            # 모델 및 데이터 저장
```

## 실행 순서

### 1단계: 공격 라벨(Ground Truth) 생성 (`0.attack_result.py`)

테스트 데이터에서 윈도우별 공격 여부를 라벨링합니다.

```bash
python 0.attack_result.py \
    --input "../data/attack.jsonl" \
    --window-size 41 \
    --step-size 30 \
    --output "../result/attack_result.csv"
```

#### 파라미터

| 파라미터 | 필수 | 기본값 | 설명 |
|----------|------|--------|------|
| `-i`, `--input` | O | - | 패킷 단위 JSONL 경로 |
| `-w`, `--window-size` | X | 80 | 윈도우 크기 |
| `-s`, `--step-size` | X | window-size | 슬라이딩 stride |
| `-o`, `--output` | O | - | 출력 CSV 경로 |

#### 공격 판정 기준

- 윈도우 내에 `protocol == "modbus"` 이고 `modbus.fc == 6` 인 패킷이 하나라도 있으면 공격(is_anomaly=1)

#### 출력 CSV 형식

```csv
window_index,start_packet_idx,end_packet_idx,valid_len,is_anomaly
0,0,40,41,0
1,30,70,41,1
...
```

---

### 2단계: 모델 추론 및 이상 탐지 (`1.benchmark.py`)

학습된 LSTM-AE 모델을 사용하여 윈도우별 재구성 오차(MSE)를 계산하고 이상 여부를 판정합니다.

```bash
# 기본 실행 (feature만 추출)
python 1.benchmark.py \
    --input "../data/attack.jsonl" \
    --pre-dir "../../preprocessing/result" \
    --window-size 76 \
    --output-dir "../result/benchmark"

# 모델 추론까지 수행
python 1.benchmark.py \
    --input "../data/attack.jsonl" \
    --pre-dir "../../preprocessing/result" \
    --window-size 41 \
    --step-size 30 \
    --output-dir "../result/benchmark" \
    --model-dir "../data" \
    --batch-size 128

# threshold 직접 지정
python 1.benchmark.py \
    --input "../data/attack.jsonl" \
    --pre-dir "../../preprocessing/result" \
    --window-size 41 \
    --step-size 30 \
    --output-dir "../result/benchmark" \
    --model-dir "../data" \
    --batch-size 128 \
    --threshold 100
```

#### 파라미터

| 파라미터 | 필수 | 기본값 | 설명 |
|----------|------|--------|------|
| `-i`, `--input` | O | - | 패킷 단위 JSONL 경로 |
| `-p`, `--pre-dir` | O | - | 전처리 파라미터 디렉토리 |
| `-w`, `--window-size` | X | 80 | 윈도우 크기 |
| `-s`, `--step-size` | X | window-size | 슬라이딩 stride |
| `-o`, `--output-dir` | O | - | 출력 디렉토리 |
| `-m`, `--model-dir` | X | None | 학습된 모델 디렉토리 (model.h5, config.json) |
| `-b`, `--batch-size` | X | 128 | 추론 배치 크기 |
| `-t`, `--threshold` | X | None | 이상 판정 임계값 (미지정 시 threshold.json 사용) |
| `--no-pad-last` | X | False | 마지막 윈도우 패딩 없이 버림 |

#### 출력 파일

| 파일명 | 설명 |
|--------|------|
| `X_windows.npy` | 윈도우별 feature 배열 [N, T, D] |
| `windows_meta.jsonl` | 윈도우 메타정보 |
| `window_scores.csv` | 윈도우별 MSE 및 이상 판정 결과 |

#### window_scores.csv 형식

```csv
window_index,start_packet_idx,end_packet_idx,valid_len,mse,is_anomaly
0,0,40,41,0.0234,0
1,30,70,41,0.1567,1
...
```

---

### 3단계: 탐지 성능 평가 (`2.eval_detection_metrics.py`)

Ground Truth(attack_result.csv)와 모델 예측 결과(window_scores.csv)를 비교하여 성능 지표를 계산합니다.

```bash
python 2.eval_detection_metrics.py \
    --attack-csv ../result/attack_result.csv \
    --pred-csv ../result/benchmark/window_scores.csv \
    --output-json ../result/eval_detection_metrics.json
```

#### 파라미터

| 파라미터 | 필수 | 기본값 | 설명 |
|----------|------|--------|------|
| `-a`, `--attack-csv` | O | - | Ground Truth CSV |
| `-p`, `--pred-csv` | O | - | 모델 예측 결과 CSV |
| `-o`, `--output-json` | X | metrics.json | 성능 지표 JSON |
| `--ignore-pred-minus1` | X | False | is_anomaly=-1인 행 제외 |

#### 계산되는 지표

- **Confusion Matrix**: TP, TN, FP, FN
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP)
- **Recall (TPR)**: TP / (TP + FN)
- **F1-score**: 2 * Precision * Recall / (Precision + Recall)
- **FPR**: FP / (FP + TN)
- **TNR**: TN / (TN + FP)
- **FNR**: FN / (FN + TP)

#### 출력 JSON 형식

```json
{
  "num_samples": 100,
  "confusion_matrix": {
    "TP": 45,
    "TN": 40,
    "FP": 5,
    "FN": 10
  },
  "accuracy": 0.85,
  "precision": 0.90,
  "recall": 0.818,
  "f1": 0.857,
  "tpr": 0.818,
  "fpr": 0.111,
  "tnr": 0.889,
  "fnr": 0.182
}
```

---

### 4단계: MSE 분포 분석 (`3.analyze_mse_dist.py`)

공격/정상 윈도우별 MSE 분포를 상세 분석합니다.

```bash
python 3.analyze_mse_dist.py \
    --attack-csv ../result/attack_result.csv \
    --pred-csv ../result/benchmark/window_scores.csv \
    --output-json ../result/analyze_mse_dist.json \
    --top-k 20
```

#### 파라미터

| 파라미터 | 필수 | 기본값 | 설명 |
|----------|------|--------|------|
| `--attack-csv` | O | - | Ground Truth CSV |
| `--pred-csv` | O | - | 모델 예측 결과 CSV |
| `--output-json` | X | None | 분석 결과 JSON |
| `--top-k` | X | 20 | 오차가 큰 윈도우 상위 K개 저장 |

#### 분석 내용

- **전체 통계**: attack/normal별 MSE mean, std, min, max, percentiles
- **그룹별 통계**: pattern, protocol 등 그룹별 MSE 분포
- **Top-K 윈도우**: 오차가 가장 큰 공격/정상 윈도우 목록

#### 출력 JSON 구조

```json
{
  "meta": {
    "label_col": "is_anomaly",
    "score_col": "mse",
    "n_total": 100,
    "n_attack": 50,
    "n_normal": 50
  },
  "attack": {
    "count": 50,
    "mean": 0.0523,
    "std": 0.0234,
    "min": 0.0012,
    "max": 0.2134,
    "p50": 0.0456,
    "p90": 0.0891,
    "p95": 0.1234,
    "p99": 0.1890
  },
  "normal": {...},
  "group_stats": {...},
  "top_attack_windows": [...],
  "top_normal_windows": [...]
}
```

## 보조 모듈

### packet_feature_extractor.py

패킷 데이터를 feature 벡터로 변환하는 유틸리티입니다.

```python
from packet_feature_extractor import (
    load_preprocess_params,
    sequence_group_to_feature_matrix,
    PACKET_FEATURE_COLUMNS,
)

# 전처리 파라미터 로딩
params = load_preprocess_params(Path("../preprocessing/result"))

# 패킷 리스트 → Feature 행렬
X = sequence_group_to_feature_matrix(packets, params)
# X: [seq_len, feat_dim]
```

## 전체 파이프라인 요약

```
┌─────────────────────┐
│  attack.jsonl       │  (테스트 패킷 데이터)
└─────────┬───────────┘
          │
          ├─────────────────────────────────┐
          │                                 │
          ▼                                 ▼
┌─────────────────────┐         ┌─────────────────────┐
│  0.attack_result.py │         │   1.benchmark.py    │
│  (GT 라벨 생성)      │         │  (모델 추론)        │
└─────────┬───────────┘         └─────────┬───────────┘
          │                               │
          ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│  attack_result.csv  │         │  window_scores.csv  │
└─────────┬───────────┘         └─────────┬───────────┘
          │                               │
          └───────────┬───────────────────┘
                      │
                      ▼
          ┌─────────────────────┐
          │ 2.eval_detection_   │
          │    metrics.py       │
          │   (성능 평가)        │
          └─────────┬───────────┘
                    │
                    ▼
          ┌─────────────────────┐
          │   metrics.json      │
          │ (Accuracy, F1 등)   │
          └─────────────────────┘
```

## 필요 라이브러리

```
numpy
pandas
tensorflow>=2.0
```

## 주의사항

1. **window-size 일치**: 학습 시 사용한 window_size와 동일하게 설정
2. **step-size**: 비중첩 윈도우는 step-size = window-size, 중첩 윈도우는 더 작은 값 사용
3. **전처리 파라미터**: 학습 데이터로 생성된 파라미터를 사용해야 함
4. **threshold 선택**: p99 또는 mean+3std 중 데이터 특성에 맞는 것 선택
