# Train

LSTM Autoencoder 모델 학습을 위한 모듈입니다.

## 개요

이 모듈은 전처리된 패턴 윈도우 데이터를 패딩 처리하고, LSTM Autoencoder 모델을 학습시켜 이상 탐지에 사용합니다.

## 폴더 구조

```
train/
├── code/
│   ├── 1.padding.py        # 윈도우 데이터 패딩 처리
│   └── 2.LSTM_AE.py         # LSTM Autoencoder 학습
└── result/                  # 학습 결과 저장 디렉토리
```

## 실행 순서

### 1단계: 패딩 처리 (`1.padding.py`)

가변 길이의 윈도우 데이터를 고정 길이로 패딩합니다.

```bash
# 0-padding (기본)
python 1.padding.py \
    -i "../data/pattern_features.jsonl" \
    -o "../result/pattern_features_padded_0.jsonl" \
    --pad_value 0 \
    --window_size 76

# -1 padding (마스킹용)
python 1.padding.py \
    -i "../data/pattern_features.jsonl" \
    -o "../result/pattern_features_padded_-1.jsonl" \
    --pad_value -1 \
    --window_size 76

# 특정 feature 제거
python 1.padding.py \
    -i "../data/pattern_features.jsonl" \
    -o "../result/pattern_features_padded_0.jsonl" \
    --pad_value 0 \
    --window_size 76 \
    --drop_keys delta_t
```

#### 파라미터

| 파라미터 | 필수 | 기본값 | 설명 |
|----------|------|--------|------|
| `-i`, `--input_jsonl` | O | - | 입력 feature JSONL 경로 |
| `-o`, `--output_jsonl` | O | - | 출력 JSONL 경로 |
| `--window_size` | X | 자동 | 윈도우 길이 (미지정 시 max(index)+1) |
| `--pad_value` | X | -1.0 | 패딩 값 (0.0 또는 -1.0 권장) |
| `--drop_keys` | X | [] | 제거할 feature key 리스트 |

#### 패딩 방식

- **Index 기반 패딩**: 원본 데이터의 `index` 배열 위치에 맞게 패킷을 배치
- **빈 위치**: `pad_value`로 채움
- **출력 구조**:
  - `index`: [0, 1, ..., window_size-1]
  - `sequence_group`: 길이 window_size인 리스트

---

### 2단계: LSTM Autoencoder 학습 (`2.LSTM_AE.py`)

Keras/TensorFlow 기반 LSTM Autoencoder를 학습합니다.

```bash
# 기본 학습
python 2.LSTM_AE.py \
    -i "../result/pattern_features_padded_0.jsonl" \
    -o "../result/LSTM_AE" \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 128 \
    --latent_dim 64 \
    --pad_value 0.0 \
    --device cuda \
    --seed 42

# -1 패딩 데이터 학습
python 2.LSTM_AE.py \
    -i "../result/pattern_features_padded_-1.jsonl" \
    -o "../result/LSTM_AE(-1)" \
    --epochs 100 \
    --batch_size 128 \
    --hidden_dim 128 \
    --latent_dim 64 \
    --pad_value -1.0 \
    --device cuda \
    --seed 42
```

#### 파라미터

| 파라미터 | 필수 | 기본값 | 설명 |
|----------|------|--------|------|
| `-i`, `--input_jsonl` | O | - | 패딩된 JSONL 경로 |
| `-o`, `--output_dir` | O | - | 모델/로그 저장 디렉토리 |
| `--epochs` | X | 50 | 학습 epoch 수 |
| `--batch_size` | X | 64 | 배치 크기 |
| `--hidden_dim` | X | 128 | LSTM hidden dimension |
| `--latent_dim` | X | 64 | Latent space dimension |
| `--num_layers` | X | 1 | LSTM layer 수 |
| `--bidirectional` | X | False | 양방향 LSTM 사용 여부 |
| `--lr` | X | 1e-3 | Learning rate |
| `--val_ratio` | X | 0.2 | Validation 데이터 비율 |
| `--pad_value` | X | 0.0 | 패딩 값 (loss 마스킹용) |
| `--device` | X | cuda | 디바이스 (cuda/cpu) |
| `--seed` | X | 42 | 랜덤 시드 |

## 모델 아키텍처

```
┌─────────────┐
│   Input     │ (batch, T, D)
│   (Encoder) │
└──────┬──────┘
       │ LSTM
       ▼
┌─────────────┐
│   Latent    │ (batch, latent_dim)
└──────┬──────┘
       │ Dense
       ▼
┌─────────────┐
│   Repeat    │ (batch, T, latent_dim)
└──────┬──────┘
       │ LSTM (Decoder)
       ▼
┌─────────────┐
│   Output    │ (batch, T, D)
│TimeDistributed
└─────────────┘
```

## 출력 파일

학습 완료 후 `output_dir`에 다음 파일들이 생성됩니다:

| 파일명 | 설명 |
|--------|------|
| `model.h5` | Keras 모델 파일 |
| `config.json` | 학습 설정 및 데이터 차원 정보 (N, T, D) |
| `feature_keys.txt` | Feature 이름 순서 (한 줄에 하나) |
| `train_log.json` | Epoch별 train/val loss 기록 |
| `threshold.json` | 이상 탐지 임계값 (p99, mean+3std) |
| `loss_curve.png` | Loss 곡선 그래프 |

### threshold.json 구조
```json
{
  "threshold_p99": 0.0234,
  "threshold_mu3": 0.0189,
  "stats": {
    "mean": 0.0052,
    "std": 0.0046,
    "min": 0.0001,
    "max": 0.0567
  }
}
```

## 손실 함수

패딩된 timestep을 마스킹하는 Masked MSE Loss를 사용합니다:

- 모든 feature가 `pad_value`인 timestep은 loss 계산에서 제외
- 유효한 timestep만 사용하여 평균 MSE 계산

## 입력 데이터 형식

```json
{
  "window_id": 1,
  "pattern": "normal_pattern",
  "index": [0, 1, 2, ..., 75],
  "sequence_group": [
    {"protocol": 4.0, "delta_t": 0.001, "src_host_id": 1.0, ...},
    {"protocol": 0.0, "delta_t": 0.0, ...},  // 패딩된 timestep
    ...
  ]
}
```

## 필요 라이브러리

```
numpy
tensorflow>=2.0
matplotlib
```

## 주의사항

1. **GPU 메모리**: 대용량 데이터 학습 시 `batch_size`를 조절하세요
2. **Early Stopping**: 5 epoch 동안 val_loss 개선이 없으면 자동 중단
3. **패딩 값 일치**: 학습 시 사용한 `pad_value`를 추론 시에도 동일하게 사용해야 합니다
4. **시드 고정**: 재현 가능한 결과를 위해 `--seed`를 지정하세요

## 다음 단계

학습된 모델을 사용한 탐지 및 평가는 `result_train` 폴더의 코드를 사용합니다.
