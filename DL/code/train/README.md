# `train` 폴더

## 1. 폴더 개요 (Folder Overview)

이 폴더는 이상 탐지를 위한 LSTM 기반 오토인코더(Autoencoder) 모델을 학습시키는 파이프라인을 포함합니다. `preprocessing` 단계에서 생성된 피처 벡터 시퀀스(`preprocess_pattern.jsonl`)를 입력으로 받아, 이를 재구성(reconstruct)하도록 모델을 학습시킵니다.

두 가지 모드를 지원합니다:
1.  **Normal**: 표준적인 LSTM 오토인코더 모델.
2.  **Bayesian**: TensorFlow Probability를 사용하여 모델의 가중치에 불확실성을 도입한 베이지안 LSTM 오토인코더 모델.

학습이 완료되면, 추론 및 평가(`result_train` 폴더)에 필요한 모든 산출물(모델 파일, 설정 파일, 피처 정보, 이상 탐지 임계값 등)을 지정된 출력 디렉토리에 저장합니다.

## 2. 파일별 설명 (File Descriptions)

| 파일명 | 설명 |
| --- | --- |
| `train.py` | 모델 학습 파이프라인의 메인 실행 파일입니다. `train.yaml` 설정을 읽어 `normal` 또는 `bayesian` 모드를 결정하고, `train_common.py`의 핵심 로직을 호출하여 학습을 실행합니다. |
| `train.yaml` | 모델 학습에 필요한 모든 설정을 정의하는 YAML 파일입니다. 학습 모드, 입/출력 경로, 윈도우 크기, 모델 하이퍼파라미터(은닉층 크기, 학습률 등)를 포함합니다. |
| `train_common.py` | 실제 학습 프로세스의 핵심 로직을 담고 있는 모듈입니다. 데이터 로딩, 피처 정책 적용, 모델 컴파일, `model.fit`을 통한 학습, 학습 후 산출물 저장 등 전체적인 학습 과정을 총괄합니다. |
| `padding.py` | 입력 데이터의 시퀀스를 고정된 `window_size`에 맞게 패딩하거나 잘라내는 유틸리티 함수를 포함합니다. |
| `train_normal.py`, `train_bayesian.py` | (사용되지 않음) 과거에 사용되었을 것으로 추정되는 스크립트. 현재 로직은 `train.py`와 `train_common.py`에 통합되어 있습니다. |

## 3. 작업 흐름 (Workflow)

1.  **설정 구성**: `train.yaml` 파일에 학습에 필요한 파라미터를 설정합니다.
    *   `mode`: `normal` 또는 `bayesian` 중 선택합니다.
    *   `input_jsonl`: `preprocessing` 단계의 최종 산출물인 `preprocess_pattern.jsonl` 파일 경로를 지정합니다.
    *   `output_dir`: 학습된 모델과 모든 결과물이 저장될 디렉토리 경로를 지정합니다.
    *   `feature_policy_file`: 학습에 사용할 피처를 선택하거나 가중치를 부여하기 위한 `feature_weights.txt` 파일 경로를 지정합니다.
    *   기타 하이퍼파라미터(epochs, batch_size, hidden_dim 등)를 설정합니다.

2.  **학습 실행**: 터미널에서 `train.py`를 실행합니다.
    ```bash
    python train.py
    ```
    *   스크립트는 같은 디렉토리에 있는 `train.yaml`을 자동으로 읽습니다.

3.  **프로세스 진행**:
    *   **데이터 로딩**: `train_common.py`가 `input_jsonl` 파일을 읽고, 각 시퀀스를 `window_size`에 맞게 패딩/절단하여 NumPy 배열로 변환합니다. 이 과정에서 `feature_policy_file`을 참조하여 특정 피처를 제외하거나 가중치를 적용할 준비를 합니다.
    *   **모델 생성**: `train.py`가 `mode` 설정에 따라 `normal` 또는 `bayesian` 모델을 생성합니다.
    *   **학습**: `model.fit`이 호출되어 학습 데이터(`X_train`)를 입력과 출력으로 삼아 모델을 학습시킵니다. 모델은 입력을 최대한 원본과 가깝게 재구성하도록 훈련됩니다. `EarlyStopping` 콜백을 사용하여 과적합을 방지합니다.
    *   **임계값 계산**: 학습이 완료된 모델을 사용하여 학습 데이터셋에 대한 재구성 오류(MSE)를 계산하고, 이 오류 분포를 기반으로 이상 탐지에 사용할 임계값(예: 99 백분위수)을 계산합니다.
    *   **산출물 저장**: `output_dir`에 다음과 같은 파일들이 저장됩니다.
        *   `model.h5`: 학습된 Keras 모델.
        *   `config.json`: 학습에 사용된 모든 하이퍼파라미터.
        *   `threshold.json`: 계산된 이상 탐지 임계값.
        *   `feature_keys.txt`: 모델이 학습한 피처의 순서 목록.
        *   `training_loss_curve.png`: 학습 손실 그래프.

4.  **최종 결과물**: `result_train` 폴더에서 사용할 모델 및 관련 파일들이 `output_dir`에 모두 생성됩니다.
