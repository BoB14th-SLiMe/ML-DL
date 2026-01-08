# `result_train` 폴더

## 1. 폴더 개요 (Folder Overview)

이 폴더는 사전 학습된 오토인코더(Autoencoder) 모델을 사용하여 데이터셋에 대한 이상 탐지(Anomaly Detection) 추론 및 평가를 수행하는 파이프라인을 포함합니다. 폴더 이름과 달리 모델을 '학습'하는 기능은 없으며, 이미 존재하는 모델을 '사용'하여 결과를 도출하는 데 초점을 맞춥니다.

주요 워크플로우는 YAML 설정 파일에 정의된 여러 데이터셋에 대해 동일한 모델을 실행하고, 각 데이터셋에 대한 재구성 오류(Reconstruction Error)를 계산하여 이상 여부를 판별합니다. 최종적으로 각 실행에 대한 상세한 성능 지표(Precision, Recall, F1-score), ROC 곡선, 재구성 오류 그래프 등 다양한 분석 결과물을 생성합니다.

## 2. 파일별 설명 (File Descriptions)

| 파일명 | 설명 |
| --- | --- |
| `1.result_train.py` | 전체 추론 및 평가 파이프라인을 실행하는 메인 오케스트레이터 스크립트입니다. `1.result_train.yaml` 설정을 기반으로 여러 평가 작업을 순차적으로 실행합니다. |
| `1.result_train.yaml` | 파이프라인의 실행 방법을 정의하는 설정 파일입니다. 사용할 모델의 경로, 평가할 여러 입력 데이터셋, 결과 저장 위치, 이상 탐지 임계값(threshold) 등의 파라미터를 지정합니다. |
| `1_1.common.py` | 추론 및 평가에 필요한 모든 공통 유틸리티 함수를 포함하는 모듈입니다. 데이터 로딩, 윈도우 생성, MSE 계산, 성능 지표 계산, 결과 시각화(그래프, ROC) 등의 핵심 로직이 담겨 있습니다. |
| `1_2.model.py` | 저장된 Keras 모델(`model.h5`)과 관련 메타데이터(피처 목록, 설정 등)를 로드하는 기능을 담당합니다. 오래된 버전의 Keras/TensorFlow로 학습된 모델을 호환성 문제없이 로드하기 위한 코드를 포함하고 있습니다. |
| `1_3.runner.py` | 단일 평가 작업(`run_one`)의 전체 프로세스를 담당하는 핵심 워커(worker) 스크립트입니다. 데이터 준비, 모델 추론, 재구성 오류 계산, 결과 저장 등 모든 단계를 수행합니다. |
| `0.attack_result.py` | `1.result_train.py` 실행 결과 생성된 `metrics_{tag}.json` 파일들을 취합하여 하나의 CSV 파일로 요약하는 유틸리티 스크립트입니다. |
| `0.attack_result.yaml` | `0.attack_result.py`가 취합할 `metrics` 파일들의 위치를 지정하는 설정 파일입니다. |

## 3. 작업 흐름 (Workflow)

1.  **설정 구성**: `1.result_train.yaml` 파일에 평가 작업을 정의합니다.
    *   `defaults`: 모든 작업에 공통적으로 적용될 파라미터(모델 경로, 기본 결과 폴더 등)를 설정합니다.
    *   `runs`: 평가할 개별 데이터셋 목록을 정의합니다. 각 항목은 고유한 `tag`와 `input` 데이터 파일 경로를 가집니다.

2.  **추론 및 평가 실행**: 터미널에서 `1.result_train.py`를 실행합니다.
    ```bash
    python 1.result_train.py --config 1.result_train.yaml
    ```

3.  **프로세스 진행 (각 `run` 항목에 대해 반복)**:
    *   **모델 로딩**: `1_2.model.py`를 통해 사전 학습된 오토인코더 모델과 관련 설정을 로드합니다.
    *   **데이터 준비**: `1_1.common.py`의 `build_windows` 함수가 입력 JSONL 파일을 읽어 모델에 입력할 수 있는 슬라이딩 윈도우(sliding window) 형태의 NumPy 배열로 변환합니다.
    *   **추론**: 모델이 입력 윈도우를 재구성(`predict`)합니다.
    *   **이상 점수 계산**: 원본 윈도우와 재구성된 윈도우 간의 평균 제곱 오차(MSE)를 계산합니다. 이 MSE 값이 '이상 점수(anomaly score)'가 됩니다.
    *   **이상 판별**: 계산된 MSE가 설정된 임계값(`threshold`)보다 크면 '이상(anomaly)', 작으면 '정상(normal)'으로 판별합니다.
    *   **결과 저장**: `output_dir`에 `tag`별로 다음과 같은 결과가 저장됩니다.
        *   `mse_per_window_{tag}.csv`: 모든 윈도우의 MSE 값.
        *   `metrics_{tag}.json`: 정밀도, 재현율, F1 점수 등 최종 성능 지표.
        *   `recon_error_{tag}.png`: 시간에 따른 재구성 오류를 시각화한 그래프.
        *   `roc_curve_{tag}.png`: ROC 곡선 그래프.

4.  **결과 요약 (선택 사항)**: 모든 평가 작업이 끝난 후, `0.attack_result.py`를 실행하여 생성된 모든 `metrics_{tag}.json` 파일의 핵심 내용을 하나의 CSV 파일로 취합할 수 있습니다.
    ```bash
    python 0.attack_result.py --config 0.attack_result.yaml
    ```
