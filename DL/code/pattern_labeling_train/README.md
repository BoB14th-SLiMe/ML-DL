# `pattern_labeling_train` 폴더

## 1. 폴더 개요 (Folder Overview)

이 폴더는 시퀀스 데이터를 기반으로 패턴을 분류하는 LSTM 모델의 학습, 평가, 및 테스트 파이프라인을 포함합니다. 전체 프로세스는 YAML 설정 파일을 통해 제어되며, K-Fold 교차 검증, 조기 종료(Early Stopping), 학습 곡선 및 ROC 곡선 생성 등 모델 개발에 필요한 전반적인 기능을 제공합니다. 최종적으로 가장 성능이 좋은 모델을 저장하고 테스트 데이터셋에 대한 성능 리포트를 생성합니다.

## 2. 파일별 설명 (File Descriptions)

| 파일명 | 설명 |
| --- | --- |
| `train.py` | 모델 학습 파이프라인의 메인 실행 파일입니다. 데이터 로딩, 전처리, 교차 검증, 모델 학습, 평가, 결과 저장 등 전체 프로세스를 총괄합니다. |
| `train.yaml` | `train.py`에서 사용하는 모든 설정을 정의하는 YAML 파일입니다. 데이터 경로, 모델 하이퍼파라미터, 학습 옵션 등을 포함합니다. |
| `train_common.py` | 학습 파이프라인 전반에서 사용되는 공통 유틸리티 함수들을 모아놓은 모듈입니다. 데이터 로딩, 전처리, 시드 설정, 체크포인트 저장/로드, 학습 결과 시각화(학습 곡선, ROC 곡선) 등의 기능을 담당합니다. |
| `train_model.py` | PyTorch를 사용한 LSTM 모델(`LSTMClassifier`)의 아키텍처를 정의하고, 한 에폭(epoch) 동안의 학습(`train_one_epoch`) 및 평가(`eval_epoch`)를 수행하는 함수를 포함합니다. |

## 3. 작업 흐름 (Workflow)

1.  **설정 구성**: `train.yaml` 파일에 다음과 같은 주요 파라미터를 설정합니다.
    *   `paths`: 데이터, 피처 목록, 결과 저장 디렉토리 등의 경로.
    *   `data`: 시퀀스 최대 길이, 패딩 값, 테스트 데이터셋 비율 등 데이터 관련 설정.
    *   `training`: 에폭 수, 배치 사이즈, 학습률(learning rate), K-Fold 교차 검증의 폴드(fold) 수 등 학습 관련 설정.
    *   `model`: LSTM 모델의 은닉층(hidden dimension), 레이어 수, 드롭아웃 비율 등 모델 구조 관련 설정.
    *   `early_stopping`: 조기 종료를 위한 `patience` (성능 개선이 없을 때 기다릴 에폭 수) 및 기준 지표 설정.

2.  **학습 실행**: 터미널에서 `train.py`를 실행합니다.
    ```bash
    python train.py --config train.yaml
    ```

3.  **프로세스 진행**:
    *   **데이터 로딩 및 전처리**: `train_common.py`의 함수를 사용하여 JSONL 데이터와 피처 목록을 로드하고, 레이블을 인코딩하며, 시퀀스를 패딩/절단하여 모델 입력에 맞는 형태로 변환합니다.
    *   **데이터 분할**: 전체 데이터를 학습/검증(Train/Validation) 세트와 테스트(Test) 세트로 분리합니다. 학습/검증 세트는 K-Fold 교차 검증을 위해 다시 여러 폴드로 나뉩니다.
    *   **교차 검증 및 학습**:
        *   각 폴드에 대해 `train_model.py`의 `LSTMClassifier` 모델을 초기화합니다.
        *   `train_one_epoch` 함수로 모델을 학습시키고, `eval_epoch` 함수로 검증 데이터에 대한 성능(F1 점수 또는 손실)을 측정합니다.
        *   조기 종료 조건이 충족되면 해당 폴드의 학습을 중단합니다.
        *   모든 폴드를 통틀어 검증 성능이 가장 좋았던 시점의 모델을 "최고 모델"로 저장(`best_model.h5`, `best_model_meta.json`)합니다.
    *   **최종 평가**: 저장된 최고 모델을 불러와 테스트 세트에서 최종 성능을 평가합니다.
    *   **결과 저장**: 최종 테스트 성능 지표(`test_metrics.json`), 교차 검증 요약(`cv_summary.json`), 최고 모델의 학습 곡선(`learning_curve_best.png`), 테스트 ROC 곡선(`roc_curve_test.png`) 등 다양한 결과물을 `result_dir`에 저장합니다.
