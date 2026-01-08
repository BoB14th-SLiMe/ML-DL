# Model Prediction Package

## 1. 개요 (Overview)

`model_predict` 패키지는 `model_load` 패키지를 통해 로드된 모델들을 사용하여 실제 예측을 수행하는 로직을 포함합니다. 입력 데이터(윈도우)에 대한 이상 점수 계산, 패턴 분류, 최종 위험도 산출 등의 기능을 제공합니다.

## 2. 파일별 상세 설명 (File Descriptions)

-   **`ML_predict.py`**: 로드된 Scikit-learn 모델을 사용하여 각 패킷의 이상 확률을 계산하고, 주요 기여 특징(Top-K)을 분석하는 `MLAnomalyProbEngine` 클래스를 제공합니다.
-   **`ae_predict.py`**: LSTM Autoencoder 모델을 사용하여 입력 윈도우의 재구성 오류(reconstruction error)를 계산하고, 이를 통해 이상 점수(MSE)와 이상 여부를 판단합니다.
-   **`pattern_predict.py`**: LSTM 분류 모델을 사용하여 입력 윈도우가 어떤 공격/정상 패턴에 해당하는지 예측합니다. 각 패턴에 대한 확률과 유사도 점수를 계산합니다.
-   **`DL_predict.py`**: `ae_predict`와 `pattern_predict`의 결과를 종합하여 최종적인 판단을 내리는 모듈입니다. 이상 점수, 패턴 확률, 자산 정보 등을 바탕으로 정교한 위험도(Risk Score)를 산출하고, 상용 수준의 탐지 결과를 생성합니다.

## 3. 실행 및 의존성 (Execution & Deps)

이 패키지의 모듈들은 `pipeline.py`에서 호출되어 사용되며, 직접 실행되지 않습니다.

### 주요 의존성 (Primary Dependencies)

-   `numpy`: 예측 과정에서의 모든 수치 연산에 사용됩니다.
-   `tensorflow`: `ae_predict.py`에서 LSTM Autoencoder 모델 예측을 위해 필요합니다.
-   `torch`: `pattern_predict.py`에서 LSTM 분류 모델 예측을 위해 필요합니다.
