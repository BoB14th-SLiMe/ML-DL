# Model Loading Package

## 1. 개요 (Overview)

`model_load` 패키지는 사전에 훈련된 머신러닝 및 딥러닝 모델 파일들을 디스크에서 읽어와 메모리에 로드하는 역할을 담당합니다. 각 스크립트는 특정 유형의 모델(ML, DL-Anomaly, DL-Pattern)을 로드하는 데 특화되어 있습니다.

## 2. 파일별 상세 설명 (File Descriptions)

-   **`loader_ml.py`**: Scikit-learn 기반의 머신러닝 모델을 로드합니다. `joblib`을 사용하여 직렬화된 모델 파일(`model.pkl`), 스케일러(`scaler.pkl`), 그리고 사용된 특징 목록(`selected_features.json`)을 읽어옵니다.
-   **`loader_dl_anomaly.py`**: Keras/TensorFlow로 구현된 LSTM Autoencoder 이상 징후 탐지 모델을 로드합니다. 모델 구조 파일(`model.h5`), 설정(`config.json`), 특징 정보(`feature_keys.txt`), 임계값(`threshold.json`) 등을 읽어와 모델 번들(bundle)을 구성합니다.
-   **`loader_dl_pattern.py`**: PyTorch로 구현된 LSTM 기반 패턴 분류 모델을 로드합니다. 모델의 state dictionary, 구조 정보, 특징 및 레이블 정보가 포함된 체크포인트 파일(`.pt`, `.pth`)을 로드하여 모델 번들을 생성합니다.
-   **`__init__.py`**: 이 디렉토리를 파이썬 패키지로 인식하도록 합니다.

## 3. 실행 및 의존성 (Execution & Deps)

이 패키지의 모듈들은 외부(`models.py`)에서 임포트하여 사용하는 라이브러리 형태이며, 직접 실행되지 않습니다.

### 주요 의존성 (Primary Dependencies)

-   `joblib` & `scikit-learn`: `loader_ml.py`에서 ML 모델을 로드하기 위해 필요합니다.
-   `numpy`: 모든 로더에서 데이터 처리를 위해 사용됩니다.
-   `tensorflow`: `loader_dl_anomaly.py`에서 Keras 모델을 로드하기 위해 필요합니다.
-   `torch`: `loader_dl_pattern.py`에서 PyTorch 모델을 로드하기 위해 필요합니다.
