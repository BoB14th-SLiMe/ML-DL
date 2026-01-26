# AI 기반 산업 제어 시스템 이상 탐지 시스템

## 1. 프로젝트 개요 (Project Overview)

본 프로젝트는 산업 제어 시스템(ICS)의 네트워크 트래픽을 실시간으로 분석하여, 머신러닝(ML) 및 딥러닝(DL) 모델을 통해 이상 징후를 탐지하는 AI 기반 보안 솔루션입니다. Modbus, S7comm 등 주요 산업용 프로토콜 데이터를 처리하며, 복합적인 모델 분석을 통해 정교한 위협 탐지 및 위험도 평가를 수행합니다.

## 2. 주요 기능 (Features)

-   **실시간 데이터 처리**: Redis 또는 JSONL 파일로부터 네트워크 패킷 데이터를 실시간으로 수집하고 처리합니다.
-   **다중 모델 기반 탐지 (Multi-Model Detection)**:
    -   **ML 모델 (Scikit-learn)**: 개별 패킷의 이상 확률을 계산합니다.
    -   **DL 이상 탐지 모델 (TensorFlow/Keras LSTM Autoencoder)**: 트래픽 시퀀스의 재구성 오류(Reconstruction Error)를 기반으로 이상 점수를 산출합니다.
    -   **DL 패턴 분류 모델 (PyTorch LSTM Classifier)**: 알려진 공격 또는 정상 트래픽 패턴을 분류하고 유사도를 측정합니다.
-   **정교한 위험도 산출**: 여러 모델의 탐지 결과를 종합하여 최종 위험 점수(Risk Score)를 계산하고, 탐지 결과의 신뢰도를 높입니다.
-   **모듈화된 파이프라인**: 데이터 수집, 전처리, 모델 추론, 결과 저장/알람에 이르는 전 과정이 체계적인 파이프라인으로 구성되어 있습니다.
-   **분리된 학습/운영 환경**: 모델 학습(`DL/`, `ML/`)과 실제 운영(`operator/`) 코드가 분리되어 유지보수 및 확장성을 확보했습니다.

## 3. 기술 스택 (Tech Stack)

-   **언어 (Language)**: Python
-   **ML/DL 프레임워크**:
    -   TensorFlow / Keras
    -   PyTorch
    -   Scikit-learn
-   **핵심 라이브러리**:
    -   NumPy
    -   Pandas
    -   PyYAML
    -   Joblib
    -   Redis
    -   Requests

## 4. 디렉토리 구조 (Directory Structure)

```
.
├── DL/                   # 딥러닝 모델 학습 및 평가 관련 코드
│   ├── code/             # 데이터 전처리, 모델 학습/평가 스크립트
│   └── utils/            # 유틸리티 함수
├── ML/                   # 머신러닝 모델 관련 코드
│   └── code/             # 데이터 전처리 스크립트
└── operator/             # 실시간 탐지 오퍼레이터 메인 코드
    ├── main.py           # 메인 실행 파일
    ├── main.yaml         # 오퍼레이터 설정 파일
    ├── pipeline.py       # 데이터 처리 핵심 파이프라인
    ├── model_load/       # 사전 학습된 모델 로딩 모듈
    ├── model_predict/    # 모델을 사용한 예측 수행 모듈
    ├── sources.py        # 데이터 소스(Redis, JSONL) 연동 모듈
    └── ...
```

-   **`operator/`**: 실시간 이상 탐지를 수행하는 메인 애플리케이션입니다. 데이터 소스에서 패킷을 읽어 전처리 후, 사전 학습된 모델을 로드하여 이상 징후를 예측합니다.
-   **`DL/`**: LSTM Autoencoder(이상 탐지), LSTM Classifier(패턴 분류) 등 딥러닝 모델의 학습, 평가, 결과 분석을 위한 전체 파이프라인을 포함합니다.
-   **`ML/`**: Scikit-learn 기반 머신러닝 모델의 학습에 필요한 전처리 코드를 포함합니다.

## 5. 시작 가이드 (Getting Started)

### 1. 의존성 설치 (Install Dependencies)

프로젝트 실행에 필요한 라이브러리들을 설치합니다. (가상환경 사용을 권장합니다)

```bash
pip install numpy pandas pyyaml joblib redis requests tensorflow torch scikit-learn
```

### 2. 설정 확인 (Check Configuration)

`operator/main.yaml` 파일을 열어 데이터 소스(`source_type`), 모델 경로(`model_paths`), 임계값 등 주요 파라미터가 현재 환경에 맞게 올바르게 설정되었는지 확인합니다.

```yaml
# operator/main.yaml 예시

# 데이터 소스 설정 (redis 또는 jsonl)
source_type: jsonl
# ...

# 모델 파일 경로 설정
model_paths:
  dl_anomaly: "path/to/your/dl_anomaly_model_dir"
  dl_pattern: "path/to/your/dl_pattern_model_dir"
  ml: "path/to/your/ml_model_dir"
# ...
```

### 3. 오퍼레이터 실행 (Run Operator)

아래 명령어를 통해 이상 탐지 오퍼레이터를 실행합니다.

```bash
python operator/main.py --config operator/main.yaml
```

-   `--config`: 사용할 설정 파일을 지정합니다. 기본값은 `operator/main.yaml` 입니다.
-   오퍼레이터가 실행되면 설정된 데이터 소스로부터 데이터를 수신하여 이상 탐지 파이프라인을 수행합니다.