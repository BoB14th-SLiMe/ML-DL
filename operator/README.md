# AI-based Anomaly Detection Operator

## 1. 개요 (Overview)

본 프로젝트는 네트워크 패킷 데이터를 실시간으로 처리하여 머신러닝(ML) 및 딥러닝(DL) 모델을 통해 이상 징후를 탐지하는 오퍼레이터입니다. 수집된 데이터는 전처리, 특징 추출, 모델 예측의 파이프라인을 거쳐 최종적으로 이상 탐지 결과를 생성하고, 필요시 알람을 발생시킵니다.

## 2. 파일별 상세 설명 (File Descriptions)

-   **`main.py`**: 프로젝트의 메인 실행 파일입니다. 설정 로드, 모델 preload, 파이프라인 실행 등 전체 프로세스를 총괄합니다.
-   **`main.yaml`**: 프로젝트의 주요 설정을 담고 있는 YAML 파일입니다. 데이터 소스, 모델 경로, 윈도우 크기, 임계값 등 다양한 파라미터를 정의합니다.
-   **`pipeline.py`**: 데이터 처리의 핵심 로직을 담고 있는 파이프라인 클래스(`OperationalPipeline`)가 정의되어 있습니다. 데이터 수집, 병합, 전처리, 모델 예측, 결과 저장 및 알람 전송의 흐름을 관리합니다.
-   **`config.py`**: `main.yaml` 설정 파일을 로드하고, 기본값과 병합하며, CLI 인자로 전달된 값을 오버라이드하는 기능을 제공합니다.
-   **`models.py`**: `model_load` 디렉토리의 로더들을 사용하여 ML, DL-Anomaly, DL-Pattern 세 종류의 모델을 로드하고 관리하는 함수를 제공합니다.
-   **`sources.py`**: 데이터 소스로부터 패킷을 가져오는 클래스(`RedisPopServer`, `JsonlPopServer`)를 정의합니다. Redis 스트림 또는 JSONL 파일로부터 데이터를 읽어옵니다.
-   **`io_writers.py`**: 파이프라인 실행 중 발생하는 각종 로그(수신 패킷, 재조립 전/후, 최종 결과)를 비동기적으로 파일에 쓰는 클래스들을 정의합니다.
-   **`utils.py`**: JSON 직렬화, 로거 설정, 시간 측정 등 프로젝트 전반에서 사용되는 유틸리티 함수들을 포함합니다.

## 3. 실행 및 의존성 (Execution & Deps)

### 실행 방법

프로젝트 루트 디렉토리에서 아래 명령어를 통해 파이프라인을 실행합니다.

```bash
python main.py --config main.yaml
```

-   `--config`: 사용할 YAML 설정 파일을 지정합니다. (기본값: `main.yaml`)
-   다양한 실행 옵션(예: `--input-jsonl`, `--pps`)은 `main.py`의 `build_arg_parser` 함수를 참조하세요.

### 주요 의존성 (Primary Dependencies)

-   `PyYAML`: `main.yaml` 설정 파일을 파싱하기 위해 필요합니다.
-   `numpy`: 데이터의 수치 연산을 위해 사용됩니다.
-   `redis`: Redis를 데이터 소스로 사용할 경우 필요합니다.
-   `requests`: 이상 징후 탐지 시 외부 API로 알람을 보내기 위해 사용됩니다.
-   `tensorflow`: DL-Anomaly (LSTM Autoencoder) 모델 실행을 위해 필요합니다.
-   `torch`: DL-Pattern (LSTM Classifier) 모델 실행을 위해 필요합니다.
-   `scikit-learn` / `joblib`: ML 모델을 로드하고 실행하기 위해 필요합니다.
