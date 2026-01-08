# Preprocessing

## 1. 폴더 개요 (Folder Overview)

이 `preprocessing` 폴더는 전체 데이터 파이프라인에서 원시 데이터(Raw Data)를 머신러닝 모델이 학습할 수 있는 형태의 피처(Feature)로 변환하는 **피처 엔지니어링(Feature Engineering)** 단계를 담당합니다.

네트워크 패킷의 원시 로그(JSONL 형식)를 입력으로 받아, 각 패킷의 프로토콜(TCP, UDP, S7COMM, MODBUS 등) 특성에 맞는 정규화(Normalization) 및 수치 변환(Numerical Transformation)을 수행합니다. 최종적으로 모델의 입력으로 사용될 수 있는 고정된 형태의 피처 벡터(Feature Vector)를 생성합니다.

- **Input**: `RAW.jsonl` (패킷 단위의 원시 로그)
- **Process**: 피처 추출, 정규화, 수치 변환
- **Output**: `preprocessing.jsonl` (정규화된 피처 벡터)

## 2. 파일별 상세 설명 (File Descriptions)

| 파일명 | 설명 |
| --- | --- |
| `preprocessing.py` | **메인 전처리 실행 스크립트**입니다. `preprocessing.yaml` 설정 파일을 기반으로 동작하며, 다음과 같은 작업을 수행합니다:<br/>- 원시 패킷 데이터(JSONL)를 한 줄씩 읽어들입니다.<br/>- 각 패킷의 프로토콜을 식별하고, 해당 프로토콜에 맞는 피처 추출 함수를 호출합니다.<br/>- 사전에 계산된 통계치(`pre_dir` 내의 `*_norm_params.json` 파일)를 사용하여 각 피처를 Min-Max 정규화합니다.<br/>- MAC 주소, IP, 변수명 등 문자열 데이터를 사전 정의된 ID(`*_map.json`, `*_vocab.json`)로 변환합니다.<br/>- 최종 피처 벡터를 JSONL 파일로 저장합니다. |
| `preprocessing.yaml` | **전처리 스크립트의 설정 파일**입니다.<br/>- 데이터 입/출력 경로, 사전 학습된 통계치 및 어휘 파일이 저장된 디렉토리(`pre_dir`) 경로를 지정합니다.<br/>- `feature_weights.txt` 파일 생성 여부 등 스크립트의 동작 옵션을 제어합니다. |

## 3. 데이터 흐름 (Data Flow)

1.  **사전 준비 (Prerequisites)**
    - `preprocessing.yaml` 파일의 `paths.pre_dir` 경로에 전처리에 필요한 통계 및 매핑 파일들이 준비되어 있어야 합니다. 이 파일들은 일반적으로 학습 데이터셋 전체를 분석하여 미리 생성됩니다.
      - `common_host_map.json`: 호스트(MAC, IP)와 숫자 ID 매핑 정보
      - `*_norm_params.json`: 각 프로토콜별 피처(길이, 포트 등)의 Min-Max 값
      - `*_vocab.json`: 프로토콜 내 변수명과 숫자 ID 매핑 정보

2.  **실행 (Execution)**
    - `preprocessing.py` 스크립트를 실행합니다.
    - 스크립트는 `preprocessing.yaml`에 정의된 `input_jsonl` 파일을 읽습니다.

3.  **변환 (Transformation)**
    - 스크립트는 각 패킷에 대해 다음을 수행합니다:
      - 프로토콜 식별
      - 공통 피처(IP/MAC, 길이, 방향 등) 추출 및 정규화
      - 프로토콜별 특화 피처(e.g., `modbus.fc`, `s7comm.fn`) 추출 및 정규화
      - 모든 피처를 결합하여 하나의 JSON 객체(피처 벡터)로 생성

4.  **출력 (Output)**
    - 변환된 피처 벡터들이 `preprocessing.yaml`에 정의된 `output_jsonl` 경로에 새로운 JSONL 파일로 저장됩니다.
    - `options.write_feature_weights`가 `true`일 경우, 모델에서 사용할 피처 목록과 가중치를 담은 `feature_weights.txt` 파일이 함께 생성됩니다.

## 4. 주요 의존성 (Dependencies)

- **PyYAML**: `preprocessing.yaml` 설정 파일을 파싱하기 위해 사용됩니다.
- **Standard Libraries**: `json`, `argparse`, `re`, `pathlib` 등 Python 표준 라이브러리를 사용합니다.

*별도의 `requirements.txt`는 없으나, `PyYAML` 라이브러리가 설치되어 있어야 합니다.*
```bash
pip install pyyaml
```
