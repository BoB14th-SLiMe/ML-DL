# `preprocessing` 폴더

## 1. 폴더 개요 (Folder Overview)

이 폴더는 `PLS-JSONL` 단계에서 생성된 `pattern.jsonl` 파일을 머신러닝 모델이 학습할 수 있는 최종적인 숫자 형식의 피처(feature)로 변환하는 전처리 파이프라인을 담당합니다.

프로세스는 크게 두 단계로 나뉩니다:
1.  **피팅(Fitting) 단계**: 전체 원본 데이터(`RAW.jsonl`)를 스캔하여 피처 엔지니어링에 필요한 각종 통계 정보(예: Min-Max 정규화를 위한 최솟값/최댓값, IP/MAC 주소와 고유 ID 매핑 테이블 등)를 계산하고 파일로 저장합니다.
2.  **변환(Transform) 단계**: `pattern.jsonl`의 각 패킷에 대해, 1단계에서 생성된 통계 정보를 이용해 정규화, 인코딩 등을 적용하여 최종 피처 벡터 시퀀스를 생성합니다.

최종적으로 각 패턴의 원본 패킷 시퀀스가 숫자 피처 시퀀스로 대체된 `preprocess_pattern.jsonl` 파일이 생성됩니다.

## 2. 파일별 설명 (File Descriptions)

| 파일명 | 설명 |
| --- | --- |
| `1.run_all_preprocess.py` | **피팅(Fitting) 단계**를 실행하는 오케스트레이터 스크립트입니다. `RAW.jsonl` 파일을 입력으로 받아, 하위의 프로토콜별 스크립트들을 순차적으로 실행시켜 피처 생성에 필요한 통계/매핑 파일들을 생성합니다. |
| `2.extract_feature.py` | **변환(Transform) 단계**를 실행하는 메인 스크립트입니다. `pattern.jsonl`과 `1.run_all_preprocess.py`가 생성한 통계 파일들을 입력으로 받아, 각 패턴의 패킷들을 최종 피처 벡터로 변환하고 `preprocess_pattern.jsonl` 파일로 저장합니다. |
| `preprocessing.yaml` | 이 폴더의 전체 워크플로우에 필요한 주요 파일 및 스크립트 경로를 정의하는 설정 파일입니다. (주로 참조용) |
| `common.py`, `arp.py`, `dns.py`, `modbus.py`, `s7comm.py`, `xgt-fen.py` | `1.run_all_preprocess.py`에 의해 호출되는 프로토콜별 피팅 스크립트입니다. 각자 담당하는 프로토콜의 필드를 분석하여 정규화에 필요한 파라미터(min/max 등)나 어휘(vocabulary)를 계산하고 JSON 파일로 저장합니다. |
| `preprocess_translated_addr_slot.py` | Modbus, XGT 등 특정 프로토콜의 메모리 주소(slot) 값에 대한 통계를 계산하는 피팅 스크립트입니다. |
| *기타 .py 파일* | 피처 엔지니어링에 사용되는 보조 스크립트 및 유틸리티입니다. |

## 3. 작업 흐름 (Workflow)

1.  **피팅 단계 실행**: `1.run_all_preprocess.py`를 실행하여 전체 원본 데이터로부터 피처 정규화 및 인코딩에 필요한 파라미터를 학습합니다.
    ```bash
    # <raw_data_path>: RAW.jsonl 파일 경로
    # <output_dir_path>: 통계/매핑 파일들이 저장될 디렉토리 경로
    python 1.run_all_preprocess.py --input <raw_data_path> --output <output_dir_path>
    ```
    *   이 명령은 `output_dir_path`에 `common_host_map.json`, `modbus_norm_params.json` 등 다수의 JSON 파일을 생성합니다.

2.  **변환 단계 실행**: `2.extract_feature.py`를 실행하여 패턴 데이터를 최종 피처 데이터로 변환합니다.
    ```bash
    # <pattern_data_path>: pattern.jsonl 파일 경로
    # <pre_dir_path>: 1단계에서 생성된 통계 파일들이 있는 디렉토리
    # <output_file_path>: 최종 결과물(preprocess_pattern.jsonl) 저장 경로
    python 2.extract_feature.py --input <pattern_data_path> --pre_dir <pre_dir_path> --output <output_file_path>
    ```
    *   이 스크립트는 `pattern.jsonl`의 각 패킷을 읽어 `pre_dir_path`의 통계 정보를 바탕으로 숫자 피처 벡터로 변환합니다.
    *   예를 들어, `ip.src` 주소는 `common_host_map.json`을 참조하여 숫자 ID로 바뀌고, `tcp.srcport`는 `common_norm_params.json`의 `sp_min`, `sp_max` 값을 이용해 0과 1 사이의 값으로 정규화됩니다.

3.  **최종 결과물**: `train` 폴더에서 사용할 `preprocess_pattern.jsonl` 파일과 모든 피처의 목록이 담긴 `feature_weights.txt` 파일이 생성됩니다.
