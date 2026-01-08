# `DATA` 폴더

## 1. 폴더 개요 (Folder Overview)

이 폴더는 데이터 파이프라인의 초기 단계에서 사용되는 스크립트를 포함하고 있습니다. 주요 역할은 분산된 로그 데이터를 하나의 파일로 병합하는 것입니다. `modbus` 및 `xgt_fen` 프로토콜과 같이 여러 패킷으로 나뉘어 수집될 수 있는 로그들을 연속성(동일 타임스탬프 및 시퀀스 번호)을 기준으로 그룹화하고, 관련 필드를 통합하여 단일 JSONL 객체로 만듭니다.

## 2. 파일별 설명 (File Descriptions)

| 파일명 | 설명 |
| --- | --- |
| `data_merge.py` | JSONL 형식의 입력 파일을 읽어 특정 프로토콜(`modbus`, `xgt_fen`)의 연속적인 패킷들을 병합하는 메인 스크립트입니다. 설정 파일(`data_merge.yaml`)에 정의된 작업 목록에 따라 동작합니다. |
| `data_merge.yaml` | `data_merge.py` 스크립트의 설정 파일입니다. 병합할 데이터의 입력 경로와 병합된 결과를 저장할 출력 경로, 인코딩 방식 등을 지정하는 'job' 목록을 정의합니다. |

## 3. 작업 흐름 (Workflow)

1.  **설정 정의**: `data_merge.yaml` 파일에 병합할 데이터의 위치(`input`)와 결과를 저장할 위치(`output`)를 포함한 하나 이상의 작업을 정의합니다.
2.  **스크립트 실행**: `data_merge.py`를 실행합니다.
    ```bash
    python data_merge.py [data_merge.yaml 경로]
    ```
    *   YAML 파일 경로를 지정하지 않으면 스크립트와 동일한 위치에 있는 `data_merge.yaml`을 기본값으로 사용합니다.
3.  **병합 처리**: 스크립트는 설정 파일에 명시된 각 작업에 대해 다음을 수행합니다.
    *   입력 JSONL 파일을 한 줄씩 읽습니다.
    *   `protocol`, `@timestamp`, `sq` 필드를 기준으로 연속적인 패킷들을 식별하고 그룹화합니다.
    *   그룹화된 패킷들을 하나의 JSON 객체로 병합합니다.
    *   병합된 결과를 출력 JSONL 파일에 씁니다.
4.  **결과 확인**: 작업이 완료되면 `data_merge.yaml`에 지정된 `output` 경로에서 병합된 파일을 확인할 수 있습니다.

---
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

---
# `PLS-JSONL` 폴더

## 1. 폴더 개요 (Folder Overview)

이 폴더는 원본 데이터(`RAW.jsonl`)와 패턴 분석 시스템(SLM)의 출력물(`PLS.jsonl`, `window_pls_80.jsonl`)을 결합하고 가공하여 후속 분석 단계에 필요한 데이터를 생성하는 파이프라인을 포함합니다.

주요 목표는 두 가지입니다:
1.  SLM이 탐지한 추상적인 패턴 시퀀스(`PLS`)를 실제 원본 패킷 데이터(`RAW`)로 변환합니다.
2.  각 패턴 시퀀스에 포함된 패킷들이 원래의 전체 트래픽 윈도우 내에서 몇 번째 위치에 있었는지, 그 인덱스 정보를 찾아 추가합니다.

최종적으로 레이블, 원본 패킷 시퀀스, 그리고 위치 인덱스 정보가 모두 포함된 `pattern.jsonl` 파일을 생성하여 다음 전처리 단계로 전달합니다.

## 2. 파일별 설명 (File Descriptions)

| 파일명 | 설명 |
| --- | --- |
| `PLS-JSONL.py` | 전체 파이프라인을 순차적으로 실행하는 메인 오케스트레이터 스크립트입니다. `PLS-JSONL.yaml` 설정 파일을 읽어 각 단계를 실행합니다. |
| `PLS-JSONL.yaml` | 파이프라인의 각 단계에서 사용할 스크립트, 입력 파일, 출력 파일의 경로를 정의하는 설정 파일입니다. |
| `1.PLS_to_RAW.py` | SLM의 패턴 분석 결과(`PLS.jsonl`)에 포함된 각 시퀀스를 원본 RAW 패킷으로 매핑합니다. `@timestamp`, `sq`, `ak`, `fl`을 키로 사용하여 매칭합니다. |
| `2.PLS_to_RAW_windows.py` | SLM이 분석에 사용한 전체 트래픽 윈도우(`window_pls_80.jsonl`)를 원본 RAW 패킷으로 재구성합니다. `@timestamp`를 키로 사용하여 매칭하며, 윈도우 내 모든 패킷이 존재해야 유효한 것으로 간주합니다. |
| `3.window_index.py` | `1.PLS_to_RAW.py`의 결과물(패턴)과 `2.PLS_to_RAW_windows.py`의 결과물(전체 윈도우)을 비교하여, 패턴 내 각 패킷이 전체 윈도우에서 몇 번째 인덱스에 위치하는지 찾아내고, 이 정보를 `"index"` 필드로 추가합니다. |

## 3. 작업 흐름 (Workflow)

1.  **설정 확인**: `PLS-JSONL.yaml` 파일에 각 단계에 필요한 입출력 파일 경로가 올바르게 지정되었는지 확인합니다.

2.  **파이프라인 실행**: 터미널에서 `PLS-JSONL.py`를 실행합니다.
    ```bash
    python PLS-JSONL.py --config PLS-JSONL.yaml
    ```

3.  **프로세스 진행**:
    *   **1단계 (패턴-RAW 매핑)**: `1.PLS_to_RAW.py`가 실행됩니다.
        *   `PLS.jsonl`의 레이블된 패턴 시퀀스를 읽습니다.
        *   `RAW.jsonl`의 원본 패킷과 매칭하여, 레이블 정보와 원본 패킷 시퀀스가 포함된 `PLS_to_RAW_mapped.jsonl`을 생성합니다.
    *   **2단계 (윈도우-RAW 재구성)**: `2.PLS_to_RAW_windows.py`가 실행됩니다.
        *   `window_pls_80.jsonl`의 전체 윈도우 정보를 읽습니다.
        *   `RAW.jsonl`의 원본 패킷과 매칭하여, `window_id`별로 모든 원본 패킷이 포함된 `PLS_to_RAW_windows_mapped.jsonl`을 생성합니다.
    *   **3단계 (인덱스 추가)**: `3.window_index.py`가 실행됩니다.
        *   1단계 결과물(`PLS_to_RAW_mapped.jsonl`)과 2단계 결과물(`PLS_to_RAW_windows_mapped.jsonl`)을 로드합니다.
        *   각 패턴의 `window_id`를 기준으로, 패턴에 속한 패킷들이 전체 윈도우의 몇 번째에 위치하는지 인덱스를 계산합니다.
        *   계산된 인덱스 리스트를 `"index"` 필드로 추가하여 최종 결과물인 `pattern.jsonl`을 생성합니다.

4.  **최종 결과물**: `preprocessing` 폴더에서 사용할 `pattern.jsonl` 파일이 생성됩니다.

---
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

---
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

---
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
