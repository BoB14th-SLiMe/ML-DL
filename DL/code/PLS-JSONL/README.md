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
