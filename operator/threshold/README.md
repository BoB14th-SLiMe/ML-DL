# Threshold Calculation Utility

## 1. 개요 (Overview)

이 유틸리티는 DL-Anomaly 모델의 재구성 오류(reconstruction error) 통계 정보가 담긴 `threshold.json` 파일을 기반으로, 새로운 이상 징후 탐지 임계값(threshold)을 계산하는 스크립트입니다.

## 2. 파일별 상세 설명 (File Descriptions)

-   **`threshold.py`**: `threshold.json` 파일에 저장된 분포의 평균(mean)과 표준편차(std)를 읽어옵니다. 사용자가 지정한 `k` 값을 이용하여 `평균 - k * 표준편차` 공식으로 새로운 임계값을 계산하고, 그 결과를 별도의 JSON 파일로 저장합니다.

## 3. 실행 및 의존성 (Execution & Deps)

### 실행 방법

아래와 같이 CLI에서 직접 실행하여 새로운 임계값 설정 파일을 생성할 수 있습니다.

```bash
# 예시: k=0.3을 적용하여 새로운 임계값 파일 생성
python threshold.py --threshold-json ../data/threshold.json --k 0.3
```

-   `--threshold-json`: 통계 정보가 담긴 소스 파일 경로를 지정합니다.
-   `--k`: 표준편차에 곱해질 계수(k)를 지정합니다. 이 값이 클수록 임계값은 낮아져 탐지에 더 민감해집니다.
-   `--include-original`: 결과 파일에 원본 `threshold.json` 내용을 포함할지 여부를 결정합니다.

### 주요 의존성 (Primary Dependencies)

-   `argparse`: CLI 인자를 파싱하기 위해 사용됩니다.
-   `dataclasses`: 데이터 구조를 정의하기 위해 사용됩니다.
