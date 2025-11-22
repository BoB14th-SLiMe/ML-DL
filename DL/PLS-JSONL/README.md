# PLS-JSONL

PLS(Pattern Learning System) 패턴 데이터와 RAW 패킷 데이터를 매핑하여 학습용 윈도우 데이터를 생성하는 전처리 모듈입니다.

## 개요

이 모듈은 PLS에서 추출된 패턴 시퀀스(`sequence_group`)와 원본 RAW 패킷 데이터를 timestamp 또는 timestamp+sq 기준으로 매핑합니다.

## 폴더 구조

```
PLS-JSONL/
├── code/
│   ├── 0.prepare.py              # 데이터 준비 상태 점검
│   ├── 1-1.PLS_to_Raw(timestamp).py    # timestamp 기준 매핑
│   └── 1-2.PLS_to_Raw(timestamp_sq).py # timestamp + sq 기준 매핑
└── config/
    ├── 0.prepare.yaml            # prepare 설정 파일
    └── 1.PLS_to_Raw.yaml         # PLS_to_Raw 설정 파일
```

## 실행 순서

### 1. 데이터 준비 점검 (`0.prepare.py`)

입력 데이터 파일의 존재 여부와 JSONL 형식 유효성을 검사합니다.

```bash
python 0.prepare.py -c "../config/0.prepare.yaml"
```

**config 파일 필수 항목:**
```yaml
pipeline:
  PLS_file: "path/to/PLS_patterns.jsonl"   # PLS 패턴 파일 경로
  RAW_file: "path/to/RAW_packets.jsonl"    # RAW 패킷 파일 경로
  result_dir: "path/to/output"             # 결과 저장 디렉토리
  prepare_log: "path/to/prepare.log"       # 로그 파일 경로
```

**검사 항목:**
- 파일 존재 여부
- 파일 비어있는지 확인
- JSONL 파싱 가능 여부
- 총 라인 수 카운트

---

### 2. PLS to Raw 매핑

두 가지 매핑 방식 중 하나를 선택하여 실행합니다.

#### 2-1. Timestamp 기준 매핑 (`1-1.PLS_to_Raw(timestamp).py`)

`@timestamp` 필드만 사용하여 PLS 패턴과 RAW 패킷을 매핑합니다.

```bash
python "1-1.PLS_to_Raw(timestamp).py" -c "../config/1.PLS_to_Raw.yaml"
```

#### 2-2. Timestamp + SQ 기준 매핑 (`1-2.PLS_to_Raw(timestamp_sq).py`)

`@timestamp`와 `sq`(sequence number) 필드를 모두 사용하여 더 정확한 매핑을 수행합니다.

```bash
python "1-2.PLS_to_Raw(timestamp_sq).py" -c "../config/1.PLS_to_Raw.yaml"
```

**config 파일 필수 항목:**
```yaml
pipeline:
  PLS_file: "path/to/PLS_patterns.jsonl"
  RAW_file: "path/to/RAW_packets.jsonl"
  1-1_result_file: "path/to/result_1-1.jsonl"   # timestamp 매핑 결과
  1-2_result_file: "path/to/result_1-2.jsonl"   # timestamp+sq 매핑 결과
  1-1_log_file: "path/to/1-1.log"
  1-2_log_file: "path/to/1-2.log"
```

## 입력 데이터 형식

### PLS 패턴 파일 (JSONL)
```json
{
  "window_id": 1,
  "label": "normal",
  "sequence_group": [
    {"@timestamp": "2025-01-01T00:00:00.000Z", "sq": 1, ...},
    {"@timestamp": "2025-01-01T00:00:00.001Z", "sq": 2, ...}
  ]
}
```

### RAW 패킷 파일 (JSONL)
```json
{
  "@timestamp": "2025-01-01T00:00:00.000Z",
  "sq": 1,
  "protocol": "modbus",
  "smac": "00:11:22:33:44:55",
  "sip": "192.168.0.1",
  ...
}
```

## 출력 데이터 형식

```json
{
  "window_id": 1,
  "label": "normal",
  "window_group": [
    {
      "@timestamp": "2025-01-01T00:00:00.000Z",
      "protocol": "modbus",
      "smac": "...",
      "match": "O",
      ...
    }
  ]
}
```

- `match: "O"`: 매핑 성공을 표시

## 필요 라이브러리

```
orjson
pyyaml
tqdm
```

## 주의사항

1. PLS 파일과 RAW 파일의 timestamp 형식이 일치해야 합니다
2. 대용량 파일 처리 시 메모리 사용량에 주의하세요 (RAW 패킷을 메모리에 인덱싱)
3. timestamp+sq 매핑이 더 정확하지만, sq 필드가 없는 경우 timestamp 매핑을 사용하세요
