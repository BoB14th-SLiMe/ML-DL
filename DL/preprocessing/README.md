# Preprocessing

네트워크 패킷 데이터를 딥러닝 모델 학습에 적합한 feature로 변환하는 전처리 모듈입니다.

## 개요

이 모듈은 다양한 산업용 프로토콜(Modbus, S7comm, XGT-FEN, ARP, DNS 등)의 패킷 데이터를 정규화하고 feature를 추출합니다.

## 폴더 구조

```
preprocessing/
├── code/
│   ├── 1.run_all_preprocess.py       # 전체 전처리 통합 실행 스크립트
│   ├── 2.window_to_feature_csv.py    # 윈도우 → Feature CSV 변환
│   ├── 3.window_to_feature_csv_dynamic_index.py  # 동적 인덱스 기반 변환
│   ├── common.py                     # 공통 feature 전처리 (MAC, IP, Port 등)
│   ├── arp.py                        # ARP 프로토콜 전처리
│   ├── dns.py                        # DNS 프로토콜 전처리
│   ├── modbus.py                     # Modbus 프로토콜 전처리
│   ├── s7comm.py                     # S7comm 프로토콜 전처리
│   └── xgt-fen.py                    # XGT-FEN 프로토콜 전처리
└── config/
    ├── main.yaml
    ├── preprocessing.yaml
    └── exclude.yaml
```

## 실행 순서

### 1단계: 프로토콜별 전처리 파라미터 생성

#### 방법 1: 통합 실행 (권장)

모든 프로토콜 전처리를 한 번에 병렬로 실행합니다.

```bash
# 학습 데이터로 fit (파라미터 생성)
python 1.run_all_preprocess.py --input "../data/ML_DL_학습.jsonl" --output "../result" --mode fit

# 특정 프로토콜 제외
python 1.run_all_preprocess.py --input "../data/ML_DL_학습.jsonl" --output "../result" --mode fit --skip dns.py modbus.py

# worker 수 지정
python 1.run_all_preprocess.py --input "../data/ML_DL_학습.jsonl" --output "../result" --mode fit --workers 4
```

#### 방법 2: 개별 실행

각 프로토콜별로 개별 실행할 수 있습니다.

```bash
# Common (공통 feature)
python common.py --fit -i "../data/ML_DL_학습.jsonl" -o "../result"

# ARP
python arp.py --fit -i "../data/ML_DL_학습.jsonl" -o "../result"

# DNS
python dns.py --fit -i "../data/ML_DL_학습.jsonl" -o "../result"

# Modbus
python modbus.py --fit -i "../data/ML_DL_학습.jsonl" -o "../result"

# S7comm
python s7comm.py --fit -i "../data/ML_DL_학습.jsonl" -o "../result"

# XGT-FEN
python xgt-fen.py --fit -i "../data/ML_DL_학습.jsonl" -o "../result"
```

### 2단계: 윈도우 데이터를 Feature로 변환

#### 기본 변환 (`2.window_to_feature_csv.py`)

```bash
python 2.window_to_feature_csv.py \
    --input "../data/pattern_windows.jsonl" \
    --pre_dir "../result" \
    --output "../result/pattern_features.csv"
```

#### 동적 인덱스 기반 변환 (`3.window_to_feature_csv_dynamic_index.py`)

윈도우 크기를 고정하고 패딩을 적용합니다.

```bash
# 자동 window_size (최대 span 기반)
python 3.window_to_feature_csv_dynamic_index.py \
    --input "../data/pattern_windows.jsonl" \
    --pre_dir "../result" \
    --output "../result/pattern_features.csv"

# window_size 수동 지정
python 3.window_to_feature_csv_dynamic_index.py \
    --input "../data/pattern_windows.jsonl" \
    --pre_dir "../result" \
    --output "../result/pattern_features.csv" \
    --max-index 40
```

## 전처리 모드

각 프로토콜 스크립트는 두 가지 모드를 지원합니다:

| 모드 | 설명 |
|------|------|
| `--fit` | 학습 데이터로 정규화 파라미터(min/max) 생성 및 저장 |
| `--transform` | 기존 파라미터를 사용하여 새 데이터 변환 |

## 생성되는 파라미터 파일

| 파일명 | 설명 |
|--------|------|
| `common_host_map.json` | MAC+IP 조합 → Host ID 매핑 |
| `common_norm_params.json` | sp, dp, len의 min/max 값 |
| `arp_host_map.json` | ARP용 Host ID 매핑 |
| `dns_norm_params.json` | dns.qc, dns.ac min/max 값 |
| `modbus_norm_params.json` | modbus.addr, fc, qty, bc min/max 값 |
| `s7comm_norm_params.json` | s7comm.ros, db, addr min/max 값 |
| `xgt_var_vocab.json` | XGT 변수명 → ID 매핑 |
| `xgt_fen_norm_params.json` | XGT-FEN 정규화 파라미터 |

## Feature 컬럼 목록

### 공통 (Common) - 6개
- `src_host_id`, `dst_host_id`: 소스/목적지 호스트 ID
- `sp_norm`, `dp_norm`: 정규화된 소스/목적지 포트
- `dir_code`: 방향 (request=1, response=0)
- `len_norm`: 정규화된 패킷 길이

### S7comm - 4개
- `s7comm_ros_norm`, `s7comm_fn`, `s7comm_db_norm`, `s7comm_addr_norm`

### Modbus - 12개
- `modbus_addr_norm`, `modbus_fc_norm`, `modbus_qty_norm`, `modbus_bc_norm`
- `modbus_regs_count`, `modbus_regs_addr_min`, `modbus_regs_addr_max`, `modbus_regs_addr_range`
- `modbus_regs_val_min`, `modbus_regs_val_max`, `modbus_regs_val_mean`, `modbus_regs_val_std`

### XGT-FEN - 21개
- `xgt_var_id`, `xgt_var_cnt`, `xgt_source`, `xgt_fenet_base`, `xgt_fenet_slot`
- `xgt_cmd`, `xgt_dtype`, `xgt_blkcnt`, `xgt_err_flag`, `xgt_err_code`
- `xgt_datasize`, `xgt_data_missing`, `xgt_data_len_chars`, `xgt_data_num_spaces`
- `xgt_data_is_hex`, `xgt_data_n_bytes`, `xgt_data_zero_ratio`
- `xgt_data_first_byte`, `xgt_data_last_byte`, `xgt_data_mean_byte`, `xgt_data_bucket`

### ARP - 3개
- `arp_src_host_id`, `arp_tgt_host_id`, `arp_op_num`

### DNS - 4개
- `dns_qc`, `dns_ac`, `dns_qc_norm`, `dns_ac_norm`

## 입력 데이터 형식

### 패킷 단위 JSONL
```json
{
  "protocol": "modbus",
  "smac": "00:11:22:33:44:55",
  "dmac": "AA:BB:CC:DD:EE:FF",
  "sip": "192.168.0.10",
  "dip": "192.168.0.11",
  "sp": 502,
  "dp": 510,
  "dir": "request",
  "len": 100,
  "modbus.addr": 0,
  "modbus.fc": 3,
  ...
}
```

## 출력 데이터 형식

### Feature CSV
- 패킷 단위 row (window_id, pattern, index, packet_idx, protocol, delta_t, ... features)

### Feature JSONL
```json
{
  "window_id": 1,
  "pattern": "normal_pattern",
  "index": [0, 1, 2, ...],
  "sequence_group": [
    {"protocol": 4.0, "delta_t": 0.001, "src_host_id": 1.0, ...},
    ...
  ]
}
```

## 필요 라이브러리

```
numpy
orjson (선택)
pyyaml
```

## 주의사항

1. **순서 중요**: 반드시 `--fit`으로 파라미터를 먼저 생성한 후 `--transform`을 사용하세요
2. **메모리 사용**: 대용량 데이터 처리 시 배치 처리를 고려하세요
3. **프로토콜 필터링**: 각 스크립트는 해당 프로토콜 패킷만 처리합니다
4. **새 호스트/변수**: fit 후 transform 시 새로운 호스트나 변수는 자동으로 새 ID가 부여됩니다
