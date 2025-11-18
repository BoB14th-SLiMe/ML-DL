# 📘 JSONL Key-Value Extractor  
**JSONL 파일에서 프로토콜 기반 key-value(unique) 목록을 추출하는 도구**  
**(특정 key 제외 기능 포함)**

---

## 📌 개요  
이 스크립트는 JSONL(Network Packet Log) 파일을 분석하여,  
각 **protocol**(예: `modbus`, `s7comm`, `tcp_session` 등)별로 등장하는:

- key 목록  
- 각 key에 대응하는 **고유(unique) 값 목록**  
- 특정 key는 제외 가능  

등을 자동으로 추출하여 **JSON 형식으로 저장**하는 도구입니다.

이 결과는 다음과 같은 용도로 유용합니다:

- ML/DL Feature Engineering (사용할 key 선택)
- 프로토콜별 데이터 스키마 분석
- 정상 트래픽 값 분포 확인
- ICS/OT 패킷 구조 분석
- 로그 데이터 표준화 작업

---

## ✨ 주요 기능

### ✔ 프로토콜별 key 자동 분류  
각 JSON 객체의 `"protocol"` 값을 기준으로 key를 그룹화합니다.

### ✔ key에 대한 값들을 중복 없이 추출  
각 key가 가진 모든 값들의 unique set을 저장합니다.

### ✔ 특정 key 제외 기능 (`--exclude-key`)  
분석에서 제외하고 싶은 key를 건너뛸 수 있습니다.

예:  
`@timestamp`, `sq`, `ak`, `fl`, `smac`, `dmac` 등  
ML/DL 학습에서 의미 없는 key를 제거할 때 매우 유용합니다.

### ✔ JSON 출력  
결과 파일은 아래와 같은 구조로 저장됩니다:

```json
{
    "modbus": {
        "dip": ["192.168.10.45", "192.168.10.80"],
        "modbus.fc": ["3", "6", "16"],
        "modbus.regs.addr": ["17", "20", "24"]
    },
    "s7comm": {
        "s7comm.addr": ["132", "564"],
        "s7comm.db": ["112", "32"]
    }
}
