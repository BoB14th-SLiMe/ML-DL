#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
0.attack_result.py (multi-scenario)

여러 종류의 공격 시나리오(JSONL)를 처리해서,
슬라이딩 윈도우 단위로 is_anomaly 플래그를 생성하는 스크립트.

공통 동작:
  - 입력: "한 줄 = 1 패킷" 형식의 JSONL (예: attack_ver5_1.jsonl)
  - window_size, step_size 로 슬라이딩 윈도우 생성
  - 윈도우 안에 "공격 패킷"이 하나라도 있으면 is_anomaly = 1, 없으면 0
  - 출력 CSV 컬럼:
      window_index, start_packet_idx, end_packet_idx, valid_len, is_anomaly

공격 판정 모드 (attack_mode):

  1) modbus_fc6
     - protocol == "modbus" 이고, modbus.fc == 6 인 패킷이 윈도우에 존재하면 공격

  2) xgt_last_zero (ver5 계열)
     - protocol == "xgt_fen"
     - xgt_fen.vars == "%DB001046"
     - xgt_fen.data 의 마지막 워드(마지막 4 hex)가 "0000" 인 패킷 → 공격
     - 단, data 전체가 0000... 이면 정상 취급

  3) xgt_mid_zero (ver5_1 계열)
     - protocol == "xgt_fen"
     - xgt_fen.vars == "%DB001046"
     - xgt_fen.data 의 3번째 워드(8~11 문자)가 "0000" 인 패킷 → 공격
     - 단, data 전체가 0000... 이면 정상 취급

  4) xgt_head_zero (ver5_2 계열)
     - protocol == "xgt_fen"
     - xgt_fen.vars == "%DB001046"
     - xgt_fen.data 의 첫 번째 워드(0~3 문자)가 "0000" 인 패킷 → 공격
     - 단, data 전체가 0000... 이면 정상 취급

  5) xgt_write_or_fc6
     - XGT 쓰기 또는 Modbus FC6 를 공격으로 간주:
       * protocol == "xgt_fen" 이고 xgt_fen.cmd in {0x58, 0x59} → 무조건 공격
       * 또는 protocol == "modbus" 이고 modbus.fc == 6 → 공격

추가 공통 규칙:
  - xgt_fen.cmd 가 0x58 또는 0x59 인 패킷은 attack_mode와 상관 없이 항상 공격으로 간주.
    (윈도우 안에 하나라도 있으면 그 윈도우는 is_anomaly = 1)

--mode auto 인 경우, 입력 파일명으로부터 모드를 추론:
  - 파일명에 "ver5_2"   → xgt_head_zero
  - 파일명에 "ver5_1"   → xgt_mid_zero
  - 파일명에 "ver5"     → xgt_last_zero
  - 파일명에 "ver2" or "ver11" or "attack" → xgt_write_or_fc6
  - 그 외                → modbus_fc6
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Any, Dict, List


# ------------------------- 기본 유틸 -------------------------

def is_all_zero_data(hex_str: str) -> bool:
    """
    xgt_fen.data가 전부 0인 경우 (예: '000000...')인지 확인.
    normalize_hex_string()을 거친 문자열을 넣는다고 가정.
    """
    if not hex_str:
        return False
    # 문자열이 있고, 모든 문자가 '0'이면 all-zero
    return all(ch == "0" for ch in hex_str)


def safe_int(val: Any, default: int = 0) -> int:
    """
    modbus.fc, xgt_fen.cmd 등 숫자 필드 안전 변환.

    - 리스트면 첫 번째 None 아닌 값 사용
    - "6", "06", 6 → 10진으로 처리
    - "0x58", "0X59" → 16진으로 처리
    - 실패하면 default
    """
    try:
        if isinstance(val, list):
            for v in val:
                if v is None:
                    continue
                val = v
                break

        if val is None:
            raise ValueError("None value")

        if isinstance(val, str):
            s = val.strip()
            if not s:
                raise ValueError("empty string")

            if s.startswith(("0x", "0X")):
                return int(s, 16)

            return int(s)

        return int(val)
    except Exception:
        return default


def normalize_hex_string(s: Any) -> str:
    """
    xgt_fen.data 같은 hex 문자열을 전처리:
      - str이 아니면 "" 반환
      - 공백 제거
      - 소문자로 변환
    """
    if not isinstance(s, str):
        return ""
    return "".join(s.split()).lower()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input", "-i", required=True,
        help="패킷 단위 JSONL 경로 (각 line = 1 packet)",
    )
    p.add_argument(
        "--window-size", "-w", type=int, default=80,
        help="윈도우 길이(묶을 패킷 개수, 기본=80)",
    )
    p.add_argument(
        "--step-size", "-s", type=int, default=None,
        help="슬라이딩 stride (기본: window-size와 동일 → non-overlap)",
    )
    p.add_argument(
        "--output", "-o", required=True,
        help="출력 CSV 경로 (window_index,start_packet_idx,end_packet_idx,valid_len,is_anomaly)",
    )
    p.add_argument(
        "--mode", "-m",
        choices=[
            "auto",
            "modbus_fc6",
            "xgt_last_zero", "xgt_mid_zero", "xgt_head_zero",
            "xgt_write_or_fc6",
            "xgt_d528_zero", "xgt_d525_zero",  # 옛 이름 호환용
        ],
        default="auto",
        help=(
            "공격 판정 모드. "
            "auto: 파일명으로부터 자동 결정, "
            "modbus_fc6: 모드버스 fc=6 기준, "
            "xgt_last_zero/xgt_d528_zero: xgt_fen D0528(마지막 워드) 0000, "
            "xgt_mid_zero/xgt_d525_zero: xgt_fen D0525(3번째 워드) 0000, "
            "xgt_head_zero: xgt_fen D0523(첫 워드) 0000, "
            "xgt_write_or_fc6: XGT 쓰기(cmd 0x58/0x59) 또는 Modbus FC6"
        ),
    )

    return p.parse_args()


# ------------------------- 패킷 단위 공격 판정 -------------------------

def packet_is_attack_xgt_word_zero(pkt: Dict[str, Any], word_idx: int) -> bool:
    """
    XGT-FEnet 블록에서 특정 워드(word_idx)가 0000인 경우를 공격으로 간주.
    단, data 전체가 0000... 인 경우는 정상으로 처리.

    - protocol == 'xgt_fen'
    - xgt_fen.vars == '%DB001046'
    - hex_str[word_idx*4 : word_idx*4+4] == '0000'
    - BUT, hex_str 전체가 '0000...' 이면 공격 아님
    """
    if pkt.get("protocol") != "xgt_fen":
        return False

    vars_field = pkt.get("xgt_fen.vars")
    if vars_field != "%DB001046":
        return False

    hex_str = normalize_hex_string(pkt.get("xgt_fen.data"))
    if not hex_str:
        return False

    # 전체가 0이면 정상 취급
    if is_all_zero_data(hex_str):
        return False

    start = word_idx * 4
    end = start + 4
    if len(hex_str) < end:
        return False

    word = hex_str[start:end]
    return word == "0000"


def packet_is_attack_modbus_fc6(pkt: Dict[str, Any]) -> bool:
    """
    기준 1: protocol == 'modbus' 이고 modbus.fc == 6 인 패킷을 공격으로 간주.
    """
    if pkt.get("protocol") != "modbus":
        return False

    fc_raw = pkt.get("modbus.fc")
    fc = safe_int(fc_raw, default=-1)
    return fc == 6


def packet_is_attack_xgt_write(pkt: Dict[str, Any]) -> bool:
    """
    XGT-FEnet 쓰기 패킷을 공격으로 간주하는 규칙.
    - protocol == 'xgt_fen'
    - xgt_fen.cmd in {0x58, 0x59} (쓰기 계열)

    ※ vars는 제한하지 않음: 58/59면 무조건 공격으로 간주.
    """
    if pkt.get("protocol") != "xgt_fen":
        return False

    cmd_raw = pkt.get("xgt_fen.cmd")
    cmd = safe_int(cmd_raw, default=-1)
    if cmd in (0x58, 0x59):
        return True
    return False


# ver5: 마지막 워드 (W5)
def packet_is_attack_xgt_last_zero(pkt: Dict[str, Any]) -> bool:
    # W5: word_idx = 5 (6워드 기준)
    return packet_is_attack_xgt_word_zero(pkt, word_idx=5)


# ver5_1: 가운데 워드 (8~11) → W2
def packet_is_attack_xgt_mid_zero(pkt: Dict[str, Any]) -> bool:
    # W2: word_idx = 2
    return packet_is_attack_xgt_word_zero(pkt, word_idx=3)


# ver5_2: 처음 워드 (0~3) → W0
def packet_is_attack_xgt_head_zero(pkt: Dict[str, Any]) -> bool:
    # W0: word_idx = 0
    return packet_is_attack_xgt_word_zero(pkt, word_idx=0)


def window_has_attack(packets: List[Dict[str, Any]], attack_mode: str) -> bool:
    """
    윈도우 내에 공격 패킷이 하나라도 있으면 True.
    attack_mode 에 따라 다른 기준 사용.
    """
    for pkt in packets:
        # 0) XGT 쓰기(cmd 0x58, 0x59)는 모드와 상관없이 항상 공격
        if packet_is_attack_xgt_write(pkt):
            return True

        if attack_mode in ("modbus_fc6", "xgt_write_or_fc6"):
            if packet_is_attack_modbus_fc6(pkt):
                return True

        elif attack_mode == "xgt_last_zero":
            if packet_is_attack_xgt_last_zero(pkt):
                return True

        elif attack_mode == "xgt_mid_zero":
            if packet_is_attack_xgt_mid_zero(pkt):
                return True

        elif attack_mode == "xgt_head_zero":
            if packet_is_attack_xgt_head_zero(pkt):
                return True

    return False


def infer_attack_mode_from_filename(filename: str) -> str:
    """
    --mode auto 일 때, 입력 파일명으로부터 공격 모드를 추론.

    우선순위:
      1) 'ver5_2'  → xgt_head_zero
      2) 'ver5_1'  → xgt_mid_zero
      3) 'ver5'    → xgt_last_zero
      4) 'ver2' or 'ver11' or 'attack' → xgt_write_or_fc6
      5) 기타      → modbus_fc6
    """
    name = filename.lower()

    if "ver5_2" in name:
        return "xgt_head_zero"
    if "ver5_1" in name:
        return "xgt_mid_zero"
    if "ver5" in name:
        return "xgt_last_zero"

    if "ver2" in name or "ver11" in name or "attack" in name:
        return "xgt_write_or_fc6"

    return "modbus_fc6"


# ------------------------- 메인 로직 -------------------------

def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    window_size = args.window_size
    step_size = args.step_size if args.step_size is not None else window_size

    # 공격 모드 결정
    if args.mode == "auto":
        attack_mode = infer_attack_mode_from_filename(input_path.name)
        print(f"[INFO] --mode=auto → 파일명 '{input_path.name}' 기준으로 공격 모드 추론: {attack_mode}")
    else:
        # 옛 이름(xgt_d528_zero/xgt_d525_zero)도 새 이름으로 매핑
        if args.mode in ("xgt_d528_zero", "xgt_last_zero"):
            attack_mode = "xgt_last_zero"
        elif args.mode in ("xgt_d525_zero", "xgt_mid_zero"):
            attack_mode = "xgt_mid_zero"
        elif args.mode == "xgt_head_zero":
            attack_mode = "xgt_head_zero"
        elif args.mode == "xgt_write_or_fc6":
            attack_mode = "xgt_write_or_fc6"
        elif args.mode == "modbus_fc6":
            attack_mode = "modbus_fc6"
        else:
            attack_mode = args.mode

        print(f"[INFO] --mode={args.mode} (사용자 지정) → 내부 attack_mode={attack_mode}")

    print(f"[INFO] 입력 JSONL : {input_path}")
    print(f"[INFO] window_size = {window_size}, step_size = {step_size}")
    print(f"[INFO] 출력 CSV    : {output_path}")
    print(f"[INFO] 공격 판정 모드 = {attack_mode}")

    # 1) JSONL 전체 읽어서 패킷 리스트 생성
    packets: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] JSON 파싱 실패 (line {line_no}), 스킵: {e}")
                continue
            packets.append(obj)

    total_packets = len(packets)
    print(f"[INFO] 총 패킷 수 = {total_packets}")

    if total_packets == 0:
        print("[WARN] 유효한 패킷이 없습니다. 종료합니다.")
        return

    # 2) 슬라이딩 윈도우 만들면서 공격 포함 여부 체크
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow([
            "window_index",
            "start_packet_idx",
            "end_packet_idx",
            "valid_len",
            "is_anomaly",
        ])

        window_index = 0
        start_idx = 0

        while start_idx < total_packets:
            end_idx = start_idx + window_size
            window_packets = packets[start_idx:end_idx]

            if not window_packets:
                break

            valid_len = len(window_packets)
            end_packet_idx = start_idx + valid_len - 1

            # 윈도우 안에 공격 패킷이 하나라도 있으면 공격(1)
            is_attack = 1 if window_has_attack(window_packets, attack_mode) else 0

            writer.writerow([
                window_index,
                start_idx,
                end_packet_idx,
                valid_len,
                is_attack,
            ])

            window_index += 1
            start_idx += step_size

    print(f"[INFO] 완료: 총 {window_index}개 윈도우 결과를 {output_path} 에 저장했습니다.")


if __name__ == "__main__":
    main()


"""
사용 예시:

1) attack_ver5.jsonl (또는 attack_ver5_1.jsonl)
   - 파일명에 'ver5' 또는 'ver5_1' 이 포함 → xgt_d528_zero 모드 자동 선택

python mark_attack_windows_multi.py \
  --input "../data/attack_ver5_1.jsonl" \
  --window-size 40 \
  --step-size 10 \
  --output "../result/attack_ver5_1_windows.csv"

2) attack_ver5_2.jsonl
   - 파일명에 'ver5_2' 포함 → xgt_d525_zero 모드 자동 선택

python mark_attack_windows_multi.py \
  --input "../data/attack_ver5_2.jsonl" \
  --window-size 40 \
  --step-size 10 \
  --output "../result/attack_ver5_2_windows.csv"

3) 기존 Modbus fc=6 공격 데이터 (파일명에 ver5_x 안 들어가는 경우)

python mark_attack_windows_multi.py \
  --input "../data/attack_fc6.jsonl" \
  --window-size 80 \
  --output "../result/attack_fc6_windows.csv"

또는 모드를 강제로 지정:

python mark_attack_windows_multi.py \
  --input "../data/ML_DL_학습_sample.jsonl" \
  --window-size 80 \
  --output "../result/sample_windows.csv" \
  --mode xgt_d528_zero

"""
