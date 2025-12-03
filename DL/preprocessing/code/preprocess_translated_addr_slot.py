#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocess_translated_addr_slot.py

translated_addr 전용 high-resolution feature 전처리.

지원 프로토콜:
  - modbus
      - 주소: modbus.translated_addr (없으면 modbus.regs.translated_addr, 그래도 없으면 modbus.regs.addr)
      - 값  : modbus.regs.val

  - xgt_fen
      - 주소: xgt_fen.translated_addr (예: ["M1","M2","M3",...])
      - 값  : xgt_fen.word_value

두 모드 제공:
  --fit        : vocab + 주소별 min/max 생성 + npy 저장
  --transform  : 기존 vocab + min/max 사용해서 npy 생성

출력 파일 (out_dir 기준):

  protocol = "modbus" 인 경우:
    - modbus_addr_slot_vocab.json         : {"300024": 0, "300025": 1, ...}
    - modbus_addr_slot_norm_params.json   : {"300024": {"min": 0.0, "max": 100.0}, ...}
    - modbus_addr_slot.npy                : shape (N_modbus_packets, num_addrs)
        dtype fields:
          ("modbus_addr_300024", "f4"),
          ("modbus_addr_300025", "f4"), ...

  protocol = "xgt_fen" 인 경우:
    - xgt_addr_slot_vocab.json
    - xgt_addr_slot_norm_params.json
    - xgt_addr_slot.npy
        dtype fields:
          ("xgt_addr_M1", "f4"),
          ("xgt_addr_M2", "f4"), ...

실시간 단일 패킷:
  from preprocess_translated_addr_slot import preprocess_translated_addr_slot_one

  vocab = json.loads((out_dir / "modbus_addr_slot_vocab.json").read_text(encoding="utf-8"))
  norm  = json.loads((out_dir / "modbus_addr_slot_norm_params.json").read_text(encoding="utf-8"))

  feat = preprocess_translated_addr_slot_one(obj, "modbus", vocab, norm)
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ============================================================
# 공용 util
# ============================================================

def _flatten_str_like(x: Any) -> List[str]:
    """
    어떤 형태든 문자열 리스트로 평탄화:
      - "M1"
      - "M1,M2"
      - ["M1","M2"]
      - '["M1","M2"]'
      - 중첩 리스트 등
    """
    out: List[str] = []

    if x is None:
        return out

    if isinstance(x, (list, tuple)):
        for y in x:
            out.extend(_flatten_str_like(y))
        return out

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return out

        # JSON 리스트 형태 시도
        if s.startswith("[") and s.endswith("]"):
            try:
                loaded = json.loads(s)
                out.extend(_flatten_str_like(loaded))
                return out
            except Exception:
                pass

        # 콤마로 구분된 리스트
        if "," in s:
            parts = [p.strip() for p in s.split(",")]
            for p in parts:
                if p:
                    out.append(p)
            return out

        # 그냥 단일 토큰
        out.append(s)
        return out

    # 그 외 타입은 str()로 변환
    out.append(str(x))
    return out


def _flatten_float_like(x: Any) -> List[float]:
    """
    숫자/문자열/리스트/중첩리스트/JSON리스트 등 → float 리스트로 평탄화
    """
    out: List[float] = []

    if x is None:
        return out

    if isinstance(x, (list, tuple)):
        for y in x:
            out.extend(_flatten_float_like(y))
        return out

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return out

        # JSON 리스트 문자열
        if s.startswith("[") and s.endswith("]"):
            try:
                loaded = json.loads(s)
                out.extend(_flatten_float_like(loaded))
                return out
            except Exception:
                pass

        # 공백/콤마로 나눠진 경우
        if "," in s:
            parts = [p.strip() for p in s.split(",")]
            for p in parts:
                try:
                    out.append(float(p))
                except Exception:
                    continue
            return out

        # 단일 숫자 문자열
        try:
            out.append(float(s))
        except Exception:
            pass
        return out

    # 숫자 등
    try:
        out.append(float(x))
    except Exception:
        pass
    return out


def minmax_norm(v: float, vmin: float, vmax: float) -> float:
    """
    단순 min-max 정규화 (vmin == vmax 이면 0.0)
    """
    try:
        v = float(v)
    except Exception:
        return 0.0

    if vmin is None or vmax is None:
        return 0.0
    if vmax <= vmin:
        return 0.0
    return float((v - vmin) / (vmax - vmin + 1e-9))


# ============================================================
# 프로토콜별 설정
# ============================================================

def get_protocol_config(proto: str) -> Dict[str, Any]:
    proto = proto.strip().lower()
    if proto == "modbus":
        return {
            "protocol": "modbus",
            "addr_fields": [
                "modbus.translated_addr",
                "modbus.regs.translated_addr",
                "modbus.regs.addr",
            ],
            "val_field": "modbus.regs.val",
            "vocab_file": "modbus_addr_slot_vocab.json",
            "norm_file": "modbus_addr_slot_norm_params.json",
            "npy_file": "modbus_addr_slot.npy",
            "feature_prefix": "modbus_addr_",
            "sort_numeric": True,  # 주소 숫자 기준 정렬
        }
    elif proto == "xgt_fen":
        return {
            "protocol": "xgt_fen",
            "addr_fields": [
                "xgt_fen.translated_addr",
            ],
            "val_field": "xgt_fen.word_value",
            "vocab_file": "xgt_addr_slot_vocab.json",
            "norm_file": "xgt_addr_slot_norm_params.json",
            "npy_file": "xgt_addr_slot.npy",
            "feature_prefix": "xgt_addr_",
            "sort_numeric": False,  # "M1","M2"... 는 문자열 기준 정렬
        }
    else:
        raise ValueError(f"지원하지 않는 protocol: {proto}")


# ============================================================
# addr / value 추출
# ============================================================

def extract_addr_val_pairs(obj: Dict[str, Any],
                           cfg: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    한 패킷에서 translated_addr + value 쌍 추출.

    반환: [(addr_str, value_float), ...]
    """
    if obj.get("protocol") != cfg["protocol"]:
        return []

    raw_addr = None
    for key in cfg["addr_fields"]:
        raw_addr = obj.get(key)
        if raw_addr:
            break

    if raw_addr is None:
        return []

    raw_val = obj.get(cfg["val_field"])
    if raw_val is None:
        return []

    # 주소 리스트
    if cfg["protocol"] == "modbus":
        # modbus는 숫자 주소라 int → str 로 다루기
        addr_nums = _flatten_float_like(raw_addr)
        addr_list = [str(int(a)) for a in addr_nums]
    else:
        # xgt_fen: "M1","M2" 그대로 사용
        addr_list = _flatten_str_like(raw_addr)

    # 값 리스트
    val_list = _flatten_float_like(raw_val)

    if not addr_list or not val_list:
        return []

    n = min(len(addr_list), len(val_list))
    pairs: List[Tuple[str, float]] = []
    for i in range(n):
        a = addr_list[i]
        try:
            v = float(val_list[i])
        except Exception:
            continue
        pairs.append((a, v))

    return pairs


# ============================================================
# FIT
# ============================================================

def fit_preprocess(input_path: Path, out_dir: Path, proto: str):
    cfg = get_protocol_config(proto)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 주소별 min/max 모으기
    addr_stats: Dict[str, Dict[str, float]] = {}
    # 패킷별 addr/value 리스트 (나중에 npy 만들 때 사용)
    packet_pairs: List[List[Tuple[str, float]]] = []

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            pairs = extract_addr_val_pairs(obj, cfg)
            if not pairs:
                continue

            packet_pairs.append(pairs)

            for a, v in pairs:
                if a not in addr_stats:
                    addr_stats[a] = {"min": v, "max": v}
                else:
                    if v < addr_stats[a]["min"]:
                        addr_stats[a]["min"] = v
                    if v > addr_stats[a]["max"]:
                        addr_stats[a]["max"] = v

    if not packet_pairs:
        print(f"⚠ {proto} translated_addr + value 쌍이 하나도 없습니다. 빈 npy를 생성합니다.")
        vocab = {}
        norm_params = {}
        addr_list_sorted: List[str] = []
    else:
        # 주소 정렬 (modbus는 숫자, xgt_fen은 문자열)
        addrs = list(addr_stats.keys())
        if cfg["sort_numeric"]:
            def sort_key(a: str):
                try:
                    return int(a)
                except Exception:
                    return 10**12
            addr_list_sorted = sorted(addrs, key=sort_key)
        else:
            addr_list_sorted = sorted(addrs)

        # vocab & norm_params 생성
        vocab = {addr: idx for idx, addr in enumerate(addr_list_sorted)}
        norm_params = {addr: {"min": s["min"], "max": s["max"]}
                       for addr, s in addr_stats.items()}

    # JSON 저장
    vocab_path = out_dir / cfg["vocab_file"]
    norm_path = out_dir / cfg["norm_file"]

    vocab_path.write_text(
        json.dumps(vocab, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    norm_path.write_text(
        json.dumps(norm_params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"✅ FIT 완료 ({proto})")
    print(f"- vocab 저장: {vocab_path}")
    print(f"- norm_params 저장: {norm_path}")
    print(f"- 총 addr 개수: {len(vocab)}")
    print(f"- 총 패킷(행) 수: {len(packet_pairs)}")

    # npy 생성
    if not vocab:
        # 빈 npy
        data_empty = np.zeros(0, dtype=[])
        np.save(out_dir / cfg["npy_file"], data_empty)
        print(f"- {cfg['npy_file']} (empty) 저장 완료")
        return

    dtype = np.dtype([
        (cfg["feature_prefix"] + addr, "f4")
        for addr in sorted(vocab.keys(), key=lambda a: vocab[a])  # index 순서대로
    ])

    data = np.zeros(len(packet_pairs), dtype=dtype)

    # addr_list_sorted는 vocab index 기준과 동일 순서로 맞춰줌
    addr_list_sorted = [addr for addr, _ in sorted(vocab.items(), key=lambda x: x[1])]

    for i, pairs in enumerate(packet_pairs):
        # addr → value 맵 (패킷 하나)
        val_map: Dict[str, float] = {}
        for a, v in pairs:
            val_map[a] = v  # 같은 주소 여러 번 있으면 마지막 것 기준

        for addr in addr_list_sorted:
            field = cfg["feature_prefix"] + addr
            v = val_map.get(addr, None)
            if v is None:
                data[field][i] = 0.0
            else:
                p = norm_params.get(addr) or {}
                vmin = p.get("min")
                vmax = p.get("max")
                data[field][i] = minmax_norm(v, vmin, vmax)

    npy_path = out_dir / cfg["npy_file"]
    np.save(npy_path, data)
    print(f"- {cfg['npy_file']} 저장: {npy_path} shape={data.shape}")


# ============================================================
# TRANSFORM
# ============================================================

def transform_preprocess(input_path: Path, out_dir: Path, proto: str):
    cfg = get_protocol_config(proto)

    vocab_path = out_dir / cfg["vocab_file"]
    norm_path = out_dir / cfg["norm_file"]

    if not vocab_path.exists():
        raise FileNotFoundError(f"❌ {vocab_path} 가 없습니다. 먼저 --fit 을 실행하세요.")
    if not norm_path.exists():
        raise FileNotFoundError(f"❌ {norm_path} 가 없습니다. 먼저 --fit 을 실행하세요.")

    vocab: Dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))
    norm_params: Dict[str, Dict[str, float]] = json.loads(norm_path.read_text(encoding="utf-8"))

    # index 기준으로 정렬된 addr 리스트
    addr_list_sorted = [addr for addr, _ in sorted(vocab.items(), key=lambda x: x[1])]

    if not addr_list_sorted:
        data_empty = np.zeros(0, dtype=[])
        np.save(out_dir / cfg["npy_file"], data_empty)
        print(f"⚠ vocab 이 비어 있습니다. {cfg['npy_file']} empty 저장.")
        return

    # 입력에서 protocol 해당하는 패킷만 다시 모으기
    packet_pairs: List[List[Tuple[str, float]]] = []

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            pairs = extract_addr_val_pairs(obj, cfg)
            if not pairs:
                continue

            # vocab 에 없는 주소는 무시
            filtered_pairs = [(a, v) for a, v in pairs if a in vocab]
            if not filtered_pairs:
                # 해당 패킷에 vocab에 있는 주소가 하나도 없으면 그래도 row는 만들고 모두 0.0
                packet_pairs.append([])
            else:
                packet_pairs.append(filtered_pairs)

    dtype = np.dtype([
        (cfg["feature_prefix"] + addr, "f4")
        for addr in addr_list_sorted
    ])
    data = np.zeros(len(packet_pairs), dtype=dtype)

    for i, pairs in enumerate(packet_pairs):
        val_map: Dict[str, float] = {}
        for a, v in pairs:
            val_map[a] = v

        for addr in addr_list_sorted:
            field = cfg["feature_prefix"] + addr
            v = val_map.get(addr, None)
            if v is None:
                data[field][i] = 0.0
            else:
                p = norm_params.get(addr) or {}
                vmin = p.get("min")
                vmax = p.get("max")
                data[field][i] = minmax_norm(v, vmin, vmax)

    npy_path = out_dir / cfg["npy_file"]
    np.save(npy_path, data)
    print("✅ TRANSFORM 완료")
    print(f"- {cfg['npy_file']} 저장: {npy_path} shape={data.shape}")


# ============================================================
# 단일 패킷용 함수
# ============================================================

def preprocess_translated_addr_slot_one(
    obj: Dict[str, Any],
    proto: str,
    vocab: Dict[str, int],
    norm_params: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    단일 패킷 → translated_addr slot feature dict.

    반환 예시 (proto="modbus" 인 경우):

      {
        "modbus_addr_300024": 0.0~1.0,
        "modbus_addr_300025": 0.0~1.0,
        ...
      }

    proto="xgt_fen" 인 경우:

      {
        "xgt_addr_M1": 0.0~1.0,
        "xgt_addr_M2": 0.0~1.0,
        ...
      }
    """
    cfg = get_protocol_config(proto)

    # index 순서대로 addr 나열
    addr_list_sorted = [addr for addr, _ in sorted(vocab.items(), key=lambda x: x[1])]

    feat: Dict[str, float] = {
        cfg["feature_prefix"] + addr: 0.0 for addr in addr_list_sorted
    }

    pairs = extract_addr_val_pairs(obj, cfg)
    if not pairs:
        return feat

    val_map: Dict[str, float] = {}
    for a, v in pairs:
        if a in vocab:
            val_map[a] = v

    for addr in addr_list_sorted:
        field = cfg["feature_prefix"] + addr
        v = val_map.get(addr, None)
        if v is None:
            feat[field] = 0.0
        else:
            p = norm_params.get(addr) or {}
            vmin = p.get("min")
            vmax = p.get("max")
            feat[field] = minmax_norm(v, vmin, vmax)

    return feat


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="translated_addr slot feature 전처리 (modbus / xgt_fen)"
    )
    parser.add_argument("-i", "--input", required=True, help="입력 JSONL 파일 경로")
    parser.add_argument("-o", "--output", required=True, help="출력 디렉토리 경로")
    parser.add_argument(
        "-P",
        "--protocol",
        required=True,
        choices=["modbus", "xgt_fen"],
        help="처리할 프로토콜 (modbus 또는 xgt_fen)",
    )
    parser.add_argument("--fit", action="store_true", help="vocab + norm_params 생성 + npy 생성")
    parser.add_argument("--transform", action="store_true", help="기존 vocab + norm_params로 npy 생성")

    args = parser.parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output)

    if args.fit and args.transform:
        raise ValueError("❌ --fit 과 --transform 는 동시에 사용할 수 없습니다.")
    if not args.fit and not args.transform:
        raise ValueError("❌ 반드시 --fit 또는 --transform 중 하나를 선택하세요.")

    if args.fit:
        fit_preprocess(input_path, out_dir, args.protocol)
    else:
        transform_preprocess(input_path, out_dir, args.protocol)


if __name__ == "__main__":
    main()
