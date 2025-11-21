#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocess_modbus_feat.py
modbus 전용 feature 전처리

두 모드 제공:
  --fit        : min-max 정규화 파라미터 생성 + modbus.npy 저장
  --transform  : 기존 정규화 파라미터 사용하여 modbus.npy 생성

입력 JSONL에서 사용하는 필드:
  - protocol == "modbus" 또는 "modbus_tcp" 등 (필요시 수정)
  - modbus.addr       : 정수화 후 min-max 정규화
  - modbus.fc         : 정수화 후 min-max 정규화
  - modbus.qty        : 정수화 후 min-max 정규화
  - modbus.bc         : 정수화 후 min-max 정규화
  - modbus.regs.addr  : 레지스터 주소 리스트
  - modbus.regs.val   : 레지스터 값 리스트

출력 feature (modbus.npy, structured numpy):
  - modbus_addr_norm   (float32)  ← modbus.addr min-max 정규화
  - modbus_fc_norm     (float32)  ← modbus.fc min-max 정규화
  - modbus_qty_norm    (float32)  ← modbus.qty min-max 정규화
  - modbus_bc_norm     (float32)  ← modbus.bc min-max 정규화

  - regs_addr_count    (float32)  ← len(modbus.regs.addr)
  - regs_addr_min      (float32)  ← min(modbus.regs.addr)
  - regs_addr_max      (float32)  ← max(modbus.regs.addr)
  - regs_addr_range    (float32)  ← max - min

  - regs_val_min       (float32)  ← min(modbus.regs.val)
  - regs_val_max       (float32)  ← max(modbus.regs.val)
  - regs_val_mean      (float32)  ← mean(modbus.regs.val)
  - regs_val_std       (float32)  ← std(modbus.regs.val) (ddof=0)

보조 파일:
  - modbus_norm_params.json
      {
        "modbus.addr": {"min": ..., "max": ...},
        "modbus.fc":   {"min": ..., "max": ...},
        "modbus.qty":  {"min": ..., "max": ...},
        "modbus.bc":   {"min": ..., "max": ...}
      }
"""

import json
import argparse
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------
# 공용 유틸
# ---------------------------------------------
def parse_int_field(val: Any) -> Optional[int]:
    """
    JSONL에서 들어오는 값이
      - 10
      - "10"
      - [10]
      - ["10"]
    등일 수 있으므로 통일해서 int로 변환.
    변환 실패 시 None 반환.
    """
    if isinstance(val, list) and val:
        val = val[0]
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def parse_float_field(val: Any) -> Optional[float]:
    """
    int/float/str/[...] 형태를 float 하나로 파싱.
    """
    if isinstance(val, list) and val:
        val = val[0]
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def parse_int_list_field(val: Any) -> List[int]:
    """
    modbus.regs.addr 같이 리스트로 들어오는 필드를 int 리스트로 변환.
    - None 또는 변환 불가능한 값은 스킵.
    - 단일 값인 경우 [값] 형태로 변환.
    """
    if val is None:
        return []
    if not isinstance(val, list):
        val = [val]
    out: List[int] = []
    for v in val:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out


def parse_float_list_field(val: Any) -> List[float]:
    """
    modbus.regs.val 같이 리스트로 들어오는 필드를 float 리스트로 변환.
    """
    if val is None:
        return []
    if not isinstance(val, list):
        val = [val]
    out: List[float] = []
    for v in val:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def minmax_norm(v: Optional[int], vmin: Optional[int], vmax: Optional[int]) -> float:
    """
    단순 min-max 정규화:
      (v - vmin) / (vmax - vmin)

    - v가 None 이거나,
    - vmin/vmax가 없거나,
    - vmin == vmax 인 경우 → 0.0 반환
    """
    if v is None or vmin is None or vmax is None:
        return 0.0
    if vmax == vmin:
        return 0.0
    return float(v - vmin) / float(vmax - vmin)


# ---------------------------------------------
# 한 레코드(modbus)에서 raw 값 추출
# ---------------------------------------------
def extract_modbus_raw(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    protocol == "modbus" 인 레코드에서 필요한 필드만 파싱.

    반환:
      {
        "addr": int or None,
        "fc":   int or None,
        "qty":  int or None,
        "bc":   int or None,
        "regs_addr": List[int],
        "regs_val":  List[float],
      }
    모든 주요 필드가 None/빈 리스트면 None 반환.
    """
    addr = parse_int_field(obj.get("modbus.addr"))
    fc   = parse_int_field(obj.get("modbus.fc"))
    qty  = parse_int_field(obj.get("modbus.qty"))
    bc   = parse_int_field(obj.get("modbus.bc"))

    regs_addr = parse_int_list_field(obj.get("modbus.regs.addr"))
    regs_val  = parse_float_list_field(obj.get("modbus.regs.val"))

    if all(v is None for v in (addr, fc, qty, bc)) and (not regs_addr) and (not regs_val):
        return None

    return {
        "addr": addr,
        "fc": fc,
        "qty": qty,
        "bc": bc,
        "regs_addr": regs_addr,
        "regs_val": regs_val,
    }


# ---------------------------------------------
# regs.* 통계 계산
# ---------------------------------------------
def compute_regs_addr_stats(addrs: List[int]) -> Dict[str, float]:
    if not addrs:
        return {
            "count": 0.0,
            "min": 0.0,
            "max": 0.0,
            "range": 0.0,
        }
    count = float(len(addrs))
    amin = float(min(addrs))
    amax = float(max(addrs))
    arange = float(amax - amin)
    return {
        "count": count,
        "min": amin,
        "max": amax,
        "range": arange,
    }


def compute_regs_val_stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
        }
    vmin = float(min(vals))
    vmax = float(max(vals))
    mean = float(sum(vals) / len(vals))
    # 분산 계산 (ddof=0)
    var = float(sum((v - mean) ** 2 for v in vals) / len(vals))
    std = float(math.sqrt(var))
    return {
        "min": vmin,
        "max": vmax,
        "mean": mean,
        "std": std,
    }


# ---------------------------------------------
# FIT
# ---------------------------------------------
def fit_preprocess(input_path: Path, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    rows_raw: List[Dict[str, Any]] = []

    # min/max 추적용
    addr_min = addr_max = None
    fc_min = fc_max = None
    qty_min = qty_max = None
    bc_min = bc_max = None

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # 필요시 protocol 값 조건 수정 (예: "modbus_tcp")
            if obj.get("protocol") != "modbus":
                continue

            raw = extract_modbus_raw(obj)
            if raw is None:
                continue

            rows_raw.append(raw)

            # min/max 업데이트 (None 은 스킵)
            if raw["addr"] is not None:
                if addr_min is None or raw["addr"] < addr_min:
                    addr_min = raw["addr"]
                if addr_max is None or raw["addr"] > addr_max:
                    addr_max = raw["addr"]

            if raw["fc"] is not None:
                if fc_min is None or raw["fc"] < fc_min:
                    fc_min = raw["fc"]
                if fc_max is None or raw["fc"] > fc_max:
                    fc_max = raw["fc"]

            if raw["qty"] is not None:
                if qty_min is None or raw["qty"] < qty_min:
                    qty_min = raw["qty"]
                if qty_max is None or raw["qty"] > qty_max:
                    qty_max = raw["qty"]

            if raw["bc"] is not None:
                if bc_min is None or raw["bc"] < bc_min:
                    bc_min = raw["bc"]
                if bc_max is None or raw["bc"] > bc_max:
                    bc_max = raw["bc"]

    # 정규화 파라미터 저장
    norm_params = {
        "modbus.addr": {"min": addr_min, "max": addr_max},
        "modbus.fc": {"min": fc_min, "max": fc_max},
        "modbus.qty": {"min": qty_min, "max": qty_max},
        "modbus.bc": {"min": bc_min, "max": bc_max},
    }

    norm_path = out_dir / "modbus_norm_params.json"
    norm_path.write_text(
        json.dumps(norm_params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("✅ FIT 완료")
    print(f"- modbus_norm_params.json 저장: {norm_path}")

    # numpy 구조화 배열 생성
    dtype = np.dtype([
        ("modbus_addr_norm", "f4"),
        ("modbus_fc_norm", "f4"),
        ("modbus_qty_norm", "f4"),
        ("modbus_bc_norm", "f4"),

        ("regs_addr_count", "f4"),
        ("regs_addr_min", "f4"),
        ("regs_addr_max", "f4"),
        ("regs_addr_range", "f4"),

        ("regs_val_min", "f4"),
        ("regs_val_max", "f4"),
        ("regs_val_mean", "f4"),
        ("regs_val_std", "f4"),
    ])

    data = np.zeros(len(rows_raw), dtype=dtype)

    for idx, raw in enumerate(rows_raw):
        addr = raw["addr"]
        fc   = raw["fc"]
        qty  = raw["qty"]
        bc   = raw["bc"]
        regs_addr = raw["regs_addr"]
        regs_val  = raw["regs_val"]

        # 1) min-max 정규화 필드
        data["modbus_addr_norm"][idx] = minmax_norm(addr, addr_min, addr_max)
        data["modbus_fc_norm"][idx]   = minmax_norm(fc, fc_min, fc_max)
        data["modbus_qty_norm"][idx]  = minmax_norm(qty, qty_min, qty_max)
        data["modbus_bc_norm"][idx]   = minmax_norm(bc, bc_min, bc_max)

        # 2) regs.addr 통계
        addr_stats = compute_regs_addr_stats(regs_addr)
        data["regs_addr_count"][idx] = addr_stats["count"]
        data["regs_addr_min"][idx]   = addr_stats["min"]
        data["regs_addr_max"][idx]   = addr_stats["max"]
        data["regs_addr_range"][idx] = addr_stats["range"]

        # 3) regs.val 통계
        val_stats = compute_regs_val_stats(regs_val)
        data["regs_val_min"][idx]   = val_stats["min"]
        data["regs_val_max"][idx]   = val_stats["max"]
        data["regs_val_mean"][idx]  = val_stats["mean"]
        data["regs_val_std"][idx]   = val_stats["std"]

    npy_path = out_dir / "modbus.npy"
    np.save(npy_path, data)

    print(f"- modbus.npy 저장: {npy_path}")
    print(f"- shape: {data.shape}")

    print("\n===== 앞 5개 modbus 전처리 샘플 =====")
    for i in range(min(5, len(data))):
        sample = {name: data[name][i] for name in data.dtype.names}
        print(sample)


# ---------------------------------------------
# TRANSFORM
# ---------------------------------------------
def transform_preprocess(input_path: Path, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    norm_path = out_dir / "modbus_norm_params.json"
    if not norm_path.exists():
        raise FileNotFoundError(
            f"❌ {norm_path} 가 없습니다. 먼저 --fit 을 실행해서 정규화 파라미터를 생성하세요."
        )

    norm_params = json.loads(norm_path.read_text(encoding="utf-8"))

    addr_min = norm_params.get("modbus.addr", {}).get("min")
    addr_max = norm_params.get("modbus.addr", {}).get("max")
    fc_min   = norm_params.get("modbus.fc", {}).get("min")
    fc_max   = norm_params.get("modbus.fc", {}).get("max")
    qty_min  = norm_params.get("modbus.qty", {}).get("min")
    qty_max  = norm_params.get("modbus.qty", {}).get("max")
    bc_min   = norm_params.get("modbus.bc", {}).get("min")
    bc_max   = norm_params.get("modbus.bc", {}).get("max")

    rows_feat: List[Dict[str, float]] = []

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("protocol") != "modbus":
                continue

            raw = extract_modbus_raw(obj)
            if raw is None:
                continue

            addr = raw["addr"]
            fc   = raw["fc"]
            qty  = raw["qty"]
            bc   = raw["bc"]
            regs_addr = raw["regs_addr"]
            regs_val  = raw["regs_val"]

            addr_stats = compute_regs_addr_stats(regs_addr)
            val_stats  = compute_regs_val_stats(regs_val)

            feat = {
                "modbus_addr_norm": minmax_norm(addr, addr_min, addr_max),
                "modbus_fc_norm":   minmax_norm(fc, fc_min, fc_max),
                "modbus_qty_norm":  minmax_norm(qty, qty_min, qty_max),
                "modbus_bc_norm":   minmax_norm(bc, bc_min, bc_max),

                "regs_addr_count":  addr_stats["count"],
                "regs_addr_min":    addr_stats["min"],
                "regs_addr_max":    addr_stats["max"],
                "regs_addr_range":  addr_stats["range"],

                "regs_val_min":     val_stats["min"],
                "regs_val_max":     val_stats["max"],
                "regs_val_mean":    val_stats["mean"],
                "regs_val_std":     val_stats["std"],
            }
            rows_feat.append(feat)

    dtype = np.dtype([
        ("modbus_addr_norm", "f4"),
        ("modbus_fc_norm", "f4"),
        ("modbus_qty_norm", "f4"),
        ("modbus_bc_norm", "f4"),

        ("regs_addr_count", "f4"),
        ("regs_addr_min", "f4"),
        ("regs_addr_max", "f4"),
        ("regs_addr_range", "f4"),

        ("regs_val_min", "f4"),
        ("regs_val_max", "f4"),
        ("regs_val_mean", "f4"),
        ("regs_val_std", "f4"),
    ])

    data = np.zeros(len(rows_feat), dtype=dtype)

    for idx, feat in enumerate(rows_feat):
        for name in data.dtype.names:
            data[name][idx] = float(feat[name])

    npy_path = out_dir / "modbus.npy"
    np.save(npy_path, data)

    print("✅ TRANSFORM 완료")
    print(f"- modbus.npy 저장: {npy_path} shape={data.shape}")


# ---------------------------------------------
# MAIN
# ---------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="입력 JSONL 파일 경로")
    parser.add_argument("-o", "--output", required=True, help="출력 디렉토리 경로")
    parser.add_argument("--fit", action="store_true", help="정규화 파라미터 생성 + modbus.npy 생성")
    parser.add_argument("--transform", action="store_true", help="기존 파라미터로 modbus.npy 생성")

    args = parser.parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output)

    if args.fit:
        fit_preprocess(input_path, out_dir)
    elif args.transform:
        transform_preprocess(input_path, out_dir)
    else:
        raise ValueError("❌ 반드시 --fit 또는 --transform 중 하나를 선택하세요.")


"""
최종 데이터 사용 예시 (modbus.npy)
    import numpy as np

    data = np.load("../result/output_modbus/modbus.npy")

    # shape: (N, )
    features = np.stack([
        data["modbus_addr_norm"],
        data["modbus_fc_norm"],
        data["modbus_qty_norm"],
        data["modbus_bc_norm"],
        data["regs_addr_count"],
        data["regs_addr_min"],
        data["regs_addr_max"],
        data["regs_addr_range"],
        data["regs_val_min"],
        data["regs_val_max"],
        data["regs_val_mean"],
        data["regs_val_std"],
    ], axis=1).astype("float32")
"""

"""
usage:
    # 1) 학습용 modbus 데이터에서 정규화 파라미터 + feature 생성
    python modbus.py --fit -i "../data/ML_DL 학습.jsonl" -o "../result/output_modbus"

    # 2) 이후 새 데이터에 대해 같은 파라미터로 전처리
    python modbus.py --transform -i "../data/ML_DL_새데이터.jsonl" -o "../result/output_modbus"
"""
