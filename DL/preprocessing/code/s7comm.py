#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocess_s7comm_feat.py
s7comm 전용 feature 전처리 (ARP 버전 참고)

두 모드 제공:
  --fit        : min-max 정규화 파라미터 생성 + s7comm.npy 저장
  --transform  : 기존 정규화 파라미터 사용하여 s7comm.npy 생성

입력 JSONL에서 사용하는 필드:
  - protocol == "s7comm"
  - s7comm.ros    : 1 또는 3 (min-max 정규화)
  - s7comm.fn     : 기능 코드 (int로 변환만, 정규화 X)
  - s7comm.db     : 데이터 블록 번호 (int → min-max 정규화)
  - s7comm.addr   : 레지스터 주소 (int → min-max 정규화)

출력 feature (s7comm.npy, structured numpy):
  - s7comm_ros_norm    (float32)  ← s7comm.ros min-max 정규화
  - s7comm_fn          (float32)  ← s7comm.fn 정수값 (그대로)
  - s7comm_db_norm     (float32)  ← s7comm.db min-max 정규화
  - s7comm_addr_norm   (float32)  ← s7comm.addr min-max 정규화

보조 파일:
  - s7comm_norm_params.json
      {
        "s7comm.ros":  {"min": ..., "max": ...},
        "s7comm.db":   {"min": ..., "max": ...},
        "s7comm.addr": {"min": ..., "max": ...}
      }
"""

import json
import argparse
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
# 한 레코드(s7comm)에서 raw 값 추출
# ---------------------------------------------
def extract_s7comm_raw(obj: Dict[str, Any]) -> Optional[Dict[str, Optional[int]]]:
    """
    protocol == "s7comm" 인 레코드에서 필요한 필드만 int로 파싱.

    반환:
      {
        "ros":  int or None,
        "fn":   int or None,
        "db":   int or None,
        "addr": int or None,
      }
    모든 값이 None 이면 None 반환.
    """
    ros = parse_int_field(obj.get("s7comm.ros"))
    fn = parse_int_field(obj.get("s7comm.fn"))
    db = parse_int_field(obj.get("s7comm.db"))
    addr = parse_int_field(obj.get("s7comm.addr"))

    if all(v is None for v in (ros, fn, db, addr)):
        return None

    return {
        "ros": ros,
        "fn": fn,
        "db": db,
        "addr": addr,
    }


# ---------------------------------------------
# FIT
# ---------------------------------------------
def fit_preprocess(input_path: Path, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    rows_raw: List[Dict[str, Optional[int]]] = []

    # min/max 추적용
    ros_min = ros_max = None
    db_min = db_max = None
    addr_min = addr_max = None

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("protocol") != "s7comm":
                continue

            raw = extract_s7comm_raw(obj)
            if raw is None:
                continue

            rows_raw.append(raw)

            # min/max 업데이트 (None 은 스킵)
            if raw["ros"] is not None:
                if ros_min is None or raw["ros"] < ros_min:
                    ros_min = raw["ros"]
                if ros_max is None or raw["ros"] > ros_max:
                    ros_max = raw["ros"]

            if raw["db"] is not None:
                if db_min is None or raw["db"] < db_min:
                    db_min = raw["db"]
                if db_max is None or raw["db"] > db_max:
                    db_max = raw["db"]

            if raw["addr"] is not None:
                if addr_min is None or raw["addr"] < addr_min:
                    addr_min = raw["addr"]
                if addr_max is None or raw["addr"] > addr_max:
                    addr_max = raw["addr"]

    # 정규화 파라미터 저장
    norm_params = {
        "s7comm.ros": {"min": ros_min, "max": ros_max},
        "s7comm.db": {"min": db_min, "max": db_max},
        "s7comm.addr": {"min": addr_min, "max": addr_max},
    }

    norm_path = out_dir / "s7comm_norm_params.json"
    norm_path.write_text(
        json.dumps(norm_params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("✅ FIT 완료")
    print(f"- s7comm_norm_params.json 저장: {norm_path}")

    # numpy 구조화 배열 생성
    dtype = np.dtype([
        ("s7comm_ros_norm", "f4"),
        ("s7comm_fn", "f4"),
        ("s7comm_db_norm", "f4"),
        ("s7comm_addr_norm", "f4"),
    ])

    data = np.zeros(len(rows_raw), dtype=dtype)

    for idx, raw in enumerate(rows_raw):
        ros = raw["ros"]
        fn = raw["fn"]
        db = raw["db"]
        addr = raw["addr"]

        data["s7comm_ros_norm"][idx] = minmax_norm(ros, ros_min, ros_max)
        data["s7comm_fn"][idx] = float(fn if fn is not None else 0.0)
        data["s7comm_db_norm"][idx] = minmax_norm(db, db_min, db_max)
        data["s7comm_addr_norm"][idx] = minmax_norm(addr, addr_min, addr_max)

    npy_path = out_dir / "s7comm.npy"
    np.save(npy_path, data)

    print(f"- s7comm.npy 저장: {npy_path}")
    print(f"- shape: {data.shape}")

    print("\n===== 앞 5개 s7comm 전처리 샘플 =====")
    for i in range(min(5, len(data))):
        sample = {name: data[name][i] for name in data.dtype.names}
        print(sample)


# ---------------------------------------------
# TRANSFORM
# ---------------------------------------------
def transform_preprocess(input_path: Path, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    norm_path = out_dir / "s7comm_norm_params.json"
    if not norm_path.exists():
        raise FileNotFoundError(
            f"❌ {norm_path} 가 없습니다. 먼저 --fit 을 실행해서 정규화 파라미터를 생성하세요."
        )

    norm_params = json.loads(norm_path.read_text(encoding="utf-8"))

    ros_min = norm_params.get("s7comm.ros", {}).get("min")
    ros_max = norm_params.get("s7comm.ros", {}).get("max")
    db_min = norm_params.get("s7comm.db", {}).get("min")
    db_max = norm_params.get("s7comm.db", {}).get("max")
    addr_min = norm_params.get("s7comm.addr", {}).get("min")
    addr_max = norm_params.get("s7comm.addr", {}).get("max")

    rows_norm: List[Dict[str, float]] = []

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("protocol") != "s7comm":
                continue

            raw = extract_s7comm_raw(obj)
            if raw is None:
                continue

            ros = raw["ros"]
            fn = raw["fn"]
            db = raw["db"]
            addr = raw["addr"]

            feat = {
                "s7comm_ros_norm": minmax_norm(ros, ros_min, ros_max),
                "s7comm_fn": float(fn if fn is not None else 0.0),
                "s7comm_db_norm": minmax_norm(db, db_min, db_max),
                "s7comm_addr_norm": minmax_norm(addr, addr_min, addr_max),
            }
            rows_norm.append(feat)

    dtype = np.dtype([
        ("s7comm_ros_norm", "f4"),
        ("s7comm_fn", "f4"),
        ("s7comm_db_norm", "f4"),
        ("s7comm_addr_norm", "f4"),
    ])

    data = np.zeros(len(rows_norm), dtype=dtype)

    for idx, feat in enumerate(rows_norm):
        data["s7comm_ros_norm"][idx] = float(feat["s7comm_ros_norm"])
        data["s7comm_fn"][idx] = float(feat["s7comm_fn"])
        data["s7comm_db_norm"][idx] = float(feat["s7comm_db_norm"])
        data["s7comm_addr_norm"][idx] = float(feat["s7comm_addr_norm"])

    npy_path = out_dir / "s7comm.npy"
    np.save(npy_path, data)

    print("✅ TRANSFORM 완료")
    print(f"- s7comm.npy 저장: {npy_path} shape={data.shape}")


# ---------------------------------------------
# MAIN
# ---------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="입력 JSONL 파일 경로")
    parser.add_argument("-o", "--output", required=True, help="출력 디렉토리 경로")
    parser.add_argument("--fit", action="store_true", help="정규화 파라미터 생성 + s7comm.npy 생성")
    parser.add_argument("--transform", action="store_true", help="기존 파라미터로 s7comm.npy 생성")

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
최종 데이터 사용 예시 (s7comm.npy)
    import numpy as np

    data = np.load("../result/output_s7comm/s7comm.npy")

    # shape: (N, )
    ros = data["s7comm_ros_norm"].astype("float32")
    fn  = data["s7comm_fn"].astype("float32")
    db  = data["s7comm_db_norm"].astype("float32")
    addr = data["s7comm_addr_norm"].astype("float32")

    features = np.stack([ros, fn, db, addr], axis=1).astype("float32")
"""

"""
usage:
    # 1) 학습용 s7comm 데이터에서 정규화 파라미터 + feature 생성
    python s7comm.py --fit -i "../data/ML_DL 학습.jsonl" -o "../result/output_s7comm"

    # 2) 이후 새 데이터에 대해 같은 파라미터로 전처리
    python s7comm.py --transform -i "../data/ML_DL_새데이터.jsonl" -o "../result/output_s7comm"
"""
