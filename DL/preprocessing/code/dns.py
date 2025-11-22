#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocess_dns_embed.py
A버전: DNS 전용 feature 전처리 (dns.qc, dns.ac만 사용)

두 모드 제공:
  --fit        : dns_norm_params 생성 후 dns.npy 저장
  --transform  : 기존 dns_norm_params 사용

입력 JSONL에서 사용하는 필드:
  - protocol == "dns"
  - dns.qc    : Query Count (예: "0","1","2","20"...)
  - dns.ac    : Answer Count (예: "0","1","2","4","6","17"...)

출력 feature (dns.npy, structured numpy):
  - dns_qc        (float32)  ← dns.qc 정수화 원본
  - dns_ac        (float32)  ← dns.ac 정수화 원본
  - dns_qc_norm   (float32)  ← Min-Max 정규화된 dns.qc
  - dns_ac_norm   (float32)  ← Min-Max 정규화된 dns.ac

정규화 파라미터:
  - dns_norm_params.json 에 저장:
    {
      "dns_qc_min": ...,
      "dns_qc_max": ...,
      "dns_ac_min": ...,
      "dns_ac_max": ...
    }
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------
# dns.qc / dns.ac 공통 파서
# ---------------------------------------------
def parse_int_field(val: Any) -> int:
    """
    "1", 1, ["1"], None 등 들어올 수 있는 값을
    안전하게 int로 변환. 실패하면 0.
    """
    if isinstance(val, list) and val:
        val = val[0]
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------
# 한 레코드(DNS) 전처리
# ---------------------------------------------
def preprocess_dns_record(obj: Dict[str, Any]) -> Dict[str, float]:
    """
    protocol == "dns" 인 레코드를 feature dict로 변환
    """
    feat: Dict[str, float] = {}

    qc_raw = obj.get("dns.qc")
    ac_raw = obj.get("dns.ac")

    qc_int = parse_int_field(qc_raw)
    ac_int = parse_int_field(ac_raw)

    feat["dns_qc"] = float(qc_int)
    feat["dns_ac"] = float(ac_int)

    return feat


# ---------------------------------------------
# FIT
# ---------------------------------------------
def fit_preprocess(input_path: Path, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # DNS만 처리
            if obj.get("protocol") != "dns":
                continue

            feat = preprocess_dns_record(obj)
            rows.append(feat)

    if not rows:
        raise ValueError("❌ DNS 레코드를 하나도 찾지 못했습니다.")

    # -------------------------------
    # Min/Max 계산
    # -------------------------------
    qc_vals = [r["dns_qc"] for r in rows]
    ac_vals = [r["dns_ac"] for r in rows]

    qc_min, qc_max = float(min(qc_vals)), float(max(qc_vals))
    ac_min, ac_max = float(min(ac_vals)), float(max(ac_vals))

    # 0 division 방지용
    qc_range = qc_max - qc_min if qc_max > qc_min else 1.0
    ac_range = ac_max - ac_min if ac_max > ac_min else 1.0

    norm_params = {
        "dns_qc_min": qc_min,
        "dns_qc_max": qc_max,
        "dns_ac_min": ac_min,
        "dns_ac_max": ac_max,
    }

    (out_dir / "dns_norm_params.json").write_text(
        json.dumps(norm_params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("✅ FIT 완료")
    print(f"- dns_norm_params.json 저장: {out_dir/'dns_norm_params.json'}")

    # -------------------------------
    # numpy 구조화 배열 생성
    # -------------------------------
    dtype = np.dtype([
        ("dns_qc", "f4"),
        ("dns_ac", "f4"),
        ("dns_qc_norm", "f4"),
        ("dns_ac_norm", "f4"),
    ])

    data = np.zeros(len(rows), dtype=dtype)

    for idx, feat in enumerate(rows):
        qc = float(feat.get("dns_qc", 0.0))
        ac = float(feat.get("dns_ac", 0.0))

        qc_norm = (qc - qc_min) / qc_range
        ac_norm = (ac - ac_min) / ac_range

        data["dns_qc"][idx]      = qc
        data["dns_ac"][idx]      = ac
        data["dns_qc_norm"][idx] = qc_norm
        data["dns_ac_norm"][idx] = ac_norm

    np.save(out_dir / "dns.npy", data)

    print(f"- dns.npy 저장: {out_dir/'dns.npy'}")
    print(f"- shape: {data.shape}")

    # 앞 5개 샘플 출력
    print("\n===== 앞 5개 DNS 전처리 샘플 =====")
    for i in range(min(5, len(data))):
        sample = {name: float(data[name][i]) for name in data.dtype.names}
        print(sample)


# ---------------------------------------------
# TRANSFORM
# ---------------------------------------------
def transform_preprocess(input_path: Path, out_dir: Path):

    norm_path = out_dir / "dns_norm_params.json"
    if not norm_path.exists():
        raise FileNotFoundError(f"❌ {norm_path} 가 없습니다. 먼저 --fit 을 실행하세요.")

    norm_params = json.loads(norm_path.read_text(encoding="utf-8"))

    qc_min = float(norm_params["dns_qc_min"])
    qc_max = float(norm_params["dns_qc_max"])
    ac_min = float(norm_params["dns_ac_min"])
    ac_max = float(norm_params["dns_ac_max"])

    qc_range = qc_max - qc_min if qc_max > qc_min else 1.0
    ac_range = ac_max - ac_min if ac_max > ac_min else 1.0

    rows: List[Dict[str, float]] = []

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("protocol") != "dns":
                continue

            feat = preprocess_dns_record(obj)
            rows.append(feat)

    if not rows:
        raise ValueError("❌ DNS 레코드를 하나도 찾지 못했습니다.")

    dtype = np.dtype([
        ("dns_qc", "f4"),
        ("dns_ac", "f4"),
        ("dns_qc_norm", "f4"),
        ("dns_ac_norm", "f4"),
    ])

    data = np.zeros(len(rows), dtype=dtype)

    for idx, feat in enumerate(rows):
        qc = float(feat.get("dns_qc", 0.0))
        ac = float(feat.get("dns_ac", 0.0))

        qc_norm = (qc - qc_min) / qc_range
        ac_norm = (ac - ac_min) / ac_range

        data["dns_qc"][idx]      = qc
        data["dns_ac"][idx]      = ac
        data["dns_qc_norm"][idx] = qc_norm
        data["dns_ac_norm"][idx] = ac_norm

    np.save(out_dir / "dns.npy", data)

    print("✅ TRANSFORM 완료")
    print(f"- dns.npy 저장: {out_dir/'dns.npy'} shape={data.shape}")


# ---------------------------------------------
# MAIN
# ---------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--transform", action="store_true")

    args = parser.parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output)

    if args.fit and args.transform:
        raise ValueError("❌ --fit 과 --transform 는 동시에 사용할 수 없습니다.")
    if not args.fit and not args.transform:
        raise ValueError("❌ 반드시 --fit 또는 --transform 중 하나를 선택하세요.")

    if args.fit:
        fit_preprocess(input_path, out_dir)
    else:
        transform_preprocess(input_path, out_dir)


"""
최종 데이터 사용 (dns.npy)
    import numpy as np

    data = np.load("../result/output_dns/dns.npy")

    # 원본 정수 값
    dns_qc = data["dns_qc"].astype("float32")
    dns_ac = data["dns_ac"].astype("float32")

    # 정규화 값 (0~1)
    dns_feat = np.stack([
        data["dns_qc_norm"],
        data["dns_ac_norm"],
    ], axis=1).astype("float32")
"""

"""
usage:
    # 학습용 DNS 데이터에서 norm_params + feature 생성
    python dns.py --fit -i "../data/ML_DL 학습.jsonl" -o "../result/output_dns"
    
    # 이후 새 데이터에 대해 같은 norm_params로 전처리
    python dns.py --transform -i "../data/ML_DL 학습.jsonl" -o "../result/output_dns"
"""
