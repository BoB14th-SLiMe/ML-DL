# 여기서 사용해야 하는 데이터 셋은 다음과 같음
# - smac : 11:22:33:44:55:66
# - dmac : AA:BB:CC:DD:EE:FF
# - sip : 192.168.0.10
# - dip : 192.168.0.11
# - sp : 502
# - dp : 510
# - dir : response, request
# - len : n 개수

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocess_host_embed.py  
A버전: MAC+IP 묶어서 host 단위 embedding

두 모드 제공:
  --fit        : host_map + norm_params 생성 후 common.npy 저장
  --transform  : 기존 host_map + norm_params 사용

출력 feature (common.npy):
  [src_host_id, dst_host_id, sp_norm, dp_norm, dir_code, len_norm]
"""

import json
import argparse
import numpy as np
from pathlib import Path


# ---------------------------------------------
# Host ID 생성기 (MAC+IP 묶어서 host 기준)
# ---------------------------------------------
def get_host_id_factory(host_map):
    next_id = max(host_map.values()) + 1 if host_map else 1

    def get_host_id(mac, ip):
        nonlocal next_id
        if not mac or not ip:
            return 0  # UNK
        key = f"{mac}|{ip}"
        if key not in host_map:
            host_map[key] = next_id
            next_id += 1
        return host_map[key]

    return get_host_id


# ---------------------------------------------
# FIT
# ---------------------------------------------
def fit_preprocess(input_path: Path, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    host_map = {}
    get_host_id = get_host_id_factory(host_map)

    rows = []
    sp_vals, dp_vals, len_vals = [], [], []

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            try:
                obj = json.loads(line.strip())
            except:
                continue

            smac = obj.get("smac")
            sip  = obj.get("sip")
            dmac = obj.get("dmac")
            dip  = obj.get("dip")
            sp   = obj.get("sp")
            dp   = obj.get("dp")
            dir_raw = obj.get("dir")
            length  = obj.get("len")

            # Host ID (MAC+IP 묶기)
            src_id = get_host_id(smac, sip)
            dst_id = get_host_id(dmac, dip)

            # dir code
            dir_code = 1 if dir_raw == "request" else 0

            # numeric
            try: sp = int(sp) if sp else 0
            except: sp = 0
            try: dp = int(dp) if dp else 0
            except: dp = 0
            try: length = int(length) if length else 0
            except: length = 0

            sp_vals.append(sp)
            dp_vals.append(dp)
            len_vals.append(length)

            rows.append((src_id, dst_id, sp, dp, dir_code, length))

    # min/max 계산
    norm_params = {
        "sp_min": min(sp_vals),  "sp_max": max(sp_vals),
        "dp_min": min(dp_vals),  "dp_max": max(dp_vals),
        "len_min": min(len_vals), "len_max": max(len_vals),
    }

    # JSON 저장
    (out_dir / "common_host_map.json").write_text(
        json.dumps(host_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "common_norm_params.json").write_text(
        json.dumps(norm_params, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("✅ FIT 완료")
    print(f"- host_map.json 저장: {out_dir/'common_host_map.json'}")
    print(f"- norm_params.json 저장: {out_dir/'common_norm_params.json'}")

    # numpy 구조화 배열 생성 (int + float 같이 저장)
    dtype = np.dtype([
        ("src_host_id", "i4"),   # int32
        ("dst_host_id", "i4"),   # int32
        ("sp_norm",    "f4"),    # float32
        ("dp_norm",    "f4"),    # float32
        ("dir_code",   "f4"),    # float32 (0.0 / 1.0)
        ("len_norm",   "f4"),    # float32
    ])

    data = np.zeros(len(rows), dtype=dtype)

    for idx, (src_id, dst_id, sp, dp, dir_code, length) in enumerate(rows):
        sp_norm  = (sp - norm_params["sp_min"])  / (norm_params["sp_max"] - norm_params["sp_min"] + 1e-9)
        dp_norm  = (dp - norm_params["dp_min"])  / (norm_params["dp_max"] - norm_params["dp_min"] + 1e-9)
        len_norm = (length - norm_params["len_min"]) / (norm_params["len_max"] - norm_params["len_min"] + 1e-9)

        data["src_host_id"][idx] = src_id
        data["dst_host_id"][idx] = dst_id
        data["sp_norm"][idx]     = sp_norm
        data["dp_norm"][idx]     = dp_norm
        data["dir_code"][idx]    = float(dir_code)
        data["len_norm"][idx]    = len_norm

    np.save(out_dir / "common.npy", data)

    print(f"- common.npy 저장: {out_dir/'common.npy'}")
    print(f"- shape: {data.shape}")

    # ----- 앞 5개 전처리 샘플 출력 -----
    print("\n===== 앞 5개 전처리 샘플 =====")
    for i in range(min(5, len(data))):
        print({
            "src_host_id": data['src_host_id'][i],
            "dst_host_id": data['dst_host_id'][i],
            "sp_norm":     data['sp_norm'][i],
            "dp_norm":     data['dp_norm'][i],
            "dir_code":    data['dir_code'][i],
            "len_norm":    data['len_norm'][i],
        })


# ---------------------------------------------
# TRANSFORM
# ---------------------------------------------
def transform_preprocess(input_path: Path, out_dir: Path):

    host_map = json.loads((out_dir / "common_host_map.json").read_text(encoding="utf-8"))
    norm_params = json.loads((out_dir / "common_norm_params.json").read_text(encoding="utf-8"))

    get_host_id = get_host_id_factory(host_map)

    rows = []

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            try:
                obj = json.loads(line.strip())
            except:
                continue

            smac = obj.get("smac")
            sip  = obj.get("sip")
            dmac = obj.get("dmac")
            dip  = obj.get("dip")
            sp   = obj.get("sp")
            dp   = obj.get("dp")
            dir_raw = obj.get("dir")
            length  = obj.get("len")

            src_id = get_host_id(smac, sip)
            dst_id = get_host_id(dmac, dip)

            try: sp = int(sp) if sp else 0
            except: sp = 0
            try: dp = int(dp) if dp else 0
            except: dp = 0
            try: length = int(length) if length else 0
            except: length = 0

            dir_code = 1 if dir_raw == "request" else 0

            sp_norm  = (sp - norm_params["sp_min"])  / (norm_params["sp_max"] - norm_params["sp_min"] + 1e-9)
            dp_norm  = (dp - norm_params["dp_min"])  / (norm_params["dp_max"] - norm_params["dp_min"] + 1e-9)
            len_norm = (length - norm_params["len_min"]) / (norm_params["len_max"] - norm_params["len_min"] + 1e-9)

            rows.append([src_id, dst_id, sp_norm, dp_norm, dir_code, len_norm])

    dtype = np.dtype([
        ("src_host_id", "i4"),
        ("dst_host_id", "i4"),
        ("sp_norm",    "f4"),
        ("dp_norm",    "f4"),
        ("dir_code",   "f4"),
        ("len_norm",   "f4"),
    ])

    data = np.zeros(len(rows), dtype=dtype)
    for idx, (src_id, dst_id, sp_norm, dp_norm, dir_code, len_norm) in enumerate(rows):
        data["src_host_id"][idx] = src_id
        data["dst_host_id"][idx] = dst_id
        data["sp_norm"][idx]     = sp_norm
        data["dp_norm"][idx]     = dp_norm
        data["dir_code"][idx]    = float(dir_code)
        data["len_norm"][idx]    = len_norm

    np.save(out_dir / "common.npy", data)

    print("✅ TRANSFORM 완료")
    print(f"- common.npy 저장: {out_dir/'common.npy'} shape={data.shape}")


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

    if args.fit:
        fit_preprocess(input_path, out_dir)
    elif args.transform:
        transform_preprocess(input_path, out_dir)
    else:
        raise ValueError("❌ 반드시 --fit 또는 --transform 중 하나를 선택하세요.")


"""
최종 데이터 사용 (common.npy)
    import numpy as np

    data = np.load("output_dir/common.npy")   # dtype: structured

    src_ids = data["src_host_id"]   # int32 → Embedding에 그대로
    dst_ids = data["dst_host_id"]   # int32

    numeric = np.stack([
        data["sp_norm"],
        data["dp_norm"],
        data["dir_code"],
        data["len_norm"],
    ], axis=1).astype("float32")    # (N, 4)
"""

"""
usage:
    python common.py --fit -i "../data/ML_DL 학습.jsonl" -o "../result/output_common"
    python common.py --transform -i new.jsonl -o output_dir
"""