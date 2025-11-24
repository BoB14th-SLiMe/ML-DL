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
common.py  
A버전: MAC+IP 묶어서 host 단위 embedding

두 모드 제공:
  --fit        : host_map + norm_params 생성 후 common.npy 저장
  --transform  : 기존 host_map + norm_params 사용

입력 JSONL에서 사용하는 필드:
  - smac : 소스 MAC
  - dmac : 목적지 MAC
  - sip  : 소스 IP
  - dip  : 목적지 IP
  - sp   : 소스 포트
  - dp   : 목적지 포트
  - dir  : "request" / "response" / "unknown" ...
  - len  : TCP 이후 payload 길이

출력 feature (common.npy, structured numpy):
  - src_host_id (int32)  : (smac, sip) 조합 → ID, Embedding용
  - dst_host_id (int32)  : (dmac, dip) 조합 → ID, Embedding용
  - sp_norm    (float32) : sp min-max 정규화
  - dp_norm    (float32) : dp min-max 정규화
  - dir_code   (float32) : request=1.0, 그 외=0.0
  - len_norm   (float32) : len min-max 정규화

또한, 단일 패킷 dict 에 대해서도 아래 함수를 직접 호출해서
바로 feature dict 를 얻을 수 있음:

    feat = preprocess_common_record(pkt, get_host_id, norm_params)
"""

import json
import argparse
import numpy as np
from pathlib import Path


# ---------------------------------------------
# Host ID 생성기 (MAC+IP 묶어서 host 기준)
# ---------------------------------------------
def get_host_id_factory(host_map):
    """
    host_map: {"11:22:33:44:55:66|192.168.0.10": 1, ...}

    반환된 get_host_id 는 (mac, ip) → host_id 를 매핑하고,
    새로운 조합이 들어오면 host_map 에 추가하면서 ID 증가.
    """
    next_id = max(host_map.values()) + 1 if host_map else 1

    def get_host_id(mac, ip):
        nonlocal next_id
        if not mac or not ip:
            return 0  # UNK
        mac_str = str(mac).strip()
        ip_str = str(ip).strip()
        if not mac_str or not ip_str:
            return 0
        key = f"{mac_str}|{ip_str}"
        if key not in host_map:
            host_map[key] = next_id
            next_id += 1
        return host_map[key]

    return get_host_id


# ---------------------------------------------
# 단일 패킷 전처리 함수 (실시간/운영에서 사용)
# ---------------------------------------------
def preprocess_common_record(obj, get_host_id, norm_params):
    """
    단일 패킷(obj)을 common feature로 변환.
    host_map 은 get_host_id 의 클로저 안에서 갱신됨.

    obj 예시:
        {
          "smac": "11:22:33:44:55:66",
          "dmac": "AA:BB:CC:DD:EE:FF",
          "sip": "192.168.0.10",
          "dip": "192.168.0.11",
          "sp": 502,
          "dp": 510,
          "dir": "request",
          "len": 80,
        }

    norm_params 예시(common_norm_params.json):
        {
          "sp_min": ...,
          "sp_max": ...,
          "dp_min": ...,
          "dp_max": ...,
          "len_min": ...,
          "len_max": ...
        }

    반환:
        {
          "src_host_id": int,
          "dst_host_id": int,
          "sp_norm": float,
          "dp_norm": float,
          "dir_code": float,
          "len_norm": float,
        }
    """
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

    # dir code: request=1, 나머지(response/unknown 등)=0
    dir_code = 1 if dir_raw == "request" else 0

    # 안전한 int 변환
    try:
        sp = int(sp) if sp not in (None, "") else 0
    except Exception:
        sp = 0
    try:
        dp = int(dp) if dp not in (None, "") else 0
    except Exception:
        dp = 0
    try:
        length = int(length) if length not in (None, "") else 0
    except Exception:
        length = 0

    sp_min  = norm_params["sp_min"]
    sp_max  = norm_params["sp_max"]
    dp_min  = norm_params["dp_min"]
    dp_max  = norm_params["dp_max"]
    len_min = norm_params["len_min"]
    len_max = norm_params["len_max"]

    # min-max 정규화 (0으로 나누기 방지)
    def mm(v, vmin, vmax):
        if vmax <= vmin:
            return 0.0
        return (v - vmin) / (vmax - vmin + 1e-9)

    sp_norm  = mm(sp,  sp_min,  sp_max)
    dp_norm  = mm(dp,  dp_min,  dp_max)
    len_norm = mm(length, len_min, len_max)

    return {
        "src_host_id": src_id,
        "dst_host_id": dst_id,
        "sp_norm":     float(sp_norm),
        "dp_norm":     float(dp_norm),
        "dir_code":    float(dir_code),
        "len_norm":    float(len_norm),
    }


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
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
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
            try:
                sp = int(sp) if sp else 0
            except Exception:
                sp = 0
            try:
                dp = int(dp) if dp else 0
            except Exception:
                dp = 0
            try:
                length = int(length) if length else 0
            except Exception:
                length = 0

            sp_vals.append(sp)
            dp_vals.append(dp)
            len_vals.append(length)

            rows.append((src_id, dst_id, sp, dp, dir_code, length))

    if not sp_vals:
        # 데이터가 아예 없을 경우 방어
        sp_vals = [0]
        dp_vals = [0]
        len_vals = [0]

    # min/max 계산
    norm_params = {
        "sp_min": min(sp_vals),   "sp_max": max(sp_vals),
        "dp_min": min(dp_vals),   "dp_max": max(dp_vals),
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
    print(f"- common_host_map.json 저장: {out_dir/'common_host_map.json'}")
    print(f"- common_norm_params.json 저장: {out_dir/'common_norm_params.json'}")

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

    # 동일한 mm 함수 재사용
    def mm(v, vmin, vmax):
        if vmax <= vmin:
            return 0.0
        return (v - vmin) / (vmax - vmin + 1e-9)

    for idx, (src_id, dst_id, sp, dp, dir_code, length) in enumerate(rows):
        sp_norm  = mm(sp,  norm_params["sp_min"],  norm_params["sp_max"])
        dp_norm  = mm(dp,  norm_params["dp_min"],  norm_params["dp_max"])
        len_norm = mm(length, norm_params["len_min"], norm_params["len_max"])

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
            "src_host_id": int(data['src_host_id'][i]),
            "dst_host_id": int(data['dst_host_id'][i]),
            "sp_norm":     float(data['sp_norm'][i]),
            "dp_norm":     float(data['dp_norm'][i]),
            "dir_code":    float(data['dir_code'][i]),
            "len_norm":    float(data['len_norm'][i]),
        })


# ---------------------------------------------
# TRANSFORM
# ---------------------------------------------
def transform_preprocess(input_path: Path, out_dir: Path):

    host_map_path = out_dir / "common_host_map.json"
    norm_path = out_dir / "common_norm_params.json"

    if not host_map_path.exists():
        raise FileNotFoundError(f"❌ {host_map_path} 가 없습니다. 먼저 --fit 을 실행하세요.")
    if not norm_path.exists():
        raise FileNotFoundError(f"❌ {norm_path} 가 없습니다. 먼저 --fit 을 실행하세요.")

    host_map = json.loads(host_map_path.read_text(encoding="utf-8"))
    norm_params = json.loads(norm_path.read_text(encoding="utf-8"))

    # host_map 을 사용하는 host_id 생성기
    get_host_id = get_host_id_factory(host_map)

    rows = []

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            feat = preprocess_common_record(obj, get_host_id, norm_params)
            rows.append(feat)

    dtype = np.dtype([
        ("src_host_id", "i4"),
        ("dst_host_id", "i4"),
        ("sp_norm",    "f4"),
        ("dp_norm",    "f4"),
        ("dir_code",   "f4"),
        ("len_norm",   "f4"),
    ])

    data = np.zeros(len(rows), dtype=dtype)
    for idx, feat in enumerate(rows):
        data["src_host_id"][idx] = int(feat["src_host_id"])
        data["dst_host_id"][idx] = int(feat["dst_host_id"])
        data["sp_norm"][idx]     = float(feat["sp_norm"])
        data["dp_norm"][idx]     = float(feat["dp_norm"])
        data["dir_code"][idx]    = float(feat["dir_code"])
        data["len_norm"][idx]    = float(feat["len_norm"])

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

실시간 단일 패킷 예시:

    import json
    from pathlib import Path
    from preprocess_host_embed import get_host_id_factory, preprocess_common_record

    out_dir = Path("../result/output_common")
    host_map = json.loads((out_dir / "common_host_map.json").read_text(encoding="utf-8"))
    norm_params = json.loads((out_dir / "common_norm_params.json").read_text(encoding="utf-8"))

    get_host_id = get_host_id_factory(host_map)

    pkt = {
        "smac": "11:22:33:44:55:66",
        "dmac": "AA:BB:CC:DD:EE:FF",
        "sip": "192.168.0.10",
        "dip": "192.168.0.11",
        "sp": 502,
        "dp": 510,
        "dir": "request",
        "len": 80,
    }

    feat = preprocess_common_record(pkt, get_host_id, norm_params)
    # feat 딕셔너리에서 바로 model input으로 변환해서 사용

usage:
    python common.py --fit -i "../data/ML_DL 학습.jsonl" -o "../result/output_common"
    python common.py --transform -i new.jsonl -o output_dir
"""
