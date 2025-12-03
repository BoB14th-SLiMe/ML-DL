#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
window_to_feature_csv_dynamic_index.py

패턴 윈도우 JSONL (window_id, pattern, index, sequence_group 포함) →

1) --max-index 가 주어지면:
   - global_window_size = max_index 로 고정
   - 각 윈도우에 대해
       span = max(index) - min(index)
       → span < max_index 인 윈도우만 사용 (span 이 크거나 같은 윈도우는 버림)
         예) max_index=30 일 때
             index = [0, 30]  → span = 30  → 제거
             index = [1, 4, 5] → span = 4  → 통과

2) --max-index 를 주지 않으면:
   - global_window_size =
       • 우선 index 리스트 길이의 최댓값
       • 만약 index 가 없으면 sequence_group 길이의 최댓값
       • 그래도 없으면 1

추가 필터:
  - index 중복 제거 (같은 index는 첫 번째 패킷만 유지) + 오름차순 정렬
  - 중복 제거 후 index 개수가 1개인 윈도우는 제거

출력:
  - JSONL: 각 window에 대해
        {
          "window_id": ...,
          "pattern": ...,
          "index": [... base_idx 기준 0부터 시작하는 index ...],
          "base_idx": ...,
          "span": ...,
          "window_size": T_real,    # 실제 패킷 개수
          "sequence_group": [
             {
               "protocol": <code>,
               "delta_t": <float>,
               <FEATURE_COLUMNS ...>
             },
             ...
          ]
        }

추가:
  - 슬롯 메타 파일:
      modbus_addr_slot_vocab.json
      modbus_addr_slot_norm_params.json
      xgt_addr_slot_vocab.json
      xgt_addr_slot_norm_params.json
    을 읽어서, 각 슬롯(40012, 40013, D523, D524, ...)별로
      modbus_slot_40012_norm, xgt_slot_D523_norm
    같은 피처를 동적으로 추가 (alias 없이 주소 문자열 그대로 사용)
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re  # 슬롯 이름 sanitize 용

# ==========================
# 공통 유틸
# ==========================
PROTOCOL_MAP = {
    "s7comm": 1,
    "tcp": 2,
    "xgt_fen": 3,
    "modbus": 4,
    "arp": 5,
    "udp": 6,
    "unknown": 7,
    "dns": 8,
}

PROTOCOL_MIN = 0
PROTOCOL_MAX = max(PROTOCOL_MAP.values())


def protocol_to_code(p: str) -> int:
    if not p:
        return 0
    return PROTOCOL_MAP.get(p, 0)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"❌ 필요 파일 없음: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# def minmax_norm(x: float, vmin: float, vmax: float) -> float:
#     """
#     vmin/vmax 가 없거나 이상하면 0.0,
#     vmin == vmax 이면 (훈련 데이터가 상수였던 경우)
#       - x <= vmin → 0.0
#       - x  > vmin → 1.0 로 처리
#     그 외에는 [0, 1] 로 클램핑해서 반환
#     """
#     if vmin is None or vmax is None:
#         return 0.0

#     if vmax == vmin:
#         return 0.0 if x <= vmin else 1.0

#     val = (x - vmin) / (vmax - vmin + 1e-9)
#     if val < 0.0:
#         return 0.0
#     if val > 1.0:
#         return 1.0
#     return val

def minmax_norm_with_sentinel(
    x: float,
    vmin: float,
    vmax: float,
    sentinel: float = -2.0,
) -> float:
    """
    이산 코드(예: xgt_cmd, protocol 등)에 쓰기 좋은 버전:
      - vmin/vmax 밖이면 sentinel 반환
      - 안에 있으면 0~1로 스케일
    """
    if vmin is None or vmax is None:
        return 0.0

    # 범위를 벗어나면 센티널
    if x < vmin or x > vmax:
        return float(sentinel)

    if vmax == vmin:
        return 0.0

    val = (x - vmin) / (vmax - vmin + 1e-9)
    # 혹시 수치 에러 대비해서 0~1 클립
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return float(val)


def minmax_norm(x: float, vmin: float, vmax: float) -> float:
    """
    vmin/vmax 가 없거나 이상하면 0.0,
    vmin == vmax 이면 (훈련 데이터가 상수였던 경우)
      - x <= vmin → 0.0
      - x  > vmin → -2.0 (범위 밖 센티널)
    그 외:
      - vmin <= x <= vmax → [0, 1] 로 변환
      - x < vmin or x > vmax → -2.0
    """
    if vmin is None or vmax is None:
        return 0.0

    # 상수인 경우: 훈련 데이터는 항상 vmin==vmax
    if vmax == vmin:
        return 0.0 if x <= vmin else -2.0

    # 범위 밖이면 바로 센티널
    if x < vmin or x > vmax:
        return -2.0

    val = (x - vmin) / (vmax - vmin + 1e-9)
    # 이 경우는 이론상 0~1 안이므로 추가 클램핑은 생략해도 OK
    return val



def safe_int(val: Any, default: int = 0) -> int:
    try:
        if isinstance(val, list) and val:
            val = val[0]

        s = str(val).strip()
        if not s:
            return default

        # base=0 → "0x10", "010", "10" 모두 자동 처리
        return int(s, 0)
    except Exception:
        return default


def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if isinstance(val, list) and val:
            val = val[0]
        return float(val)
    except Exception:
        return default


def sanitize_slot_name(name: str) -> str:
    """슬롯 이름을 컬럼명으로 쓰기 좋게 변환 (영숫자/언더스코어만 유지)"""
    s = str(name)
    s = s.replace("%", "").replace(" ", "")
    s = re.sub(r"[^0-9A-Za-z_]", "_", s)
    return s


# ==========================
# common host embed (smac/sip, dmac/dip)
# ==========================


def get_host_id_factory(host_map: Dict[str, int]):
    next_id = max(host_map.values()) + 1 if host_map else 1

    def get_host_id(mac: Any, ip: Any) -> int:
        nonlocal next_id
        if not mac or not ip:
            return 0  # UNK
        key = f"{mac}|{ip}"
        if key not in host_map:
            host_map[key] = next_id
            next_id += 1
        return host_map[key]

    return get_host_id


def build_common_features(
    obj: Dict[str, Any],
    host_map: Dict[str, int],
    norm_params: Dict[str, Any],
) -> Dict[str, float]:
    get_host_id = get_host_id_factory(host_map)

    smac = obj.get("smac")
    sip = obj.get("sip")
    dmac = obj.get("dmac")
    dip = obj.get("dip")
    sp = safe_int(obj.get("sp"))
    dp = safe_int(obj.get("dp"))
    length = safe_int(obj.get("len"))
    dir_raw = obj.get("dir")

    src_id = get_host_id(smac, sip)
    dst_id = get_host_id(dmac, dip)

    dir_code = 1.0 if dir_raw == "request" else 0.0

    sp_min = norm_params["sp_min"]
    sp_max = norm_params["sp_max"]
    dp_min = norm_params["dp_min"]
    dp_max = norm_params["dp_max"]
    len_min = norm_params["len_min"]
    len_max = norm_params["len_max"]

    sp_norm = minmax_norm(float(sp), sp_min, sp_max)
    dp_norm = minmax_norm(float(dp), dp_min, dp_max)
    len_norm = minmax_norm(float(length), len_min, len_max)

    return {
        "src_host_id": float(src_id),
        "dst_host_id": float(dst_id),
        "sp_norm": float(sp_norm),
        "dp_norm": float(dp_norm),
        "dir_code": float(dir_code),
        "len_norm": float(len_norm),
    }


# ==========================
# s7comm feature
# ==========================


def build_s7comm_features(
    obj: Dict[str, Any],
    norm_params: Dict[str, Any],
) -> Dict[str, float]:
    ros = safe_int(obj.get("s7comm.ros"))
    fn = safe_int(obj.get("s7comm.fn"))
    db = safe_int(obj.get("s7comm.db"))
    addr = safe_int(obj.get("s7comm.addr"))

    ros_cfg = norm_params.get("s7comm.ros", {})
    db_cfg = norm_params.get("s7comm.db", {})
    addr_cfg = norm_params.get("s7comm.addr", {})

    ros_min = ros_cfg.get("min")
    ros_max = ros_cfg.get("max")
    db_min = db_cfg.get("min")
    db_max = db_cfg.get("max")
    addr_min = addr_cfg.get("min")
    addr_max = addr_cfg.get("max")

    ros_norm = minmax_norm(float(ros), ros_min, ros_max)
    db_norm = minmax_norm(float(db), db_min, db_max)
    addr_norm = minmax_norm(float(addr), addr_min, addr_max)

    return {
        "s7comm_ros_norm": float(ros_norm),
        "s7comm_fn": float(fn),
        "s7comm_db_norm": float(db_norm),
        "s7comm_addr_norm": float(addr_norm),
    }


# ==========================
# modbus feature
# ==========================


def _parse_int_list(val: Any) -> List[int]:
    if isinstance(val, list):
        out: List[int] = []
        for v in val:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out
    return []


def _to_str_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(v).strip() for v in val if str(v).strip()]
    if val is None:
        return []
    s = str(val).strip()
    if not s:
        return []
    # "a,b", "a b" 둘 다 대충 쪼갬
    s = s.replace(";", ",").replace(" ", ",")
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _to_float_list(val: Any) -> List[float]:
    if isinstance(val, list):
        out: List[float] = []
        for v in val:
            try:
                out.append(float(v))
            except Exception:
                continue
        return out
    if val is None:
        return []
    s = str(val).strip()
    if not s:
        return []
    s = s.replace(";", ",")
    out: List[float] = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            continue
    return out


def _compute_regs_addr_stats(addrs: List[int]) -> Tuple[int, float, float, float]:
    if not addrs:
        return 0, 0.0, 0.0, 0.0
    c = len(addrs)
    amin = float(min(addrs))
    amax = float(max(addrs))
    return c, amin, amax, amax - amin


def _compute_regs_val_stats(vals: List[int]) -> Tuple[float, float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0, 0.0
    vmin = float(min(vals))
    vmax = float(max(vals))
    mean = float(sum(vals)) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = var ** 0.5
    return vmin, vmax, mean, std


def build_modbus_features(
    obj: Dict[str, Any],
    norm_params: Dict[str, Any],
    slot_config: Dict[str, Any] = None,
) -> Dict[str, float]:
    # --- 기본 필드 ---
    addr = safe_int(obj.get("modbus.addr"))
    fc = safe_int(obj.get("modbus.fc"))
    qty = safe_int(obj.get("modbus.qty"))
    bc = safe_int(obj.get("modbus.bc"))

    # 🔸 주소 리스트는 modbus.py와 동일하게 "translated_addr" 우선 사용
    #    - modbus.translated_addr 가 있으면 그걸 사용
    #    - 없으면 modbus.regs.addr 사용
    addr_source = obj.get("modbus.translated_addr")
    if addr_source is None:
        addr_source = obj.get("modbus.regs.addr")

    regs_addr = addr_source
    regs_val = obj.get("modbus.regs.val")

    # --- 기본 modbus 필드용 min/max ---
    addr_cfg = norm_params.get("modbus.addr", {})
    fc_cfg = norm_params.get("modbus.fc", {})
    qty_cfg = norm_params.get("modbus.qty", {})
    bc_cfg = norm_params.get("modbus.bc", {})

    addr_min = addr_cfg.get("min")
    addr_max = addr_cfg.get("max")
    fc_min = fc_cfg.get("min")
    fc_max = fc_cfg.get("max")
    qty_min = qty_cfg.get("min")
    qty_max = qty_cfg.get("max")
    bc_min = bc_cfg.get("min")
    bc_max = bc_cfg.get("max")

    addr_norm = minmax_norm(float(addr), addr_min, addr_max)
    fc_norm = minmax_norm(float(fc), fc_min, fc_max)
    qty_norm = minmax_norm(float(qty), qty_min, qty_max)
    bc_norm = minmax_norm(float(bc), bc_min, bc_max)

    # --- regs.* 통계 계산 (raw) ---
    addrs = _parse_int_list(regs_addr)
    vals = _parse_int_list(regs_val)  # 필요하면 float 리스트 파서로 바꿔도 OK

    c, amin, amax, arange = _compute_regs_addr_stats(addrs)
    vmin, vmax, vmean, vstd = _compute_regs_val_stats(vals)

    # --- regs_addr.* / regs_val.* min/max 로드 ---
    ra_count_cfg = norm_params.get("regs_addr.count", {})
    ra_min_cfg = norm_params.get("regs_addr.min", {})
    ra_max_cfg = norm_params.get("regs_addr.max", {})
    ra_range_cfg = norm_params.get("regs_addr.range", {})

    rv_min_cfg = norm_params.get("regs_val.min", {})
    rv_max_cfg = norm_params.get("regs_val.max", {})
    rv_mean_cfg = norm_params.get("regs_val.mean", {})
    rv_std_cfg = norm_params.get("regs_val.std", {})

    ra_count_min = ra_count_cfg.get("min")
    ra_count_max = ra_count_cfg.get("max")
    ra_min_min = ra_min_cfg.get("min")
    ra_min_max = ra_min_cfg.get("max")
    ra_max_min = ra_max_cfg.get("min")
    ra_max_max = ra_max_cfg.get("max")
    ra_range_min = ra_range_cfg.get("min")
    ra_range_max = ra_range_cfg.get("max")

    rv_min_min = rv_min_cfg.get("min")
    rv_min_max = rv_min_cfg.get("max")
    rv_max_min = rv_max_cfg.get("min")
    rv_max_max = rv_max_cfg.get("max")
    rv_mean_min = rv_mean_cfg.get("min")
    rv_mean_max = rv_mean_cfg.get("max")
    rv_std_min = rv_std_cfg.get("min")
    rv_std_max = rv_std_cfg.get("max")

    # --- regs.* 값들 min-max 정규화 ---
    c_norm = minmax_norm(float(c), ra_count_min, ra_count_max)
    amin_norm = minmax_norm(float(amin), ra_min_min, ra_min_max)
    amax_norm = minmax_norm(float(amax), ra_max_min, ra_max_max)
    arange_norm = minmax_norm(float(arange), ra_range_min, ra_range_max)

    vmin_norm = minmax_norm(float(vmin), rv_min_min, rv_min_max)
    vmax_norm = minmax_norm(float(vmax), rv_max_min, rv_max_max)
    vmean_norm = minmax_norm(float(vmean), rv_mean_min, rv_mean_max)
    vstd_norm = minmax_norm(float(vstd), rv_std_min, rv_std_max)

    feat: Dict[str, float] = {
        "modbus_addr_norm": float(addr_norm),
        "modbus_fc_norm": float(fc_norm),
        "modbus_qty_norm": float(qty_norm),
        "modbus_bc_norm": float(bc_norm),
        "modbus_regs_count": float(c_norm),
        "modbus_regs_addr_min": float(amin_norm),
        "modbus_regs_addr_max": float(amax_norm),
        "modbus_regs_addr_range": float(arange_norm),
        "modbus_regs_val_min": float(vmin_norm),
        "modbus_regs_val_max": float(vmax_norm),
        "modbus_regs_val_mean": float(vmean_norm),
        "modbus_regs_val_std": float(vstd_norm),
    }

    # --- translated_addr 슬롯별 feature (옵션) ---
    if slot_config:
        slot_names: List[str] = slot_config.get("slot_names", [])
        stats_cfg: Dict[str, Any] = slot_config.get("stats", {})
        feat_names: Dict[str, str] = slot_config.get("feat_names", {})

        addr_list = _to_str_list(obj.get("modbus.translated_addr"))
        if not addr_list:
            addr_list = _to_str_list(obj.get("modbus.regs.addr"))

        val_list = _to_float_list(obj.get("modbus.word_value"))
        if not val_list:
            val_list = _to_float_list(obj.get("modbus.regs.val"))

        value_map: Dict[str, float] = {}
        for a, v in zip(addr_list, val_list):
            if a not in value_map:
                value_map[a] = v

        for slot_name in slot_names:
            feat_name = feat_names.get(slot_name)
            if not feat_name:
                continue
            stat = stats_cfg.get(slot_name, {})
            vmin = stat.get("min")
            vmax = stat.get("max")
            raw_v = value_map.get(slot_name)
            if raw_v is None:
                feat[feat_name] = 0.0
            else:
                feat[feat_name] = float(minmax_norm(float(raw_v), vmin, vmax))

    return feat


# ==========================
# xgt_fen feature
# ==========================
XGT_NORM_FIELDS = [
    "xgt_var_cnt",
    "xgt_source",
    "xgt_fenet_base",
    "xgt_fenet_slot",
    "xgt_cmd",
    "xgt_dtype",
    "xgt_blkcnt",
    "xgt_err_code",
    "xgt_datasize",
    "xgt_data_len_chars",
    "xgt_data_num_spaces",
    "xgt_data_n_bytes",
]


def get_xgt_minmax(norm_params: Dict[str, Any], key: str) -> Tuple[float, float]:
    """
    xgt_fen_norm_params.json 에서 min/max 를 가져옴.
    - 먼저 key 그대로 (예: "xgt_cmd")
    - 없으면 legacy 이름 "xgt_fen.<suffix>" (예: "xgt_fen.cmd") 도 한 번 더 찾음
    """
    cfg = norm_params.get(key)
    if isinstance(cfg, dict):
        return cfg.get("min"), cfg.get("max")

    # 옛날 형식: xgt_fen.cmd, xgt_fen.dtype, ...
    if key.startswith("xgt_"):
        suffix = key[len("xgt_"):]  # "cmd", "dtype", "source" ...
        legacy_key = f"xgt_fen.{suffix}"
        cfg = norm_params.get(legacy_key)
        if isinstance(cfg, dict):
            return cfg.get("min"), cfg.get("max")

    return None, None


def get_var_id_factory(var_map: Dict[str, int]):
    next_id = max(var_map.values()) + 1 if var_map else 1

    def get_var_id(var: Any) -> int:
        nonlocal next_id
        if not var:
            return 0
        if not isinstance(var, str):
            var_str = str(var)
        else:
            var_str = var
        if var_str not in var_map:
            var_map[var_str] = next_id
            next_id += 1
        return var_map[var_str]

    return get_var_id


def _bucket_by_mean(mean_byte: float) -> int:
    if mean_byte <= 64:
        return 0
    elif mean_byte <= 128:
        return 1
    elif mean_byte <= 192:
        return 2
    else:
        return 3


def build_xgt_fen_features(
    obj: Dict[str, Any],
    var_map: Dict[str, int],
    norm_params: Dict[str, Any],
    slot_config: Dict[str, Any] = None,
) -> Dict[str, float]:
    # 1) RAW feature 우선 계산
    feat_raw: Dict[str, float] = {}

    source = safe_int(obj.get("xgt_fen.source"))
    datasize = safe_int(obj.get("xgt_fen.datasize"))
    cmd = safe_int(obj.get("xgt_fen.cmd"))  # 0x0055 → 85 이런 식
    dtype = safe_int(obj.get("xgt_fen.dtype"))
    blkcnt = safe_int(obj.get("xgt_fen.blkcnt"))
    errstat = safe_int(obj.get("xgt_fen.errstat"))
    errinfo = safe_int(obj.get("xgt_fen.errinfo"))
    fenetpos = safe_int(obj.get("xgt_fen.fenetpos"))

    xgt_fenet_base = fenetpos >> 4
    xgt_fenet_slot = fenetpos & 0x0F

    var_raw = obj.get("xgt_fen.vars")
    get_var_id = get_var_id_factory(var_map)
    var_id = get_var_id(var_raw)
    var_cnt = 1.0 if var_raw else 0.0

    data_raw = obj.get("xgt_fen.data")
    data_missing = 1.0 if data_raw is None else 0.0
    data_len_chars = float(len(data_raw)) if isinstance(data_raw, str) else 0.0
    num_spaces = float(data_raw.count(" ")) if isinstance(data_raw, str) else 0.0

    is_hex = 0.0
    bytes_list: List[int] = []
    if isinstance(data_raw, str):
        hex_str = data_raw.replace(" ", "")
        try:
            bs = bytes.fromhex(hex_str)
            is_hex = 1.0
            bytes_list = list(bs)
        except Exception:
            is_hex = 0.0

    n_bytes = float(len(bytes_list))
    zero_ratio = 0.0
    first_b = 0.0
    last_b = 0.0
    mean_b = 0.0
    bucket = 0.0

    if bytes_list:
        first_b = float(bytes_list[0])
        last_b = float(bytes_list[-1])
        mean_b = float(sum(bytes_list)) / len(bytes_list)
        zero_cnt = sum(1 for b in bytes_list if b == 0)
        zero_ratio = float(zero_cnt) / len(bytes_list)
        bucket = float(_bucket_by_mean(mean_b))

    # RAW 채우기
    feat_raw["xgt_var_id"] = float(var_id)  # 정규화 안 함 (ID)
    feat_raw["xgt_var_cnt"] = float(var_cnt)
    feat_raw["xgt_source"] = float(source)
    feat_raw["xgt_fenet_base"] = float(xgt_fenet_base)
    feat_raw["xgt_fenet_slot"] = float(xgt_fenet_slot)
    feat_raw["xgt_cmd"] = float(cmd)
    feat_raw["xgt_dtype"] = float(dtype)
    feat_raw["xgt_blkcnt"] = float(blkcnt)
    feat_raw["xgt_err_flag"] = 1.0 if (errstat != 0 or errinfo != 0) else 0.0
    feat_raw["xgt_err_code"] = float(errinfo)
    feat_raw["xgt_datasize"] = float(datasize)
    feat_raw["xgt_data_missing"] = float(data_missing)
    feat_raw["xgt_data_len_chars"] = float(data_len_chars)
    feat_raw["xgt_data_num_spaces"] = float(num_spaces)
    feat_raw["xgt_data_is_hex"] = float(is_hex)
    feat_raw["xgt_data_n_bytes"] = float(n_bytes)
    feat_raw["xgt_data_zero_ratio"] = float(zero_ratio)
    feat_raw["xgt_data_first_byte"] = float(first_b)
    feat_raw["xgt_data_last_byte"] = float(last_b)
    feat_raw["xgt_data_mean_byte"] = float(mean_b)
    feat_raw["xgt_data_bucket"] = float(bucket)

    # 2) 정규화 적용
    feat: Dict[str, float] = {}

    for k, v in feat_raw.items():
        if k == "xgt_cmd":
            vmin, vmax = get_xgt_minmax(norm_params, k)
            if v < vmin or v > vmax:
                feat[k] = -2.0
            else:
                feat[k] = minmax_norm(v, vmin, vmax)
        elif k in XGT_NORM_FIELDS:
            vmin, vmax = get_xgt_minmax(norm_params, k)
            feat[k] = float(minmax_norm(v, vmin, vmax))
        else:
            # 정규화 안 하는 필드는 raw 값 그대로
            feat[k] = float(v)

    # 3) translated_addr 슬롯별 feature (옵션)
    if slot_config:
        slot_names: List[str] = slot_config.get("slot_names", [])
        stats_cfg: Dict[str, Any] = slot_config.get("stats", {})
        feat_names: Dict[str, str] = slot_config.get("feat_names", {})

        addr_list = _to_str_list(obj.get("xgt_fen.translated_addr"))
        val_list = _to_float_list(obj.get("xgt_fen.word_value"))

        value_map: Dict[str, float] = {}
        for a, v in zip(addr_list, val_list):
            if a not in value_map:
                value_map[a] = v

        for slot_name in slot_names:
            feat_name = feat_names.get(slot_name)
            if not feat_name:
                continue
            stat = stats_cfg.get(slot_name, {})
            vmin = stat.get("min")
            vmax = stat.get("max")
            raw_v = value_map.get(slot_name)
            if raw_v is None:
                feat[feat_name] = 0.0
            else:
                feat[feat_name] = float(minmax_norm(float(raw_v), vmin, vmax))

    return feat


# ==========================
# arp feature
# ==========================


def build_arp_features(
    obj: Dict[str, Any],
    host_map: Dict[str, int],
) -> Dict[str, float]:
    get_host_id = get_host_id_factory(host_map)

    smac = obj.get("smac")
    sip = obj.get("sip")
    tmac = obj.get("arp.tmac")
    tip = obj.get("arp.tip")
    op = safe_int(obj.get("arp.op"))

    src_id = get_host_id(smac, sip)
    tgt_id = get_host_id(tmac, tip)

    return {
        "arp_src_host_id": float(src_id),
        "arp_tgt_host_id": float(tgt_id),
        "arp_op_num": float(op),
    }


# ==========================
# dns feature (정규화만 사용)
# ==========================


def build_dns_features(
    obj: Dict[str, Any],
    norm_params: Dict[str, Any],
) -> Dict[str, float]:
    qc = safe_int(obj.get("dns.qc"))
    ac = safe_int(obj.get("dns.ac"))

    qc_min = norm_params["dns_qc_min"]
    qc_max = norm_params["dns_qc_max"]
    ac_min = norm_params["dns_ac_min"]
    ac_max = norm_params["dns_ac_max"]

    qc_norm = minmax_norm(float(qc), qc_min, qc_max)
    ac_norm = minmax_norm(float(ac), ac_min, ac_max)

    return {
        "dns_qc_norm": float(qc_norm),
        "dns_ac_norm": float(ac_norm),
    }


# ==========================
# 메인 변환 로직
# ==========================

META_COLUMNS = [
    "window_id",
    "pattern",
    "protocol",
    "delta_t",
]

# 기존 고정 feature 목록 (translated_addr 슬롯 제외)
BASE_FEATURE_COLUMNS = [
    # protocol one-hot 대신 scalar + 정규화
    "protocol_norm",
    # common
    "src_host_id",
    "dst_host_id",
    "sp_norm",
    "dp_norm",
    "dir_code",
    "len_norm",
    # s7comm
    "s7comm_ros_norm",
    "s7comm_fn",
    "s7comm_db_norm",
    "s7comm_addr_norm",
    # modbus
    "modbus_addr_norm",
    "modbus_fc_norm",
    "modbus_qty_norm",
    "modbus_bc_norm",
    "modbus_regs_count",
    "modbus_regs_addr_min",
    "modbus_regs_addr_max",
    "modbus_regs_addr_range",
    "modbus_regs_val_min",
    "modbus_regs_val_max",
    "modbus_regs_val_mean",
    "modbus_regs_val_std",
    # xgt_fen
    "xgt_var_id",
    "xgt_var_cnt",
    "xgt_source",
    "xgt_fenet_base",
    "xgt_fenet_slot",
    "xgt_cmd",
    "xgt_dtype",
    "xgt_blkcnt",
    "xgt_err_flag",
    "xgt_err_code",
    "xgt_datasize",
    "xgt_data_missing",
    "xgt_data_len_chars",
    "xgt_data_num_spaces",
    "xgt_data_is_hex",
    "xgt_data_n_bytes",
    "xgt_data_zero_ratio",
    "xgt_data_first_byte",
    "xgt_data_last_byte",
    "xgt_data_mean_byte",
    "xgt_data_bucket",
    # arp
    "arp_src_host_id",
    "arp_tgt_host_id",
    "arp_op_num",
    # dns
    "dns_qc_norm",
    "dns_ac_norm",
]

# 동적으로 채울 전역 리스트 (main에서 설정)
FEATURE_COLUMNS: List[str] = []
COLUMNS: List[str] = []


def main():
    global FEATURE_COLUMNS, COLUMNS

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="패턴 윈도우 JSONL 경로",
    )
    parser.add_argument(
        "-p",
        "--pre_dir",
        required=True,
        help="전처리 파라미터 JSON들이 모여있는 디렉토리",
    )
    parser.add_argument(
        "-o1",
        "--output1",
        required=True,
        help="출력 기준 경로 (기본: .jsonl)",
    )
    parser.add_argument(
        "-o2",
        "--output2",
        required=True,
        help="출력 기준 경로 (기본: .jsonl)",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="저장할 feature JSONL 경로 (생략 시 --output 의 .jsonl로 저장)",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=None,
        help="(옵션) window_size (T). 지정하면 "
             "span = max(index) - min(index) < T 인 윈도우만 사용",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    pre_dir = Path(args.pre_dir)
    output1_path = Path(args.output1)
    output2_path = Path(args.output2)
    output1_path.parent.mkdir(parents=True, exist_ok=True)
    output2_path.parent.mkdir(parents=True, exist_ok=True)

    jsonl_path1 = Path(args.output1)
    jsonl_path2 = Path(args.output2)

    jsonl_path1.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path2.parent.mkdir(parents=True, exist_ok=True)

    # ----- 파라미터 로딩 -----
    common_host_map = load_json(pre_dir / "common_host_map.json")
    common_norm_params = load_json(pre_dir / "common_norm_params.json")

    s7comm_norm_params = load_json(pre_dir / "s7comm_norm_params.json")
    modbus_norm_params = load_json(pre_dir / "modbus_norm_params.json")

    xgt_var_vocab = load_json(pre_dir / "xgt_var_vocab.json")
    xgt_fen_norm_params = load_json(pre_dir / "xgt_fen_norm_params.json")

    arp_host_map = load_json(pre_dir / "arp_host_map.json")
    dns_norm_params = load_json(pre_dir / "dns_norm_params.json")

    # 슬롯 메타 (있으면 슬롯별 feature 추가)
    modbus_slot_vocab = None
    modbus_slot_norm_params = None
    xgt_slot_vocab = None
    xgt_slot_norm_params = None

    # Modbus 슬롯 메타 로딩
    try:
        modbus_slot_vocab = load_json(pre_dir / "modbus_addr_slot_vocab.json")
    except FileNotFoundError:
        print("[WARN] modbus_addr_slot_vocab.json 없음 → modbus 슬롯 feature 미사용")
    try:
        modbus_slot_norm_params = load_json(pre_dir / "modbus_addr_slot_norm_params.json")
    except FileNotFoundError:
        print("[WARN] modbus_addr_slot_norm_params.json 없음 → modbus 슬롯 정규화 파라미터 없음 (0.0으로 대체)")

    # XGT-FEnet 슬롯 메타 로딩
    try:
        xgt_slot_vocab = load_json(pre_dir / "xgt_addr_slot_vocab.json")
    except FileNotFoundError:
        print("[WARN] xgt_addr_slot_vocab.json 없음 → xgt_fen 슬롯 feature 미사용")
    try:
        xgt_slot_norm_params = load_json(pre_dir / "xgt_addr_slot_norm_params.json")
    except FileNotFoundError:
        print("[WARN] xgt_addr_slot_norm_params.json 없음 → xgt_fen 슬롯 정규화 파라미터 없음 (0.0으로 대체)")

    # 동적 FEATURE_COLUMNS 구성
    FEATURE_COLUMNS = list(BASE_FEATURE_COLUMNS)
    modbus_slot_config = None
    xgt_slot_config = None

    if modbus_slot_vocab is not None:
        # vocab 의 index 순서대로 슬롯 정렬
        slot_names = sorted(modbus_slot_vocab.keys(), key=lambda k: modbus_slot_vocab[k])
        stats = modbus_slot_norm_params if modbus_slot_norm_params is not None else {}
        feat_names: Dict[str, str] = {}
        for addr in slot_names:
            safe = sanitize_slot_name(addr)
            col = f"modbus_slot_{safe}_norm"
            FEATURE_COLUMNS.append(col)
            feat_names[addr] = col
        modbus_slot_config = {
            "slot_names": slot_names,
            "stats": stats,
            "feat_names": feat_names,
        }

    if xgt_slot_vocab is not None:
        slot_names = sorted(xgt_slot_vocab.keys(), key=lambda k: xgt_slot_vocab[k])
        stats = xgt_slot_norm_params if xgt_slot_norm_params is not None else {}
        feat_names: Dict[str, str] = {}
        for addr in slot_names:
            safe = sanitize_slot_name(addr)
            col = f"xgt_slot_{safe}_norm"
            FEATURE_COLUMNS.append(col)
            feat_names[addr] = col
        xgt_slot_config = {
            "slot_names": slot_names,
            "stats": stats,
            "feat_names": feat_names,
        }

    COLUMNS = META_COLUMNS + FEATURE_COLUMNS

    # ----- 1PASS: 윈도우 로딩 -----
    windows: List[Dict[str, Any]] = []
    line_cnt_raw = 0

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                win_obj = json.loads(line)
            except Exception:
                continue
            windows.append(win_obj)
            line_cnt_raw += 1

    # ----- global_window_size 결정 (참고용) -----
    if args.max_index is not None:
        global_window_size = args.max_index
    else:
        global_window_size = 0
        # 1순위: index 리스트 길이의 최대값
        for w in windows:
            idx_list = w.get("index", [])
            if isinstance(idx_list, list) and len(idx_list) > global_window_size:
                global_window_size = len(idx_list)
        # 2순위: sequence_group 길이의 최대값
        if global_window_size <= 0:
            for w in windows:
                seq = w.get("sequence_group", [])
                if isinstance(seq, list) and len(seq) > global_window_size:
                    global_window_size = len(seq)
        if global_window_size <= 0:
            global_window_size = 1

    print(f"📦 총 윈도우 수: {len(windows)}")
    print(f"📏 span 필터 기준 global_window_size (--max-index): {global_window_size}")

    # ----- JSONL 작성 -----
    with jsonl_path1.open("w", encoding="utf-8") as fout_jsonl1, jsonl_path2.open("w", encoding="utf-8") as fout_jsonl2:

        win_cnt = 0
        skipped_by_span = 0
        skipped_empty = 0
        skipped_single_index = 0  # index 개수 1개인 윈도우 스킵 카운트
        total_row_cnt = 0  # 실제 출력 row 수 (모든 윈도우의 실제 패킷 합)

        for win_obj in windows:
            window_id = win_obj.get("window_id")
            pattern = win_obj.get("pattern") or win_obj.get("label")
            description = win_obj.get("description")

            # 1) 패킷 시퀀스 가져오기 (sequence_group / window_group / RAW fallback)
            seq_group = win_obj.get("sequence_group")
            if not isinstance(seq_group, list) or not seq_group:
                seq_group = win_obj.get("window_group") or win_obj.get("RAW") or []
            if not isinstance(seq_group, list):
                seq_group = []

            # 2) index 리스트 (없으면 0..len(seq_group)-1 로 생성)
            index_list = win_obj.get("index")
            if not isinstance(index_list, list):
                index_list = []
            if not index_list and seq_group:
                index_list = list(range(len(seq_group)))

            # 👉 index 중복 제거 + 오름차순 정렬 + sequence_group 재정렬
            if index_list:
                pair_list = list(zip(index_list, seq_group))
                unique_map: Dict[int, Any] = {}
                for idx, pkt in pair_list:
                    try:
                        idx_int = int(idx)
                    except Exception:
                        # 숫자로 못 바꾸면 그냥 스킵
                        continue
                    # 같은 index가 여러 개 있으면 첫 번째 것만 유지
                    if idx_int not in unique_map:
                        unique_map[idx_int] = pkt

                sorted_items = sorted(unique_map.items(), key=lambda x: x[0])
                index_list = [idx for idx, _ in sorted_items]
                seq_group = [pkt for _, pkt in sorted_items]

            # 👉 중복 제거 후 index 개수가 1개인 윈도우는 제거
            if len(index_list) == 1:
                skipped_single_index += 1
                continue

            # ----- span 계산 및 필터링 -----
            span = None
            if index_list:
                try:
                    idx_min = int(min(index_list))
                    idx_max = int(max(index_list))
                    span = idx_max - idx_min
                except Exception:
                    span = None

            if args.max_index is not None and span is not None:
                # span >= max_index → 제거
                if span >= args.max_index:
                    skipped_by_span += 1
                    continue

            # 실제 패킷이 하나도 없으면 스킵
            if not seq_group:
                skipped_empty += 1
                continue

            # base_idx = min(index_list) (비어있으면 0)
            if index_list:
                try:
                    base_idx = int(min(index_list))
                except Exception:
                    base_idx = 0
            else:
                base_idx = 0

            # 👉 상대 인덱스 리스트(0부터 시작) 생성 (이미 index_list는 정렬된 상태)
            rel_index_list: List[int] = []
            for idx in index_list:
                try:
                    idx_int = int(idx)
                except Exception:
                    continue
                rel_index_list.append(idx_int - base_idx)

            # 이 윈도우의 feature 시퀀스 (JSONL용)
            seq_feature_group: List[Dict[str, Any]] = []

            # index 기준으로 정렬하면서 feature 생성
            # (index_list / seq_group 둘 다 이미 오름차순 정렬된 상태라 정렬은 idempotent)
            for orig_idx, pkt in sorted(zip(index_list, seq_group), key=lambda x: int(x[0])):
                protocol_str = pkt.get("protocol", "")
                protocol_code = protocol_to_code(protocol_str)
                delta_t = safe_float(pkt.get("delta_t", 0.0))

                # feature용 row 딕셔너리
                row: Dict[str, Any] = {col: 0.0 for col in COLUMNS}
                row["window_id"] = window_id
                row["pattern"] = pattern
                row["description"] = description
                row["protocol"] = float(protocol_code)
                row["delta_t"] = float(delta_t)

                protocol_norm = minmax_norm(float(protocol_code), PROTOCOL_MIN, PROTOCOL_MAX)
                row["protocol_norm"] = float(protocol_norm)

                # 공통 feature
                common_feat = build_common_features(
                    pkt, common_host_map, common_norm_params
                )
                row.update(common_feat)

                # 프로토콜별 feature
                if protocol_str == "s7comm":
                    s7_feat = build_s7comm_features(pkt, s7comm_norm_params)
                    row.update(s7_feat)
                elif protocol_str == "modbus":
                    mb_feat = build_modbus_features(pkt, modbus_norm_params, modbus_slot_config)
                    row.update(mb_feat)
                elif protocol_str == "xgt_fen":
                    xgt_feat = build_xgt_fen_features(
                        pkt, xgt_var_vocab, xgt_fen_norm_params, xgt_slot_config
                    )
                    row.update(xgt_feat)
                elif protocol_str == "arp":
                    arp_feat = build_arp_features(pkt, arp_host_map)
                    row.update(arp_feat)
                elif protocol_str == "dns":
                    dns_feat = build_dns_features(pkt, dns_norm_params)
                    row.update(dns_feat)

                total_row_cnt += 1

                # JSONL 용 feature만 추출
                pkt_feat: Dict[str, Any] = {
                    "protocol": float(protocol_code),
                    "delta_t": float(delta_t),
                }
                for k in FEATURE_COLUMNS:
                    pkt_feat[k] = row[k]
                seq_feature_group.append(pkt_feat)

            window_size_real = len(seq_feature_group)

            # JSONL 출력 (원본 패킷 X, feature 시퀀스만)
            out_obj = {
                "window_id": window_id,
                "pattern": pattern,
                "orig_index": index_list,   # 이제는 중복 제거 + 오름차순 index
                "index": rel_index_list,    # base_idx 기준 0부터 시작하는 index
                "base_idx": base_idx,
                "span": span,
                "window_size": window_size_real,  # 실제 패킷 개수
                "sequence_group": seq_feature_group,
            }
            line = json.dumps(out_obj, ensure_ascii=False) + "\n"
            fout_jsonl1.write(line)
            fout_jsonl2.write(line)
            win_cnt += 1

    print(f"✅ 완료: 원본 {line_cnt_raw}개 라인 / {win_cnt}개 윈도우 처리")
    if args.max_index is not None:
        print(f"   ↳ span >= {args.max_index} 조건으로 스킵된 윈도우 수: {skipped_by_span}")
    print(f"   ↳ index 개수 == 1 이라 스킵된 윈도우 수: {skipped_single_index}")
    print(f"   ↳ 유효 패킷이 없어 스킵된 윈도우 수: {skipped_empty}")
    print(f"→ span 기준 global_window_size(--max-index 또는 자동): {global_window_size}")
    print(f"→ 총 row 수(실제 패킷 수 합): {total_row_cnt}")


if __name__ == "__main__":
    main()

"""
예시:
python 3.window_to_feature_csv_dynamic_index.py \
  --input "../data/pattern_windows.jsonl" \
  --pre_dir "../result" \
  --output "../../train/data/pattern_features1.jsonl" \
  --output2 "../../train/data/pattern_features2.jsonl" \
  --max-index 8
"""
