#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
packet_feature_extractor.py

- 하나의 패킷(dict)을 통합 feature 벡터로 변환
- sequence_group (패킷 리스트)를 받아서 [seq_len, feat_dim] 행렬로 변환
- 슬라이딩 윈도우는 이 벡터 시퀀스에 대해서 네가 직접 만들면 됨

필요 전처리 파라미터:
  pre_dir/
    - common_host_map.json
    - common_norm_params.json
    - s7comm_norm_params.json
    - modbus_norm_params.json
    - xgt_var_vocab.json
    - xgt_fen_norm_params.json
    - arp_host_map.json
    - dns_norm_params.json
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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


def protocol_to_code(p: str) -> int:
    if not p:
        return 0
    return PROTOCOL_MAP.get(p, 0)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"❌ 필요 파일 없음: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def minmax_norm(x: float, vmin: Optional[float], vmax: Optional[float]) -> float:
    if vmin is None or vmax is None:
        return 0.0
    if vmax <= vmin:
        return 0.0
    return (x - vmin) / (vmax - vmin + 1e-9)


def safe_int(val: Any, default: int = 0) -> int:
    """
    JSON에서 읽은 값이
      - 정수
      - "10" 같은 10진 문자열
      - "0x0058" 같은 16진 문자열
      - ["0x0058"] 같은 리스트
    도 전부 안전하게 int로 바꿔주는 함수.
    """
    try:
        if isinstance(val, list) and val:
            val = val[0]

        if isinstance(val, str):
            s = val.strip()
            # 먼저 base=0으로 시도 (0x.., 0o.., 0b.. 지원)
            try:
                return int(s, 0)
            except Exception:
                # 그래도 안 되면 그냥 10진수로 한 번 더 시도
                return int(s)

        return int(val)
    except Exception:
        return default



def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if isinstance(val, list) and val:
            val = val[0]
        return float(val)
    except Exception:
        return default


# ==========================
# host / var id factory
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


def get_var_id_factory(var_map: Dict[str, int]):
    next_id = max(var_map.values()) + 1 if var_map else 1

    def get_var_id(var: Any) -> int:
        nonlocal next_id
        if not var:
            return 0
        var_str = str(var)
        if var_str not in var_map:
            var_map[var_str] = next_id
            next_id += 1
        return var_map[var_str]

    return get_var_id


# ==========================
# common host embed
# ==========================

def build_common_features(
    obj: Dict[str, Any],
    get_host_id_common: Any,
    norm_params: Dict[str, Any],
) -> Dict[str, float]:
    smac = obj.get("smac")
    sip = obj.get("sip")
    dmac = obj.get("dmac")
    dip = obj.get("dip")
    sp = safe_int(obj.get("sp"))
    dp = safe_int(obj.get("dp"))
    length = safe_int(obj.get("len"))
    dir_raw = obj.get("dir")

    src_id = get_host_id_common(smac, sip)
    dst_id = get_host_id_common(dmac, dip)

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

    ros_norm = minmax_norm(float(ros), ros_cfg.get("min"), ros_cfg.get("max"))
    db_norm = minmax_norm(float(db), db_cfg.get("min"), db_cfg.get("max"))
    addr_norm = minmax_norm(float(addr), addr_cfg.get("min"), addr_cfg.get("max"))

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
        out = []
        for v in val:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out
    return []


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
) -> Dict[str, float]:
    addr = safe_int(obj.get("modbus.addr"))
    fc = safe_int(obj.get("modbus.fc"))
    qty = safe_int(obj.get("modbus.qty"))
    bc = safe_int(obj.get("modbus.bc"))

    regs_addr = obj.get("modbus.regs.addr")
    regs_val = obj.get("modbus.regs.val")

    addr_cfg = norm_params.get("modbus.addr", {})
    fc_cfg = norm_params.get("modbus.fc", {})
    qty_cfg = norm_params.get("modbus.qty", {})
    bc_cfg = norm_params.get("modbus.bc", {})

    addr_norm = minmax_norm(float(addr), addr_cfg.get("min"), addr_cfg.get("max"))
    fc_norm = minmax_norm(float(fc), fc_cfg.get("min"), fc_cfg.get("max"))
    qty_norm = minmax_norm(float(qty), qty_cfg.get("min"), qty_cfg.get("max"))
    bc_norm = minmax_norm(float(bc), bc_cfg.get("min"), bc_cfg.get("max"))

    addrs = _parse_int_list(regs_addr)
    vals = _parse_int_list(regs_val)

    c, amin, amax, arange = _compute_regs_addr_stats(addrs)
    vmin, vmax, vmean, vstd = _compute_regs_val_stats(vals)

    return {
        "modbus_addr_norm": float(addr_norm),
        "modbus_fc_norm": float(fc_norm),
        "modbus_qty_norm": float(qty_norm),
        "modbus_bc_norm": float(bc_norm),
        "modbus_regs_count": float(c),
        "modbus_regs_addr_min": float(amin),
        "modbus_regs_addr_max": float(amax),
        "modbus_regs_addr_range": float(arange),
        "modbus_regs_val_min": float(vmin),
        "modbus_regs_val_max": float(vmax),
        "modbus_regs_val_mean": float(vmean),
        "modbus_regs_val_std": float(vstd),
    }


# ==========================
# xgt_fen feature
# ==========================

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
    get_var_id_xgt: Any,
    norm_params: Dict[str, Any],
) -> Dict[str, float]:
    feat: Dict[str, float] = {}

    source = safe_int(obj.get("xgt_fen.source"))
    datasize = safe_int(obj.get("xgt_fen.datasize"))
    cmd = safe_int(obj.get("xgt_fen.cmd"))
    dtype = safe_int(obj.get("xgt_fen.dtype"))
    blkcnt = safe_int(obj.get("xgt_fen.blkcnt"))
    errstat = safe_int(obj.get("xgt_fen.errstat"))
    errinfo = safe_int(obj.get("xgt_fen.errinfo"))
    fenetpos = safe_int(obj.get("xgt_fen.fenetpos"))

    xgt_fenet_base = fenetpos >> 4
    xgt_fenet_slot = fenetpos & 0x0F

    var_raw = obj.get("xgt_fen.vars")
    var_id = get_var_id_xgt(var_raw)
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

    # 🔥 여기서만 xgt_cmd 값을 한 번만 결정
    cmd_cfg = norm_params.get("xgt_cmd", {})
    cmd_min = cmd_cfg.get("min")
    cmd_max = cmd_cfg.get("max")

    if cmd_min is not None and cmd_max is not None:
        if cmd < cmd_min or cmd > cmd_max:
            xgt_cmd_feat = -2.0
        else:
            xgt_cmd_feat = minmax_norm(float(cmd), cmd_min, cmd_max)
    else:
        xgt_cmd_feat = float(cmd)

    feat["xgt_cmd"] = float(xgt_cmd_feat)

    # 나머지 피처들
    feat["xgt_var_id"] = float(var_id)
    feat["xgt_var_cnt"] = float(var_cnt)
    feat["xgt_source"] = float(source)
    feat["xgt_fenet_base"] = float(xgt_fenet_base)
    feat["xgt_fenet_slot"] = float(xgt_fenet_slot)
    feat["xgt_dtype"] = float(dtype)
    feat["xgt_blkcnt"] = float(blkcnt)
    feat["xgt_err_flag"] = 1.0 if (errstat != 0 or errinfo != 0) else 0.0
    feat["xgt_err_code"] = float(errinfo)
    feat["xgt_datasize"] = float(datasize)
    feat["xgt_data_missing"] = float(data_missing)
    feat["xgt_data_len_chars"] = float(data_len_chars)
    feat["xgt_data_num_spaces"] = float(num_spaces)
    feat["xgt_data_is_hex"] = float(is_hex)
    feat["xgt_data_n_bytes"] = float(n_bytes)
    feat["xgt_data_zero_ratio"] = float(zero_ratio)
    feat["xgt_data_first_byte"] = float(first_b)
    feat["xgt_data_last_byte"] = float(last_b)
    feat["xgt_data_mean_byte"] = float(mean_b)
    feat["xgt_data_bucket"] = float(bucket)

    return feat


# ==========================
# arp feature
# ==========================

def build_arp_features(
    obj: Dict[str, Any],
    get_host_id_arp: Any,
) -> Dict[str, float]:
    smac = obj.get("smac")
    sip = obj.get("sip")
    tmac = obj.get("arp.tmac")
    tip = obj.get("arp.tip")
    op = safe_int(obj.get("arp.op"))

    src_id = get_host_id_arp(smac, sip)
    tgt_id = get_host_id_arp(tmac, tip)

    return {
        "arp_src_host_id": float(src_id),
        "arp_tgt_host_id": float(tgt_id),
        "arp_op_num": float(op),
    }


# ==========================
# dns feature
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
        "dns_qc": float(qc),
        "dns_ac": float(ac),
        "dns_qc_norm": float(qc_norm),
        "dns_ac_norm": float(ac_norm),
    }


# ==========================
# feature 컬럼 정의
# (모델에 넣을 순서는 여기 순서 그대로)
# ==========================

PACKET_FEATURE_COLUMNS = [
    # 기본 메타
    "protocol", "delta_t",
    # common
    "src_host_id", "dst_host_id", "sp_norm", "dp_norm", "dir_code", "len_norm",
    # s7comm
    "s7comm_ros_norm", "s7comm_fn", "s7comm_db_norm", "s7comm_addr_norm",
    # modbus
    "modbus_addr_norm", "modbus_fc_norm", "modbus_qty_norm", "modbus_bc_norm",
    "modbus_regs_count", "modbus_regs_addr_min", "modbus_regs_addr_max",
    "modbus_regs_addr_range",
    "modbus_regs_val_min", "modbus_regs_val_max",
    "modbus_regs_val_mean", "modbus_regs_val_std",
    # xgt_fen
    "xgt_var_id", "xgt_var_cnt", "xgt_source", "xgt_fenet_base",
    "xgt_fenet_slot", "xgt_cmd", "xgt_dtype", "xgt_blkcnt",
    "xgt_err_flag", "xgt_err_code", "xgt_datasize", "xgt_data_missing",
    "xgt_data_len_chars", "xgt_data_num_spaces", "xgt_data_is_hex",
    "xgt_data_n_bytes", "xgt_data_zero_ratio",
    "xgt_data_first_byte", "xgt_data_last_byte",
    "xgt_data_mean_byte", "xgt_data_bucket",
    # arp
    "arp_src_host_id", "arp_tgt_host_id", "arp_op_num",
    # dns
    "dns_qc", "dns_ac", "dns_qc_norm", "dns_ac_norm",
]


# ==========================
# 전처리 파라미터 로딩
# ==========================

def load_preprocess_params(pre_dir: Path) -> Dict[str, Any]:
    """
    ../result 같은 디렉토리에서 필요한 json들을 한 번에 로딩해서
    dict로 리턴.
    """
    params: Dict[str, Any] = {}

    # common
    params["common_host_map"] = load_json(pre_dir / "common_host_map.json")
    params["common_norm_params"] = load_json(pre_dir / "common_norm_params.json")
    params["get_host_id_common"] = get_host_id_factory(params["common_host_map"])

    # s7comm
    params["s7comm_norm_params"] = load_json(pre_dir / "s7comm_norm_params.json")

    # modbus
    params["modbus_norm_params"] = load_json(pre_dir / "modbus_norm_params.json")

    # xgt_fen
    params["xgt_var_vocab"] = load_json(pre_dir / "xgt_var_vocab.json")
    params["xgt_fen_norm_params"] = load_json(pre_dir / "xgt_fen_norm_params.json")
    params["get_var_id_xgt"] = get_var_id_factory(params["xgt_var_vocab"])

    # arp
    params["arp_host_map"] = load_json(pre_dir / "arp_host_map.json")
    params["get_host_id_arp"] = get_host_id_factory(params["arp_host_map"])

    # dns
    params["dns_norm_params"] = load_json(pre_dir / "dns_norm_params.json")

    return params


# ==========================
# 핵심: 하나의 패킷 → feature dict / 벡터
# ==========================

def packet_to_feature_dict(
    pkt: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, float]:
    """
    하나의 패킷(dict)을 받아서 PACKET_FEATURE_COLUMNS 의
    모든 키를 가진 feature dict 로 변환.
    """
    protocol_str = pkt.get("protocol", "")
    protocol_code = protocol_to_code(protocol_str)
    delta_t = safe_float(pkt.get("delta_t", 0.0))

    feat: Dict[str, float] = {k: 0.0 for k in PACKET_FEATURE_COLUMNS}

    # 기본 메타
    feat["protocol"] = float(protocol_code)
    feat["delta_t"] = float(delta_t)

    # common
    common_feat = build_common_features(
        pkt,
        params["get_host_id_common"],
        params["common_norm_params"],
    )
    feat.update(common_feat)

    # 프로토콜별
    if protocol_str == "s7comm":
        s7_feat = build_s7comm_features(pkt, params["s7comm_norm_params"])
        feat.update(s7_feat)

    elif protocol_str == "modbus":
        mb_feat = build_modbus_features(pkt, params["modbus_norm_params"])
        feat.update(mb_feat)

    elif protocol_str == "xgt_fen":
        xgt_feat = build_xgt_fen_features(
            pkt,
            params["get_var_id_xgt"],
            params["xgt_fen_norm_params"],
        )
        feat.update(xgt_feat)

    elif protocol_str == "arp":
        arp_feat = build_arp_features(pkt, params["get_host_id_arp"])
        feat.update(arp_feat)

    elif protocol_str == "dns":
        dns_feat = build_dns_features(pkt, params["dns_norm_params"])
        feat.update(dns_feat)

    return feat


def packet_to_vector(
    pkt: Dict[str, Any],
    params: Dict[str, Any],
) -> List[float]:
    """
    하나의 패킷을 모델 입력용 feature 벡터(list[float])로 변환.
    순서는 PACKET_FEATURE_COLUMNS 순서 그대로.
    """
    feat_dict = packet_to_feature_dict(pkt, params)
    return [feat_dict[col] for col in PACKET_FEATURE_COLUMNS]


# ==========================
# sequence_group (패킷 리스트) → [seq_len, feat_dim]
# ==========================

def sequence_group_to_feature_matrix(
    sequence_group: List[Dict[str, Any]],
    params: Dict[str, Any],
) -> List[List[float]]:
    """
    window_obj["sequence_group"] 같은 패킷 리스트를
    [seq_len, feat_dim] 형태(파이썬 리스트)로 변환.
    """
    return [packet_to_vector(pkt, params) for pkt in sequence_group]
