#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
packet_to_feature_jsonl.py (기존 window_to_feature_csv_dynamic_index 로직을
패킷 단위로 단순화한 버전)

역할:
  - 패킷 단위 JSONL (각 line이 하나의 패킷)을 입력으로 받아서,
  - 각 패킷에 대해 feature를 추출한 뒤
  - 패킷 1개당 JSON 1줄로 출력한다.

입력 예시 (한 줄에 하나의 패킷):
{"@timestamp": "2025-09-22T11:52:00.727524Z", "protocol": "s7comm", ...}
{"@timestamp": "2025-09-22T11:52:00.728740Z", "protocol": "tcp", ...}
...

출력 예시 (한 줄에 하나의 feature row):
{
  "packet_idx": 0,
  "@timestamp": "...",
  "protocol_str": "s7comm",
  "protocol": 1.0,
  "delta_t": 0.0,
  "protocol_norm": ...,
  "src_host_id": ...,
  ...
}

delta_t 처리:
  - 입력에 "delta_t" 필드가 있으면 그대로 사용.
  - 없으면 @timestamp 기준으로 이전 패킷과의 시간 차이(초)를 계산.
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone

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


def minmax_norm(x: float, vmin: float, vmax: float) -> float:
    """
    vmin/vmax 가 없거나 이상하면 0.0,
    vmin == vmax 이면 (훈련 데이터가 상수였던 경우)
      - x <= vmin → 0.0
      - x  > vmin → 1.0 로 처리
    그 외에는 [0, 1] 로 클램핑해서 반환
    """
    if vmin is None or vmax is None:
        return 0.0

    if vmax == vmin:
        return 0.0 if x <= vmin else 1.0

    val = (x - vmin) / (vmax - vmin + 1e-9)
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
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


def parse_timestamp(ts: Any) -> float:
    """
    @timestamp 문자열을 epoch seconds 로 변환.
    포맷 예: "2025-09-22T11:52:00.727524Z"
    """
    if not ts:
        return 0.0
    s = str(ts).strip()
    # 끝에 'Z' 붙어 있으면 제거
    if s.endswith("Z"):
        s = s[:-1]
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


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

    addrs = _parse_int_list(regs_addr)
    vals = _parse_int_list(regs_val)

    c, amin, amax, arange = _compute_regs_addr_stats(addrs)
    vmin, vmax, vmean, vstd = _compute_regs_val_stats(vals)

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

    c_norm = minmax_norm(float(c), ra_count_min, ra_count_max)
    amin_norm = minmax_norm(float(amin), ra_min_min, ra_min_max)
    amax_norm = minmax_norm(float(amax), ra_max_min, ra_max_max)
    arange_norm = minmax_norm(float(arange), ra_range_min, ra_range_max)

    vmin_norm = minmax_norm(float(vmin), rv_min_min, rv_min_max)
    vmax_norm = minmax_norm(float(vmax), rv_max_min, rv_max_max)
    vmean_norm = minmax_norm(float(vmean), rv_mean_min, rv_mean_max)
    vstd_norm = minmax_norm(float(vstd), rv_std_min, rv_std_max)

    return {
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
    cfg = norm_params.get(key)
    if isinstance(cfg, dict):
        return cfg.get("min"), cfg.get("max")

    if key.startswith("xgt_"):
        suffix = key[len("xgt_"):]
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
) -> Dict[str, float]:
    feat_raw: Dict[str, float] = {}

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

    feat: Dict[str, float] = {}

    for k, v in feat_raw.items():
        if k in XGT_NORM_FIELDS:
            vmin, vmax = get_xgt_minmax(norm_params, k)
            feat[k] = float(minmax_norm(v, vmin, vmax))
        else:
            feat[k] = float(v)

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
        "dns_qc_norm": float(qc_norm),
        "dns_ac_norm": float(ac_norm),
    }


# ==========================
# 메타 + feature 컬럼 정의
# ==========================

FEATURE_COLUMNS = [
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


# ==========================
# 메인
# ==========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="패킷 단위 JSONL 경로"
    )
    parser.add_argument(
        "-p", "--pre_dir",
        required=True,
        help="전처리 파라미터 JSON들이 모여있는 디렉토리"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="출력 기준 경로 (기본: .jsonl)"
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="저장할 feature JSONL 경로 (생략 시 --output 의 .jsonl로 저장)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    pre_dir = Path(args.pre_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.json_output:
        jsonl_path = Path(args.json_output)
    else:
        jsonl_path = output_path.with_suffix(".jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # ----- 파라미터 로딩 -----
    common_host_map = load_json(pre_dir / "common_host_map.json")
    common_norm_params = load_json(pre_dir / "common_norm_params.json")

    s7comm_norm_params = load_json(pre_dir / "s7comm_norm_params.json")
    modbus_norm_params = load_json(pre_dir / "modbus_norm_params.json")

    xgt_var_vocab = load_json(pre_dir / "xgt_var_vocab.json")
    xgt_fen_norm_params = load_json(pre_dir / "xgt_fen_norm_params.json")

    arp_host_map = load_json(pre_dir / "arp_host_map.json")
    dns_norm_params = load_json(pre_dir / "dns_norm_params.json")

    packet_idx = 0
    total_output = 0
    prev_ts_sec = None

    with input_path.open("r", encoding="utf-8") as fin, \
            jsonl_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                pkt = json.loads(line)
            except Exception:
                continue

            ts_str = pkt.get("@timestamp")
            ts_sec = parse_timestamp(ts_str)

            # delta_t 우선순위: 필드가 있으면 그걸 사용, 없으면 timestamp 차이
            if "delta_t" in pkt:
                delta_t = safe_float(pkt.get("delta_t", 0.0))
            else:
                if prev_ts_sec is None:
                    delta_t = 0.0
                else:
                    delta_t = max(0.0, ts_sec - prev_ts_sec)
            prev_ts_sec = ts_sec

            protocol_str = pkt.get("protocol", "")
            protocol_code = protocol_to_code(protocol_str)

            # feature row 생성
            row: Dict[str, Any] = {}

            # 메타 정보
            row["@timestamp"] = ts_str
            row["protocol_str"] = protocol_str

            row["protocol"] = float(protocol_code)
            row["delta_t"] = float(delta_t)

            protocol_norm = minmax_norm(float(protocol_code), PROTOCOL_MIN, PROTOCOL_MAX)
            row["protocol_norm"] = float(protocol_norm)

            # 공통 feature
            common_feat = build_common_features(pkt, common_host_map, common_norm_params)
            row.update(common_feat)

            # 프로토콜별 feature
            if protocol_str == "s7comm":
                s7_feat = build_s7comm_features(pkt, s7comm_norm_params)
                row.update(s7_feat)
            elif protocol_str == "modbus":
                mb_feat = build_modbus_features(pkt, modbus_norm_params)
                row.update(mb_feat)
            elif protocol_str == "xgt_fen":
                xgt_feat = build_xgt_fen_features(pkt, xgt_var_vocab, xgt_fen_norm_params)
                row.update(xgt_feat)
            elif protocol_str == "arp":
                arp_feat = build_arp_features(pkt, arp_host_map)
                row.update(arp_feat)
            elif protocol_str == "dns":
                dns_feat = build_dns_features(pkt, dns_norm_params)
                row.update(dns_feat)

            # 없는 feature는 0으로 채우기 (일관된 스키마 유지)
            for k in FEATURE_COLUMNS:
                row.setdefault(k, 0.0)

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            packet_idx += 1
            total_output += 1

    print(f"✅ 완료: 총 {packet_idx}개 패킷 중 {total_output}개 feature row 출력")
    print(f"→ JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
