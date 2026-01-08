#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Constants
# =============================================================================
MISSING_DEFAULT = -1.0
SLOT_MISSING_SENTINEL = -1.0
SLOT_OOR_SENTINEL = -2.0

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

BASE_FEATURE_COLUMNS = [
    "protocol_norm",
    "src_host_id",
    "dst_host_id",
    "sp_norm",
    "dp_norm",
    "dir_code",
    "len_norm",
    "s7comm_ros_norm",
    "s7comm_fn",
    "s7comm_db_norm",
    "s7comm_addr_norm",
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
    "arp_src_host_id",
    "arp_tgt_host_id",
    "arp_op_num",
    "dns_qc_norm",
    "dns_ac_norm",
]


JsonDict = Dict[str, Any]


# =============================================================================
# Config helpers
# =============================================================================
def _load_yaml(path: Path) -> JsonDict:
    try:
        import yaml  # type: ignore
    except Exception:
        raise RuntimeError("PyYAML이 필요합니다. 설치: pip install pyyaml")
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("YAML root는 dict여야 합니다.")
    return obj


def _get(cfg: JsonDict, key: str, default: Any = None) -> Any:
    return cfg.get(key, default)


def _resolve_path(base_dir: Path, v: Any) -> Optional[Path]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    p = Path(s).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()


# =============================================================================
# IO helpers
# =============================================================================
def write_feature_weights_txt(out_jsonl_path: Path, feature_columns: List[str], weight: float = 1.0) -> Path:
    weights_path = out_jsonl_path.parent / "feature_weights.txt"
    ordered = ["protocol", "delta_t"] + list(feature_columns) + ["match"]

    seen = set()
    names: List[str] = []
    for n in ordered:
        if n in seen:
            continue
        seen.add(n)
        names.append(n)

    lines = ["#   use feature_name    weight"]
    for name in names:
        lines.append(f"O {name:<28} {float(weight):.1f}")

    weights_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return weights_path


def load_json(path: Path) -> JsonDict:
    if not path.exists():
        raise FileNotFoundError(str(path))
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError(f"json root must be dict: {path}")
    return obj


def load_json_or_default(path: Path, default: Optional[JsonDict] = None) -> JsonDict:
    if default is None:
        default = {}
    try:
        return load_json(path)
    except Exception:
        return default


# =============================================================================
# Parsing / numeric helpers
# =============================================================================
def protocol_to_code(p: Any) -> int:
    if not p:
        return 0
    return int(PROTOCOL_MAP.get(str(p), 0))


def safe_int(val: Any, default: int = 0) -> int:
    try:
        if isinstance(val, list) and val:
            val = val[0]
        s = str(val).strip()
        if not s:
            return default
        return int(s, 0)
    except Exception:
        return default


def safe_float(val: Any, default: float = MISSING_DEFAULT) -> float:
    try:
        if isinstance(val, list) and val:
            val = val[0]
        if val is None:
            return default
        if isinstance(val, str) and not val.strip():
            return default
        return float(val)
    except Exception:
        return default


def match_to_float(v: Any) -> float:
    if v is None:
        return float(MISSING_DEFAULT)
    if v in (1, True, "1", "true", "True", "O", "o", "OK", "ok"):
        return 1.0
    if v in (0, False, "0", "false", "False", "X", "x"):
        return 0.0
    return float(MISSING_DEFAULT)


def parse_packet_obj(pkt: Any) -> Optional[JsonDict]:
    if isinstance(pkt, dict):
        return pkt
    if isinstance(pkt, str):
        s = pkt.strip()
        if not s:
            return None
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
    return None


# =============================================================================
# Normalization
# =============================================================================
def minmax_norm(x: Any, vmin: Any, vmax: Any, oor: float = SLOT_OOR_SENTINEL, missing: float = 0.0) -> float:
    if vmin is None or vmax is None:
        return float(missing)
    try:
        x = float(x)
        vmin = float(vmin)
        vmax = float(vmax)
    except Exception:
        return float(missing)
    if x < vmin or x > vmax:
        return float(oor)
    if vmax == vmin:
        return 0.0
    return float((x - vmin) / (vmax - vmin + 1e-9))


def minmax_norm_with_sentinel(x: Any, vmin: Any, vmax: Any, sentinel: float = SLOT_OOR_SENTINEL) -> float:
    if vmin is None or vmax is None:
        return 0.0
    try:
        x = float(x)
        vmin = float(vmin)
        vmax = float(vmax)
    except Exception:
        return 0.0
    if vmax == vmin:
        return 0.0 if x == vmin else float(sentinel)
    if x < vmin or x > vmax:
        return float(sentinel)
    val = (x - vmin) / (vmax - vmin + 1e-9)
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return float(val)


# =============================================================================
# Small utils
# =============================================================================
def sanitize_slot_name(name: Any) -> str:
    s = str(name)
    s = s.replace("%", "").replace(" ", "")
    s = re.sub(r"[^0-9A-Za-z_]", "_", s)
    return s


def to_str_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(v).strip() for v in val if str(v).strip()]
    if val is None:
        return []
    s = str(val).strip()
    if not s:
        return []
    s = s.replace(";", ",").replace(" ", ",")
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def to_float_list(val: Any) -> List[float]:
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


def host_id_lookup(host_map: JsonDict, mac: Any, ip: Any) -> int:
    if not mac or not ip:
        return 0
    key = f"{str(mac).strip()}|{str(ip).strip()}"
    try:
        return int(host_map.get(key, 0))
    except Exception:
        return 0


def var_id_lookup(var_vocab: JsonDict, var_raw: Any) -> int:
    if not var_raw:
        return 0
    s = str(var_raw)
    try:
        return int(var_vocab.get(s, 0))
    except Exception:
        return 0


def get_xgt_minmax(norm_params: JsonDict, key: str):
    cfg = norm_params.get(key)
    if isinstance(cfg, dict):
        return cfg.get("min"), cfg.get("max")
    if key.startswith("xgt_"):
        suffix = key[len("xgt_") :]
        legacy = f"xgt_fen.{suffix}"
        cfg = norm_params.get(legacy)
        if isinstance(cfg, dict):
            return cfg.get("min"), cfg.get("max")
    return None, None


def bucket_by_mean(mean_byte: float) -> int:
    if mean_byte <= 64:
        return 0
    if mean_byte <= 128:
        return 1
    if mean_byte <= 192:
        return 2
    return 3


def parse_int_list(val: Any) -> List[int]:
    if isinstance(val, list):
        out: List[int] = []
        for v in val:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        parts = [p.strip() for p in s.replace(";", ",").replace(" ", ",").split(",")]
        out = []
        for p in parts:
            if not p:
                continue
            try:
                out.append(int(p, 0))
            except Exception:
                continue
        return out
    return []


def compute_regs_addr_stats(addrs: List[int]):
    if not addrs:
        return 0, 0.0, 0.0, 0.0
    c = len(addrs)
    amin = float(min(addrs))
    amax = float(max(addrs))
    return c, amin, amax, amax - amin


def compute_regs_val_stats(vals: List[int]):
    if not vals:
        return 0.0, 0.0, 0.0, 0.0
    vmin = float(min(vals))
    vmax = float(max(vals))
    mean = float(sum(vals)) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = var ** 0.5
    return vmin, vmax, mean, std


# =============================================================================
# Feature builders
# =============================================================================
def build_common_features(pkt: JsonDict, common_host_map: JsonDict, common_norm: JsonDict) -> Dict[str, float]:
    smac = pkt.get("smac")
    sip = pkt.get("sip")
    dmac = pkt.get("dmac")
    dip = pkt.get("dip")
    sp = safe_int(pkt.get("sp"), 0)
    dp = safe_int(pkt.get("dp"), 0)
    length = safe_int(pkt.get("len"), 0)
    dir_raw = pkt.get("dir")

    src_id = host_id_lookup(common_host_map, smac, sip)
    dst_id = host_id_lookup(common_host_map, dmac, dip)
    dir_code = 1.0 if dir_raw == "request" else 0.0

    sp_norm = minmax_norm(float(sp), common_norm.get("sp_min"), common_norm.get("sp_max"))
    dp_norm = minmax_norm(float(dp), common_norm.get("dp_min"), common_norm.get("dp_max"))
    len_norm = minmax_norm(float(length), common_norm.get("len_min"), common_norm.get("len_max"))

    return {
        "src_host_id": float(src_id),
        "dst_host_id": float(dst_id),
        "sp_norm": float(sp_norm),
        "dp_norm": float(dp_norm),
        "dir_code": float(dir_code),
        "len_norm": float(len_norm),
    }


def build_s7comm_features(pkt: JsonDict, s7_norm: JsonDict) -> Dict[str, float]:
    ros = safe_int(pkt.get("s7comm.ros"), 0)
    fn = safe_int(pkt.get("s7comm.fn"), 0)
    db = safe_int(pkt.get("s7comm.db"), 0)
    addr = safe_int(pkt.get("s7comm.addr"), 0)

    ros_cfg = s7_norm.get("s7comm.ros", {})
    db_cfg = s7_norm.get("s7comm.db", {})
    addr_cfg = s7_norm.get("s7comm.addr", {})

    return {
        "s7comm_ros_norm": float(minmax_norm(float(ros), ros_cfg.get("min"), ros_cfg.get("max"))),
        "s7comm_fn": float(fn),
        "s7comm_db_norm": float(minmax_norm(float(db), db_cfg.get("min"), db_cfg.get("max"))),
        "s7comm_addr_norm": float(minmax_norm(float(addr), addr_cfg.get("min"), addr_cfg.get("max"))),
    }


def build_modbus_features(pkt: JsonDict, modbus_norm: JsonDict, modbus_slot_config: Optional[JsonDict]) -> Dict[str, float]:
    addr = safe_int(pkt.get("modbus.addr"), 0)
    fc = safe_int(pkt.get("modbus.fc"), 0)
    qty = safe_int(pkt.get("modbus.qty"), 0)
    bc = safe_int(pkt.get("modbus.bc"), 0)

    addr_cfg = modbus_norm.get("modbus.addr", {})
    fc_cfg = modbus_norm.get("modbus.fc", {})
    qty_cfg = modbus_norm.get("modbus.qty", {})
    bc_cfg = modbus_norm.get("modbus.bc", {})

    feat: Dict[str, float] = {
        "modbus_addr_norm": float(minmax_norm(float(addr), addr_cfg.get("min"), addr_cfg.get("max"))),
        "modbus_fc_norm": float(minmax_norm(float(fc), fc_cfg.get("min"), fc_cfg.get("max"))),
        "modbus_qty_norm": float(minmax_norm(float(qty), qty_cfg.get("min"), qty_cfg.get("max"))),
        "modbus_bc_norm": float(minmax_norm(float(bc), bc_cfg.get("min"), bc_cfg.get("max"))),
    }

    addr_source = pkt.get("modbus.translated_addr")
    if addr_source is None:
        addr_source = pkt.get("modbus.regs.addr")
    regs_addr = addr_source
    regs_val = pkt.get("modbus.regs.val")

    addrs = parse_int_list(regs_addr)
    vals = parse_int_list(regs_val)

    c, amin, amax, arange = compute_regs_addr_stats(addrs)
    vmin, vmax, vmean, vstd = compute_regs_val_stats(vals)

    ra_count_cfg = modbus_norm.get("regs_addr.count", {})
    ra_min_cfg = modbus_norm.get("regs_addr.min", {})
    ra_max_cfg = modbus_norm.get("regs_addr.max", {})
    ra_range_cfg = modbus_norm.get("regs_addr.range", {})

    rv_min_cfg = modbus_norm.get("regs_val.min", {})
    rv_max_cfg = modbus_norm.get("regs_val.max", {})
    rv_mean_cfg = modbus_norm.get("regs_val.mean", {})
    rv_std_cfg = modbus_norm.get("regs_val.std", {})

    feat.update(
        {
            "modbus_regs_count": float(minmax_norm(float(c), ra_count_cfg.get("min"), ra_count_cfg.get("max"))),
            "modbus_regs_addr_min": float(minmax_norm(float(amin), ra_min_cfg.get("min"), ra_min_cfg.get("max"))),
            "modbus_regs_addr_max": float(minmax_norm(float(amax), ra_max_cfg.get("min"), ra_max_cfg.get("max"))),
            "modbus_regs_addr_range": float(minmax_norm(float(arange), ra_range_cfg.get("min"), ra_range_cfg.get("max"))),
            "modbus_regs_val_min": float(minmax_norm(float(vmin), rv_min_cfg.get("min"), rv_min_cfg.get("max"))),
            "modbus_regs_val_max": float(minmax_norm(float(vmax), rv_max_cfg.get("min"), rv_max_cfg.get("max"))),
            "modbus_regs_val_mean": float(minmax_norm(float(vmean), rv_mean_cfg.get("min"), rv_mean_cfg.get("max"))),
            "modbus_regs_val_std": float(minmax_norm(float(vstd), rv_std_cfg.get("min"), rv_std_cfg.get("max"))),
        }
    )

    if modbus_slot_config:
        slot_names = modbus_slot_config["slot_names"]
        stats_cfg = modbus_slot_config["stats"]
        feat_names = modbus_slot_config["feat_names"]

        addr_list = to_str_list(pkt.get("modbus.translated_addr"))
        if not addr_list:
            addr_list = to_str_list(pkt.get("modbus.regs.addr"))
        val_list = to_float_list(pkt.get("modbus.word_value"))
        if not val_list:
            val_list = to_float_list(pkt.get("modbus.regs.val"))

        value_map: Dict[str, float] = {}
        for a, v in zip(addr_list, val_list):
            if a not in value_map:
                value_map[a] = float(v)

        for slot_name in slot_names:
            feat_name = feat_names.get(slot_name)
            if not feat_name:
                continue
            raw_v = value_map.get(slot_name)
            if raw_v is None:
                feat[feat_name] = float(SLOT_MISSING_SENTINEL)
                continue
            stat = stats_cfg.get(slot_name, {}) if isinstance(stats_cfg, dict) else {}
            feat[feat_name] = float(
                minmax_norm_with_sentinel(raw_v, stat.get("min"), stat.get("max"), sentinel=SLOT_OOR_SENTINEL)
            )

    return feat


def build_xgt_fen_features(pkt: JsonDict, xgt_var_vocab: JsonDict, xgt_norm: JsonDict, xgt_slot_config: Optional[JsonDict]) -> Dict[str, float]:
    source = safe_int(pkt.get("xgt_fen.source"), 0)
    datasize = safe_int(pkt.get("xgt_fen.datasize"), 0)
    cmd = safe_int(pkt.get("xgt_fen.cmd"), 0)
    dtype = safe_int(pkt.get("xgt_fen.dtype"), 0)
    blkcnt = safe_int(pkt.get("xgt_fen.blkcnt"), 0)
    errstat = safe_int(pkt.get("xgt_fen.errstat"), 0)
    errinfo = safe_int(pkt.get("xgt_fen.errinfo"), 0)
    fenetpos = safe_int(pkt.get("xgt_fen.fenetpos"), 0)

    xgt_fenet_base = fenetpos >> 4
    xgt_fenet_slot = fenetpos & 0x0F

    var_raw = pkt.get("xgt_fen.vars")
    var_id = var_id_lookup(xgt_var_vocab, var_raw)
    var_cnt = 1.0 if var_raw else 0.0

    data_raw = pkt.get("xgt_fen.data")
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
        bucket = float(bucket_by_mean(mean_b))

    feat: Dict[str, float] = {"xgt_var_id": float(var_id)}

    for key, raw_val in [
        ("xgt_var_cnt", var_cnt),
        ("xgt_source", float(source)),
        ("xgt_fenet_base", float(xgt_fenet_base)),
        ("xgt_fenet_slot", float(xgt_fenet_slot)),
        ("xgt_cmd", float(cmd)),
        ("xgt_dtype", float(dtype)),
        ("xgt_blkcnt", float(blkcnt)),
        ("xgt_err_code", float(errinfo)),
        ("xgt_datasize", float(datasize)),
        ("xgt_data_len_chars", float(data_len_chars)),
        ("xgt_data_num_spaces", float(num_spaces)),
        ("xgt_data_n_bytes", float(n_bytes)),
    ]:
        vmin, vmax = get_xgt_minmax(xgt_norm, key)
        feat[key] = float(minmax_norm(raw_val, vmin, vmax)) if (vmin is not None and vmax is not None) else 0.0

    feat["xgt_err_flag"] = 1.0 if (errstat != 0 or errinfo != 0) else 0.0
    feat["xgt_err_code"] = feat.get("xgt_err_code", 0.0)
    feat["xgt_data_missing"] = float(data_missing)
    feat["xgt_data_is_hex"] = float(is_hex)
    feat["xgt_data_zero_ratio"] = float(zero_ratio)
    feat["xgt_data_first_byte"] = float(first_b)
    feat["xgt_data_last_byte"] = float(last_b)
    feat["xgt_data_mean_byte"] = float(mean_b)
    feat["xgt_data_bucket"] = float(bucket)

    if xgt_slot_config:
        slot_names = xgt_slot_config["slot_names"]
        stats_cfg = xgt_slot_config["stats"]
        feat_names = xgt_slot_config["feat_names"]

        addr_list = to_str_list(pkt.get("xgt_fen.translated_addr"))
        val_list = to_float_list(pkt.get("xgt_fen.word_value"))

        value_map: Dict[str, float] = {}
        for a, v in zip(addr_list, val_list):
            if a not in value_map:
                value_map[a] = float(v)

        for slot_name in slot_names:
            feat_name = feat_names.get(slot_name)
            if not feat_name:
                continue
            raw_v = value_map.get(slot_name)
            if raw_v is None:
                feat[feat_name] = float(SLOT_MISSING_SENTINEL)
                continue
            stat = stats_cfg.get(slot_name, {}) if isinstance(stats_cfg, dict) else {}
            feat[feat_name] = float(
                minmax_norm_with_sentinel(raw_v, stat.get("min"), stat.get("max"), sentinel=SLOT_OOR_SENTINEL)
            )

    return feat


def build_arp_features(pkt: JsonDict, arp_host_map: JsonDict) -> Dict[str, float]:
    smac = pkt.get("smac")
    sip = pkt.get("sip")
    tmac = pkt.get("arp.tmac")
    tip = pkt.get("arp.tip")
    op = safe_int(pkt.get("arp.op"), 0)

    src_id = host_id_lookup(arp_host_map, smac, sip)
    tgt_id = host_id_lookup(arp_host_map, tmac, tip)

    return {"arp_src_host_id": float(src_id), "arp_tgt_host_id": float(tgt_id), "arp_op_num": float(op)}


def build_dns_features(pkt: JsonDict, dns_norm: JsonDict) -> Dict[str, float]:
    qc = safe_int(pkt.get("dns.qc"), 0)
    ac = safe_int(pkt.get("dns.ac"), 0)

    qc_norm = minmax_norm(float(qc), dns_norm.get("dns_qc_min"), dns_norm.get("dns_qc_max"))
    ac_norm = minmax_norm(float(ac), dns_norm.get("dns_ac_min"), dns_norm.get("dns_ac_max"))

    return {"dns_qc_norm": float(qc_norm), "dns_ac_norm": float(ac_norm)}


def build_slot_config(slot_vocab: JsonDict, slot_norm: JsonDict, prefix: str, feature_columns: List[str]) -> Optional[JsonDict]:
    if not isinstance(slot_vocab, dict) or not slot_vocab:
        return None

    slot_names = sorted(slot_vocab.keys(), key=lambda k: slot_vocab[k])
    stats = slot_norm if isinstance(slot_norm, dict) else {}
    feat_names: Dict[str, str] = {}

    for addr in slot_names:
        safe = sanitize_slot_name(addr)
        col = f"{prefix}_slot_{safe}_norm"
        feature_columns.append(col)
        feat_names[str(addr)] = col

    return {"slot_names": slot_names, "stats": stats, "feat_names": feat_names}


def extract_one_packet(
    pkt: JsonDict,
    feature_columns: List[str],
    base_feat_template: Dict[str, float],
    common_host_map: JsonDict,
    common_norm: JsonDict,
    s7_norm: JsonDict,
    modbus_norm: JsonDict,
    xgt_var_vocab: JsonDict,
    xgt_norm: JsonDict,
    arp_host_map: JsonDict,
    dns_norm: JsonDict,
    modbus_slot_config: Optional[JsonDict],
    xgt_slot_config: Optional[JsonDict],
) -> JsonDict:
    protocol_str = str(pkt.get("protocol", "") or "")
    protocol_code = protocol_to_code(protocol_str)
    protocol_norm = minmax_norm(float(protocol_code), PROTOCOL_MIN, PROTOCOL_MAX, missing=0.0)

    pkt_feat = dict(base_feat_template)
    pkt_feat["protocol_norm"] = float(protocol_norm)

    pkt_feat.update(build_common_features(pkt, common_host_map, common_norm))

    if protocol_str == "s7comm":
        pkt_feat.update(build_s7comm_features(pkt, s7_norm))
    elif protocol_str == "modbus":
        pkt_feat.update(build_modbus_features(pkt, modbus_norm, modbus_slot_config))
    elif protocol_str == "xgt_fen":
        pkt_feat.update(build_xgt_fen_features(pkt, xgt_var_vocab, xgt_norm, xgt_slot_config))
    elif protocol_str == "arp":
        pkt_feat.update(build_arp_features(pkt, arp_host_map))
    elif protocol_str == "dns":
        pkt_feat.update(build_dns_features(pkt, dns_norm))

    out_pkt: JsonDict = {
        "protocol": float(protocol_code),
        "delta_t": float(safe_float(pkt.get("delta_t"), MISSING_DEFAULT)),
        "match": float(match_to_float(pkt.get("match"))),
    }
    for k in feature_columns:
        out_pkt[k] = float(pkt_feat.get(k, MISSING_DEFAULT))
    return out_pkt


# =============================================================================
# CLI / Main
# =============================================================================
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="preprocessing.yaml")
    p.add_argument("--packet", default=None, help="단일 패킷 JSON 문자열(있으면 YAML input.packet_json보다 우선)")
    p.add_argument("-i", "--input", default=None, help="패킷 JSONL(1줄=1패킷). 우선순위: CLI > YAML")
    p.add_argument("-o", "--output", default=None, help="출력 JSONL. 우선순위: CLI > YAML")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        print(f"[ERROR] config not found: {cfg_path}", file=sys.stderr)
        return 1

    cfg = _load_yaml(cfg_path)
    base_dir = cfg_path.parent.resolve()

    paths = _get(cfg, "paths", {})
    options = _get(cfg, "options", {})
    input_cfg = _get(cfg, "input", {})

    pre_dir = _resolve_path(base_dir, paths.get("pre_dir"))
    if pre_dir is None or not pre_dir.exists():
        print(f"[ERROR] pre_dir not found: {pre_dir}", file=sys.stderr)
        return 1

    yaml_input = _resolve_path(base_dir, paths.get("input_jsonl"))
    yaml_output = _resolve_path(base_dir, paths.get("output_jsonl"))

    input_path = _resolve_path(base_dir, args.input) if args.input else yaml_input
    output_path = _resolve_path(base_dir, args.output) if args.output else yaml_output

    encoding = str(_get(input_cfg, "encoding", "utf-8-sig"))
    write_weights = bool(_get(options, "write_feature_weights", False))
    default_weight = float(_get(options, "default_weight", 1.0))

    # 단일 패킷 우선순위: CLI --packet > YAML input.packet_json
    packet_json = args.packet if args.packet else _get(input_cfg, "packet_json", None)

    # --- load preprocess artifacts
    common_host_map = load_json_or_default(pre_dir / "common_host_map.json")
    common_norm = load_json_or_default(pre_dir / "common_norm_params.json")
    s7_norm = load_json_or_default(pre_dir / "s7comm_norm_params.json")
    modbus_norm = load_json_or_default(pre_dir / "modbus_norm_params.json")
    xgt_var_vocab = load_json_or_default(pre_dir / "xgt_var_vocab.json")
    xgt_norm = load_json_or_default(pre_dir / "xgt_norm_params.json")
    arp_host_map = load_json_or_default(pre_dir / "arp_host_map.json")
    dns_norm = load_json_or_default(pre_dir / "dns_norm_params.json")

    modbus_slot_vocab = load_json_or_default(pre_dir / "modbus_addr_slot_vocab.json")
    modbus_slot_norm = load_json_or_default(pre_dir / "modbus_addr_slot_norm_params.json")
    xgt_slot_vocab = load_json_or_default(pre_dir / "xgt_addr_slot_vocab.json")
    xgt_slot_norm = load_json_or_default(pre_dir / "xgt_addr_slot_norm_params.json")

    feature_columns = list(BASE_FEATURE_COLUMNS)
    modbus_slot_config = build_slot_config(modbus_slot_vocab, modbus_slot_norm, "modbus", feature_columns)
    xgt_slot_config = build_slot_config(xgt_slot_vocab, xgt_slot_norm, "xgt", feature_columns)

    base_feat_template = {k: float(MISSING_DEFAULT) for k in feature_columns}

    # --- output stream
    fout = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fout = output_path.open("w", encoding=encoding)

        if write_weights:
            wpath = write_feature_weights_txt(output_path, feature_columns, weight=default_weight)
            print(f"[INFO] feature_weights saved: {wpath}")
    else:
        fout = sys.stdout
        if write_weights:
            print("[WARN] output_jsonl이 없어서 feature_weights.txt는 생성하지 않습니다.", file=sys.stderr)

    def _emit(obj: JsonDict):
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        if fout is not sys.stdout:
            fout.flush()

    def _handle(pkt_obj: JsonDict):
        feat_pkt = extract_one_packet(
            pkt=pkt_obj,
            feature_columns=feature_columns,
            base_feat_template=base_feat_template,
            common_host_map=common_host_map,
            common_norm=common_norm,
            s7_norm=s7_norm,
            modbus_norm=modbus_norm,
            xgt_var_vocab=xgt_var_vocab,
            xgt_norm=xgt_norm,
            arp_host_map=arp_host_map,
            dns_norm=dns_norm,
            modbus_slot_config=modbus_slot_config,
            xgt_slot_config=xgt_slot_config,
        )
        _emit(feat_pkt)

    # --- run
    if packet_json:
        pkt = parse_packet_obj(packet_json)
        if not pkt:
            print("[ERROR] invalid packet_json (--packet or YAML input.packet_json)", file=sys.stderr)
            return 1
        _handle(pkt)
    else:
        fin = None
        if input_path:
            fin = input_path.open("r", encoding=encoding)
        else:
            fin = sys.stdin

        with fin:
            for line in fin:
                s = line.strip()
                if not s:
                    continue
                pkt = parse_packet_obj(s)
                if not pkt:
                    continue
                _handle(pkt)

    if fout is not None and fout is not sys.stdout:
        fout.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
