#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2.extract_feature.py
pattern windows JSONL -> sequence_group 기반 feature만 추출하여 JSONL 저장

"""
import argparse
import json
import re
import sys
from pathlib import Path


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


def protocol_to_code(p):
    if not p:
        return 0
    return PROTOCOL_MAP.get(p, 0)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8-sig"))


def try_load_json(path: Path, warn: str):
    try:
        return load_json(path)
    except FileNotFoundError:
        print(warn, file=sys.stderr)
        return None


def safe_int(val, default=0):
    try:
        if isinstance(val, list) and val:
            val = val[0]
        s = str(val).strip()
        if not s:
            return default
        return int(s, 0)
    except Exception:
        return default


def safe_float(val, default=MISSING_DEFAULT):
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


def match_to_float(v):
    if v is None:
        return MISSING_DEFAULT
    if v in (1, True, "1", "true", "True", "O", "o", "OK", "ok"):
        return 1.0
    if v in (0, False, "0", "false", "False", "X", "x"):
        return 0.0
    return MISSING_DEFAULT


def parse_packet_obj(pkt):
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


def minmax_norm(x, vmin, vmax, oor=SLOT_OOR_SENTINEL, missing=0.0):
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


def minmax_norm_with_sentinel(x, vmin, vmax, sentinel=SLOT_OOR_SENTINEL):
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


def sanitize_slot_name(name):
    s = str(name)
    s = s.replace("%", "").replace(" ", "")
    s = re.sub(r"[^0-9A-Za-z_]", "_", s)
    return s


def to_str_list(val):
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


def to_float_list(val):
    if isinstance(val, list):
        out = []
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
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            continue
    return out


def host_id_lookup(host_map, mac, ip):
    if not mac or not ip:
        return 0
    key = f"{str(mac).strip()}|{str(ip).strip()}"
    try:
        return int(host_map.get(key, 0))
    except Exception:
        return 0


def var_id_lookup(var_vocab, var_raw):
    if not var_raw:
        return 0
    s = str(var_raw)
    try:
        return int(var_vocab.get(s, 0))
    except Exception:
        return 0


def get_xgt_minmax(norm_params, key):
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


def bucket_by_mean(mean_byte):
    if mean_byte <= 64:
        return 0
    elif mean_byte <= 128:
        return 1
    elif mean_byte <= 192:
        return 2
    else:
        return 3


def build_common_features(pkt, common_host_map, common_norm):
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


def build_s7comm_features(pkt, s7_norm):
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


def parse_int_list(val):
    if isinstance(val, list):
        out = []
        for v in val:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out
    return []


def compute_regs_addr_stats(addrs):
    if not addrs:
        return 0, 0.0, 0.0, 0.0
    c = len(addrs)
    amin = float(min(addrs))
    amax = float(max(addrs))
    return c, amin, amax, amax - amin


def compute_regs_val_stats(vals):
    if not vals:
        return 0.0, 0.0, 0.0, 0.0
    vmin = float(min(vals))
    vmax = float(max(vals))
    mean = float(sum(vals)) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = var ** 0.5
    return vmin, vmax, mean, std


def build_modbus_features(pkt, modbus_norm, modbus_slot_config):
    addr = safe_int(pkt.get("modbus.addr"), 0)
    fc = safe_int(pkt.get("modbus.fc"), 0)
    qty = safe_int(pkt.get("modbus.qty"), 0)
    bc = safe_int(pkt.get("modbus.bc"), 0)

    addr_cfg = modbus_norm.get("modbus.addr", {})
    fc_cfg = modbus_norm.get("modbus.fc", {})
    qty_cfg = modbus_norm.get("modbus.qty", {})
    bc_cfg = modbus_norm.get("modbus.bc", {})

    feat = {
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

        value_map = {}
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


def build_xgt_fen_features(pkt, xgt_var_vocab, xgt_norm, xgt_slot_config):
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
    bytes_list = []
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

    feat = {}
    feat["xgt_var_id"] = float(var_id)

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
        if vmin is None or vmax is None:
            feat[key] = 0.0
        else:
            feat[key] = float(minmax_norm(raw_val, vmin, vmax))

    feat["xgt_err_flag"] = 1.0 if (errstat != 0 or errinfo != 0) else 0.0
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

        value_map = {}
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


def build_arp_features(pkt, arp_host_map):
    smac = pkt.get("smac")
    sip = pkt.get("sip")
    tmac = pkt.get("arp.tmac")
    tip = pkt.get("arp.tip")
    op = safe_int(pkt.get("arp.op"), 0)

    src_id = host_id_lookup(arp_host_map, smac, sip)
    tgt_id = host_id_lookup(arp_host_map, tmac, tip)

    return {
        "arp_src_host_id": float(src_id),
        "arp_tgt_host_id": float(tgt_id),
        "arp_op_num": float(op),
    }


def build_dns_features(pkt, dns_norm):
    qc = safe_int(pkt.get("dns.qc"), 0)
    ac = safe_int(pkt.get("dns.ac"), 0)

    qc_norm = minmax_norm(float(qc), dns_norm.get("dns_qc_min"), dns_norm.get("dns_qc_max"))
    ac_norm = minmax_norm(float(ac), dns_norm.get("dns_ac_min"), dns_norm.get("dns_ac_max"))

    return {
        "dns_qc_norm": float(qc_norm),
        "dns_ac_norm": float(ac_norm),
    }


def determine_global_window_size(input_path: Path):
    gws = 0
    with input_path.open("r", encoding="utf-8-sig") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                win_obj = json.loads(line)
            except Exception:
                continue
            idx = win_obj.get("index")
            if isinstance(idx, list):
                gws = max(gws, len(idx))
            seq = win_obj.get("sequence_group")
            if isinstance(seq, list):
                gws = max(gws, len(seq))
    return gws if gws > 0 else 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-p", "--pre_dir", required=True)
    p.add_argument("-o", "--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    pre_dir = Path(args.pre_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    common_host_map = load_json(pre_dir / "common_host_map.json")
    common_norm = load_json(pre_dir / "common_norm_params.json")

    s7_norm = load_json(pre_dir / "s7comm_norm_params.json")
    modbus_norm = load_json(pre_dir / "modbus_norm_params.json")

    xgt_var_vocab = load_json(pre_dir / "xgt_var_vocab.json")

    xgt_norm = try_load_json(pre_dir / "xgt_norm_params.json", "[WARN] xgt_norm_params.json 없음")
    if xgt_norm is None:
        xgt_norm = load_json(pre_dir / "xgt_norm_params.json")

    arp_host_map = load_json(pre_dir / "arp_host_map.json")
    dns_norm = load_json(pre_dir / "dns_norm_params.json")

    modbus_slot_vocab = try_load_json(
        pre_dir / "modbus_addr_slot_vocab.json",
        "[WARN] modbus_addr_slot_vocab.json 없음",
    )
    modbus_slot_norm = try_load_json(
        pre_dir / "modbus_addr_slot_norm_params.json",
        "[WARN] modbus_addr_slot_norm_params.json 없음",
    )

    xgt_slot_vocab = try_load_json(
        pre_dir / "xgt_addr_slot_vocab.json",
        "[WARN] xgt_addr_slot_vocab.json 없음",
    )
    xgt_slot_norm = try_load_json(
        pre_dir / "xgt_addr_slot_norm_params.json",
        "[WARN] xgt_addr_slot_norm_params.json 없음",
    )

    feature_columns = list(BASE_FEATURE_COLUMNS)

    modbus_slot_config = None
    if isinstance(modbus_slot_vocab, dict) and modbus_slot_vocab:
        slot_names = sorted(modbus_slot_vocab.keys(), key=lambda k: modbus_slot_vocab[k])
        stats = modbus_slot_norm if isinstance(modbus_slot_norm, dict) else {}
        feat_names = {}
        for addr in slot_names:
            safe = sanitize_slot_name(addr)
            col = f"modbus_slot_{safe}_norm"
            feature_columns.append(col)
            feat_names[addr] = col
        modbus_slot_config = {"slot_names": slot_names, "stats": stats, "feat_names": feat_names}

    xgt_slot_config = None
    if isinstance(xgt_slot_vocab, dict) and xgt_slot_vocab:
        slot_names = sorted(xgt_slot_vocab.keys(), key=lambda k: xgt_slot_vocab[k])
        stats = xgt_slot_norm if isinstance(xgt_slot_norm, dict) else {}
        feat_names = {}
        for addr in slot_names:
            safe = sanitize_slot_name(addr)
            col = f"xgt_slot_{safe}_norm"
            feature_columns.append(col)
            feat_names[addr] = col
        xgt_slot_config = {"slot_names": slot_names, "stats": stats, "feat_names": feat_names}

    gws = determine_global_window_size(input_path)
    print(f"[INFO] global_window_size: {gws}")
    print(f"[INFO] feature_dim: {len(feature_columns)}")

    base_feat_template = {k: float(MISSING_DEFAULT) for k in feature_columns}

    line_cnt_raw = 0
    win_cnt = 0
    skipped_leq1 = 0
    total_row_cnt = 0

    with input_path.open("r", encoding="utf-8-sig") as fin, out_path.open("w", encoding="utf-8-sig") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                win_obj = json.loads(line)
            except Exception:
                continue

            line_cnt_raw += 1

            window_id = win_obj.get("window_id")
            pattern = win_obj.get("pattern") or win_obj.get("label")
            description = win_obj.get("description")

            index_list = win_obj.get("index")
            index_list = index_list if isinstance(index_list, list) else []

            if len(index_list) <= 1:
                skipped_leq1 += 1
                continue

            raw_seq = win_obj.get("sequence_group")
            if not isinstance(raw_seq, list):
                raw_seq = win_obj.get("window_group") or win_obj.get("RAW") or []
            raw_seq = raw_seq if isinstance(raw_seq, list) else []

            seq_group = []
            for x in raw_seq:
                p = parse_packet_obj(x)
                if isinstance(p, dict):
                    seq_group.append(p)
                else:
                    seq_group.append({})

            seq_feature_group = []

            for pkt in seq_group:
                protocol_str = pkt.get("protocol", "") if isinstance(pkt, dict) else ""
                protocol_code = protocol_to_code(protocol_str)
                protocol_norm = minmax_norm(float(protocol_code), PROTOCOL_MIN, PROTOCOL_MAX, missing=0.0)

                pkt_feat = dict(base_feat_template)
                pkt_feat["protocol_norm"] = float(protocol_norm)

                if isinstance(pkt, dict):
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

                out_pkt = {
                    "protocol": float(protocol_code),
                    "delta_t": float(safe_float(pkt.get("delta_t"), MISSING_DEFAULT)) if isinstance(pkt, dict) else float(MISSING_DEFAULT),
                    "match": float(match_to_float(pkt.get("match"))) if isinstance(pkt, dict) else float(MISSING_DEFAULT),
                }
                for k in feature_columns:
                    out_pkt[k] = float(pkt_feat.get(k, MISSING_DEFAULT))

                seq_feature_group.append(out_pkt)
                total_row_cnt += 1

            out_obj = {
                "window_id": window_id,
                "pattern": pattern,
                "description": description,
                "index": index_list,
                "sequence_group": seq_feature_group,
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            win_cnt += 1

            if win_cnt % 1000 == 0:
                fout.flush()
                print(f"[INFO] {win_cnt} windows processed...", flush=True)

    print(f"[INFO] total_lines(valid json): {line_cnt_raw}")
    print(f"[INFO] windows_written: {win_cnt}")
    print(f"[INFO] skipped_len_index_leq1: {skipped_leq1}")
    print(f"[INFO] total_rows(packets): {total_row_cnt}")

if __name__ == "__main__":
    main()
