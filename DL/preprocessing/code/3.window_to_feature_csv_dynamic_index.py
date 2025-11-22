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
   - 각 윈도우 내에서
       base_idx = min(index_list)
       각 패킷을 pos = (orig_index - base_idx) 위치에 매핑
       pos 가 [0, global_window_size-1] 범위 안일 때만 사용
       비어 있는 pos 는 0-padding row로 채움

2) --max-index 를 주지 않으면:
   - global_window_size =
       • 우선 index 리스트 길이의 최댓값
       • 만약 index 가 없으면 sequence_group 길이의 최댓값
       • 그래도 없으면 1

출력:
  - JSONL: 원본이 아니라, 각 window에 대해
        {
          "window_id": ...,
          "pattern": ...,
          "index": [... 원본 index ...],
          "base_idx": ...,
          "span": ...,
          "window_size": T,
          "sequence_group": [
             {
               "pos": 0,
               "orig_index": <원본 index 또는 null>,
               "has_real_pkt": 0/1,
               "protocol": <code>,
               "delta_t": <float>,
               <FEATURE_COLUMNS ...>
             },
             ...
          ]
        }
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def minmax_norm(x: float, vmin: float, vmax: float) -> float:
    if vmax is None or vmin is None or vmax <= vmin:
        return 0.0
    return (x - vmin) / (vmax - vmin + 1e-9)


def safe_int(val: Any, default: int = 0) -> int:
    try:
        if isinstance(val, list) and val:
            val = val[0]
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

    feat["xgt_var_id"] = float(var_id)
    feat["xgt_var_cnt"] = float(var_cnt)
    feat["xgt_source"] = float(source)
    feat["xgt_fenet_base"] = float(xgt_fenet_base)
    feat["xgt_fenet_slot"] = float(xgt_fenet_slot)
    feat["xgt_cmd"] = float(cmd)
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

FEATURE_COLUMNS = [
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

COLUMNS = META_COLUMNS + FEATURE_COLUMNS


def main():
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
        "-o",
        "--output",
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
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.json_output:
        jsonl_path = Path(args.json_output)
    else:
        # 예전처럼 .csv 주면 같은 이름의 .jsonl로 저장
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

    # ----- global_window_size 결정 (span 기반 X) -----
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
    print(f"📏 실제 사용 global_window_size (== window_size): {global_window_size}")

    # ----- JSONL 작성 -----
    # ----- JSONL 작성 -----
    with jsonl_path.open("w", encoding="utf-8") as fout_jsonl:

        win_cnt = 0
        skipped_by_span = 0
        total_row_cnt = 0  # 윈도우 * window_size (정보용)

        for win_obj in windows:
            window_id = win_obj.get("window_id")
            pattern = win_obj.get("pattern")
            seq_group = win_obj.get("sequence_group", [])
            index_list = win_obj.get("index", [])

            if not isinstance(seq_group, list):
                seq_group = []
            if not isinstance(index_list, list):
                index_list = []

            # ----- span 계산 및 필터링 -----
            span = None
            idx_min = None
            idx_max = None
            if index_list:
                try:
                    idx_min = int(min(index_list))
                    idx_max = int(max(index_list))
                    span = idx_max - idx_min
                except Exception:
                    span = None

            if args.max_index is not None and span is not None:
                # span >= max_index → 제거 (JSONL에도 안 나감)
                if span >= args.max_index:
                    skipped_by_span += 1
                    continue

            # base_idx = min(index_list) (비어있으면 0)
            if index_list:
                try:
                    base_idx = int(min(index_list))
                except Exception:
                    base_idx = 0
            else:
                base_idx = 0

            # 👉 상대 인덱스 리스트(0부터 시작) 생성
            rel_index_list: List[int] = []
            for idx in index_list:
                try:
                    idx_int = int(idx)
                except Exception:
                    # 변환 안 되면 0 기준으로만 넣어도 됨 (혹은 건너뛰기)
                    continue
                rel_index_list.append(idx_int - base_idx)

            # pos -> (pkt, orig_index) 매핑
            pos_to_info: Dict[int, Tuple[Dict[str, Any], int]] = {}
            for pkt, orig_idx in zip(seq_group, index_list):
                try:
                    orig_idx_int = int(orig_idx)
                except Exception:
                    continue
                pos = orig_idx_int - base_idx
                if 0 <= pos < global_window_size and pos not in pos_to_info:
                    pos_to_info[pos] = (pkt, orig_idx_int)

            # 이 윈도우의 feature 시퀀스 (JSONL용)
            seq_feature_group: List[Dict[str, Any]] = []

            # 0 ~ global_window_size-1 까지 dense하게 채우기 (중간은 0-padding)
            for pos in range(global_window_size):
                if pos in pos_to_info:
                    pkt, orig_idx_int = pos_to_info[pos]
                    has_real_pkt = 1.0
                else:
                    pkt = {}
                    orig_idx_int = -1
                    has_real_pkt = 0.0

                protocol_str = pkt.get("protocol", "") if has_real_pkt else ""
                protocol_code = protocol_to_code(protocol_str)
                delta_t = safe_float(pkt.get("delta_t", 0.0)) if has_real_pkt else 0.0

                # feature용 row 딕셔너리 (CSV 안 쓰지만 동일 구조 활용)
                row: Dict[str, Any] = {col: 0.0 for col in COLUMNS}
                row["window_id"] = window_id
                row["pattern"] = pattern
                row["protocol"] = float(protocol_code)
                row["delta_t"] = float(delta_t)

                if has_real_pkt:
                    common_feat = build_common_features(
                        pkt, common_host_map, common_norm_params
                    )
                    row.update(common_feat)

                    if protocol_str == "s7comm":
                        s7_feat = build_s7comm_features(pkt, s7comm_norm_params)
                        row.update(s7_feat)
                    elif protocol_str == "modbus":
                        mb_feat = build_modbus_features(pkt, modbus_norm_params)
                        row.update(mb_feat)
                    elif protocol_str == "xgt_fen":
                        xgt_feat = build_xgt_fen_features(
                            pkt, xgt_var_vocab, xgt_fen_norm_params
                        )
                        row.update(xgt_feat)
                    elif protocol_str == "arp":
                        arp_feat = build_arp_features(pkt, arp_host_map)
                        row.update(arp_feat)
                    elif protocol_str == "dns":
                        dns_feat = build_dns_features(pkt, dns_norm_params)
                        row.update(dns_feat)

                total_row_cnt += 1

                # JSONL 용 feature만 추출 (pos, orig_index, has_real_pkt 제거)
                pkt_feat: Dict[str, Any] = {
                    "protocol": float(protocol_code),
                    "delta_t": float(delta_t),
                }
                for k in FEATURE_COLUMNS:
                    pkt_feat[k] = row[k]
                seq_feature_group.append(pkt_feat)

            # JSONL 출력 (원본 패킷 X, feature 시퀀스만)
            out_obj = {
                "window_id": window_id,
                "pattern": pattern,
                "orig_index": index_list,      # 원본 index 그대로
                "index": rel_index_list,       # ✅ base_idx 기준 0부터 시작하는 index
                "base_idx": base_idx,
                "span": span,
                "window_size": global_window_size,
                "sequence_group": seq_feature_group,
            }
            fout_jsonl.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            win_cnt += 1

    print(f"✅ 완료: 원본 {line_cnt_raw}개 라인 / {win_cnt}개 윈도우 처리")
    if args.max_index is not None:
        print(f"   ↳ span >= {args.max_index} 조건으로 스킵된 윈도우 수: {skipped_by_span}")
    print(f"→ window 당 길이(global_window_size): {global_window_size}")
    print(f"→ 총 row 수(윈도우 * window_size): {total_row_cnt}")
    print(f"→ JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()

"""
python 3.window_to_feature_csv_dynamic_index.py --input "../data/pattern_windows.jsonl"  --pre_dir "../result" --output "../result/pattern_features.csv"

python 3.window_to_feature_csv_dynamic_index.py --input "../data/pattern_windows.jsonl" --pre_dir "../result" --output "../result/pattern_features.csv" --max-index 8

"""
