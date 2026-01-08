#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
packet_feature_preprocessor.py

- 전처리 파라미터(pre_dir)를 1번 로드하고,
- 패킷 1개({origin:{...}} 또는 origin dict)를 넣으면
  모델 입력용 feature dict 1개를 반환한다.

반환 dict:
  {
    "protocol": <code>,
    "delta_t": <float>,
    "match": <0/1>,
    "protocol_norm": ...,
    ... (BASE_FEATURE_COLUMNS + 동적 슬롯 컬럼 전부)
    (+ 옵션) "index": ...,
  }
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ==========================
# constants
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

SLOT_MISSING_SENTINEL = -1.0
SLOT_OOR_SENTINEL = -2.0

BASE_FEATURE_COLUMNS = [
    "protocol_norm",
    # common
    "src_host_id", "dst_host_id", "sp_norm", "dp_norm", "dir_code", "len_norm",
    # s7comm
    "s7comm_ros_norm", "s7comm_fn", "s7comm_db_norm", "s7comm_addr_norm",
    # modbus
    "modbus_addr_norm", "modbus_fc_norm", "modbus_qty_norm", "modbus_bc_norm",
    "modbus_regs_count", "modbus_regs_addr_min", "modbus_regs_addr_max", "modbus_regs_addr_range",
    "modbus_regs_val_min", "modbus_regs_val_max", "modbus_regs_val_mean", "modbus_regs_val_std",
    # xgt_fen
    "xgt_var_id", "xgt_var_cnt", "xgt_source", "xgt_fenet_base", "xgt_fenet_slot",
    "xgt_cmd", "xgt_dtype", "xgt_blkcnt", "xgt_err_flag", "xgt_err_code", "xgt_datasize",
    "xgt_data_missing", "xgt_data_len_chars", "xgt_data_num_spaces", "xgt_data_is_hex",
    "xgt_data_n_bytes", "xgt_data_zero_ratio", "xgt_data_first_byte", "xgt_data_last_byte",
    "xgt_data_mean_byte", "xgt_data_bucket",
    # arp
    "arp_src_host_id", "arp_tgt_host_id", "arp_op_num",
    # dns
    "dns_qc_norm", "dns_ac_norm",
]

XGT_NORM_FIELDS = [
    "xgt_var_cnt", "xgt_source", "xgt_fenet_base", "xgt_fenet_slot", "xgt_cmd", "xgt_dtype",
    "xgt_blkcnt", "xgt_err_code", "xgt_datasize", "xgt_data_len_chars",
    "xgt_data_num_spaces", "xgt_data_n_bytes",
]


# ==========================
# utils
# ==========================
def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_slot_name(name: str) -> str:
    s = str(name).replace("%", "").replace(" ", "")
    return re.sub(r"[^0-9A-Za-z_]", "_", s)


def protocol_to_code(p: Any) -> int:
    if not p:
        return 0
    s = str(p)
    if s == "xgt-fen":
        s = "xgt_fen"
    return int(PROTOCOL_MAP.get(s, 0))


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


def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if isinstance(val, list) and val:
            val = val[0]
        return float(val)
    except Exception:
        return default


def minmax_norm(x: float, vmin: Any, vmax: Any) -> float:
    if vmin is None or vmax is None:
        return 0.0
    vmin, vmax = float(vmin), float(vmax)
    x = float(x)
    if x < vmin or x > vmax:
        return SLOT_OOR_SENTINEL
    if vmax == vmin:
        return 0.0 if x <= vmin else SLOT_OOR_SENTINEL
    return (x - vmin) / (vmax - vmin + 1e-9)


def minmax_norm_with_sentinel(x: float, vmin: Any, vmax: Any, sentinel: float = -2.0) -> float:
    if vmin is None or vmax is None:
        return 0.0
    vmin, vmax = float(vmin), float(vmax)
    x = float(x)
    if vmax == vmin:
        return 0.0 if x == vmin else float(sentinel)
    if x < vmin or x > vmax:
        return float(sentinel)
    val = (x - vmin) / (vmax - vmin + 1e-9)
    return max(0.0, min(1.0, val))


def parse_ts_ms(origin: Dict[str, Any]) -> Optional[int]:
    rts = origin.get("redis_timestamp_ms")
    if rts is not None:
        try:
            return int(rts)
        except Exception:
            pass

    ts = origin.get("@timestamp")
    if isinstance(ts, str) and ts.strip():
        s = ts.strip()
        try:
            if s.endswith("Z"):
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(s)
            return int(dt.timestamp() * 1000)
        except Exception:
            return None
    return None


def parse_redis_id(redis_id: Any) -> Optional[Tuple[int, int]]:
    try:
        s = str(redis_id)
        ts_ms, seq = s.split("-")
        return int(ts_ms), int(seq)
    except Exception:
        return None


def extract_match(pkt: Dict[str, Any]) -> int:
    if "match" in pkt:
        try:
            return int(pkt["match"])
        except Exception:
            pass
    ml = pkt.get("ML")
    if isinstance(ml, dict):
        m = ml.get("match")
        if m in ("O", "o", 1, "1", True):
            return 1
        if m in ("X", "x", 0, "0", False):
            return 0
    return 1


def _parse_int_list(val: Any) -> List[int]:
    if not isinstance(val, list):
        return []
    out: List[int] = []
    for v in val:
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def _to_str_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(v).strip() for v in val if str(v).strip()]
    if val is None:
        return []
    s = str(val).strip()
    if not s:
        return []
    s = s.replace(";", ",").replace(" ", ",")
    return [p.strip() for p in s.split(",") if p.strip()]


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
    return vmin, vmax, mean, var ** 0.5


# ==========================
# ID assigner (freeze by default)
# ==========================
class IdAssigner:
    def __init__(self, mapping: Dict[str, int], *, allow_new: bool):
        self.mapping = mapping or {}
        self.allow_new = bool(allow_new)
        self._next_id = (max(self.mapping.values()) + 1) if self.mapping else 1

    def get(self, key: str) -> int:
        if not key:
            return 0
        if key in self.mapping:
            return int(self.mapping[key])
        if not self.allow_new:
            return 0
        nid = self._next_id
        self.mapping[key] = nid
        self._next_id += 1
        return nid


@dataclass
class SlotConfig:
    slot_names: List[str]
    stats: Dict[str, Any]
    feat_names: Dict[str, str]  # raw slot -> column


# ==========================
# main preprocessor (packet 1 -> feature 1)
# ==========================
@dataclass
class PacketStreamState:
    prev_ts_ms: Optional[int] = None
    seq_counter: int = 0


class PacketFeaturePreprocessor:
    """
    - pre_dir 로 파라미터 로드(1회)
    - preprocess()에 패킷 1개 넣으면 feature dict 1개 반환
    """

    def __init__(
        self,
        pre_dir: Path,
        *,
        allow_new_ids: bool = False,
        index_source: str = "seq",        # "seq"|"redis_id"|"redis_ts"|"ts"
        include_index: bool = False,
    ):
        self.pre_dir = Path(pre_dir)
        self.allow_new_ids = bool(allow_new_ids)
        self.index_source = str(index_source)
        self.include_index = bool(include_index)

        # params
        self.common_norm = load_json(self.pre_dir / "common_norm_params.json", {}) or {}
        self.s7_norm = load_json(self.pre_dir / "s7comm_norm_params.json", {}) or {}
        self.mb_norm = load_json(self.pre_dir / "modbus_norm_params.json", {}) or {}
        self.xgt_norm = load_json(self.pre_dir / "xgt_fen_norm_params.json", {}) or {}
        self.dns_norm = load_json(self.pre_dir / "dns_norm_params.json", {}) or {}

        # id maps
        self.common_host_id = IdAssigner(load_json(self.pre_dir / "common_host_map.json", {}) or {}, allow_new=self.allow_new_ids)
        self.arp_host_id = IdAssigner(load_json(self.pre_dir / "arp_host_map.json", {}) or {}, allow_new=self.allow_new_ids)
        self.xgt_var_id = IdAssigner(load_json(self.pre_dir / "xgt_var_vocab.json", {}) or {}, allow_new=self.allow_new_ids)

        # dynamic slots
        self.feature_columns: List[str] = list(BASE_FEATURE_COLUMNS)
        self.modbus_slot: Optional[SlotConfig] = None
        self.xgt_slot: Optional[SlotConfig] = None
        self._load_slot_meta()

        # stream state
        self.state = PacketStreamState()

    def _load_slot_meta(self) -> None:
        # modbus
        mb_vocab = load_json(self.pre_dir / "modbus_addr_slot_vocab.json", None)
        mb_stats = load_json(self.pre_dir / "modbus_addr_slot_norm_params.json", {}) or {}
        if isinstance(mb_vocab, dict) and mb_vocab:
            names = sorted(mb_vocab.keys(), key=lambda k: mb_vocab[k])
            feat_names: Dict[str, str] = {}
            for addr in names:
                col = f"modbus_slot_{sanitize_slot_name(addr)}_norm"
                self.feature_columns.append(col)
                feat_names[addr] = col
            self.modbus_slot = SlotConfig(slot_names=names, stats=mb_stats, feat_names=feat_names)

        # xgt
        xgt_vocab = load_json(self.pre_dir / "xgt_addr_slot_vocab.json", None)
        xgt_stats = load_json(self.pre_dir / "xgt_addr_slot_norm_params.json", {}) or {}
        if isinstance(xgt_vocab, dict) and xgt_vocab:
            names = sorted(xgt_vocab.keys(), key=lambda k: xgt_vocab[k])
            feat_names: Dict[str, str] = {}
            for addr in names:
                col = f"xgt_slot_{sanitize_slot_name(addr)}_norm"
                self.feature_columns.append(col)
                feat_names[addr] = col
            self.xgt_slot = SlotConfig(slot_names=names, stats=xgt_stats, feat_names=feat_names)

    # ---------- index/delta_t ----------
    def _index_from_origin(self, origin: Dict[str, Any]) -> int:
        mode = self.index_source
        fallback = self.state.seq_counter
        if mode == "seq":
            return int(fallback)

        if mode == "redis_id":
            parsed = parse_redis_id(origin.get("redis_id"))
            if parsed is not None:
                ts_ms, seq = parsed
                return int(ts_ms) * 100000 + int(seq)
            return int(fallback)

        if mode == "redis_ts":
            try:
                return int(origin.get("redis_timestamp_ms"))
            except Exception:
                return int(fallback)

        if mode == "ts":
            tms = parse_ts_ms(origin)
            return int(tms) if tms is not None else int(fallback)

        return int(fallback)

    def _delta_t_stream(self, origin: Dict[str, Any]) -> float:
        # delta_t가 있으면 우선 사용 + ts 있으면 prev 갱신
        if "delta_t" in origin:
            dt = float(safe_float(origin.get("delta_t"), 0.0))
            cur_ms = parse_ts_ms(origin)
            if cur_ms is not None:
                self.state.prev_ts_ms = cur_ms
            return dt

        cur_ms = parse_ts_ms(origin)
        if cur_ms is None:
            return 0.0
        if self.state.prev_ts_ms is None:
            self.state.prev_ts_ms = cur_ms
            return 0.0
        dt = max(0.0, (cur_ms - self.state.prev_ts_ms) / 1000.0)
        self.state.prev_ts_ms = cur_ms
        return float(dt)

    # ---------- per-protocol features ----------
    def _common_features(self, pkt: Dict[str, Any]) -> Dict[str, float]:
        smac, sip = pkt.get("smac"), pkt.get("sip")
        dmac, dip = pkt.get("dmac"), pkt.get("dip")
        sp, dp = safe_int(pkt.get("sp")), safe_int(pkt.get("dp"))
        ln = safe_int(pkt.get("len"))
        dir_code = 1.0 if pkt.get("dir") == "request" else 0.0

        src_key = f"{smac}|{sip}" if smac and sip else ""
        dst_key = f"{dmac}|{dip}" if dmac and dip else ""

        sp_norm = minmax_norm(float(sp), self.common_norm.get("sp_min"), self.common_norm.get("sp_max"))
        dp_norm = minmax_norm(float(dp), self.common_norm.get("dp_min"), self.common_norm.get("dp_max"))
        ln_norm = minmax_norm(float(ln), self.common_norm.get("len_min"), self.common_norm.get("len_max"))

        return {
            "src_host_id": float(self.common_host_id.get(src_key)),
            "dst_host_id": float(self.common_host_id.get(dst_key)),
            "sp_norm": float(sp_norm),
            "dp_norm": float(dp_norm),
            "dir_code": float(dir_code),
            "len_norm": float(ln_norm),
        }

    def _s7_features(self, pkt: Dict[str, Any]) -> Dict[str, float]:
        ros = safe_int(pkt.get("s7comm.ros"))
        fn = safe_int(pkt.get("s7comm.fn"))
        db = safe_int(pkt.get("s7comm.db"))
        addr = safe_int(pkt.get("s7comm.addr"))

        ros_cfg = self.s7_norm.get("s7comm.ros", {})
        db_cfg = self.s7_norm.get("s7comm.db", {})
        ad_cfg = self.s7_norm.get("s7comm.addr", {})

        return {
            "s7comm_ros_norm": float(minmax_norm(float(ros), ros_cfg.get("min"), ros_cfg.get("max"))),
            "s7comm_fn": float(fn),
            "s7comm_db_norm": float(minmax_norm(float(db), db_cfg.get("min"), db_cfg.get("max"))),
            "s7comm_addr_norm": float(minmax_norm(float(addr), ad_cfg.get("min"), ad_cfg.get("max"))),
        }

    def _modbus_features(self, pkt: Dict[str, Any]) -> Dict[str, float]:
        addr = safe_int(pkt.get("modbus.addr"))
        fc = safe_int(pkt.get("modbus.fc"))
        qty = safe_int(pkt.get("modbus.qty"))
        bc = safe_int(pkt.get("modbus.bc"))

        addr_cfg = self.mb_norm.get("modbus.addr", {})
        fc_cfg = self.mb_norm.get("modbus.fc", {})
        qty_cfg = self.mb_norm.get("modbus.qty", {})
        bc_cfg = self.mb_norm.get("modbus.bc", {})

        # regs stats
        addr_source = pkt.get("modbus.translated_addr")
        if addr_source is None:
            addr_source = pkt.get("modbus.regs.addr")
        addrs = _parse_int_list(addr_source)
        vals = _parse_int_list(pkt.get("modbus.regs.val"))

        c, amin, amax, arange = _compute_regs_addr_stats(addrs)
        vmin, vmax, vmean, vstd = _compute_regs_val_stats(vals)

        # regs norm params
        ra_count = self.mb_norm.get("regs_addr.count", {})
        ra_min = self.mb_norm.get("regs_addr.min", {})
        ra_max = self.mb_norm.get("regs_addr.max", {})
        ra_rng = self.mb_norm.get("regs_addr.range", {})

        rv_min = self.mb_norm.get("regs_val.min", {})
        rv_max = self.mb_norm.get("regs_val.max", {})
        rv_mean = self.mb_norm.get("regs_val.mean", {})
        rv_std = self.mb_norm.get("regs_val.std", {})

        feat: Dict[str, float] = {
            "modbus_addr_norm": float(minmax_norm(float(addr), addr_cfg.get("min"), addr_cfg.get("max"))),
            "modbus_fc_norm": float(minmax_norm(float(fc), fc_cfg.get("min"), fc_cfg.get("max"))),
            "modbus_qty_norm": float(minmax_norm(float(qty), qty_cfg.get("min"), qty_cfg.get("max"))),
            "modbus_bc_norm": float(minmax_norm(float(bc), bc_cfg.get("min"), bc_cfg.get("max"))),

            "modbus_regs_count": float(minmax_norm(float(c), ra_count.get("min"), ra_count.get("max"))),
            "modbus_regs_addr_min": float(minmax_norm(float(amin), ra_min.get("min"), ra_min.get("max"))),
            "modbus_regs_addr_max": float(minmax_norm(float(amax), ra_max.get("min"), ra_max.get("max"))),
            "modbus_regs_addr_range": float(minmax_norm(float(arange), ra_rng.get("min"), ra_rng.get("max"))),

            "modbus_regs_val_min": float(minmax_norm(float(vmin), rv_min.get("min"), rv_min.get("max"))),
            "modbus_regs_val_max": float(minmax_norm(float(vmax), rv_max.get("min"), rv_max.get("max"))),
            "modbus_regs_val_mean": float(minmax_norm(float(vmean), rv_mean.get("min"), rv_mean.get("max"))),
            "modbus_regs_val_std": float(minmax_norm(float(vstd), rv_std.get("min"), rv_std.get("max"))),
        }

        # slots
        if self.modbus_slot is not None:
            cfg = self.modbus_slot
            addr_list = _to_str_list(pkt.get("modbus.translated_addr")) or _to_str_list(pkt.get("modbus.regs.addr"))
            val_list = _to_float_list(pkt.get("modbus.word_value")) or _to_float_list(pkt.get("modbus.regs.val"))

            value_map: Dict[str, float] = {}
            for a, v in zip(addr_list, val_list):
                if a not in value_map:
                    value_map[a] = v

            for slot_name in cfg.slot_names:
                col = cfg.feat_names.get(slot_name)
                if not col:
                    continue
                stat = cfg.stats.get(slot_name, {})
                raw_v = value_map.get(slot_name)
                if raw_v is None:
                    feat[col] = float(SLOT_MISSING_SENTINEL)
                else:
                    feat[col] = float(minmax_norm_with_sentinel(raw_v, stat.get("min"), stat.get("max"), sentinel=SLOT_OOR_SENTINEL))

        return feat

    def _get_xgt_minmax(self, key: str) -> Tuple[Any, Any]:
        cfg = self.xgt_norm.get(key)
        if isinstance(cfg, dict):
            return cfg.get("min"), cfg.get("max")
        if key.startswith("xgt_"):
            legacy = f"xgt_fen.{key[len('xgt_'):]}"
            cfg2 = self.xgt_norm.get(legacy)
            if isinstance(cfg2, dict):
                return cfg2.get("min"), cfg2.get("max")
        return None, None

    @staticmethod
    def _bucket_by_mean(mean_byte: float) -> int:
        if mean_byte <= 64:
            return 0
        if mean_byte <= 128:
            return 1
        if mean_byte <= 192:
            return 2
        return 3

    def _xgt_features(self, pkt: Dict[str, Any]) -> Dict[str, float]:
        source = safe_int(pkt.get("xgt_fen.source"))
        datasize = safe_int(pkt.get("xgt_fen.datasize"))
        cmd = safe_int(pkt.get("xgt_fen.cmd"))
        dtype = safe_int(pkt.get("xgt_fen.dtype"))
        blkcnt = safe_int(pkt.get("xgt_fen.blkcnt"))
        errstat = safe_int(pkt.get("xgt_fen.errstat"))
        errinfo = safe_int(pkt.get("xgt_fen.errinfo"))
        fenetpos = safe_int(pkt.get("xgt_fen.fenetpos"))

        xgt_fenet_base = fenetpos >> 4
        xgt_fenet_slot = fenetpos & 0x0F

        var_raw = pkt.get("xgt_fen.vars")
        var_str = str(var_raw) if var_raw is not None else ""
        var_id = self.xgt_var_id.get(var_str) if var_str else 0
        var_cnt = 1.0 if var_str else 0.0

        data_raw = pkt.get("xgt_fen.data")
        data_missing = 1.0 if data_raw is None else 0.0
        data_len_chars = float(len(data_raw)) if isinstance(data_raw, str) else 0.0
        num_spaces = float(data_raw.count(" ")) if isinstance(data_raw, str) else 0.0

        is_hex = 0.0
        bytes_list: List[int] = []
        if isinstance(data_raw, str):
            try:
                bs = bytes.fromhex(data_raw.replace(" ", ""))
                is_hex = 1.0
                bytes_list = list(bs)
            except Exception:
                is_hex = 0.0

        n_bytes = float(len(bytes_list))
        if bytes_list:
            first_b = float(bytes_list[0])
            last_b = float(bytes_list[-1])
            mean_b = float(sum(bytes_list)) / len(bytes_list)
            zero_ratio = float(sum(1 for b in bytes_list if b == 0)) / len(bytes_list)
            bucket = float(self._bucket_by_mean(mean_b))
        else:
            first_b = last_b = mean_b = zero_ratio = bucket = 0.0

        feat_raw: Dict[str, float] = {
            "xgt_var_id": float(var_id),
            "xgt_var_cnt": float(var_cnt),
            "xgt_source": float(source),
            "xgt_fenet_base": float(xgt_fenet_base),
            "xgt_fenet_slot": float(xgt_fenet_slot),
            "xgt_cmd": float(cmd),
            "xgt_dtype": float(dtype),
            "xgt_blkcnt": float(blkcnt),
            "xgt_err_flag": 1.0 if (errstat != 0 or errinfo != 0) else 0.0,
            "xgt_err_code": float(errinfo),
            "xgt_datasize": float(datasize),
            "xgt_data_missing": float(data_missing),
            "xgt_data_len_chars": float(data_len_chars),
            "xgt_data_num_spaces": float(num_spaces),
            "xgt_data_is_hex": float(is_hex),
            "xgt_data_n_bytes": float(n_bytes),
            "xgt_data_zero_ratio": float(zero_ratio),
            "xgt_data_first_byte": float(first_b),
            "xgt_data_last_byte": float(last_b),
            "xgt_data_mean_byte": float(mean_b),
            "xgt_data_bucket": float(bucket),
        }

        feat: Dict[str, float] = {}
        for k, v in feat_raw.items():
            if k == "xgt_cmd" or k in XGT_NORM_FIELDS:
                vmin, vmax = self._get_xgt_minmax(k)
                if vmin is None or vmax is None:
                    feat[k] = 0.0
                else:
                    if k == "xgt_cmd":
                        feat[k] = float(minmax_norm_with_sentinel(v, vmin, vmax, sentinel=SLOT_OOR_SENTINEL))
                    else:
                        feat[k] = float(minmax_norm(v, vmin, vmax))
            else:
                feat[k] = float(v)

        # slots
        if self.xgt_slot is not None:
            cfg = self.xgt_slot
            addr_list = _to_str_list(pkt.get("xgt_fen.translated_addr"))
            val_list = _to_float_list(pkt.get("xgt_fen.word_value"))

            value_map: Dict[str, float] = {}
            for a, v in zip(addr_list, val_list):
                if a not in value_map:
                    value_map[a] = v

            for slot_name in cfg.slot_names:
                col = cfg.feat_names.get(slot_name)
                if not col:
                    continue
                stat = cfg.stats.get(slot_name, {})
                raw_v = value_map.get(slot_name)
                if raw_v is None:
                    feat[col] = float(SLOT_MISSING_SENTINEL)
                else:
                    feat[col] = float(minmax_norm_with_sentinel(raw_v, stat.get("min"), stat.get("max"), sentinel=SLOT_OOR_SENTINEL))

        return feat

    def _arp_features(self, pkt: Dict[str, Any]) -> Dict[str, float]:
        smac, sip = pkt.get("smac"), pkt.get("sip")
        tmac, tip = pkt.get("arp.tmac"), pkt.get("arp.tip")
        op = safe_int(pkt.get("arp.op"))

        src_key = f"{smac}|{sip}" if smac and sip else ""
        tgt_key = f"{tmac}|{tip}" if tmac and tip else ""

        return {
            "arp_src_host_id": float(self.arp_host_id.get(src_key)),
            "arp_tgt_host_id": float(self.arp_host_id.get(tgt_key)),
            "arp_op_num": float(op),
        }

    def _dns_features(self, pkt: Dict[str, Any]) -> Dict[str, float]:
        qc = safe_int(pkt.get("dns.qc"))
        ac = safe_int(pkt.get("dns.ac"))
        return {
            "dns_qc_norm": float(minmax_norm(float(qc), self.dns_norm.get("dns_qc_min"), self.dns_norm.get("dns_qc_max"))),
            "dns_ac_norm": float(minmax_norm(float(ac), self.dns_norm.get("dns_ac_min"), self.dns_norm.get("dns_ac_max"))),
        }

    # ---------- public: packet -> features ----------
    def preprocess(self, record_or_origin: Dict[str, Any]) -> Dict[str, Any]:
        """
        record_or_origin:
          - {"origin": {...}} 또는 origin dict
        """
        origin = record_or_origin.get("origin") if isinstance(record_or_origin, dict) and "origin" in record_or_origin else record_or_origin
        if not isinstance(origin, dict):
            raise TypeError("input must be dict (either {'origin': {...}} or origin dict)")

        idx = self._index_from_origin(origin)
        self.state.seq_counter += 1

        dt = self._delta_t_stream(origin)
        m = extract_match(origin)

        proto_str = origin.get("protocol", "")
        proto_code = protocol_to_code(proto_str)
        feat: Dict[str, Any] = {
            "protocol": float(proto_code),
            "delta_t": float(dt),
            "match": int(m),
            "protocol_norm": float(minmax_norm(float(proto_code), PROTOCOL_MIN, PROTOCOL_MAX)),
        }

        # common always
        feat.update(self._common_features(origin))

        # per-proto
        p = str(proto_str) if proto_str is not None else ""
        if p == "xgt-fen":
            p = "xgt_fen"

        if p == "s7comm":
            feat.update(self._s7_features(origin))
        elif p == "modbus":
            feat.update(self._modbus_features(origin))
        elif p == "xgt_fen":
            feat.update(self._xgt_features(origin))
        elif p == "arp":
            feat.update(self._arp_features(origin))
        elif p == "dns":
            feat.update(self._dns_features(origin))

        # 안정: 누락 컬럼 0.0 채움
        for k in self.feature_columns:
            if k not in feat:
                feat[k] = 0.0

        if self.include_index:
            feat["index"] = int(idx)

        return feat
