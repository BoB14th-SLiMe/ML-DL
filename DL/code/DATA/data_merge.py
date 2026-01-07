#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


MERGE_PROTOCOLS = {"modbus", "xgt_fen"}
SPECIAL_MERGE_PAIRS = {
    "modbus": ("modbus.regs.translated_addr", "modbus.regs.val"),
    "xgt_fen": ("xgt_fen.word_offset", "xgt_fen.word_value"),
}

JsonDict = Dict[str, Any]
GroupKey = Tuple[str, Any]


def _as_list_str(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def _resolve_path(base_dir: Path, v: Any) -> Path:
    if v is None:
        raise ValueError("path is required")
    s = str(v).strip()
    if not s:
        raise ValueError("path is empty")
    p = Path(s).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()


def read_jsonl(path: Path, encoding: str) -> Iterator[JsonDict]:
    with path.open("r", encoding=encoding) as fin:
        for line_no, line in enumerate(fin, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                print(f"[WARN] JSON parse fail: {path.name}:{line_no}", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                yield obj


def iter_consecutive_groups(logs: Iterable[JsonDict]) -> Iterator[List[JsonDict]]:
    cur: List[JsonDict] = []
    cur_proto: Optional[str] = None
    cur_key: Optional[GroupKey] = None

    for log in logs:
        proto = str(log.get("protocol") or "")

        if proto not in MERGE_PROTOCOLS:
            if cur:
                yield cur
                cur = []
                cur_proto = None
                cur_key = None
            yield [log]
            continue

        key: GroupKey = (str(log.get("@timestamp") or ""), log.get("sq"))

        if cur_proto is None or not (proto == cur_proto and key == cur_key):
            if cur:
                yield cur
            cur = [log]
            cur_proto = proto
            cur_key = key
        else:
            cur.append(log)

    if cur:
        yield cur


def merge_group(pkts: List[JsonDict]) -> JsonDict:
    if len(pkts) == 1:
        return pkts[0]

    base = dict(pkts[0])
    proto = str(base.get("protocol") or "")
    if proto not in MERGE_PROTOCOLS:
        return base

    pair = SPECIAL_MERGE_PAIRS.get(proto)
    special_keys = set(pair) if pair else set()

    if pair:
        a, v = pair
        addrs: List[str] = []
        vals: List[str] = []
        for p in pkts:
            addrs.extend(_as_list_str(p.get(a)))
            vals.extend(_as_list_str(p.get(v)))
        base[a] = addrs
        base[v] = vals

    keys: set[str] = set()
    for p in pkts:
        keys.update(p.keys())

    prefix = proto + "."
    for k in sorted(keys):
        if k in special_keys or not k.startswith(prefix):
            continue
        vals = [p.get(k) for p in pkts]
        first = vals[0]
        base[k] = first if all(x == first for x in vals[1:]) else vals

    return base


def merge_stream(input_path: Path, output_path: Path, encoding: str) -> Tuple[int, int]:
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    in_valid = 0
    out_count = 0
    with output_path.open("w", encoding=encoding) as fout:
        for g in iter_consecutive_groups(read_jsonl(input_path, encoding)):
            in_valid += 1
            fout.write(json.dumps(merge_group(g), ensure_ascii=False) + "\n")
            out_count += 1
    return in_valid, out_count


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        raise RuntimeError("PyYAML 필요: pip install pyyaml")
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("YAML root must be dict")
    return obj


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_yaml = script_dir / "data_merge.yaml"

    cfg_path = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) >= 2 else default_yaml
    if not cfg_path.exists():
        print(f"[ERROR] config not found: {cfg_path}", file=sys.stderr)
        return 1

    cfg = load_yaml(cfg_path)
    base = cfg_path.parent.resolve()

    defaults = cfg.get("defaults") if isinstance(cfg.get("defaults"), dict) else {}
    jobs = cfg.get("jobs")
    if not isinstance(jobs, list):
        print("[ERROR] YAML jobs must be a list", file=sys.stderr)
        return 1

    def_enc = str(defaults.get("encoding", "utf-8-sig"))

    for j in jobs:
        if not isinstance(j, dict):
            continue
        if j.get("enabled", True) is False:
            continue

        name = str(j.get("name", "(no-name)"))
        enc = str(j.get("encoding", def_enc))

        try:
            in_path = _resolve_path(base, j.get("input"))
            out_path = _resolve_path(base, j.get("output"))
        except Exception as e:
            print(f"[ERROR] job '{name}': {e}", file=sys.stderr)
            return 1

        print(f"[INFO] job={name}")
        print(f"       input={in_path}")
        print(f"       output={out_path}")
        try:
            in_valid, out_count = merge_stream(in_path, out_path, enc)
        except Exception as e:
            print(f"[ERROR] job '{name}' failed: {e}", file=sys.stderr)
            return 1
        print(f"[INFO] done: in_valid_dict={in_valid}, out={out_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
