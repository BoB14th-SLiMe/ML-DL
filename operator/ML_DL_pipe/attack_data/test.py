#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rewrite_xgt_from_timestamp.py

--start-ts 로 지정한 타임스탬프(ISO8601) 이상이 되는 시점부터,
파일 끝까지(나머지 데이터) xgt_fen D523~D528 대상의 word_value를 강제 치환한다.

지원:
- flat keys:   "xgt_fen.translated_addr", "xgt_fen.word_value"
- nested:      "xgt_fen": {"translated_addr": ..., "word_value": ...}
- 키 변형: translated_addr / word_value 를 포함하는 키 자동 탐색
- addr/value 가 리스트인 경우도 지원

예)
python rewrite_xgt_from_timestamp.py --in attack.jsonl --out attack_changed.jsonl \
  --start-ts "2025-11-28T07:28:44.000000Z" --debug
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

Json = Dict[str, Any]

TARGET_LIST = ["D523", "D524", "D525", "D526", "D527", "D528"]
TARGET_SET = set(TARGET_LIST)

# 시작 시점 이후에 강제 치환할 값
FIXED = {
    "D523": "5",
    "D524": "30",
    "D525": "245",
    "D526": "0",
    "D527": "0",   # 핵심
    "D528": "39",
}


def _norm(v: Any) -> str:
    return str(v).strip().upper()


def _parse_ts(ts: str) -> Optional[datetime]:
    """
    ISO8601 파싱 (예: 2025-11-28T07:28:44.436776Z)
    - Z -> +00:00 처리
    - tzinfo 없으면 UTC로 간주
    """
    if not ts:
        return None
    s = ts.strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _coerce_like(old_value: Any, new_value_str: str) -> Any:
    if isinstance(old_value, int):
        try:
            return int(new_value_str)
        except Exception:
            return new_value_str
    if isinstance(old_value, float):
        try:
            return float(new_value_str)
        except Exception:
            return new_value_str
    return new_value_str


def _auto_find_flat_keys(obj: Json) -> Tuple[Optional[str], Optional[str]]:
    keys = list(obj.keys())
    addr_candidates = [k for k in keys if "translated_addr" in k]
    val_candidates = [k for k in keys if "word_value" in k]
    if not addr_candidates or not val_candidates:
        return None, None

    def _score(k: str) -> Tuple[int, int, int]:
        return (
            1 if "xgt_fen" in k else 0,
            1 if k.startswith("xgt_fen.") else 0,
            len(k),  # 약간 더 구체적인 키를 우선
        )

    addr_candidates.sort(key=_score, reverse=True)
    val_candidates.sort(key=_score, reverse=True)
    return addr_candidates[0], val_candidates[0]


def _locate_fields(obj: Json) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    # nested 우선
    xgt = obj.get("xgt_fen")
    if isinstance(xgt, dict) and ("translated_addr" in xgt) and ("word_value" in xgt):
        return xgt, "translated_addr", "word_value"

    # flat exact
    if "xgt_fen.translated_addr" in obj and "xgt_fen.word_value" in obj:
        return obj, "xgt_fen.translated_addr", "xgt_fen.word_value"

    # flat auto
    ak, vk = _auto_find_flat_keys(obj)
    if ak and vk:
        return obj, ak, vk

    return None, None, None


def _rewrite_addr_value(container: Dict[str, Any], addr_key: str, val_key: str) -> bool:
    """
    - addr/val 스칼라: addr가 D523~D528이면 val을 FIXED로 치환
    - addr/val 리스트: addr 리스트에서 D523~D528에 해당하는 위치의 val을 FIXED로 치환
    """
    addr = container.get(addr_key)
    val = container.get(val_key)
    changed = False

    if isinstance(addr, list) and isinstance(val, list):
        if len(addr) != len(val):
            return False
        for i, a in enumerate(addr):
            a_norm = _norm(a)
            if a_norm in TARGET_SET:
                old = val[i]
                val[i] = _coerce_like(old, FIXED[a_norm])
                if _norm(old) != _norm(val[i]):
                    changed = True
        if changed:
            container[val_key] = val
        return changed

    if not isinstance(addr, list) and not isinstance(val, list):
        a_norm = _norm(addr)
        if a_norm in TARGET_SET:
            old = container.get(val_key)
            container[val_key] = _coerce_like(old, FIXED[a_norm])
            return _norm(old) != _norm(container[val_key])

    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="input JSONL")
    ap.add_argument("--out", dest="out_path", required=True, help="output JSONL")

    # ✅ start-ts를 선택 옵션으로 변경
    ap.add_argument(
        "--start-ts",
        default=None,
        help='start timestamp (ISO8601), e.g. 2025-11-28T07:28:44.000000Z'
    )

    # ✅ 파일 전체 강제 적용 옵션
    ap.add_argument("--all", action="store_true", help="rewrite from beginning (ignore start-ts)")

    ap.add_argument("--debug", action="store_true", help="print some rewrite logs")
    ap.add_argument("--max-debug", type=int, default=30)
    ap.add_argument("--keep-bad-lines", action="store_true", help="keep bad JSON lines as-is (default: skip)")
    args = ap.parse_args()

    # ✅ all이면 시작 시점 파싱 불필요
    start_dt = None
    if not args.all:
        if not args.start_ts:
            raise SystemExit("[ERROR] --start-ts is required unless --all is set")
        start_dt = _parse_ts(args.start_ts)
        if start_dt is None:
            raise SystemExit(f"[ERROR] cannot parse --start-ts: {args.start_ts}")

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    bad_json = 0

    # ✅ all이면 처음부터 started=True
    started = bool(args.all)
    started_at_line = 1 if args.all else None

    rewrote = 0
    debug_left = args.max_debug
    key_info_printed = False

    with in_path.open("r", encoding="utf-8-sig") as fin, out_path.open("w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.rstrip("\n")
            if not line.strip():
                continue

            total += 1

            try:
                obj = json.loads(line)
            except Exception:
                bad_json += 1
                if args.keep_bad_lines:
                    fout.write(line + "\n")
                continue

            if not isinstance(obj, dict):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            # ✅ start-ts 모드일 때만 started 전환 체크
            if (not args.all) and (not started):
                ts_str = obj.get("@timestamp") or obj.get("timestamp") or obj.get("ts")
                dt = _parse_ts(ts_str) if isinstance(ts_str, str) else None
                if dt is not None and dt >= start_dt:
                    started = True
                    started_at_line = total

            # started 이후부터 끝까지 치환 (all이면 처음부터)
            if started:
                container, addr_key, val_key = _locate_fields(obj)
                if container is not None and addr_key and val_key:
                    if args.debug and not key_info_printed:
                        print(f"[DEBUG] detected keys: addr_key={addr_key}, val_key={val_key}")
                        key_info_printed = True

                    before = container.get(val_key)
                    ch = _rewrite_addr_value(container, addr_key, val_key)
                    after = container.get(val_key)

                    if ch:
                        rewrote += 1
                        if args.debug and debug_left > 0:
                            debug_left -= 1
                            ts = obj.get("@timestamp", "")
                            addr_show = container.get(addr_key)
                            print(f"[DEBUG] ts={ts} addr={addr_show} word_value: {before} -> {after}")

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[INFO] total_lines={total}")
    if args.all:
        print("[INFO] mode=ALL (rewrite from beginning)")
    else:
        print(f"[INFO] start_ts={args.start_ts} (parsed_utc={start_dt.isoformat()})")
    print(f"[INFO] started={started}, started_at_line={started_at_line}")
    print(f"[INFO] rewrote={rewrote}")
    print(f"[INFO] bad_json={bad_json}")
    print(f"[INFO] output -> {out_path}")


if __name__ == "__main__":
    main()



"""
python test.py --in attack_ver5_1.jsonl --out attack_ver5_1_change.jsonl --start-ts "2025-11-28T07:28:44.436776Z" --debug --all

"""

