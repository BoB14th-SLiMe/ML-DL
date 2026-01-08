#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple


Json = Dict[str, Any]


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    s = str(ts).strip()
    if not s:
        return None
    # "Z" -> "+00:00"
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _to_iso_z(dt: datetime) -> str:
    # 항상 UTC Z로 출력
    dt_utc = dt.astimezone(timezone.utc)
    # 입력 예시처럼 microsecond 유지
    return dt_utc.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _extract_dt(rec: Json) -> Optional[datetime]:
    # 우선순위: @timestamp -> timestamp
    ts = rec.get("@timestamp") or rec.get("timestamp")
    if ts is None:
        return None
    return _parse_iso(str(ts))


def _read_first_valid_dt(path: Path) -> Optional[datetime]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if isinstance(rec, dict):
                dt = _extract_dt(rec)
                if dt is not None:
                    return dt
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Remap @timestamp for JSONL: shift by preserving Δt (default) or fixed interval."
    )
    ap.add_argument("--in", dest="in_path", required=True, help="input JSONL path")
    ap.add_argument("--out", dest="out_path", required=True, help="output JSONL path")
    ap.add_argument(
        "--start",
        required=True,
        help='new start timestamp for the FIRST record (ISO8601). e.g. "2025-12-20T00:00:00.000000Z"',
    )
    ap.add_argument(
        "--mode",
        choices=["shift", "fixed"],
        default="shift",
        help="shift: preserve original deltas from first timestamp; fixed: generate timestamps by fixed step.",
    )
    ap.add_argument(
        "--fixed-ms",
        type=float,
        default=1.0,
        help="(mode=fixed) step in milliseconds for each next record. default=1.0",
    )
    ap.add_argument(
        "--set-field",
        default="@timestamp",
        help='which field to set. default="@timestamp" (also updates "timestamp" if it exists)',
    )

    args = ap.parse_args()

    in_path = Path(args.in_path).resolve()
    out_path = Path(args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_start_dt = _parse_iso(args.start)
    if new_start_dt is None:
        raise SystemExit(f"Invalid --start: {args.start}")

    mode = args.mode
    set_field = str(args.set_field)

    old_first_dt = None
    if mode == "shift":
        old_first_dt = _read_first_valid_dt(in_path)
        if old_first_dt is None:
            raise SystemExit("mode=shift requires at least one valid @timestamp/timestamp in input.")

    fixed_step = timedelta(milliseconds=float(args.fixed_ms))

    n = 0
    base_old = old_first_dt
    base_new = new_start_dt

    with in_path.open("r", encoding="utf-8-sig") as fi, out_path.open("w", encoding="utf-8") as fo:
        for line in fi:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                # 깨진 라인은 그대로 쓰되, timestamp만 fixed 모드에서라도 부여할지 여부는 정책에 따라 다름
                continue

            if not isinstance(rec, dict):
                continue

            if mode == "shift":
                cur_dt = _extract_dt(rec)
                if cur_dt is None:
                    # timestamp가 없는 레코드는 스킵(원하시면 여기서 fixed처럼 부여하도록 변경 가능)
                    continue
                delta = cur_dt - base_old  # type: ignore[arg-type]
                new_dt = base_new + delta
            else:
                # fixed
                new_dt = base_new + (fixed_step * n)

            new_ts = _to_iso_z(new_dt)

            # 지정 필드 세팅
            rec[set_field] = new_ts

            # 흔히 timestamp도 같이 쓰는 경우가 있어서, 기존에 있으면 같이 갱신
            if set_field == "@timestamp" and "timestamp" in rec:
                rec["timestamp"] = new_ts

            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"[OK] remapped={n} -> {out_path}")


if __name__ == "__main__":
    main()

"""
python timestamp.py --in  ./attack_ver5_1_change.jsonl --out ./1.jsonl --start "2025-11-28T04:42:56.757705Z" --mode shift
python timestamp.py --in  ./attack_ver2.jsonl --out ./2.jsonl --start "2025-11-28T04:43:34.255562Z" --mode shift

"""