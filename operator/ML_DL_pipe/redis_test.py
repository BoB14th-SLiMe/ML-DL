#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import heapq

import redis


# DEFAULT_PROTOCOLS = [ "modbus", "s7comm", "xgt_fen", "xgt-fen" ]
DEFAULT_PROTOCOLS = [ "modbus", "s7comm", "xgt_fen", "xgt-fen", "tcp", "udp", "dns", "arp" ]


def parse_id(sid: str) -> Tuple[int, int]:
    try:
        ts_ms, seq = sid.split("-")
        return int(ts_ms), int(seq)
    except Exception:
        return (0, 0)


def safe_json_load(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {"raw": s}


def connect(host: str, port: int, db: int, password: Optional[str]) -> redis.Redis:
    r = redis.Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
    )
    r.ping()
    return r


def stream_name(proto: str) -> str:
    return f"stream:protocol:{proto}"


def peek_stream(r: redis.Redis, proto: str, count: int, newest: bool = False):
    sname = stream_name(proto)
    if newest:
        # newest부터 count개
        return r.xrevrange(sname, "+", "-", count=count)
    # oldest부터 count개
    return r.xrange(sname, "-", "+", count=count)


def pop_entries(r: redis.Redis, proto: str, ids: List[str]) -> int:
    if not ids:
        return 0
    sname = stream_name(proto)
    # xdel(stream, id1, id2, ...)
    return int(r.xdel(sname, *ids))


def mode_per_stream(
    r: redis.Redis,
    protocols: List[str],
    count: int,
    action: str,
    newest: bool,
) -> None:
    for p in protocols:
        sname = stream_name(p)
        entries = peek_stream(r, p, count=count, newest=newest)
        if not entries:
            continue

        print(f"\n=== {sname} ({len(entries)}개) ===")
        del_ids: List[str] = []

        for msg_id, fields in entries:
            payload = safe_json_load(fields.get("data", ""))
            payload.setdefault("protocol", p)
            payload.setdefault("redis_id", msg_id)
            print(json.dumps(payload, ensure_ascii=False))

            del_ids.append(msg_id)

        if action == "pop":
            deleted = pop_entries(r, p, del_ids)
            print(f"-> XDEL 삭제: {deleted}개")


def mode_global_oldest(
    r: redis.Redis,
    protocols: List[str],
    count: int,
    action: str,
) -> None:
    """
    각 스트림에서 최대 count개를 가져온 뒤,
    전역(전체 프로토콜)에서 가장 오래된 것부터 총 count개를 출력/삭제.
    """
    heap: List[Tuple[int, int, str, str, Dict[str, str]]] = []

    # 1) 각 프로토콜에서 up to count개 미리 조회
    for p in protocols:
        entries = peek_stream(r, p, count=count, newest=False)
        for msg_id, fields in entries:
            ts, seq = parse_id(msg_id)
            heapq.heappush(heap, (ts, seq, p, msg_id, fields))

    # 2) 전역 oldest count개 선택
    picked: List[Tuple[str, str]] = []  # (proto, id)
    printed = 0

    while heap and printed < count:
        ts, seq, p, msg_id, fields = heapq.heappop(heap)
        payload = safe_json_load(fields.get("data", ""))
        payload.setdefault("protocol", p)
        payload.setdefault("redis_id", msg_id)
        payload.setdefault("redis_timestamp_ms", ts)
        payload.setdefault("pop_time_local", datetime.now().isoformat())

        print(json.dumps(payload, ensure_ascii=False))
        picked.append((p, msg_id))
        printed += 1

    if action == "pop":
        # 프로토콜별로 묶어서 XDEL
        by_proto: Dict[str, List[str]] = {}
        for p, mid in picked:
            by_proto.setdefault(p, []).append(mid)

        total_deleted = 0
        for p, ids in by_proto.items():
            total_deleted += pop_entries(r, p, ids)

        print(f"-> XDEL 삭제(전역 기준): {total_deleted}개")


def main():
    ap = argparse.ArgumentParser(description="Dump Redis Streams in batches (10씩)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=6379)
    ap.add_argument("--db", type=int, default=0)
    ap.add_argument("--password", default=None)

    ap.add_argument("--protocols", nargs="+", default=DEFAULT_PROTOCOLS)
    ap.add_argument("--count", type=int, default=10, help="한 번에 뽑을 개수(기본 10)")

    ap.add_argument("--mode", choices=["per-stream", "global-oldest"], default="per-stream")
    ap.add_argument("--action", choices=["peek", "pop"], default="peek", help="peek=조회만, pop=조회+삭제")
    ap.add_argument("--newest", action="store_true", help="per-stream 모드에서 최신부터 뽑기(xrevrange)")

    args = ap.parse_args()

    r = connect(args.host, args.port, args.db, args.password)
    print(f"✓ connected: {args.host}:{args.port} (db={args.db})")

    if args.mode == "per-stream":
        mode_per_stream(r, args.protocols, args.count, args.action, newest=args.newest)
    else:
        mode_global_oldest(r, args.protocols, args.count, args.action)


if __name__ == "__main__":
    main()

"""
python redis_test.py --host <REDIS_IP> --mode per-stream --count 10 --action peek

"""
