#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import json_loads_fast

try:
    import redis  # type: ignore
except Exception:
    redis = None

JsonDict = Dict[str, Any]


def iso_to_epoch_ms(ts: Any) -> int:
    if ts is None:
        return 0
    s = str(ts).strip()
    if not s:
        return 0
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return 0


class RedisPopServer:
    __slots__ = ("host", "port", "db", "password", "protocols", "redis_client")

    def __init__(self, host: str, port: int, db: int, password: Optional[str], protocols: List[str]):
        if redis is None:
            raise RuntimeError("redis package not available. Install: pip install redis")
        self.host = host
        self.port = int(port)
        self.db = int(db)
        self.password = password
        self.protocols = list(protocols)
        self.redis_client: Optional[redis.Redis] = None
        self.connect()

    def connect(self) -> None:
        while True:
            try:
                self.redis_client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                )
                self.redis_client.ping()
                break
            except Exception as e:
                print(f"🚫 Redis 연결 실패: {e}. 3초 후 재시도합니다.")
                time.sleep(3)

    @staticmethod
    def parse_id(sid: str) -> Tuple[int, int]:
        try:
            ts_ms, seq = sid.split("-")
            return int(ts_ms), int(seq)
        except Exception:
            return (0, 0)

    def pop_oldest(self) -> Optional[JsonDict]:
        try:
            rc = self.redis_client
            if rc is None:
                self.connect()
                rc = self.redis_client
                if rc is None:
                    return None

            pipe = rc.pipeline(transaction=False)
            stream_infos = []
            for p in self.protocols:
                sname = f"stream:protocol:{p}"
                stream_infos.append((p, sname))
                pipe.xrange(sname, "-", "+", count=1)

            results = pipe.execute()

            best = None  # (proto, sname, msg_id, raw, id_ts, id_seq)
            for (proto, sname), msgs in zip(stream_infos, results):
                if not msgs:
                    continue
                msg_id, fields = msgs[0]
                raw = fields.get("data")
                if raw is None:
                    continue
                id_ts, id_seq = self.parse_id(msg_id)
                cand = (proto, sname, msg_id, raw, int(id_ts), int(id_seq))
                if best is None:
                    best = cand
                else:
                    _, _, _, _, bts, bseq = best
                    if (id_ts < bts) or (id_ts == bts and id_seq < bseq):
                        best = cand

            if best is None:
                return None

            proto, sname, msg_id, raw, id_ts, _ = best
            rc.xdel(sname, msg_id)

            meta = {
                "redis_id": msg_id,
                "redis_timestamp_ms": int(id_ts or 0),
                "packet_timestamp_ms": int(id_ts or 0),
                "pop_time": datetime.now().isoformat(),
                "protocol": proto,
            }
            return {"origin_raw": raw, "protocol": proto, "_meta": meta}

        except Exception as e:
            print(f"❌ pop_oldest() 오류: {e}")
            self.connect()
            return None


class JsonlPopServer:
    __slots__ = ("path", "fp", "line_no", "done")

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self.fp = self.path.open("r", encoding="utf-8-sig")
        self.line_no = 0
        self.done = False
        print(f"✓ JSONL 입력 모드: {self.path}")

    def close(self) -> None:
        try:
            self.fp.close()
        except Exception:
            pass

    def pop_oldest(self) -> Optional[JsonDict]:
        if self.done:
            return None

        while True:
            line = self.fp.readline()
            if not line:
                self.done = True
                return None

            self.line_no += 1
            s = line.strip()
            if not s:
                continue

            try:
                obj = json_loads_fast(s)
            except Exception:
                obj = {"raw": s}

            if isinstance(obj, dict) and isinstance(obj.get("origin"), dict):
                origin = obj.get("origin") or {}
                meta = obj.get("_meta") or {}
            else:
                origin = obj if isinstance(obj, dict) else {"value": obj}
                meta = {}

            proto = origin.get("protocol") or meta.get("protocol") or "unknown"
            origin["protocol"] = proto

            ts_ms = iso_to_epoch_ms(origin.get("@timestamp") or origin.get("timestamp"))
            if ts_ms <= 0:
                ts_ms = int(self.line_no)

            meta_out = {
                "redis_id": meta.get("redis_id") or f"file:{self.path.name}:{self.line_no}",
                "redis_timestamp_ms": int(meta.get("redis_timestamp_ms") or ts_ms),
                "packet_timestamp_ms": int(meta.get("packet_timestamp_ms") or ts_ms),
                "pop_time": datetime.now().isoformat(),
                "protocol": proto,
            }
            return {"origin": origin, "_meta": meta_out, "protocol": proto}