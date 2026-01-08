#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from utils import json_dumps_bytes, sanitize_and_drop_none

JsonDict = Dict[str, Any]


def count_jsonl_records(path: Path) -> int:
    try:
        p = Path(path)
        if not p.exists():
            return 0
        n = 0
        with p.open("rb") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n
    except Exception:
        return 0


class AsyncJsonlWriter:
    __slots__ = ("path", "batch", "flush_sec", "q", "stop_event", "thread")

    def __init__(self, path: Path, *, batch: int, flush_sec: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.batch = int(max(1, batch))
        self.flush_sec = float(max(0.05, flush_sec))

        self.q: "queue.Queue[JsonDict]" = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        buf: List[JsonDict] = []
        last_flush = time.monotonic()

        with self.path.open("ab", buffering=1024 * 1024) as f:
            while not self.stop_event.is_set() or not self.q.empty():
                timeout = max(0.05, self.flush_sec - (time.monotonic() - last_flush))
                try:
                    obj = self.q.get(timeout=timeout)
                    buf.append(obj)
                    self.q.task_done()
                except queue.Empty:
                    pass

                now = time.monotonic()
                if buf and (len(buf) >= self.batch or (now - last_flush) >= self.flush_sec):
                    self._flush(f, buf)
                    buf.clear()
                    last_flush = now

            if buf:
                self._flush(f, buf)

    @staticmethod
    def _flush(f, buf: List[JsonDict]) -> None:
        try:
            out = bytearray()
            for o in buf:
                o = sanitize_and_drop_none(o)
                try:
                    out += json_dumps_bytes(o)
                except Exception:
                    out += b'{"_error":"json_dumps_failed"}'
                out += b"\n"
            f.write(out)
            f.flush()
        except Exception:
            pass

    def write_obj(self, obj: JsonDict) -> None:
        self.q.put(obj)

    def close(self) -> None:
        self.stop_event.set()
        self.q.join()
        self.thread.join()


class AsyncJsonArrayWriter:
    __slots__ = ("path", "batch", "flush_sec", "q", "stop_event", "thread", "_first")

    def __init__(self, path: Path, *, batch: int, flush_sec: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.batch = int(max(1, batch))
        self.flush_sec = float(max(0.05, flush_sec))

        self.q: "queue.Queue[JsonDict]" = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)

        self._first = True
        self.thread.start()

    def _worker(self):
        buf: List[JsonDict] = []
        last_flush = time.monotonic()

        with self.path.open("wb", buffering=1024 * 1024) as f:
            f.write(b"[\n")
            f.flush()

            while not self.stop_event.is_set() or not self.q.empty():
                timeout = max(0.05, self.flush_sec - (time.monotonic() - last_flush))
                try:
                    obj = self.q.get(timeout=timeout)
                    buf.append(obj)
                    self.q.task_done()
                except queue.Empty:
                    pass

                now = time.monotonic()
                if buf and (len(buf) >= self.batch or (now - last_flush) >= self.flush_sec):
                    self._flush_buf(f, buf)
                    buf.clear()
                    last_flush = now

            if buf:
                self._flush_buf(f, buf)

            f.write(b"\n]\n")
            f.flush()

    def _flush_buf(self, f, buf: List[JsonDict]) -> None:
        try:
            out = bytearray()
            for o in buf:
                o = sanitize_and_drop_none(o)
                try:
                    b = json_dumps_bytes(o)
                except Exception:
                    b = b'{"_error":"json_dumps_failed"}'
                if self._first:
                    out += b
                    self._first = False
                else:
                    out += b",\n" + b
            f.write(out)
            f.flush()
        except Exception:
            pass

    def write_obj(self, obj: JsonDict) -> None:
        self.q.put(obj)

    def close(self) -> None:
        self.stop_event.set()
        self.q.join()
        self.thread.join()


class RunStores:
    __slots__ = ("incoming", "before", "after", "final", "dl_out")

    def __init__(
        self,
        incoming_path: Path,
        before_path: Path,
        after_path: Path,
        final_path: Path,
        dl_out_path: Path,
        *,
        async_batch: int,
        async_flush_sec: float,
    ):
        self.incoming = AsyncJsonlWriter(incoming_path, batch=async_batch, flush_sec=async_flush_sec)
        self.before = AsyncJsonlWriter(before_path, batch=async_batch, flush_sec=async_flush_sec)
        self.after = AsyncJsonlWriter(after_path, batch=async_batch, flush_sec=async_flush_sec)
        self.final = AsyncJsonArrayWriter(final_path, batch=async_batch, flush_sec=async_flush_sec)
        self.dl_out = AsyncJsonlWriter(dl_out_path, batch=async_batch, flush_sec=async_flush_sec)

    def close(self) -> None:
        self.incoming.close()
        self.before.close()
        self.after.close()
        self.final.close()
        self.dl_out.close()
