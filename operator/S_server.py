#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
redis_stream_server.py — Redis POP + (ML_start, DL_start) 호출
──────────────────────────────────────────────────────────────
 - Redis에서 모든 stream:protocol:* 중 가장 오래된 항목을 pop
 - ML_start(data)로 단일패킷 추론, DL_start(data)로 시퀀스 추론
 - 자동 재연결, 안전 출력, 디버그 프린트 포함
 - RedisPopper 클래스 기능 통합
Usage:
  실전용
    python3 redis_stream_server.py --loop --interval 0.1

  테스트용
    python3 redis_stream_server.py --max 30 --interval 0.1
"""

import json
import time
import argparse
from datetime import datetime

import redis

class RedisPopServer:
    """
    Redis Stream에서 데이터를 POP하여 ML/DL 추론을 수행하는 서버
    RedisPopper 기능 통합 (재연결 로직 포함)
    """
    
    def __init__(self, host="localhost", port=6379, db=0, password=None):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.redis_client = None
        self.connect()

    def connect(self):
        """Redis 연결 (실패 시 자동 재시도)"""
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
                print(f"✓ Connected to Redis at {self.host}:{self.port}")
                break
            except Exception as e:
                print(f"🚫 Redis connection failed: {e}")
                time.sleep(3)

    def get_protocols(self):
        """
        [수정됨] 분석할 OT 프로토콜 목록을 하드코딩합니다.
        - 'KEYS *'는 Redis 성능을 저하시키므로 사용하지 않습니다.
        - 'dhcp', 'unknown' 등 IT 트래픽을 제외합니다.
        """
        # redis-cli KEYS * 출력에서 확인된 OT 프로토콜 목록
        # (이 목록에 있는 Stream만 읽습니다)
        return [
            "modbus",
            "s7comm",
            "xgt-fen",       # (이전 redis-cli 출력 기반)
            "xgt_fen",       # (이전 redis-cli 출력 기반)
            "mms",
            "iec104",
            "dnp3",
            "ethernet_ip",
            "opc_ua",
            # --- 의도적으로 제외된 스트림 ---
            # "tcp_session", # 너무 일반적이어서 제외 (필요시 추가)
            # "dhcp",
            # "arp",
            # "dns",
            # "unknown"
        ]

        # [이전 코드 - 성능 문제 유발]
        # try:
        #     keys = self.redis_client.keys("stream:protocol:*")
        #     return [key.replace("stream:protocol:", "") for key in keys]
        # except Exception:
        #     self.connect()
        #     return []

    @staticmethod
    def parse_id(sid):
        """Redis Stream ID를 타임스탬프와 시퀀스로 분리"""
        try:
            ts, seq = sid.split("-")
            return int(ts), int(seq)
        except Exception:
            return (0, 0)

    def pop_oldest(self):
        """
        가장 오래된 데이터 1건 POP (삭제 후 반환)
        반환 형식: { "origin": { redis_pop_core 데이터 } }
        """
        try:
            protocols = self.get_protocols()
            if not protocols:
                return None

            oldest_entry, oldest_id, oldest_proto = None, None, None
            oldest_ts, oldest_seq = None, None

            # 모든 프로토콜에서 가장 오래된 항목 찾기
            for p in protocols:
                sname = f"stream:protocol:{p}"
                msgs = self.redis_client.xrange(sname, "-", "+", count=1)
                if not msgs:
                    continue

                msg_id, fields = msgs[0]
                if "data" not in fields:
                    continue

                ts, seq = self.parse_id(msg_id)
                if oldest_ts is None or (ts < oldest_ts or (ts == oldest_ts and seq < oldest_seq)):
                    oldest_ts, oldest_seq = ts, seq
                    oldest_id, oldest_proto, oldest_entry = msg_id, p, fields["data"]

            if not oldest_entry:
                return None

            # 삭제 후 반환 (at-most-once)
            self.redis_client.xdel(f"stream:protocol:{oldest_proto}", oldest_id)

            # JSON 파싱
            try:
                data = json.loads(oldest_entry)
            except json.JSONDecodeError:
                # JSON이 아닐 경우 raw 형태로 저장
                if isinstance(oldest_entry, dict):
                    data = dict(oldest_entry)
                else:
                    data = {"raw": oldest_entry}

            # 메타데이터 추가
            data["protocol"] = oldest_proto
            data["redis_id"] = oldest_id
            data["pop_time"] = datetime.now().isoformat()

            # origin 키로 감싸서 반환
            return {"origin": data}

        except Exception as e:
            print(f"❌ pop_oldest() error: {e}")
            self.connect()
            return None

    def run(self, interval=0.5, max_count=None, loop=False):
        """
        메인 실행 루프
        - Redis에서 데이터 POP
        - ML/DL 추론 수행
        - 결과 출력
        """
        print(f"\n🚀 Redis POP Server started (interval={interval}s, loop={loop})")
        print("=" * 70)

        count = 0
        while True:
            # 1) Redis에서 가장 오래된 데이터 POP
            wrapped_data = self.pop_oldest()
            if not wrapped_data:
                print("⚠️ No data available... waiting.")
                time.sleep(interval)
                if not loop and not max_count:
                    break
                continue

            count += 1
            data = wrapped_data["origin"]  # origin 키에서 실제 데이터 추출
            proto = data.get("protocol", "unknown")
            
            print("\n" + "=" * 80)
            print(f"#{count:05d} | 🧩 protocol={proto} | redis_id={data.get('redis_id', '-')}")
            print(f"📅 pop_time={data.get('pop_time', '-')}")
            print("-" * 80)

            # RAW 데이터 출력 (wrapped 형태)
            try:
                print("📦 [RAW Redis Data - Wrapped]")
                print(json.dumps(wrapped_data, ensure_ascii=False, indent=2)[:500] + "...")
            except Exception as e:
                print(f"⚠️ Error while printing raw data: {e}")

            # 평탄화 키 프리뷰
            try:
                flat = {}

                def flatten(prefix, obj):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            flatten(f"{prefix}.{k}" if prefix else k, v)
                    else:
                        flat[prefix] = obj

                flatten("", data)

                print("-" * 80)
                print("🧾 [Flattened Keys Preview] (최대 30개)")
                for i, (k, v) in enumerate(list(flat.items())[:30]):
                    print(f" {i+1:02d}. {k}: {v}")
            except Exception as e:
                print(f"⚠️ Flatten preview error: {e}")

            print("=" * 80)

            # 2) ML 추론 (실제 데이터 전달)
            ml_out = ML_start(data)
            if not ml_out:
                print("❌ ML 추론 실패 — 스킵")
                time.sleep(interval)
                if not loop and not max_count:
                    break
                continue

            match = ml_out["match"]
            label_text = "🟢 Normal" if match == "O" else "🔴 Anomaly"
            print(f"→ [ML] {label_text} | protocol={ml_out['protocol']}")

            impact = ml_out.get("impact", {})
            if impact and "info" not in impact:
                top_feats = list(impact.keys())[:3]
                print(f"   ⚙️ 주요 영향 feature → {', '.join(top_feats)}")
            else:
                print("   ⚙️ 영향 feature 정보 없음")

            used = ml_out.get("used_features", [])
            if used:
                sample_feats = ', '.join([f"{n}:{v}" for n, v in used[:3]])
                print(f"   📊 feature 샘플 → {sample_feats} ...")

            # 3) DL 추론 (슬라이딩 윈도우)
            dl_out = DL_start(data)
            if not dl_out:
                print("❌ DL 추론 실패")
            elif not dl_out.get("ready"):
                w = dl_out.get("window_size")
                fd = dl_out.get("feature_dim")
                print(f"⏳ [DL] window {w}/{fd} — 대기")
            else:
                print(f"→ [DL] {dl_out['label']} | MSE={dl_out['mse']:.6f} (TH={dl_out['threshold']:.6f})")

            # 4) 루프 제어
            if max_count and count >= max_count:
                print(f"\n✅ Reached max_count={max_count}. Stopping gracefully.\n")
                break

            time.sleep(interval)
            if not loop and not max_count:
                print("\nℹ️ Loop disabled — stopping after one iteration.")
                break


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Redis ICS POP Server (uses ML_start & DL_start)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--db", type=int, default=0)
    parser.add_argument("--password")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--max", type=int)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    server = RedisPopServer(host=args.host, port=args.port, db=args.db, password=args.password)
    server.run(interval=args.interval, max_count=args.max, loop=args.loop)


if __name__ == "__main__":
    main()