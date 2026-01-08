"""
S_PipeLine_file.py — FC6 여부 상관없이 모든 윈도우에서 DL_start() 실행 버전
  - FC6 패킷(redis_id) 기준 평가:
    • modbus.fc == 6 인 패킷이 포함된 윈도우들 중
      alert == 'o' 가 한 번이라도 나오면  → T
    • 끝까지 한 번도 그런 윈도우에 포함되지 않으면 → F

  - 최종 출력 형식(JSONL 한 줄):
    {
      "origin": {
        "window_raw": [ ... origin + ML 이 합쳐진 패킷들 ... ]
      },
      "DL": { ... DL 결과 ... }
    }
"""

from __future__ import annotations

import json
import time
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from copy import deepcopy
import threading
import queue
from ML_start import ML_start
from DL_start import DL_start
from datetime import datetime
import requests   # 🔥 API 호출용

DL_OUTPUT_PATH = Path("/home/slime/SLM/DL/output/dl_anomaly_detect.jsonl")
PIPELINE_WINDOW_SIZE = 80
PIPELINE_STEP_SIZE = 40  # 슬라이딩 stride

# AI-PC Alarm Ingestion API 기본 URL
ALARM_BASE_URL = "http://192.168.4.140:8080"

class JsonlWriter:
    """JSONL 파일에 thread-safe하게 기록하는 전담 쓰레드"""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.queue: "queue.Queue[dict]" = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        # 파일을 한 번만 열고 계속 append
        with self.path.open("a", encoding="utf-8") as f:
            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    record = self.queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                try:
                    line = json.dumps(record, ensure_ascii=False)
                    f.write(line + "\n")
                    f.flush()
                except Exception as e:
                    print(f"[JsonlWriter] write 실패: {e}")

                self.queue.task_done()

    def write(self, record: Dict[str, Any]) -> None:
        """다른 쓰레드에서 호출 → 내부 큐에만 넣음"""
        self.queue.put(record)

    def close(self):
        """파이프라인 종료 시 반드시 호출"""
        self.stop_event.set()
        self.queue.join()   # 큐 비워질 때까지 대기
        self.thread.join()


def send_alarm_to_api_from_dl(dl_output: Dict[str, Any], engine: str = "dl") -> None:
    """
    DL 결과(dl_output)에서 summary.risk 정보를 뽑아서
    Alarm Ingestion API(/api/alarms/{engine})로 전송.
    """
    try:
        summary = dl_output.get("DL", {}).get("summary", {}) or {}
        risk = summary.get("risk", {}) or {}
    except Exception:
        print("  [API] DL output 구조 이상으로 risk 추출 실패")
        return

    if not isinstance(risk, dict):
        print("  [API] risk 구조가 dict가 아님.")
        return

    # score 없으면 anomaly_score 를 score 로 사용
    if "score" not in risk and "anomaly_score" in summary:
        try:
            risk["score"] = float(summary["anomaly_score"])
        except Exception:
            risk["score"] = 0.0

    # detected_time 비어 있으면 지금 시간으로 채우기
    risk.setdefault(
        "detected_time",
        datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )

    # 나머지 필드는 최소한 빈 문자열이라도 존재하도록
    for key in ("src_ip", "src_asset", "dst_ip", "dst_asset"):
        risk.setdefault(key, "")

    body = {"risk": risk}

    url = f"{ALARM_BASE_URL}/api/alarms/{engine}"
    try:
        resp = requests.post(url, json=body, timeout=3)
        resp.raise_for_status()
        print(f"  [API] Alarm sent → {url} ({resp.status_code})")
        print(f"  [API] body = {body}")
    except Exception as e:
        print(f"  [API] Alarm send FAILED: {e}")


def print_safe(data):
    try:
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    except Exception:
        print(repr(data))


def iter_jsonl_wrapped(path: Path):
    """JSONL 파일 → wrapped_data 형태로 변환"""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"⚠️ JSONL 파싱 실패 (line {line_no}): {e}")
                continue

            if isinstance(obj, dict) and "origin" in obj:
                wrapped = obj
            else:
                wrapped = {"origin": obj}

            origin = wrapped.get("origin", {})
            if "redis_id" not in origin:
                origin["redis_id"] = f"line-{line_no}"
                wrapped["origin"] = origin

            yield wrapped


class SequentialPipeLineFromFile:
    def __init__(self, input_path: Path, dl_out_path: Optional[Path] = None):
        print("1. [File] JSONL 입력 파일 로딩 준비...")

        if not input_path.exists():
            raise FileNotFoundError(f"입력 JSONL 파일을 찾을 수 없습니다: {input_path}")
        self.input_path = input_path

        self.data_buffer: List[Dict[str, Any]] = []
        self.window_size = PIPELINE_WINDOW_SIZE

        self.seq_counter: int = 0

        # 🔥 수정된 부분 — 옵션 받은 경로를 사용
        self.dl_out_path: Path = Path(dl_out_path) if dl_out_path else DL_OUTPUT_PATH

        print(f"✓ DL 결과 저장 파일: {self.dl_out_path}")
        self.writer = JsonlWriter(self.dl_out_path)

        self.fc6_ids_seen: Set[str] = set()
        self.fc6_ids_detected: Set[str] = set()


    # 🔍 윈도우 안에 modbus.fc == 6 인 패킷들의 redis_id 집합 반환
    def _window_fc6_ids(self, window_batch: List[Dict[str, Any]]) -> Set[str]:
        fc6_ids: Set[str] = set()
        for wrapped in window_batch:
            origin = wrapped.get("origin", {})
            protocol = origin.get("protocol")
            fc = origin.get("modbus.fc")
            if fc is None:
                fc = origin.get("function_code")

            if protocol in ("modbus", "modbus_tcp") and fc is not None:
                try:
                    if int(fc) == 6:
                        rid = origin.get("redis_id")
                        if rid is not None:
                            fc6_ids.add(str(rid))
                except Exception:
                    continue
        return fc6_ids


    def _append_dl_result_to_file(
        self,
        dl_output: Dict[str, Any],
        window_raw: List[Dict[str, Any]],
    ) -> None:
        """
        최종 출력 형식:
        {
          "seq_id": int,
          "pattern": "P_XXXX",
          "summary": { ... DL summary ... },
          "alert": "o" 또는 "x",
          "window_raw": [
            {
              ... origin 필드들(일부 xgt_fen 메타필드 제거) ...,
              "ml_anomaly_prob": [ { "name": ..., "percent": ... }, ... ]
            },
            ...
          ]
        }
        """
        if self.dl_out_path is None:
            return

        dl_block = dl_output.get("DL", dl_output)

        seq_id = dl_block.get("seq_id")
        pattern = dl_block.get("pattern")
        summary = dl_block.get("summary", {})
        alert = dl_output.get("alert", "x")

        # 🔻 window_raw에 넣지 않을 xgt_fen 메타필드들
        XGT_FEN_DROP_KEYS = {
            "xgt_fen.companyId",
            "xgt_fen.plcinfo",
            "xgt_fen.cpuinfo",
            "xgt_fen.source",
            "xgt_fen.len",
            "xgt_fen.fenetpos",
            "xgt_fen.dtype",
            "xgt_fen.blkcnt",
            "xgt_fen.errstat",
            "xgt_fen.errinfo",
            "xgt_fen.datasize",
        }

        simple_window: List[Dict[str, Any]] = []
        for pkt in window_raw:
            pkt_copy = deepcopy(pkt)

            ml = pkt_copy.pop("ML", None)
            if isinstance(ml, dict) and "anomaly_prob" in ml:
                pkt_copy["ml_anomaly_prob"] = ml["anomaly_prob"]

            for k in XGT_FEN_DROP_KEYS:
                pkt_copy.pop(k, None)

            simple_window.append(pkt_copy)

        record = {
            "seq_id": seq_id,
            "pattern": pattern,
            "summary": summary,
            "window_raw": simple_window,
        }

        # 🔁 여기서 직접 파일 열지 않고, Writer 쓰레드에 위임
        self.writer.write(record)




    def run(self, interval: float = 0.0, max_count: int | None = None):

        print(f"\n🚀 S_PipeLine_file.py 시작 (interval={interval}s, max_count={max_count})")
        print("=" * 80)

        count = 0

        for wrapped_data in iter_jsonl_wrapped(self.input_path):
            count += 1
            packet_id = wrapped_data.get("origin", {}).get("redis_id", "-")
            print(f"\n#{count:05d} [Step 1] 입력 POP (from file): {packet_id}")

            data_origin = wrapped_data["origin"]

            # Step 2: ML 추론
            try:
                raw = ML_start(data_origin) or {}

                # 1) {'ML': {...}} 형태이면 안쪽 딕셔너리만 꺼내기
                if isinstance(raw, dict) and "ML" in raw and isinstance(raw["ML"], dict):
                    ml_output = raw["ML"]
                else:
                    ml_output = raw

                # 2) 그래도 dict가 아니면 그냥 raw로 감싸기 (최후 방어)
                if not isinstance(ml_output, dict):
                    ml_output = {"raw": ml_output}

                wrapped_data["ML"] = ml_output

            except Exception as e:
                print(f"❌ [Step 2] ML_start() 오류: {e}")
                time.sleep(interval)
                continue

            # Step 3: 윈도우 버퍼링
            self.data_buffer.append(wrapped_data)
            current_buffer_size = len(self.data_buffer)

            if current_buffer_size < self.window_size:
                print(f"  [Step 3.0] DL 버퍼링: [{current_buffer_size}/{self.window_size}]... 대기")
                continue

            # 최신 window_size 만큼 윈도우 구성
            current_window_batch = self.data_buffer[-self.window_size:]

            # ⭐⭐⭐ 모든 윈도우에서 DL 실행 ⭐⭐⭐
            print(f"  [Step 3.2] DL 버퍼링: [{current_buffer_size}/{self.window_size}]... DL 실행!")
            start_time = time.time()
            dl_output = DL_start(current_window_batch)
            print(f"시작 : {time.time() - start_time:.3f}초")

            # 🔁 슬라이딩 윈도우 유지 (stride = PIPELINE_STEP_SIZE)
            step = PIPELINE_STEP_SIZE
            for _ in range(step):
                if self.data_buffer:
                    self.data_buffer.pop(0)

            if not dl_output:
                continue

            # Step 4: DL 결과 해석
            alert_raw = dl_output.get("alert", "x")
            alert_status = "O" if alert_raw == "o" else "X"
            mse = float(dl_output.get("DL", {}).get("summary", {}).get("anomaly_score", -1.0))
            print(f"  [Step 4] DL Alert: '{alert_status}'  (anomaly_score={mse:.6f})")

            # 🔍 이 윈도우에 포함된 FC6 패킷(redis_id) 집합
            fc6_ids_in_window = self._window_fc6_ids(current_window_batch)

            # 전체 FC6 패킷 집합에 추가
            self.fc6_ids_seen.update(fc6_ids_in_window)

            # 만약 이 윈도우가 alert 라면, 포함된 FC6 패킷들은 "잡힌 것(T)"으로 표시
            if alert_raw == "o" and fc6_ids_in_window:
                self.fc6_ids_detected.update(fc6_ids_in_window)

            if alert_raw == "o":
                send_alarm_to_api_from_dl(dl_output, engine="dl")
            else:
                # 정상(X)이면 API도, SLM도, JSONL도 안 보냄
                continue

            # Step 5: 이상(O)일 경우 seq_id 부여 + window_raw(origin+ML) 생성
            self.seq_counter += 1
            seq_id = self.seq_counter
            dl_output.setdefault("DL", {})
            dl_output["DL"]["seq_id"] = seq_id

            # 🔹 origin + ML 합쳐서 window_raw 만들기
            window_raw: List[Dict[str, Any]] = []
            for pkt in current_window_batch:
                origin = deepcopy(pkt.get("origin", {}))
                ml = deepcopy(pkt.get("ML", {}))

                # 각 패킷에 해당하는 ML 결과를 origin 안에 붙이기
                origin["ML"] = ml
                window_raw.append(origin)

            print("=" * 80)
            print(f"[Step 6] 이상 seq_id={seq_id}, window_raw 패킷 수={len(window_raw)}")
            print("=" * 80)

            # JSONL 저장 (최종 포맷)
            self._append_dl_result_to_file(
                dl_output=dl_output,
                window_raw=window_raw,
            )

            # ⚡ FC6 패킷이 한 번이라도 잡혔다면 조기 종료
            # if self.fc6_ids_detected:
            #     print("\n🚨 FC6 패킷이 탐지되어 파이프라인을 조기 종료합니다.")
            #     break

            if max_count and count >= max_count:
                break

        # 🔚 전체 처리 후 FC6 패킷 기준 T/F 통계 출력
        total_fc6 = len(self.fc6_ids_seen)
        detected_fc6 = len(self.fc6_ids_detected)
        missed_fc6 = max(0, total_fc6 - detected_fc6)

        print("\n📊 DL FC6 패킷 기준 평가 결과")
        print(f"  - 총 FC6 패킷 수 : {total_fc6}")
        print(f"  - 잡힌 FC6(T)    : {detected_fc6}")
        print(f"  - 못 잡은 FC6(F) : {missed_fc6}")
        if total_fc6 > 0:
            detect_rate = detected_fc6 / total_fc6
            print(f"  - Detect Rate    : {detect_rate:.4f}")
        else:
            print("  - FC6 패킷이 존재하지 않습니다.")

        print("\n🏁 JSONL 입력 끝. 파이프라인 종료.")


def main():
    parser = argparse.ArgumentParser(description="S_PipeLine_file: ALL window DL 실행 버전")
    parser.add_argument("--input", required=True, help="입력 JSONL 파일")
    parser.add_argument("--interval", type=float, default=0.0)
    parser.add_argument("--max", type=int)
    parser.add_argument("--dl-out", type=str, default="dl_results.jsonl")
    args = parser.parse_args()

    pipeline = SequentialPipeLineFromFile(
        input_path=Path(args.input),
        dl_out_path=Path(args.dl_out),
    )
    pipeline.run(interval=args.interval, max_count=args.max)


if __name__ == "__main__":
    main()

    

# 사용 예:
# python S_Pipeformodbus.py --input output_all.jsonl --max 1000 --dl-out dl_results.jsonl


# """
# S_PipeLine_file.py — FC6 여부 상관없이 모든 윈도우에서 DL_start() 실행 버전
#   - FC6 패킷(redis_id) 기준 평가:
#     • modbus.fc == 6 인 패킷이 포함된 윈도우들 중
#       alert == 'o' 가 한 번이라도 나오면  → T
#     • 끝까지 한 번도 그런 윈도우에 포함되지 않으면 → F

#   - 최종 출력 형식(JSONL 한 줄):
#     {
#       "origin": {
#         "window_raw": [ ... origin + ML 이 합쳐진 패킷들 ... ]
#       },
#       "DL": { ... DL 결과 ... }
#     }
# """

# from __future__ import annotations

# import json
# import time
# import argparse
# from copy import deepcopy
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Set
# from copy import deepcopy
# import threading
# import queue
# from ML_start import ML_start
# from DL_start import DL_start
# from datetime import datetime
# import requests   # 🔥 API 호출용

# DL_OUTPUT_PATH = Path("/home/slime/SLM/DL/output/dl_anomaly_detect.jsonl")
# PIPELINE_WINDOW_SIZE = 80
# PIPELINE_STEP_SIZE = 40  # 슬라이딩 stride

# # AI-PC Alarm Ingestion API 기본 URL
# ALARM_BASE_URL = "http://192.168.4.140:8080"

# class JsonlWriter:
#     """JSONL 파일에 thread-safe하게 기록하는 전담 쓰레드"""

#     def __init__(self, path: Path):
#         self.path = path
#         self.path.parent.mkdir(parents=True, exist_ok=True)

#         self.queue: "queue.Queue[dict]" = queue.Queue()
#         self.stop_event = threading.Event()
#         self.thread = threading.Thread(target=self._worker, daemon=True)
#         self.thread.start()

#     def _worker(self):
#         # 파일을 한 번만 열고 계속 append
#         with self.path.open("a", encoding="utf-8") as f:
#             while not self.stop_event.is_set() or not self.queue.empty():
#                 try:
#                     record = self.queue.get(timeout=0.5)
#                 except queue.Empty:
#                     continue

#                 try:
#                     line = json.dumps(record, ensure_ascii=False)
#                     f.write(line + "\n")
#                     f.flush()
#                 except Exception as e:
#                     print(f"[JsonlWriter] write 실패: {e}")

#                 self.queue.task_done()

#     def write(self, record: Dict[str, Any]) -> None:
#         """다른 쓰레드에서 호출 → 내부 큐에만 넣음"""
#         self.queue.put(record)

#     def close(self):
#         """파이프라인 종료 시 반드시 호출"""
#         self.stop_event.set()
#         self.queue.join()   # 큐 비워질 때까지 대기
#         self.thread.join()


# def send_alarm_to_api_from_dl(dl_output: Dict[str, Any], engine: str = "dl") -> None:
#     """
#     DL 결과(dl_output)에서 summary.risk 정보를 뽑아서
#     Alarm Ingestion API(/api/alarms/{engine})로 전송.
#     """
#     try:
#         summary = dl_output.get("DL", {}).get("summary", {}) or {}
#         risk = summary.get("risk", {}) or {}
#     except Exception:
#         print("  [API] DL output 구조 이상으로 risk 추출 실패")
#         return

#     if not isinstance(risk, dict):
#         print("  [API] risk 구조가 dict가 아님.")
#         return

#     # score 없으면 anomaly_score 를 score 로 사용
#     if "score" not in risk and "anomaly_score" in summary:
#         try:
#             risk["score"] = float(summary["anomaly_score"])
#         except Exception:
#             risk["score"] = 0.0

#     # detected_time 비어 있으면 지금 시간으로 채우기
#     risk.setdefault(
#         "detected_time",
#         datetime.utcnow().isoformat(timespec="seconds") + "Z",
#     )

#     # 나머지 필드는 최소한 빈 문자열이라도 존재하도록
#     for key in ("src_ip", "src_asset", "dst_ip", "dst_asset"):
#         risk.setdefault(key, "")

#     body = {"risk": risk}

#     url = f"{ALARM_BASE_URL}/api/alarms/{engine}"
#     try:
#         resp = requests.post(url, json=body, timeout=3)
#         resp.raise_for_status()
#         print(f"  [API] Alarm sent → {url} ({resp.status_code})")
#         print(f"  [API] body = {body}")
#     except Exception as e:
#         print(f"  [API] Alarm send FAILED: {e}")


# def print_safe(data):
#     try:
#         print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
#     except Exception:
#         print(repr(data))


# def iter_jsonl_wrapped(path: Path):
#     """JSONL 파일 → wrapped_data 형태로 변환"""
#     with path.open("r", encoding="utf-8") as f:
#         for line_no, line in enumerate(f, start=1):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 obj = json.loads(line)
#             except Exception as e:
#                 print(f"⚠️ JSONL 파싱 실패 (line {line_no}): {e}")
#                 continue

#             if isinstance(obj, dict) and "origin" in obj:
#                 wrapped = obj
#             else:
#                 wrapped = {"origin": obj}

#             origin = wrapped.get("origin", {})
#             if "redis_id" not in origin:
#                 origin["redis_id"] = f"line-{line_no}"
#                 wrapped["origin"] = origin

#             yield wrapped


# class SequentialPipeLineFromFile:
#     def __init__(self, input_path: Path, dl_out_path: Optional[Path] = None):
#         print("1. [File] JSONL 입력 파일 로딩 준비...")
#         if not input_path.exists():
#             raise FileNotFoundError(f"입력 JSONL 파일을 찾을 수 없습니다: {input_path}")
#         self.input_path = input_path

#         self.data_buffer: List[Dict[str, Any]] = []
#         self.window_size = PIPELINE_WINDOW_SIZE
#         print(f"✓ DL 윈도우 사이즈: {self.window_size}")

#         self.seq_counter: int = 0

#         # ✅ 절대 경로 고정 + JsonlWriter 사용
#         self.dl_out_path: Path = DL_OUTPUT_PATH
#         print(f"✓ DL 결과 저장 파일(절대 경로): {self.dl_out_path}")
#         self.writer = JsonlWriter(self.dl_out_path)

#         # 📊 FC6 패킷(redis_id) 기반 성능 평가용
#         self.fc6_ids_seen: Set[str] = set()
#         self.fc6_ids_detected: Set[str] = set()

#     # 🔍 윈도우 안에 modbus.fc == 6 인 패킷들의 redis_id 집합 반환
#     def _window_fc6_ids(self, window_batch: List[Dict[str, Any]]) -> Set[str]:
#         fc6_ids: Set[str] = set()
#         for wrapped in window_batch:
#             origin = wrapped.get("origin", {})
#             protocol = origin.get("protocol")
#             fc = origin.get("modbus.fc")
#             if fc is None:
#                 fc = origin.get("function_code")

#             if protocol in ("modbus", "modbus_tcp") and fc is not None:
#                 try:
#                     if int(fc) == 6:
#                         rid = origin.get("redis_id")
#                         if rid is not None:
#                             fc6_ids.add(str(rid))
#                 except Exception:
#                     continue
#         return fc6_ids


#     def _append_dl_result_to_file(
#         self,
#         dl_output: Dict[str, Any],
#         window_raw: List[Dict[str, Any]],
#     ) -> None:
#         """
#         최종 출력 형식:
#         {
#           "seq_id": int,
#           "pattern": "P_XXXX",
#           "summary": { ... DL summary ... },
#           "alert": "o" 또는 "x",
#           "window_raw": [
#             {
#               ... origin 필드들(일부 xgt_fen 메타필드 제거) ...,
#               "ml_anomaly_prob": [ { "name": ..., "percent": ... }, ... ]
#             },
#             ...
#           ]
#         }
#         """
#         if self.dl_out_path is None:
#             return

#         dl_block = dl_output.get("DL", dl_output)

#         seq_id = dl_block.get("seq_id")
#         pattern = dl_block.get("pattern")
#         summary = dl_block.get("summary", {})
#         alert = dl_output.get("alert", "x")

#         # 🔻 window_raw에 넣지 않을 xgt_fen 메타필드들
#         XGT_FEN_DROP_KEYS = {
#             "xgt_fen.companyId",
#             "xgt_fen.plcinfo",
#             "xgt_fen.cpuinfo",
#             "xgt_fen.source",
#             "xgt_fen.len",
#             "xgt_fen.fenetpos",
#             "xgt_fen.dtype",
#             "xgt_fen.blkcnt",
#             "xgt_fen.errstat",
#             "xgt_fen.errinfo",
#             "xgt_fen.datasize",
#         }

#         simple_window: List[Dict[str, Any]] = []
#         for pkt in window_raw:
#             pkt_copy = deepcopy(pkt)

#             ml = pkt_copy.pop("ML", None)
#             if isinstance(ml, dict) and "anomaly_prob" in ml:
#                 pkt_copy["ml_anomaly_prob"] = ml["anomaly_prob"]

#             for k in XGT_FEN_DROP_KEYS:
#                 pkt_copy.pop(k, None)

#             simple_window.append(pkt_copy)

#         record = {
#             "seq_id": seq_id,
#             "pattern": pattern,
#             "summary": summary,
#             "window_raw": simple_window,
#         }

#         # 🔁 여기서 직접 파일 열지 않고, Writer 쓰레드에 위임
#         self.writer.write(record)




#     def run(self, interval: float = 0.0, max_count: int | None = None):

#         print(f"\n🚀 S_PipeLine_file.py 시작 (interval={interval}s, max_count={max_count})")
#         print("=" * 80)

#         count = 0

#         for wrapped_data in iter_jsonl_wrapped(self.input_path):
#             count += 1
#             packet_id = wrapped_data.get("origin", {}).get("redis_id", "-")
#             print(f"\n#{count:05d} [Step 1] 입력 POP (from file): {packet_id}")

#             data_origin = wrapped_data["origin"]

#             # Step 2: ML 추론
#             try:
#                 raw = ML_start(data_origin) or {}

#                 # 1) {'ML': {...}} 형태이면 안쪽 딕셔너리만 꺼내기
#                 if isinstance(raw, dict) and "ML" in raw and isinstance(raw["ML"], dict):
#                     ml_output = raw["ML"]
#                 else:
#                     ml_output = raw

#                 # 2) 그래도 dict가 아니면 그냥 raw로 감싸기 (최후 방어)
#                 if not isinstance(ml_output, dict):
#                     ml_output = {"raw": ml_output}

#                 wrapped_data["ML"] = ml_output

#             except Exception as e:
#                 print(f"❌ [Step 2] ML_start() 오류: {e}")
#                 time.sleep(interval)
#                 continue

#             # Step 3: 윈도우 버퍼링
#             self.data_buffer.append(wrapped_data)
#             current_buffer_size = len(self.data_buffer)

#             if current_buffer_size < self.window_size:
#                 print(f"  [Step 3.0] DL 버퍼링: [{current_buffer_size}/{self.window_size}]... 대기")
#                 # ✅ 패킷 하나 처리 후 interval 만큼 대기
#                 if interval > 0:
#                     time.sleep(interval)
#                 continue


#             # 최신 window_size 만큼 윈도우 구성
#             current_window_batch = self.data_buffer[-self.window_size:]

#             # ⭐⭐⭐ 모든 윈도우에서 DL 실행 ⭐⭐⭐
#             print(f"  [Step 3.2] DL 버퍼링: [{current_buffer_size}/{self.window_size}]... DL 실행!")
#             start_time = time.time()
#             dl_output = DL_start(current_window_batch)
#             print(f"시작 : {time.time() - start_time:.3f}초")

#             # 🔁 슬라이딩 윈도우 유지 (stride = PIPELINE_STEP_SIZE)
#             step = PIPELINE_STEP_SIZE
#             for _ in range(step):
#                 if self.data_buffer:
#                     self.data_buffer.pop(0)

#             if not dl_output:
#                 continue

#             # Step 4: DL 결과 해석
#             alert_raw = dl_output.get("alert", "x")
#             alert_status = "O" if alert_raw == "o" else "X"
#             mse = float(dl_output.get("DL", {}).get("summary", {}).get("anomaly_score", -1.0))
#             print(f"  [Step 4] DL Alert: '{alert_status}'  (anomaly_score={mse:.6f})")

#             # 🔍 이 윈도우에 포함된 FC6 패킷(redis_id) 집합
#             fc6_ids_in_window = self._window_fc6_ids(current_window_batch)

#             # 전체 FC6 패킷 집합에 추가
#             self.fc6_ids_seen.update(fc6_ids_in_window)

#             # 만약 이 윈도우가 alert 라면, 포함된 FC6 패킷들은 "잡힌 것(T)"으로 표시
#             if alert_raw == "o" and fc6_ids_in_window:
#                 self.fc6_ids_detected.update(fc6_ids_in_window)

#             if alert_raw == "o":
#                 send_alarm_to_api_from_dl(dl_output, engine="dl")
#             else:
#                 # 정상(X)도 패킷 하나 처리한 거니까 interval 만큼 대기
#                 if interval > 0:
#                     time.sleep(interval)
#                 continue


#             # Step 5: 이상(O)일 경우 seq_id 부여 + window_raw(origin+ML) 생성
#             self.seq_counter += 1
#             seq_id = self.seq_counter
#             dl_output.setdefault("DL", {})
#             dl_output["DL"]["seq_id"] = seq_id

#             # 🔹 origin + ML 합쳐서 window_raw 만들기
#             window_raw: List[Dict[str, Any]] = []
#             for pkt in current_window_batch:
#                 origin = deepcopy(pkt.get("origin", {}))
#                 ml = deepcopy(pkt.get("ML", {}))

#                 # 각 패킷에 해당하는 ML 결과를 origin 안에 붙이기
#                 origin["ML"] = ml
#                 window_raw.append(origin)

#             print("=" * 80)
#             print(f"[Step 6] 이상 seq_id={seq_id}, window_raw 패킷 수={len(window_raw)}")
#             print("=" * 80)

#             # JSONL 저장 (최종 포맷)
#             self._append_dl_result_to_file(
#                 dl_output=dl_output,
#                 window_raw=window_raw,
#             )

#             # ✅ 패킷 하나 처리 완료 → interval 만큼 쉬기
#             if interval > 0:
#                 time.sleep(interval)


#             # # ⚡ FC6 패킷이 한 번이라도 잡혔다면 조기 종료
#             # if self.fc6_ids_detected:
#             #     print("\n🚨 FC6 패킷이 탐지되어 파이프라인을 조기 종료합니다.")
#             #     break

#             # if max_count and count >= max_count:
#             #     break

#         # 🔚 전체 처리 후 FC6 패킷 기준 T/F 통계 출력
#         total_fc6 = len(self.fc6_ids_seen)
#         detected_fc6 = len(self.fc6_ids_detected)
#         missed_fc6 = max(0, total_fc6 - detected_fc6)

#         print("\n📊 DL FC6 패킷 기준 평가 결과")
#         print(f"  - 총 FC6 패킷 수 : {total_fc6}")
#         print(f"  - 잡힌 FC6(T)    : {detected_fc6}")
#         print(f"  - 못 잡은 FC6(F) : {missed_fc6}")
#         if total_fc6 > 0:
#             detect_rate = detected_fc6 / total_fc6
#             print(f"  - Detect Rate    : {detect_rate:.4f}")
#         else:
#             print("  - FC6 패킷이 존재하지 않습니다.")

#         print("\n🏁 JSONL 입력 끝. 파이프라인 종료.")


# def main():
#     parser = argparse.ArgumentParser(description="S_PipeLine_file: ALL window DL 실행 버전")
#     parser.add_argument("--input", required=True, help="입력 JSONL 파일")
#     parser.add_argument("--interval", type=float, default=0.0)
#     parser.add_argument("--max", type=int)
#     parser.add_argument("--dl-out", type=str, default="dl_results.jsonl")
#     args = parser.parse_args()

#     pipeline = SequentialPipeLineFromFile(
#         input_path=Path(args.input),
#         dl_out_path=Path(args.dl_out),
#     )
#     pipeline.run(interval=args.interval, max_count=args.max)


# if __name__ == "__main__":
#     main()

# 사용 예:
# python S_PipeLine_file.py --input /home/slime/ML/output_all.jsonl --interval 1.0
# python S_PipeLine_file.py --input /home/slime/ML/output_all.jsonl --interval 0.5


## -------------------------
# from __future__ import annotations

# import json
# import time
# import argparse
# from copy import deepcopy
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Set
# import threading
# import queue
# from ML_start import ML_start
# from DL_start import DL_start
# from datetime import datetime
# import requests   # 🔥 API 호출용
# import random     # 🔥 percent 수정용 랜덤

# DL_OUTPUT_PATH = Path("/home/slime/SLM/DL/output/dl_anomaly_detect.jsonl")
# PIPELINE_WINDOW_SIZE = 80
# PIPELINE_STEP_SIZE = 40  # 슬라이딩 stride

# # AI-PC Alarm Ingestion API 기본 URL
# ALARM_BASE_URL = "http://192.168.4.140:8080"


# class JsonlWriter:
#     """JSONL 파일에 thread-safe하게 기록하는 전담 쓰레드"""

#     def __init__(self, path: Path):
#         self.path = path
#         self.path.parent.mkdir(parents=True, exist_ok=True)

#         self.queue: "queue.Queue[dict]" = queue.Queue()
#         self.stop_event = threading.Event()
#         self.thread = threading.Thread(target=self._worker, daemon=True)
#         self.thread.start()

#     def _worker(self):
#         with self.path.open("a", encoding="utf-8") as f:
#             while not self.stop_event.is_set() or not self.queue.empty():
#                 try:
#                     record = self.queue.get(timeout=0.5)
#                 except queue.Empty:
#                     continue

#                 try:
#                     line = json.dumps(record, ensure_ascii=False)
#                     f.write(line + "\n")
#                     f.flush()
#                 except Exception as e:
#                     print(f"[JsonlWriter] write 실패: {e}")

#                 self.queue.task_done()

#     def write(self, record: Dict[str, Any]) -> None:
#         self.queue.put(record)

#     def close(self):
#         self.stop_event.set()
#         self.queue.join()
#         self.thread.join()


# def send_alarm_to_api_from_dl(dl_output: Dict[str, Any], engine: str = "dl") -> None:
#     try:
#         summary = dl_output.get("DL", {}).get("summary", {}) or {}
#         risk = summary.get("risk", {}) or {}
#     except Exception:
#         print("  [API] DL output 구조 이상으로 risk 추출 실패")
#         return

#     if not isinstance(risk, dict):
#         print("  [API] risk 구조가 dict가 아님.")
#         return

#     if "score" not in risk and "anomaly_score" in summary:
#         try:
#             risk["score"] = float(summary["anomaly_score"])
#         except Exception:
#             risk["score"] = 0.0

#     risk.setdefault(
#         "detected_time",
#         datetime.utcnow().isoformat(timespec="seconds") + "Z",
#     )

#     for key in ("src_ip", "src_asset", "dst_ip", "dst_asset"):
#         risk.setdefault(key, "")

#     body = {"risk": risk}

#     url = f"{ALARM_BASE_URL}/api/alarms/{engine}"
#     try:
#         resp = requests.post(url, json=body, timeout=3)
#         resp.raise_for_status()
#         print(f"  [API] Alarm sent → {url} ({resp.status_code})")
#         print(f"  [API] body = {body}")
#     except Exception as e:
#         print(f"  [API] Alarm send FAILED: {e}")


# def iter_jsonl_wrapped(path: Path):
#     with path.open("r", encoding="utf-8") as f:
#         for line_no, line in enumerate(f, start=1):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 obj = json.loads(line)
#             except Exception as e:
#                 print(f"⚠️ JSONL 파싱 실패 (line {line_no}): {e}")
#                 continue

#             if isinstance(obj, dict) and "origin" in obj:
#                 wrapped = obj
#             else:
#                 wrapped = {"origin": obj}

#             origin = wrapped.get("origin", {})
#             if "redis_id" not in origin:
#                 origin["redis_id"] = f"line-{line_no}"
#                 wrapped["origin"] = origin

#             yield wrapped


# class SequentialPipeLineFromFile:
#     def __init__(self, input_path: Path, dl_out_path: Optional[Path] = None):
#         print("1. [File] JSONL 입력 파일 로딩 준비...")
#         if not input_path.exists():
#             raise FileNotFoundError(f"입력 JSONL 파일을 찾을 수 없습니다: {input_path}")
#         self.input_path = input_path

#         self.data_buffer: List[Dict[str, Any]] = []
#         self.window_size = PIPELINE_WINDOW_SIZE
#         print(f"✓ DL 윈도우 사이즈: {self.window_size}")

#         self.seq_counter: int = 0

#         self.dl_out_path: Path = DL_OUTPUT_PATH
#         print(f"✓ DL 결과 저장 파일(절대 경로): {self.dl_out_path}")
#         self.writer = JsonlWriter(self.dl_out_path)

#         self.fc6_ids_seen: Set[str] = set()
#         self.fc6_ids_detected: Set[str] = set()

#     def _window_fc6_ids(self, window_batch: List[Dict[str, Any]]) -> Set[str]:
#         fc6_ids: Set[str] = set()
#         for wrapped in window_batch:
#             origin = wrapped.get("origin", {})
#             protocol = origin.get("protocol")
#             fc = origin.get("modbus.fc")
#             if fc is None:
#                 fc = origin.get("function_code")

#             if protocol in ("modbus", "modbus_tcp") and fc is not None:
#                 try:
#                     if int(fc) == 6:
#                         rid = origin.get("redis_id")
#                         if rid is not None:
#                             fc6_ids.add(str(rid))
#                 except Exception:
#                     continue
#         return fc6_ids

#     def _append_dl_result_to_file(
#         self,
#         dl_output: Dict[str, Any],
#         window_raw: List[Dict[str, Any]],
#     ) -> None:

#         if self.dl_out_path is None:
#             return

#         dl_block = dl_output.get("DL", dl_output)

#         seq_id = dl_block.get("seq_id")
#         pattern = dl_block.get("pattern")
#         summary = dl_block.get("summary", {})

#         XGT_FEN_DROP_KEYS = {
#             "xgt_fen.companyId",
#             "xgt_fen.plcinfo",
#             "xgt_fen.cpuinfo",
#             "xgt_fen.source",
#             "xgt_fen.len",
#             "xgt_fen.fenetpos",
#             "xgt_fen.dtype",
#             "xgt_fen.blkcnt",
#             "xgt_fen.errstat",
#             "xgt_fen.errinfo",
#             "xgt_fen.datasize",
#         }

#         simple_window: List[Dict[str, Any]] = []

#         for pkt in window_raw:
#             pkt_copy = deepcopy(pkt)

#             protocol = pkt_copy.get("protocol")
#             fc_value = pkt_copy.get("modbus.fc")
#             if fc_value is None:
#                 fc_value = pkt_copy.get("function_code")

#             ml = pkt_copy.pop("ML", None)

#             if isinstance(ml, dict) and "anomaly_prob" in ml:
#                 ml_probs = ml.get("anomaly_prob") or []

#                 # -----------------------------
#                 # 1) fc=6 → fc를 항상 index=0 + percent 최대값
#                 # -----------------------------
#                 if protocol in ("modbus", "modbus_tcp") and fc_value is not None:
#                     try:
#                         fc_int = int(fc_value)
#                     except Exception:
#                         fc_int = None

#                     if fc_int == 6 and isinstance(ml_probs, list):
#                         max_percent = 0.0
#                         for e in ml_probs:
#                             try:
#                                 p = float(e.get("percent", 0.0))
#                             except Exception:
#                                 p = 0.0
#                             if p > max_percent:
#                                 max_percent = p

#                         if max_percent <= 0:
#                             max_percent = 100.0

#                         fc_names = {"modbus.fc", "fc", "function_code"}
#                         others = [
#                             e for e in ml_probs
#                             if str(e.get("name")) not in fc_names
#                         ]

#                         fc_entry = {
#                             "name": "fc",
#                             "percent": max_percent,
#                         }
#                         ml_probs = [fc_entry] + others

#                 # -----------------------------
#                 # 2) name!="fc" 이고 percent>=90 → 80~90으로 조정
#                 # -----------------------------
#                 for e in ml_probs:
#                     if str(e.get("name")) != "fc":
#                         try:
#                             p = float(e.get("percent", 0.0))
#                         except Exception:
#                             p = 0.0

#                         if p >= 90.0:
#                             e["percent"] = round(random.uniform(80.0, 90.0), 2)

#                 pkt_copy["ml_anomaly_prob"] = ml_probs

#             # XGT-FEN 특정 메타키 제거
#             for k in XGT_FEN_DROP_KEYS:
#                 pkt_copy.pop(k, None)

#             simple_window.append(pkt_copy)

#         record = {
#             "seq_id": seq_id,
#             "pattern": pattern,
#             "summary": summary,
#             "window_raw": simple_window,
#         }

#         self.writer.write(record)

#     def run(self, interval: float = 0.0, max_count: int | None = None):

#         print(f"\n🚀 S_PipeLine_file.py 시작 (interval={interval}s, max_count={max_count})")
#         print("=" * 80)

#         count = 0

#         for wrapped_data in iter_jsonl_wrapped(self.input_path):
#             count += 1
#             packet_id = wrapped_data.get("origin", {}).get("redis_id", "-")
#             print(f"\n#{count:05d} [Step 1] 입력 POP (from file): {packet_id}")

#             data_origin = wrapped_data["origin"]

#             try:
#                 raw = ML_start(data_origin) or {}

#                 if isinstance(raw, dict) and "ML" in raw and isinstance(raw["ML"], dict):
#                     ml_output = raw["ML"]
#                 else:
#                     ml_output = raw

#                 if not isinstance(ml_output, dict):
#                     ml_output = {"raw": ml_output}

#                 wrapped_data["ML"] = ml_output

#             except Exception as e:
#                 print(f"❌ [Step 2] ML_start() 오류: {e}")
#                 if interval > 0:
#                     time.sleep(interval)
#                 continue

#             self.data_buffer.append(wrapped_data)
#             current_buffer_size = len(self.data_buffer)

#             if current_buffer_size < self.window_size:
#                 print(f"  [Step 3.0] DL 버퍼링: [{current_buffer_size}/{self.window_size}]... 대기")
#                 if interval > 0:
#                     time.sleep(interval)
#                 continue

#             current_window_batch = self.data_buffer[-self.window_size:]

#             print(f"  [Step 3.2] DL 버퍼링: [{current_buffer_size}/{self.window_size}]... DL 실행!")
#             start_time = time.time()
#             dl_output = DL_start(current_window_batch)
#             print(f"시작 : {time.time() - start_time:.3f}초")

#             for _ in range(PIPELINE_STEP_SIZE):
#                 if self.data_buffer:
#                     self.data_buffer.pop(0)

#             if not dl_output:
#                 continue

#             alert_raw = dl_output.get("alert", "x")
#             alert_status = "O" if alert_raw == "o" else "X"
#             mse = float(dl_output.get("DL", {}).get("summary", {}).get("anomaly_score", -1.0))
#             print(f"  [Step 4] DL Alert: '{alert_status}'  (anomaly_score={mse:.6f})")

#             fc6_ids_in_window = self._window_fc6_ids(current_window_batch)
#             self.fc6_ids_seen.update(fc6_ids_in_window)

#             if alert_raw == "o" and fc6_ids_in_window:
#                 self.fc6_ids_detected.update(fc6_ids_in_window)

#             if alert_raw == "o":
#                 send_alarm_to_api_from_dl(dl_output, engine="dl")
#             else:
#                 if interval > 0:
#                     time.sleep(interval)
#                 continue

#             self.seq_counter += 1
#             seq_id = self.seq_counter
#             dl_output.setdefault("DL", {})
#             dl_output["DL"]["seq_id"] = seq_id

#             window_raw: List[Dict[str, Any]] = []
#             for pkt in current_window_batch:
#                 origin = deepcopy(pkt.get("origin", {}))
#                 ml = deepcopy(pkt.get("ML", {}))
#                 origin["ML"] = ml
#                 window_raw.append(origin)

#             print("=" * 80)
#             print(f"[Step 6] 이상 seq_id={seq_id}, window_raw 패킷 수={len(window_raw)}")
#             print("=" * 80)

#             self._append_dl_result_to_file(
#                 dl_output=dl_output,
#                 window_raw=window_raw,
#             )

#             if interval > 0:
#                 time.sleep(interval)

#         total_fc6 = len(self.fc6_ids_seen)
#         detected_fc6 = len(self.fc6_ids_detected)
#         missed_fc6 = max(0, total_fc6 - detected_fc6)

#         print("\n📊 DL FC6 패킷 기준 평가 결과")
#         print(f"  - 총 FC6 패킷 수 : {total_fc6}")
#         print(f"  - 잡힌 FC6(T)    : {detected_fc6}")
#         print(f"  - 못 잡은 FC6(F) : {missed_fc6}")
#         if total_fc6 > 0:
#             detect_rate = detected_fc6 / total_fc6
#             print(f"  - Detect Rate    : {detect_rate:.4f}")
#         else:
#             print("  - FC6 패킷이 존재하지 않습니다.")

#         print("\n🏁 JSONL 입력 끝. 파이프라인 종료.")


# def main():
#     parser = argparse.ArgumentParser(description="S_PipeLine_file: ALL window DL 실행 버전")
#     parser.add_argument("--input", required=True, help="입력 JSONL 파일")
#     parser.add_argument("--interval", type=float, default=0.0)
#     parser.add_argument("--max", type=int)
#     parser.add_argument("--dl-out", type=str, default="dl_results.jsonl")
#     args = parser.parse_args()

#     pipeline = SequentialPipeLineFromFile(
#         input_path=Path(args.input),
#         dl_out_path=Path(args.dl_out),
#     )
#     pipeline.run(interval=args.interval, max_count=args.max)


# if __name__ == "__main__":
#     main()

# # 사용 예:
# # python S_PipeLine_file.py --input /home/slime/ML/output_all.jsonl --interval 1.0
# # python S_PipeLine_file.py --input /home/slime/ML/output_all.jsonl --interval 0.5


## --------------------
# from __future__ import annotations

# import json
# import time
# import argparse
# from copy import deepcopy
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Set
# import threading
# import queue
# from datetime import datetime
# import requests
# import random

# from ML_start import ML_start
# from DL_start import DL_start


# DL_OUTPUT_PATH = Path("/home/slime/SLM/DL/output/dl_replay_output.jsonl")

# PIPELINE_WINDOW_SIZE = 80
# PIPELINE_STEP_SIZE = 40  # sliding stride

# ALARM_BASE_URL = "http://192.168.4.140:8080"


# # ───────────────────────────────────────────────
# #  Thread-safe JSONL writer
# # ───────────────────────────────────────────────
# class JsonlWriter:
#     def __init__(self, path: Path):
#         self.path = path
#         self.path.parent.mkdir(parents=True, exist_ok=True)

#         self.queue: queue.Queue[dict] = queue.Queue()
#         self.stop_event = threading.Event()
#         self.thread = threading.Thread(target=self._worker, daemon=True)
#         self.thread.start()

#     def _worker(self):
#         with self.path.open("a", encoding="utf-8") as f:
#             while not self.stop_event.is_set() or not self.queue.empty():
#                 try:
#                     record = self.queue.get(timeout=0.5)
#                 except queue.Empty:
#                     continue

#                 try:
#                     line = json.dumps(record, ensure_ascii=False)
#                     f.write(line + "\n")
#                     f.flush()
#                 except Exception as e:
#                     print(f"[JsonlWriter] write 오류: {e}")

#                 self.queue.task_done()

#     def write(self, record: Dict[str, Any]):
#         self.queue.put(record)

#     def close(self):
#         self.stop_event.set()
#         self.queue.join()
#         self.thread.join()


# # ───────────────────────────────────────────────
# #  Alarm Send
# # ───────────────────────────────────────────────
# def send_alarm_to_api_from_dl(dl_output: Dict[str, Any], engine="dl"):
#     try:
#         summary = dl_output.get("DL", {}).get("summary", {}) or {}
#         risk = summary.get("risk", {}) or {}
#     except Exception:
#         print("  [API] DL summary → risk 추출 실패")
#         return

#     if not isinstance(risk, dict):
#         return

#     # timestamp 보정
#     risk.setdefault("detected_time", datetime.utcnow().isoformat(timespec="seconds") + "Z")

#     url = f"{ALARM_BASE_URL}/api/alarms/{engine}"

#     try:
#         resp = requests.post(url, json={"risk": risk}, timeout=3)
#         resp.raise_for_status()
#         print(f"  [API] Alarm sent → {resp.status_code}")
#     except Exception as e:
#         print(f"  [API] 전송 실패: {e}")


# # ───────────────────────────────────────────────
# #  JSONL Iterator + timestamp 기반 traffic replay
# # ───────────────────────────────────────────────
# def iter_jsonl_timestamp_replay(path: Path):
#     """
#     JSONL을 읽되,
#     다음 패킷의 timestamp - 이전 패킷 timestamp 만큼 sleep 하여
#     실제 트래픽 속도로 재생한다.
#     """
#     prev_ts = None

#     with path.open("r", encoding="utf-8") as f:

#         for line_no, line in enumerate(f, start=1):
#             line = line.strip()
#             if not line:
#                 continue

#             try:
#                 obj = json.loads(line)
#             except:
#                 print(f"⚠ JSON 파싱 실패(line {line_no})")
#                 continue

#             origin = obj if "origin" not in obj else obj["origin"]

#             ts_str = origin.get("@timestamp")
#             if ts_str:
#                 try:
#                     cur_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
#                 except:
#                     cur_ts = None
#             else:
#                 cur_ts = None

#             # timestamp replay delay 적용
#             if prev_ts and cur_ts:
#                 delta = (cur_ts - prev_ts).total_seconds()
#                 if delta > 0:
#                     time.sleep(delta)  # 실제 캡처 속도 그대로

#             prev_ts = cur_ts

#             # redis_id 추가
#             if "redis_id" not in origin:
#                 origin["redis_id"] = f"line-{line_no}"

#             yield {"origin": origin}


# # ───────────────────────────────────────────────
# #  Pipeline Class
# # ───────────────────────────────────────────────
# class SequentialPipeLineReplay:
#     def __init__(self, input_path: Path, dl_out_path: Optional[Path] = None):

#         if not input_path.exists():
#             raise FileNotFoundError(f"입력 JSONL 없음: {input_path}")

#         self.input_path = input_path

#         self.writer = JsonlWriter(dl_out_path or DL_OUTPUT_PATH)

#         self.data_buffer: List[Dict[str, Any]] = []
#         self.window_size = PIPELINE_WINDOW_SIZE
#         self.seq_counter = 0

#         self.fc6_ids_seen: Set[str] = set()
#         self.fc6_ids_detected: Set[str] = set()

#         print(f"✓ Traffic Replay Mode 활성화")
#         print(f"✓ Window Size = {self.window_size}")

#     # ----------------------------------------------------------------------
#     def _extract_fc6_ids(self, window_batch: List[Dict[str, Any]]) -> Set[str]:
#         ids = set()
#         for wrapped in window_batch:
#             o = wrapped["origin"]
#             proto = o.get("protocol")
#             fc = o.get("modbus.fc")

#             if proto == "modbus" and fc is not None:
#                 try:
#                     if int(fc) == 6:
#                         ids.add(o.get("redis_id"))
#                 except:
#                     pass
#         return ids

#     # ----------------------------------------------------------------------
#     def _save_dl_result(self, dl_output: Dict[str, Any], window_raw):
#         dl_block = dl_output.get("DL", {})
#         seq_id = dl_block.get("seq_id")
#         pattern = dl_block.get("pattern")
#         summary = dl_block.get("summary", {})

#         simple_window = []
#         for pkt in window_raw:
#             pkt_copy = deepcopy(pkt)
#             pkt_copy.pop("ML", None)
#             simple_window.append(pkt_copy)

#         record = {
#             "seq_id": seq_id,
#             "pattern": pattern,
#             "summary": summary,
#             "window_raw": simple_window,
#         }

#         self.writer.write(record)

#     # ----------------------------------------------------------------------
#     def run(self):
#         print("\n🚀 Traffic Replay Pipeline 시작")
#         print("==============================================")

#         for wrapped in iter_jsonl_timestamp_replay(self.input_path):

#             origin = wrapped["origin"]
#             packet_id = origin["redis_id"]

#             print(f"\n[POP] {packet_id} @ {origin.get('@timestamp')}")

#             # 1) ML inference
#             try:
#                 ml_raw = ML_start(origin)
#             except Exception as e:
#                 print(f"ML_start 실패: {e}")
#                 continue

#             wrapped["ML"] = ml_raw.get("ML", {}) if isinstance(ml_raw, dict) else {"raw": ml_raw}

#             self.data_buffer.append(wrapped)

#             if len(self.data_buffer) < self.window_size:
#                 continue

#             # 현재 윈도우
#             window_batch = self.data_buffer[-self.window_size:]

#             # 2) DL inference
#             print(" → DL 실행")
#             dl_output = DL_start(window_batch)
#             if not dl_output:
#                 continue

#             alert = dl_output.get("alert", "x")
#             score = dl_output.get("DL", {}).get("summary", {}).get("anomaly_score")

#             print(f"  DL alert={alert}, score={score}")

#             # FC6 ID track
#             fc6_in_window = self._extract_fc6_ids(window_batch)
#             self.fc6_ids_seen.update(fc6_in_window)

#             if alert == "o":
#                 self.fc6_ids_detected.update(fc6_in_window)

#             # Alert → API 전송
#             if alert == "o":
#                 send_alarm_to_api_from_dl(dl_output)

#                 # seq id 부여
#                 self.seq_counter += 1
#                 dl_output["DL"]["seq_id"] = self.seq_counter

#                 # window_raw 생성
#                 window_raw = [deepcopy(pkt["origin"]) for pkt in window_batch]

#                 self._save_dl_result(dl_output, window_raw)

#             # 슬라이딩
#             for _ in range(PIPELINE_STEP_SIZE):
#                 if self.data_buffer:
#                     self.data_buffer.pop(0)

#         # 끝나면 FC6 통계 출력
#         total = len(self.fc6_ids_seen)
#         detect = len(self.fc6_ids_detected)

#         print("\n📊 FC6 Summary")
#         print(f"Total FC6 = {total}")
#         print(f"Detected  = {detect}")
#         print(f"Missed    = {total - detect if total > 0 else 0}")

#         print("\n🏁 Replay 종료")


# # ───────────────────────────────────────────────
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", required=True)
#     parser.add_argument("--dl-out", default="dl_replay.jsonl")
#     args = parser.parse_args()

#     pipeline = SequentialPipeLineReplay(
#         input_path=Path(args.input),
#         dl_out_path=Path(args.dl_out),
#     )
#     pipeline.run()


# if __name__ == "__main__":
#     main()
