#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_all_preprocess.py
----------------------------------------
각 프로토콜 전처리 스크립트를 한 번에 (병렬로) 실행하는 통합 런처.

- 현재 실행 중인 파이썬(sys.executable)을 그대로 사용하므로
  Windows / Linux 상관없이 python3 / python 문제 없이 동작함.
- ThreadPoolExecutor로 병렬 실행.
"""

import subprocess
import argparse
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# 한 번에 돌릴 스크립트 리스트
SCRIPTS = [
    "common.py",
    "arp.py",
    "dns.py",
    "modbus.py",
    "s7comm.py",
    "xgt-fen.py",
    # translated_addr slot feature
    "preprocess_translated_addr_slot.py",
]


def run_script(script_path: Path, input_path: Path, output_dir: Path, mode: str) -> int:
    """
    하나의 전처리 스크립트를 subprocess로 실행하고, exit code 반환

    - 일반 스크립트: 1번 실행
    - preprocess_translated_addr_slot.py: modbus / xgt_fen 두 번 실행
    """
    python_cmd = sys.executable  # 지금 이 파일을 실행 중인 파이썬 그대로 사용

    # 공통 인자(--fit/--transform, -i, -o)까지는 동일
    base_cmd = [
        python_cmd,
        str(script_path),
        f"--{mode}",
        "-i", str(input_path),
        "-o", str(output_dir),
    ]

    cmds = []

    # 🔹 translated_addr slot 스크립트는 modbus / xgt_fen 두 번 실행
    if script_path.name == "preprocess_translated_addr_slot.py":
        for proto in ("modbus", "xgt_fen"):
            cmds.append(base_cmd + ["-P", proto])
    else:
        # 그 외 스크립트는 한 번만 실행
        cmds.append(base_cmd)

    last_returncode = 0

    for cmd in cmds:
        print(f"\n[▶] 실행 중: {script_path.name}")
        print(" ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
            print(f"[✅] 완료: {' '.join(cmd)}")
        except subprocess.CalledProcessError as e:
            print(f"[❌] 실패: {' '.join(cmd)} (exit code {e.returncode})")
            last_returncode = e.returncode
            # 한 번이라도 실패하면 바로 중단
            return last_returncode
        except Exception as e:
            print(f"[❌] 실패: {' '.join(cmd)} (unexpected error: {e})")
            return -1

    return last_returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="입력 JSONL 경로")
    parser.add_argument("--output", "-o", required=True, help="출력 디렉토리 경로")
    parser.add_argument(
        "--mode",
        "-m",
        default="fit",
        choices=["fit", "transform"],
        help="실행 모드 (--fit / --transform)",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="건너뛸 스크립트 이름 (예: --skip dns.py modbus.py)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="병렬 실행 worker 수 (0이면 CPU 개수 기반 자동 설정)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(__file__).parent

    print(f"[INFO] 사용 파이썬: {sys.executable}")

    # worker 수 결정
    if args.workers > 0:
        max_workers = args.workers
    else:
        # 너무 많지 않게 최소 2, 최대 스크립트 수 / CPU 개수 내로
        cpu_cnt = os.cpu_count() or 4
        max_workers = min(len(SCRIPTS), max(2, cpu_cnt))
    print(f"[INFO] 병렬 worker 수: {max_workers}")

    # 실행할 작업들만 모으기
    tasks = []
    for script_name in SCRIPTS:
        if script_name in args.skip:
            print(f"[⏭] {script_name} 건너뜀 (--skip 지정)")
            continue

        script_path = base_dir / script_name
        if not script_path.exists():
            print(f"[⚠] {script_name} 없음, 건너뜀")
            continue

        tasks.append((script_name, script_path))

    if not tasks:
        print("[WARN] 실행할 스크립트가 없습니다.")
        return

    # 병렬 실행
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(run_script, script_path, input_path, output_dir, args.mode): name
            for (name, script_path) in tasks
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                code = future.result()
            except Exception as e:
                print(f"[❌] {name} 실행 중 예외 발생: {e}")
                code = -1
            results[name] = code

    print("\n================ 실행 결과 요약 ================")
    for name in SCRIPTS:
        if name in args.skip:
            print(f"{name:10s} : SKIPPED")
            continue
        if name not in results:
            print(f"{name:10s} : NOT RUN")
            continue
        code = results[name]
        if code == 0:
            status = "OK"
        else:
            status = f"FAIL({code})"
        print(f"{name:10s} : {status}")
    print("================================================")


if __name__ == "__main__":
    main()

"""
python 2.run_all_preprocess.py --input "../data/ML_DL 학습.jsonl" --output "../result" --mode fit --skip dns.py modbus.py
"""
