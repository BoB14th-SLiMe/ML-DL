#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_all_preprocess.py
각 프로토콜 전처리를 순차적으로 실행

"""
import subprocess
import argparse
from pathlib import Path
import sys
import json
import threading


SCRIPTS = [
    "common.py",
    "arp.py",
    "dns.py",
    "modbus.py",
    "s7comm.py",
    "xgt-fen.py",
    "preprocess_translated_addr_slot.py",
]


def _resolve_input_path(raw_input: str, base_dir: Path) -> Path:
    p = Path(raw_input).expanduser()

    if p.is_absolute():
        rp = p.resolve()
        if rp.exists():
            return rp
        raise FileNotFoundError(str(rp))

    cand0 = (Path.cwd() / p).resolve()
    if cand0.exists():
        return cand0

    cand1 = (base_dir / p).resolve()
    if cand1.exists():
        return cand1

    fname = p.name

    cand2 = (base_dir.parent / "data" / fname).resolve()
    if cand2.exists():
        return cand2

    cand3 = (base_dir.parent.parent / "data" / fname).resolve()
    if cand3.exists():
        return cand3

    raise FileNotFoundError(str(cand0))


def _quick_jsonl_sanity_check(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.stat().st_size == 0:
        raise ValueError(f"입력 파일이 비어있음: {path}")

    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for _ in range(50):
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue
            json.loads(s)
            return

    raise ValueError(f"입력 JSONL에서 유효한 JSON object 라인을 찾지 못함: {path}")


def run_cmd(name: str, cmd: list[str]) -> int:
    print(f"\n[▶] 실행 중: {name}")
    print(subprocess.list2cmdline(cmd))

    def _reader(pipe, is_err: bool):
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                prefix = f"[{name}][ERR] " if is_err else f"[{name}] "
                print(prefix + line.rstrip("\n"))
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    try:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        t_out = threading.Thread(target=_reader, args=(p.stdout, False), daemon=True)
        t_err = threading.Thread(target=_reader, args=(p.stderr, True), daemon=True)
        t_out.start()
        t_err.start()

        rc = p.wait()  # ✅ 중간 "동작중" 표기 없이 종료까지 대기

        t_out.join(timeout=1)
        t_err.join(timeout=1)

        if rc == 0:
            print(f"[✅] 완료: {name}")
        else:
            print(f"[❌] 실패: {name} (exit code {rc})")
        return int(rc)

    except Exception as e:
        print(f"[❌] 실패: {name} (unexpected error: {e})")
        return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    input_path = _resolve_input_path(args.input, base_dir)
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 사용 파이썬: {sys.executable}")
    print(f"[INFO] 입력 파일: {input_path}")
    print(f"[INFO] 출력 디렉토리: {output_dir}")

    _quick_jsonl_sanity_check(input_path)

    python_cmd = sys.executable
    jobs: list[tuple[str, list[str]]] = []

    for script_name in SCRIPTS:
        script_path = base_dir / script_name
        if not script_path.exists():
            print(f"[⚠] {script_name} 없음, 건너뜀")
            continue

        cmd = [
            python_cmd, "-u",
            str(script_path),
            "--fit",
            "-i", str(input_path),
            "-o", str(output_dir),
        ]
        jobs.append((script_name, cmd))

    if not jobs:
        print("[WARN] 실행할 스크립트가 없습니다.")
        return

    results: dict[str, int] = {}
    for name, cmd in jobs:
        results[name] = int(run_cmd(name, cmd))

    print("\n================ 실행 결과 요약 ================")
    for name in SCRIPTS:
        if name not in results:
            print(f"{name:30s} : NOT RUN")
            continue
        code = results[name]
        print(f"{name:30s} : {'OK' if code == 0 else f'FAIL({code})'}")
    print("================================================")


if __name__ == "__main__":
    main()
