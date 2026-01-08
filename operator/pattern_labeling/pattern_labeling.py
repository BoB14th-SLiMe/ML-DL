#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
attach_pattern_by_order.py

순서(라인 번호) 기준 매핑:
  attack_ver*.jsonl 의 i번째 레코드에
  windows_pred*.jsonl 의 i번째 레코드의 pattern을 붙임.

출력(JSONL):
  {"pattern": "<pred_pattern>", "data": { ...attack_record... }}

사용:
  python attach_pattern_by_order.py --attack attack_ver2.jsonl --pred windows_pred_attack_ver2.jsonl --out result_attack_ver2.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

JsonDict = Dict[str, Any]


def read_jsonl(path: Path) -> Iterable[JsonDict]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                raise ValueError(f"JSON parse error at {path}:{ln}: {e}") from e


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pred_pattern(rec: Optional[JsonDict]) -> str:
    if not isinstance(rec, dict):
        return ""
    p = rec.get("pattern")
    return p if isinstance(p, str) else ""


def derive_out_path(attack_path: Path) -> Path:
    name = attack_path.name
    if "attack_ver" in name:
        name = name.replace("attack_ver", "result_attack", 1)
    else:
        name = f"result_{attack_path.stem}.jsonl"
    return attack_path.parent / name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", required=True, type=Path, help="attack_ver*.jsonl")
    ap.add_argument("--pred", required=True, type=Path, help="windows_pred*.jsonl")
    ap.add_argument("--out", type=Path, default=None, help="result_attack*.jsonl (default auto)")
    args = ap.parse_args()

    attack_path: Path = args.attack
    pred_path: Path = args.pred
    out_path: Path = args.out if args.out else derive_out_path(attack_path)

    attack_iter = read_jsonl(attack_path)
    pred_iter = read_jsonl(pred_path)

    total = 0
    matched = 0
    empty_pattern = 0
    extra_attack = 0
    extra_pred = 0

    def gen() -> Iterable[JsonDict]:
        nonlocal total, matched, empty_pattern, extra_attack, extra_pred
        for a, p in zip_longest(attack_iter, pred_iter, fillvalue=None):
            if a is None and p is not None:
                # pred가 더 김: attack이 없으니 무시
                extra_pred += 1
                continue
            if a is not None and p is None:
                # attack이 더 김: pattern 없는 채로 출력
                extra_attack += 1
                pat = ""
                empty_pattern += 1
                total += 1
                yield {"pattern": pat, "data": a}
                continue

            # 정상 1:1
            total += 1
            pat = pred_pattern(p)
            if pat:
                matched += 1
            else:
                empty_pattern += 1
            yield {"pattern": pat, "data": a}

    write_jsonl(out_path, gen())

    print(f"[DONE] out={out_path}", file=sys.stderr)
    print(
        f"[STAT] total_out={total} matched_pattern={matched} empty_pattern={empty_pattern} "
        f"extra_attack_lines={extra_attack} extra_pred_lines_ignored={extra_pred}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

"""
python pattern_labeling.py --pred ./attack_ver2_window.jsonl --attack "./windows_pred(attack_ver2).jsonl" --out ./result_attack_ver2.jsonl
python pattern_labeling.py --pred ./attack_ver5_window.jsonl --attack "./windows_pred(attack_ver5).jsonl" --out ./result_attack_ver5.jsonl
python pattern_labeling.py --pred ./attack_ver5_1_window.jsonl --attack "./windows_pred(attack_ver5_1).jsonl" --out ./result_attack_ver5_1.jsonl
python pattern_labeling.py --pred ./attack_ver5_2_window.jsonl --attack "./windows_pred(attack_ver5_2).jsonl" --out ./result_attack_ver5_2.jsonl
python pattern_labeling.py --pred ./attack_ver11_window.jsonl --attack "./windows_pred(attack_ver11).jsonl" --out ./result_attack_ver11.jsonl

"""