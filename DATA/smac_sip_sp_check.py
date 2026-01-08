#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import defaultdict

def find_smac_conflicts(jsonl_path: str):
    smac_to_pairs = defaultdict(set)
    conflicts = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
            except json.JSONDecodeError:
                continue  # 잘못된 JSONL 라인은 건너뜀

            # 필요 필드 가져오기
            smac = obj.get("smac")
            sip = obj.get("sip")
            sp   = obj.get("sp")  # ✅ source port 추가

            # 유효성 체크
            if not smac or not sip or sp is None:
                continue

            # smac별 (sip, sp) 조합 저장
            smac_to_pairs[smac].add((sip, sp))

    # smac이 동일하지만 sip/sp 조합이 여러 개인 경우 탐지
    for smac, pairs in smac_to_pairs.items():
        if len(pairs) > 1:
            conflicts.append({
                "smac": smac,
                "pairs": list(pairs),
                "count": len(pairs)
            })

    # 결과 출력
    print("⚠️ 동일한 SMAC이 서로 다른 SIP 또는 SP를 가진 경우:")
    for item in conflicts:
        pair_str = ", ".join([f"{sip}:{sp}" for sip, sp in item["pairs"]])
        print(f"SMAC: {item['smac']} → {pair_str} (총 {item['count']}개 조합)")

    print(f"\n총 {len(conflicts)}개의 충돌이 탐지됨.")
    return conflicts


if __name__ == "__main__":
    find_smac_conflicts("./ML_DL 학습.jsonl")
