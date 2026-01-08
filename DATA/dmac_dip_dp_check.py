#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import defaultdict

def find_dmac_conflicts(jsonl_path: str):
    dmac_to_pairs = defaultdict(set)
    conflicts = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
            except json.JSONDecodeError:
                continue  # 잘못된 JSONL 라인은 건너뜀

            # 필요 필드 가져오기
            dmac = obj.get("dmac")
            dip = obj.get("dip")
            dp   = obj.get("dp")  # ✅ source port 추가

            # 유효성 체크
            if not dmac or not dip or dp is None:
                continue

            # dmac별 (dip, dp) 조합 저장
            dmac_to_pairs[dmac].add((dip, dp))

    # dmac이 동일하지만 dip/dp 조합이 여러 개인 경우 탐지
    for dmac, pairs in dmac_to_pairs.items():
        if len(pairs) > 1:
            conflicts.append({
                "dmac": dmac,
                "pairs": list(pairs),
                "count": len(pairs)
            })

    # 결과 출력
    print("⚠️ 동일한 DMAC이 서로 다른 DIP 또는 SP를 가진 경우:")
    for item in conflicts:
        pair_str = ", ".join([f"{dip}:{dp}" for dip, dp in item["pairs"]])
        print(f"DMAC: {item['dmac']} → {pair_str} (총 {item['count']}개 조합)")

    print(f"\n총 {len(conflicts)}개의 충돌이 탐지됨.")
    return conflicts


if __name__ == "__main__":
    find_dmac_conflicts("./ML_DL 학습.jsonl")
