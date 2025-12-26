"""
file_loader.py

다음 파일을 로드하는 함수
  - json
  - jsonl

input
  - file_type : 파일 종류
  - file_path : 파일 경로
  
"""
import re
from collections import Counter

import re
from typing import Optional, Dict, Any

_TS = re.compile(r'timestamp[=:]"?([0-9T:\.\-Z\+]+)"?')
_SQ = re.compile(r'\bsq[=:]"?([^"\s]+)"?')
_AK = re.compile(r'\bak[=:]"?([^"\s]+)"?')
_FL = re.compile(r'\bfl[=:]"?([^"\s]+)"?')

def pls_extract(pls: Any) -> Optional[Dict[str, str]]:
    # 핵심: string 아니면 스킵(혼합 타입 때문에 죽는 문제 해결)
    if not isinstance(pls, str):
        return None

    ts = _TS.search(pls)
    sq = _SQ.search(pls)
    ak = _AK.search(pls)
    fl = _FL.search(pls)

    if not (ts and sq and ak and fl):
        return None

    return {
        "@timestamp": ts.group(1),   # RAW는 @timestamp를 쓰므로 통일
        "sq": sq.group(1),
        "ak": ak.group(1),
        "fl": fl.group(1),
    }


def raw_extract(RAW: list, required: list):
  skipped = 0
  missing_by_key = Counter()
  missing_by_key_reason = Counter() 
  valid_records = []

  for i, raw in enumerate(RAW):
    missing = []
    for key in required:
      v = raw.get(key)
      if v is None:
        missing.append((key, "None"))
      elif v == "":
        missing.append((key, "empty"))

    if missing:
      skipped += 1
      print(f"[SKIP] idx={i} missing={missing}")

      # 레코드 기준 집계 (키 중복 방지)
      for key, reason in missing:
        missing_by_key[key] += 1
        missing_by_key_reason[(key, reason)] += 1
      continue

    valid_records.append(tuple(raw[k] for k in required))
  
  print("\n=== MISSING SUMMARY ===")
  print(f"number of skipped_record: {skipped}")
 
  if (skipped > 0):
    print(f"number of skipped_record: {skipped}")
    print("\n=== missing by key (record count) ===")
    for k, n in missing_by_key.items():
      print(f"{k}: {n}")

    print("\n=== MISSING BY KEY/REASON (record count) ===")
    for (k, reason), n in missing_by_key_reason.items():
      print(f"{k}/{reason}: {n}")

  return valid_records