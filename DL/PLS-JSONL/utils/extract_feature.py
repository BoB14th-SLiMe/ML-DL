"""
extract_feature.py

pls_extract : PLS 데이터에서 timestamp, sq, ak, fl 추출

raw_extract : 원본 데이터에서 required에 해당하는 key 추출

"""
import re
from collections import Counter

import re
from typing import Optional, Dict, Any

_TS_RE = re.compile(r'@timestamp\s*[:=]\s*"?([0-9T:\.\-Z\+]+)"?')
_SQ_RE = re.compile(r'\bsq\s*[:=]\s*"?([^",>\s]+)"?')
_AK_RE = re.compile(r'\bak\s*[:=]\s*"?([^",>\s]+)"?')
_FL_RE = re.compile(r'\bfl\s*[:=]\s*"?([^",>\s]+)"?')

def pls_extract(pls_line: Any) -> Optional[Dict[str, str]]:
    if isinstance(pls_line, dict):
        ts = pls_line.get("@timestamp") or pls_line.get("timestamp")
        sq = pls_line.get("sq")
        ak = pls_line.get("ak")
        fl = pls_line.get("fl")
        if ts and sq is not None and ak is not None and fl is not None:
            return {"@timestamp": str(ts), "sq": str(sq), "ak": str(ak), "fl": str(fl)}
        return None

    if not isinstance(pls_line, str):
        return None

    pls = pls_line

    ts = _TS_RE.search(pls)
    sq = _SQ_RE.search(pls)
    ak = _AK_RE.search(pls)
    fl = _FL_RE.search(pls)

    if not (ts and sq and ak and fl):
        return None

    return {
        "@timestamp": ts.group(1),
        "sq": sq.group(1),
        "ak": ak.group(1),
        "fl": fl.group(1),
    }


def timestamp_extract(pls_line: str) -> Optional[str]:
    if not isinstance(pls_line, str):
        return None

    ts = re.search(r'timestamp="([\w\-\:\.TZ\+]+)"', pls_line)
    if ts:
        return ts.group(1)

    return None


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