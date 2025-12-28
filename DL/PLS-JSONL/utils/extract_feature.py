"""
extract_feature.py

pls_extract : PLS 데이터에서 timestamp, sq, ak, fl 추출

raw_extract : 원본 데이터에서 required에 해당하는 key 추출

"""
import re
from collections import Counter

import re
from typing import Optional, Dict, Any, List, Tuple, Sequence, Union, overload

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



RawRecord = Dict[str, Any]
ValidTuple = Tuple[Any, ...]

@overload
def raw_extract(RAW: List[RawRecord], required: Sequence[str]) -> List[ValidTuple]: ...
@overload
def raw_extract(RAW: RawRecord, required: Sequence[str]) -> Union[ValidTuple, None]: ...


def raw_extract(
    RAW: Union[List[RawRecord], RawRecord],
    required: Sequence[str],
) -> Union[List[ValidTuple], ValidTuple, None]:
    """
    - RAW가 list[dict]면: (유효 레코드들) -> list[tuple] 반환 + 요약 출력(1회)
    - RAW가 dict(단일 레코드)면: 유효하면 tuple, 아니면 None 반환 (출력 없음)
    """

    if isinstance(required, (str, bytes, bytearray)):
        raise TypeError("required must be a sequence of keys, not a string.")

    if isinstance(RAW, dict):
        for k in required:
            v = RAW.get(k)
            if v is None or v == "":
                return None
        return tuple(RAW[k] for k in required)

    if not isinstance(RAW, list):
        raise TypeError(f"RAW must be list or dict. got={type(RAW)}")

    skipped = 0
    missing_by_key = Counter()
    missing_by_key_reason = Counter()
    valid_records: List[ValidTuple] = []

    for i, raw in enumerate(RAW):
        if not isinstance(raw, dict):
            skipped += 1
            missing_by_key["__not_dict__"] += 1
            missing_by_key_reason[("__not_dict__", str(type(raw)))] += 1
            continue

        missing = []
        for key in required:
            v = raw.get(key)
            if v is None:
                missing.append((key, "None"))
            elif v == "":
                missing.append((key, "empty"))

        if missing:
            skipped += 1
            for key, reason in missing:
                missing_by_key[key] += 1
                missing_by_key_reason[(key, reason)] += 1
            continue

        valid_records.append(tuple(raw[k] for k in required))

    print("\n=== MISSING SUMMARY ===")
    print(f"number of skipped_record: {skipped}")

    if skipped > 0:
        print("\n=== missing by key (record count) ===")
        for k, n in missing_by_key.items():
            print(f"{k}: {n}")

        print("\n=== MISSING BY KEY/REASON (record count) ===")
        for (k, reason), n in missing_by_key_reason.items():
            print(f"{k}/{reason}: {n}")

    return valid_records