"""
file_save.py

다음 파일을 저장하는 함수
  - jsonl

input
  - records : 저장할 데이터
  - out_path : 출력 경로

"""

import json
from pathlib import Path

def save_jsonl(records: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")