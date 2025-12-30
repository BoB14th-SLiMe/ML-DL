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
from typing import Any

def file_save(file_type: str, data: Any, out_path: str, indent: int = 2) -> None:
    try:
        if file_type == "json":
            return save_json(data, out_path, indent=indent)
        elif file_type == "jsonl":
            return save_jsonl(data, out_path)
        else:
          print("Error no file type (file_save)")
    except Exception as e:
        print(f"{e}")

def save_json(obj: Any, out_path: Path, indent: int = 2) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.write("\n")

def save_jsonl(records: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
