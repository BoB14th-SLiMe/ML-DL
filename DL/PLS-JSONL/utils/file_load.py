"""
file_loader.py

다음 파일을 로드하는 함수
  - json
  - jsonl

input
  - file_type : 파일 종류
  - file_path : 파일 경로

output
  - 
  
"""

import json, sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve()
sys.path.insert(0, str(ROOT))
from .path_validation import path_validation

def file_load(file_type: str, file_path: str):
  try:
    if(file_type == "json"):
      return json_load(file_path)
    elif(file_type == "jsonl"):
      return jsonl_load(file_path)
    else:
      print("Error no file type (file_load)")
  except Exception as e:
    print(f"{e}")


def json_load(file_path: str):
  try:
    path_validation(file_path)
    return json.loads(Path(file_path).read_text())  
  except Exception as e:
    print(f"{e}")


def jsonl_load(file_path: str):
  try:
    path_validation(file_path)
    file_data = []
    with open(file_path, "r", encoding="utf-8-sig") as fin:
      for file_line in tqdm(fin, desc="jsonl load", leave=True, ncols=90):
        file_line = file_line.strip()
        if not file_line:
          continue
        try:
          file_data.append(json.loads(file_line))
        except json.JSONDecodeError:
          print(f"Error json parsing : {file_line[:50]}")
    return file_data
  except Exception as e:
    print(f"{e}")
