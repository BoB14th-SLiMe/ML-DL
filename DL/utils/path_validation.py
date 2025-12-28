"""
path_validation.py

파일의 경로를 검증하는 함수

input
  - file_path : 파일 경로
 
  
"""

import os

def path_validation(file_path: str):
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"Error not found file path(file_loader): {file_path}")
  return True
