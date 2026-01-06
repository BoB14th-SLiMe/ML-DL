#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
0.attack_result.py (multi-scenario, JSONL only + 옵션으로 feature 추출까지)

최신 설계 요약:

1단계 (항상 수행):
  - 패킷 단위 JSONL (각 줄 = 1 패킷, 예: attack_ver5_1.jsonl)을 입력으로 받아
  - window_size, step_size 로 슬라이딩 윈도우를 만들고
  - 각 윈도우가 "공격 패킷"을 포함하는지 판단해서 pattern(NORMAL/ATTACK)을 붙인 뒤
  - 3.window_to_feature_csv_dynamic_index.py 가 그대로 사용할 수 있는
    "패턴 윈도우 JSONL" (raw packet 기반)을 출력한다.

2단계 (옵션: --pre-dir, --feat-output1 를 주면 수행):
  - 1단계에서 만든 윈도우 JSONL을
    3.window_to_feature_csv_dynamic_index.py 에 넣어서
    훈련과 동일한 전처리 (slot 포함 FEATURE_COLUMNS)를 적용한
    feature JSONL을 생성한다. (각 줄 = 1 window, sequence_group = feature row 리스트)

3단계 (옵션: --model-dir 를 주면 수행):
  - 2단계에서 만든 feature JSONL 을 다시 읽어서
  - sequence_group 안의 각 feature row 에 대해 ML 모델을 돌려
    anomaly 여부를 0/1 로 판단하고
  - 그 결과를 해당 row 의 "match" 필드에 기록한다.
    (0 = 이상, 1 = 정상, ML 미사용/실패 시 None)
  - 이때 "match" 는 반드시 sequence_group 내부에만 존재하도록 보장하며,
    top-level 에 있는 match 키는 모두 제거한다.
  - 동시에 window_index, pattern, sequence_group 만 남기는 slim 처리까지 한 번에 수행한다.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import subprocess
import sys
import importlib.util

import numpy as np  # pandas 제거, numpy만 사용


# ------------------------- ML 관련 전역 변수 -------------------------

ML_MODEL = None
ML_SCALER = None
ML_SELECTED_FEATURES: Optional[List[str]] = None
ML_META: Dict[str, Any] = {}
ML_THRESHOLD: Optional[float] = None
FEATURE_NAMES_CACHE: Optional[List[str]] = None  # feature 이름 캐시


# ------------------------- 기본 유틸 -------------------------

def is_all_zero_data(hex_str: str) -> bool:
    """xgt_fen.data가 전부 0(예: '000000...')인지 확인."""
    if not hex_str:
        return False
    return all(ch == "0" for ch in hex_str)


def safe_int(val: Any, default: int = 0) -> int:
    """
    modbus.fc, xgt_fen.cmd 등 숫자 필드 안전 변환.

    - 리스트면 첫 번째 None 아닌 값 사용
    - "6", "06", 6 → 10진
    - "0x58", "0X59" → 16진
    """
    try:
        if isinstance(val, list):
            for v in val:
                if v is None:
                    continue
                val = v
                break

        if val is None:
            raise ValueError("None value")

        if isinstance(val, str):
            s = val.strip()
            if not s:
                raise ValueError("empty string")
            if s.startswith(("0x", "0X")):
                return int(s, 16)
            return int(s)

        return int(val)
    except Exception:
        return default


def normalize_hex_string(s: Any) -> str:
    """xgt_fen.data 같은 hex 문자열: 공백 제거 + 소문자."""
    if not isinstance(s, str):
        return ""
    return "".join(s.split()).lower()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # 1단계: 공격 윈도우 생성용
    p.add_argument(
        "--input", "-i", required=True,
        help="패킷 단위 JSONL 경로 (각 line = 1 packet)",
    )
    p.add_argument(
        "--window-size", "-w", type=int, default=80,
        help="윈도우 길이(묶을 패킷 개수, 기본=80)",
    )
    p.add_argument(
        "--step-size", "-s", type=int, default=None,
        help="슬라이딩 stride (기본: window-size와 동일 → non-overlap)",
    )
    p.add_argument(
        "--output", "-o", required=True,
        help="패턴 윈도우 JSONL 출력 경로 (1단계 결과, raw packet 포함)",
    )
    p.add_argument(
        "--mode", "-m",
        choices=[
            "auto",
            "modbus_fc6",
            "xgt_last_zero", "xgt_mid_zero", "xgt_head_zero",
            "xgt_write_or_fc6",
            "xgt_write_or_fc6_after",
            "xgt_d528_zero", "xgt_d525_zero",
        ],
        default="auto",
        help=(
            "공격 판정 모드. "
            "auto: 파일명으로부터 자동 결정, "
            "modbus_fc6: 모드버스 fc=6 기준, "
            "xgt_last_zero/xgt_d528_zero: xgt_fen D0528(마지막 워드) 0000, "
            "xgt_mid_zero/xgt_d525_zero: xgt_fen D0525(3번째 워드) 0000, "
            "xgt_head_zero: xgt_fen D0523(첫 워드) 0000, "
            "xgt_write_or_fc6: XGT 쓰기(cmd 0x58/0x59) 또는 Modbus FC6, "
            "xgt_write_or_fc6_after: XGT 쓰기/Modbus FC6/xgt_head_zero 중 "
            "하나라도 포함된 윈도우를 ATTACK으로 라벨링 "
            "(write/fc6/head_zero 합집합; after 전파 없음)"
        ),
    )

    # 2단계: feature 추출 (3.window_to_feature_csv_dynamic_index.py 호출용)
    p.add_argument(
        "--pre-dir",
        help="전처리 파라미터 JSON들이 모여있는 디렉토리 (주면 feature 추출까지 수행)",
    )
    p.add_argument(
        "--feat-output1",
        help="feature JSONL 출력 경로 (최종 slim + ML match 반영된 결과)",
    )
    p.add_argument(
        "--window-script",
        default=None,
        help=(
            "3.window_to_feature_csv_dynamic_index.py 경로 "
            "(생략 시 repo 구조 기준으로 자동 추정: "
            "../../preprocessing/code/3.window_to_feature_csv_dynamic_index.py)"
        ),
    )

    # 3단계: ML 모델 관련 옵션
    p.add_argument(
        "--model-dir",
        help="ML 모델 번들이 들어있는 디렉터리 (model.pkl, scaler.pkl, selected_features.json 등)",
    )
    p.add_argument(
        "--model-loader",
        default=None,
        help=(
            "load_model_bundle 함수가 들어있는 1-1.model_loader.py 경로 "
            "(생략 시 0.attack_result.py 기준으로 자동 추정 시도)"
        ),
    )
    p.add_argument(
        "--ml-threshold",
        type=float,
        default=None,
        help=(
            "anomaly score 기준 threshold "
            "(score 가 threshold 를 넘으면 이상으로 간주; "
            "score_higher_is_anom 에 따라 부호 해석)"
        ),
    )

    return p.parse_args()


# ------------------------- ML 번들 로더 -------------------------

def load_model_bundle_from_file(model_dir: Path | str, loader_path: Path | str):
    """
    1-1.model_loader.py 파일 경로(loader_path)와
    모델 디렉터리(model_dir)를 받아서

    model, scaler, selected_features, metadata 를 반환한다.
    """
    loader_path = Path(loader_path)
    model_dir = Path(model_dir)

    if not loader_path.exists():
        raise FileNotFoundError(f"model loader not found: {loader_path}")

    spec = importlib.util.spec_from_file_location("model_loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    # 1-1.model_loader.py 안에 정의된 load_model_bundle 사용
    return module.load_model_bundle(model_dir)


# ------------------------- 패킷 단위 공격 판정 -------------------------

def packet_is_attack_xgt_word_zero(pkt: Dict[str, Any], word_idx: int) -> bool:
    """
    XGT-FEnet 블록에서 특정 워드(word_idx)가 0000인 경우를 공격으로 간주.
    단, data 전체가 0000... 인 경우는 정상 처리.

    - protocol == 'xgt_fen'
    - xgt_fen.vars == '%DB001046'
    - hex_str[word_idx*4 : word_idx*4+4] == '0000'
    """
    if pkt.get("protocol") != "xgt_fen":
        return False

    vars_field = pkt.get("xgt_fen.vars")
    if vars_field != "%DB001046":
        return False

    hex_str = normalize_hex_string(pkt.get("xgt_fen.data"))
    if not hex_str:
        return False

    if is_all_zero_data(hex_str):
        # 전체 0이면 정상 패턴으로 간주
        return False

    start = word_idx * 4
    end = start + 4
    if len(hex_str) < end:
        return False

    word = hex_str[start:end]
    return word == "0000"


def packet_is_attack_modbus_fc6(pkt: Dict[str, Any]) -> bool:
    """protocol == 'modbus' && modbus.fc == 6 → 공격."""
    if pkt.get("protocol") != "modbus":
        return False

    fc_raw = pkt.get("modbus.fc")
    fc = safe_int(fc_raw, default=-1)
    return fc == 6


def packet_is_attack_xgt_write(pkt: Dict[str, Any]) -> bool:
    """
    XGT-FEnet 쓰기 패킷:
      - protocol == 'xgt_fen'
      - xgt_fen.cmd in {0x58, 0x59}
    """
    if pkt.get("protocol") != "xgt_fen":
        return False

    cmd_raw = pkt.get("xgt_fen.cmd")
    cmd = safe_int(cmd_raw, default=-1)
    return cmd in (0x58, 0x59)


def packet_is_attack_xgt_last_zero(pkt: Dict[str, Any]) -> bool:
    """ver5: 마지막 워드 (word_idx=5) 0000."""
    return packet_is_attack_xgt_word_zero(pkt, word_idx=5)


def packet_is_attack_xgt_mid_zero(pkt: Dict[str, Any]) -> bool:
    """ver5_1: 가운데 워드 (8~11 문자, word_idx=4) 0000."""
    return packet_is_attack_xgt_word_zero(pkt, word_idx=4)


def packet_is_attack_xgt_head_zero(pkt: Dict[str, Any]) -> bool:
    """
    ver5_2: 첫 워드가 0000 또는 0005가 아닌 경우 → 공격
    (즉, 정상은 정확히 0005일 때만)
    """
    if pkt.get("protocol") != "xgt_fen":
        return False

    vars_field = pkt.get("xgt_fen.vars")
    if vars_field != "%DB001046":
        return False

    hex_str = normalize_hex_string(pkt.get("xgt_fen.data"))
    if not hex_str:
        return False

    if is_all_zero_data(hex_str):
        return False  # 전체 0인 경우는 정상 처리

    # 첫 워드(0~3)
    if len(hex_str) < 4:
        return False

    head_word = hex_str[0:4]
    return head_word == "0000" or head_word != "0500"


def compute_attack_flags_for_mode(
    packets: List[Dict[str, Any]],
    attack_mode: str,
) -> List[bool]:
    """
    각 패킷에 대해 '이 패킷이 공격이냐?'를 attack_mode 기준으로 한 번만 계산해서
    bool 리스트로 반환.

    이후 슬라이딩 윈도우에서는 이 리스트의 슬라이스에 대해 any()만 보면 되므로,
    같은 패킷에 대해 hex 파싱/조건 체크를 여러 번 반복하지 않게 된다.
    """
    flags: List[bool] = []

    for pkt in packets:
        # 공통: XGT 쓰기(cmd 0x58/0x59)는 항상 공격
        if packet_is_attack_xgt_write(pkt):
            flags.append(True)
            continue

        # Modbus FC6 기반 모드
        if attack_mode in ("modbus_fc6", "xgt_write_or_fc6", "xgt_write_or_fc6_head_zero"):
            if packet_is_attack_modbus_fc6(pkt):
                flags.append(True)
                continue

        # D0528 마지막 워드 0000
        if attack_mode == "xgt_last_zero":
            if packet_is_attack_xgt_last_zero(pkt):
                flags.append(True)
                continue

        # D0525 중간 워드 0000
        if attack_mode == "xgt_mid_zero":
            if packet_is_attack_xgt_mid_zero(pkt):
                flags.append(True)
                continue

        # D0523 헤드 워드 조건
        if attack_mode in ("xgt_head_zero", "xgt_write_or_fc6_head_zero"):
            if packet_is_attack_xgt_head_zero(pkt):
                flags.append(True)
                continue

        flags.append(False)

    return flags


def infer_attack_mode_from_filename(filename: str) -> str:
    """
    --mode auto 일 때, 입력 파일명으로부터 공격 모드를 추론.
    """
    name = filename.lower()

    if "ver5_2" in name:
        return "xgt_head_zero"
    if "ver5_1" in name:
        return "xgt_mid_zero"
    if "ver5" in name:
        return "xgt_last_zero"

    if "ver2" in name or "ver11" in name or "attack" in name:
        return "xgt_write_or_fc6"

    return "modbus_fc6"


# ------------------------- ML 기반 match 계산 (feature row 단위) -------------------------

def infer_match_with_ml(feat_row: Dict[str, Any]) -> Optional[int]:
    """
    하나의 feature row(= sequence_group 안의 원소 1개)에 대해
    ML 모델로 anomaly 여부를 판단하고 match 값을 반환.

    반환:
      - 0 : ML 기준 "이상(anomaly)"
      - 1 : ML 기준 "정상(normal)"
      - None : ML_MODEL 이 없거나, feature 계산 실패 등으로 판단 불가

    최적화:
      - feature 이름(feature_names)은 전역 캐시에 한 번만 계산
      - pandas.DataFrame 대신 numpy 배열만 사용
    """
    global FEATURE_NAMES_CACHE

    if ML_MODEL is None:
        return None
    if feat_row is None:
        return None

    # 1) feature 이름 결정 (캐시)
    if FEATURE_NAMES_CACHE is None:
        if ML_SELECTED_FEATURES is not None:
            feature_names = list(ML_SELECTED_FEATURES)
        elif ML_SCALER is not None and hasattr(ML_SCALER, "feature_names_in_"):
            feature_names = list(ML_SCALER.feature_names_in_)
        elif hasattr(ML_MODEL, "feature_names_in_"):
            feature_names = list(ML_MODEL.feature_names_in_)
        else:
            exclude_cols = {"window_id", "window_index", "pattern", "match"}
            feature_names = sorted(k for k in feat_row.keys() if k not in exclude_cols)

        FEATURE_NAMES_CACHE = feature_names
    else:
        feature_names = FEATURE_NAMES_CACHE

    if not feature_names:
        return None

    # 2) 하나의 row를 numpy 벡터로 구성
    vec: List[float] = []
    for f in feature_names:
        v = feat_row.get(f, 0.0)
        if v is None:
            v = 0.0
        try:
            vec.append(float(v))
        except Exception:
            vec.append(0.0)

    X = np.asarray([vec], dtype=float)  # shape (1, n_features)

    # 3) scaler 적용 (있으면)
    if ML_SCALER is not None:
        X_scaled = ML_SCALER.transform(X)
    else:
        X_scaled = X

    # 4) score 계산
    if hasattr(ML_MODEL, "decision_function"):
        score = float(ML_MODEL.decision_function(X_scaled)[0])
    elif hasattr(ML_MODEL, "predict_proba"):
        proba = ML_MODEL.predict_proba(X_scaled)[0]
        score = float(proba[1]) if len(proba) > 1 else float(proba[0])
    else:
        # fallback: predict 결과 자체를 score처럼 사용
        score = float(ML_MODEL.predict(X_scaled)[0])

    # 5) threshold 기준 anomaly 판단
    score_higher_is_anom = ML_META.get("score_higher_is_anom", True)
    thr = ML_THRESHOLD if ML_THRESHOLD is not None else 0.0

    if score_higher_is_anom:
        is_anom = score >= thr
    else:
        is_anom = score <= thr

    # ML에서 이상(attack)으로 잡은 경우 → 0
    # ML에서 정상으로 잡은 경우 → 1
    return 0 if is_anom else 1


# ------------------------- feature JSONL 에 ML match 적용 (+ slim) -------------------------

def apply_ml_to_feature_jsonl(jsonl_path: Path) -> None:
    """
    feature JSONL 을 읽어서,
    sequence_group 내 각 row 에 대해 ML 기반 match 값을 계산/주입한 뒤,

    최종적으로는 아래와 같은 "슬림" 구조만 남기면서 파일을 덮어쓴다:

      {
        "window_index": 0,
        "pattern": "NORMAL" or "ATTACK",
        "sequence_group": [ {...feat..., "match": 0/1/None}, ... ]
      }

    - ML_MODEL 이 없는 경우:
        * top-level match 제거 + slim만 수행 (sequence_group 내 match 는 그대로 둠)
    - ML_MODEL 이 있는 경우:
        * sequence_group 의 각 row 에 대해 infer_match_with_ml 을 호출해서
          0/1/None 을 계산하고, row["match"] 에 덮어쓴다.
        * top-level match 키는 모두 제거.
    """
    if not jsonl_path.exists():
        print(f"[WARN] apply_ml_to_feature_jsonl: 파일이 없음: {jsonl_path}")
        return

    tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".mltmp")

    n_in = 0
    n_out = 0
    n_rows = 0
    n_ml_0 = 0
    n_ml_1 = 0
    n_ml_none = 0

    with jsonl_path.open("r", encoding="utf-8") as fin, \
         tmp_path.open("w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            n_in += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] apply_ml: line {line_no} JSON 파싱 실패, 스킵: {e}")
                continue

            # sequence_group 가져오기
            seq_list = obj.get("sequence_group")
            if not isinstance(seq_list, list):
                seq_list = []

            # ML 적용
            if ML_MODEL is not None:
                for row in seq_list:
                    if not isinstance(row, dict):
                        continue
                    n_rows += 1
                    ml_match = infer_match_with_ml(row)
                    row["match"] = ml_match
                    if ml_match == 0:
                        n_ml_0 += 1
                    elif ml_match == 1:
                        n_ml_1 += 1
                    else:
                        n_ml_none += 1
            else:
                # ML 모델이 없으면 구조만 정리 (top-level match 제거 + slim)
                pass

            # top-level match 키는 무조건 제거
            if "match" in obj:
                del obj["match"]

            win_idx = obj.get("window_index", obj.get("window_id"))

            # 슬림 구조로 재구성해서 바로 기록
            slim_obj = {
                "window_index": win_idx,
                "pattern": obj.get("pattern"),
                "sequence_group": seq_list,
            }

            fout.write(json.dumps(slim_obj, ensure_ascii=False) + "\n")
            n_out += 1

    # 원본을 ML+slim 버전으로 교체
    tmp_path.replace(jsonl_path)

    if ML_MODEL is not None:
        print(f"[INFO] feature JSONL에 ML match 적용 + slim 완료: {jsonl_path}")
        print(f"[INFO]   윈도우 수(라인 수)     = {n_in}")
        print(f"[INFO]   sequence_group row 수 = {n_rows}")
        print(f"[INFO]   match == 0 (이상) 수  = {n_ml_0}")
        print(f"[INFO]   match == 1 (정상) 수  = {n_ml_1}")
        print(f"[INFO]   match None 수         = {n_ml_none}")
    else:
        print(f"[INFO] ML 모델이 없어 feature JSONL에는 slim 처리만 수행했습니다: {jsonl_path}")


# ------------------------- 메인 -------------------------

def main():
    global ML_MODEL, ML_SCALER, ML_SELECTED_FEATURES, ML_META, ML_THRESHOLD, FEATURE_NAMES_CACHE

    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    window_size = args.window_size
    step_size = args.step_size if args.step_size is not None else window_size

    # 공격 모드 결정
    if args.mode == "auto":
        attack_mode = infer_attack_mode_from_filename(input_path.name)
        print(f"[INFO] --mode=auto → 파일명 '{input_path.name}' 기준 공격 모드: {attack_mode}")
    else:
        if args.mode in ("xgt_d528_zero", "xgt_last_zero"):
            attack_mode = "xgt_last_zero"
        elif args.mode in ("xgt_d525_zero", "xgt_mid_zero"):
            attack_mode = "xgt_mid_zero"
        else:
            attack_mode = args.mode
        print(f"[INFO] --mode={args.mode} (사용자 지정) → 내부 attack_mode={attack_mode}")

    # window_has_attack 에 넘길 실제 모드 (xgt_write_or_fc6_after용 내부 합집합 모드)
    if attack_mode == "xgt_write_or_fc6_after":
        # XGT 쓰기 + Modbus FC6 + xgt_head_zero 를 모두 공격으로 보는 내부 모드
        base_attack_mode = "xgt_write_or_fc6_head_zero"
    else:
        base_attack_mode = attack_mode

    print(f"[INFO] 입력 JSONL : {input_path}")
    print(f"[INFO] window_size = {window_size}, step_size = {step_size}")
    print(f"[INFO] 윈도우 JSONL 출력 : {output_path}")
    print(f"[INFO] 공격 판정 모드 = {attack_mode} (base={base_attack_mode})")

    # ML 모델 로드 (옵션, 실제 적용은 feature JSONL에 대해 수행)
    FEATURE_NAMES_CACHE = None  # 모델 로딩할 때마다 feature 이름 캐시 초기화

    if args.model_dir:
        if args.model_loader:
            loader_path = Path(args.model_loader)
        else:
            # 0.attack_result.py 기준 같은 디렉터리에 1-1.model_loader.py 있다고 가정
            loader_path = Path(__file__).resolve().parent / "1-1.model_loader.py"

        print(f"[INFO] ML model bundle 로딩: model_dir={args.model_dir}")
        print(f"[INFO] ML loader 파일     : {loader_path}")

        ML_MODEL, ML_SCALER, ML_SELECTED_FEATURES, ML_META = \
            load_model_bundle_from_file(args.model_dir, loader_path)

        # ⚠️ MinMaxScaler 가 feature_names_in_ 을 들고 있으면
        # numpy 배열을 넣을 때마다 "X does not have valid feature names" 경고가 뜸.
        # 우리는 selected_features 순서대로 벡터를 만들기 때문에 이름 검증은 필요 없음.
        if ML_SCALER is not None and hasattr(ML_SCALER, "feature_names_in_"):
            try:
                del ML_SCALER.feature_names_in_
                print("[INFO] MinMaxScaler.feature_names_in_ 제거 → 경고 억제")
            except Exception as e:
                print(f"[WARN] feature_names_in_ 제거 실패: {e}")

        # threshold 결정: 우선 CLI → metadata.json → 디폴트 0.0
        if args.ml_threshold is not None:
            ML_THRESHOLD = args.ml_threshold
        else:
            ML_THRESHOLD = ML_META.get("threshold", 0.0)

        print(f"[INFO] ML threshold = {ML_THRESHOLD}")
        print(f"[INFO] ML meta      = {ML_META}")

    else:
        ML_MODEL = None
        ML_SCALER = None
        ML_SELECTED_FEATURES = None
        ML_META = {}
        ML_THRESHOLD = None
        print("[INFO] model_dir 미지정 → ML 기반 match 는 사용하지 않습니다.")

    # 1) JSONL 전체 읽기 (raw packet)
    packets: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] JSON 파싱 실패 (line {line_no}), 스킵: {e}")
                continue
            packets.append(obj)

    total_packets = len(packets)
    print(f"[INFO] 총 패킷 수 = {total_packets}")
    if total_packets == 0:
        print("[WARN] 유효한 패킷이 없습니다. 종료합니다.")
        return

    # 1-1) 공격 플래그를 패킷 단위로 미리 계산 (슬라이딩 윈도우에서 재사용)
    print("[INFO] 패킷 단위 공격 플래그 사전 계산 중...")
    attack_flags = compute_attack_flags_for_mode(packets, base_attack_mode)
    print("[INFO] 패킷 단위 공격 플래그 계산 완료.")

    # 2) 슬라이딩 윈도우 + 윈도우 JSONL(output_path) 출력
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        window_index = 0
        start_idx = 0

        while start_idx < total_packets:
            end_idx = start_idx + window_size
            window_packets = packets[start_idx:end_idx]
            if not window_packets:
                break

            valid_len = len(window_packets)
            end_packet_idx = start_idx + valid_len - 1

            # (1) 윈도우 내부 공격 여부: precomputed attack_flags 사용
            has_attack_in_window = any(attack_flags[start_idx:end_idx])

            # (2) after 전파 없음: 윈도우 안에 공격이 있을 때만 ATTACK (heuristic label)
            is_attack = 1 if has_attack_in_window else 0

            out_obj = {
                "window_id": window_index,
                "pattern": "ATTACK" if is_attack == 1 else "NORMAL",
                # match 는 여기서 절대로 넣지 않는다 (sequence_group 내부에서만 존재해야 함)
                "index": list(range(valid_len)),
                "window_size": valid_len,
                "sequence_group": window_packets,
                "description": None,
                "start_packet_idx": start_idx,
                "end_packet_idx": end_packet_idx,
                "is_anomaly": is_attack,
            }

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            window_index += 1
            start_idx += step_size

    print(f"[INFO] 1단계 완료: 총 {window_index}개 윈도우를 {output_path} 에 JSONL로 저장했습니다.")

    # 3) 옵션: 바로 feature 추출까지 실행 + ML match 적용 + slim
    if args.pre_dir and args.feat_output1:
        # 3.window_to_feature_csv_dynamic_index.py 경로 결정
        if args.window_script:
            window_script = Path(args.window_script)
        else:
            # 0.attack_result.py 가 DL/result_train/code 에 있다고 가정하면
            # DL/preprocessing/code/3.window_to_feature_csv_dynamic_index.py 로 올라감
            window_script = (
                Path(__file__).resolve()
                .parents[2]
                / "preprocessing"
                / "code"
                / "3.window_to_feature_csv_dynamic_index.py"
            )

        if not window_script.exists():
            print(f"[WARN] feature 추출 스킵: window_script 경로를 찾을 수 없음: {window_script}")
            return

        pre_dir = Path(args.pre_dir)
        feat_out1 = Path(args.feat_output1)
        feat_out1.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(window_script),
            "--input", str(output_path),          # 방금 만든 윈도우 JSONL (raw packet 기반)
            "--pre_dir", str(pre_dir),
            "--output1", str(feat_out1),
            "--output2", str(feat_out1),
        ]
        if args.window_size is not None:
            cmd += ["--max-index", str(args.window_size)]

        print("[INFO] 2단계: 3.window_to_feature_csv_dynamic_index.py 호출 시작")
        print("       명령:", " ".join(cmd))

        subprocess.run(cmd, check=True)
        print("[INFO] 2단계 완료: feature JSONL 생성됨")
        print(f"       raw feature output = {feat_out1}")

        # 4) sequence_group 내부에 ML 기반 match 적용 + 동시에 slim 처리
        apply_ml_to_feature_jsonl(feat_out1)
    else:
        print("[INFO] pre_dir/feat-output1 이 설정되지 않아 "
              "feature 추출/ML match 적용 단계는 건너뜁니다.")


if __name__ == "__main__":
    main()

"""
python 0.attack_result.py --input "..\data\attack.jsonl" --window-size 16 --step-size 2 --output "..\result\attack.jsonl" --mode auto
python 0.attack_result.py --input "..\data\attack_ver2.jsonl" --window-size 16 --step-size 3 --output "..\result\attack_ver2.jsonl" --mode auto
python 0.attack_result.py --input "..\data\attack_ver5.jsonl" --window-size 16 --step-size  --output "..\result\attack_ver5.jsonl" --mode auto

python 0.attack_result.py --input "..\data\attack_ver2.jsonl" --window-size 80 --step-size 5 --output "..\result\attack_ver2_window.jsonl" --mode xgt_write_or_fc6 --pre-dir "..\..\preprocessing\result" --feat-output1 "..\result\attack_ver2.jsonl"
python 0.attack_result.py --input "..\data\attack_ver5.jsonl" --window-size 80 --step-size 5 --output "..\result\attack_ver5_window.jsonl" --mode xgt_last_zero --pre-dir "..\..\preprocessing\result" --feat-output1 "..\result\attack_ver5.jsonl"
python 0.attack_result.py --input "..\data\attack_ver5_1.jsonl" --window-size 80 --step-size 5 --output "..\result\attack_ver5_1_window.jsonl" --mode xgt_mid_zero --pre-dir "..\..\preprocessing\result" --feat-output1 "..\result\attack_ver5_1.jsonl"
python 0.attack_result.py --input "..\data\attack_ver5_2.jsonl" --window-size 80 --step-size 5 --output "..\result\attack_ver5_2_window.jsonl" --mode xgt_head_zero --pre-dir "..\..\preprocessing\result" --feat-output1 "..\result\attack_ver5_2.jsonl"
python 0.attack_result.py --input "..\data\attack_ver11.jsonl" --window-size 80 --step-size 5 --output "..\result\attack_ver11_window.jsonl" --mode xgt_write_or_fc6 --pre-dir "..\..\preprocessing\result" --feat-output1 "..\result\attack_ver11.jsonl"


python 0.attack_result.py --input "..\data\attack_ver2.jsonl" --window-size 16 --step-size 2 --output "..\result\attack_ver2_window.jsonl" --mode xgt_write_or_fc6_after --pre-dir "..\..\preprocessing\result" --feat-output1 "..\result\attack_ver2.jsonl"
python 0.attack_result.py --input "..\data\attack_ver5.jsonl" --window-size 16 --step-size 4 --output "..\result\attack_ver5_window.jsonl" --mode xgt_last_zero --pre-dir "..\..\preprocessing\result" --feat-output1 "..\result\attack_ver5.jsonl"
python 0.attack_result.py --input "..\data\attack_ver5_1.jsonl" --window-size 16 --step-size 4 --output "..\result\attack_ver5_1_window.jsonl" --mode xgt_mid_zero --pre-dir "..\..\preprocessing\result" --feat-output1 "..\result\attack_ver5_1.jsonl"
python 0.attack_result.py --input "..\data\attack_ver5_2.jsonl" --window-size 16 --step-size 4 --output "..\result\attack_ver5_2_window.jsonl" --mode xgt_head_zero --pre-dir "..\..\preprocessing\result" --feat-output1 "..\result\attack_ver5_2.jsonl"
python 0.attack_result.py --input "..\data\attack_ver11.jsonl" --window-size 16 --step-size 2 --output "..\result\attack_ver11_window.jsonl" --mode xgt_write_or_fc6_after --pre-dir "..\..\preprocessing\result" --feat-output1 "..\result\attack_ver11.jsonl"


match
python 0.attack_result.py --input "..\data\attack_ver2.jsonl" --window-size 80 --step-size 5 --output "..\result\attack_ver2_window.jsonl" --mode xgt_write_or_fc6_after --pre-dir "..\..\preprocessing\result" --feat-output1 "..\result\attack_ver2.jsonl" --model-dir  "..\..\..\operating\ML\model"   --model-loader "..\..\..\operating\ML\code\1-1.model_load.py" --ml-threshold -126
python 0.attack_result.py --input "..\data\attack_ver5.jsonl" --window-size 80 --step-size 5 --output "..\result\attack_ver5_window.jsonl" --mode xgt_last_zero --pre-dir "..\..\preprocessing\result" --feat-output1 "..\result\attack_ver5.jsonl" --model-dir  "..\..\..\operating\ML\model"   --model-loader "..\..\..\operating\ML\code\1-1.model_load.py" --ml-threshold -126
python 0.attack_result.py --input "..\data\attack_ver5_1.jsonl" --window-size 80 --step-size 5 --output "..\result\attack_ver5_1_window.jsonl" --mode xgt_mid_zero --pre-dir "..\..\preprocessing\result" --feat-output1 "..\result\attack_ver5_1.jsonl" --model-dir  "..\..\..\operating\ML\model"   --model-loader "..\..\..\operating\ML\code\1-1.model_load.py" --ml-threshold -126
python 0.attack_result.py --input "..\data\attack_ver5_2.jsonl" --window-size 80 --step-size 5 --output "..\result\attack_ver5_2_window.jsonl" --mode xgt_head_zero --pre-dir "..\..\preprocessing\result" --feat-output1 "..\result\attack_ver5_2.jsonl" --model-dir  "..\..\..\operating\ML\model"   --model-loader "..\..\..\operating\ML\code\1-1.model_load.py" --ml-threshold -126
python 0.attack_result.py --input "..\data\attack_ver11.jsonl" --window-size 80 --step-size 5 --output "..\result\attack_ver11_window.jsonl" --mode xgt_write_or_fc6_after --pre-dir "..\..\preprocessing\result" --feat-output1 "..\result\attack_ver11.jsonl" --model-dir  "..\..\..\operating\ML\model"   --model-loader "..\..\..\operating\ML\code\1-1.model_load.py" --ml-threshold -126

"""