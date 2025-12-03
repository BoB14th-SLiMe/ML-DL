#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
feature_single_run.py  (조합 실험 + 항상 제외할 feature 지원 버전)

역할:
  - ALL_FEATURES 중에서,
      1) ESSENTIAL_FEATURES 는 항상 포함하고
      2) EXCLUDED_FEATURES 는 어떤 조합에서도 절대 사용하지 않으며
      3) OPTIONAL_FEATURES(= ALL − ESSENTIAL − EXCLUDED)에 대해서는
         가능한 모든 조합(부분집합)을 만들어
         각 조합별로 파이프라인을 실행한다.

루프 한 번에 하는 일:
  1) train/data/exclude.txt 를 "해당 조합의 feature + EXCLUDED_FEATURES 를 제외한 나머지"가 되도록 작성
     → 결국, keep_features 에 없는 feature 와 EXCLUDED_FEATURES 가 모두 exclude.txt 에 기록됨
  2) train/code/0.run_pipeline_pattern.py 실행 (패딩 + LSTM-AE 학습)
  3) result_train/code/0.run_pipeline_pattern.py 실행 (attack 파이프라인 + MSE 분석)
  4) result_train/result/analyze_mse_dist.json 을
     result_train/result/feature_combo/<combo_name>/analyze_mse_dist.json 으로 복사
  5) 모든 조합 루프가 끝나면 exclude.txt 를 원래 상태로 복구
"""

from __future__ import annotations
import subprocess
import shutil
import sys
import json
from pathlib import Path
from itertools import combinations


# ============================================================
# 1) 고정 하이퍼파라미터
# ============================================================
WINDOW_SIZE = 16        # padding & attack 윈도우 크기 (둘 다 동일하게 사용)
EPOCHS = 50
BATCH_SIZE = 64
HIDDEN_DIM = 64
LATENT_DIM = 64

# result_train 쪽 슬라이딩 파라미터
STEP_SIZE = 4           # None이면 WINDOW_SIZE와 동일 (non-overlap)
THRESHOLD = None        # None이면 threshold.json / -1 사용 (네 기존 로직 그대로)


# ============================================================
# 2) feature 리스트들
# ============================================================

# 전체 feature 풀 (기존 FEATURES 그대로)
ALL_FEATURES = [
    "protocol",
    "delta_t",
    "protocol_norm",
    "src_host_id",
    "dst_host_id",
    "sp_norm",
    "dp_norm",
    "dir_code",
    "len_norm",
    "s7comm_ros_norm",
    "s7comm_fn",
    "s7comm_db_norm",
    "s7comm_addr_norm",
    "modbus_addr_norm",
    "modbus_fc_norm",
    "modbus_qty_norm",
    "modbus_bc_norm",
    "modbus_regs_count",
    "modbus_regs_addr_min",
    "modbus_regs_addr_max",
    "modbus_regs_addr_range",
    "modbus_regs_val_min",
    "modbus_regs_val_max",
    "modbus_regs_val_mean",
    "modbus_regs_val_std",
    "xgt_var_id",
    "xgt_var_cnt",
    "xgt_source",
    "xgt_fenet_base",
    "xgt_fenet_slot",
    "xgt_cmd",
    "xgt_dtype",
    "xgt_blkcnt",
    "xgt_err_flag",
    "xgt_err_code",
    "xgt_datasize",
    "xgt_data_missing",
    "xgt_data_len_chars",
    "xgt_data_num_spaces",
    "xgt_data_is_hex",
    "xgt_data_n_bytes",
    "xgt_data_zero_ratio",
    "xgt_data_first_byte",
    "xgt_data_last_byte",
    "xgt_data_mean_byte",
    "xgt_data_bucket",
    "arp_src_host_id",
    "arp_tgt_host_id",
    "arp_op_num",
    "dns_qc_norm",
    "dns_ac_norm",
    "modbus_slot_300001_norm",
    "modbus_slot_300024_norm",
    "modbus_slot_300025_norm",
    "modbus_slot_300026_norm",
    "modbus_slot_300027_norm",
    "modbus_slot_300028_norm",
    "modbus_slot_300029_norm",
    "modbus_slot_300096_norm",
    "modbus_slot_400013_norm",
    "modbus_slot_400014_norm",
    "modbus_slot_400015_norm",
    "modbus_slot_400064_norm",
    "modbus_slot_400065_norm",
    "modbus_slot_400068_norm",
    "xgt_slot_D500_norm",
    "xgt_slot_D523_norm",
    "xgt_slot_D524_norm",
    "xgt_slot_D525_norm",
    "xgt_slot_D526_norm",
    "xgt_slot_D527_norm",
    "xgt_slot_D528_norm",
    "xgt_slot_D597_norm",
    "xgt_slot_D598_norm",
    "xgt_slot_M1_norm",
    "xgt_slot_M2_norm",
    "xgt_slot_M3_norm",
    "xgt_slot_M4_norm",
    "xgt_slot_M5_norm",
    "xgt_slot_M6_norm",
]

# ------------------------------------------------------------
# [1] 항상 제외할 feature (어떤 조합에서도 절대 사용 X)
#     → 여기 넣어둔 애들은 항상 exclude.txt에 들어감
# ------------------------------------------------------------
EXCLUDED_FEATURES: list[str] = [
    "protocol",
    # "delta_t",
    # "protocol_norm",
    "src_host_id",
    "dst_host_id",
    "sp_norm",
    "dp_norm",
    "dir_code",
    # "len_norm",
    "s7comm_ros_norm",
    # "s7comm_fn",
    "s7comm_db_norm",
    "s7comm_addr_norm",
    # "modbus_addr_norm",
    # "modbus_fc_norm",
    # "modbus_qty_norm",
    "modbus_bc_norm",
    "modbus_regs_count",
    "modbus_regs_addr_min",
    "modbus_regs_addr_max",
    "modbus_regs_addr_range",
    "modbus_regs_val_min",
    "modbus_regs_val_max",
    "modbus_regs_val_mean",
    "modbus_regs_val_std",
    "xgt_var_id",
    "xgt_var_cnt",
    "xgt_source",
    "xgt_fenet_base",
    "xgt_fenet_slot",
    # "xgt_cmd",
    "xgt_dtype",
    "xgt_blkcnt",
    "xgt_err_flag",
    "xgt_err_code",
    "xgt_datasize",
    "xgt_data_missing",
    "xgt_data_len_chars",
    "xgt_data_num_spaces",
    "xgt_data_is_hex",
    "xgt_data_n_bytes",
    "xgt_data_zero_ratio",
    "xgt_data_first_byte",
    "xgt_data_last_byte",
    "xgt_data_mean_byte",
    "xgt_data_bucket",
    "arp_src_host_id",
    "arp_tgt_host_id",
    "arp_op_num",
    "dns_qc_norm",
    "dns_ac_norm",
]

# ------------------------------------------------------------
# [2] 항상 포함할 필수 feature
#     → 모든 조합에서 무조건 포함되는 컬럼
# ------------------------------------------------------------
ESSENTIAL_FEATURES: list[str] = [
    "protocol_norm",
    "len_norm",
    "s7comm_fn",
    "modbus_addr_norm",
    "modbus_fc_norm",
    "xgt_cmd",
]

# OPTIONAL_FEATURES = ALL_FEATURES - ESSENTIAL - EXCLUDED
OPTIONAL_FEATURES = [
    f for f in ALL_FEATURES
    if f not in ESSENTIAL_FEATURES and f not in EXCLUDED_FEATURES
]

# 조합 크기 범위 설정
#   - COMBO_MIN_K = 0  이면: "필수만 사용하는 경우"도 포함
#   - COMBO_MAX_K = None 이면: len(OPTIONAL_FEATURES) 까지 전부
COMBO_MIN_K = 0
COMBO_MAX_K: int | None = None


# ============================================================
# 유틸 함수들
# ============================================================
def run_cmd(cmd, cwd: Path):
    print("\n[RUN] (cwd =", cwd, ")")
    print("   ", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        print(f"[ERROR] 명령 실패 (exit code={result.returncode}): {' '.join(str(c) for c in cmd)}")
        sys.exit(result.returncode)


def write_exclude_for_feature_set(
    exclude_path: Path,
    all_features: list[str],
    excluded_features: list[str],
    keep_features: list[str],
) -> None:
    if not keep_features:
        raise RuntimeError("keep_features 가 비어 있습니다. 최소 1개 이상의 feature는 남아야 합니다.")

    unknown = [f for f in keep_features if f not in all_features]
    if unknown:
        raise RuntimeError(
            f"keep_features 에 ALL_FEATURES 에 없는 feature 가 있습니다: {unknown}"
        )

    conflict = [f for f in keep_features if f in excluded_features]
    if conflict:
        raise RuntimeError(
            f"keep_features 에 항상 제외해야 하는 feature(EXCLUDED_FEATURES)가 포함되어 있습니다: {conflict}"
        )

    to_exclude: list[str] = []

    for f in all_features:
        if f in excluded_features:
            to_exclude.append(f)
        else:
            if f not in keep_features:
                to_exclude.append(f)

    if len(to_exclude) == len(all_features):
        raise RuntimeError(
            "all_features 전부가 exclude 대상입니다. "
            "즉, 남는 feature가 없습니다. ESSENTIAL_FEATURES/EXCLUDED_FEATURES 설정을 확인하세요."
        )

    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    with exclude_path.open("w", encoding="utf-8") as f:
        for name in to_exclude:
            f.write(name + "\n")


def safe_dir_name(name: str) -> str:
    bad_chars = ['/', '\\', ' ', ':', '*', '?', '"', '<', '>', '|', ',']
    safe = name
    for ch in bad_chars:
        safe = safe.replace(ch, "_")
    return safe


def build_combos(optional_features: list[str]) -> list[list[str]]:
    if not optional_features and COMBO_MIN_K > 0:
        return []

    min_k = max(0, COMBO_MIN_K)
    max_k = len(optional_features) if COMBO_MAX_K is None else min(COMBO_MAX_K, len(optional_features))

    combos: list[list[str]] = []
    for k in range(min_k, max_k + 1):
        for combo in combinations(optional_features, k):
            combos.append(list(combo))
    return combos


# ============================================================
# 메인 로직
# ============================================================
def main():
    base_dir = Path(__file__).resolve().parent  # DL/ 디렉토리

    if not ALL_FEATURES:
        raise RuntimeError("ALL_FEATURES 가 비어 있습니다.")

    print("========== feature_single_run.py (combo + exclude mode) ==========")
    print(f"[INFO] window_size  = {WINDOW_SIZE}")
    print(f"[INFO] step_size    = {STEP_SIZE if STEP_SIZE is not None else WINDOW_SIZE}")
    print(f"[INFO] threshold    = {THRESHOLD if THRESHOLD is not None else 'None (threshold.json / -1 사용)'}")
    print(f"[INFO] #ALL_FEATURES       = {len(ALL_FEATURES)}")
    print(f"[INFO] #EXCLUDED_FEATURES  = {len(EXCLUDED_FEATURES)}")
    print(f"[INFO] #ESSENTIAL_FEATURES = {len(ESSENTIAL_FEATURES)}")
    print(f"[INFO] #OPTIONAL_FEATURES  = {len(OPTIONAL_FEATURES)}")
    print("  EXCLUDED :", ", ".join(EXCLUDED_FEATURES) if EXCLUDED_FEATURES else "(none)")
    print("  ESSENTIAL:", ", ".join(ESSENTIAL_FEATURES) if ESSENTIAL_FEATURES else "(none)")
    print("  OPTIONAL :", ", ".join(OPTIONAL_FEATURES) if OPTIONAL_FEATURES else "(none)")

    union_set = set(EXCLUDED_FEATURES).union(ESSENTIAL_FEATURES).union(OPTIONAL_FEATURES)
    if union_set != set(ALL_FEATURES):
        print("⚠ 경고: EXCLUDED + ESSENTIAL + OPTIONAL 이 ALL_FEATURES 와 정확히 일치하지 않습니다.")
        print("   - 일부 feature 가 누락/중복되었을 수 있습니다. 계속 진행은 하지만 결과를 주의하세요.")

    if set(EXCLUDED_FEATURES).intersection(ESSENTIAL_FEATURES):
        print("⚠ 경고: EXCLUDED_FEATURES 와 ESSENTIAL_FEATURES 가 겹칩니다. 겹치는 feature 는 제외 리스트로 취급됩니다.")

    train_script = base_dir / "train" / "code" / "0.run_pipeline_pattern.py"
    result_script = base_dir / "result_train" / "code" / "0.run_pipeline_pattern.py"

    if not train_script.exists():
        print(f"[ERROR] train 쪽 0.run_pipeline_pattern.py 를 찾을 수 없습니다: {train_script}")
        sys.exit(1)
    if not result_script.exists():
        print(f"[ERROR] result_train 쪽 0.run_pipeline_pattern.py 를 찾을 수 없습니다: {result_script}")
        sys.exit(1)

    exclude_path = base_dir / "train" / "data" / "exclude.txt"
    analyze_src = base_dir / "result_train" / "result" / "analyze_mse_dist.json"
    analyze_dst_root = base_dir / "result_train" / "result" / "feature_combo"

    step_size = STEP_SIZE if (STEP_SIZE is not None) else WINDOW_SIZE

    combos = build_combos(OPTIONAL_FEATURES)
    total_combos = len(combos)

    print(f"[INFO] OPTIONAL_FEATURES 조합 개수 = {total_combos} (COMBO_MIN_K={COMBO_MIN_K}, COMBO_MAX_K={COMBO_MAX_K})")
    print("========================================================\n")

    if total_combos == 0:
        print("⚠ 생성된 조합이 없습니다. ESSENTIAL_FEATURES / OPTIONAL_FEATURES / COMBO_MIN_K / COMBO_MAX_K 를 확인하세요.")
        return

    original_exclude = None
    if exclude_path.exists():
        original_exclude = exclude_path.read_text(encoding="utf-8")

    all_combo_summaries: list[dict] = []

    try:
        for idx, opt_feats in enumerate(combos, start=1):
            keep_features = list(ESSENTIAL_FEATURES) + list(opt_feats)

            combo_label = "+".join(sorted(keep_features))
            # 🔹 여기서부터: safe_label 대신 숫자 폴더 이름 사용
            folder_name = str(idx)  # "1", "2", "3", ...
            print("=" * 80)
            print(f"[{idx}/{total_combos}] Feature 조합 사용 (folder={folder_name}):")
            print(f"    keep_features ({len(keep_features)}개): {combo_label}")
            print("=" * 80)

            write_exclude_for_feature_set(exclude_path, ALL_FEATURES, EXCLUDED_FEATURES, keep_features)
            print(f"→ exclude.txt 갱신 (EXCLUDED + 이 조합에 포함되지 않은 feature 제외)")

            cmd_train = [
                sys.executable,
                str(train_script),
                "--window-size", str(WINDOW_SIZE),
                "--epochs", str(EPOCHS),
                "--batch-size", str(BATCH_SIZE),
                "--hidden-dim", str(HIDDEN_DIM),
                "--latent-dim", str(LATENT_DIM),
            ]
            run_cmd(cmd_train, cwd=train_script.parent)

            cmd_result = [
                sys.executable,
                str(result_script),
                "--window-size", str(WINDOW_SIZE),
                "--step-size", str(step_size),
            ]
            if THRESHOLD is not None:
                cmd_result.extend(["--threshold", str(THRESHOLD)])

            run_cmd(cmd_result, cwd=result_script.parent)

            if not analyze_src.exists():
                print(f"⚠ 경고: {analyze_src} 가 생성되지 않았습니다. 이 조합은 스킵합니다.", file=sys.stderr)
                continue

            # 🔹 디렉토리 이름 = "1", "2", "3", ...
            dst_dir = analyze_dst_root / folder_name
            dst_dir.mkdir(parents=True, exist_ok=True)

            dst_path = dst_dir / "analyze_mse_dist.json"
            shutil.copy2(analyze_src, dst_path)

            meta_path = dst_dir / "features.txt"
            with meta_path.open("w", encoding="utf-8") as f:
                f.write("# keep_features (ESSENTIAL + OPTIONAL 조합)\n")
                for name in keep_features:
                    f.write(name + "\n")
                f.write("\n# 항상 제외된 features:\n")
                for name in EXCLUDED_FEATURES:
                    f.write(name + "\n")

            selected_json_path = dst_dir / "selected_features.json"
            combo_info = {
                "combo_index": idx,
                "folder_name": folder_name,
                "combo_label": combo_label,
                "essential_features": ESSENTIAL_FEATURES,
                "optional_features": list(opt_feats),
                "keep_features": keep_features,
                "excluded_features": EXCLUDED_FEATURES,
            }
            with selected_json_path.open("w", encoding="utf-8") as jf:
                json.dump(combo_info, jf, ensure_ascii=False, indent=2)

            all_combo_summaries.append(combo_info)

            print(f"→ {analyze_src} 를 {dst_path} 로 복사 완료")
            print(f"→ 사용 feature 목록은 {meta_path} / {selected_json_path} 에 저장\n")

    except subprocess.CalledProcessError as e:
        print("❌ 외부 스크립트 실행 중 오류 발생:", e, file=sys.stderr)
        print("  - returncode:", e.returncode, file=sys.stderr)
    finally:
        if all_combo_summaries:
            summary_path = analyze_dst_root / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", encoding="utf-8") as jf:
                json.dump(all_combo_summaries, jf, ensure_ascii=False, indent=2)
            print(f"\n[INFO] 전체 조합 요약을 {summary_path} 에 저장했습니다.")

        if original_exclude is not None:
            exclude_path.write_text(original_exclude, encoding="utf-8")
            print("\n[INFO] exclude.txt 를 원래 내용으로 복구했습니다.")
        else:
            if exclude_path.exists():
                exclude_path.unlink()
                print("\n[INFO] exclude.txt 를 삭제해 원래 상태로 돌렸습니다.")


if __name__ == "__main__":
    main()