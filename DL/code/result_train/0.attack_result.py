#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


ML_MODEL = None
ML_SCALER = None
ML_SELECTED_FEATURES: Optional[List[str]] = None
ML_META: Dict[str, Any] = {}
ML_THRESHOLD: Optional[float] = None
FEATURE_NAMES_CACHE: Optional[List[str]] = None

_DL_ROOT_CACHE: Optional[Path] = None

REQUIRED_PRE_FILES = [
    "common_host_map.json",
    "common_norm_params.json",
    "s7comm_norm_params.json",
    "modbus_norm_params.json",
    "xgt_var_vocab.json",
    "xgt_norm_params.json",
    "arp_host_map.json",
    "dns_norm_params.json",
]


def safe_int(val: Any, default: int = 0) -> int:
    try:
        if isinstance(val, list):
            for v in val:
                if v is not None:
                    val = v
                    break
        if val is None:
            return default
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return default
            if s.startswith(("0x", "0X")):
                return int(s, 16)
            return int(s)
        return int(val)
    except Exception:
        return default


def normalize_hex_string(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    return "".join(s.split()).lower()


def is_all_zero_data(hex_str: str) -> bool:
    return bool(hex_str) and all(ch == "0" for ch in hex_str)


def dl_root() -> Path:
    global _DL_ROOT_CACHE
    if _DL_ROOT_CACHE is not None:
        return _DL_ROOT_CACHE

    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if p.name.lower() == "dl":
            _DL_ROOT_CACHE = p
            return p

    for p in [here.parent, *here.parents]:
        if (p / "code" / "preprocessing" / "2.extract_feature.py").exists():
            _DL_ROOT_CACHE = p
            return p

    raise FileNotFoundError(f"Cannot locate DL root from: {here}")


def extract_feature_script() -> Path:
    p = dl_root() / "code" / "preprocessing" / "2.extract_feature.py"
    if not p.exists():
        raise FileNotFoundError(f"2.extract_feature.py not found: {p}")
    return p


def default_model_loader_path() -> Path:
    p = dl_root() / "operating" / "ML" / "code" / "1-1.model_load.py"
    if p.exists():
        return p
    return Path(__file__).resolve().parent / "1-1.model_load.py"


def _check_pre_dir(p: Path) -> bool:
    return p.is_dir() and all((p / fn).exists() for fn in REQUIRED_PRE_FILES)


def resolve_pre_dir(user_pre_dir: Union[str, Path]) -> Path:
    raw = Path(user_pre_dir).expanduser()
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    dl = dl_root()

    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([cwd / raw, here / raw, dl / raw])

    candidates.extend(
        [
            dl / "preprocessing" / "result",
            dl / "preprocessing" / "results",
            dl / "code" / "preprocessing" / "result",
            dl / "code" / "preprocessing" / "results",
        ]
    )

    seen = set()
    for c in candidates:
        c = c.resolve()
        if str(c) in seen:
            continue
        seen.add(str(c))
        if _check_pre_dir(c):
            return c

    search_roots = [c.resolve() for c in candidates if c.exists()]
    search_roots.append(dl)

    for root in search_roots:
        try:
            hits = list(root.rglob("common_host_map.json"))
        except Exception:
            continue
        for h in hits[:80]:
            parent = h.parent.resolve()
            if _check_pre_dir(parent):
                return parent

    tried = "\n".join(f"- {c.resolve()}" for c in candidates[:25])
    raise FileNotFoundError(
        "pre_dir를 찾을 수 없습니다. (필수 파일 세트가 있는 폴더가 필요)\n"
        f"요청 pre_dir: {raw}\n"
        "다음 경로들을 시도했지만 실패했습니다:\n"
        f"{tried}\n\n"
        "pre_dir 폴더에는 아래 파일들이 모두 있어야 합니다:\n"
        + "\n".join(f"- {x}" for x in REQUIRED_PRE_FILES)
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--input", "-i", required=True)
    p.add_argument("--window-size", "-w", type=int, default=80)
    p.add_argument("--step-size", "-s", type=int, default=None)
    p.add_argument("--output", "-o", required=True)

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
    )

    p.add_argument("--pre-dir", default=None)
    p.add_argument("--feat-output1", default=None)

    p.add_argument("--model-dir", default=None)
    p.add_argument("--model-loader", default=None)
    p.add_argument("--ml-threshold", type=float, default=None)

    return p.parse_args(argv)


def load_model_bundle_from_file(model_dir: Union[Path, str], loader_path: Union[Path, str]):
    loader_path = Path(loader_path)
    model_dir = Path(model_dir)

    if not loader_path.exists():
        raise FileNotFoundError(f"model loader not found: {loader_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"model dir not found: {model_dir}")

    spec = importlib.util.spec_from_file_location("model_loader", str(loader_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    if not hasattr(module, "load_model_bundle"):
        raise AttributeError(f"load_model_bundle not found in loader: {loader_path}")

    return module.load_model_bundle(model_dir)


def is_modbus_fc6(pkt: Dict[str, Any]) -> bool:
    return pkt.get("protocol") == "modbus" and safe_int(pkt.get("modbus.fc"), -1) == 6


def is_xgt_write(pkt: Dict[str, Any]) -> bool:
    if pkt.get("protocol") != "xgt_fen":
        return False
    cmd = safe_int(pkt.get("xgt_fen.cmd"), -1)
    return cmd in (0x58, 0x59)


def is_xgt_word_zero(pkt: Dict[str, Any], word_idx: int) -> bool:
    if pkt.get("protocol") != "xgt_fen":
        return False
    if pkt.get("xgt_fen.vars") != "%DB001046":
        return False

    hex_str = normalize_hex_string(pkt.get("xgt_fen.data"))
    if not hex_str or is_all_zero_data(hex_str):
        return False

    end = (word_idx * 4) + 4
    if len(hex_str) < end:
        return False
    return hex_str[word_idx * 4:end] == "0000"


def is_xgt_head_anom(pkt: Dict[str, Any]) -> bool:
    if pkt.get("protocol") != "xgt_fen":
        return False
    if pkt.get("xgt_fen.vars") != "%DB001046":
        return False

    hex_str = normalize_hex_string(pkt.get("xgt_fen.data"))
    if not hex_str or is_all_zero_data(hex_str) or len(hex_str) < 4:
        return False
    return hex_str[0:4] != "0500"


def infer_attack_mode_from_filename(filename: str) -> str:
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


def normalize_mode(mode: str, input_name: str) -> str:
    if mode == "auto":
        return infer_attack_mode_from_filename(input_name)
    if mode in ("xgt_d528_zero", "xgt_last_zero"):
        return "xgt_last_zero"
    if mode in ("xgt_d525_zero", "xgt_mid_zero"):
        return "xgt_mid_zero"
    return mode


def compute_attack_flags(packets: List[Dict[str, Any]], mode: str) -> List[bool]:
    flags: List[bool] = []
    for pkt in packets:
        if is_xgt_write(pkt):
            flags.append(True)
            continue

        if mode == "modbus_fc6":
            flags.append(is_modbus_fc6(pkt))
        elif mode == "xgt_last_zero":
            flags.append(is_xgt_word_zero(pkt, 5))
        elif mode == "xgt_mid_zero":
            flags.append(is_xgt_word_zero(pkt, 4))
        elif mode == "xgt_head_zero":
            flags.append(is_xgt_head_anom(pkt))
        elif mode == "xgt_write_or_fc6":
            flags.append(is_modbus_fc6(pkt))
        elif mode == "xgt_write_or_fc6_after":
            flags.append(is_modbus_fc6(pkt) or is_xgt_head_anom(pkt))
        else:
            flags.append(False)
    return flags


def infer_match_with_ml(feat_row: Dict[str, Any]) -> Optional[int]:
    global FEATURE_NAMES_CACHE

    if ML_MODEL is None or not isinstance(feat_row, dict):
        return None

    if FEATURE_NAMES_CACHE is None:
        if ML_SELECTED_FEATURES:
            FEATURE_NAMES_CACHE = list(ML_SELECTED_FEATURES)
        elif ML_SCALER is not None and hasattr(ML_SCALER, "feature_names_in_"):
            FEATURE_NAMES_CACHE = list(getattr(ML_SCALER, "feature_names_in_"))
        elif hasattr(ML_MODEL, "feature_names_in_"):
            FEATURE_NAMES_CACHE = list(getattr(ML_MODEL, "feature_names_in_"))
        else:
            exclude = {"window_id", "window_index", "pattern", "match"}
            FEATURE_NAMES_CACHE = sorted(k for k in feat_row.keys() if k not in exclude)

    feature_names = FEATURE_NAMES_CACHE or []
    if not feature_names:
        return None

    vec: List[float] = []
    getv = feat_row.get
    for f in feature_names:
        v = getv(f, 0.0)
        try:
            vec.append(float(0.0 if v is None else v))
        except Exception:
            vec.append(0.0)

    X = np.asarray([vec], dtype=float)
    if ML_SCALER is not None:
        X = ML_SCALER.transform(X)

    if hasattr(ML_MODEL, "decision_function"):
        score = float(ML_MODEL.decision_function(X)[0])
    elif hasattr(ML_MODEL, "predict_proba"):
        proba = ML_MODEL.predict_proba(X)[0]
        score = float(proba[1]) if len(proba) > 1 else float(proba[0])
    else:
        score = float(ML_MODEL.predict(X)[0])

    thr = float(ML_THRESHOLD if ML_THRESHOLD is not None else 0.0)
    higher_is_anom = bool(ML_META.get("score_higher_is_anom", True))
    is_anom = (score >= thr) if higher_is_anom else (score <= thr)
    return 0 if is_anom else 1


def apply_ml_to_feature_jsonl(jsonl_path: Path) -> None:
    if not jsonl_path.exists():
        print(f"[WARN] feature jsonl not found: {jsonl_path}")
        return

    tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".mltmp")

    n_win = 0
    n_rows = 0
    n0 = n1 = nn = 0

    with jsonl_path.open("r", encoding="utf-8-sig") as fin, tmp_path.open("w", encoding="utf-8-sig") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] line {line_no} json parse fail: {e}")
                continue

            seq_list = obj.get("sequence_group")
            if not isinstance(seq_list, list):
                seq_list = []

            if ML_MODEL is not None:
                for row in seq_list:
                    if not isinstance(row, dict):
                        continue
                    n_rows += 1
                    m = infer_match_with_ml(row)
                    row["match"] = m
                    if m == 0:
                        n0 += 1
                    elif m == 1:
                        n1 += 1
                    else:
                        nn += 1

            obj.pop("match", None)
            win_idx = obj.get("window_index", obj.get("window_id"))

            fout.write(
                json.dumps(
                    {"window_index": win_idx, "pattern": obj.get("pattern"), "sequence_group": seq_list},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_win += 1

    tmp_path.replace(jsonl_path)

    if ML_MODEL is not None:
        print(f"[INFO] ML match + slim done: {jsonl_path}")
        print(f"[INFO] windows={n_win}, rows={n_rows}, match0={n0}, match1={n1}, matchNone={nn}")
    else:
        print(f"[INFO] slim done (no ML): {jsonl_path}")


def read_jsonl_packets(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def write_windows_jsonl(
    packets: List[Dict[str, Any]],
    attack_flags: List[bool],
    out_path: Path,
    window_size: int,
    step_size: int,
) -> int:
    total = len(packets)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    win_count = 0
    with out_path.open("w", encoding="utf-8-sig") as fout:
        start = 0
        while start < total:
            end = start + window_size
            window_packets = packets[start:end]
            if not window_packets:
                break

            has_attack = any(attack_flags[start:end])
            valid_len = len(window_packets)

            fout.write(
                json.dumps(
                    {
                        "window_id": win_count,
                        "pattern": "ATTACK" if has_attack else "NORMAL",
                        "index": list(range(valid_len)),
                        "sequence_group": window_packets,
                        "description": None,
                        "start_packet_idx": start,
                        "end_packet_idx": start + valid_len - 1,
                        "is_anomaly": 1 if has_attack else 0,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            win_count += 1
            start += step_size

    return win_count


def maybe_load_ml(args: argparse.Namespace) -> None:
    global ML_MODEL, ML_SCALER, ML_SELECTED_FEATURES, ML_META, ML_THRESHOLD, FEATURE_NAMES_CACHE

    FEATURE_NAMES_CACHE = None

    if not args.model_dir:
        ML_MODEL = None
        ML_SCALER = None
        ML_SELECTED_FEATURES = None
        ML_META = {}
        ML_THRESHOLD = None
        return

    model_dir = Path(args.model_dir).expanduser().resolve()
    loader_path = Path(args.model_loader).expanduser().resolve() if args.model_loader else default_model_loader_path()

    print(f"[INFO] ML load: model_dir={model_dir}")
    print(f"[INFO] loader : {loader_path}")

    ML_MODEL, ML_SCALER, ML_SELECTED_FEATURES, ML_META = load_model_bundle_from_file(model_dir, loader_path)

    if ML_SCALER is not None and hasattr(ML_SCALER, "feature_names_in_"):
        try:
            delattr(ML_SCALER, "feature_names_in_")
        except Exception:
            pass

    ML_THRESHOLD = float(args.ml_threshold) if args.ml_threshold is not None else float(ML_META.get("threshold", 0.0))
    print(f"[INFO] ML threshold={ML_THRESHOLD}")


def run_once(argv: List[str]) -> None:
    args = parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    window_size = int(args.window_size)
    step_size = int(args.step_size) if args.step_size is not None else window_size
    if window_size <= 0:
        raise ValueError("window-size must be > 0")
    if step_size <= 0:
        raise ValueError("step-size must be > 0")

    attack_mode = normalize_mode(args.mode, input_path.name)
    maybe_load_ml(args)

    packets = read_jsonl_packets(input_path)
    total_packets = len(packets)
    print(f"[INFO] total_packets={total_packets}")
    if total_packets == 0:
        print("[WARN] no packets, exit")
        return

    print("[INFO] precompute attack flags...")
    attack_flags = compute_attack_flags(packets, attack_mode)

    win_count = write_windows_jsonl(packets, attack_flags, output_path, window_size, step_size)
    print(f"[INFO] 1단계 완료: windows={win_count} -> {output_path}")

    if not (args.pre_dir and args.feat_output1):
        print("[INFO] pre_dir/feat-output1 미설정 -> feature 추출/ML 적용 스킵")
        return

    script = extract_feature_script()
    pre_dir = resolve_pre_dir(args.pre_dir)

    feat_out = Path(args.feat_output1).expanduser().resolve()
    feat_out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script),
        "-i", str(output_path),
        "-p", str(pre_dir),
        "-o", str(feat_out),
    ]

    print("[INFO] 2단계: 2.extract_feature.py 실행")
    print("       cmd:", " ".join(cmd))

    subprocess.run(cmd, check=True)
    apply_ml_to_feature_jsonl(feat_out)


def _resolve_from_base(base_dir: Path, v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        raise RuntimeError("PyYAML이 필요합니다. 설치: pip install pyyaml")

    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid YAML root (dict expected): {path}")
    return obj


def run_from_yaml(config_path: Path, job_name: Optional[str], list_jobs: bool, dry_run: bool) -> None:
    cfg = _load_yaml(config_path)
    defaults = cfg.get("defaults") if isinstance(cfg.get("defaults"), dict) else {}
    jobs = cfg.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("YAML에 jobs: [ ... ] 리스트가 필요합니다.")

    if list_jobs:
        for j in jobs:
            if isinstance(j, dict):
                print(j.get("name"))
        return

    base_dir = config_path.parent.resolve()

    for j in jobs:
        if not isinstance(j, dict):
            continue

        name = str(j.get("name", "")).strip()
        if job_name and name != job_name:
            continue

        enabled = j.get("enabled", True)
        if enabled is False:
            continue

        ws = int(j.get("window_size", defaults.get("window_size", 80)))
        ss = int(j.get("step_size", defaults.get("step_size", ws)))
        mode = str(j.get("mode", defaults.get("mode", "auto")))

        inp = _resolve_from_base(base_dir, j.get("input"))
        out = _resolve_from_base(base_dir, j.get("output"))
        if not inp or not out:
            raise ValueError(f"job '{name}': input/output 필수")

        run_extract = bool(j.get("run_extract", defaults.get("run_extract", False)))
        run_ml = bool(j.get("run_ml", defaults.get("run_ml", False)))

        pre_dir = _resolve_from_base(base_dir, j.get("pre_dir", defaults.get("pre_dir")))
        feat_out = _resolve_from_base(base_dir, j.get("feat_output1"))

        model_dir = _resolve_from_base(base_dir, j.get("model_dir", defaults.get("model_dir")))
        model_loader = _resolve_from_base(base_dir, j.get("model_loader", defaults.get("model_loader")))
        ml_thr = j.get("ml_threshold", defaults.get("ml_threshold", None))

        argv = [
            "--input", inp,
            "--window-size", str(ws),
            "--step-size", str(ss),
            "--output", out,
            "--mode", mode,
        ]

        if run_extract:
            if not pre_dir or not feat_out:
                raise ValueError(f"job '{name}': run_extract=true 이면 pre_dir/feat_output1 필요")
            argv += ["--pre-dir", pre_dir, "--feat-output1", feat_out]

        if run_ml:
            if not model_dir:
                raise ValueError(f"job '{name}': run_ml=true 이면 model_dir 필요")
            argv += ["--model-dir", model_dir]
            if model_loader:
                argv += ["--model-loader", model_loader]
            if ml_thr is not None:
                argv += ["--ml-threshold", str(float(ml_thr))]

        print(f"\n[INFO] YAML job: {name or '(no-name)'}")
        print("       argv:", " ".join(argv))

        if dry_run:
            continue

        run_once(argv)


def entry() -> None:
    script_dir = Path(__file__).resolve().parent
    default_yaml = Path(__file__).with_suffix(".yaml")

    if len(sys.argv) == 1:
        if not default_yaml.exists():
            raise FileNotFoundError(f"default yaml not found: {default_yaml}")
        run_from_yaml(default_yaml, job_name=None, list_jobs=False, dry_run=False)
        return

    if ("--input" in sys.argv) or ("-i" in sys.argv):
        run_once(sys.argv[1:])
        return

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(default_yaml), help="path to 0.attack_result.yaml")
    p.add_argument("--job", default=None, help="run only one job by name")
    p.add_argument("--list-jobs", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    ns, unknown = p.parse_known_args()
    if unknown:
        raise SystemExit(f"Unknown args (use --input mode or yaml mode): {unknown}")

    cfg_path = Path(ns.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (script_dir / cfg_path).resolve()

    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")

    run_from_yaml(cfg_path, job_name=ns.job, list_jobs=ns.list_jobs, dry_run=ns.dry_run)


if __name__ == "__main__":
    entry()
