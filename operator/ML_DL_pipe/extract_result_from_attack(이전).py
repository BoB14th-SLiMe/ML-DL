#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_result_from_attack.py (HARD-CODED: full packets -> ML per packet + DL sliding windows)

- window_size=80, step=5
- 전체 패킷: ML 예측(패킷 단위)
- 전체 윈도우: DL 예측(윈도우 단위) + 패턴 매핑(DL-pattern)
- 출력: windows_pred.jsonl (1 line = 1 window)

요구 반영:
- 저장 시(window_raw)에서 redis_id, ml_anomaly_score 제거 (내부 추론에는 유지)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

JsonDict = Dict[str, Any]

# ============================================================
# HARD-CODED CONFIG
# ============================================================
TOPK: int = 2

WINDOW_SIZE: int = 80
STEP: int = 5

DL_BATCH_SIZE: int = 128         # anomaly(LSTM-AE) batch
PAT_BATCH_SIZE: int = 256        # pattern(torch) batch

ALWAYS_RUN_PATTERN: bool = True  # normal이어도 pattern 추론 수행
RISK_ZERO_WHEN_NORMAL: bool = True

INPUT_JSONL_NAME: str = "attack_ver2.jsonl"
OUTPUT_JSONL_NAME: str = "windows_pred(attack_ver2).jsonl"
# ============================================================

# ============================================================
# OUTPUT STRIP (저장 시 제거할 origin 필드)
# ============================================================
DROP_ORIGIN_KEYS = {
    "redis_id",
    "ml_anomaly_score",
    # 필요 시 아래도 추가 가능
    # "ml_anomaly_prob",
    # "ml_anomaly_prob_topk",
}
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_predict.ML_predict import (  # noqa: E402
    MLAnomalyProbEngine,
    predict_enrich_origin_records_with_bundle_fast,
)

from model_predict.DL_predict import (  # noqa: E402
    load_lstm_ae_bundle,
    load_dl_pattern_bundle,          # ✅ DL-pattern loader (DL_predict.py에 구현돼 있어야 함)
    predict_dl_models_windows_sliding,  # ✅ anomaly + pattern + risk 통합
)


# ============================================================
# Utils
# ============================================================
def _import_from_file(mod_name: str, py_path: Path):
    import importlib.util

    py_path = Path(py_path)
    if not py_path.exists():
        raise FileNotFoundError(str(py_path))

    spec = importlib.util.spec_from_file_location(mod_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"spec load failed: {py_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sanitize(x: Any) -> Any:
    try:
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            v = float(x)
            return None if (not np.isfinite(v)) else v
        if isinstance(x, (np.ndarray,)):
            return [_sanitize(v) for v in x.tolist()]
    except Exception:
        pass
    if isinstance(x, float):
        return None if (not np.isfinite(x)) else x
    if isinstance(x, dict):
        return {str(k): _sanitize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_sanitize(v) for v in x]
    return x


def _strip_origin_for_save(origin: Dict[str, Any]) -> Dict[str, Any]:
    """
    저장 파일(window_raw)에만 적용.
    내부 추론/파이프라인에서는 redis_id, ml_anomaly_score를 계속 사용.
    """
    o = dict(origin)  # shallow copy
    for k in DROP_ORIGIN_KEYS:
        if k in o:
            o.pop(k, None)
    return o


def _resolve_input_jsonl(script_dir: Path) -> Path:
    for p in [
        script_dir / "attack_data" / INPUT_JSONL_NAME,
        script_dir.parent / "attack_data" / INPUT_JSONL_NAME,
    ]:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError("input JSONL not found")


def _resolve_output_jsonl(script_dir: Path) -> Path:
    return (script_dir / OUTPUT_JSONL_NAME).resolve()


def _resolve_pre_dir(script_dir: Path) -> Path:
    for p in [
        script_dir / "preprocessing" / "result",
        script_dir.parent / "preprocessing" / "result",
        script_dir.parent.parent / "preprocessing" / "result",
    ]:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError("preprocessing result dir not found")


def _resolve_model_load_dir(script_dir: Path) -> Path:
    for p in [
        script_dir / "model_load",
        script_dir / ".model_load",
        script_dir.parent / "model_load",
        script_dir.parent / ".model_load",
    ]:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError("model_load dir not found")


def _resolve_dl_anom_model_dir(model_load_dir: Path) -> Path:
    model_load_dir = Path(model_load_dir).resolve()
    cands = [
        model_load_dir / "DL_model_anomaly",
        model_load_dir / "DL_model",
        model_load_dir / "DL_anomaly",
        model_load_dir / "lstm_ae",
    ]
    for d in cands:
        if d.exists() and (d / "model.h5").exists() and (d / "config.json").exists() and (d / "feature_keys.txt").exists():
            return d.resolve()

    for h5 in model_load_dir.rglob("model.h5"):
        d = h5.parent
        if (d / "config.json").exists() and (d / "feature_keys.txt").exists():
            return d.resolve()

    raise FileNotFoundError(f"DL-anomaly model dir not found under: {model_load_dir}")


def _resolve_dl_pattern_loader_py(model_load_dir: Path) -> Path:
    """
    model_load(DL-pattern).py 자동 탐색
    """
    model_load_dir = Path(model_load_dir).resolve()
    cands = [
        model_load_dir / "model_load(DL-pattern).py",
        model_load_dir / "model_load_DL-pattern.py",
    ]
    for p in cands:
        if p.exists():
            return p.resolve()

    found = list(model_load_dir.rglob("model_load(DL-pattern).py"))
    if found:
        found.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return found[0].resolve()

    raise FileNotFoundError(f"model_load(DL-pattern).py not found under: {model_load_dir}")


def _resolve_dl_pattern_ckpt_dir(model_load_dir: Path) -> Path:
    """
    DL-pattern 체크포인트(best_model.* / checkpoint.* / model.*)가 있는 디렉토리 자동 탐색
    """
    model_load_dir = Path(model_load_dir).resolve()

    preferred_dirs = [
        model_load_dir / "DL_model_pattern",
        model_load_dir / "DL_pattern",
        model_load_dir / "DL_model_label",
        model_load_dir / "pattern",
    ]
    preferred_names = [
        "best_model.h5", "best_model.pt", "best_model.pth",
        "checkpoint.h5", "checkpoint.pt", "checkpoint.pth",
        "model.h5", "model.pt", "model.pth",
    ]

    for d in preferred_dirs:
        if d.exists() and d.is_dir():
            for nm in preferred_names:
                if (d / nm).exists():
                    return d.resolve()

    hits: List[Path] = []
    for nm in preferred_names:
        hits.extend(model_load_dir.rglob(nm))
    if hits:
        hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return hits[0].parent.resolve()

    raise FileNotFoundError(f"DL-pattern checkpoint dir not found under: {model_load_dir}")


def _normalize_protocol_for_features(proto: str) -> str:
    return "xgt_fen" if proto == "xgt-fen" else proto


def try_init_featurizer(pre_dir: Path):
    from preprocessing.packet_feature_preprocessor import PacketFeaturePreprocessor
    return PacketFeaturePreprocessor(
        pre_dir,
        allow_new_ids=False,
        index_source="redis_id",
        include_index=False,
    )


def build_features(origin: Dict[str, Any], fz, *, redis_id: str) -> Dict[str, Any]:
    origin_for_feat = dict(origin)
    origin_for_feat["protocol"] = _normalize_protocol_for_features(str(origin_for_feat.get("protocol", "")))
    wrapped = {"origin": origin_for_feat, "_meta": {"redis_id": redis_id}}
    if hasattr(fz, "preprocess"):
        out = fz.preprocess(wrapped)
    else:
        out = fz.process(wrapped)
    if isinstance(out, dict) and isinstance(out.get("features"), dict):
        return out["features"]
    return out if isinstance(out, dict) else {}


def _unwrap_origin_and_redis(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict) and isinstance(obj.get("origin"), dict):
        origin = obj["origin"]
        meta = obj.get("_meta") or {}
        rid = meta.get("redis_id")
        if rid is not None and "redis_id" not in origin:
            origin["redis_id"] = rid
        return origin
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"packet not dict: {type(obj)}")


def load_all_packets(path: Path) -> List[Dict[str, Any]]:
    packets: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            packets.append(_unwrap_origin_and_redis(obj))
    return packets


def load_ml_bundle(model_load_dir: Path) -> Dict[str, Any]:
    model_load_dir = Path(model_load_dir).resolve()
    ml_dir = (model_load_dir / "ML_model").resolve()
    ml_loader_py = (model_load_dir / "model_load(ML).py").resolve()
    if not ml_loader_py.exists() or not ml_dir.exists():
        raise FileNotFoundError("ML loader or model dir missing")

    ml_mod = _import_from_file("_ml_loader_full", ml_loader_py)
    model, scaler, selected_features, metadata = ml_mod.load_model_bundle(ml_dir)

    return {
        "model": model,
        "scaler": scaler,
        "selected_features": list(selected_features or []),
        "metadata": metadata or {},
        "ml_dir": str(ml_dir),
        "ml_loader_py": str(ml_loader_py),
    }


def _infer_normal_label(model: Any, metadata: Dict[str, Any]) -> int:
    try:
        nl = metadata.get("normal_label", None)
        if nl is not None:
            return int(nl)
    except Exception:
        pass
    mod = getattr(type(model), "__module__", "") or ""
    if mod.startswith("pyod."):
        return 0
    return 1


def _fill_risk_meta(summary: Dict[str, Any], window_raw: List[Dict[str, Any]]) -> None:
    if not window_raw:
        return
    last = window_raw[-1]
    risk = summary.get("risk")
    if not isinstance(risk, dict):
        risk = {}
        summary["risk"] = risk

    risk["detected_time"] = last.get("@timestamp")
    risk["src_ip"] = last.get("sip")
    risk["src_asset"] = last.get("src_asset")
    risk["dst_ip"] = last.get("dip")
    risk["dst_asset"] = last.get("dst_asset")


# ============================================================
# main
# ============================================================
def main():
    input_jsonl = _resolve_input_jsonl(SCRIPT_DIR)
    output_jsonl = _resolve_output_jsonl(SCRIPT_DIR)
    pre_dir = _resolve_pre_dir(SCRIPT_DIR)
    model_load_dir = _resolve_model_load_dir(SCRIPT_DIR)

    dl_anom_dir = _resolve_dl_anom_model_dir(model_load_dir)
    dl_pat_loader_py = _resolve_dl_pattern_loader_py(model_load_dir)
    dl_pat_ckpt_dir = _resolve_dl_pattern_ckpt_dir(model_load_dir)

    print("[INFO] ===== FULL RUN (window_size=80, step=5) =====")
    print(f"[INFO] input_jsonl       = {input_jsonl}")
    print(f"[INFO] output_jsonl      = {output_jsonl}")
    print(f"[INFO] pre_dir           = {pre_dir}")
    print(f"[INFO] model_load_dir    = {model_load_dir}")
    print(f"[INFO] dl_anom_dir       = {dl_anom_dir}")
    print(f"[INFO] dl_pat_loader_py  = {dl_pat_loader_py}")
    print(f"[INFO] dl_pat_ckpt_dir   = {dl_pat_ckpt_dir}")
    print(f"[INFO] TOPK              = {TOPK}")
    print(f"[INFO] WINDOW_SIZE       = {WINDOW_SIZE}")
    print(f"[INFO] STEP              = {STEP}")
    print(f"[INFO] ALWAYS_RUN_PATTERN= {ALWAYS_RUN_PATTERN}")
    print(f"[INFO] RISK_ZERO_WHEN_NORMAL= {RISK_ZERO_WHEN_NORMAL}")
    print(f"[INFO] DROP_ORIGIN_KEYS  = {sorted(list(DROP_ORIGIN_KEYS))}")

    # 1) load all packets
    packets = load_all_packets(input_jsonl)
    if len(packets) < WINDOW_SIZE:
        raise ValueError(f"need >= {WINDOW_SIZE} packets but got {len(packets)}")

    # 2) featurize all packets -> prepares
    fz = try_init_featurizer(pre_dir)

    prepares: List[Dict[str, Any]] = []
    for i, origin in enumerate(packets):
        rid = str(origin.get("redis_id") or f"offline-{i:08d}")
        origin["redis_id"] = rid  # 내부 추론/정합성용(저장 시에는 strip)

        features = build_features(origin, fz, redis_id=rid)
        prepares.append({"origin": origin, "features": features, "_meta": {"redis_id": rid}})

    # 3) ML inference for all packets (batch inside)
    mlb = load_ml_bundle(model_load_dir)
    normal_label = _infer_normal_label(mlb["model"], mlb.get("metadata", {}))

    engine = MLAnomalyProbEngine(
        selected_features=mlb["selected_features"],
        scaler=mlb["scaler"],
        topk=int(TOPK),
        max_batch=256,
    )

    ml_out_list = predict_enrich_origin_records_with_bundle_fast(
        prepares,
        scaler=mlb["scaler"],
        selected_features=mlb["selected_features"],
        topk=int(TOPK),
        model=mlb["model"],
        normal_label=int(normal_label),
        engine=engine,
    )

    # attach ML fields back to each packet(origin/features)
    for pr, ml_out in zip(prepares, ml_out_list):
        o = pr["origin"]
        f = pr["features"]
        o["ml_anomaly_prob"] = ml_out.get("ml_anomaly_prob") or []
        if "ml_anomaly_score" in ml_out:
            o["ml_anomaly_score"] = float(ml_out["ml_anomaly_score"])
        f["match"] = int(ml_out.get("match", 0) or 0)

    # 4) DL bundle load (anom + pattern) + sliding windows predict (MERGED)
    dl_anom_bundle = load_lstm_ae_bundle(dl_anom_dir)
    dl_pat_bundle = load_dl_pattern_bundle(
        dl_pat_ckpt_dir,
        loader_py=dl_pat_loader_py,
        device="cpu",
    )

    dl_pack = predict_dl_models_windows_sliding(
        prepares,
        dl_anom_bundle,
        dl_pat_bundle,
        window_size=int(WINDOW_SIZE),
        step=int(STEP),
        batch_size_anom=int(DL_BATCH_SIZE),
        batch_size_pat=int(PAT_BATCH_SIZE),
        always_run_pattern=bool(ALWAYS_RUN_PATTERN),
        risk_zero_when_normal=bool(RISK_ZERO_WHEN_NORMAL),
    )

    # 5) write outputs: 1 line per window
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fout:
        for w in dl_pack["windows"]:
            s = int(w.get("start", (int(w["seq_id"]) - 1) * STEP))
            e = int(w.get("end", s + WINDOW_SIZE - 1))

            window_prepares = prepares[s : e + 1]

            # ✅ 저장 시에만 strip 적용 (redis_id, ml_anomaly_score 제거)
            window_raw = [_strip_origin_for_save(pr["origin"]) for pr in window_prepares]

            summary = w.get("summary") or {}
            if not isinstance(summary, dict):
                summary = {}

            # risk meta는 window_raw 기준으로 확정 (timestamp/ip/asset 유지)
            _fill_risk_meta(summary, window_raw)

            out_obj = {
                "seq_id": int(w["seq_id"]),
                "pattern": w.get("pattern"),
                "summary": summary,
                "window_raw": window_raw,
            }

            fout.write(json.dumps(_sanitize(out_obj), ensure_ascii=False) + "\n")

    print(f"[DONE] saved -> {output_jsonl} (windows={dl_pack['num_windows']}, packets={dl_pack['num_packets']})")


if __name__ == "__main__":
    main()
