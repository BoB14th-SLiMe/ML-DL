#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
padding.py
index 데이터를 기반으로 window_size 만큼 padding을 채움

"""
import json
import argparse

META_KEYS = {"index", "packet_idx"}


def _to_dict(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}
    return {}


def pad_window(win: dict, window_size: int, pad_value: float) -> dict:
    idx_list = win.get("index", []) or []
    seq = win.get("sequence_group", []) or []
    if not isinstance(idx_list, list) or not isinstance(seq, list) or not seq:
        return {}

    first_pkt = _to_dict(seq[0])
    feature_keys = [k for k in first_pkt.keys() if k not in META_KEYS]
    if not feature_keys:
        return {}

    pos_to_pkt = {}
    T = min(len(idx_list), len(seq))
    for i in range(T):
        try:
            pos = int(idx_list[i])
        except Exception:
            pos = i
        pos_to_pkt[pos] = _to_dict(seq[i])

    new_seq = []
    for pos in range(window_size):
        if pos in pos_to_pkt:
            pkt = pos_to_pkt[pos]
            new_seq.append({k: pkt.get(k, pad_value) for k in feature_keys})
        else:
            new_seq.append({k: pad_value for k in feature_keys})

    return {
        "pattern": win.get("pattern"),
        "sequence_group": new_seq,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input jsonl path")
    ap.add_argument("--window_size", type=int, default=None, help="fixed window size (optional)")
    ap.add_argument("--pad_value", type=float, default=-1.0)
    args = ap.parse_args()

    out_path = "a.jsonl"

    with open(args.input, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                win = json.loads(line)
            except Exception:
                continue
            if not isinstance(win, dict):
                continue

            if args.window_size is None:
                mx = -1
                for v in (win.get("index", []) or []):
                    try:
                        iv = int(v)
                        if iv > mx:
                            mx = iv
                    except Exception:
                        pass
                if mx < 0:
                    continue
                ws = mx + 1
            else:
                ws = args.window_size

            out_obj = pad_window(win, ws, args.pad_value)
            if not out_obj:
                continue

            out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

"""
python padding.py -i ../../results/train/preprocess_pattern.jsonl --window_size 76 --pad_value -1

"""