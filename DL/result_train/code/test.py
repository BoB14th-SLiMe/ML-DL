from packet_feature_extractor import load_preprocess_params, packet_to_feature_dict
from pathlib import Path
import json

pre_dir = Path("../../preprocessing/result")
params = load_preprocess_params(pre_dir)

jsonl_path = Path("../data/attack_ver2.jsonl")

with jsonl_path.open("r", encoding="utf-8") as f:
    for _ in range(20):
        line = f.readline()
        obj = json.loads(line)
        feat = packet_to_feature_dict(obj, params)
        print(obj.get("protocol"), obj.get("xgt_fen.cmd"), "-> xgt_cmd:", feat["xgt_cmd"])
