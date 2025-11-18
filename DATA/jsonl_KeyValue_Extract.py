#!/usr/bin/env python3
import json
import argparse
from collections import defaultdict
from pathlib import Path


def extract_protocol_keys_and_values(jsonl_path, exclude_keys=None):
    protocol_keys = defaultdict(set)
    protocol_values = defaultdict(lambda: defaultdict(set))

    # 제외할 key들 → 소문자로 정규화
    exclude_keys = {k.lower() for k in exclude_keys} if exclude_keys else set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            protocol = obj.get("protocol")
            if protocol is None:
                continue

            for k, v in obj.items():

                # 🔥 제외할 key라면 skip
                if k.lower() in exclude_keys:
                    continue

                protocol_keys[protocol].add(k)

                # 값 정규화
                if isinstance(v, (list, dict)):
                    v = json.dumps(v, sort_keys=True)
                elif v is None:
                    v = "NULL"

                protocol_values[protocol][k].add(str(v))

    return protocol_keys, protocol_values


def save_protocol_values_as_json(protocol_values, output_path):
    converted = {
        proto: {
            key: sorted(list(vals))
            for key, vals in kv.items()
        }
        for proto, kv in protocol_values.items()
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=4)

    print(f"✔ 저장 완료: {output_path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract protocol-based key/value unique sets from a JSONL file."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input JSONL file path"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="protocol_values.json",
        help="Output JSON file path (default: protocol_values.json)"
    )
    parser.add_argument(
        "-ek", "--exclude-key",
        type=str,
        nargs="*",
        default=[],
        help="Keys to exclude (e.g. --exclude-key smac dmac sip)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    jsonl_path = Path(args.input)
    output_json = Path(args.output)
    exclude_keys = args.exclude_key

    if not jsonl_path.exists():
        raise FileNotFoundError(f"❌ 입력 파일을 찾을 수 없음: {jsonl_path}")

    print(f"📌 제외할 key 목록: {exclude_keys}")

    protocol_keys, protocol_values = extract_protocol_keys_and_values(
        jsonl_path,
        exclude_keys=exclude_keys
    )

    save_protocol_values_as_json(protocol_values, output_json)

# """
# python jsonl_KeyValue_Extract.py -i "ML_DL 학습.jsonl" -o result.json -ek @timestamp
# """