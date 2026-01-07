import json

path = "ML_DL 학습.jsonl"  # 정상 JSONL 경로로 바꿔줘

count = 0

with open(path, "r", encoding="utf-8") as f:
    for lineno, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # 깨진 라인 있으면 그냥 패스

        # 1) protocol이 xgt_fen인지 (필요하면 빼도 됨)
        if obj.get("protocol") not in ("xgt_fen", "xgt-fen"):
            continue

        # 2) xgt_fen.vars 가 %DB001046 인지
        if obj.get("xgt_fen.vars") != "%DB001046":
            continue

        data = obj.get("xgt_fen.data", "")
        if not isinstance(data, str):
            continue

        # 3) data 마지막 4글자가 0000인지 (대소문자 무시)
        if data.lower().endswith("0000"):
            count += 1
            print(f"[HIT] line {lineno}: xgt_fen.data = {data}")

print(f"\n총 {count}개 라인이 조건을 만족합니다.")
