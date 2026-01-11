import json
import random

# ===== 설정 =====
INPUT_PATH = "/home/hslee/multiagent/data/raw/phrases_no_exclude_test.jsonl"
OUTPUT_PATH = "/home/hslee/multiagent/data/processed/medqa_all_clean.jsonl"
SAMPLE_SIZE = 1273

# ===== JSONL 파일 읽기 =====
data = []
with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            data.append(obj)
        except json.JSONDecodeError:
            print(f"[JSON 파싱 오류] {line[:100]} ...")

print(f"[INFO] 전체 데이터 개수: {len(data)}")

# ===== answer 값을 A/B/C/D로 치환 =====
def convert_answer(obj):
    if "answer" not in obj or "options" not in obj:
        return None
    correct_text = obj["answer"].strip()
    options = obj["options"]

    converted = None
    for key, val in options.items():
        if val.strip() == correct_text:
            converted = key
            break

    if converted is None:
        print(f"[WARN] 정답 매칭 실패: {correct_text[:60]} ...")
        return None

    obj["answer"] = converted
    return obj

converted_data = []
for d in data:
    c = convert_answer(d)
    if c:
        # question, answer, options만 남기고 나머지 key 제거
        cleaned = {
            "question": c.get("question", ""),
            "answer": c.get("answer", ""),
            "options": c.get("options", {})
        }
        converted_data.append(cleaned)

print(f"[INFO] 변환 및 정제 완료 데이터 개수: {len(converted_data)}")

# ===== 무작위 500개 샘플링 =====
if len(converted_data) > SAMPLE_SIZE:
    sampled = random.sample(converted_data, SAMPLE_SIZE)
else:
    sampled = converted_data
    print("[WARN] 데이터가 500개 미만입니다. 전체를 저장합니다.")

# ===== JSONL 파일로 저장 =====
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for obj in sampled:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"[DONE] {len(sampled)}개 항목을 {OUTPUT_PATH} 에 저장 완료.")
