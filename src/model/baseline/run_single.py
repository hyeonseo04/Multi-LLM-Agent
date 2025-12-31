import os
import json
import re
import gc
import glob
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

# =========================================================
# ✅ 경로 및 환경 설정 (모두 코드 내에 고정)
# =========================================================
CONFIG = {
    # 입력 데이터 경로
    "raw_data": "/home/hslee/multiagent/data/raw/medqa/medqa_all_clean.jsonl",
    "processed_root": "/home/hslee/multiagent/data/processed/medqa",
    
    # 결과 저장 및 프롬프트 경로
    "results_root": "/home/hslee/multiagent/results/baseline/medqa",
    "prompt_file": "/home/hslee/multiagent/prompts/v2/classfication_prompt_1.txt",
    
    # 모델 설정
    "model_id": "openai/gpt-oss-20b",
    "batch_size": 8,
    "max_length": 1024,
    "max_new_tokens": 1,
    "do_sample": False,
    "max_retries": 1,
    "gpu_mem_per_card": "44GiB",
    "dtype": torch.bfloat16,
    
    # 기본 텍스트 필드
    "text_field_default": "question",
}

# 정답 후보군
_VALID = {"A", "B", "C", "D"}

# =========================
# 1. 파일 탐색 및 경로 파싱
# =========================
def get_all_input_paths():
    """모든 노이즈 파일(.jsonl)을 자동으로 탐색하여 리스트 반환"""
    paths = [CONFIG["raw_data"]]  # Clean 데이터 우선 추가
    
    # processed 하위 모든 jsonl 탐색 (**/seed_*.jsonl)
    search_pattern = os.path.join(CONFIG["processed_root"], "**/*.jsonl")
    found_files = glob.glob(search_pattern, recursive=True)
    
    paths.extend(found_files)
    return sorted(list(set(paths)))

def parse_experiment_id(input_path: str):
    """경로에서 노이즈종류/레벨/시드를 추출하여 결과 폴더 구조 생성"""
    p = Path(input_path).as_posix()

    if "/processed/medqa/" in p:
        # 예: .../processed/medqa/typo/wer_0.1/seed_1.jsonl
        # '/' 기준으로 분할하여 뒤에서부터 정보 추출
        parts = p.split("/processed/medqa/")[1].split("/")
        if len(parts) >= 3:
            noise = parts[0]
            level = parts[1]
            seed = parts[2].replace(".jsonl", "")
            return noise, level, seed
            
    return "clean", "clean", "seed_0"

# =========================
# 2. 로직 처리 유틸리티 (기존 로직 유지)
# =========================
def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def safe_format_prompt(template: str, text: str) -> str:
    token = "___TEXT___"
    t = template.replace("{text}", token).replace("{", "{{").replace("}", "}}").replace(token, "{text}")
    return t.format(text=str(text).replace("{", "{{").replace("}", "}}"))

class AllowOnly(LogitsProcessor):
    def __init__(self, ids):
        self.ids = set(ids)
    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        for i in self.ids: mask[:, i] = 0.0
        return scores + mask

def _single_token_ids(tokenizer, letter: str):
    out = set()
    for pref in ["", " ", "▁", "Ġ"]:
        t = tokenizer.encode(pref + letter, add_special_tokens=False)
        if len(t) == 1: out.add(t[0])
    return sorted(out)

def build_allowed_ids(tokenizer):
    ids = []
    for L in ["A", "B", "C", "D"]:
        ids += _single_token_ids(tokenizer, L)
    return sorted(set(ids))

@torch.inference_mode()
def classify_forced(model, tokenizer, prompts, max_length=1024):
    """LogitsProcessor를 사용하여 A,B,C,D 중 하나로 답변 강제"""
    chats = [tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False) + '{"label":"' for p in prompts]
    enc = tokenizer(chats, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
    
    processors = LogitsProcessorList([AllowOnly(build_allowed_ids(tokenizer))])
    
    out = model.generate(
        **enc, max_new_tokens=1, min_new_tokens=1, do_sample=False,
        pad_token_id=tokenizer.eos_token_id, logits_processor=processors
    )

    input_len = enc["input_ids"].size(1)
    letters, raws = [], []
    for i in range(out.size(0)):
        txt = tokenizer.decode(out[i, input_len:], skip_special_tokens=True)
        L = (txt.strip()[:1] or "?").upper()
        letters.append(L if L in _VALID else "?")
        raws.append(txt)
    return letters, raws

def build_texts(df: pd.DataFrame, text_field: str):
    qs = df[text_field].astype(str).tolist()
    opts_list = df["options"].tolist()
    texts = []
    for q, opts in zip(qs, opts_list):
        texts.append(f"{q}\n\nOptions:\nA) {opts.get('A','')}\nB) {opts.get('B','')}\nC) {opts.get('C','')}\nD) {opts.get('D','')}")
    return texts

# =========================
# 3. 메인 실행 프로세스
# =========================
def main():
    # GPU 체크 및 모델 로드
    n_gpus = torch.cuda.device_count()
    tok = AutoTokenizer.from_pretrained(CONFIG["model_id"], trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_id"], device_map="auto",
        max_memory={i: CONFIG["gpu_mem_per_card"] for i in range(n_gpus)},
        torch_dtype=CONFIG["dtype"], trust_remote_code=True
    ).eval()

    input_paths = get_all_input_paths()
    prompt_templ = load_prompt(CONFIG["prompt_file"])

    for p_path in input_paths:
        noise, level, seed = parse_experiment_id(p_path)
        out_dir = Path(CONFIG["results_root"]) / noise / level / seed
        
        # 중복 실행 방지 (Skip logic)
        if (out_dir / "metrics.json").exists():
            print(f"[SKIP] 이미 결과가 존재함: {out_dir}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[RUN] 실험 시작: {noise} | {level} | {seed}")

        df = pd.read_json(p_path, lines=True)
        text_field = "noisy_question" if "noisy_question" in df.columns else CONFIG["text_field_default"]
        
        texts = build_texts(df, text_field=text_field)
        prompts = [safe_format_prompt(prompt_templ, t) for t in texts]
        true_ans = df["answer"].astype(str).str.upper().str.strip().tolist()

        preds, pred_raw = ["?"] * len(df), [""] * len(df)
        
        # 배치 추론 진행
        for s in tqdm(range(0, len(prompts), CONFIG["batch_size"]), desc=f"Processing {noise}-{level}"):
            batch_prompts = prompts[s:s + CONFIG["batch_size"]]
            Ls, raws = classify_forced(model, tok, batch_prompts, max_length=CONFIG["max_length"])
            for k, L in enumerate(Ls):
                preds[s + k] = L
                pred_raw[s + k] = raws[k]

        # 결과 저장
        out_df = pd.DataFrame({"pred": preds, "true": true_ans, "correct": [int(p == t) for p, t in zip(preds, true_ans)]})
        out_df.to_csv(out_dir / "predictions.csv", index=False)

        correct_count = sum(out_df["correct"])
        metrics = {
            "noise": noise, "level": level, "seed": seed,
            "accuracy": correct_count / len(df), "timestamp": datetime.now().isoformat()
        }
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print("\n✅ 모든 실험이 완료되었습니다.")

if __name__ == "__main__":
    main()