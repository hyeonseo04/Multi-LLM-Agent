import os, re, gc, json
from pathlib import Path
from typing import List, Dict

import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
    LogitsProcessor, LogitsProcessorList,
)

# -------------------------
# 1. 환경 설정 및 디버그 설정
# -------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEBUG_CONFIG = {
    "models": {
        "reasoner_A": "Qwen/Qwen2.5-32B-Instruct",
        "reasoner_B": "google/gemma-3-27b-it",
        "adjudicator": "openai/gpt-oss-20b",
    },
    "devices": {"reasoner_A": 0, "reasoner_B": 1},
    "quant_4bit": True,
    "max_new_tokens": {"reasoner": 256, "reviewer": 128, "adjudicator": 1},
    "sample_count": 2, # 테스트할 샘플 개수
}

# -------------------------
# 2. 유틸리티 함수 (기존 로직 유지)
# -------------------------
def apply_chat(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": user_text}], add_generation_prompt=True, tokenize=False)

def safe_format(template: str, **kw) -> str:
    def repl(m):
        key = m.group(1)
        parts = key.split(".")
        val = kw.get(parts[0], "")
        for p in parts[1:]:
            if isinstance(val, dict): val = val.get(p, "")
            else: val = ""; break
        return str(val)
    return re.sub(r"\{([a-zA-Z0-9_\.]+)\}", repl, template)

def parse_json_robust(s: str, keys: List[str]) -> dict:
    s = s.replace("“", '"').replace("”", '"')
    m = re.search(r"\{[\s\S]*\}", s)
    if not m: return {k: "?" for k in keys}
    try:
        obj = json.loads(m.group(0))
        return {k: obj.get(k, "?") for k in keys}
    except:
        return {k: "?" for k in keys}

# -------------------------
# 3. 모델 로딩 함수
# -------------------------
def load_model(model_id, device):
    print(f"--- Loading {model_id} on cuda:{device} ---")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
    )
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map={"": device}, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()
    return tok, mdl

# -------------------------
# 4. 핵심 실행 엔진 (터미널 출력 강화)
# -------------------------
@torch.no_grad()
def run_debug_session(sample_data: List[Dict]):
    # --- STAGE 1: Reasoners ---
    tokA, mdlA = load_model(DEBUG_CONFIG["models"]["reasoner_A"], DEBUG_CONFIG["devices"]["reasoner_A"])
    tokB, mdlB = load_model(DEBUG_CONFIG["models"]["reasoner_B"], DEBUG_CONFIG["devices"]["reasoner_B"])

    results = []

    for idx, item in enumerate(sample_data):
        print(f"\n{'='*20} SAMPLE {idx+1} {'='*20}")
        print(f"[Q]: {item['question']}\n[Options]: {item['options']}")

        # 1. Reasoners Inference
        prompt = f"Solve this MedQA. Return JSON with 'reasoning' and 'answer' (A/B/C/D).\nQuestion: {item['question']}\nOptions: {item['options']}"
        
        # A 추론
        outA = mdlA.generate(**tokA([apply_chat(tokA, prompt)], return_tensors="pt").to(mdlA.device), max_new_tokens=256)
        txtA = tokA.decode(outA[0], skip_special_tokens=True).split("Assistant:")[-1]
        resA = parse_json_robust(txtA, ["reasoning", "answer"])
        print(f"\n[Reasoner A Choice]: {resA['answer']}\n[A's Thought]: {resA['reasoning'][:150]}...")

        # B 추론
        outB = mdlB.generate(**tokB([apply_chat(tokB, prompt)], return_tensors="pt").to(mdlB.device), max_new_tokens=256)
        txtB = tokB.decode(outB[0], skip_special_tokens=True).split("Assistant:")[-1]
        resB = parse_json_robust(txtB, ["reasoning", "answer"])
        print(f"\n[Reasoner B Choice]: {resB['answer']}\n[B's Thought]: {resB['reasoning'][:150]}...")

        # 2. Cross Review
        rev_prompt_tmpl = "Review this medical reasoning. Return JSON with 'score' (1-5) and 'feedback'.\nReasoning: {reasoning}\nAnswer: {answer}"
        
        # B reviews A
        rev_p_BA = safe_format(rev_prompt_tmpl, reasoning=resA['reasoning'], answer=resA['answer'])
        outBA = mdlB.generate(**tokB([apply_chat(tokB, rev_p_BA)], return_tensors="pt").to(mdlB.device), max_new_tokens=128)
        txtBA = tokB.decode(outBA[0], skip_special_tokens=True).split("Assistant:")[-1]
        revBA = parse_json_robust(txtBA, ["score", "feedback"])
        print(f"\n[Review: B on A]: Score {revBA['score']} | Feedback: {revBA['feedback'][:100]}...")

        # A reviews B
        rev_p_AB = safe_format(rev_prompt_tmpl, reasoning=resB['reasoning'], answer=resB['answer'])
        outAB = mdlA.generate(**tokA([apply_chat(tokA, rev_p_AB)], return_tensors="pt").to(mdlA.device), max_new_tokens=128)
        txtAB = tokA.decode(outAB[0], skip_special_tokens=True).split("Assistant:")[-1]
        revAB = parse_json_robust(txtAB, ["score", "feedback"])
        print(f"\n[Review: A on B]: Score {revAB['score']} | Feedback: {revAB['feedback'][:100]}...")

        results.append({
            "item": item, "resA": resA, "resB": resB, "revBA": revBA, "revAB": revAB
        })

    del mdlA, mdlB; gc.collect(); torch.cuda.empty_cache()

    # --- STAGE 2: Adjudicator ---
    tokJ, mdlJ = load_model(DEBUG_CONFIG["models"]["adjudicator"], 0) # Adjudicator는 보통 0번 혹은 auto
    
    print(f"\n{'='*20} FINAL ADJUDICATION {'='*20}")
    for i, r in enumerate(results):
        adj_p = f"Decide final answer A/B/C/D based on 2 models and reviews.\nQ: {r['item']['question']}\nA: {r['resA']['answer']} (Score: {r['revBA']['score']})\nB: {r['resB']['answer']} (Score: {r['revAB']['score']})\nFinal Answer (One Letter Only):"
        
        # 여기서 LogitsProcessor로 A/B/C/D 강제 가능
        outJ = mdlJ.generate(**tokJ([apply_chat(tokJ, adj_p)], return_tensors="pt").to(mdlJ.device), max_new_tokens=1)
        final_ans = tokJ.decode(outJ[0], skip_special_tokens=True).strip()[-1].upper()
        
        print(f"[Sample {i+1}] Correct: {r['item'].get('answer', 'N/A')} | Predicted: {final_ans}")
        if final_ans == r['item'].get('answer'): print("✨ CORRECT!")
        else: print("❌ INCORRECT")

# -------------------------
# 5. 실행부
# -------------------------
if __name__ == "__main__":
    # 실제 데이터 파일이 있다면 로드, 없다면 샘플로 대체
    test_samples = [
        {
            "question": "A 45-year-old male with a history of hypertension presents with sudden onset tearing chest pain radiating to the back. BP is 180/110. What is the next best step?",
            "options": {"A": "CT Angiography", "B": "Echocardiogram", "C": "Chest X-ray", "D": "Beta-blockers"},
            "answer": "A"
        }
    ]
    
    run_debug_session(test_samples)