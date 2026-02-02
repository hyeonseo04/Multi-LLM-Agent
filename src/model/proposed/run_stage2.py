# -*- coding: utf-8 -*-

"""
stage2_mediator_final_multi.py
- 여러 입력 파일(file_sets) 일괄 처리
- 입력: CSV 또는 JSONL 자동 인식
- 강제 1글자(A/B/C/D) 디코딩(LogitsProcessor)
- 출력: output_base(.csv 허용/미지정시 _mediator.csv 자동 부착)
- 로그 스타일:
  * 시작: Loading model... / Models and prompt loaded.
  * 파일: Processing file: <in_path>
  * 진행: tqdm 진행바 (한 줄)
  * 끝  : [Saved] <out_path>
  * 전체: [All tasks done]
"""

import os, re, gc, time
from pathlib import Path
from typing import List, Tuple

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from transformers.utils import logging as hf_logging
from tqdm.auto import tqdm  # 진행바
from pathlib import Path

# =========================
# Quiet / Env safety
# =========================
hf_logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE = Path("/home/hslee/multiagent/results/proposed/v4/duplication")

file_sets = []

for stage1_file in BASE.rglob("stage1_results.jsonl"):
    file_sets.append({
        "input": str(stage1_file),
        "output_base": str(stage1_file.parent / "stage2_results_ex")
    })

file_sets = sorted(file_sets, key=lambda x: x["input"])

CONFIG = {
    "file_sets": file_sets,

    "mediator_prompt": "/home/hslee/multiagent/prompts/v1/final_medical_adjudicator.txt",
    "model_id": "openai/gpt-oss-20b",
    "device_map": "auto",

    "max_length": 2048,
    "max_new_tokens_label": 1,
    "batch_size": 8,
}

# =========================
# 출력 컬럼 고정
# =========================
OUTPUT_COLS = [
    "question", "options", "answer",
    "answer_A", "reasoning_A",
    "answer_B", "reasoning_B",
    "review_score_B_on_A", "review_feedback_B_on_A",
    "review_score_A_on_B", "review_feedback_A_on_B",
    "final_answer", "mediator_raw",
]
REQUIRED_COLS = OUTPUT_COLS[:-2]  # final_answer, mediator_raw 제외



# =========================
# UTILS
# =========================
def ensure_dirs(path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def safe_format_prompt(template: str, **kwargs) -> str:
    placeholders = re.findall(r"\{(\w+)\}", template)
    tmap = {p: f"__TOK_{p.upper()}__" for p in placeholders}
    for p, tok in tmap.items():
        template = template.replace(f"{{{p}}}", tok)
    template = template.replace("{", "{{").replace("}", "}}")
    for p, tok in tmap.items():
        template = template.replace(tok, f"{{{p}}}")
    skw = {k: str(v).replace("{", "{{").replace("}", "}}") for k, v in kwargs.items()}
    return template.format(**skw)

def apply_chat(tokenizer, user_text: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            add_generation_prompt=True, tokenize=False
        )
    except Exception:
        return f"User:\n{user_text}\n\nAssistant:"

def read_table_auto(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(path)
    elif suf in (".jsonl", ".json"):
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported input format: {path}")
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = "" if "score_" not in c else 0
    return df

def resolve_out_path(output_base: str) -> Path:
    p = Path(output_base)
    # output_base가 확장자를 가지고 있으면 그대로 사용
    if p.suffix.lower() in (".csv", ".jsonl", ".json"):
        return p
    # 확장자 없으면 그냥 그 이름 그대로(나중에 저장 단계에서 csv/jsonl 둘 다 만들 것)
    return p


def expand_file_sets(cfg) -> List[Tuple[Path, Path]]:  # (input, output_csv)
    pairs: List[Tuple[Path, Path]] = []
    items = cfg.get("file_sets", [])
    if not items:
        raise ValueError("CONFIG['file_sets']가 비어 있습니다.")
    for item in items:
        in_path = Path(item["input"])
        out_path = resolve_out_path(item["output_base"])
        pairs.append((in_path, out_path))
    return pairs

# =========================
# 한 글자 강제 (A/B/C/D)
# =========================
class AllowOnly(LogitsProcessor):
    def __init__(self, ids): self.ids = set(ids)
    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, -float("inf"))
        if self.ids: mask[:, list(self.ids)] = 0.0
        return scores + mask

def _single_token_ids(tokenizer, letter: str):
    out = set()
    for pref in ["", " ", "▁", "Ġ"]:
        t = tokenizer.encode(pref + letter, add_special_tokens=False)
        if len(t) == 1:
            out.add(t[0])
    return sorted(out)

def build_allowed_ids(tokenizer):
    ids = []
    for L in ["A","B","C","D","a","b","c","d"]:
        ids += _single_token_ids(tokenizer, L)
    return sorted(set(ids))

@torch.no_grad()
def forced_label_generate(model, tokenizer, prompts, *, max_length, max_new_tokens, first_device):
    chats = [apply_chat(tokenizer, p) + '{"label":"' for p in prompts]
    enc = tokenizer(chats, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    enc = {k: v.to(first_device) for k, v in enc.items()}

    allowed_ids = build_allowed_ids(tokenizer)
    processors = LogitsProcessorList([AllowOnly(allowed_ids)]) if allowed_ids else None

    pad_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
    if not allowed_ids:
        out = model.generate(
            **enc, max_new_tokens=max(2, max_new_tokens), do_sample=False,
            pad_token_id=pad_id, eos_token_id=pad_id
        )
    else:
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=False,
            pad_token_id=pad_id, eos_token_id=pad_id,
            logits_processor=processors,
        )

    ilen = enc["input_ids"].size(1)
    raws, labels = [], []
    for i in range(out.size(0)):
        gen = out[i, ilen:]
        txt = tokenizer.decode(gen, skip_special_tokens=True)
        lab = (txt.strip().replace('"', "").replace("'", "")[:1] or "?").upper()
        if lab not in {"A","B","C","D"}:
            m = re.search(r'[ABCD]', txt.upper())
            lab = m.group(0) if m else "?"
        raws.append(f'{{"label":"{lab}"}}')
        labels.append(lab)
    return labels, raws

# =========================
# MODEL I/O
# =========================
@torch.no_grad()
def load_mediator(model_id: str, device_map="auto"):
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA GPU not detected")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "left"

    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device_map,
    ).eval()
    first_device = next(iter(mdl.hf_device_map.values())) if hasattr(mdl, "hf_device_map") else "cuda:0"
    return tok, mdl, first_device

# =========================
# MAIN
# =========================
def main():
    # 시작 로그
    print("Loading model...")
    pairs = expand_file_sets(CONFIG)
    tok, mdl, dev = load_mediator(CONFIG["model_id"], device_map=CONFIG["device_map"])
    prompt_tmpl = Path(CONFIG["mediator_prompt"]).read_text(encoding="utf-8").strip()
    bs = CONFIG["batch_size"]
    print("Models and prompt loaded.")

    for in_path, out_path in pairs:
        if not in_path.exists():
            print(f"Processing file: {in_path}")
            print(f"[Error] Input not found: {in_path}")
            continue
        
        # 출력 파일 경로 확인
        base = out_path
        if base.suffix.lower() in (".csv", ".jsonl", ".json"):
            base = base.with_suffix("")
        
        csv_path = base.with_suffix(".csv")
        jsonl_path = base.with_suffix(".jsonl")
        
        # 이미 두 파일이 모두 존재하면 스킵
        if csv_path.exists() and jsonl_path.exists():
            print(f"[Skipped] Output already exists: {csv_path}")
            continue
        
        # 파일 처리 시작 로그
        print(f"Processing file: {in_path}")

        # 데이터 로드
        df = read_table_auto(in_path)

        # 프롬프트 준비
        prompts = [
            safe_format_prompt(
                prompt_tmpl,
                question=r.get("question", ""),
                options=r.get("options", ""),  # dict 그대로 들어가도 문자열로 변환됨
                reasoning_model1=r.get("reasoning_A", ""),
                answer_model1=r.get("answer_A", "?"),
                reasoning_model2=r.get("reasoning_B", ""),
                answer_model2=r.get("answer_B", "?"),
                score_model2_on_model1=r.get("review_score_B_on_A", 0),
                feedback_model2_on_model1=r.get("review_feedback_B_on_A", ""),
                score_model1_on_model2=r.get("review_score_A_on_B", 0),
                feedback_model1_on_model2=r.get("review_feedback_A_on_B", ""),
            )
            for _, r in df.iterrows()
        ]
        # =========================
        # PROMPT TOKEN LENGTH CHECK
        # =========================
        try:
            sample_n = min(200, len(prompts))  # 너무 오래 걸리면 여기 숫자 줄이기 (ex. 50)
            lens = []
            for p in prompts[:sample_n]:
                chat = apply_chat(tok, p)
                ids = tok(chat, add_special_tokens=False)["input_ids"]
                lens.append(len(ids))

            lens_sorted = sorted(lens)
            p95 = lens_sorted[int(len(lens_sorted) * 0.95)] if lens_sorted else 0
            p99 = lens_sorted[int(len(lens_sorted) * 0.99)] if lens_sorted else 0
            mx = max(lens) if lens else 0
            avg = sum(lens) / len(lens) if lens else 0

            print(f"[Prompt Token Stats] samples={sample_n} avg={avg:.1f} p95={p95} p99={p99} max={mx} (max_length={CONFIG['max_length']})")

            # 잘림 위험 경고
            if mx >= CONFIG["max_length"]:
                print(f"[Warning] max prompt tokens ({mx}) >= max_length ({CONFIG['max_length']}). Truncation likely happening.")
            elif p95 >= int(CONFIG["max_length"] * 0.9):
                print(f"[Warning] p95 ({p95}) is close to max_length ({CONFIG['max_length']}). Consider increasing max_length.")
        except Exception as e:
            print(f"[Prompt Token Stats] skipped due to error: {e}")


        # 배치 생성 (진행바)
        all_labels, all_raw = [], []
        for i in tqdm(range(0, len(prompts), bs), desc=f"Processing {in_path.name}"):
            batch = prompts[i:i+bs]
            labels, raws = forced_label_generate(
                mdl, tok, batch,
                max_length=CONFIG["max_length"],
                max_new_tokens=CONFIG["max_new_tokens_label"],
                first_device=dev
            )
            all_labels.extend(labels)
            all_raw.extend(raws)

        # 결과 저장 (정확히 OUTPUT_COLS만)
        out_df = pd.DataFrame(index=df.index)
        for c in OUTPUT_COLS:
            if c in df.columns:
                out_df[c] = df[c]
            else:
                out_df[c] = "" if "score_" not in c else 0
        out_df["final_answer"] = [str(x).upper() for x in all_labels]
        out_df["mediator_raw"] = all_raw

        ensure_dirs(out_path)

        # out_path가 파일명+확장자를 포함할 수도, 확장자가 없을 수도 있음
        base = out_path
        if base.suffix.lower() in (".csv", ".jsonl", ".json"):
            # 확장자가 있으면 stem까지만 베이스로
            base = base.with_suffix("")

        csv_path = base.with_suffix(".csv")
        jsonl_path = base.with_suffix(".jsonl")

        # 1) CSV 저장
        out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 2) JSONL 저장
        out_df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

        print(f"[Saved] {csv_path}")
        print(f"[Saved] {jsonl_path}")


if __name__ == "__main__":
    main()
