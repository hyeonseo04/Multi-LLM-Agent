# -*- coding: utf-8 -*-
"""
run_multiagent_model_centric.py

[Model-Centric Pipeline]
1. Load A -> Check if done -> Run (Answer A) -> Unload A
2. Load B -> Check if done -> Run (Answer B & Review B->A) -> Unload B
3. Load A -> Check if done -> Run (Review A->B) -> Unload A
4. Adjudicator -> Check if done -> Run (Final Decision) -> Unload
   * Stage 2: Forced Decoding (A/B/C/D), Token Stats, CSV+JSONL Output
   
타임아웃 시   
export HF_HUB_HTTP_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=300
"""

import os, re, gc, json, glob, time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any

import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList
)
from transformers.utils import logging as hf_logging

# =========================
# Quiet / Env safety
# =========================
hf_logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600" 
os.environ["HF_HUB_READ_TIMEOUT"] = "600"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "600"
try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

# -------------------------
# 0) Project root autodetect
# -------------------------
THIS_FILE = Path(__file__).resolve()
BASE_DIR = THIS_FILE.parent
for _ in range(8):
    if (BASE_DIR / "pyproject.toml").exists() or (BASE_DIR / "README.md").exists():
        break
    if BASE_DIR.parent == BASE_DIR:
        break
    BASE_DIR = BASE_DIR.parent
BASE_DIR = BASE_DIR.resolve()

# -------------------------
# 1) Paths & Config
# -------------------------
RESULTS_ROOT = BASE_DIR / "results" / "proposed" / "v4"
PROMPT_DIR = BASE_DIR / "prompts" / "v1"

CONFIG = {
    "run": {
        "stage": "all",
    },
    "models": {
        "reasoner_A": "Qwen/Qwen2.5-32B-Instruct",
        "reasoner_B": "google/gemma-3-27b-it",
        "adjudicator": "openai/gpt-oss-20b",
    },
    "devices": "auto", 
    "quant_4bit": {"reasoner_A": False, "reasoner_B": False}, 
    
    # 기본 배치 사이즈 (Stage 1용)
    "batch_size": 32, 
    
    "max_length": {"reasoner": 1600, "reviewer": 1600, "adjudicator": 2048},
    "max_new_tokens": {"reasoner": 300, "reviewer": 200, "adjudicator": 1},
    "sampling": {
        "reasoner_A": {"do_sample": True, "temperature": 0.7},
        "reasoner_B": {"do_sample": True, "temperature": 0.7},
        "reviewer":   {"do_sample": False, "temperature": 0.0},
    },
    "safety": {
        "oom_fallback": True,
        "warn_missing_placeholders": False,
        "torch_compile": True,
    },
    
    # ✅ Clean 데이터용 5개 random seed
    "clean_seeds": [42, 123, 456, 789, 1024],
    "default_seed": 42,
}

# =========================================================
# Utilities
# =========================================================
def parse_path_info(input_path: str) -> Tuple[str, str, str]:
    p = Path(input_path).as_posix()
    key = "/data/processed/medqa/"
    if key in p:
        tail = p.split(key, 1)[1]
        parts = tail.split("/")
        if len(parts) >= 3:
            noise = parts[0]
            level = parts[1]
            seed_tag = parts[2].replace(".jsonl", "")
            return noise, level, seed_tag
    return "clean", "clean", "seed_0"

def read_text(p: Path) -> str:
    if not p.exists():
        print(f"[WARN] Prompt file not found: {p}")
        return ""
    return p.read_text(encoding="utf-8").strip()

def apply_chat(tokenizer, user_text: str) -> str:
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                add_generation_prompt=True, tokenize=False
            )
    except: pass
    return f"User:\n{user_text}\n\nAssistant:"

def safe_format_prompt(template: str, **kwargs) -> str:
    # 템플릿 내의 {변수}를 잠시 임시 토큰으로 변경
    placeholders = re.findall(r"\{(\w+)\}", template)
    tmap = {p: f"__TOK_{p.upper()}__" for p in placeholders}
    for p, tok in tmap.items():
        template = template.replace(f"{{{p}}}", tok)
    
    # JSON 구조를 위해 중괄호 이스케이프 ({ -> {{, } -> }})
    template = template.replace("{", "{{").replace("}", "}}")
    
    # 임시 토큰을 다시 포맷팅 가능한 {변수} 형태로 복구
    for p, tok in tmap.items():
        template = template.replace(tok, f"{{{p}}}")
        
    # 값들 중에도 중괄호가 있으면 이스케이프 처리
    skw = {k: str(v).replace("{", "{{").replace("}", "}}") for k, v in kwargs.items()}
    
    return template.format(**skw)

def format_options(options):
    if isinstance(options, str): 
        try:
            options = json.loads(options.replace("'", '"'))
        except:
            return options 
            
    if isinstance(options, dict):
        return "\n".join([f"{k}) {v}" for k, v in sorted(options.items())])
    
    return str(options)

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

def _normalize_and_extract_json(s: str) -> Optional[str]:
    if not isinstance(s, str) or not s.strip(): return None
    s = s.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if m: s = m.group(1).strip()
    start = s.find("{")
    if start == -1: return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{": depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0: return s[start:i+1]
    return None

def parse_reasoner_json_robust(s: str) -> dict:
    default = {"answer": "?", "reasoning": "Parsing Failed"}
    js = _normalize_and_extract_json(s)
    if not js: js = s 
    obj = None
    try: obj = json.loads(js)
    except:
        if repair_json:
            try: obj = repair_json(js, return_objects=True)
            except: pass
    if not isinstance(obj, dict): return default
    ans = str(obj.get("answer", "?")).strip().upper()
    rea = str(obj.get("reasoning", "No reasoning provided.")).strip()
    match = re.search(r"[ABCD]", ans)
    ans = match.group(0) if match else "?"
    return {"answer": ans, "reasoning": rea}

def parse_reviewer_json_robust(s: str) -> dict:
    default = {"score": 0, "feedback": "Parsing Failed"}
    js = _normalize_and_extract_json(s)
    if not js: js = s
    obj = None
    try: obj = json.loads(js)
    except:
        if repair_json:
            try: obj = repair_json(js, return_objects=True)
            except: pass
    if not isinstance(obj, dict): return default
    score = obj.get("score")
    feedback = str(obj.get("feedback", "No feedback provided.")).strip()
    v = 0
    try:
        if isinstance(score, (int, float)): v = int(score)
        elif isinstance(score, str):
            clean = re.sub(r"[^0-9\.]", "", score)
            if clean: v = int(float(clean))
    except: v = 0
    return {"score": max(1, min(5, v)), "feedback": feedback}

def force_clean_gpu():
    """강력한 GPU 메모리 청소 및 torch.compile 캐시 초기화"""
    print("\n[Memory] Starting Force Cleanup...")
    for _ in range(3): gc.collect()
    torch.cuda.empty_cache()
    try:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
    except Exception as e:
        print(f"[Memory] Warning: Dynamo reset failed: {e}")
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[Memory] Post-Cleanup -> Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB\n")

# =========================================================
# Model & Generation
# =========================================================
def build_bnb_4bit():
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
    )

@torch.no_grad()
def load_model_auto(model_id: str, quant4bit: bool = False):
    print(f"[Load] Loading {model_id} (device='auto')...")
    
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("[WARN] CUDA is NOT available. Loading on CPU... (Flash Attention disabled)")
    
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    
    kwargs = dict(
        trust_remote_code=True, 
        low_cpu_mem_usage=True, 
        dtype=torch.bfloat16 if use_cuda else torch.float32, 
        device_map="auto",
    )
    
    if use_cuda:
        kwargs["attn_implementation"] = "flash_attention_2"
    else:
        kwargs["attn_implementation"] = "eager"

    if quant4bit and use_cuda:
        kwargs["quantization_config"] = build_bnb_4bit()
        kwargs.pop("dtype", None)
        
    mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    
    if CONFIG["safety"].get("torch_compile", False) and use_cuda:
        if not quant4bit: 
            print("[Compile] Applying torch.compile(mode='reduce-overhead')...")
            try:
                mdl = torch.compile(mdl, mode="reduce-overhead")
            except Exception as e:
                print(f"[Warn] torch.compile failed: {e}.")

    first_device = "cpu"
    if use_cuda and hasattr(mdl, "hf_device_map") and mdl.hf_device_map:
        for v in mdl.hf_device_map.values():
            if isinstance(v, str) and v.startswith("cuda"): first_device = v; break
            elif isinstance(v, int): first_device = f"cuda:{v}"; break
    elif use_cuda:
        first_device = "cuda:0"
            
    return tok, mdl, first_device

@torch.no_grad()
def generate_full_output(model, tokenizer, prompts, *, max_len, max_new, do_sample, temperature, device):
    chats = [apply_chat(tokenizer, p) for p in prompts]
    enc = tokenizer(chats, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    enc = {k: v.to(device) for k, v in enc.items()}
    
    gen_kwargs = dict(
        max_new_tokens=int(max_new),
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        do_sample=bool(do_sample),
    )
    if do_sample: gen_kwargs["temperature"] = temperature
    
    outputs = model.generate(**enc, **gen_kwargs)
    in_len = enc["input_ids"].size(1)
    return [tokenizer.decode(o[in_len:], skip_special_tokens=True).strip() for o in outputs]

def batched_generate_with_fallback_local(generate_fn, prompts: List[str], batch_size: int, desc: str = "") -> List[str]:
    outs = []
    i = 0
    pbar = tqdm(total=len(prompts), desc=desc, leave=False)
    while i < len(prompts):
        bs = batch_size
        while True:
            try:
                chunk = prompts[i:i+bs]
                outs.extend(generate_fn(chunk))
                i += bs
                pbar.update(bs)
                break
            except torch.cuda.OutOfMemoryError:
                force_clean_gpu()
                if bs <= 1:
                    print(f"[FATAL OOM] Failed at index {i}")
                    raise
                bs = max(1, bs // 2)
                print(f"[OOM] Retry batch={bs}")
    pbar.close()
    return outs

# =========================================================
# 단일 파일 + 단일 Seed 처리 함수 (Stage 1 전체)
# =========================================================
def run_stage1_single_file_seed(
    input_path: str,
    noise: str,
    level: str,
    seed_tag: str,
    actual_seed: int,
    prompts: dict,
):
    """
    하나의 파일을 하나의 seed로 Stage 1 전체 실행
    (Phase 1: A, Phase 2: B + Review B->A, Phase 3: Review A->B)
    """
    # Seed 설정
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)
    
    work_dir = RESULTS_ROOT / noise / level / seed_tag
    work_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = work_dir / "stage1_results.jsonl"
    
    reasoner_tmpl = prompts["reasoner"]
    reviewer_tmpl = prompts["reviewer"]
    bs = int(CONFIG["batch_size"])
    
    # =========================================================================
    # Phase 1: Reasoner A
    # =========================================================================
    print(f"\n[Phase 1/3] {noise}/{level}/{seed_tag} (seed={actual_seed}) - Reasoner A")
    
    need_phase1 = True
    if stage1_path.exists():
        try:
            df = pd.read_json(stage1_path, lines=True)
            if "answer_A" in df.columns and df["answer_A"].astype(str).str.strip().all():
                need_phase1 = False
                print("  [Skip] Answer A already exists")
        except:
            pass
    
    if need_phase1:
        tok, mdl, dev = load_model_auto(CONFIG['models']['reasoner_A'], quant4bit=CONFIG['quant_4bit']['reasoner_A'])
        
        file_start = time.time()
        
        if stage1_path.exists():
            df = pd.read_json(stage1_path, lines=True)
        else:
            df = pd.read_json(input_path, lines=True)

        for c in ["reasoning_A", "answer_A", "raw_reasoner_A"]:
            if c not in df.columns:
                df[c] = ""
        
        target_questions = [r.get("noisy_question") if r.get("noisy_question") else r["question"] for _, r in df.iterrows()]
        
        p_list = []
        for i, r in df.iterrows():
            if df.iloc[i]["answer_A"]:
                p_list.append("dummy")
            else:
                opts = r["options"] if isinstance(r["options"], dict) else {}
                formatted_opts = format_options(opts)
                p_list.append(safe_format(reasoner_tmpl, question=target_questions[i], options=formatted_opts))
        
        todo_indices = [i for i, p in enumerate(p_list) if p != "dummy"]
        if todo_indices:
            real_prompts = [p_list[i] for i in todo_indices]
            def _gen(chunk):
                return generate_full_output(mdl, tok, chunk, 
                    max_len=CONFIG["max_length"]["reasoner"], max_new=CONFIG["max_new_tokens"]["reasoner"],
                    do_sample=CONFIG["sampling"]["reasoner_A"]["do_sample"], 
                    temperature=CONFIG["sampling"]["reasoner_A"]["temperature"], device=dev)
            
            raw_outs = batched_generate_with_fallback_local(_gen, real_prompts, bs, desc="ReasonA")
            
            for idx, raw in zip(todo_indices, raw_outs):
                parsed = parse_reasoner_json_robust(raw)
                df.at[idx, "raw_reasoner_A"] = raw
                df.at[idx, "answer_A"] = parsed["answer"]
                df.at[idx, "reasoning_A"] = parsed["reasoning"]
                
        df.to_json(stage1_path, orient="records", lines=True, force_ascii=False)
        elapsed = time.time() - file_start
        print(f"  >>> Done. ⏱ Time: {str(timedelta(seconds=int(elapsed)))}")
        
        del mdl, tok
        force_clean_gpu()
    
    # =========================================================================
    # Phase 2: Reasoner B + Review B->A
    # =========================================================================
    print(f"\n[Phase 2/3] {noise}/{level}/{seed_tag} - Reasoner B + Review B->A")
    
    need_phase2 = True
    if stage1_path.exists():
        try:
            df = pd.read_json(stage1_path, lines=True)
            cols = ["answer_B", "review_feedback_B_on_A"]
            if all(c in df.columns and df[c].astype(str).str.strip().all() for c in cols):
                need_phase2 = False
                print("  [Skip] Answer B & Review B->A already exist")
        except:
            pass
    
    if need_phase2:
        tok, mdl, dev = load_model_auto(CONFIG['models']['reasoner_B'], quant4bit=CONFIG['quant_4bit']['reasoner_B'])
        
        file_start = time.time()
        df = pd.read_json(stage1_path, lines=True)

        for c in ["reasoning_B", "answer_B", "raw_reasoner_B", "review_score_B_on_A", "review_feedback_B_on_A", "raw_review_B_on_A"]:
            if c not in df.columns:
                df[c] = ""
        
        target_questions = [r.get("noisy_question") if r.get("noisy_question") else r["question"] for _, r in df.iterrows()]
        
        # 1. Answer B
        p_list = []
        for i, r in df.iterrows():
            if df.iloc[i]["answer_B"]:
                p_list.append("dummy")
            else:
                opts = r["options"] if isinstance(r["options"], dict) else {}
                formatted_opts = format_options(opts)
                p_list.append(safe_format(reasoner_tmpl, question=target_questions[i], options=formatted_opts))
        
        todo_indices = [i for i, p in enumerate(p_list) if p != "dummy"]
        if todo_indices:
            def _gen_b(chunk):
                return generate_full_output(mdl, tok, chunk, 
                    max_len=CONFIG["max_length"]["reasoner"], max_new=CONFIG["max_new_tokens"]["reasoner"],
                    do_sample=CONFIG["sampling"]["reasoner_B"]["do_sample"], 
                    temperature=CONFIG["sampling"]["reasoner_B"]["temperature"], device=dev)
            raw_outs = batched_generate_with_fallback_local(_gen_b, [p_list[i] for i in todo_indices], bs, desc="ReasonB")
            for idx, raw in zip(todo_indices, raw_outs):
                parsed = parse_reasoner_json_robust(raw)
                df.at[idx, "raw_reasoner_B"] = raw
                df.at[idx, "answer_B"] = parsed["answer"]
                df.at[idx, "reasoning_B"] = parsed["reasoning"]

        # 2. Review B->A
        p_rev = []
        for i, r in df.iterrows():
            if df.iloc[i]["review_feedback_B_on_A"]:
                p_rev.append("dummy")
            else:
                opts = r["options"] if isinstance(r["options"], dict) else {}
                formatted_opts = format_options(opts)
                p_rev.append(safe_format(reviewer_tmpl, question=target_questions[i], options=formatted_opts, 
                                        reasoning=r["reasoning_A"], answer=r["answer_A"]))
        
        todo_rev = [i for i, p in enumerate(p_rev) if p != "dummy"]
        if todo_rev:
            def _gen_rev(chunk):
                return generate_full_output(mdl, tok, chunk, 
                    max_len=CONFIG["max_length"]["reviewer"], max_new=CONFIG["max_new_tokens"]["reviewer"],
                    do_sample=CONFIG["sampling"]["reviewer"]["do_sample"], 
                    temperature=CONFIG["sampling"]["reviewer"]["temperature"], device=dev)
            raw_outs = batched_generate_with_fallback_local(_gen_rev, [p_rev[i] for i in todo_rev], bs, desc="Review B->A")
            for idx, raw in zip(todo_rev, raw_outs):
                parsed = parse_reviewer_json_robust(raw)
                df.at[idx, "raw_review_B_on_A"] = raw
                df.at[idx, "review_score_B_on_A"] = parsed["score"]
                df.at[idx, "review_feedback_B_on_A"] = parsed["feedback"]

        df.to_json(stage1_path, orient="records", lines=True, force_ascii=False)
        elapsed = time.time() - file_start
        print(f"  >>> Done. ⏱ Time: {str(timedelta(seconds=int(elapsed)))}")
        
        del mdl, tok
        force_clean_gpu()
    
    # =========================================================================
    # Phase 3: Review A->B
    # =========================================================================
    print(f"\n[Phase 3/3] {noise}/{level}/{seed_tag} - Review A->B")
    
    need_phase3 = True
    if stage1_path.exists():
        try:
            df = pd.read_json(stage1_path, lines=True)
            if "review_feedback_A_on_B" in df.columns and df["review_feedback_A_on_B"].astype(str).str.strip().all():
                need_phase3 = False
                print("  [Skip] Review A->B already exists")
        except:
            pass
    
    if need_phase3:
        tok, mdl, dev = load_model_auto(CONFIG['models']['reasoner_A'], quant4bit=CONFIG['quant_4bit']['reasoner_A'])
        
        file_start = time.time()
        df = pd.read_json(stage1_path, lines=True)

        for c in ["review_score_A_on_B", "review_feedback_A_on_B", "raw_review_A_on_B"]:
            if c not in df.columns:
                df[c] = ""
        
        target_questions = [r.get("noisy_question") if r.get("noisy_question") else r["question"] for _, r in df.iterrows()]
        
        p_rev = []
        for i, r in df.iterrows():
            if df.iloc[i]["review_feedback_A_on_B"]:
                p_rev.append("dummy")
            else:
                opts = r["options"] if isinstance(r["options"], dict) else {}
                formatted_opts = format_options(opts)
                p_rev.append(safe_format(reviewer_tmpl, question=target_questions[i], options=formatted_opts, 
                                        reasoning=r["reasoning_B"], answer=r["answer_B"]))
        
        todo_rev = [i for i, p in enumerate(p_rev) if p != "dummy"]
        if todo_rev:
            def _gen_rev(chunk):
                return generate_full_output(mdl, tok, chunk, 
                    max_len=CONFIG["max_length"]["reviewer"], max_new=CONFIG["max_new_tokens"]["reviewer"],
                    do_sample=CONFIG["sampling"]["reviewer"]["do_sample"], 
                    temperature=CONFIG["sampling"]["reviewer"]["temperature"], device=dev)
            raw_outs = batched_generate_with_fallback_local(_gen_rev, [p_rev[i] for i in todo_rev], bs, desc="Review A->B")
            for idx, raw in zip(todo_rev, raw_outs):
                parsed = parse_reviewer_json_robust(raw)
                df.at[idx, "raw_review_A_on_B"] = raw
                df.at[idx, "review_score_A_on_B"] = parsed["score"]
                df.at[idx, "review_feedback_A_on_B"] = parsed["feedback"]

        df.to_json(stage1_path, orient="records", lines=True, force_ascii=False)
        elapsed = time.time() - file_start
        print(f"  >>> Done. ⏱ Time: {str(timedelta(seconds=int(elapsed)))}")
        
        del mdl, tok
        force_clean_gpu()


# =========================================================
# STAGE 1: 파일별 + Seed별 처리
# =========================================================
def run_stage1_model_centric(files: List[str], prompts: dict):
    print(f"\n{'='*60}\n[STAGE 1] Multi-Agent Reasoning with 5 Seeds for Clean\n{'='*60}")
    
    for input_path in tqdm(files, desc="Files"):
        noise, level, seed_orig = parse_path_info(input_path)
        
        # ✅ Clean 데이터면 5개 seed로 반복
        if noise == "clean":
            seed_list = [(f"seed_{i}", s) for i, s in enumerate(CONFIG["clean_seeds"])]
        else:
            # 노이즈 데이터는 1번만
            seed_list = [(seed_orig, CONFIG["default_seed"])]
        
        for seed_tag, actual_seed in seed_list:
            run_stage1_single_file_seed(
                input_path=input_path,
                noise=noise,
                level=level,
                seed_tag=seed_tag,
                actual_seed=actual_seed,
                prompts=prompts,
            )


# =========================================================
# STAGE 2: Adjudicator Utils
# =========================================================
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
    # [중요] 성공한 코드의 핵심: 프롬프트 끝에 JSON 시작 구문을 강제로 붙임
    chats = [apply_chat(tokenizer, p) + '{"label":"' for p in prompts]
    
    enc = tokenizer(chats, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    enc = {k: v.to(first_device) for k, v in enc.items()}

    allowed_ids = build_allowed_ids(tokenizer)
    processors = LogitsProcessorList([AllowOnly(allowed_ids)]) if allowed_ids else None

    pad_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
    
    gen_kwargs = dict(
        max_new_tokens=1,
        min_new_tokens=1,
        do_sample=False,
        pad_token_id=pad_id,
        eos_token_id=pad_id,
    )
    if allowed_ids:
        gen_kwargs["logits_processor"] = processors

    out = model.generate(**enc, **gen_kwargs)

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

OUTPUT_COLS = [
    "question", "options", "answer",
    "answer_A", "reasoning_A",
    "answer_B", "reasoning_B",
    "review_score_B_on_A", "review_feedback_B_on_A",
    "review_score_A_on_B", "review_feedback_A_on_B",
    "final_answer", "mediator_raw",
]

def run_stage2_all(files: List[str], prompts: dict):
    print("Loading model...")
    
    # 1) 대상 결정
    todo_tasks = []
    for fp in files:
        noise, level, seed_orig = parse_path_info(fp)
        
        # Clean이면 5개 seed 모두 처리
        if noise == "clean":
            seed_list = [f"seed_{i}" for i in range(len(CONFIG["clean_seeds"]))]
        else:
            seed_list = [seed_orig]
        
        for seed_tag in seed_list:
            stage1_path = RESULTS_ROOT / noise / level / seed_tag / "stage1_results.jsonl"
            final_csv = RESULTS_ROOT / noise / level / seed_tag / "stage2_results.csv"
            
            if stage1_path.exists() and not final_csv.exists():
                todo_tasks.append((stage1_path, final_csv))
    
    if not todo_tasks:
        print("Nothing to adjudicate.")
        return

    # 2) 모델 로드
    tok, mdl, dev = load_model_auto(CONFIG["models"]["adjudicator"], quant4bit=False)
    prompt_tmpl = prompts["adjudicator"]
    bs = 8
    
    print("Models and prompt loaded.")

    # 3) 처리
    for in_path, out_path_base in todo_tasks:
        if not in_path.exists():
            print(f"Processing file: {in_path}")
            print(f"[Error] Input not found: {in_path}")
            continue
            
        print(f"Processing file: {in_path}")
        
        df = pd.read_json(in_path, lines=True)
        for c in OUTPUT_COLS[:-2]:
             if c not in df.columns:
                 df[c] = "" if "score_" not in c else 0

        prompts_list = [
            safe_format_prompt(
                prompt_tmpl,
                question=r.get("noisy_question") or r.get("question", ""),
                options=format_options(r.get("options", "")),
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

        # Token stats
        try:
            sample_n = min(200, len(prompts_list))
            lens = []
            for p in prompts_list[:sample_n]:
                chat = apply_chat(tok, p)
                ids = tok(chat, add_special_tokens=False)["input_ids"]
                lens.append(len(ids))

            lens_sorted = sorted(lens)
            p95 = lens_sorted[int(len(lens_sorted) * 0.95)] if lens_sorted else 0
            p99 = lens_sorted[int(len(lens_sorted) * 0.99)] if lens_sorted else 0
            mx = max(lens) if lens else 0
            avg = sum(lens) / len(lens) if lens else 0

            print(f"[Prompt Token Stats] samples={sample_n} avg={avg:.1f} p95={p95} p99={p99} max={mx}")

            if mx >= CONFIG["max_length"]["adjudicator"]:
                print(f"[Warning] max prompt tokens ({mx}) >= max_length ({CONFIG['max_length']['adjudicator']})")
            elif p95 >= int(CONFIG["max_length"]["adjudicator"] * 0.9):
                print(f"[Warning] p95 ({p95}) close to max_length ({CONFIG['max_length']['adjudicator']})")
        except Exception as e:
            print(f"[Prompt Token Stats] skipped: {e}")

        # 배치 생성
        all_labels, all_raw = [], []
        for i in tqdm(range(0, len(prompts_list), bs), desc=f"Processing {in_path.name}"):
            batch = prompts_list[i:i+bs]
            labels, raws = forced_label_generate(
                mdl, tok, batch,
                max_length=CONFIG["max_length"]["adjudicator"],
                max_new_tokens=CONFIG["max_new_tokens"]["adjudicator"],
                first_device=dev
            )
            all_labels.extend(labels)
            all_raw.extend(raws)

        # 결과 저장
        out_df = pd.DataFrame(index=df.index)
        for c in OUTPUT_COLS:
            if c in df.columns:
                out_df[c] = df[c]
            else:
                out_df[c] = "" if "score_" not in c else 0
        out_df["final_answer"] = [str(x).upper() for x in all_labels]
        out_df["mediator_raw"] = all_raw
        
        csv_path = out_path_base.with_suffix(".csv")
        jsonl_path = out_path_base.with_suffix(".jsonl")
        
        out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        out_df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

        print(f"[Saved] {csv_path}")
        print(f"[Saved] {jsonl_path}")

    del mdl, tok
    force_clean_gpu()
    print("[Stage 2] Done.")


# =========================================================
# Main
# =========================================================
def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"[Config] Stage={CONFIG['run']['stage']} | Devices={CONFIG['devices']}")

    prompts = {
        "reasoner": read_text(PROMPT_DIR / "medical_reasoner.txt"),
        "reviewer": read_text(PROMPT_DIR / "critical_medical_reviewer.txt"),
        "adjudicator": read_text(PROMPT_DIR / "final_medical_adjudicator.txt"),
    }
    
    files = sorted(glob.glob(str(BASE_DIR / "data/processed/medqa/mlm/**/*.jsonl"), recursive=True))
    print(f"[Files] {len(files)} target files.")

    if CONFIG["run"]["stage"] in {"1", "all"}:
        run_stage1_model_centric(files, prompts)

    if CONFIG["run"]["stage"] in {"2", "all"}:
        run_stage2_all(files, prompts)

    print("\n[Done] All tasks finished.")

if __name__ == "__main__":
    main()