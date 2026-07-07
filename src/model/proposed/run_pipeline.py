# -*- coding: utf-8 -*-
"""
run_pipeline.py

Multi-agent cross-review pipeline (MedMCR) — end-to-end runner.

Stage 1 : Independent Reasoning -> Cross-Review
    Phase 1 : Reasoner A  (answer_A)
    Phase 2 : Reasoner B  (answer_B) + Review B->A
    Phase 3 : Review A->B
Stage 2 : Final Adjudication (forced A/B/C/D decoding, gpt-oss-20b)

This file merges the two scripts that were previously run separately
(model_centric_baseline.py for stage 1, final_adjudiator.py for stage 2).

IMPORTANT — reproducibility notes
---------------------------------
* Stage 2 settings (max_length=1500, options passed as-is, forced 1-token
  decoding) reproduce the numbers reported in the paper. DO NOT change them.
* Stage 2 auto-discovers every stage1_results.jsonl under RESULTS_ROOT, so
  clean + all 6 noise types are adjudicated in a single run. (Previously BASE
  had to be edited by hand per noise type.)

Run:
    # everything (stage 1 then stage 2)
    python run_pipeline.py                       # CONFIG["run"]["stage"] = "all"

    # or a single stage — edit CONFIG["run"]["stage"] to "1" / "2" / "all"

Timeout workaround:
    export HF_HUB_HTTP_TIMEOUT=300
    export HF_HUB_ETAG_TIMEOUT=300
"""

import os, re, gc, json, glob, time
from pathlib import Path
from datetime import timedelta
from typing import List, Dict, Tuple, Optional

import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList,
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
# Project root autodetect
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
# Paths & Config
# -------------------------
RESULTS_ROOT = BASE_DIR / "results" / "proposed" / "v4"
PROMPT_DIR   = BASE_DIR / "prompts" / "v1"         # flat: prompts/<name>.txt  (no version subfolder)
DATA_CLEAN   = BASE_DIR / "data" / "raw" / "medqa" / "medqa_all_clean.jsonl"
DATA_NOISE   = BASE_DIR / "data" / "processed" / "medqa"   # <noise>/<level>/<seed>.jsonl

CONFIG = {
    "run": {
        "stage": "all",          # "1" | "2" | "all"
    },
    "models": {
        "reasoner_A":  "Qwen/Qwen2.5-32B-Instruct",
        "reasoner_B":  "google/gemma-3-27b-it",
        "adjudicator": "openai/gpt-oss-20b",
    },
    "devices": "auto",
    "quant_4bit": {"reasoner_A": False, "reasoner_B": False},

    "batch_size": 32,            # stage 1
    "adjudicator_batch_size": 8, # stage 2

    "max_length": {"reasoner": 1600, "reviewer": 1600, "adjudicator": 1500},  # 1500 = paper value, do not change
    "max_new_tokens": {"reasoner": 300, "reviewer": 200, "adjudicator": 1},
    "sampling": {
        "reasoner_A": {"do_sample": True,  "temperature": 0.7},
        "reasoner_B": {"do_sample": True,  "temperature": 0.7},
        "reviewer":   {"do_sample": False, "temperature": 0.0},
    },
    "safety": {
        "oom_fallback": True,
        "torch_compile": True,
    },

    # 5 random seeds for the clean set (noise sets use their embedded seed)
    "clean_seeds": [42, 123, 456, 789, 1024],
    "default_seed": 42,
}

PROMPT_FILES = {
    "reasoner":    "medical_reasoner.txt",
    "reviewer":    "critical_medical_reviewer.txt",
    "adjudicator": "final_medical_adjudicator.txt",
}

# Stage 2 output columns (canonical order from final_adjudiator.py)
OUTPUT_COLS = [
    "question", "noisy_question", "noise_meta", "options", "answer",
    "answer_A", "reasoning_A",
    "answer_B", "reasoning_B",
    "review_score_B_on_A", "review_feedback_B_on_A",
    "review_score_A_on_B", "review_feedback_A_on_B",
    "final_answer", "mediator_raw",
]


# =========================================================
# Utilities
# =========================================================
def parse_path_info(input_path: str) -> Tuple[str, str, str]:
    """Extract (noise, level, seed_tag) from a processed-data path."""
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
                add_generation_prompt=True, tokenize=False,
            )
    except Exception:
        pass
    return f"User:\n{user_text}\n\nAssistant:"


def safe_format_prompt(template: str, **kwargs) -> str:
    """Fill {placeholders} while escaping stray braces (for JSON-shaped prompts)."""
    placeholders = re.findall(r"\{(\w+)\}", template)
    tmap = {p: f"__TOK_{p.upper()}__" for p in placeholders}
    for p, tok in tmap.items():
        template = template.replace(f"{{{p}}}", tok)
    template = template.replace("{", "{{").replace("}", "}}")
    for p, tok in tmap.items():
        template = template.replace(tok, f"{{{p}}}")
    skw = {k: str(v).replace("{", "{{").replace("}", "}}") for k, v in kwargs.items()}
    return template.format(**skw)


def safe_format(template: str, **kw) -> str:
    """Lightweight {a.b} substitution used by stage-1 prompts."""
    def repl(m):
        key = m.group(1)
        parts = key.split(".")
        val = kw.get(parts[0], "")
        for p in parts[1:]:
            if isinstance(val, dict):
                val = val.get(p, "")
            else:
                val = ""
                break
        return str(val)
    return re.sub(r"\{([a-zA-Z0-9_\.]+)\}", repl, template)


def format_options(options):
    """dict -> 'A) ...\nB) ...' (used in STAGE 1 only)."""
    if isinstance(options, str):
        try:
            options = json.loads(options.replace("'", '"'))
        except Exception:
            return options
    if isinstance(options, dict):
        return "\n".join([f"{k}) {v}" for k, v in sorted(options.items())])
    return str(options)


def _normalize_and_extract_json(s: str) -> Optional[str]:
    if not isinstance(s, str) or not s.strip():
        return None
    s = (s.replace("\u201c", '"').replace("\u201d", '"')
           .replace("\u2018", "'").replace("\u2019", "'"))
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if m:
        s = m.group(1).strip()
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def parse_reasoner_json_robust(s: str) -> dict:
    default = {"answer": "?", "reasoning": "Parsing Failed"}
    js = _normalize_and_extract_json(s) or s
    obj = None
    try:
        obj = json.loads(js)
    except Exception:
        if repair_json:
            try:
                obj = repair_json(js, return_objects=True)
            except Exception:
                pass
    if not isinstance(obj, dict):
        return default
    ans = str(obj.get("answer", "?")).strip().upper()
    rea = str(obj.get("reasoning", "No reasoning provided.")).strip()
    match = re.search(r"[ABCD]", ans)
    ans = match.group(0) if match else "?"
    return {"answer": ans, "reasoning": rea}


def parse_reviewer_json_robust(s: str) -> dict:
    default = {"score": 0, "feedback": "Parsing Failed"}
    js = _normalize_and_extract_json(s) or s
    obj = None
    try:
        obj = json.loads(js)
    except Exception:
        if repair_json:
            try:
                obj = repair_json(js, return_objects=True)
            except Exception:
                pass
    if not isinstance(obj, dict):
        return default
    score = obj.get("score")
    feedback = str(obj.get("feedback", "No feedback provided.")).strip()
    v = 0
    try:
        if isinstance(score, (int, float)):
            v = int(score)
        elif isinstance(score, str):
            clean = re.sub(r"[^0-9\.]", "", score)
            if clean:
                v = int(float(clean))
    except Exception:
        v = 0
    return {"score": max(1, min(5, v)), "feedback": feedback}


def force_clean_gpu():
    print("\n[Memory] Force cleanup...")
    for _ in range(3):
        gc.collect()
    torch.cuda.empty_cache()
    try:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
    except Exception as e:
        print(f"[Memory] Dynamo reset failed: {e}")
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024 ** 3
        r = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"[Memory] Allocated {a:.2f} GB | Reserved {r:.2f} GB\n")


# =========================================================
# Model loading
# =========================================================
def build_bnb_4bit():
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )


@torch.no_grad()
def load_model_auto(model_id: str, quant4bit: bool = False, *, compile_ok: bool = True):
    """Stage-1 loader: flash-attn + optional torch.compile."""
    print(f"[Load] {model_id} (device='auto')")
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("[WARN] CUDA not available -> CPU (flash-attn disabled)")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"

    kwargs = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="auto",
        attn_implementation="flash_attention_2" if use_cuda else "eager",
    )
    if quant4bit and use_cuda:
        kwargs["quantization_config"] = build_bnb_4bit()
        kwargs.pop("dtype", None)

    mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()

    if compile_ok and CONFIG["safety"].get("torch_compile", False) and use_cuda and not quant4bit:
        print("[Compile] torch.compile(mode='reduce-overhead')")
        try:
            mdl = torch.compile(mdl, mode="reduce-overhead")
        except Exception as e:
            print(f"[Warn] torch.compile failed: {e}")

    first_device = "cpu"
    if use_cuda and getattr(mdl, "hf_device_map", None):
        for v in mdl.hf_device_map.values():
            if isinstance(v, str) and v.startswith("cuda"):
                first_device = v
                break
            if isinstance(v, int):
                first_device = f"cuda:{v}"
                break
    elif use_cuda:
        first_device = "cuda:0"
    return tok, mdl, first_device


@torch.no_grad()
def load_adjudicator(model_id: str, device_map="auto"):
    """
    Stage-2 loader — kept deliberately minimal to reproduce the paper's
    adjudication results (no flash-attn, no compile, greedy forced decode).
    """
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA GPU not detected")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "left"
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, low_cpu_mem_usage=True, device_map=device_map,
    ).eval()
    first_device = next(iter(mdl.hf_device_map.values())) if hasattr(mdl, "hf_device_map") else "cuda:0"
    return tok, mdl, first_device


# =========================================================
# Stage-1 generation
# =========================================================
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
    if do_sample:
        gen_kwargs["temperature"] = temperature
    outputs = model.generate(**enc, **gen_kwargs)
    in_len = enc["input_ids"].size(1)
    return [tokenizer.decode(o[in_len:], skip_special_tokens=True).strip() for o in outputs]


def batched_generate_with_fallback(generate_fn, prompts: List[str], batch_size: int, desc: str = "") -> List[str]:
    outs, i = [], 0
    pbar = tqdm(total=len(prompts), desc=desc, leave=False)
    while i < len(prompts):
        bs = batch_size
        while True:
            try:
                chunk = prompts[i:i + bs]
                outs.extend(generate_fn(chunk))
                i += bs
                pbar.update(bs)
                break
            except torch.cuda.OutOfMemoryError:
                force_clean_gpu()
                if bs <= 1:
                    print(f"[FATAL OOM] index {i}")
                    raise
                bs = max(1, bs // 2)
                print(f"[OOM] retry batch={bs}")
    pbar.close()
    return outs


def run_stage1_single_file_seed(input_path, noise, level, seed_tag, actual_seed, prompts):
    """Run all 3 stage-1 phases for one file + one seed."""
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)

    work_dir = RESULTS_ROOT / noise / level / seed_tag
    work_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = work_dir / "stage1_results.jsonl"

    reasoner_tmpl = prompts["reasoner"]
    reviewer_tmpl = prompts["reviewer"]
    bs = int(CONFIG["batch_size"])

    # ---------- Phase 1: Reasoner A ----------
    print(f"\n[Phase 1/3] {noise}/{level}/{seed_tag} (seed={actual_seed}) - Reasoner A")
    need = True
    if stage1_path.exists():
        try:
            df = pd.read_json(stage1_path, lines=True)
            if "answer_A" in df.columns and df["answer_A"].astype(str).str.strip().all():
                need = False
                print("  [Skip] answer_A exists")
        except Exception:
            pass
    if need:
        tok, mdl, dev = load_model_auto(CONFIG["models"]["reasoner_A"], CONFIG["quant_4bit"]["reasoner_A"])
        t0 = time.time()
        df = pd.read_json(stage1_path if stage1_path.exists() else input_path, lines=True)
        for c in ["reasoning_A", "answer_A", "raw_reasoner_A"]:
            df[c] = df.get(c, "")
        qs = [r.get("noisy_question") or r["question"] for _, r in df.iterrows()]
        p_list = []
        for i, r in df.iterrows():
            if df.iloc[i]["answer_A"]:
                p_list.append("dummy")
            else:
                opts = r["options"] if isinstance(r["options"], dict) else {}
                p_list.append(safe_format(reasoner_tmpl, question=qs[i], options=format_options(opts)))
        todo = [i for i, p in enumerate(p_list) if p != "dummy"]
        if todo:
            def _gen(chunk):
                return generate_full_output(
                    mdl, tok, chunk,
                    max_len=CONFIG["max_length"]["reasoner"], max_new=CONFIG["max_new_tokens"]["reasoner"],
                    do_sample=CONFIG["sampling"]["reasoner_A"]["do_sample"],
                    temperature=CONFIG["sampling"]["reasoner_A"]["temperature"], device=dev)
            raw_outs = batched_generate_with_fallback(_gen, [p_list[i] for i in todo], bs, "ReasonA")
            for idx, raw in zip(todo, raw_outs):
                parsed = parse_reasoner_json_robust(raw)
                df.at[idx, "raw_reasoner_A"] = raw
                df.at[idx, "answer_A"] = parsed["answer"]
                df.at[idx, "reasoning_A"] = parsed["reasoning"]
        df.to_json(stage1_path, orient="records", lines=True, force_ascii=False)
        print(f"  >>> Done. {timedelta(seconds=int(time.time() - t0))}")
        del mdl, tok
        force_clean_gpu()

    # ---------- Phase 2: Reasoner B + Review B->A ----------
    print(f"\n[Phase 2/3] {noise}/{level}/{seed_tag} - Reasoner B + Review B->A")
    need = True
    if stage1_path.exists():
        try:
            df = pd.read_json(stage1_path, lines=True)
            cols = ["answer_B", "review_feedback_B_on_A"]
            if all(c in df.columns and df[c].astype(str).str.strip().all() for c in cols):
                need = False
                print("  [Skip] answer_B & review B->A exist")
        except Exception:
            pass
    if need:
        tok, mdl, dev = load_model_auto(CONFIG["models"]["reasoner_B"], CONFIG["quant_4bit"]["reasoner_B"])
        t0 = time.time()
        df = pd.read_json(stage1_path, lines=True)
        for c in ["reasoning_B", "answer_B", "raw_reasoner_B",
                  "review_score_B_on_A", "review_feedback_B_on_A", "raw_review_B_on_A"]:
            df[c] = df.get(c, "")
        qs = [r.get("noisy_question") or r["question"] for _, r in df.iterrows()]

        # answer B
        p_list = []
        for i, r in df.iterrows():
            if df.iloc[i]["answer_B"]:
                p_list.append("dummy")
            else:
                opts = r["options"] if isinstance(r["options"], dict) else {}
                p_list.append(safe_format(reasoner_tmpl, question=qs[i], options=format_options(opts)))
        todo = [i for i, p in enumerate(p_list) if p != "dummy"]
        if todo:
            def _gen_b(chunk):
                return generate_full_output(
                    mdl, tok, chunk,
                    max_len=CONFIG["max_length"]["reasoner"], max_new=CONFIG["max_new_tokens"]["reasoner"],
                    do_sample=CONFIG["sampling"]["reasoner_B"]["do_sample"],
                    temperature=CONFIG["sampling"]["reasoner_B"]["temperature"], device=dev)
            raw_outs = batched_generate_with_fallback(_gen_b, [p_list[i] for i in todo], bs, "ReasonB")
            for idx, raw in zip(todo, raw_outs):
                parsed = parse_reasoner_json_robust(raw)
                df.at[idx, "raw_reasoner_B"] = raw
                df.at[idx, "answer_B"] = parsed["answer"]
                df.at[idx, "reasoning_B"] = parsed["reasoning"]

        # review B->A
        p_rev = []
        for i, r in df.iterrows():
            if df.iloc[i]["review_feedback_B_on_A"]:
                p_rev.append("dummy")
            else:
                opts = r["options"] if isinstance(r["options"], dict) else {}
                p_rev.append(safe_format(reviewer_tmpl, question=qs[i], options=format_options(opts),
                                         reasoning=r["reasoning_A"], answer=r["answer_A"]))
        todo_rev = [i for i, p in enumerate(p_rev) if p != "dummy"]
        if todo_rev:
            def _gen_rev(chunk):
                return generate_full_output(
                    mdl, tok, chunk,
                    max_len=CONFIG["max_length"]["reviewer"], max_new=CONFIG["max_new_tokens"]["reviewer"],
                    do_sample=CONFIG["sampling"]["reviewer"]["do_sample"],
                    temperature=CONFIG["sampling"]["reviewer"]["temperature"], device=dev)
            raw_outs = batched_generate_with_fallback(_gen_rev, [p_rev[i] for i in todo_rev], bs, "Review B->A")
            for idx, raw in zip(todo_rev, raw_outs):
                parsed = parse_reviewer_json_robust(raw)
                df.at[idx, "raw_review_B_on_A"] = raw
                df.at[idx, "review_score_B_on_A"] = parsed["score"]
                df.at[idx, "review_feedback_B_on_A"] = parsed["feedback"]

        df.to_json(stage1_path, orient="records", lines=True, force_ascii=False)
        print(f"  >>> Done. {timedelta(seconds=int(time.time() - t0))}")
        del mdl, tok
        force_clean_gpu()

    # ---------- Phase 3: Review A->B ----------
    print(f"\n[Phase 3/3] {noise}/{level}/{seed_tag} - Review A->B")
    need = True
    if stage1_path.exists():
        try:
            df = pd.read_json(stage1_path, lines=True)
            if "review_feedback_A_on_B" in df.columns and df["review_feedback_A_on_B"].astype(str).str.strip().all():
                need = False
                print("  [Skip] review A->B exists")
        except Exception:
            pass
    if need:
        tok, mdl, dev = load_model_auto(CONFIG["models"]["reasoner_A"], CONFIG["quant_4bit"]["reasoner_A"])
        t0 = time.time()
        df = pd.read_json(stage1_path, lines=True)
        for c in ["review_score_A_on_B", "review_feedback_A_on_B", "raw_review_A_on_B"]:
            df[c] = df.get(c, "")
        qs = [r.get("noisy_question") or r["question"] for _, r in df.iterrows()]
        p_rev = []
        for i, r in df.iterrows():
            if df.iloc[i]["review_feedback_A_on_B"]:
                p_rev.append("dummy")
            else:
                opts = r["options"] if isinstance(r["options"], dict) else {}
                p_rev.append(safe_format(reviewer_tmpl, question=qs[i], options=format_options(opts),
                                         reasoning=r["reasoning_B"], answer=r["answer_B"]))
        todo_rev = [i for i, p in enumerate(p_rev) if p != "dummy"]
        if todo_rev:
            def _gen_rev(chunk):
                return generate_full_output(
                    mdl, tok, chunk,
                    max_len=CONFIG["max_length"]["reviewer"], max_new=CONFIG["max_new_tokens"]["reviewer"],
                    do_sample=CONFIG["sampling"]["reviewer"]["do_sample"],
                    temperature=CONFIG["sampling"]["reviewer"]["temperature"], device=dev)
            raw_outs = batched_generate_with_fallback(_gen_rev, [p_rev[i] for i in todo_rev], bs, "Review A->B")
            for idx, raw in zip(todo_rev, raw_outs):
                parsed = parse_reviewer_json_robust(raw)
                df.at[idx, "raw_review_A_on_B"] = raw
                df.at[idx, "review_score_A_on_B"] = parsed["score"]
                df.at[idx, "review_feedback_A_on_B"] = parsed["feedback"]
        df.to_json(stage1_path, orient="records", lines=True, force_ascii=False)
        print(f"  >>> Done. {timedelta(seconds=int(time.time() - t0))}")
        del mdl, tok
        force_clean_gpu()


def run_stage1(files: List[str], prompts: dict):
    print(f"\n{'=' * 60}\n[STAGE 1] Multi-Agent Reasoning (clean uses 5 seeds)\n{'=' * 60}")
    for input_path in tqdm(files, desc="Files"):
        noise, level, seed_orig = parse_path_info(input_path)
        if noise == "clean":
            seed_list = [(f"seed_{i}", s) for i, s in enumerate(CONFIG["clean_seeds"])]
        else:
            seed_list = [(seed_orig, CONFIG["default_seed"])]
        for seed_tag, actual_seed in seed_list:
            run_stage1_single_file_seed(input_path, noise, level, seed_tag, actual_seed, prompts)


# =========================================================
# Stage-2 forced decoding (A/B/C/D)
# =========================================================
class AllowOnly(LogitsProcessor):
    def __init__(self, ids):
        self.ids = set(ids)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, -float("inf"))
        if self.ids:
            mask[:, list(self.ids)] = 0.0
        return scores + mask


def _single_token_ids(tokenizer, letter: str):
    out = set()
    for pref in ["", " ", "\u2581", "\u0120"]:
        t = tokenizer.encode(pref + letter, add_special_tokens=False)
        if len(t) == 1:
            out.add(t[0])
    return sorted(out)


def build_allowed_ids(tokenizer):
    ids = []
    for L in ["A", "B", "C", "D", "a", "b", "c", "d"]:
        ids += _single_token_ids(tokenizer, L)
    return sorted(set(ids))


@torch.no_grad()
def forced_label_generate(model, tokenizer, prompts, *, max_length, first_device):
    # prime the JSON so the model only needs to emit the label token
    chats = [apply_chat(tokenizer, p) + '{"label":"' for p in prompts]
    enc = tokenizer(chats, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    enc = {k: v.to(first_device) for k, v in enc.items()}

    allowed_ids = build_allowed_ids(tokenizer)
    processors = LogitsProcessorList([AllowOnly(allowed_ids)]) if allowed_ids else None
    pad_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0

    gen_kwargs = dict(max_new_tokens=1, min_new_tokens=1, do_sample=False,
                      pad_token_id=pad_id, eos_token_id=pad_id)
    if allowed_ids:
        gen_kwargs["logits_processor"] = processors
    out = model.generate(**enc, **gen_kwargs)

    ilen = enc["input_ids"].size(1)
    raws, labels = [], []
    for i in range(out.size(0)):
        txt = tokenizer.decode(out[i, ilen:], skip_special_tokens=True)
        lab = (txt.strip().replace('"', "").replace("'", "")[:1] or "?").upper()
        if lab not in {"A", "B", "C", "D"}:
            m = re.search(r"[ABCD]", txt.upper())
            lab = m.group(0) if m else "?"
        raws.append(f'{{"label":"{lab}"}}')
        labels.append(lab)
    return labels, raws


def run_stage2(prompts: dict):
    """
    Adjudicate every stage1_results.jsonl found under RESULTS_ROOT.
    Output: stage2_results_final.csv / .jsonl next to each input.
    """
    print(f"\n{'=' * 60}\n[STAGE 2] Final Adjudication\n{'=' * 60}")

    tasks = []
    for stage1_file in sorted(RESULTS_ROOT.rglob("stage1_results.jsonl")):
        base = stage1_file.parent / "stage2_results_final"
        csv_p, jsonl_p = base.with_suffix(".csv"), base.with_suffix(".jsonl")
        if csv_p.exists() and jsonl_p.exists():
            print(f"[Skip] {csv_p}")
            continue
        tasks.append((stage1_file, base))
    if not tasks:
        print("Nothing to adjudicate.")
        return

    print("Loading model...")
    tok, mdl, dev = load_adjudicator(CONFIG["models"]["adjudicator"], CONFIG["devices"])
    prompt_tmpl = prompts["adjudicator"]
    bs = int(CONFIG["adjudicator_batch_size"])
    max_len = CONFIG["max_length"]["adjudicator"]
    print("Models and prompt loaded.")

    for in_path, out_base in tasks:
        print(f"Processing file: {in_path}")
        df = pd.read_json(in_path, lines=True)
        for c in OUTPUT_COLS[:-2]:
            if c not in df.columns:
                df[c] = 0 if "score_" in c else ""

        # NOTE: options passed as-is (paper-canonical). Do not wrap with format_options here.
        prompts_list = [
            safe_format_prompt(
                prompt_tmpl,
                question=r.get("noisy_question") or r.get("question", ""),
                options=r.get("options", ""),
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

        # prompt-length sanity check
        try:
            sample_n = min(200, len(prompts_list))
            lens = [len(tok(apply_chat(tok, p), add_special_tokens=False)["input_ids"])
                    for p in prompts_list[:sample_n]]
            if lens:
                ls = sorted(lens)
                p95 = ls[int(len(ls) * 0.95)]
                mx = max(lens)
                print(f"[Prompt Token Stats] n={sample_n} avg={sum(lens)/len(lens):.1f} p95={p95} max={mx} (max_length={max_len})")
                if mx >= max_len:
                    print(f"[Warning] max prompt tokens ({mx}) >= max_length ({max_len}). Truncation likely.")
        except Exception as e:
            print(f"[Prompt Token Stats] skipped: {e}")

        all_labels, all_raw = [], []
        for i in tqdm(range(0, len(prompts_list), bs), desc=f"Adjudicate {in_path.parent.name}"):
            labels, raws = forced_label_generate(
                mdl, tok, prompts_list[i:i + bs], max_length=max_len, first_device=dev)
            all_labels.extend(labels)
            all_raw.extend(raws)

        out_df = pd.DataFrame(index=df.index)
        for c in OUTPUT_COLS:
            if c in df.columns:
                out_df[c] = df[c]
            else:
                out_df[c] = 0 if "score_" in c else ""
        out_df["final_answer"] = [str(x).upper() for x in all_labels]
        out_df["mediator_raw"] = all_raw

        out_base.parent.mkdir(parents=True, exist_ok=True)
        csv_p, jsonl_p = out_base.with_suffix(".csv"), out_base.with_suffix(".jsonl")
        out_df.to_csv(csv_p, index=False, encoding="utf-8-sig")
        out_df.to_json(jsonl_p, orient="records", lines=True, force_ascii=False)
        print(f"[Saved] {csv_p}")
        print(f"[Saved] {jsonl_p}")

    del mdl, tok
    force_clean_gpu()
    print("[Stage 2] Done.")


# =========================================================
# Main
# =========================================================
def discover_stage1_files() -> List[str]:
    """clean (raw) + every processed noise file."""
    clean = [str(DATA_CLEAN)] if DATA_CLEAN.exists() else []
    noisy = sorted(glob.glob(str(DATA_NOISE / "**" / "*.jsonl"), recursive=True))
    return clean + noisy


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    stage = CONFIG["run"]["stage"]
    print(f"[Config] Stage={stage} | Devices={CONFIG['devices']} | Root={BASE_DIR}")

    prompts = {k: read_text(PROMPT_DIR / fn) for k, fn in PROMPT_FILES.items()}

    if stage in {"1", "all"}:
        files = discover_stage1_files()
        print(f"[Files] {len(files)} stage-1 inputs")
        run_stage1(files, prompts)

    if stage in {"2", "all"}:
        run_stage2(prompts)

    print("\n[Done] All tasks finished.")


if __name__ == "__main__":
    main()