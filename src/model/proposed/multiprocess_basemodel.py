# -*- coding: utf-8 -*-
"""
run_multiagent_medqa_mp.py

- Stage1: Reasoner A (GPU0) + Reasoner B (GPU1) 를 멀티프로세스로 동시에 generate
         + Cross-review(BA/AB)도 동시에 generate
- Stage2: Adjudicator (원래대로 device_map="auto")

CONFIG["run"]["stage"] in {"1","2","all"}
env -u LD_LIBRARY_PATH -u CUDA_HOME -u CUDA_PATH LD_LIBRARY_PATH=/usr/lib64     /home/hslee/multiagent/.venv/bin/python test_multiprocess.py
"""

import os, re, gc, json, glob, time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
    LogitsProcessor, LogitsProcessorList,
)
from llm_json_fixer import fix_json

import multiprocessing as mp


# -------------------------
# Env safety
# -------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# -------------------------
# 0) Project root autodetect
# -------------------------
THIS_FILE = Path(__file__).resolve()
BASE_DIR = THIS_FILE
for _ in range(8):
    if (BASE_DIR / "pyproject.toml").exists() or (BASE_DIR / "README.md").exists():
        break
    BASE_DIR = BASE_DIR.parent
BASE_DIR = BASE_DIR.resolve()

# -------------------------
# 1) Paths (match your tree)
# -------------------------
DATA_DIR = BASE_DIR / "data" / "processed" / "medqa"
RAW_DATA = BASE_DIR / "data" / "raw" / "medqa" / "medqa_all_clean.jsonl"
RESULTS_ROOT = BASE_DIR / "results" / "proposed" / "v2_basemodel"
PROMPT_DIR = BASE_DIR / "prompts" / "v1"

# -------------------------
# 2) CONFIG (EDIT HERE)
# -------------------------
CONFIG = {
    "run": {
        "stage": "all",          # "1" / "2" / "all"
        "include_raw": True,
    },
    "models": {
        "reasoner_A": "Qwen/Qwen2.5-32B-Instruct",
        "reasoner_B": "google/gemma-3-27b-it",
        "adjudicator": "openai/gpt-oss-20b",
    },
    "devices": {"reasoner_A": 0, "reasoner_B": 1},
    "quant_4bit": {"reasoner_A": False, "reasoner_B": False},

    "batch_size": 32,
    "max_length": {"reasoner": 2048, "reviewer": 1536, "adjudicator": 1536},
    "max_new_tokens": {"reasoner": 192, "reviewer": 80, "adjudicator": 1},

    "sampling": {
        "reasoner_A": {"do_sample": True, "temperature": 1.0},
        "reasoner_B": {"do_sample": True, "temperature": 1.0},
        "reviewer":   {"do_sample": False, "temperature": 0.0},
    },

    "safety": {
        "oom_fallback": True,
        "warn_missing_placeholders": True,
    },
    "seed": 42,

    # MP worker communication
    "mp": {
        "timeout_sec": 3600,   # worker 응답 기다리는 최대 시간(안전)
    }
}


# =========================================================
# Utilities
# =========================================================
def get_all_jsonl_files() -> List[str]:
    files = []
    if CONFIG["run"]["include_raw"] and RAW_DATA.exists():
        files.append(RAW_DATA.as_posix())
    files.extend(glob.glob(str(DATA_DIR / "**" / "*.jsonl"), recursive=True))
    return sorted(list(set(files)))


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
    return p.read_text(encoding="utf-8").strip()


def apply_chat(tokenizer, user_text: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        return f"User:\n{user_text}\n\nAssistant:"


def safe_format(template: str, **kw) -> str:
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


def safe_format_checked(template: str, values: Dict, must_have: List[str], warn_prefix: str = "") -> str:
    txt = safe_format(template, **values)
    if not CONFIG["safety"]["warn_missing_placeholders"]:
        return txt

    missing = []
    for key in must_have:
        if f"{{{key}}}" not in template:
            continue
        if "." in key:
            root, child = key.split(".", 1)
            v = values.get(root, None)
            if not isinstance(v, dict) or v.get(child, "") in ["", None]:
                missing.append(key)
        else:
            if values.get(key, "") in ["", None]:
                missing.append(key)

    if missing:
        print(f"[WARN]{warn_prefix} Missing placeholders: {missing}")
    return txt


def build_bnb_4bit():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


@torch.no_grad()
def load_reasoner_model_single_visible_gpu(model_id: str, quant4bit: bool):
    """
    IMPORTANT:
      Worker 프로세스에서 CUDA_VISIBLE_DEVICES로 GPU를 1개만 보이게 만든 뒤 호출.
      그러면 device는 항상 cuda:0만 쓰면 됨.
    """
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"

    kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    if quant4bit:
        kwargs["quantization_config"] = build_bnb_4bit()

    # Worker에서는 GPU가 1개만 보이므로 자동 배치해도 결국 cuda:0으로 올라감
    kwargs["device_map"] = {"": 0}

    torch.cuda.set_device(0)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    return tok, mdl, "cuda:0"


@torch.no_grad()
def load_adjudicator(model_id: str, device_map="auto"):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"

    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    ).eval()

    # choose one cuda device for input tensors
    first_device = "cuda:0"
    if hasattr(mdl, "hf_device_map") and mdl.hf_device_map:
        for v in mdl.hf_device_map.values():
            if isinstance(v, str) and v.startswith("cuda"):
                first_device = v
                break
    return tok, mdl, first_device


# =========================================================
# JSON extraction + parsing
# =========================================================
def _normalize_and_extract_json(s: str) -> Optional[str]:
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

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
                return s[start:i+1]
    return None


def parse_reasoner_json_robust(s: str) -> dict:
    default = {"answer": "?", "reasoning": "Parsing Failed"}
    js = _normalize_and_extract_json(s)
    if not js:
        return default
    try:
        obj = json.loads(js)
    except json.JSONDecodeError:
        try:
            obj = json.loads(fix_json(js))
        except Exception:
            return default
    if not isinstance(obj, dict):
        return default

    ans = str(obj.get("answer", "?")).strip().upper()
    rea = str(obj.get("reasoning", "No reasoning provided.")).strip()
    if ans not in {"A", "B", "C", "D"}:
        ans = "?"
    return {"answer": ans, "reasoning": rea if rea else "No reasoning provided."}


def parse_reviewer_json_robust(s: str) -> dict:
    default = {"score": 0, "feedback": "Parsing Failed"}
    js = _normalize_and_extract_json(s)
    if not js:
        return default
    try:
        obj = json.loads(js)
    except json.JSONDecodeError:
        try:
            obj = json.loads(fix_json(js))
        except Exception:
            return default
    if not isinstance(obj, dict):
        return default

    score = obj.get("score")
    feedback = str(obj.get("feedback", "No feedback provided.")).strip()

    v = 0
    if isinstance(score, int) and 1 <= score <= 5:
        v = score
    elif isinstance(score, str) and score.isdigit() and 1 <= int(score) <= 5:
        v = int(score)
    return {"score": v, "feedback": feedback if feedback else "No feedback provided."}


# =========================================================
# Generation helpers
# =========================================================
@torch.no_grad()
def generate_full_output(model, tokenizer, prompts, *, max_len, max_new, do_sample, temperature, device):
    chats = [apply_chat(tokenizer, p) for p in prompts]
    enc = tokenizer(chats, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    enc = {k: v.to(device) for k, v in enc.items()}

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id

    gen_kwargs = dict(
        max_new_tokens=int(max_new),
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        do_sample=bool(do_sample),
        top_p=1.0,
        top_k=0,
    )
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)

    outputs = model.generate(**enc, **gen_kwargs)
    in_len = enc["input_ids"].size(1)
    return [tokenizer.decode(o[in_len:], skip_special_tokens=True).strip() for o in outputs]


def batched_generate_with_fallback_local(generate_fn, prompts: List[str], batch_size: int, desc: str = "") -> List[str]:
    """
    Worker 내부에서만 쓰는 OOM fallback.
    (스레드/프로세스 경쟁 없이 해당 GPU만 관리)
    """
    outs = []
    i = 0
    while i < len(prompts):
        bs = batch_size
        while True:
            try:
                chunk = prompts[i:i+bs]
                outs.extend(generate_fn(chunk))
                i += bs
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); gc.collect()
                if not CONFIG["safety"]["oom_fallback"] or bs <= 1:
                    raise
                bs = max(1, bs // 2)
                print(f"[OOM]{desc} retry with batch={bs} (at index {i})")
    return outs


# =========================================================
# MP Worker Protocol
# =========================================================
# request: dict
# {
#   "job_id": str,
#   "kind": "reason" | "review",
#   "prompts": List[str],
#   "max_len": int,
#   "max_new": int,
#   "do_sample": bool,
#   "temperature": float,
#   "batch_size": int,
# }
#
# response: dict
# {
#   "job_id": str,
#   "ok": bool,
#   "outputs": List[str] | None,
#   "error": str | None,
# }

def mp_worker_loop(gpu_id: int, model_id: str, quant4bit: bool,
                   in_q: mp.Queue, out_q: mp.Queue):
    """
    각 워커는 자신 GPU만 보게 고정 -> cuda:0만 사용.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch as _torch
    import gc as _gc

    assert _torch.cuda.is_available(), "CUDA not available in worker"
    _torch.cuda.set_device(0)

    tok, mdl, dev = load_reasoner_model_single_visible_gpu(model_id, quant4bit=quant4bit)

    while True:
        req = in_q.get()
        if req is None:
            break

        job_id = req["job_id"]
        prompts = req["prompts"]
        max_len = int(req["max_len"])
        max_new = int(req["max_new"])
        do_sample = bool(req["do_sample"])
        temperature = float(req["temperature"])
        batch_size = int(req["batch_size"])
        kind = req.get("kind", "gen")

        try:
            def _gen(chunk):
                return generate_full_output(
                    mdl, tok, chunk,
                    max_len=max_len,
                    max_new=max_new,
                    do_sample=do_sample,
                    temperature=temperature,
                    device=dev,
                )

            outs = batched_generate_with_fallback_local(_gen, prompts, batch_size=batch_size, desc=f"{kind}:{model_id}")
            out_q.put({"job_id": job_id, "ok": True, "outputs": outs, "error": None})
        except Exception as e:
            # best-effort cleanup
            try:
                _torch.cuda.empty_cache()
                _gc.collect()
            except Exception:
                pass
            out_q.put({"job_id": job_id, "ok": False, "outputs": None, "error": repr(e)})


class MPGenPool:
    def __init__(self):
        self.ctx = mp.get_context("spawn")   # 핵심!
        self.inA = self.ctx.Queue()
        self.outA = self.ctx.Queue()
        self.inB = self.ctx.Queue()
        self.outB = self.ctx.Queue()
        self.pA = None
        self.pB = None

    def start(self):
        # set_start_method는 여기서 굳이 안 해도 됨(이미 ctx로 통일했으니까)
        self.pA = self.ctx.Process(
            target=mp_worker_loop,
            args=(int(CONFIG["devices"]["reasoner_A"]),
                  CONFIG["models"]["reasoner_A"],
                  bool(CONFIG["quant_4bit"]["reasoner_A"]),
                  self.inA, self.outA),
            daemon=True
        )
        self.pB = self.ctx.Process(
            target=mp_worker_loop,
            args=(int(CONFIG["devices"]["reasoner_B"]),
                  CONFIG["models"]["reasoner_B"],
                  bool(CONFIG["quant_4bit"]["reasoner_B"]),
                  self.inB, self.outB),
            daemon=True
        )
        self.pA.start()
        self.pB.start()
        print(f"[MP] Workers started: A->GPU{CONFIG['devices']['reasoner_A']} B->GPU{CONFIG['devices']['reasoner_B']}")


    def stop(self):
        try:
            self.inA.put(None)
            self.inB.put(None)
        except Exception:
            pass
        if self.pA is not None:
            self.pA.join(timeout=10)
        if self.pB is not None:
            self.pB.join(timeout=10)
        print("[MP] Workers stopped")

    def submit_pair(self, job_id: str, reqA: Dict[str, Any], reqB: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        A/B 요청을 동시에 넣고, 두 결과를 모두 받을 때까지 기다림.
        """
        reqA = dict(reqA); reqB = dict(reqB)
        reqA["job_id"] = job_id
        reqB["job_id"] = job_id

        self.inA.put(reqA)
        self.inB.put(reqB)

        gotA = gotB = None
        deadline = time.time() + float(CONFIG["mp"]["timeout_sec"])

        while (gotA is None) or (gotB is None):
            if time.time() > deadline:
                raise TimeoutError(f"[MP] Timeout waiting for job_id={job_id}")

            if gotA is None:
                if not self.outA.empty():
                    resp = self.outA.get()
                    if resp["job_id"] == job_id:
                        gotA = resp
            if gotB is None:
                if not self.outB.empty():
                    resp = self.outB.get()
                    if resp["job_id"] == job_id:
                        gotB = resp

            # outQ가 empty일 때 바쁜 대기 방지
            if (gotA is None) or (gotB is None):
                time.sleep(0.01)

        if not gotA["ok"]:
            raise RuntimeError(f"WorkerA failed: {gotA['error']}")
        if not gotB["ok"]:
            raise RuntimeError(f"WorkerB failed: {gotB['error']}")

        return gotA["outputs"], gotB["outputs"]


# =========================================================
# Forced A/B/C/D (Adjudicator only)
# =========================================================
_VALID = {"A", "B", "C", "D"}

class AllowOnly(LogitsProcessor):
    def __init__(self, ids):
        self.ids = set(ids)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, -float("inf"))
        mask[:, list(self.ids)] = 0.0
        return scores + mask


def _single_token_ids(tokenizer, text: str):
    t = tokenizer.encode(text, add_special_tokens=False)
    return {t[0]} if len(t) == 1 else set()


def build_allowed_ids(tokenizer):
    candidates = []
    for L in ["A", "B", "C", "D", "a", "b", "c", "d"]:
        candidates += [L, f" {L}", f"\n{L}", f"{L}."]
    for L in ["A", "B", "C", "D", "a", "b", "c", "d"]:
        candidates += [f"▁{L}", f"Ġ{L}"]

    ids = set()
    for c in candidates:
        ids |= _single_token_ids(tokenizer, c)
    return sorted(ids)


@torch.no_grad()
def forced_abcd_only(model, tokenizer, prompts, *, max_length, first_device):
    chats = [apply_chat(tokenizer, p) for p in prompts]
    enc = tokenizer(chats, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    enc = {k: v.to(first_device) for k, v in enc.items()}

    allowed_ids = build_allowed_ids(tokenizer)
    if not allowed_ids:
        raise RuntimeError("No single-token ids found for A/B/C/D in this tokenizer.")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id

    out = model.generate(
        **enc,
        max_new_tokens=1,
        min_new_tokens=1,
        do_sample=False,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        logits_processor=LogitsProcessorList([AllowOnly(allowed_ids)]),
    )

    ilen = enc["input_ids"].size(1)
    labels = []
    for i in range(out.size(0)):
        txt = tokenizer.decode(out[i, ilen:], skip_special_tokens=True).strip()
        lab = (txt[:1] or "?").upper()
        if lab not in _VALID:
            m = re.search(r"[ABCD]", txt.upper())
            lab = m.group(0) if m else "?"
        labels.append(lab)
    return labels


# =========================================================
# Stage 1 and Stage 2 per-file functions
# =========================================================
def run_stage1_only_mp(
    input_path: str,
    prompts: dict,
    pool: MPGenPool,
):
    noise, level, seed_tag = parse_path_info(input_path)
    work_dir = RESULTS_ROOT / noise / level / seed_tag
    work_dir.mkdir(parents=True, exist_ok=True)

    stage1_path = work_dir / "stage1_results.jsonl"
    if stage1_path.exists():
        print(f"[Stage1 Skip] exists: {noise}/{level}/{seed_tag}")
        return

    print(f"\n[Stage1 Run] {noise}/{level}/{seed_tag}")
    df = pd.read_json(input_path, lines=True)
    if "question" not in df.columns or "options" not in df.columns:
        raise ValueError(f"Missing required columns in {input_path}. Need 'question' and 'options'.")

    if "answer" in df.columns:
        df["true_answer"] = df["answer"].astype(str).str.strip().str.upper()
    else:
        df["true_answer"] = ""

    reasoner_tmpl = prompts["reasoner"]
    reviewer_tmpl = prompts["reviewer"]
    bs = int(CONFIG["batch_size"])

    # init columns
    df["reasoning_A"] = ""; df["answer_A"] = "?"; df["raw_reasoner_A"] = ""
    df["reasoning_B"] = ""; df["answer_B"] = "?"; df["raw_reasoner_B"] = ""
    df["review_score_B_on_A"] = 0; df["review_feedback_B_on_A"] = ""; df["raw_review_B_on_A"] = ""
    df["review_score_A_on_B"] = 0; df["review_feedback_A_on_B"] = ""; df["raw_review_A_on_B"] = ""

    for i in tqdm(range(0, len(df), bs), desc=f"Stage1 {noise}/{level}/{seed_tag}"):
        b = df.iloc[i:i+bs]

        # Reasoner prompts
        p_reasoner = []
        for _, r in b.iterrows():
            opts = r["options"] if isinstance(r["options"], dict) else {}
            p_reasoner.append(
                safe_format_checked(
                    reasoner_tmpl,
                    values={"question": r["question"], "options": opts},
                    must_have=["question", "options.A", "options.B", "options.C", "options.D"],
                    warn_prefix=f"[{noise}/{level}/{seed_tag}] ",
                )
            )

        job_id = f"{noise}|{level}|{seed_tag}|reason|{i}"

        # Reasoning A/B simultaneously
        raw_A, raw_B = pool.submit_pair(
            job_id=job_id,
            reqA={
                "kind": "reason_A",
                "prompts": p_reasoner,
                "max_len": CONFIG["max_length"]["reasoner"],
                "max_new": CONFIG["max_new_tokens"]["reasoner"],
                "do_sample": CONFIG["sampling"]["reasoner_A"]["do_sample"],
                "temperature": CONFIG["sampling"]["reasoner_A"]["temperature"],
                "batch_size": bs,
            },
            reqB={
                "kind": "reason_B",
                "prompts": p_reasoner,
                "max_len": CONFIG["max_length"]["reasoner"],
                "max_new": CONFIG["max_new_tokens"]["reasoner"],
                "do_sample": CONFIG["sampling"]["reasoner_B"]["do_sample"],
                "temperature": CONFIG["sampling"]["reasoner_B"]["temperature"],
                "batch_size": bs,
            }
        )

        parsed_A = [parse_reasoner_json_robust(s) for s in raw_A]
        parsed_B = [parse_reasoner_json_robust(s) for s in raw_B]

        df.loc[b.index, "raw_reasoner_A"] = raw_A
        df.loc[b.index, "answer_A"] = [x["answer"] for x in parsed_A]
        df.loc[b.index, "reasoning_A"] = [x["reasoning"] for x in parsed_A]

        df.loc[b.index, "raw_reasoner_B"] = raw_B
        df.loc[b.index, "answer_B"] = [x["answer"] for x in parsed_B]
        df.loc[b.index, "reasoning_B"] = [x["reasoning"] for x in parsed_B]

        # Cross-review prompts
        vprom_B_on_A = []
        for (_, r), pa in zip(b.iterrows(), parsed_A):
            opts = r["options"] if isinstance(r["options"], dict) else {}
            vprom_B_on_A.append(
                safe_format_checked(
                    reviewer_tmpl,
                    values={"question": r["question"], "options": opts, "reasoning": pa["reasoning"], "answer": pa["answer"]},
                    must_have=["question", "options.A", "options.B", "options.C", "options.D", "reasoning", "answer"],
                    warn_prefix=f"[{noise}/{level}/{seed_tag}] ",
                )
            )

        vprom_A_on_B = []
        for (_, r), pb in zip(b.iterrows(), parsed_B):
            opts = r["options"] if isinstance(r["options"], dict) else {}
            vprom_A_on_B.append(
                safe_format_checked(
                    reviewer_tmpl,
                    values={"question": r["question"], "options": opts, "reasoning": pb["reasoning"], "answer": pb["answer"]},
                    must_have=["question", "options.A", "options.B", "options.C", "options.D", "reasoning", "answer"],
                    warn_prefix=f"[{noise}/{level}/{seed_tag}] ",
                )
            )

        job_id2 = f"{noise}|{level}|{seed_tag}|review|{i}"

        # Reviews simultaneously:
        # - B reviews A  => worker B
        # - A reviews B  => worker A
        raw_BA, raw_AB = pool.submit_pair(
            job_id=job_id2,
            reqA={  # Worker A does A->B review
                "kind": "review_A_on_B",
                "prompts": vprom_A_on_B,
                "max_len": CONFIG["max_length"]["reviewer"],
                "max_new": CONFIG["max_new_tokens"]["reviewer"],
                "do_sample": CONFIG["sampling"]["reviewer"]["do_sample"],
                "temperature": CONFIG["sampling"]["reviewer"]["temperature"],
                "batch_size": bs,
            },
            reqB={  # Worker B does B->A review
                "kind": "review_B_on_A",
                "prompts": vprom_B_on_A,
                "max_len": CONFIG["max_length"]["reviewer"],
                "max_new": CONFIG["max_new_tokens"]["reviewer"],
                "do_sample": CONFIG["sampling"]["reviewer"]["do_sample"],
                "temperature": CONFIG["sampling"]["reviewer"]["temperature"],
                "batch_size": bs,
            }
        )

        parsed_BA = [parse_reviewer_json_robust(s) for s in raw_BA]
        parsed_AB = [parse_reviewer_json_robust(s) for s in raw_AB]

        df.loc[b.index, "raw_review_B_on_A"] = raw_BA
        df.loc[b.index, "review_score_B_on_A"] = [x["score"] for x in parsed_BA]
        df.loc[b.index, "review_feedback_B_on_A"] = [x["feedback"] for x in parsed_BA]

        df.loc[b.index, "raw_review_A_on_B"] = raw_AB
        df.loc[b.index, "review_score_A_on_B"] = [x["score"] for x in parsed_AB]
        df.loc[b.index, "review_feedback_A_on_B"] = [x["feedback"] for x in parsed_AB]

        gc.collect()

    stage1_cols = [
        "question", "options", "true_answer",
        "answer_A", "reasoning_A", "raw_reasoner_A",
        "answer_B", "reasoning_B", "raw_reasoner_B",
        "review_score_B_on_A", "review_feedback_B_on_A", "raw_review_B_on_A",
        "review_score_A_on_B", "review_feedback_A_on_B", "raw_review_A_on_B",
    ]
    df[stage1_cols].to_json(stage1_path, orient="records", lines=True, force_ascii=False)
    print(f"[Stage1 Saved] {stage1_path}")


def run_stage2_only(
    input_path: str,
    prompts: dict,
    tokJ, mdlJ, devJ,
):
    noise, level, seed_tag = parse_path_info(input_path)
    work_dir = RESULTS_ROOT / noise / level / seed_tag
    work_dir.mkdir(parents=True, exist_ok=True)

    stage1_path = work_dir / "stage1_results.jsonl"
    final_csv = work_dir / "final_predictions.csv"
    metrics_path = work_dir / "metrics.json"

    if metrics_path.exists() and final_csv.exists():
        print(f"[Stage2 Skip] done: {noise}/{level}/{seed_tag}")
        return
    if not stage1_path.exists():
        print(f"[Stage2 Skip] missing stage1: {noise}/{level}/{seed_tag}")
        return

    print(f"\n[Stage2 Run] {noise}/{level}/{seed_tag}")
    stage1_df = pd.read_json(stage1_path, lines=True)
    adjudicator_tmpl = prompts["adjudicator"]
    bs = int(CONFIG["batch_size"])

    adjudicator_prompts = []
    for _, r in stage1_df.iterrows():
        opts = r["options"] if isinstance(r["options"], dict) else {}
        adjudicator_prompts.append(
            safe_format_checked(
                adjudicator_tmpl,
                values={
                    "question": r["question"],
                    "options": opts,
                    "reasoning_model1": r["reasoning_A"],
                    "answer_model1": r["answer_A"],
                    "reasoning_model2": r["reasoning_B"],
                    "answer_model2": r["answer_B"],
                    "score_model2_on_model1": r["review_score_B_on_A"],
                    "feedback_model2_on_model1": r["review_feedback_B_on_A"],
                    "score_model1_on_model2": r["review_score_A_on_B"],
                    "feedback_model1_on_model2": r["review_feedback_A_on_B"],
                },
                must_have=[
                    "question", "options.A", "options.B", "options.C", "options.D",
                    "answer_model1", "answer_model2",
                    "score_model2_on_model1", "score_model1_on_model2",
                ],
                warn_prefix=f"[{noise}/{level}/{seed_tag}] ",
            )
        )

    final_answers = []
    for i in tqdm(range(0, len(adjudicator_prompts), bs), desc=f"Stage2 {noise}/{level}/{seed_tag}"):
        batch_prompts = adjudicator_prompts[i:i+bs]
        labels = forced_abcd_only(
            mdlJ, tokJ, batch_prompts,
            max_length=CONFIG["max_length"]["adjudicator"],
            first_device=devJ,
        )
        final_answers.extend(labels)

    out_df = stage1_df.copy()
    out_df["final_answer"] = final_answers

    ordered = [
        "question", "options", "true_answer",
        "answer_A", "reasoning_A",
        "answer_B", "reasoning_B",
        "review_score_B_on_A", "review_feedback_B_on_A",
        "review_score_A_on_B", "review_feedback_A_on_B",
        "final_answer",
    ]
    out_df[ordered].to_csv(final_csv, index=False, encoding="utf-8-sig")
    print(f"[Stage2 Saved] {final_csv}")

    parsed = int((out_df["final_answer"] != "?").sum())
    correct = int((out_df["final_answer"] == out_df["true_answer"]).sum()) if "true_answer" in out_df.columns else 0
    acc = (correct / parsed) if parsed > 0 else 0.0

    metrics = {
        "noise": noise,
        "level": level,
        "seed": seed_tag,
        "n_total": int(len(out_df)),
        "n_parsed": parsed,
        "n_correct": correct,
        "accuracy_parsed": acc,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": input_path,
        "output_dir": str(work_dir),
        "models": CONFIG["models"],
        "batch_size": int(CONFIG["batch_size"]),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Stage2 Saved] {metrics_path}")
    print(f"[Stats] parsed={parsed}/{len(out_df)} correct={correct} acc={acc:.4f}")


# =========================================================
# main
# =========================================================
def main():
    mp.freeze_support()
    assert torch.cuda.is_available(), "CUDA required"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])

    print(f"[BASE_DIR]      {BASE_DIR}")
    print(f"[RAW_DATA]      {RAW_DATA}")
    print(f"[DATA_DIR]      {DATA_DIR}")
    print(f"[RESULTS_ROOT]  {RESULTS_ROOT}")
    print(f"[PROMPT_DIR]    {PROMPT_DIR}")
    print(f"[STAGE]         {CONFIG['run']['stage']}")
    print(f"[BATCH_SIZE]    {CONFIG['batch_size']}")

    prompts = {
        "reasoner": read_text(PROMPT_DIR / "medical_reasoner.txt"),
        "reviewer": read_text(PROMPT_DIR / "critical_medical_reviewer.txt"),
        "adjudicator": read_text(PROMPT_DIR / "final_medical_adjudicator.txt"),
    }

    files = get_all_jsonl_files()
    print(f"[Files] {len(files)} jsonl found")

    stage = CONFIG["run"]["stage"]
    if stage not in {"1", "2", "all"}:
        raise ValueError('CONFIG["run"]["stage"] must be one of {"1","2","all"}')

    pool = None

    # -------------------------
    # Stage 1 (MP workers)
    # -------------------------
    if stage in {"1", "all"}:
        print("\n[Global] Starting MP workers (Reasoner A/B) ...")
        pool = MPGenPool()
        pool.start()

        for fp in files:
            run_stage1_only_mp(fp, prompts, pool)

        print("\n[Global] Stopping MP workers ...")
        pool.stop()
        pool = None
        gc.collect()

    # -------------------------
    # Stage 2 (single process)
    # -------------------------
    if stage in {"2", "all"}:
        print("\n[Global] Loading Adjudicator once ...")
        tokJ, mdlJ, devJ = load_adjudicator(CONFIG["models"]["adjudicator"], device_map="auto")

        for fp in files:
            run_stage2_only(fp, prompts, tokJ, mdlJ, devJ)

        print("\n[Global] Unloading Adjudicator ...")
        del mdlJ, tokJ
        torch.cuda.empty_cache()
        gc.collect()

    print("\n[All done]")


if __name__ == "__main__":
    main()
