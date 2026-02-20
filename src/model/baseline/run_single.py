import os
import json
import gc
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
)

# =========================================================
# ✅ CONFIG: 전체 실험(클린 + 노이즈 전부)
# =========================================================
CONFIG = {
    # Clean 데이터
    "raw_data": "/home/hslee/multiagent/data/raw/medqa/medqa_all_clean.jsonl",
    # 노이즈 데이터 루트(여기 아래 모든 *.jsonl 자동 탐색)
    "processed_root": "/home/hslee/multiagent/data/processed/medqa",

    # 결과 저장 루트
    "results_root": "/home/hslee/multiagent/results/baseline/medqa",

    # 프롬프트 파일 (label-only JSON 버전)
    "prompt_file": "/home/hslee/multiagent/prompts/v1/baseline.txt",

    # 모델
    "model_id": "openai/gpt-oss-20b",
    "batch_size": 8,
    "max_length": 1024,

    # GPU
    "gpu_mem_per_card": "44GiB",
    "dtype": torch.bfloat16,
    "device_map": "auto",

    # 재시도(원하면 0으로 둬도 됨)
    "max_retries": 1,

    # ✅ Clean 데이터용 5개 random seed
    "clean_seeds": [42, 123, 456, 789, 1024],
    # 노이즈 데이터용 기본 seed
    "default_seed": 42,
}

_VALID = {"A", "B", "C", "D"}


# =========================================================
# 1) 파일 탐색 / 실험 ID 파싱
# =========================================================
def get_all_input_paths() -> List[str]:
    """
    clean 1개 + processed_root 하위 모든 jsonl 파일 탐색
    """
    paths = [CONFIG["raw_data"]]
    pattern = os.path.join(CONFIG["processed_root"], "**/*.jsonl")
    found = glob.glob(pattern, recursive=True)
    paths.extend(found)
    # 중복 제거 + 정렬
    return sorted(set(paths))

def parse_experiment_id(input_path: str) -> Tuple[str, str, str]:
    """
    /processed/medqa/{noise}/{level}/{seed}.jsonl 형태에서
    noise, level, seed 추출
    clean은 (clean, clean, seed_0)로 통일
    """
    p = Path(input_path).as_posix()
    if "/processed/medqa/" in p:
        tail = p.split("/processed/medqa/")[1]
        parts = tail.split("/")
        # 기대: noise / level / seed.jsonl
        if len(parts) >= 3:
            noise = parts[0]
            level = parts[1]
            seed = parts[2].replace(".jsonl", "")
            return noise, level, seed
    return "clean", "clean", "seed_0"


# =========================================================
# 2) 프롬프트 유틸
# =========================================================
def load_prompt_template(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def safe_format_prompt(template: str, text: str) -> str:
    token = "___TEXT___"
    t = template.replace("{text}", token).replace("{", "{{").replace("}", "}}").replace(token, "{text}")
    return t.format(text=str(text).replace("{", "{{").replace("}", "}}"))

def build_text_block(q: str, opts: Dict[str, str]) -> str:
    q = str(q).strip()
    return (
        f"{q}\n\n"
        f"Options:\n"
        f"A) {opts.get('A','')}\n"
        f"B) {opts.get('B','')}\n"
        f"C) {opts.get('C','')}\n"
        f"D) {opts.get('D','')}"
    )

def apply_chat(tokenizer, user_text: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        return user_text


# =========================================================
# 3) 강제 1토큰(A/B/C/D) 디코딩
# =========================================================
class AllowOnly(LogitsProcessor):
    def __init__(self, ids: List[int]):
        self.ids = set(ids)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        for i in self.ids:
            mask[:, i] = 0.0
        return scores + mask

def _single_token_ids(tokenizer, letter: str) -> List[int]:
    out = set()
    # 네가 성공한 코드의 prefix 그대로 유지
    for pref in ["", " ", "▁", "Ġ"]:
        t = tokenizer.encode(pref + letter, add_special_tokens=False)
        if len(t) == 1:
            out.add(t[0])
    return sorted(out)

def build_allowed_ids(tokenizer) -> List[int]:
    ids = []
    for L in ["A", "B", "C", "D"]:
        ids += _single_token_ids(tokenizer, L)
    return sorted(set(ids))

def classify_forced_one_token(model, tokenizer, prompts: List[str], max_length: int) -> List[str]:
    """
    성공한 방식 그대로:
    chat_template + '{"label":"' 붙이기 + A/B/C/D만 허용 + 1토큰 생성
    """
    chats = [apply_chat(tokenizer, p) + '{"label":"' for p in prompts]

    enc = tokenizer(
        chats,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    processors = LogitsProcessorList([AllowOnly(build_allowed_ids(tokenizer))])

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=1,
            min_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=processors,
        )

    input_len = enc["input_ids"].size(1)
    preds = []
    for i in range(out.size(0)):
        gen = out[i, input_len:]
        txt = tokenizer.decode(gen, skip_special_tokens=True)
        L = (txt.strip()[:1] or "?").upper()
        preds.append(L if L in _VALID else "?")
    return preds


# =========================================================
# 4) 요약(summary.json) 생성
# =========================================================
def generate_all_summaries(results_root: str):
    results_root = Path(results_root)
    all_data = []

    for mf in results_root.glob("*/*/seed_*/metrics.json"):
        try:
            all_data.append(json.loads(mf.read_text(encoding="utf-8")))
        except Exception:
            continue

    if not all_data:
        print("[SUMMARY] No metrics.json found.")
        return

    df_res = pd.DataFrame(all_data)
    if df_res.empty:
        print("[SUMMARY] Empty metrics dataframe.")
        return

    summary_df = (
        df_res.groupby(["noise", "level"])["accuracy"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    for _, row in summary_df.iterrows():
        noise, level = row["noise"], row["level"]
        save_path = results_root / noise / level / "summary.json"

        seeds_data = df_res[(df_res["noise"] == noise) & (df_res["level"] == level)]
        seed_acc = dict(zip(seeds_data["seed"], seeds_data["accuracy"]))

        summary = {
            "noise": noise,
            "level": level,
            "num_seeds": int(row["count"]),
            "accuracy": {
                "mean": float(row["mean"]),
                "std": float(row["std"]) if pd.notna(row["std"]) else 0.0,
                "min": float(row["min"]),
                "max": float(row["max"]),
            },
            "seeds": seed_acc,
            "generated_at": datetime.now().isoformat(),
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[SUMMARY] Updated summaries in {results_root}")


# =========================================================
# 5) 단일 실험 실행 함수
# =========================================================
def run_single_experiment(
    model,
    tokenizer,
    prompt_templ: str,
    p_path: str,
    noise: str,
    level: str,
    seed_name: str,
    actual_seed: int,
    results_root: Path,
):
    """
    단일 실험 실행 (clean이든 noisy든 동일한 로직)
    """
    out_dir = results_root / noise / level / seed_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # skip if already done
    if (out_dir / "metrics.json").exists():
        tqdm.write(f"[SKIP] Exists: {noise}/{level}/{seed_name}")
        return

    tqdm.write(f"[RUN] {noise} | {level} | {seed_name} (random_seed={actual_seed})")

    # 재현성 설정
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)

    df = pd.read_json(p_path, lines=True)

    # 텍스트 필드 결정
    text_field = "noisy_question" if "noisy_question" in df.columns else "question"
    if "options" not in df.columns or "answer" not in df.columns:
        tqdm.write(f"[WARN] Missing columns in {p_path}, skip")
        return

    texts = [build_text_block(row[text_field], row["options"]) for _, row in df.iterrows()]
    prompts = [safe_format_prompt(prompt_templ, t) for t in texts]
    true_ans = df["answer"].astype(str).str.upper().str.strip().tolist()

    n = len(df)
    preds = ["?"] * n

    # retries 구조는 유지
    idxs = list(range(n))
    for attempt in range(CONFIG["max_retries"] + 1):
        if not idxs:
            break
        for s in tqdm(range(0, len(idxs), CONFIG["batch_size"]), desc=f"Attempt {attempt+1}", leave=False):
            b = idxs[s:s + CONFIG["batch_size"]]
            batch_prompts = [prompts[i] for i in b]
            batch_preds = classify_forced_one_token(model, tokenizer, batch_prompts, CONFIG["max_length"])
            for k, i0 in enumerate(b):
                if batch_preds[k] in _VALID:
                    preds[i0] = batch_preds[k]
        idxs = [i for i in idxs if preds[i] == "?"]

    out_df = pd.DataFrame({
        "pred": preds,
        "true": true_ans,
        "correct": [int(p == t) for p, t in zip(preds, true_ans)],
        "used_question": df[text_field].astype(str).tolist(),
    })
    out_df.to_csv(out_dir / "predictions.csv", index=False)

    parsed = sum(p in _VALID for p in preds)
    acc = float(out_df["correct"].mean())

    metrics = {
        "noise": noise,
        "level": level,
        "seed": seed_name,
        "actual_random_seed": actual_seed,
        "accuracy": acc,
        "num_samples": int(n),
        "parsed": int(parsed),
        "timestamp": datetime.now().isoformat(),
        "model_id": CONFIG["model_id"],
        "prompt_file": CONFIG["prompt_file"],
        "batch_size": CONFIG["batch_size"],
        "max_length": CONFIG["max_length"],
        "decode": "forced_generate_1_token_allowed_{A,B,C,D}",
        "input_path": str(p_path),
        "text_field_used": text_field,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # cleanup
    del df, out_df, texts, prompts
    gc.collect()
    torch.cuda.empty_cache()


# =========================================================
# 6) Main
# =========================================================
def main():
    assert torch.cuda.is_available(), "CUDA GPU not detected"
    n_gpus = torch.cuda.device_count()
    print(f"[Info] GPUs: {n_gpus} | model: {CONFIG['model_id']}")

    # tokenizer/model (seed 적용 전에 로드)
    tok = AutoTokenizer.from_pretrained(CONFIG["model_id"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_id"],
        device_map=CONFIG["device_map"],
        max_memory={i: CONFIG["gpu_mem_per_card"] for i in range(max(n_gpus, 1))},
        dtype=CONFIG["dtype"],
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()

    prompt_templ = load_prompt_template(CONFIG["prompt_file"])
    input_paths = get_all_input_paths()

    print(f"[Info] Total input files: {len(input_paths)}")
    results_root = Path(CONFIG["results_root"])
    results_root.mkdir(parents=True, exist_ok=True)

    for p_path in tqdm(input_paths, desc="🚀 Overall Experiments"):
        noise, level, seed_orig = parse_experiment_id(p_path)
        
        # ✅ Clean 데이터면 5개 seed로 반복 실행
        if noise == "clean":
            seed_list = [(f"seed_{i}", s) for i, s in enumerate(CONFIG["clean_seeds"], start=1)]  # ← start=1 추가!
        else:
            # 노이즈 데이터는 원래대로 1번만 (파일명의 seed 사용)
            seed_list = [(seed_orig, CONFIG["default_seed"])]
        
        for seed_name, actual_seed in seed_list:
            run_single_experiment(
                model=model,
                tokenizer=tok,
                prompt_templ=prompt_templ,
                p_path=p_path,
                noise=noise,
                level=level,
                seed_name=seed_name,
                actual_seed=actual_seed,
                results_root=results_root,
            )

    # summaries
    generate_all_summaries(str(results_root))
    print("\n✅ All experiments completed.")


if __name__ == "__main__":
    main()