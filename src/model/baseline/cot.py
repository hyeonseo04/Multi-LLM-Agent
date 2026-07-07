import os
import json
import gc
import glob
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from json_repair import repair_json
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False
    print("[Warning] json_repair not installed. Run: pip install json-repair")
    print("[Warning] Falling back to regex-only parsing.")

# 프로젝트 루트 찾기
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG = {
    "raw_data": str(PROJECT_ROOT / "data/raw/medqa/medqa_all_clean.jsonl"),
    "processed_root": str(PROJECT_ROOT / "data/processed/medqa"),
    "results_root": str(PROJECT_ROOT / "results/cot/qwen_32"),
    "prompt_file": str(PROJECT_ROOT / "prompts/v1/cot.txt"),

    # 모델
    "model_id": "Qwen/Qwen2.5-32B-Instruct",
    "max_length": 1024,
    "batch_size": 16,

    # CoT 생성 설정
    "cot_max_new_tokens": 1024,

    # GPU
    "dtype": torch.bfloat16,
    "device_map": "auto",

    # 재시도
    "max_retries": 1,

    "clean_seeds": [42, 123, 456, 789, 1024],
    "default_seed": 42,

    # ========================================
    # 🎯 실행 모드 선택 (주석 처리/해제)
    # ========================================

    # "run_mode": "clean_only",

    # "run_mode": "wer_only",
    # "target_wer_levels": ["wer_0.4"],

    "run_mode": "clean_and_wer",
    "target_wer_levels": ["wer_0.1", "wer_0.4"],

    # "run_mode": "all",
}

_VALID = {"A", "B", "C", "D"}


# =========================================================
# 1) 파일 탐색 / 필터링
# =========================================================
def get_all_input_paths() -> List[str]:
    mode = CONFIG["run_mode"]
    paths = []

    print("\n" + "="*60)
    print(f"🎯 Run Mode: {mode.upper()}")
    print("="*60)

    if mode in ["clean_only", "clean_and_wer"]:
        paths.append(CONFIG["raw_data"])
        print("✓ Clean data included")

    if mode != "clean_only":
        pattern = os.path.join(CONFIG["processed_root"], "**/*.jsonl")
        found = glob.glob(pattern, recursive=True)
        target_levels = CONFIG.get("target_wer_levels", [])

        if mode == "all" or not target_levels:
            filtered = found
            print("✓ All WER levels included")
        else:
            filtered = [p for p in found if any(lv in p for lv in target_levels)]
            print(f"✓ Target WER levels: {target_levels}")

        paths.extend(sorted(filtered))

    print(f"✓ Total files: {len(paths)}")
    print("="*60 + "\n")
    return paths


def parse_experiment_id(input_path: str) -> Tuple[str, str, str]:
    p = Path(input_path).as_posix()
    if "/processed/medqa/" in p:
        tail = p.split("/processed/medqa/")[1]
        parts = tail.split("/")
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2].replace(".jsonl", "")
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
# 3) CoT 생성 + 파싱
# =========================================================
def parse_answer_from_cot(text: str) -> str:
    text = text.strip()

    # ── 1순위: json_repair (JSON 블록 추출 후 repair) ──
    if JSON_REPAIR_AVAILABLE:
        # "label" 키 시도
        json_match = re.search(r'\{[^}]*"label"[^}]*\}', text, re.IGNORECASE)
        if json_match:
            try:
                fixed = repair_json(json_match.group())
                data = json.loads(fixed)
                label = str(data.get("label", "")).upper()
                if label in _VALID:
                    return label
            except Exception:
                pass

        # "answer" 키 시도
        json_match = re.search(r'\{[^}]*"answer"[^}]*\}', text, re.IGNORECASE)
        if json_match:
            try:
                fixed = repair_json(json_match.group())
                data = json.loads(fixed)
                label = str(data.get("answer", "")).upper()
                if label in _VALID:
                    return label
            except Exception:
                pass

    # ── 2순위: 정규식 JSON 패턴 ──
    for pattern in [
        r'\{"label"\s*:\s*"([A-D])"\}',
        r'\{"answer"\s*:\s*"([A-D])"\}',
        r'"label"\s*:\s*"([A-D])"',
        r'"answer"\s*:\s*"([A-D])"',
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # ── 3순위: 명시적 마커 ──
    for pattern in [
        r'(?:final\s+)?answer\s*:\s*([A-D])\b',
        r'(?:the\s+)?(?:correct\s+)?answer\s+is\s+([A-D])\b',
        r'(?:i\s+)?(?:select|choose)\s+(?:option\s+)?([A-D])\b',
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # ── 4순위: 마지막 단독 A/B/C/D (fallback) ──
    matches = re.findall(r'\b([A-D])\b', text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    return "?"


def classify_cot(
    model,
    tokenizer,
    prompts: List[str],
    max_length: int,
    max_new_tokens: int,
) -> Tuple[List[str], List[str]]:
    chats = [apply_chat(tokenizer, p) for p in prompts]

    enc = tokenizer(
        chats,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = enc["input_ids"].size(1)
    preds, responses = [], []
    for i in range(out.size(0)):
        resp = tokenizer.decode(out[i, input_len:], skip_special_tokens=True)
        responses.append(resp)
        preds.append(parse_answer_from_cot(resp))

    return preds, responses


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

        summary = {
            "method": "cot",
            "noise": noise,
            "level": level,
            "num_seeds": int(row["count"]),
            "accuracy": {
                "mean": float(row["mean"]),
                "std": float(row["std"]) if pd.notna(row["std"]) else 0.0,
                "min": float(row["min"]),
                "max": float(row["max"]),
            },
            "seeds": dict(zip(seeds_data["seed"], seeds_data["accuracy"])),
            "generated_at": datetime.now().isoformat(),
        }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[SUMMARY] Updated summaries in {results_root}")


# =========================================================
# 5) 단일 실험 실행
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
    out_dir = results_root / noise / level / seed_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "metrics.json").exists():
        tqdm.write(f"[SKIP] {noise}/{level}/{seed_name}")
        return

    tqdm.write(f"\n[RUN] {noise} | {level} | {seed_name} (seed={actual_seed})")

    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)

    df = pd.read_json(p_path, lines=True)
    text_field = "noisy_question" if "noisy_question" in df.columns else "question"

    if "options" not in df.columns or "answer" not in df.columns:
        tqdm.write(f"[WARN] Missing columns, skip")
        return

    texts = [build_text_block(row[text_field], row["options"]) for _, row in df.iterrows()]
    prompts = [safe_format_prompt(prompt_templ, t) for t in texts]
    true_ans = df["answer"].astype(str).str.upper().str.strip().tolist()

    n = len(df)
    preds = ["?"] * n
    full_responses = [""] * n

    idxs = list(range(n))

    for attempt in range(CONFIG["max_retries"] + 1):
        if not idxs:
            break

        for s in tqdm(range(0, len(idxs), CONFIG["batch_size"]), desc=f"Attempt {attempt+1}", leave=False):
            b = idxs[s:s + CONFIG["batch_size"]]
            batch_prompts = [prompts[i] for i in b]

            try:
                batch_preds, batch_responses = classify_cot(
                    model, tokenizer, batch_prompts,
                    CONFIG["max_length"], CONFIG["cot_max_new_tokens"]
                )
                for k, i0 in enumerate(b):
                    if batch_preds[k] in _VALID:
                        preds[i0] = batch_preds[k]
                        full_responses[i0] = batch_responses[k]

            except torch.cuda.OutOfMemoryError:
                tqdm.write(f"[OOM] batch_size={len(b)}, falling back to sample-by-sample")
                torch.cuda.empty_cache()
                gc.collect()

                for i0 in b:
                    try:
                        p, r = classify_cot(
                            model, tokenizer, [prompts[i0]],
                            CONFIG["max_length"], CONFIG["cot_max_new_tokens"]
                        )
                        if p[0] in _VALID:
                            preds[i0] = p[0]
                            full_responses[i0] = r[0]
                    except Exception as e:
                        tqdm.write(f"[ERROR] Sample {i0} failed: {e}")

        idxs = [i for i in idxs if preds[i] == "?"]

    # 결과 저장
    out_df = pd.DataFrame({
        "pred": preds,
        "true": true_ans,
        "correct": [int(p == t) for p, t in zip(preds, true_ans)],
        "used_question": df[text_field].astype(str).tolist(),
        "reasoning": full_responses,
    })
    out_df.to_csv(out_dir / "predictions.csv", index=False)

    parsed = sum(p in _VALID for p in preds)
    acc = float(out_df["correct"].mean())
    tqdm.write(f"[Result] Accuracy: {acc:.4f} ({int(acc*n)}/{n}) | Parsed: {parsed}/{n}")

    metrics = {
        "method": "cot",
        "noise": noise,
        "level": level,
        "seed": seed_name,
        "actual_random_seed": actual_seed,
        "accuracy": acc,
        "num_samples": n,
        "parsed": parsed,
        "timestamp": datetime.now().isoformat(),
        "model_id": CONFIG["model_id"],
        "prompt_file": CONFIG["prompt_file"],
        "batch_size": CONFIG["batch_size"],
        "max_length": CONFIG["max_length"],
        "cot_max_new_tokens": CONFIG["cot_max_new_tokens"],
        "decode_method": "free_generation",
        "json_repair": JSON_REPAIR_AVAILABLE,
        "input_path": str(p_path),
        "text_field_used": text_field,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    del df, out_df, texts, prompts
    gc.collect()
    torch.cuda.empty_cache()


# =========================================================
# 6) Main
# =========================================================
def main():
    assert torch.cuda.is_available(), "CUDA GPU not detected"
    n_gpus = torch.cuda.device_count()

    print("\n" + "="*60)
    print("⚡ CoT Experiments")
    print("="*60)
    print(f"GPUs: {n_gpus} | Model: {CONFIG['model_id']}")
    print(f"Batch size: {CONFIG['batch_size']} | CoT tokens: {CONFIG['cot_max_new_tokens']}")
    print(f"json_repair: {'✅ enabled' if JSON_REPAIR_AVAILABLE else '❌ not installed'}")
    print("="*60)

    tok = AutoTokenizer.from_pretrained(CONFIG["model_id"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_id"],
        device_map=CONFIG["device_map"],
        torch_dtype=CONFIG["dtype"],
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()

    prompt_templ = load_prompt_template(CONFIG["prompt_file"])
    input_paths = get_all_input_paths()
    results_root = Path(CONFIG["results_root"])
    results_root.mkdir(parents=True, exist_ok=True)

    for p_path in tqdm(input_paths, desc="CoT Experiments"):
        noise, level, seed_orig = parse_experiment_id(p_path)

        if noise == "clean":
            seed_list = [(f"seed_{i}", s) for i, s in enumerate(CONFIG["clean_seeds"], start=1)]
        else:
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

    generate_all_summaries(str(results_root))

    print("\n" + "="*60)
    print("✅ All CoT experiments completed!")
    print("="*60)


if __name__ == "__main__":
    main()