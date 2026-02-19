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

# =========================================================
# ✅ CONFIG: 모든 최적화 + 자동 batch_size 조정
# =========================================================
CONFIG = {
    # Clean 데이터
    "raw_data": "/home/hslee/multiagent/data/raw/medqa/medqa_all_clean.jsonl",
    
    # 노이즈 데이터 루트
    "processed_root": "/home/hslee/multiagent/data/processed/medqa",
    
    # 결과 저장 루트
    "results_root": "/home/hslee/multiagent/results/cot/medqa",
    
    # 프롬프트 파일
    "prompt_file": "/home/hslee/multiagent/prompts/v1/cot.txt",
    
    # 모델
    "model_id": "openai/gpt-oss-20b",
    "max_length": 1024,
    
    # CoT 생성 설정
    "cot_max_new_tokens": 512,
    
    # GPU
    "gpu_mem_per_card": "44GiB",
    "dtype": torch.bfloat16,
    "device_map": "auto",
    
    # 재시도
    "max_retries": 1,
    
    # ✅ Clean 데이터용 5개 random seed
    "clean_seeds": [42, 123, 456, 789, 1024],
    # 노이즈 데이터용 기본 seed
    "default_seed": 42,
    
    # ========================================
    # ⚡ 성능 최적화
    # ========================================
    "use_torch_compile": True,
    "torch_compile_mode": "reduce-overhead",  # "default", "reduce-overhead", "max-autotune"
    
    "use_flash_attention": False,
    
    # ========================================
    # 🔄 자동 Batch Size 조정
    # ========================================
    "initial_batch_size": 12,  # 시작값 (크게!)
    "min_batch_size": 1,  # 최소값
    "auto_reduce_batch_size": True,  # 자동 조정 활성화
    "oom_threshold": 3,  # 연속 OOM 이 횟수 넘으면 영구 감소
    
    # ========================================
    # 🎯 실행 모드 선택 (주석 처리/해제)
    # ========================================
    
    # ─── 모드 1: Clean만 (1-1.5시간) ───
    #"run_mode": "clean_only",
    # "target_wer_levels": [],
    
    # ─── 모드 2: wer_0.2, wer_0.4만 (10-14시간) ───
    "run_mode": "wer_only",
    "target_wer_levels": ["wer_0.4"],
    
    # ─── 모드 3: Clean + wer_0.2, wer_0.4 (11-16시간) ───
    # "run_mode": "clean_and_wer",
    # "target_wer_levels": ["wer_0.2", "wer_0.4"],
    
    # ─── 모드 4: 전체 (매우 오래 걸림) ───
    # "run_mode": "all",
    # "target_wer_levels": [],
}

_VALID = {"A", "B", "C", "D"}


# =========================================================
# 1) 파일 탐색 / 필터링
# =========================================================
def get_all_input_paths() -> List[str]:
    """run_mode에 따라 파일 필터링"""
    mode = CONFIG["run_mode"]
    paths = []
    
    print("\n" + "="*60)
    print(f"🎯 Run Mode: {mode.upper()}")
    print("="*60)
    
    if mode in ["clean_only", "clean_and_wer"]:
        paths.append(CONFIG["raw_data"])
        print(f"✓ Clean data included")
    
    if mode != "clean_only":
        pattern = os.path.join(CONFIG["processed_root"], "**/*.jsonl")
        found = glob.glob(pattern, recursive=True)
        
        target_levels = CONFIG.get("target_wer_levels", [])
        
        if mode == "all" or not target_levels:
            filtered = found
            print(f"✓ All WER levels included")
        else:
            filtered = []
            for p in found:
                if any(level in p for level in target_levels):
                    filtered.append(p)
            print(f"✓ Target WER levels: {target_levels}")
        
        paths.extend(sorted(filtered))
    
    print(f"✓ Total files: {len(paths)}")
    print("="*60 + "\n")
    
    return paths


def parse_experiment_id(input_path: str) -> Tuple[str, str, str]:
    """실험 ID 파싱"""
    p = Path(input_path).as_posix()
    if "/processed/medqa/" in p:
        tail = p.split("/processed/medqa/")[1]
        parts = tail.split("/")
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
# 3) CoT: 자유 생성 후 답변 파싱 (OOM 핸들링 포함)
# =========================================================
def classify_cot_generation(
    model, 
    tokenizer, 
    prompts: List[str], 
    max_length: int, 
    max_new_tokens: int
) -> Tuple[List[str], List[str]]:
    """CoT 방식: 자유 생성 + 파싱"""
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
    preds = []
    full_responses = []
    
    for i in range(out.size(0)):
        gen = out[i, input_len:]
        response = tokenizer.decode(gen, skip_special_tokens=True)
        full_responses.append(response)
        parsed = parse_answer_from_cot(response)
        preds.append(parsed)

    return preds, full_responses


def classify_cot_with_auto_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_length: int,
    max_new_tokens: int,
    initial_batch_size: int = None
) -> Tuple[List[str], List[str]]:
    """
    OOM 발생시 자동으로 batch_size 줄여서 재시도
    """
    batch_size = initial_batch_size or len(prompts)
    batch_size = min(batch_size, len(prompts))
    
    while batch_size >= 1:
        try:
            # 작은 배치로 나누어 처리
            all_preds = []
            all_responses = []
            
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                batch_preds, batch_resps = classify_cot_generation(
                    model, tokenizer, batch_prompts, max_length, max_new_tokens
                )
                all_preds.extend(batch_preds)
                all_responses.extend(batch_resps)
            
            return all_preds, all_responses
            
        except torch.cuda.OutOfMemoryError:
            old_size = batch_size
            batch_size = max(1, batch_size // 2)
            
            print(f"\n[OOM] Batch size {old_size} → {batch_size}, retrying...")
            
            # 메모리 정리
            torch.cuda.empty_cache()
            gc.collect()
            
            if batch_size < 1:
                raise RuntimeError("OOM even with batch_size=1")
    
    raise RuntimeError("Failed to process batch")


def parse_answer_from_cot(text: str) -> str:
    """CoT 텍스트에서 답변 추출"""
    text = text.strip()
    
    # 1. JSON 파싱
    json_patterns = [
        r'\{"label"\s*:\s*"([A-D])"\}',
        r'\{"answer"\s*:\s*"([A-D])"\}',
        r'"label"\s*:\s*"([A-D])"',
        r'"answer"\s*:\s*"([A-D])"',
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 2. 명시적 마커
    answer_patterns = [
        r'(?:final\s+)?answer\s*:\s*([A-D])\b',
        r'(?:the\s+)?(?:correct\s+)?answer\s+is\s+([A-D])\b',
        r'(?:i\s+)?(?:select|choose)\s+(?:option\s+)?([A-D])\b',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 3. 마지막 A/B/C/D
    matches = re.findall(r'\b([A-D])\b', text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    
    return "?"


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
    current_global_batch_size: int,
) -> int:
    """
    단일 실험 실행 (clean이든 noisy든 동일한 로직)
    Returns: 업데이트된 global batch_size
    """
    out_dir = results_root / noise / level / seed_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    if (out_dir / "metrics.json").exists():
        tqdm.write(f"[SKIP] {noise}/{level}/{seed_name}")
        return current_global_batch_size

    tqdm.write(f"\n[RUN] {noise} | {level} | {seed_name} (random_seed={actual_seed})")
    tqdm.write(f"[Batch] Starting with size: {current_global_batch_size}")

    # 재현성 설정
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)

    df = pd.read_json(p_path, lines=True)
    text_field = "noisy_question" if "noisy_question" in df.columns else "question"
    
    if "options" not in df.columns or "answer" not in df.columns:
        tqdm.write(f"[WARN] Missing columns, skip")
        return current_global_batch_size

    texts = [build_text_block(row[text_field], row["options"]) for _, row in df.iterrows()]
    prompts = [safe_format_prompt(prompt_templ, t) for t in texts]
    true_ans = df["answer"].astype(str).str.upper().str.strip().tolist()
    
    n = len(df)
    preds = ["?"] * n
    full_responses = [""] * n
    
    idxs = list(range(n))
    
    # OOM 카운터
    oom_consecutive_count = 0
    
    for attempt in range(CONFIG["max_retries"] + 1):
        if not idxs:
            break
        
        # 현재 시도의 batch_size
        attempt_batch_size = current_global_batch_size
        
        s = 0
        while s < len(idxs):
            # 남은 샘플 수
            remaining = len(idxs) - s
            current_batch_size = min(attempt_batch_size, remaining)
            
            b = idxs[s:s + current_batch_size]
            batch_prompts = [prompts[i] for i in b]
            
            try:
                # 배치 처리 (자동 축소 포함)
                if CONFIG["auto_reduce_batch_size"]:
                    batch_preds, batch_responses = classify_cot_with_auto_batch(
                        model, tokenizer, batch_prompts,
                        CONFIG["max_length"], CONFIG["cot_max_new_tokens"],
                        initial_batch_size=current_batch_size
                    )
                else:
                    batch_preds, batch_responses = classify_cot_generation(
                        model, tokenizer, batch_prompts,
                        CONFIG["max_length"], CONFIG["cot_max_new_tokens"]
                    )
                
                # 성공
                for k, i0 in enumerate(b):
                    if batch_preds[k] in _VALID:
                        preds[i0] = batch_preds[k]
                        full_responses[i0] = batch_responses[k]
                
                s += current_batch_size
                oom_consecutive_count = 0  # 리셋
                
            except torch.cuda.OutOfMemoryError:
                oom_consecutive_count += 1
                
                tqdm.write(f"[OOM] Batch size {current_batch_size} failed")
                tqdm.write(f"[OOM] Consecutive count: {oom_consecutive_count}")
                
                # 연속 OOM → 전역 감소
                if oom_consecutive_count >= CONFIG["oom_threshold"]:
                    old_global = current_global_batch_size
                    current_global_batch_size = max(CONFIG["min_batch_size"], current_global_batch_size // 2)
                    attempt_batch_size = current_global_batch_size
                    
                    tqdm.write(f"[OOM] Global batch_size: {old_global} → {current_global_batch_size}")
                    oom_consecutive_count = 0
                
                # 현재 배치만 축소해서 재시도
                retry_size = max(CONFIG["min_batch_size"], current_batch_size // 2)
                tqdm.write(f"[Retry] With batch_size: {retry_size}")
                
                torch.cuda.empty_cache()
                gc.collect()
                
                # 재시도
                try:
                    batch_preds, batch_responses = classify_cot_with_auto_batch(
                        model, tokenizer, batch_prompts,
                        CONFIG["max_length"], CONFIG["cot_max_new_tokens"],
                        initial_batch_size=retry_size
                    )
                    
                    for k, i0 in enumerate(b):
                        if batch_preds[k] in _VALID:
                            preds[i0] = batch_preds[k]
                            full_responses[i0] = batch_responses[k]
                    
                    s += current_batch_size
                    
                except Exception as e:
                    tqdm.write(f"[ERROR] Retry failed: {e}")
                    s += current_batch_size  # 건너뛰기
        
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
    
    tqdm.write(f"[Result] Accuracy: {acc:.4f} ({int(acc*n)}/{n})")
    tqdm.write(f"[Result] Parsed: {parsed}/{n}")
    
    # Metrics 저장
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
        "initial_batch_size": CONFIG["initial_batch_size"],
        "final_batch_size": current_global_batch_size,
        "max_length": CONFIG["max_length"],
        "cot_max_new_tokens": CONFIG["cot_max_new_tokens"],
        "decode_method": "free_generation",
        "optimizations": {
            "flash_attention": CONFIG.get("use_flash_attention", False),
            "torch_compile": CONFIG.get("use_torch_compile", False),
            "auto_batch_reduce": CONFIG.get("auto_reduce_batch_size", False),
        },
        "input_path": str(p_path),
        "text_field_used": text_field,
    }
    
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Cleanup
    del df, out_df, texts, prompts
    gc.collect()
    torch.cuda.empty_cache()
    
    return current_global_batch_size


# =========================================================
# 6) Main
# =========================================================
def main():
    assert torch.cuda.is_available(), "CUDA GPU not detected"
    n_gpus = torch.cuda.device_count()
    
    print("\n" + "="*60)
    print("⚡ CoT Experiments with Full Optimization")
    print("="*60)
    print(f"GPUs: {n_gpus} | Model: {CONFIG['model_id']}")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(CONFIG["model_id"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # 모델 로드 설정
    model_kwargs = {
        "device_map": CONFIG["device_map"],
        "max_memory": {i: CONFIG["gpu_mem_per_card"] for i in range(max(n_gpus, 1))},
        "dtype": CONFIG["dtype"],
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    
    # ⚡ FlashAttention-2
    if CONFIG.get("use_flash_attention", False):
        print("[Optimization] Attempting FlashAttention-2...")
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("[Optimization] ✅ FlashAttention-2 enabled")
        except ImportError:
            print("[Warning] flash-attn not installed")
            print("[Warning] Install: pip install flash-attn --no-build-isolation")
            print("[Warning] Continuing without FlashAttention...")
    else:
        print("[Optimization] FlashAttention disabled")

    # 모델 로드
    print("[Loading] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_id"],
        **model_kwargs
    ).eval()
    print("[Loading] ✅ Model loaded")

    # ⚡ torch.compile
    if CONFIG.get("use_torch_compile", False):
        mode = CONFIG["torch_compile_mode"]
        print(f"[Optimization] Applying torch.compile (mode: {mode})...")
        print("[Optimization] First batch will be slower (compilation)")
        try:
            model = torch.compile(model, mode=mode)
            print("[Optimization] ✅ torch.compile enabled")
        except Exception as e:
            print(f"[Warning] torch.compile failed: {e}")
            print("[Warning] Continuing without compilation...")
    else:
        print("[Optimization] torch.compile disabled")

    # Batch Size 설정
    initial_batch_size = CONFIG["initial_batch_size"]
    min_batch_size = CONFIG["min_batch_size"]
    auto_reduce = CONFIG["auto_reduce_batch_size"]
    
    print(f"\n[Batch Size] Initial: {initial_batch_size}")
    print(f"[Batch Size] Minimum: {min_batch_size}")
    print(f"[Batch Size] Auto-reduce: {auto_reduce}")
    print("="*60 + "\n")

    prompt_templ = load_prompt_template(CONFIG["prompt_file"])
    input_paths = get_all_input_paths()
    
    results_root = Path(CONFIG["results_root"])
    results_root.mkdir(parents=True, exist_ok=True)

    # 전역 batch_size (적응형)
    current_global_batch_size = initial_batch_size

    for p_path in tqdm(input_paths, desc="🚀 CoT Experiments"):
        noise, level, seed_orig = parse_experiment_id(p_path)
        
        # ✅ Clean 데이터면 5개 seed로 반복 실행
        if noise == "clean":
            seed_list = [(f"seed_{i}", s) for i, s in enumerate(CONFIG["clean_seeds"])]
        else:
            # 노이즈 데이터는 원래대로 1번만
            seed_list = [(seed_orig, CONFIG["default_seed"])]
        
        for seed_name, actual_seed in seed_list:
            current_global_batch_size = run_single_experiment(
                model=model,
                tokenizer=tok,
                prompt_templ=prompt_templ,
                p_path=p_path,
                noise=noise,
                level=level,
                seed_name=seed_name,
                actual_seed=actual_seed,
                results_root=results_root,
                current_global_batch_size=current_global_batch_size,
            )

    # Summaries
    generate_all_summaries(str(results_root))
    
    print("\n" + "="*60)
    print("✅ All CoT experiments completed!")
    print(f"Final global batch_size: {current_global_batch_size}")
    print("="*60)


if __name__ == "__main__":
    main()