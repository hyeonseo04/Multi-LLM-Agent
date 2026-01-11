#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MedQA jsonl의 question-field에 대해 PubMedBERT MLM 기반 word-level substitution 노이즈 생성.
- 문장별 목표 WER (= round(wer * n_words))에 맞춰 치환 시도 (retry 포함)
- UMLS 없이 lightweight: negation/숫자/단위/약어/의학접미사/하이픈/슬래시 토큰 보호
- PubMedBERT 사용 (기본 모델)
- A5000 기준: 기본적으로 GPU(cuda) 사용 (원하면 --cpu로 CPU 강제)

설치:
  pip install transformers torch tqdm

실행 예:
  python PubMedBERT.py \
    --input /home/hslee/multiagent/data/raw/medqa/medqa_all_clean.jsonl \
    --output /home/hslee/multiagent/data/processed/medqa/substitution_pubmedbert \
    --wer 0.1,0.2,0.3,0.4 \
    --seeds 1,2,3,4,5 \
    --top-k 50 \
    --retry-factor 2.5 \
    --batch-cap 128 \
    --max-len 256

출력:
  - output/wer_{r}/seed_{seed}.jsonl
  - output/wer_{r}/summary.json
"""

import argparse
import json
import math
import os
import random
import re
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM


# -----------------------------
# Word-ish tokenizer (fast & stable for WER control)
# -----------------------------
TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|[0-9]+(?:[/.-][0-9]+)*|[^\s]")

def simple_tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)

def simple_detokenize(tokens: List[str]) -> str:
    out = []
    for t in tokens:
        if not out:
            out.append(t)
        elif t in [".", ",", ":", ";", ")", "]", "}", "?", "!", "%"]:
            out[-1] += t
        elif out[-1] in ["(", "[", "{"]:
            out[-1] += t
        elif t in ["/", "-"] or out[-1] in ["/", "-"]:
            out[-1] += t
        else:
            out.append(" " + t)
    return "".join(out).strip()

def preserve_case(src: str, dst: str) -> str:
    if src.isupper():
        return dst.upper()
    if src.istitle():
        return dst.title()
    return dst


# -----------------------------
# Lightweight safety filters (no UMLS)
# -----------------------------
NEGATION = {"no", "not", "never", "none", "denies", "deny", "without", "neither", "nor"}
UNITS = {
    "mg","mcg","g","kg","ml","l","mmhg","bpm","meq","mmol","mol","iu","units","kpa","cm","mm","m","°c","°f"
}
MED_SUFFIXES = (
    "itis","emia","oma","osis","ectomy","algia","pathy","tomy","plasty","uria","rrhea","rrhagia","penia"
)

def looks_sensitive_or_medicalish(tok: str) -> bool:
    """
    UMLS 없이도 label shift 위험이 큰 토큰은 치환하지 않도록 보호.
    - 숫자/단위/비율/용량/생체징후
    - negation
    - 약어/대문자
    - 하이픈/슬래시 포함(의학 복합어/검사명/약물명일 확률)
    - 의학 접미사
    """
    t = tok.strip()
    if not t:
        return True
    low = t.lower()

    # 너무 짧은 토큰은 대체로 불안정
    if len(low) <= 2:
        return True

    # 숫자 포함(용량, 활력징후, 범위 등)
    if any(ch.isdigit() for ch in low):
        return True

    # 단위
    if low in UNITS:
        return True

    # 부정어
    if low in NEGATION:
        return True

    # 약어/대문자
    if t.isalpha() and any(ch.isupper() for ch in t):
        return True

    # 하이픈/슬래시(의학 복합어, 검사/약물 표기)
    if "-" in t or "/" in t:
        return True

    # 의학 접미사(질환/수술/상태)
    for s in MED_SUFFIXES:
        if low.endswith(s):
            return True

    return False


# -----------------------------
# MLM batch prediction: one forward per chunk
# -----------------------------
@torch.inference_mode()
def predict_replacements_batch(
    tokenizer,
    model,
    tokens: List[str],
    positions: List[int],
    top_k: int,
    max_len: int,
    device: str,
) -> Dict[int, List[str]]:
    """
    각 position마다 문장 하나씩 만들되,
    positions를 batch로 묶어서 한 번에 forward.
    반환: {pos: [cand1, cand2, ...]} (단일 토큰 후보만)
    """
    if not positions:
        return {}

    masked_texts = []
    for pos in positions:
        tmp = tokens[:]
        tmp[pos] = tokenizer.mask_token
        masked_texts.append(simple_detokenize(tmp))

    enc = tokenizer(
        masked_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    logits = model(**enc).logits  # [B, T, V]
    input_ids = enc["input_ids"]  # [B, T]
    mask_id = tokenizer.mask_token_id

    out: Dict[int, List[str]] = {}

    # 각 배치 샘플마다 [MASK] 위치 로짓에서 top-k 후보 추출
    for i, pos in enumerate(positions):
        mask_pos = (input_ids[i] == mask_id).nonzero(as_tuple=False).view(-1)
        if len(mask_pos) == 0:
            out[pos] = []
            continue
        mp = mask_pos[0].item()

        probs = torch.softmax(logits[i, mp], dim=-1)
        topk = torch.topk(probs, k=top_k)
        cand_ids = topk.indices.tolist()

        cands = []
        for tid in cand_ids:
            wp_tok = tokenizer.convert_ids_to_tokens(tid)

            # WordPiece 조각 제외(##...), 알파벳 단일 토큰만 허용
            if wp_tok.startswith("##"):
                continue
            if not re.fullmatch(r"[A-Za-z]+", wp_tok):
                continue

            w = tokenizer.convert_tokens_to_string([wp_tok]).strip()
            if w:
                cands.append(w)

        out[pos] = cands

    return out


# -----------------------------
# Injector
# -----------------------------
class PubMedBERTSubstitutionInjector:
    def __init__(
        self,
        model_name: str,
        seed: int,
        device: str,
        top_k: int,
        max_len: int,
        retry_factor: float,
        batch_cap: int,
        forbid_same: bool = True,
    ):
        self.rng = random.Random(seed)
        self.device = device
        self.top_k = top_k
        self.max_len = max_len
        self.retry_factor = retry_factor
        self.batch_cap = batch_cap
        self.forbid_same = forbid_same

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.mask_token is None:
            raise ValueError("Tokenizer has no [MASK] token. Choose an MLM checkpoint.")
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.model.eval()

    @staticmethod
    def _word_positions(tokens: List[str]) -> List[int]:
        # 알파벳 단어(하이픈/아포스트로피 포함)는 word로 카운트
        return [i for i, t in enumerate(tokens) if re.fullmatch(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", t)]

    def inject_exact_wer(self, text: str, wer_r: float) -> Tuple[str, Dict]:
        """
        문장별 목표 치환수 = round(wer_r * n_words)
        - 후보 부족/MLM 후보 부족으로 shortfall이 생길 수 있음 (meta로 기록)
        - retry_factor로 '더 많은 위치'를 시도해서 shortfall을 줄임
        """
        if not text or not text.strip():
            return text, {"n_words": 0, "target_subs": 0, "n_substituted": 0, "shortfall": 0}

        tokens = simple_tokenize(text)
        word_pos = self._word_positions(tokens)
        n_words = len(word_pos)
        if n_words == 0:
            return text, {"n_words": 0, "target_subs": 0, "n_substituted": 0, "shortfall": 0}

        target_subs = int(round(wer_r * n_words))
        if target_subs <= 0:
            return text, {"n_words": n_words, "target_subs": 0, "n_substituted": 0, "shortfall": 0}

        # 안전한 후보만 선택 (핵심 의학 토큰을 건드리지 않도록)
        safe_candidates = [p for p in word_pos if not looks_sensitive_or_medicalish(tokens[p])]
        self.rng.shuffle(safe_candidates)

        # 목표치보다 더 많이 시도 (실패 대비)
        max_attempts = min(
            len(safe_candidates),
            max(target_subs, int(math.ceil(target_subs * self.retry_factor)))
        )
        attempt_positions = safe_candidates[:max_attempts]

        substituted = 0
        details = []
        used_positions = set()

        # chunk 단위로 batch forward (GPU 메모리/속도 균형)
        idx = 0
        while substituted < target_subs and idx < len(attempt_positions):
            chunk = []
            while idx < len(attempt_positions) and len(chunk) < self.batch_cap:
                p = attempt_positions[idx]
                idx += 1
                if p not in used_positions:
                    chunk.append(p)
                    used_positions.add(p)

            if not chunk:
                continue

            cand_map = predict_replacements_batch(
                tokenizer=self.tokenizer,
                model=self.model,
                tokens=tokens,
                positions=chunk,
                top_k=self.top_k,
                max_len=self.max_len,
                device=self.device,
            )

            # 치환 적용
            for pos in chunk:
                if substituted >= target_subs:
                    break
                orig = tokens[pos]
                cands = cand_map.get(pos, [])
                # 후보 필터링
                filtered = []
                for c in cands:
                    if self.forbid_same and c.lower() == orig.lower():
                        continue
                    # 너무 짧은 토큰 배제(추가 안정장치)
                    if len(c) <= 2:
                        continue
                    filtered.append(c)

                if not filtered:
                    continue

                repl = preserve_case(orig, self.rng.choice(filtered))
                tokens[pos] = repl
                substituted += 1
                details.append({"orig": orig, "new": repl, "pos": pos})

        noisy = simple_detokenize(tokens)
        return noisy, {
            "n_words": n_words,
            "target_subs": target_subs,
            "n_candidates": len(safe_candidates),
            "n_attempted": len(attempt_positions),
            "n_substituted": substituted,
            "shortfall": max(0, target_subs - substituted),
            "details": details[:50],
        }


# -----------------------------
# Experiment runner (multi-wer x multi-seed like your previous script)
# -----------------------------
def run_experiment(args):
    # Runtime knobs
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(max(1, min(4, args.threads // 4)))

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        # A5000에선 fp16이 속도/메모리 둘 다 좋아서 기본 ON (불안정하면 --fp32)
        if not args.fp32:
            torch.set_default_dtype(torch.float16)  # 모델 로드 전에 설정
    print(f"[INFO] device={device}  model={args.model}")

    # Load model/tokenizer once per seed? (seed별 랜덤만 다르고 모델은 동일)
    # 하지만 fp16 dtype 설정/seed 고정을 깔끔하게 하려면 seed 루프 밖에서 로드하고,
    # injector 내부 RNG만 seed별로 다르게 만들어도 됨.
    # 여기선 간단히 seed마다 injector 생성(모델 재로딩 X)하도록 구성.
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.mask_token is None:
        raise ValueError("Tokenizer has no [MASK] token. Choose an MLM checkpoint.")
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    if device == "cuda" and not args.fp32:
        model.half()

    # Data load
    with open(args.input, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]
        if args.limit and args.limit > 0:
            all_data = all_data[:args.limit]

    out_root = Path(args.output)
    wer_list = [float(x.strip()) for x in args.wer.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    print(f"[INFO] total samples={len(all_data)}  wers={wer_list}  seeds={seeds}")

    for r in wer_list:
        r_dir = out_root / f"wer_{r}"
        r_dir.mkdir(parents=True, exist_ok=True)
        realized_rates = []

        for seed in seeds:
            # injector: reuse same tokenizer/model, seed만 다르게
            injector = PubMedBERTSubstitutionInjector.__new__(PubMedBERTSubstitutionInjector)
            injector.rng = random.Random(seed)
            injector.device = device
            injector.top_k = args.top_k
            injector.max_len = args.max_len
            injector.retry_factor = args.retry_factor
            injector.batch_cap = args.batch_cap
            injector.forbid_same = True
            injector.tokenizer = tokenizer
            injector.model = model

            results = []
            total_words = 0
            total_target = 0
            total_sub = 0
            total_shortfall = 0

            desc = f"[PubMedBERT] wer={r} seed={seed}"
            for obj in tqdm(all_data, desc=desc, unit="line", leave=False):
                q = str(obj.get(args.question_field, ""))
                noisy, meta = injector.inject_exact_wer(q, r)

                total_words += meta.get("n_words", 0)
                total_target += meta.get("target_subs", 0)
                total_sub += meta.get("n_substituted", 0)
                total_shortfall += meta.get("shortfall", 0)

                out_obj = dict(obj)
                out_obj[args.noisy_field] = noisy
                out_obj["noise_meta"] = {
                    "noise_type": "mlm_substitution",
                    "model": args.model,
                    "seed": seed,
                    "wer": r,
                    "stats": meta,
                }
                results.append(out_obj)

            # realized: 목표 대비 달성률(문장별 목표를 합산한 기준)
            realized = (total_sub / total_target) if total_target > 0 else 0.0
            realized_rates.append(realized)

            out_path = r_dir / f"seed_{seed}.jsonl"
            with out_path.open("w", encoding="utf-8") as wf:
                for res in results:
                    wf.write(json.dumps(res, ensure_ascii=False) + "\n")

            print(
                f"[DONE] wer={r} seed={seed} "
                f"target={total_target} substituted={total_sub} shortfall={total_shortfall} "
                f"realized={realized:.4f}"
            )

        summary = {
            "noise_type": "mlm_substitution",
            "model": args.model,
            "wer": r,
            "mean_realized": statistics.mean(realized_rates) if realized_rates else 0.0,
            "seeds": seeds,
            "n_samples": len(all_data),
            "notes": {
                "shortfall_reason": "MLM candidate 없음/필터로 인한 거절로 목표 치환수 미달 가능",
                "increase_realized": "top_k↑, retry_factor↑, batch_cap↑(GPU), 또는 필터 완화(비추천: negation/숫자/단위는 계속 보호)",
            },
        }
        with open(r_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    # PubMedBERT (MLM) 기본값
    parser.add_argument(
        "--model",
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    )

    parser.add_argument("--wer", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--seeds", default="1,2,3,4,5")

    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=256)

    # retry_factor: target_subs * retry_factor 만큼 위치를 더 시도(실패 대비)
    parser.add_argument("--retry-factor", type=float, default=2.5)

    # batch_cap: 한 번의 forward에 묶는 마스크 문장 수 (A5000이면 64~256까지 상황 따라)
    parser.add_argument("--batch-cap", type=int, default=64)

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--noisy-field", default="noisy_question")

    # runtime
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    parser.add_argument("--fp32", action="store_true", help="use fp32 on GPU (slower, more stable)")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (only matters with --cpu)")

    run_experiment(parser.parse_args())
