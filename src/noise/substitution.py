#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLM-based Contextual Word Substitution for Clinical Text Noise Injection

Injects word-level noise into medical QA text by masking randomly selected
words and replacing them with the top-1 prediction from PubMedBERT MLM.

Method:
  1. Randomly select round(n_words × WER) words
  2. For each word: mask → PubMedBERT predicts → replace with top-1
  3. Protected tokens (negation, numbers, units) are skipped

Based on:
  - BERT-Attack (Li et al., EMNLP 2020): MLM-based word substitution
  - Moradi et al. (2021): Clinical text perturbation at controlled WER
  - Kobayashi (NAACL 2018): Contextual augmentation via MLM

Usage:
    python mlm_noise.py \
        --input  data/medqa_all_clean.jsonl \
        --output data/processed/medqa/mlm_noise \
        --wer 0.1,0.2,0.3,0.4 \
        --seeds 1,2,3,4,5

    # Quick test
    python mlm_noise.py \
        --input data/medqa_all_clean.jsonl \
        --output data/test --wer 0.1 --seeds 1 --limit 10

Environment:
    export HF_HUB_HTTP_TIMEOUT=60
    export HF_HUB_ETAG_TIMEOUT=60
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

from nltk.tokenize import word_tokenize
import nltk

try:
    word_tokenize("test")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


# ═══════════════════════════════════════════════════════════════
# Protection Filters (Moradi et al., 2021)
# ═══════════════════════════════════════════════════════════════

NEGATION = {
    "no", "not", "never", "none", "nothing", "nobody", "nowhere",
    "neither", "nor", "without", "lack", "absent", "negative",
    "deny", "denies", "denied", "denying",
    "cannot", "can't", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "shouldn't", "couldn't",
    "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't",
}

UNITS = {
    "mg", "g", "kg", "ml", "l", "dl", "mcg", "μg",
    "mm", "cm", "m", "mmhg", "bpm", "mmol", "meq",
    "mg/dl", "g/dl", "ml/min", "iu",
    "%", "percent", "degree", "degrees", "db", "hz",
}

MED_SUFFIXES = (
    "itis", "oma", "emia", "osis", "pathy",
    "ectomy", "otomy", "ostomy", "algia",
    "plasty", "rrhea", "scopy", "gram",
    "lysis", "penia", "trophy", "plasia",
)


def is_protected(word: str) -> bool:
    """Protect negation, numbers, units, and medical terms."""
    w = word.lower().strip()
    if w in NEGATION or w in UNITS:
        return True
    if any(c.isdigit() for c in word):
        return True
    if len(w) < 3:
        return True
    for sfx in MED_SUFFIXES:
        if w.endswith(sfx) and len(w) > len(sfx) + 1:
            return True
    return False


# ═══════════════════════════════════════════════════════════════
# MLM Substitutor
# ═══════════════════════════════════════════════════════════════

class MLMNoiseInjector:
    """
    PubMedBERT MLM-based word substitutor.

    For each target word:
      1. Replace with [MASK] in the original context
      2. Get top-K MLM predictions
      3. Pick first valid candidate (alphabetic, ≠ original)

    No POS filter, no similarity threshold — MLM probability alone
    determines the replacement. This maximizes WER achievement rate.
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        device: str = "cuda",
        top_k: int = 50,
        seed: int = 42,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.top_k = top_k
        self.rng = random.Random(seed)

        print(f"[INFO] Loading: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        if self.device == "cuda":
            self.model.half()
        print(f"[INFO] Loaded on {self.device}")

    @torch.inference_mode()
    def _predict_mask(self, tokens: List[str], mask_idx: int) -> List[str]:
        """
        Mask token at mask_idx, return top-K valid candidates.
        """
        original = tokens[mask_idx]
        masked_tokens = tokens.copy()
        masked_tokens[mask_idx] = self.tokenizer.mask_token
        masked_text = self._detokenize(masked_tokens)

        inputs = self.tokenizer(
            masked_text, return_tensors="pt",
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        mask_id = self.tokenizer.mask_token_id
        mask_pos = (inputs["input_ids"][0] == mask_id).nonzero(as_tuple=True)[0]
        if len(mask_pos) == 0:
            return []

        logits = outputs.logits[0, mask_pos[0].item()]
        top_ids = torch.topk(logits, k=self.top_k).indices.tolist()

        candidates = []
        for tid in top_ids:
            word = self.tokenizer.decode([tid]).strip()
            if word.isalpha() and word.lower() != original.lower() and len(word) >= 2:
                candidates.append(word)
        return candidates

    def inject_noise(self, text: str, wer: float) -> Dict:
        """
        Inject noise at target WER.

        Returns dict with perturbed text and full statistics.
        """
        if not text or not text.strip():
            return self._empty(text)

        tokens = word_tokenize(text)
        n_words = sum(1 for t in tokens if t.isalpha())

        if n_words == 0:
            return self._empty(text)

        target_subs = max(1, round(n_words * wer))

        # Candidate positions: alphabetic + not protected
        candidates = [
            i for i, t in enumerate(tokens)
            if t.isalpha() and not is_protected(t)
        ]
        n_candidate_positions = len(candidates)

        self.rng.shuffle(candidates)

        n_sub = 0
        n_att = 0
        n_cand_total = 0
        details = []

        for pos in candidates:
            if n_sub >= target_subs:
                break

            orig = tokens[pos]
            n_att += 1

            preds = self._predict_mask(tokens, pos)
            n_cand_total += len(preds)

            if preds:
                repl = preds[0]

                # Preserve casing
                if orig.isupper():
                    repl = repl.upper()
                elif orig[0].isupper():
                    repl = repl.capitalize()

                tokens[pos] = repl
                n_sub += 1
                details.append({"orig": orig, "new": repl, "pos": pos})

        return {
            "perturbed": self._detokenize(tokens),
            "n_words": n_words,
            "target_subs": target_subs,
            "n_candidates": n_cand_total,
            "n_attempted": n_att,
            "n_substituted": n_sub,
            "shortfall": max(0, target_subs - n_sub),
            "details": details,
        }

    @staticmethod
    def _empty(text):
        return {
            "perturbed": text or "", "n_words": 0, "target_subs": 0,
            "n_candidates": 0, "n_attempted": 0, "n_substituted": 0,
            "shortfall": 0, "details": [],
        }

    @staticmethod
    def _detokenize(tokens: List[str]) -> str:
        NO_SPACE_BEFORE = {
            ".", ",", "!", "?", ":", ";", ")", "]", "}", "%",
            "'s", "n't", "'re", "'ve", "'ll", "'d", "'m",
        }
        NO_SPACE_AFTER = {"(", "[", "{"}
        parts = []
        for i, tok in enumerate(tokens):
            if i == 0:
                parts.append(tok)
            elif tok in NO_SPACE_BEFORE or tok.startswith("'"):
                parts.append(tok)
            elif parts and parts[-1] in NO_SPACE_AFTER:
                parts.append(tok)
            else:
                parts.append(" " + tok)
        return "".join(parts)


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

def run(args):
    print(f"[INFO] Loading: {args.input}")
    p = Path(args.input)
    if p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            data = [json.loads(l) for l in f if l.strip()]
    else:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]

    if args.limit > 0:
        data = data[: args.limit]
    print(f"[INFO] {len(data)} samples")

    wers = [float(x) for x in args.wer.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for wer in wers:
        wdir = out / f"wer_{wer}"
        wdir.mkdir(parents=True, exist_ok=True)
        realized_list = []

        for seed in seeds:
            print(f"\n{'='*50}\n  WER={wer}  Seed={seed}\n{'='*50}")

            inj = MLMNoiseInjector(
                model_name=args.model,
                device="cuda" if torch.cuda.is_available() else "cpu",
                top_k=args.top_k,
                seed=seed,
            )

            results = []
            tw, ts = 0, 0

            for item in tqdm(data, desc=f"wer={wer} seed={seed}"):
                q = item.get(args.question_field, "")
                r = inj.inject_noise(q, wer)
                tw += r["n_words"]
                ts += r["n_substituted"]

                out_item = dict(item)
                out_item[args.noisy_field] = r["perturbed"]
                out_item["noise_meta"] = {
                    "noise_type": "mlm_substitution",
                    "model": args.model,
                    "seed": seed,
                    "wer": wer,
                    "stats": {k: r[k] for k in [
                        "n_words", "target_subs", "n_candidates",
                        "n_attempted", "n_substituted", "shortfall", "details",
                    ]},
                }
                results.append(out_item)

            realized = ts / tw if tw > 0 else 0.0
            realized_list.append(realized)

            fp = wdir / f"seed_{seed}.jsonl"
            with open(fp, "w", encoding="utf-8") as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            print(f"[INFO] Saved: {fp}  |  Realized WER: {realized:.4f}")

        # Summary
        summary = {
            "noise_type": "mlm_substitution",
            "model": args.model,
            "wer": wer,
            "mean_realized": sum(realized_list) / len(realized_list),
            "seeds": seeds,
            "n_samples": len(data),
            "method": (
                "MLM-based contextual word substitution using PubMedBERT "
                "(Gu et al., 2021). Words are randomly selected and replaced "
                "with top-1 MLM prediction, following BERT-Attack mechanism "
                "(Li et al., EMNLP 2020). Noise applied at controlled WER "
                "following Moradi et al. (2021). Protected tokens: negation, "
                "numbers, units, medical terms."
            ),
            "notes": {
                "shortfall_reason": "MLM candidate 없음/필터로 인한 거절로 목표 치환수 미달 가능",
                "increase_realized": "top_k↑ 또는 필터 완화",
            },
        }
        sf = wdir / "summary.json"
        with open(sf, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[SUMMARY] WER={wer} Mean={summary['mean_realized']:.4f}")

    print("\n[DONE]")


def main():
    ap = argparse.ArgumentParser(description="MLM noise injection for medical QA")
    ap.add_argument("--input", default="/home/hslee/multiagent/data/raw/medqa/medqa_all_clean.jsonl")
    ap.add_argument("--output", default="/home/hslee/multiagent/data/processed/medqa/mlm")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    ap.add_argument("--wer", default="0.1,0.2,0.3,0.4")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--question-field", default="question")
    ap.add_argument("--noisy-field", default="noisy_question")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()