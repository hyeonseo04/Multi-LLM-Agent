import argparse
import json
import random
import re
import statistics
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import nltk
from gensim.models import KeyedVectors
from tqdm import tqdm  # 진행바 라이브러리 추가

def ensure_nltk_data():
    """NLTK 품사 태깅 데이터 자동 확보"""
    for pkg in ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"taggers/{pkg}" if "tagger" in pkg else f"tokenizers/{pkg}")
        except:
            nltk.download(pkg, quiet=True)

# --- 텍스트 처리 유틸리티 ---
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*|[^\s]")

def simple_tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)

def simple_detokenize(tokens: List[str]) -> str:
    out = []
    for t in tokens:
        if not out: out.append(t)
        elif t in [".", ",", ":", ";", ")", "]", "}", "?", "!", "%"]: out[-1] += t
        elif out[-1] in ["(", "[", "{"]: out[-1] += t
        elif t in ["/", "-"] or out[-1] in ["/", "-"]: out[-1] += t
        else: out.append(" " + t)
    return "".join(out).strip()

def preserve_case(src: str, dst: str) -> str:
    if src.isupper(): return dst.upper()
    if src.istitle(): return dst.title()
    return dst

# --- Embedding Substitution 엔진 ---
class EmbeddingSubstitutionInjector:
    def __init__(self, seed: int, kv: KeyedVectors, top_k: int, min_cos: float):
        self.rng = random.Random(seed)
        self.kv = kv
        self.top_k = top_k
        self.min_cos = min_cos
        ensure_nltk_data()

    def _get_neighbors(self, word: str) -> List[str]:
        if word not in self.kv: return []
        try:
            sims = self.kv.most_similar(word, topn=self.top_k)
            return [w for w, s in sims if s >= self.min_cos and w.lower() != word.lower()]
        except: return []

    def inject(self, text: str, r: float) -> Tuple[str, Dict]:
        if not text.strip(): return text, {"n_candidates": 0, "n_substituted": 0}
        tokens = simple_tokenize(text)
        word_idxs = [i for i, t in enumerate(tokens) if re.match(r"^[A-Za-z]+$", t)]
        if not word_idxs: return text, {"n_candidates": 0, "n_substituted": 0}
        
        words = [tokens[i] for i in word_idxs]
        tagged = nltk.pos_tag(words)
        
        candidates = [li for li, (w, tag) in enumerate(tagged) 
                      if tag.startswith(("N", "V", "J", "R")) and w in self.kv]
        
        n_target = int(round(len(candidates) * r))
        chosen_locals = self.rng.sample(candidates, min(n_target, len(candidates)))
        
        details = []
        for li in chosen_locals:
            orig = words[li]
            neighbors = self._get_neighbors(orig)
            if neighbors:
                repl = preserve_case(orig, self.rng.choice(neighbors))
                tokens[word_idxs[li]] = repl
                details.append({"orig": orig, "new": repl})
                
        return simple_detokenize(tokens), {"n_candidates": len(candidates), "n_substituted": len(details), "details": details}

# --- 메인 실행 로직 ---
def run_experiment(args):
    # 1. 모델 로딩
    print(f"[BioWordVec] 모델 로딩 중 (약 1~2분 소요): {args.bin_path}")
    kv = KeyedVectors.load_word2vec_format(args.bin_path, binary=True)
    print(f"[BioWordVec] 로딩 완료.")

    # 2. 데이터 로드
    with open(args.input, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]
        if args.limit:
            all_data = all_data[:args.limit]

    out_root = Path(args.output)
    wer_list = [float(x) for x in args.wer.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    print(f"[INFO] 전체 실험 시작: {len(wer_list)}개 비율 x {len(seeds)}개 시드")

    for r in wer_list:
        r_dir = out_root / f"wer_{r}"
        r_dir.mkdir(parents=True, exist_ok=True)
        rates = []

        for seed in seeds:
            injector = EmbeddingSubstitutionInjector(seed, kv, args.top_k, args.min_cos)
            results, total_cand, total_sub = [], 0, 0

            # --- 진행바 추가 구간 ---
            desc = f"[Batch] 비율:{r} 시드:{seed}"
            for obj in tqdm(all_data, desc=desc, unit="line", leave=False):
                noisy, meta = injector.inject(str(obj.get(args.question_field, "")), r)
                total_cand += meta.get("n_candidates", 0)
                total_sub += meta.get("n_substituted", 0)
                results.append({**obj, args.noisy_field: noisy, "noise_meta": {"noise_type": "embedding", "seed": seed, "stats": meta}})

            realized = total_sub / total_cand if total_cand > 0 else 0
            rates.append(realized)
            
            out_path = r_dir / f"seed_{seed}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for res in results: f.write(json.dumps(res, ensure_ascii=False) + "\n")
            print(f"[DONE] 비율={r}, 시드={seed}, 치환율={realized:.4f}")

        # 요약 저장
        summary = {"noise_type": "embedding", "wer_r": r, "mean_rate": statistics.mean(rates), "n_samples": len(all_data)}
        with open(r_dir / "summary.json", "w") as f: json.dump(summary, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 경로 고정
    parser.add_argument("--input", default="/home/hslee/FieldTask/path_data_poison_medqa/dataset/all/medqa_all_clean.jsonl")
    parser.add_argument("--output", default="/home/hslee/multiagent/data/processed/medqa/substitution")
    parser.add_argument("--bin-path", default="/home/hslee/multiagent/data/external/BioWordVec_PubMed_MIMICIII_d200.vec.bin")
    
    parser.add_argument("--wer", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-cos", type=float, default=0.65)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--noisy-field", default="noisy_question")
    
    run_experiment(parser.parse_args())