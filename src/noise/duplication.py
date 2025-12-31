import argparse
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# --- 불용어(Stopwords) 설정 ---
STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","when","while","where","why","how",
    "to","of","in","on","at","by","for","from","with","without","as","into","onto","over","under","within","between",
    "is","am","are","was","were","be","been","being","do","does","did","doing","done",
    "have","has","had","having","can","could","may","might","must","shall","should","will","would",
    "not","no","nor","this","that","these","those","it","its","they","them","their","theirs",
    "he","him","his","she","her","hers","we","us","our","ours","you","your","yours","i","me","my","mine",
    "there","here","who","whom","which","what","any","some","all","each","every","either","neither",
    "both","few","many","much","more","most","less","least","than","too","very","just","also","only",
}

_WORD_RE = re.compile(r"\w+")

# --- Duplication 노이즈 엔진 ---
class DuplicationInjector:
    def __init__(self, seed: int, min_len: int = 2):
        self.rng = random.Random(seed)
        self.min_len = min_len

    def inject(self, text: str, r: float) -> Tuple[str, Dict]:
        # 단어 스캔 및 후보 필터링 (길이 조건 + 불용어 제외)
        spans = [(m.start(), m.end(), m.group()) for m in _WORD_RE.finditer(text)]
        candidates = [
            i for i, (s, e, tok) in enumerate(spans) 
            if (e - s) >= self.min_len and tok.lower() not in STOPWORDS
        ]
        
        n_target = int(round(len(candidates) * r))
        # 삽입 시 인덱스 변화를 방지하기 위해 뒤에서부터 처리
        chosen_indices = sorted(self.rng.sample(candidates, min(n_target, len(candidates))), reverse=True)
        
        details = []
        noisy_text = text
        
        for idx in chosen_indices:
            s, e, tok = spans[idx]
            
            # 선택된 단어 뒤에 " [단어]"를 한 번 더 삽입
            dup_text = f" {tok}"
            noisy_text = noisy_text[:e] + dup_text + noisy_text[e:]
            
            details.append({
                "idx": idx, 
                "token": tok, 
                "duplicated_text": dup_text
            })
            
        return noisy_text, {
            "n_candidates": len(candidates), 
            "n_duplicated": len(chosen_indices), 
            "details": details
        }

# --- 메인 실행기 ---
def run_experiment(args):
    with open(args.input, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()][:args.limit if args.limit > 0 else None]

    out_root = Path(args.output)
    wer_list = [float(x) for x in args.wer.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    print(f"[INFO] Data loaded: {len(data)} samples. Output root: {out_root}")

    for r in wer_list:
        r_dir = out_root / f"wer_{r}"
        r_dir.mkdir(parents=True, exist_ok=True)
        rates = []

        for seed in seeds:
            injector = DuplicationInjector(seed, args.min_word_len)
            results = []
            total_cand, total_dup = 0, 0

            for obj in data:
                src = str(obj.get(args.question_field, ""))
                noisy, meta = injector.inject(src, r)
                
                total_cand += meta["n_candidates"]
                total_dup += meta["n_duplicated"]
                
                new_obj = {
                    **obj, 
                    args.noisy_field: noisy, 
                    "noise_meta": {
                        "noise_type": "word_duplication",
                        "seed": seed, 
                        "wer_target": r, 
                        "stats": meta
                    }
                }
                results.append(new_obj)

            realized = total_dup / total_cand if total_cand > 0 else 0
            rates.append(realized)
            
            out_path = r_dir / f"seed_{seed}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
            print(f"[DONE] r={r}, seed={seed}, realized_rate={realized:.4f}")

        summary = {
            "noise_type": "word_duplication",
            "wer_r": r,
            "seeds": seeds,
            "mean_realized_rate": statistics.mean(rates),
            "std": statistics.pstdev(rates) if len(rates) > 1 else 0,
            "n_samples": len(data),
            "note": "Stopwords excluded from duplication"
        }
        with open(r_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Word Duplication Experiment Runner")
    parser.add_argument("--input", default="/home/hslee/multiagent/data/raw/medqa/medqa_all_clean.jsonl")
    parser.add_argument("--output", default="/home/hslee/multiagent/data/processed/medqa/duplication")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--noisy-field", default="noisy_question")
    parser.add_argument("--wer", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min_word_len", type=int, default=2)

    run_experiment(parser.parse_args())