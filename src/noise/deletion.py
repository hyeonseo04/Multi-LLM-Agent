import argparse
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# --- 불용어(Stopwords) 설정: 분석에 핵심적이지 않은 단어들 제외 ---
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

# --- Deletion 노이즈 엔진 ---
class DeletionInjector:
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
        chosen_indices = sorted(self.rng.sample(candidates, min(n_target, len(candidates))), reverse=True)
        
        details = []
        char_list = list(text)
        
        for idx in chosen_indices:
            s, e, tok = spans[idx]
            
            # 공백 처리 로직: 단어 삭제 후 이중 공백 방지
            # 왼쪽 공백이 있으면 왼쪽 포함 삭제, 없으면 오른쪽 공백 포함 삭제
            ds, de = s, e
            if ds > 0 and text[ds-1].isspace():
                ds -= 1
            elif de < len(text) and text[de].isspace():
                de += 1
            
            deleted_text = "".join(char_list[ds:de])
            char_list[ds:de] = [] # 해당 구간 삭제
            
            details.append({
                "idx": idx, 
                "token": tok, 
                "deleted_segment": deleted_text
            })
            
        # 결과 텍스트 정제 (연속 공백 제거 및 양끝 정리)
        noisy_text = "".join(char_list)
        noisy_text = re.sub(r"[ \t]+", " ", noisy_text).strip()
        
        return noisy_text, {
            "n_candidates": len(candidates), 
            "n_deleted": len(chosen_indices), 
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
            injector = DeletionInjector(seed, args.min_word_len)
            results = []
            total_cand, total_del = 0, 0

            for obj in data:
                src = str(obj.get(args.question_field, ""))
                noisy, meta = injector.inject(src, r)
                
                total_cand += meta["n_candidates"]
                total_del += meta["n_deleted"]
                
                new_obj = {
                    **obj, 
                    args.noisy_field: noisy, 
                    "noise_meta": {
                        "noise_type": "word_deletion",
                        "seed": seed, 
                        "wer_target": r, 
                        "stats": meta
                    }
                }
                results.append(new_obj)

            realized = total_del / total_cand if total_cand > 0 else 0
            rates.append(realized)
            
            out_path = r_dir / f"seed_{seed}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
            print(f"[DONE] r={r}, seed={seed}, realized_rate={realized:.4f}")

        summary = {
            "noise_type": "word_deletion",
            "wer_r": r,
            "seeds": seeds,
            "mean_realized_rate": statistics.mean(rates),
            "std": statistics.pstdev(rates) if len(rates) > 1 else 0,
            "n_samples": len(data),
            "note": "Stopwords excluded from deletion"
        }
        with open(r_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Word Deletion Experiment Runner")
    parser.add_argument("--input", default="/home/hslee/multiagent/data/raw/medqa/medqa_all_clean.jsonl")
    parser.add_argument("--output", default="/home/hslee/multiagent/data/processed/medqa/deletion")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--noisy-field", default="noisy_question")
    parser.add_argument("--wer", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min_word_len", type=int, default=2)

    run_experiment(parser.parse_args())