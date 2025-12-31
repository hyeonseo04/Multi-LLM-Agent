import argparse
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# --- 문장 분리 정규식 (구두점 뒤 공백 기준) ---
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# --- Sentence Shuffle 노이즈 엔진 ---
class SentenceShuffler:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def inject(self, text: str, p: float) -> Tuple[str, Dict]:
        # 1. 문장 분리
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]
        n = len(sentences)
        
        # 2. 적용 조건 확인 (문장 2개 이상 & 확률 p)
        if n < 2 or self.rng.random() > p:
            return text, {
                "applied": False, 
                "n_sentences": n, 
                "reason": "n_sents < 2" if n < 2 else "skip_by_p"
            }

        # 3. 셔플 수행
        indices = list(range(n))
        self.rng.shuffle(indices)
        
        # 만약 셔플 결과가 원본과 같다면 (아주 드문 확률) 강제 재배치 시도
        if indices == list(range(n)) and n >= 2:
            while indices == list(range(n)):
                self.rng.shuffle(indices)
        
        shuffled_text = " ".join([sentences[i] for i in indices])
        
        return shuffled_text, {
            "applied": True, 
            "n_sentences": n, 
            "permutation": indices,
            "reason": "shuffled"
        }

# --- 메인 실험 실행기 ---
def run_experiment(args):
    # 데이터 로드
    with open(args.input, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()][:args.limit if args.limit > 0 else None]

    out_root = Path(args.output)
    p_list = [float(x) for x in args.p.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    print(f"[INFO] Data loaded: {len(data)} samples. Output root: {out_root}")

    for p in p_list:
        p_dir = out_root / f"p_{p}"
        p_dir.mkdir(parents=True, exist_ok=True)
        applied_rates = []

        for seed in seeds:
            shuffler = SentenceShuffler(seed)
            results = []
            applied_count = 0
            eligible_count = 0

            for obj in data:
                src = str(obj.get(args.question_field, ""))
                noisy, meta = shuffler.inject(src, p)
                
                if meta["n_sentences"] >= 2:
                    eligible_count += 1
                    if meta["applied"]:
                        applied_count += 1
                
                new_obj = {
                    **obj, 
                    args.noisy_field: noisy, 
                    "noise_meta": {
                        "noise_type": "sentence_shuffling",
                        "seed": seed, 
                        "p_target": p, 
                        "stats": meta
                    }
                }
                results.append(new_obj)

            realized_rate = applied_count / eligible_count if eligible_count > 0 else 0
            applied_rates.append(realized_rate)
            
            # 파일 저장
            out_path = p_dir / f"seed_{seed}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
            print(f"[DONE] p={p}, seed={seed}, applied_rate={realized_rate:.4f} (applied={applied_count})")

        # Summary 저장
        summary = {
            "noise_type": "sentence_shuffling",
            "p_target": p,
            "seeds": seeds,
            "mean_applied_rate": statistics.mean(applied_rates),
            "std": statistics.pstdev(applied_rates) if len(applied_rates) > 1 else 0,
            "n_samples": len(data),
            "note": "applied_rate is calculated over documents with >= 2 sentences"
        }
        with open(r_dir := p_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentence Shuffling Experiment Runner")
    parser.add_argument("--input", default="/home/hslee/multiagent/data/raw/medqa/medqa_all_clean.jsonl")
    parser.add_argument("--output", default="/home/hslee/multiagent/data/processed/medqa/sentence_shuffle")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--noisy-field", default="noisy_question")
    parser.add_argument("--p", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--limit", type=int, default=0)

    run_experiment(parser.parse_args())