import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Dict, Tuple, List

# --- Word Shuffle 노이즈 엔진 (B방식: 정확한 개수 통제) ---
class WordShuffleAugmenter:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def inject(self, text: str, shuffle_rate: float) -> Tuple[str, Dict]:
        """
        B 방식: 문장에서 정확히 shuffle_rate 비율만큼의 단어를 선택하여
        그 단어들끼리 순서를 섞습니다 (Permutation).
        """
        if not text or not isinstance(text, str):
             return text, {"applied": False, "reason": "empty_or_invalid"}

        words = text.split()
        n_words = len(words)

        # 단어가 2개 미만이면 섞을 수 없음
        if n_words < 2:
            return text, {
                "applied": False, 
                "n_words": n_words,
                "reason": "n_words < 2"
            }
            
        if shuffle_rate <= 0:
             return text, {"applied": False, "reason": "rate_is_zero"}

        # 1. 섞을 단어의 개수를 정확히 계산 (최소 2개)
        n_targets = max(2, int(n_words * shuffle_rate))
        
        # 문장 길이가 짧아서 전체를 다 섞어야 하는 경우 처리
        if n_targets > n_words:
            n_targets = n_words

        # 2. 전체 인덱스 중 '섞을 위치'를 중복 없이 선택
        target_indices = self.rng.sample(range(n_words), n_targets)
        target_indices.sort() # 원래 순서대로 정렬 (나중에 값 넣을 때 필요)
        
        # 3. 해당 위치의 단어들을 가져옴
        target_words = [words[i] for i in target_indices]
        
        # 4. 가져온 단어들을 섞음 (Shuffle)
        self.rng.shuffle(target_words)
        
        # 5. 섞인 단어를 원래 위치에 다시 집어넣음
        new_words = list(words)
        for idx, word in zip(target_indices, target_words):
            new_words[idx] = word

        contaminated_text = " ".join(new_words)
        
        return contaminated_text, {
            "applied": True,
            "n_words": n_words,
            "n_shuffled_words": n_targets, # 실제로 건드린 단어 개수
            "shuffle_rate_target": shuffle_rate,
            "reason": "shuffled_subset"
        }

# --- 메인 실험 실행기 ---
def run_experiment(args):
    # 1. 데이터 로드
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return

    print(f"[INFO] Loading data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()][:args.limit if args.limit > 0 else None]

    out_root = Path(args.output)
    
    # 입력 파싱
    p_list = [float(x) for x in args.p.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    print(f"[INFO] Loaded {len(data)} samples.")
    print(f"[INFO] Shuffle Rates: {p_list}")
    print(f"[INFO] Seeds: {seeds}")
    print(f"[INFO] Output Root: {out_root}")

    # 2. 실험 루프
    for p in p_list:
        p_dir = out_root / f"p_{int(p*100)}" 
        p_dir.mkdir(parents=True, exist_ok=True)
        
        experiment_stats = [] 

        for seed in seeds:
            augmenter = WordShuffleAugmenter(seed)
            results = []
            
            applied_count = 0
            eligible_count = 0 

            for obj in data:
                src = str(obj.get(args.question_field, ""))
                
                # 노이즈 주입 (Shuffle)
                noisy_text, meta = augmenter.inject(src, p)
                
                if meta.get("n_words", 0) >= 2:
                    eligible_count += 1
                    if meta.get("applied"):
                        applied_count += 1
                
                new_obj = obj.copy()
                new_obj[args.noisy_field] = noisy_text
                new_obj["noise_meta"] = {
                    "noise_type": "word_shuffle", # 이름 수정됨
                    "seed": seed,
                    "p_target": p,
                    "stats": meta
                }
                results.append(new_obj)

            realized_rate = applied_count / eligible_count if eligible_count > 0 else 0
            experiment_stats.append(realized_rate)
            
            out_path = p_dir / f"seed_{seed}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
            print(f"  [DONE] p={p}, seed={seed} -> Saved to {out_path.name} (Applied: {applied_count}/{eligible_count})")

        # 3. Summary 저장
        summary = {
            "noise_type": "word_shuffle",
            "p_target": p,
            "seeds": seeds,
            "mean_applied_rate": statistics.mean(experiment_stats) if experiment_stats else 0,
            "std_applied_rate": statistics.pstdev(experiment_stats) if len(experiment_stats) > 1 else 0,
            "total_samples": len(data),
            "note": "Word Shuffling (Subset Permutation). Exact N words selected and shuffled."
        }
        
        with open(p_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[ALL JOBS DONE]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffling Experiment Runner")
    parser.add_argument("--input", default="/home/hslee/multiagent/data/raw/medqa/medqa_all_clean.jsonl")
    parser.add_argument("--output", default="/home/hslee/multiagent/data/processed/medqa/shuffle")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--noisy-field", default="noisy_question")
    parser.add_argument("--wer", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--limit", type=int, defsault=0)

    run_experiment(parser.parse_args())