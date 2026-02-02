import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Dict, Tuple
import re

# --- Word Swap 노이즈 엔진 (확률적 방식) ---
class WordSwapAugmenter:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def inject(self, text: str, wer_rate: float) -> Tuple[str, Dict]:
        """
        문장별로 swap 적용 (문장 경계 넘지 않음)
        """
        if not text or not isinstance(text, str):
            return text, {"applied": False, "reason": "empty_or_invalid"}
        
        # 1. 문장 분리
        sentences = re.split(r'([.!?]\s+)', text)
        # 예: "A B C. D E F. G H." → ['A B C', '. ', 'D E F', '. ', 'G H.']
        
        noisy_parts = []
        total_stats = {
            "n_swaps": 0,
            "n_affected_words": 0,
            "n_words": 0
        }
        
        # 2. 각 문장마다 개별 처리
        for i, part in enumerate(sentences):
            if i % 2 == 0:  # 문장 내용
                if part.strip():
                    noisy, meta = self._swap_in_sentence(part, wer_rate)
                    noisy_parts.append(noisy)
                    
                    if meta.get("applied"):
                        total_stats["n_swaps"] += meta["n_swaps"]
                        total_stats["n_affected_words"] += meta["n_affected_words"]
                    total_stats["n_words"] += meta.get("n_words", 0)
                else:
                    noisy_parts.append(part)
            else:  # 구분자 (. ! ? 등)
                noisy_parts.append(part)
        
        contaminated_text = "".join(noisy_parts)
        
        return contaminated_text, {
            "applied": total_stats["n_swaps"] > 0,
            "n_words": total_stats["n_words"],
            "n_swaps": total_stats["n_swaps"],
            "n_affected_words": total_stats["n_affected_words"],
            "wer_target": wer_rate,
            "wer_actual": total_stats["n_affected_words"] / total_stats["n_words"] if total_stats["n_words"] > 0 else 0,
            "expected_swaps": (total_stats["n_words"] * wer_rate) / 2.0,
            "reason": "sentence_level_swap"
        }

    def _swap_in_sentence(self, text: str, wer_rate: float) -> Tuple[str, Dict]:
        """한 문장 내에서만 swap (기존 로직)"""
        words = text.split()
        n_words = len(words)
        
        if n_words < 2:
            return text, {"applied": False, "n_words": n_words, "reason": "n_words < 2"}
        
        if wer_rate <= 0:
            return text, {"applied": False, "reason": "rate_is_zero"}
        
        # Swap 개수 계산
        expected_swaps = (n_words * wer_rate) / 2.0
        n_swaps = int(expected_swaps)
        if self.rng.random() < (expected_swaps - n_swaps):
            n_swaps += 1
        if n_swaps == 0:  # swap 없으면 그냥 원본 반환
            return text, {"applied": False, "n_words": n_words}
            
        max_possible_swaps = n_words - 1
        if n_swaps > max_possible_swaps:
            n_swaps = max_possible_swaps
        
        available_positions = list(range(n_words - 1))
        swap_positions = self.rng.sample(available_positions, n_swaps)
        
        new_words = list(words)
        for pos in swap_positions:
            new_words[pos], new_words[pos + 1] = new_words[pos + 1], new_words[pos]
        
        return " ".join(new_words), {
            "applied": True,
            "n_words": n_words,
            "n_swaps": n_swaps,
            "n_affected_words": n_swaps * 2,
            "wer_target": wer_rate,
            "wer_actual": (n_swaps * 2) / n_words
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
        data = [json.loads(line) for line in f if line.strip()]
    
    if args.limit > 0:
        data = data[:args.limit]

    out_root = Path(args.output)
    
    # 입력 파싱
    wer_list = [float(x) for x in args.wer.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    print(f"[INFO] Loaded {len(data)} samples.")
    print(f"[INFO] WER Rates: {wer_list}")
    print(f"[INFO] Seeds: {seeds}")
    print(f"[INFO] Output Root: {out_root}")

    # 2. 실험 루프
    for wer in wer_list:
        wer_dir = out_root / f"wer_{wer}"  # p_10 대신 wer_0.1
        wer_dir.mkdir(parents=True, exist_ok=True)
        
        experiment_stats = {
            "applied_count": [],
            "actual_wer": []
        }

        for seed in seeds:
            augmenter = WordSwapAugmenter(seed)
            results = []
            
            applied_count = 0
            eligible_count = 0
            actual_wer_sum = 0.0

            for obj in data:
                src = str(obj.get(args.question_field, ""))
                
                # 노이즈 주입 (Swap)
                noisy_text, meta = augmenter.inject(src, wer)
                
                # 통계 수집 (2개 이상 단어가 있는 경우만)
                if meta.get("n_words", 0) >= 2:
                    eligible_count += 1
                    if meta.get("applied"):
                        applied_count += 1
                        actual_wer_sum += meta.get("wer_actual", 0)
                
                # 결과 저장
                new_obj = obj.copy()
                new_obj[args.noisy_field] = noisy_text
                new_obj["noise_meta"] = {
                    "noise_type": "word_swap",
                    "seed": seed,
                    "wer_target": wer,
                    "stats": meta
                }
                results.append(new_obj)

            # Seed별 통계
            realized_rate = applied_count / eligible_count if eligible_count > 0 else 0
            avg_actual_wer = actual_wer_sum / applied_count if applied_count > 0 else 0
            
            experiment_stats["applied_count"].append(realized_rate)
            experiment_stats["actual_wer"].append(avg_actual_wer)
            
            # 파일 저장
            out_path = wer_dir / f"seed_{seed}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
            print(f"[DONE] wer={wer}, seed={seed} -> {out_path.name}")
            print(f"       Applied: {applied_count}/{eligible_count}, Avg Actual WER: {avg_actual_wer:.4f}")

        # 3. Summary 저장
        summary = {
            "noise_type": "word_swap",
            "wer_target": wer,
            "seeds": seeds,
            "mean_applied_rate": statistics.mean(experiment_stats["applied_count"]) if experiment_stats["applied_count"] else 0,
            "std_applied_rate": statistics.pstdev(experiment_stats["applied_count"]) if len(experiment_stats["applied_count"]) > 1 else 0,
            "mean_actual_wer": statistics.mean(experiment_stats["actual_wer"]) if experiment_stats["actual_wer"] else 0,
            "std_actual_wer": statistics.pstdev(experiment_stats["actual_wer"]) if len(experiment_stats["actual_wer"]) > 1 else 0,
            "n_samples": len(data),
            "note": "Probabilistic Adjacent Word Swap. Expected WER achieved through probabilistic rounding."
        }
        
        with open(wer_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SUMMARY] WER={wer}")
        print(f"  Mean Actual WER: {summary['mean_actual_wer']:.4f} ± {summary['std_actual_wer']:.4f}")

    print("\n[ALL JOBS DONE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Word Swap Noise Injection Experiment")
    parser.add_argument("--input", default="/home/hslee/multiagent/data/raw/medqa/medqa_all_clean.jsonl")
    parser.add_argument("--output", default="/home/hslee/multiagent/data/processed/medqa/swap")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--noisy-field", default="noisy_question")
    parser.add_argument("--wer", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--limit", type=int, default=0)
    
    args = parser.parse_args()
    run_experiment(args)
