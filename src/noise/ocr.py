import argparse
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# --- 1. 사용자가 제시한 상세 OCR 혼동 그룹 (36개 그룹) ---
_OCR_CONFUSION_GROUPS = [
    ('O', '0', 'o'), ('I', '1', 'l', 'i'), ('Z', '2'), ('S', '5', '$'), ('G', '6'), ('B', '8'),
    ('g', 'q', '9'), ('A', '4'),
    ('c', 'e'), ('u', 'v', 'y'), ('a', 'd'), ('h', 'n', 'r'), ('f', 't'), ('j', 'i'),
    ('k', 'x'), ('p', 'b'), ('.', ','), ('-', '_'),
    ('m', 'rn', 'nn', 'iii'), ('w', 'vv', 'uv'), ('d', 'cl', 'dl'), ('h', 'li'),
    ('w', 'iv'), ('n', 'ri'), ('u', 'ii'), ('k', 'lc'), ('g', 'cj'),
    ('f', 'ft'), ('y', 'vj'), ('s', 'ss'), ('z', 'zz'),
    ('C', 'c'), ('K', 'k'), ('P', 'p'), ('V', 'v'), ('W', 'w')
]

def build_confusion_map(groups: List[Tuple[str, ...]]) -> Dict[str, List[str]]:
    conf_map = {}
    for group in groups:
        for item in group:
            candidates = [x for x in group if x != item]
            if item not in conf_map:
                conf_map[item] = []
            conf_map[item].extend(candidates)
    return conf_map

_CONFUSION_MAP = build_confusion_map(_OCR_CONFUSION_GROUPS)
_WORD_RE = re.compile(r"\w+")

# --- 2. OCR 노이즈 엔진 (이름을 OCRInjector로 통일하여 에러 방지) ---
class OCRInjector:
    def __init__(self, seed: int, max_replacements: int = 3, min_len: int = 2):
        self.rng = random.Random(seed)
        self.max_replacements = max_replacements
        self.min_len = min_len

    def _mutate_word(self, word: str) -> Tuple[str, List[Dict]]:
        # 변환 가능한 인덱스 추출
        eligible_indices = [i for i, c in enumerate(word) if c in _CONFUSION_MAP]
        if not eligible_indices:
            return word, []

        num_to_rep = self.rng.randint(1, min(self.max_replacements, len(eligible_indices)))
        target_indices = sorted(self.rng.sample(eligible_indices, num_to_rep), reverse=True)
        
        chars = list(word)
        logs = []
        for idx in target_indices:
            before = chars[idx]
            after = self.rng.choice(_CONFUSION_MAP[before])
            chars[idx] = after
            logs.append({"idx": idx, "from": before, "to": after})
            
        return "".join(chars), logs

    def inject(self, text: str, r: float) -> Tuple[str, Dict]:
        spans = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
        candidates = [i for i, (s, e) in enumerate(spans) if (e - s) >= self.min_len]
        
        n_target = int(round(len(candidates) * r))
        chosen_indices = sorted(self.rng.sample(candidates, min(n_target, len(candidates))), reverse=True)
        
        details = []
        noisy_text = text
        for idx in chosen_indices:
            s, e = spans[idx]
            orig = text[s:e]
            corrupted, log = self._mutate_word(orig)
            if corrupted != orig:
                noisy_text = noisy_text[:s] + corrupted + noisy_text[e:]
                details.append({"idx": idx, "orig": orig, "new": corrupted, "replacements": log})
            
        return noisy_text, {"n_candidates": len(candidates), "n_corrupted": len(details), "details": details}

# --- 3. 메인 실험 실행기 ---
def run_experiment(args):
    # 데이터 로드
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
            # 클래스 이름을 OCRInjector로 맞춰 에러 해결
            injector = OCRInjector(seed, args.max_char_replacements, args.min_word_len)
            results = []
            total_cand, total_corr = 0, 0

            for obj in data:
                src = str(obj.get(args.question_field, ""))
                noisy, meta = injector.inject(src, r)
                
                total_cand += meta["n_candidates"]
                total_corr += meta["n_corrupted"]
                
                new_obj = {
                    **obj, 
                    args.noisy_field: noisy, 
                    "noise_meta": {
                        "noise_type": "ocr_error",
                        "seed": seed, 
                        "wer_target": r, 
                        "stats": meta
                    }
                }
                results.append(new_obj)

            realized = total_corr / total_cand if total_cand > 0 else 0
            rates.append(realized)
            
            # 파일 저장
            out_path = r_dir / f"seed_{seed}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
            print(f"[DONE] r={r}, seed={seed}, realized_rate={realized:.4f}")

        # Summary 저장
        summary = {
            "noise_type": "ocr_error",
            "wer_r": r,
            "seeds": seeds,
            "mean_realized_rate": statistics.mean(rates),
            "std": statistics.pstdev(rates) if len(rates) > 1 else 0,
            "n_samples": len(data)
        }
        with open(r_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Noise Experiment Runner")
    parser.add_argument("--input", default="/home/hslee/multiagent/data/raw/medqa/medqa_all_clean.jsonl")
    parser.add_argument("--output", default="/home/hslee/multiagent/data/processed/medqa/ocr")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--noisy-field", default="noisy_question")
    parser.add_argument("--wer", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-char-replacements", type=int, default=3)
    parser.add_argument("--min-word-len", type=int, default=2)

    run_experiment(parser.parse_args())