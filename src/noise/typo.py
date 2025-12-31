import argparse
import json
import random
import re
import string
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# --- 기초 설정 ---
_WORD_RE = re.compile(r"\w+")
_ALPHANUM = string.ascii_lowercase + string.digits

def build_qwerty_adj() -> Dict[str, str]:
    rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    adj = {}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            neighbors = ""
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < len(rows) and 0 <= nc < len(rows[nr]):
                        neighbors += rows[nr][nc]
            adj[ch] = neighbors
    return adj

_QWERTY_ADJ = build_qwerty_adj()

# --- 노이즈 엔진 ---
class TypoInjector:
    def __init__(self, seed: int, max_edits: int = 3, adj_prob: float = 0.7, min_len: int = 2):
        self.rng = random.Random(seed)
        self.max_edits = max_edits
        self.adj_prob = adj_prob
        self.min_len = min_len
        self.ops = ["delete", "insert", "substitute", "transpose"]
        self.weights = [0.20, 0.20, 0.45, 0.15]

    def _get_neighbor(self, ch: str) -> str:
        base = ch.lower()
        candidates = _QWERTY_ADJ.get(base, _ALPHANUM)
        res = self.rng.choice(candidates)
        return res.upper() if ch.isupper() and res.isalpha() else res

    def _mutate_word(self, word: str) -> Tuple[str, List[Dict]]:
        cur = word
        logs = []
        for _ in range(self.rng.randint(1, self.max_edits)):
            op = self.rng.choices(self.ops, weights=self.weights, k=1)[0]
            prev = cur
            idx = self.rng.randrange(len(cur))
            
            if op == "delete" and len(cur) > 1:
                cur = cur[:idx] + cur[idx+1:]
            elif op == "insert":
                new_ch = self._get_neighbor(cur[idx])
                cur = cur[:idx] + new_ch + cur[idx:]
            elif op == "substitute":
                new_ch = self._get_neighbor(cur[idx]) if self.rng.random() < self.adj_prob else self.rng.choice(_ALPHANUM)
                cur = cur[:idx] + (new_ch.upper() if cur[idx].isupper() else new_ch) + cur[idx+1:]
            elif op == "transpose" and len(cur) > 1:
                idx = self.rng.randrange(len(cur) - 1)
                lst = list(cur)
                lst[idx], lst[idx+1] = lst[idx+1], lst[idx]
                cur = "".join(lst)
            
            if not cur: cur = prev # 안전장치
            logs.append({"op": op, "from": prev, "to": cur})
        return (cur if cur != word else self._mutate_word(word)[0]), logs # No-op 방지

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
            noisy_text = noisy_text[:s] + corrupted + noisy_text[e:]
            details.append({"idx": idx, "orig": orig, "new": corrupted, "edits": log})
            
        return noisy_text, {"n_candidates": len(candidates), "n_corrupted": len(chosen_indices), "details": details}

# --- 메인 실행기 ---
def run_experiment(args):
    with open(args.input, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()][:args.limit if args.limit > 0 else None]

    out_root = Path(args.output)
    wer_list = [float(x) for x in args.wer.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    for r in wer_list:
        r_dir = out_root / f"wer_{r}"
        r_dir.mkdir(parents=True, exist_ok=True)
        rates = []

        for seed in seeds:
            injector = TypoInjector(seed, args.max_char_edits, args.keyboard_adj_prob, args.min_word_len)
            results = []
            total_cand, total_corr = 0, 0

            for obj in data:
                src = str(obj.get(args.question_field, ""))
                noisy, meta = injector.inject(src, r)
                
                total_cand += meta["n_candidates"]
                total_corr += meta["n_corrupted"]
                
                new_obj = {**obj, args.noisy_field: noisy, "noise_meta": {"seed": seed, "wer_target": r, "stats": meta}}
                results.append(new_obj)

            realized = total_corr / total_cand if total_cand > 0 else 0
            rates.append(realized)
            
            with open(r_dir / f"seed_{seed}.jsonl", "w", encoding="utf-8") as f:
                for res in results: f.write(json.dumps(res, ensure_ascii=False) + "\n")
            print(f"[Done] r={r}, seed={seed}, rate={realized:.4f}")

        # Summary 저장
        summary = {"wer": r, "mean_rate": statistics.mean(rates), "std": statistics.pstdev(rates) if len(rates)>1 else 0}
        with open(r_dir / "summary.json", "w") as f: json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Typo noise injector runner")
    
    # 입력 경로는 기존 그대로 (Raw 데이터)
    parser.add_argument(
        "--input", 
        default="/home/hslee/multiagent/data/raw/medqa/medqa_all_clean.jsonl",
        help="Input JSONL path"
    )
    
    # 출력 경로를 요청하신 Processed 폴더로 변경
    parser.add_argument(
        "--output", 
        default="/home/hslee/multiagent/data/processed/medqa/typo",
        help="Output directory root"
    )
    
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--noisy-field", default="noisy_question")
    parser.add_argument("--wer", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-char-edits", type=int, default=3)
    parser.add_argument("--keyboard-adj-prob", type=float, default=0.7)
    parser.add_argument("--min-word-len", type=int, default=2)

    args = parser.parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main()