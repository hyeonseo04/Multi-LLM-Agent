# 실행 방법
```bash
# 1. 의존성 설치
uv sync

# 2. 실행
cd src/model/baseline/
uv run baseline_0shot.py
```

## 정상 실행 확인
```
[Info] GPUs: 2 | model: Qwen/Qwen2-72B
Loading checkpoint shards: 100%|████| 3/3 [00:11<00:00]
[Info] Total input files: 121
🚀 Overall Experiments:   0%|  | 0/121
```
위와 같이 표시되면 정상 작동 중입니다.