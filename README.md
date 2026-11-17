# Qwen3.8-Flash-Next-FP8-2xRTXPro6000

This repository serves Qwen3.8-Flash-Next-FP8 with vLLM on 2x NVIDIA RTX PRO 6000 Blackwell GPUs (tensor parallel 2, MTP speculative decoding).

- Deployment config: [`sparkdeck-deployment.json`](sparkdeck-deployment.json)
- Recipe and full benchmark results: [`RECIPE.md`](RECIPE.md)
- Benchmark script: [`benchmarks/bench.py`](benchmarks/bench.py)

## Benchmark summary (2026-11-16)

Per-request decode speed in tok/s (`max_tokens=1024`, thinking off):

| Workload | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| Code | 322.4 | 260.8 | 192.4 | 140.9 |
| Prose | 182.1 | 151.9 | 117.4 | 91.6 |
| Counting | 376.2 | 325.7 | 259.5 | 205.0 |

At C8, aggregate throughput is 1086 tok/s for Code, 715 tok/s for Prose and 1571 tok/s for Counting.

Prefill runs at about 15.2-16.2k tok/s from 4k to 128k prompt tokens: 16.2k tok/s at 16k and 15.2k tok/s at 128k. A 128k prompt takes 8.6 s to the first token.

## Reproduce

```bash
python3 benchmarks/bench.py --base-url http://localhost:8012 --model qwen3.8-flash-next \
    --out benchmarks/results/$(date +%F)
```
