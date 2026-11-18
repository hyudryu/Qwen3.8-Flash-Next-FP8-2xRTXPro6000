# Qwen3.8-Flash-Next-FP8-2xRTXPro6000

This repository serves Qwen3.8-Flash-Next-FP8 with vLLM on 2x NVIDIA RTX PRO 6000 Blackwell GPUs (tensor parallel 2, FP8 weights and KV cache, MTP speculative decoding with 4 tokens).

- Deployment config: [`sparkdeck-deployment.json`](sparkdeck-deployment.json)
- Recipe, method and full benchmark results: [`RECIPE.md`](RECIPE.md)
- Benchmark script: [`benchmarks/bench.py`](benchmarks/bench.py)
- Raw results: [`benchmarks/results/2026-11-16.json`](benchmarks/results/2026-11-16.json)

## Benchmark summary (2026-11-16)

Settings: streaming chat completions, 1024 output tokens per request, `temperature=0.7`, `top_p=0.8`, thinking off. C clients run back to back with 2 requests each. vLLM `0.1.dev21830+g8244d1c44`.

### Aggregate output throughput (tok/s)

All output tokens / wall time across all concurrent requests.

| Workload | C1 | C2 | C4 | C8 | C8 vs C1 |
|---|---:|---:|---:|---:|---:|
| Code | 317.5 | 510.5 | 753.5 | 1086.1 | 3.42x |
| Prose | 181.3 | 299.2 | 459.5 | 715.4 | 3.95x |
| Counting | 372.7 | 639.4 | 1011.1 | 1571.0 | 4.22x |

### TTFT p50 (s)

| Workload | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| Code | 0.051 | 0.070 | 0.082 | 0.110 |
| Prose | 0.029 | 0.063 | 0.078 | 0.135 |
| Counting | 0.028 | 0.046 | 0.071 | 0.164 |

### Prefill (single request, no prefix-cache hits)

| Prompt tokens | TTFT p50 (s) | Prefill tok/s |
|---:|---:|---:|
| 1054 | 0.136 | 7738 |
| 4112 | 0.263 | 15618 |
| 8190 | 0.512 | 16008 |
| 16345 | 1.008 | 16216 |
| 32657 | 2.027 | 16108 |
| 65277 | 4.118 | 15850 |
| 130521 | 8.594 | 15187 |

See [`RECIPE.md`](RECIPE.md) for hardware, the full server config, per-cell detail, latency and caveats.

## Reproduce

```bash
python3 benchmarks/bench.py --base-url http://localhost:8012 --model qwen3.8-flash-next \
    --out benchmarks/results/$(date +%F)
```

Rebuild a report from saved JSON:

```bash
python3 benchmarks/bench.py --render benchmarks/results/2026-11-16.json
```
