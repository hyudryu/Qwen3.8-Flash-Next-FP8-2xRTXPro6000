# Recipe: Qwen3.8-Flash-Next-FP8 on 2x RTX PRO 6000 (vLLM, TP2)

## Hardware

- 2x NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB each), driver 595.99.02
- PCIe only (no NVLink); NCCL tuned for PCIe all-reduce over `lo`

## Deployment

The full deployment is in [`sparkdeck-deployment.json`](sparkdeck-deployment.json).

| Setting | Value |
|---|---|
| Image | `ghcr.io/local-inference-lab/vllm:release-10152026` |
| Model | `Qwen/Qwen3.8-Flash-Next-FP8` @ `7fb48d708ca6cd8eeed8f4db0433be18af96984f` |
| Served name | `qwen3.8-flash-next` |
| Port | 8012 |
| Parallelism | `-tp 2 -pp 1` |
| Quantization | FP8 weights, FP8 KV cache, bfloat16 activations |
| Max model len | 262144 |
| Max num seqs | 16 |
| Max batched tokens | 6019 (chunked prefill) |
| GPU memory util | 0.96 |
| Speculative decoding | MTP, 4 speculative tokens, probabilistic draft sampling |
| CUDA graphs | `FULL_AND_PIECEWISE`, capture sizes up to 64 |
| Kernels | b12x linear backend and GDN decode kernel, b12x PCIe all-reduce |
| Features | prefix caching, async scheduling, auto tool choice (`qwen3_xml`), `qwen3` reasoning parser |

The runtime file mounts in the deployment overlay custom kernels and the MTP/eagle code from `~/qwen38-performance`. They also add the top-49152 draft vocabulary.

## Benchmark (2026-11-16)

Measured with [`benchmarks/bench.py`](benchmarks/bench.py), which uses only the Python standard library:

```bash
python3 benchmarks/bench.py --base-url http://localhost:8012 --model qwen3.8-flash-next \
    --out benchmarks/results/2026-11-16
```

Settings:

- Streaming chat completions with `max_tokens=1024`, `temperature=0.7` and `top_p=0.8`
- Thinking disabled
- 2 requests per concurrent slot, after one warmup request per workload
- Workloads:
  - **Code**: an LRU cache module plus its tests
  - **Prose**: a long short story
  - **Counting**: count from 1 to 3000
- Prefill: a single request with a random unique prefix, so the prefix cache never hits; median of 3 runs

Raw data: [`benchmarks/results/2026-11-16.json`](benchmarks/results/2026-11-16.json) and [`.md`](benchmarks/results/2026-11-16.md).

### Per-request decode speed (tok/s)

| Workload | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| Code | 322.4 | 260.8 | 192.4 | 140.9 |
| Prose | 182.1 | 151.9 | 117.4 | 91.6 |
| Counting | 376.2 | 325.7 | 259.5 | 205.0 |

### Aggregate output throughput (tok/s)

| Workload | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| Code | 317.5 | 510.5 | 753.5 | 1086.1 |
| Prose | 181.3 | 299.2 | 459.5 | 715.4 |
| Counting | 372.7 | 639.4 | 1011.1 | 1571.0 |

### TTFT p50 (s)

| Workload | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| Code | 0.051 | 0.070 | 0.082 | 0.110 |
| Prose | 0.029 | 0.063 | 0.078 | 0.135 |
| Counting | 0.028 | 0.046 | 0.071 | 0.164 |

### Prefill speed

| Prompt tokens | TTFT p50 (s) | Prefill tok/s |
|---:|---:|---:|
| 1054 | 0.136 | 7738 |
| 4112 | 0.263 | 15618 |
| 8190 | 0.512 | 16008 |
| 16345 | 1.008 | 16216 |
| 32657 | 2.027 | 16108 |
| 65277 | 4.118 | 15850 |
| 130521 | 8.594 | 15187 |

### Notes

- Decode speed depends on the workload because MTP acceptance depends on how predictable the text is. Counting and code accept more draft tokens than free-form prose.
- At C8, aggregate throughput is about 3.4-4.2x the C1 throughput, and per-request speed falls to about 45-55% of C1.
- Prefill holds steady at about 15.2-16.2k tok/s from 4k to 128k tokens. At 1k, fixed per-request overhead dominates.
