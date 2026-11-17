# Recipe: Qwen3.8-Flash-Next-FP8 on 2x RTX PRO 6000 (vLLM, TP2)

This recipe serves `Qwen/Qwen3.8-Flash-Next-FP8` with vLLM, tensor parallel across two RTX PRO 6000 Blackwell GPUs, with MTP speculative decoding. The benchmark below was measured on this exact deployment.

## Headline numbers (2026-11-16 run)

| Metric | Code | Prose | Counting |
|---|---:|---:|---:|
| Aggregate output tok/s, C1 | 317.5 | 181.3 | 372.7 |
| Aggregate output tok/s, C2 | 510.5 | 299.2 | 639.4 |
| Aggregate output tok/s, C4 | 753.5 | 459.5 | 1011.1 |
| Aggregate output tok/s, C8 | **1086.1** | **715.4** | **1571.0** |
| C8 vs C1 | 3.42x | 3.95x | 4.22x |
| TTFT p50, C8 | 0.110 s | 0.135 s | 0.164 s |

Prefill: **15.2-16.2k tok/s** from 4k to 128k prompt tokens. A 130k-token prompt reaches its first token in **8.6 s**.

## Hardware and host

| Item | Value |
|---|---|
| GPUs | 2x NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB each |
| GPU power limit | 450 W each |
| GPU link | PCIe Gen5 x16 each; no NVLink |
| NVIDIA driver | 595.99.02 |
| CPU | AMD Ryzen Threadripper PRO 9965WX, 24 cores / 48 threads |
| RAM | 215 GB |
| OS kernel | Linux 7.0.0-43-generic |
| vLLM | `0.1.dev21830+g8244d1c44` (from `/version`), image `ghcr.io/local-inference-lab/vllm:release-10152026` |
| Benchmark client | Python 3.14.4 on the same host, through `localhost` |

## Deployment

The full deployment is in [`sparkdeck-deployment.json`](sparkdeck-deployment.json) (alias `qwen38-fp8-gpu01-tp2`).

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3.8-Flash-Next-FP8` @ `7fb48d708ca6cd8eeed8f4db0433be18af96984f` |
| Served name | `qwen3.8-flash-next` |
| Port | 8012 |
| GPUs | `CUDA_VISIBLE_DEVICES=0,1` |
| Parallelism | `-tp 2 -pp 1`, decode context parallel 1 |
| Quantization | FP8 weights (`--quantization fp8`), FP8 KV cache, bfloat16 activations |
| Max model len | 262144 |
| Max num seqs | 16 |
| Max batched tokens | 6019, chunked prefill |
| KV block size | 64 |
| GPU memory util | 0.96 |
| Speculative decoding | MTP, 4 speculative tokens, probabilistic draft sampling, standard rejection sampling |
| Draft vocabulary | top-49152 tokens (`QWEN38_DRAFT_VOCAB`), `QWEN38_DRAFT_DYNAMIC=1024` |
| CUDA graphs | `FULL_AND_PIECEWISE`, capture sizes 1-64 |
| Kernels | b12x linear backend, b12x GDN decode kernel, FlashInfer autotune off |
| Mamba cache | `--mamba-cache-mode align`, SSM cache dtype `auto` |
| All-reduce | b12x PCIe all-reduce (`VLLM_ENABLE_PCIE_ALLREDUCE=1`), NCCL over `lo` with P2P level `SYS`, 2 channels |
| Model loading | `instanttensor` load format, buffered backend |
| Features | prefix caching, async scheduling, auto tool choice (`qwen3_xml`), `qwen3` reasoning parser |

The runtime file mounts in the deployment overlay custom kernels and the MTP/eagle code from `~/qwen38-performance`. These cover hyperconnection, expert prediction and routing kernels, the MTP model and the eagle spec-decode utilities. They also add the top-49152 draft vocabulary.

## Benchmark method

The script is [`benchmarks/bench.py`](benchmarks/bench.py). It needs only the Python standard library and talks to the OpenAI-compatible `/v1/chat/completions` endpoint.

```bash
python3 benchmarks/bench.py --base-url http://localhost:8012 --model qwen3.8-flash-next \
    --out benchmarks/results/2026-11-16
```

The Markdown report can be rebuilt from the saved JSON without rerunning the benchmark:

```bash
python3 benchmarks/bench.py --render benchmarks/results/2026-11-16.json
```

### Requests

- Streaming chat completions with `stream_options.include_usage`; token counts come from the server's `usage`.
- `max_tokens=1024`, `temperature=0.7`, `top_p=0.8`, thinking disabled (`enable_thinking: false`).
- Every decode request produced exactly 1024 output tokens, so all requests ran to the token cap and the cells compare like with like.
- One 64-token warmup request per workload runs before measurement.

### Workloads

| Workload | Prompt | Why |
|---|---|---|
| Code | Write a thread-safe LRU cache with TTL, size limits, stats and a decorator, plus a unittest suite. Output only code. | Structured text, high MTP acceptance |
| Prose | Write a long short story about a lighthouse keeper who finds a message from her future self. | Free-form text, lowest MTP acceptance |
| Counting | Count from 1 to 3000, separated by commas. | Highly predictable text, near-best-case MTP acceptance |

### Concurrency (C1, C2, C4, C8)

- Closed loop: C client threads each send a request, wait for it to finish, then send the next.
- 2 requests per client, so C1 = 2, C2 = 4, C4 = 8 and C8 = 16 requests per workload. That is 90 decode requests and 92,160 output tokens in total.
- **Aggregate output tok/s** = all output tokens / wall time from the first request sent to the last one finished. This is the server's throughput, and it includes TTFT and the tail where fewer than C requests are still running.
- **Per-request decode tok/s** = one request's output tokens / time after its first token. This is the speed a single user sees. It is only in the detail table below.
- **TTFT** is measured at the client, up to the first streamed content token.

### Prefill

- One request at a time, `max_tokens=1`, `temperature=0`.
- The prompt is random words from a fixed word list. The word count is calibrated to the target token count with the server's `/tokenize` endpoint.
- Each prompt starts with a 24-character nonce from the OS RNG, so no request can hit the prefix cache, even across reruns.
- 3 requests per length; the table shows the median TTFT. Prefill tok/s = prompt tokens / TTFT.
- With `--max-num-batched-tokens 6019`, a 130k prompt is processed in about 22 chunked-prefill steps.

## Results

The run finished at 2026-11-17 05:13 UTC (2026-11-16 local time). Raw data: [`benchmarks/results/2026-11-16.json`](benchmarks/results/2026-11-16.json) and [`.md`](benchmarks/results/2026-11-16.md).

### Aggregate output throughput (tok/s)

| Workload | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| Code | 317.5 | 510.5 | 753.5 | 1086.1 |
| Prose | 181.3 | 299.2 | 459.5 | 715.4 |
| Counting | 372.7 | 639.4 | 1011.1 | 1571.0 |

### Aggregate throughput vs C1

| Workload | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| Code | 1.00x | 1.61x | 2.37x | 3.42x |
| Prose | 1.00x | 1.65x | 2.53x | 3.95x |
| Counting | 1.00x | 1.72x | 2.71x | 4.22x |

### TTFT p50 (s)

| Workload | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| Code | 0.051 | 0.070 | 0.082 | 0.110 |
| Prose | 0.029 | 0.063 | 0.078 | 0.135 |
| Counting | 0.028 | 0.046 | 0.071 | 0.164 |

### Approximate request latency for 1024 output tokens (s), not measured

This run predates per-request duration recording, so these values are estimates: TTFT p50 + 1023 / mean per-request decode speed. They are not the measured mean or median request latency, and they understate it when decode speeds differ between requests. `bench.py` now records each request's end-to-end duration and reports p50, mean and max request latency on future runs.

| Workload | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| Code | 3.22 | 3.99 | 5.40 | 7.37 |
| Prose | 5.65 | 6.80 | 8.79 | 11.30 |
| Counting | 2.75 | 3.19 | 4.01 | 5.16 |

### Detail per cell

| Workload | C | Requests | Output tokens | Wall (s) | Aggregate tok/s | Per-request decode tok/s mean (min-max) | TTFT p50 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Code | C1 | 2 | 2048 | 6.45 | 317.5 | 322.4 (319.0-325.9) | 0.051 |
| Prose | C1 | 2 | 2048 | 11.30 | 181.3 | 182.1 (179.5-184.6) | 0.029 |
| Counting | C1 | 2 | 2048 | 5.50 | 372.7 | 376.2 (373.7-378.8) | 0.028 |
| Code | C2 | 4 | 4096 | 8.02 | 510.5 | 260.8 (254.1-271.3) | 0.070 |
| Prose | C2 | 4 | 4096 | 13.69 | 299.2 | 151.9 (149.6-153.1) | 0.063 |
| Counting | C2 | 4 | 4096 | 6.41 | 639.4 | 325.7 (317.2-334.9) | 0.046 |
| Code | C4 | 8 | 8192 | 10.87 | 753.5 | 192.4 (184.5-203.1) | 0.082 |
| Prose | C4 | 8 | 8192 | 17.83 | 459.5 | 117.4 (112.4-124.3) | 0.078 |
| Counting | C4 | 8 | 8192 | 8.10 | 1011.1 | 259.5 (255.6-263.8) | 0.071 |
| Code | C8 | 16 | 16384 | 15.08 | 1086.1 | 140.9 (130.7-149.0) | 0.110 |
| Prose | C8 | 16 | 16384 | 22.90 | 715.4 | 91.6 (88.8-95.1) | 0.135 |
| Counting | C8 | 16 | 16384 | 10.43 | 1571.0 | 205.0 (200.7-208.4) | 0.164 |

### Prefill speed

| Target tokens | Prompt tokens | TTFT p50 (s) | Prefill tok/s |
|---:|---:|---:|---:|
| 1024 | 1054 | 0.136 | 7738 |
| 4096 | 4112 | 0.263 | 15618 |
| 8192 | 8190 | 0.512 | 16008 |
| 16384 | 16345 | 1.008 | 16216 |
| 32768 | 32657 | 2.027 | 16108 |
| 65536 | 65277 | 4.118 | 15850 |
| 131072 | 130521 | 8.594 | 15187 |

## Observations

- **Throughput scales well to C8.** Aggregate output rises 3.4-4.2x from C1 to C8. It has not flattened at C8. C16 (the `max-num-seqs` limit) was not measured.
- **The workload matters more than concurrency at low C.** MTP acceptance follows how predictable the text is: Counting > Code > Prose. At C1, Prose decodes at about half the speed of Counting.
- **Batching gains are similar across workloads.** C8 vs C1 is 3.42x for Code, 3.95x for Prose and 4.22x for Counting. Counting gains the most and still reaches the highest absolute throughput.
- **TTFT stays low.** It is under 170 ms at C8 for short prompts. Long prompts are dominated by prefill time; see the prefill table.
- **Prefill is flat from 4k to 128k** at 15.2-16.2k tok/s and falls only about 6% at 128k. At 1k tokens, fixed per-request overhead halves the apparent rate.
- **At C1, aggregate is slightly below per-request speed** (317.5 vs 322.4 for Code). That is because aggregate also counts TTFT and the gap between back-to-back requests.

## Caveats

- 2 requests per client per cell is a small sample. Check the min-max range in the detail table before comparing close numbers.
- The decode prompts are short and repeat, so they can hit the prefix cache. That only affects TTFT, which is already small.
- `temperature=0.7` makes the output, and therefore MTP acceptance, differ from run to run.
- The decode results do not use long contexts. Decode speed at 64k+ context will be lower.
