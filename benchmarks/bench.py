#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible vLLM endpoint (stdlib only).

Measures:
  * decode speed at concurrency C1/C2/C4/C8 for Code, Prose and Counting prompts
  * prefill speed (prompt tokens / time-to-first-token) at several context lengths

Usage:
  python3 benchmarks/bench.py --base-url http://localhost:8012 --model qwen3.8-flash-next \
      --out benchmarks/results/latest
"""
import argparse
import http.client
import json
import os
import random
import statistics
import string
import subprocess
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

WORKLOADS = {
    "Code": (
        "Write a complete, well-documented Python module that implements a thread-safe "
        "LRU cache with per-entry TTL expiry, size limits, hit/miss statistics, and a "
        "decorator interface. Then write an extensive unittest test suite for it. "
        "Output only code."
    ),
    "Prose": (
        "Write a long, richly detailed short story about a lighthouse keeper on a remote "
        "island who finds a message in a bottle that seems to be written by her future "
        "self. Use vivid descriptive prose and dialogue."
    ),
    "Counting": "Count from 1 to 3000, separated by commas. Output only the numbers.",
}

WORDS = (
    "time year people way day man thing woman life child world school state family "
    "student group country problem hand part place case week company system program "
    "question work government number night point home water room mother area money "
    "story fact month lot right study book eye job word business issue side kind head "
    "house service friend father power hour game line end member law car city community "
    "name president team minute idea kid body information back parent face others level "
    "office door health person art war history party result change morning reason research "
    "girl guy moment air teacher force education river mountain signal engine theory"
).split()


def _conn(base, timeout):
    u = urllib.parse.urlparse(base)
    if u.scheme == "https":
        return http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=timeout)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)


def post_json(base, path, body, timeout=60):
    c = _conn(base, timeout)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    if r.status != 200:
        raise RuntimeError(f"{path} -> HTTP {r.status}: {data[:300]!r}")
    return json.loads(data)


def stream_chat(base, model, prompt, max_tokens, temperature, top_p, timeout=1800):
    """Run one streaming chat completion; return timing + usage."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": False},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    c = _conn(base, timeout)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {r.read()[:300]!r}")
    t_first = None
    usage = None
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            break
        ev = json.loads(payload)
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if t_first is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                t_first = time.perf_counter()
    t_end = time.perf_counter()
    c.close()
    if usage is None:
        raise RuntimeError("no usage in stream")
    if t_first is None:
        t_first = t_end
    out = usage["completion_tokens"]
    decode_s = t_end - t_first
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": out,
        "ttft_s": t_first - t0,
        "e2e_s": t_end - t0,
        "decode_tps": (out - 1) / decode_s if decode_s > 0 and out > 1 else float("nan"),
        "t_start": t0,
        "t_end": t_end,
    }


def run_concurrency(args, name, prompt, conc):
    n = conc * args.rounds
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [
            ex.submit(stream_chat, args.base_url, args.model, prompt, args.max_tokens,
                      args.temperature, args.top_p)
            for _ in range(n)
        ]
        res = [f.result() for f in futs]
    wall = max(r["t_end"] for r in res) - min(r["t_start"] for r in res)
    total_out = sum(r["completion_tokens"] for r in res)
    tps = [r["decode_tps"] for r in res]
    return {
        "workload": name,
        "concurrency": conc,
        "requests": n,
        "mean_completion_tokens": statistics.mean(r["completion_tokens"] for r in res),
        "per_request_decode_tps_mean": statistics.mean(tps),
        "per_request_decode_tps_min": min(tps),
        "per_request_decode_tps_max": max(tps),
        "aggregate_output_tps": total_out / wall,
        "ttft_median_s": statistics.median(r["ttft_s"] for r in res),
        "wall_s": wall,
    }


_NONCE_RNG = random.SystemRandom()  # independent of --seed: unique across runs


def make_text(rng, n_words):
    # The nonce must not come from the seeded rng, otherwise reruns recreate
    # identical prompts and can hit retained prefix-cache entries.
    nonce = "".join(_NONCE_RNG.choices(string.ascii_lowercase + string.digits, k=24))
    return f"[{nonce}] " + " ".join(rng.choice(WORDS) for _ in range(n_words))


def run_prefill(args, rng):
    # calibrate words -> tokens with the server tokenizer
    sample = make_text(rng, 4000)
    try:
        tok = post_json(args.base_url, "/tokenize", {"model": args.model, "prompt": sample})
        ratio = tok["count"] / 4000
    except Exception as e:  # noqa: BLE001
        print(f"tokenize failed ({e}); assuming 1.3 tok/word", file=sys.stderr)
        ratio = 1.3
    rows = []
    for target in args.prefill_lengths:
        words = max(1, int(target / ratio))
        samples = []
        for _ in range(args.prefill_reps):
            # unique random prefix every time -> no prefix-cache hits
            prompt = make_text(rng, words) + "\n\nReply with OK."
            r = stream_chat(args.base_url, args.model, prompt, 1, 0.0, 1.0)
            samples.append(r)
        ttft = statistics.median(s["ttft_s"] for s in samples)
        ptoks = statistics.median(s["prompt_tokens"] for s in samples)
        rows.append({
            "target_tokens": target,
            "prompt_tokens": ptoks,
            "ttft_median_s": ttft,
            "prefill_tps": ptoks / ttft,
            "reps": len(samples),
        })
        print(f"prefill {ptoks:>7} tok  ttft {ttft:7.3f}s  {ptoks/ttft:9.0f} tok/s", flush=True)
    return rows


def gpu_info():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
    except Exception:  # noqa: BLE001
        return []


def to_markdown(meta, conc_rows, prefill_rows):
    L = []
    L.append(f"Run: {meta['date']} UTC, model `{meta['model']}`, max_tokens={meta['max_tokens']}, "
             f"temperature={meta['temperature']}, top_p={meta['top_p']}, thinking disabled, "
             f"{meta['rounds']} request(s) per concurrent slot.")
    L.append("")
    L.append("### Decode throughput by concurrency")
    L.append("")
    L.append("Per-request = output tokens/s seen by one client after its first token. "
             "Aggregate = total output tokens / wall time across all clients.")
    L.append("")
    L.append("| Workload | C | Per-request tok/s (mean) | Per-request min-max | Aggregate tok/s | TTFT p50 (s) | Avg output tokens |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in conc_rows:
        L.append(f"| {r['workload']} | C{r['concurrency']} | {r['per_request_decode_tps_mean']:.1f} | "
                 f"{r['per_request_decode_tps_min']:.1f}-{r['per_request_decode_tps_max']:.1f} | "
                 f"{r['aggregate_output_tps']:.1f} | {r['ttft_median_s']:.3f} | {r['mean_completion_tokens']:.0f} |")
    L.append("")
    L.append("### Summary: per-request decode tok/s")
    L.append("")
    concs = sorted({r["concurrency"] for r in conc_rows})
    L.append("| Workload | " + " | ".join(f"C{c}" for c in concs) + " |")
    L.append("|---|" + "---:|" * len(concs))
    for w in WORKLOADS:
        cells = []
        for c in concs:
            m = [r for r in conc_rows if r["workload"] == w and r["concurrency"] == c]
            cells.append(f"{m[0]['per_request_decode_tps_mean']:.1f}" if m else "-")
        L.append(f"| {w} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### Prefill speed (single request, no prefix-cache hits)")
    L.append("")
    L.append("| Prompt tokens | TTFT p50 (s) | Prefill tok/s |")
    L.append("|---:|---:|---:|")
    for r in prefill_rows:
        L.append(f"| {r['prompt_tokens']:.0f} | {r['ttft_median_s']:.3f} | {r['prefill_tps']:.0f} |")
    L.append("")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8012")
    p.add_argument("--model", default="qwen3.8-flash-next")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--rounds", type=int, default=2, help="requests per concurrent slot")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--prefill-lengths", type=int, nargs="+",
                   default=[1024, 4096, 8192, 16384, 32768, 65536, 131072])
    p.add_argument("--prefill-reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="benchmarks/results/latest")
    args = p.parse_args()
    rng = random.Random(args.seed)

    print("warmup...", flush=True)
    for w in WORKLOADS.values():
        stream_chat(args.base_url, args.model, w, 64, args.temperature, args.top_p)

    conc_rows = []
    for c in args.concurrency:
        for name, prompt in WORKLOADS.items():
            r = run_concurrency(args, name, prompt, c)
            conc_rows.append(r)
            print(f"{name:<9} C{c}: per-req {r['per_request_decode_tps_mean']:7.1f} tok/s  "
                  f"agg {r['aggregate_output_tps']:7.1f} tok/s  ttft {r['ttft_median_s']:.3f}s", flush=True)

    prefill_rows = run_prefill(args, rng)

    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "rounds": args.rounds,
        "gpus": gpu_info(),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump({"meta": meta, "concurrency": conc_rows, "prefill": prefill_rows}, f, indent=2)
    md = to_markdown(meta, conc_rows, prefill_rows)
    with open(args.out + ".md", "w") as f:
        f.write(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
