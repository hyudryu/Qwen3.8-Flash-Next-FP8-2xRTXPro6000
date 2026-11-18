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
        # end-to-end duration of each request (send -> last token)
        "e2e_median_s": statistics.median(r["e2e_s"] for r in res),
        "e2e_mean_s": statistics.mean(r["e2e_s"] for r in res),
        "e2e_max_s": max(r["e2e_s"] for r in res),
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
    concs = sorted({r["concurrency"] for r in conc_rows})
    workloads = list(dict.fromkeys(r["workload"] for r in conc_rows))
    by = {(r["workload"], r["concurrency"]): r for r in conc_rows}

    def grid(title, note, fmt):
        out = [f"### {title}", ""]
        if note:
            out += [note, ""]
        out.append("| Workload | " + " | ".join(f"C{c}" for c in concs) + " |")
        out.append("|---|" + "---:|" * len(concs))
        for w in workloads:
            cells = [fmt(by[(w, c)]) if (w, c) in by else "-" for c in concs]
            out.append(f"| {w} | " + " | ".join(cells) + " |")
        out.append("")
        return out

    def scale(r):
        base = by.get((r["workload"], concs[0]))
        return f"{r['aggregate_output_tps'] / base['aggregate_output_tps']:.2f}x" if base else "-"

    def approx_latency(r):
        secs = (r["mean_completion_tokens"] - 1) / r["per_request_decode_tps_mean"] + r["ttft_median_s"]
        return f"~{secs:.2f}"

    has_e2e = all("e2e_median_s" in r for r in conc_rows)

    total_req = sum(r["requests"] for r in conc_rows)
    total_out = sum(r["requests"] * r["mean_completion_tokens"] for r in conc_rows)
    L = ["## Run", ""]
    L.append(f"- Finished: {meta['date']} UTC")
    L.append(f"- Endpoint: `{meta['base_url']}`, served model `{meta['model']}`")
    for g in meta.get("gpus") or []:
        L.append(f"- GPU: {g}")
    L.append(f"- Sampling: max_tokens={meta['max_tokens']}, temperature={meta['temperature']}, "
             f"top_p={meta['top_p']}, thinking disabled, streaming")
    L.append(f"- Load: closed loop with C concurrent clients, {meta['rounds']} request(s) per client; "
             f"{total_req} decode requests and {total_out:.0f} output tokens in total")
    if "prefill_reps" in meta:
        L.append(f"- Prefill: median of {meta['prefill_reps']} single requests per length")
    if "seed" in meta:
        L.append(f"- Seed: {meta['seed']}")
    L.append("")
    L += grid("Aggregate output throughput (tok/s)",
              "Total output tokens of all requests / wall time from the first request sent to the last one finished.",
              lambda r: f"{r['aggregate_output_tps']:.1f}")
    L += grid(f"Aggregate throughput vs C{concs[0]}", None, scale)
    L += grid("TTFT p50 (s)", None, lambda r: f"{r['ttft_median_s']:.3f}")
    if has_e2e:
        L += grid("Request latency p50 (s)",
                  "Median end-to-end duration of the individual requests (request sent to last token).",
                  lambda r: f"{r['e2e_median_s']:.2f}")
        L += grid("Request latency mean / max (s)",
                  "Mean and maximum end-to-end duration of the individual requests.",
                  lambda r: f"{r['e2e_mean_s']:.2f} / {r['e2e_max_s']:.2f}")
    else:
        L += grid("Approximate request latency (s), not measured",
                  "These results predate per-request duration recording, so this is an estimate: "
                  "TTFT p50 + (mean output tokens - 1) / mean per-request decode speed. "
                  "It is not the mean or median request latency and understates it when "
                  "decode speeds differ between requests.", approx_latency)
    L += ["### Detail", "",
          "Per-request decode = one request's output tokens / time after its first token.", "",
          "| Workload | C | Requests | Output tokens | Wall (s) | Aggregate tok/s "
          "| Per-request decode tok/s mean (min-max) | TTFT p50 (s) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in conc_rows:
        L.append(f"| {r['workload']} | C{r['concurrency']} | {r['requests']} | "
                 f"{r['requests'] * r['mean_completion_tokens']:.0f} | {r['wall_s']:.2f} | "
                 f"{r['aggregate_output_tps']:.1f} | {r['per_request_decode_tps_mean']:.1f} "
                 f"({r['per_request_decode_tps_min']:.1f}-{r['per_request_decode_tps_max']:.1f}) | "
                 f"{r['ttft_median_s']:.3f} |")
    L.append("")
    L += ["### Prefill (single request, unique prompt, no prefix-cache hits)", "",
          "| Target tokens | Prompt tokens | TTFT p50 (s) | Prefill tok/s | Reps |",
          "|---:|---:|---:|---:|---:|"]
    for r in prefill_rows:
        L.append(f"| {r['target_tokens']} | {r['prompt_tokens']:.0f} | {r['ttft_median_s']:.3f} | "
                 f"{r['prefill_tps']:.0f} | {r['reps']} |")
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
    p.add_argument("--render", metavar="JSON",
                   help="rebuild the .md report from an existing results .json and exit")
    args = p.parse_args()
    if args.render:
        with open(args.render) as f:
            data = json.load(f)
        out = args.render[:-5] if args.render.endswith(".json") else args.render
        md = to_markdown(data["meta"], data["concurrency"], data["prefill"])
        with open(out + ".md", "w") as f:
            f.write(md)
        print(md)
        return
    rng = random.Random(args.seed)

    print("warmup...", flush=True)
    for w in WORKLOADS.values():
        stream_chat(args.base_url, args.model, w, 64, args.temperature, args.top_p)

    conc_rows = []
    for c in args.concurrency:
        for name, prompt in WORKLOADS.items():
            r = run_concurrency(args, name, prompt, c)
            conc_rows.append(r)
            print(f"{name:<9} C{c}: agg {r['aggregate_output_tps']:7.1f} tok/s  "
                  f"per-req {r['per_request_decode_tps_mean']:7.1f} tok/s  ttft {r['ttft_median_s']:.3f}s  "
                  f"e2e p50 {r['e2e_median_s']:.2f}s", flush=True)

    prefill_rows = run_prefill(args, rng)

    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "rounds": args.rounds,
        "concurrency": args.concurrency,
        "prefill_reps": args.prefill_reps,
        "seed": args.seed,
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
