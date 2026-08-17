"""Fixed-output GSM8K throughput benchmark.

This uses the release GSM8K zero-shot chat prompt, but it is not a quality
benchmark: it sends ignore_eos=true and intentionally omits stop strings so
every request generates exactly max_new_tokens unless the server errors.
Requests run in strict waves: the next group is admitted only after every
request in the current concurrency-sized group has completed.
"""

import argparse
import asyncio
import json
import re
import statistics
import time
from pathlib import Path

import aiohttp
import datasets
import yaml
from transformers import AutoTokenizer

from dllm_stats import build_tpf_report, fetch_dllm_stats


def build_prompt(tokenizer, question: str) -> str:
    user_content = (
        f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    )
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
ANSWER_RE = re.compile(r"answer is\s*([\-0-9,\.]+)", re.IGNORECASE)
FALLBACK_RE = re.compile(r"(-?[\$0-9.,]{2,})|(-?[0-9]+)")


def extract_pred_strict(text: str) -> str:
    m = BOXED_RE.search(text)
    if m:
        nums = re.findall(r"-?[0-9,]+(?:\.[0-9]+)?", m.group(1).strip())
        if nums:
            return nums[-1].replace(",", "").rstrip(".")
    return ""


def extract_pred_flexible(text: str) -> str:
    strict = extract_pred_strict(text)
    if strict:
        return strict
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".,").replace(",", "")
    matches = FALLBACK_RE.findall(text)
    nums = [g for pair in matches for g in pair if g]
    return nums[-1].strip().rstrip(".,").replace(",", "") if nums else ""


def extract_gold(answer: str) -> str:
    m = re.search(r"####\s*([\-0-9,\.]+)", answer)
    return m.group(1).strip().rstrip(".,").replace(",", "") if m else ""


def equal_numbers(a: str, b: str) -> bool:
    try:
        return float(a.replace(",", "").replace("$", "")) == float(
            b.replace(",", "").replace("$", "")
        )
    except Exception:
        return a.replace(",", "") == b.replace(",", "")


def percentile(values, pct: float):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


async def gen_one(session, idx, prompt, max_new_tokens, url, args):
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": max_new_tokens,
        "ignore_eos": True,
    }
    body = {"text": prompt, "sampling_params": sampling_params}
    t0 = time.time()
    async with session.post(url, json=body) as resp:
        data = await resp.json()
        if resp.status >= 400:
            raise RuntimeError(f"request {idx} failed: status={resp.status}, body={data}")
    elapsed = time.time() - t0
    return idx, body, data, elapsed


async def run_batch(items, prompts, urls, args, label: str):
    results = {}

    async def run_one(local_idx, item):
        i, q, a = item
        url = urls[local_idx % len(urls)]
        idx, body, data, elapsed = await gen_one(
            session, i, prompts[i], args.max_new_tokens, url, args
        )
        results[idx] = (body, data, elapsed)

    wall_start = time.time()
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        done = 0
        for wave_start in range(0, len(items), args.concurrency):
            wave = items[wave_start : wave_start + args.concurrency]
            await asyncio.gather(
                *(
                    run_one(wave_start + offset, item)
                    for offset, item in enumerate(wave)
                )
            )
            done += len(wave)
            rate = done / max(time.time() - wall_start, 1e-9)
            print(
                f"  {label}: {done}/{len(items)} ({rate:.2f} req/s)",
                flush=True,
            )
    return results, time.time() - wall_start


async def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    ds = datasets.load_dataset("openai/gsm8k", "main", split="test")

    measured_start = args.warmup + args.sample_offset
    measured_stop = measured_start + args.limit
    if measured_stop > len(ds):
        raise ValueError(
            f"GSM8K only has {len(ds)} examples, need index {measured_stop - 1}"
        )
    source_indices = list(range(args.warmup)) + list(
        range(measured_start, measured_stop)
    )
    ds = ds.select(source_indices)
    all_items = [
        (source_idx, s["question"], s["answer"])
        for source_idx, s in zip(source_indices, ds)
    ]
    prompts = {i: build_prompt(tokenizer, q) for i, q, _ in all_items}
    prompt_token_counts = {
        i: len(tokenizer.encode(prompts[i], add_special_tokens=False))
        for i, _, _ in all_items
    }

    ports = [int(p.strip()) for p in args.ports.split(",")]
    urls = [f"http://localhost:{p}/generate" for p in ports]

    if args.warmup:
        warm_items = all_items[: args.warmup]
        print(f"warming up {len(warm_items)} requests", flush=True)
        await run_batch(warm_items, prompts, urls, args, "warmup")

    stats_before = await fetch_dllm_stats(ports)
    if args.require_dllm_stats and stats_before is None:
        raise RuntimeError("dLLM statistics are unavailable after warmup")

    measured_items = all_items[args.warmup :]
    results, wall = await run_batch(measured_items, prompts, urls, args, "measure")
    stats_after = await fetch_dllm_stats(ports)
    if args.require_dllm_stats and stats_after is None:
        raise RuntimeError("dLLM statistics are unavailable after measurement")

    total_completion_tokens = 0
    strict_correct = 0
    flex_correct = 0
    samples = []
    latencies = []
    completion_tokens = []
    finish_reasons = []

    for i, q, a in measured_items:
        body, data, elapsed = results[i]
        text = data.get("text", "")
        meta = data.get("meta_info", {})
        tokens = meta.get("completion_tokens", 0) if isinstance(meta, dict) else 0
        finish = (meta.get("finish_reason") or {}).get("type") if isinstance(meta, dict) else None
        total_completion_tokens += tokens
        completion_tokens.append(tokens)
        latencies.append(elapsed)
        finish_reasons.append(finish)

        gold = extract_gold(a)
        strict_pred = extract_pred_strict(text)
        flex_pred = extract_pred_flexible(text)
        if strict_pred and equal_numbers(strict_pred, gold):
            strict_correct += 1
        if flex_pred and equal_numbers(flex_pred, gold):
            flex_correct += 1

        samples.append(
            {
                "id": i,
                "question": q,
                "gold": gold,
                "prompt_tokens": prompt_token_counts[i],
                "request_body": body,
                "generation": text,
                "meta_info": meta,
                "elapsed": elapsed,
                "strict_pred": strict_pred,
                "flex_pred": flex_pred,
            }
        )

    yaml_config = None
    if args.algorithm_config:
        with open(args.algorithm_config, "r") as f:
            yaml_config = yaml.safe_load(f)

    latency_summary = {
        "mean": statistics.mean(latencies) if latencies else None,
        "p50": percentile(latencies, 50),
        "p90": percentile(latencies, 90),
        "p95": percentile(latencies, 95),
        "min": min(latencies) if latencies else None,
        "max": max(latencies) if latencies else None,
    }
    token_summary = {
        "mean": statistics.mean(completion_tokens) if completion_tokens else None,
        "p50": percentile(completion_tokens, 50),
        "p90": percentile(completion_tokens, 90),
        "p95": percentile(completion_tokens, 95),
        "min": min(completion_tokens) if completion_tokens else None,
        "max": max(completion_tokens) if completion_tokens else None,
    }
    finish_counts = {
        str(reason): finish_reasons.count(reason)
        for reason in sorted(set(finish_reasons), key=str)
    }
    protocol_errors = []
    if len(completion_tokens) != len(measured_items):
        protocol_errors.append(
            f"received token metadata for {len(completion_tokens)}/"
            f"{len(measured_items)} measured requests"
        )
    unexpected_lengths = [
        {"id": item[0], "completion_tokens": tokens}
        for item, tokens in zip(measured_items, completion_tokens)
        if tokens != args.max_new_tokens
    ]
    if unexpected_lengths:
        protocol_errors.append(
            f"{len(unexpected_lengths)} measured requests did not return exactly "
            f"{args.max_new_tokens} completion tokens"
        )
    unexpected_finishes = [
        {"id": item[0], "finish_reason": reason}
        for item, reason in zip(measured_items, finish_reasons)
        if reason != "length"
    ]
    if unexpected_finishes:
        protocol_errors.append(
            f"{len(unexpected_finishes)} measured requests did not finish by length"
        )
    dllm_tpf = build_tpf_report(
        stats_before,
        stats_after,
        measured_completion_tokens=total_completion_tokens,
    )
    if args.require_dllm_stats and (
        dllm_tpf is None or dllm_tpf["decode_forwards"] <= 0
    ):
        protocol_errors.append("measured dLLM decode-forward counters are unavailable")
    if args.require_verify_stats and (
        dllm_tpf is None
        or dllm_tpf["verify_decisions"] <= 0
        or dllm_tpf["verify_tpf"] is None
    ):
        protocol_errors.append("measured AR-Trust verification counters are unavailable")

    result = {
        "N": len(measured_items),
        "warmup": args.warmup,
        "sample_offset": args.sample_offset,
        "strict_correct": strict_correct,
        "flex_correct": flex_correct,
        "wall_seconds": wall,
        "total_completion_tokens": total_completion_tokens,
        "aggregate_tps": total_completion_tokens / wall if wall > 0 else None,
        "concurrency": args.concurrency,
        "request_schedule": {
            "policy": "strict_waves",
            "wave_size": args.concurrency,
            "warmup_waves": (
                (args.warmup + args.concurrency - 1) // args.concurrency
            ),
            "measured_waves": (
                (len(measured_items) + args.concurrency - 1) // args.concurrency
            ),
        },
        "max_new_tokens": args.max_new_tokens,
        "request_sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": True,
            "stop": None,
        },
        "algorithm_config_path": args.algorithm_config,
        "algorithm_config": yaml_config,
        "model_path": args.model_path,
        "ports": args.ports,
        "latency_seconds": latency_summary,
        "completion_tokens": token_summary,
        "finish_reason_counts": finish_counts,
        "protocol_valid": not protocol_errors,
        "protocol_errors": protocol_errors,
        "unexpected_completion_lengths": unexpected_lengths,
        "unexpected_finish_reasons": unexpected_finishes,
        "dllm_tpf": dllm_tpf,
        "samples": samples,
    }

    print("\n=== FIXED OUTPUT RESULTS ===")
    print(f"N: {result['N']}, warmup: {args.warmup}")
    print(f"wall clock: {wall:.1f}s")
    print(f"total completion tokens: {total_completion_tokens}")
    print(f"TPS: {result['aggregate_tps']:.1f}")
    print(f"finish reasons: {finish_counts}")
    print(f"latency seconds: {latency_summary}")
    print(f"completion tokens: {token_summary}")
    if dllm_tpf is not None:
        print(
            "dLLM TPF: "
            f"decode={dllm_tpf['decode_tpf']}, "
            f"verify={dllm_tpf['verify_tpf']}, "
            f"avg_accepted={dllm_tpf['average_accepted_draft_tokens']}"
        )
    print(
        f"quality strict/flex: {strict_correct}/{result['N']} / "
        f"{flex_correct}/{result['N']} (diagnostic only)"
    )

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
        print(f"saved: {args.save}")
    if protocol_errors:
        raise RuntimeError("fixed-output protocol validation failed: " + "; ".join(protocol_errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--ports", type=str, default="30000")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--algorithm-config", type=str, default=None)
    parser.add_argument("--require-dllm-stats", action="store_true")
    parser.add_argument("--require-verify-stats", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(main(args))
