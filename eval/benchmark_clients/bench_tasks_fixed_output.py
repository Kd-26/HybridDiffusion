"""Fixed-output throughput benchmark for task prompts.

This is a throughput-only client for HumanEval, GPQA Diamond, and
LiveCodeBench v6 prompts. It matches the GSM8K fixed-output protocol: warm up
first, then measure a fixed number of requests, send `ignore_eos=true`, and
omit stop strings so every successful request should finish by length.
Requests run in strict waves: the next group is admitted only after every
request in the current concurrency-sized group has completed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

import aiohttp
import datasets
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from dllm_stats import build_tpf_report, fetch_dllm_stats


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def apply_chat_template(
    tokenizer: Any,
    prompt: str,
    enable_thinking: bool,
    messages: list[dict[str, str]] | None = None,
) -> str:
    messages = messages or [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_humaneval_prompt(prompt_code: str) -> str:
    return (
        "Read the following function signature and docstring, and fully implement "
        "the function described. Your response should only contain the code for "
        f"this function.\n{prompt_code}"
    )


def build_gpqa_prompt(item: dict[str, Any]) -> tuple[str, str]:
    question = item["Question"]
    choices = [
        item["Correct Answer"],
        item["Incorrect Answer 1"],
        item["Incorrect Answer 2"],
        item["Incorrect Answer 3"],
    ]
    seed = int(hashlib.md5(question.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    indices = list(range(4))
    rng.shuffle(indices)
    shuffled = [choices[i] for i in indices]
    gold_idx = indices.index(0)
    gold_letter = "ABCD"[gold_idx]
    choices_str = "\n".join(f"{'ABCD'[i]}) {shuffled[i]}" for i in range(4))
    prompt = (
        "Answer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
        "Think step by step before answering.\n\n"
        f"{question}\n\n{choices_str}"
    )
    return prompt, gold_letter


def build_lcb_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    question = item["question_content"]
    starter = item.get("starter_code", "").strip()
    system = (
        "You are an expert Python programmer. You will be given a question "
        "(problem specification) and will generate a correct Python program "
        "that matches the specification and passes all tests. "
        "You will NOT return anything except for the program."
    )
    if starter:
        format_instruction = (
            "You will use the following starter code to write the solution to "
            "the problem and enclose your code within delimiters."
        )
        user = (
            f"### Question:\n{question}\n\n"
            f"### Format: {format_instruction}\n"
            f"```python\n{starter}\n```\n\n"
            "### Answer: (use the provided format with backticks)\n\n"
        )
    else:
        format_instruction = (
            "Read the inputs from stdin solve the problem and write the answer "
            "to stdout (do not directly test on the sample inputs). "
            "Enclose your code within delimiters as follows."
        )
        user = (
            f"### Question:\n{question}\n\n"
            f"### Format: {format_instruction}\n"
            "```python\n# YOUR CODE HERE\n```\n\n"
            "### Answer: (use the provided format with backticks)\n\n"
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def load_items(
    task: str,
    measured_limit: int,
    warmup: int,
    sample_offset: int,
) -> list[dict[str, Any]]:
    measured_start = warmup + sample_offset
    measured_stop = measured_start + measured_limit
    source_indices = list(range(warmup)) + list(
        range(measured_start, measured_stop)
    )
    if task == "humaneval":
        ds = datasets.load_dataset("openai/openai_humaneval", split="test")
        if measured_stop > len(ds):
            raise ValueError(
                f"HumanEval only has {len(ds)} examples, "
                f"need index {measured_stop - 1}"
            )
        items = []
        for source_idx, item in zip(source_indices, ds.select(source_indices)):
            items.append(
                {
                    "id": source_idx,
                    "task": task,
                    "source_id": item["task_id"],
                    "prompt": build_humaneval_prompt(item["prompt"]),
                    "gold": None,
                }
            )
        return items

    if task in {"gpqa", "gpqa_diamond"}:
        ds = datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        if measured_stop > len(ds):
            raise ValueError(
                f"GPQA Diamond only has {len(ds)} examples, "
                f"need index {measured_stop - 1}"
            )
        items = []
        for source_idx, item in zip(source_indices, ds.select(source_indices)):
            prompt, gold = build_gpqa_prompt(item)
            items.append(
                {
                    "id": source_idx,
                    "task": "gpqa_diamond",
                    "source_id": source_idx,
                    "prompt": prompt,
                    "gold": gold,
                }
            )
        return items

    if task == "lcb_v6":
        path = hf_hub_download(
            "livecodebench/code_generation_lite",
            "test6.jsonl",
            repo_type="dataset",
        )
        with open(path, encoding="utf-8") as handle:
            data = [json.loads(line) for line in handle]
        if measured_stop > len(data):
            raise ValueError(
                f"LiveCodeBench v6 only has {len(data)} examples, "
                f"need index {measured_stop - 1}"
            )
        items = []
        for source_idx in source_indices:
            item = data[source_idx]
            items.append(
                {
                    "id": source_idx,
                    "task": task,
                    "source_id": item["question_id"],
                    "prompt": item["question_content"],
                    "messages": build_lcb_messages(item),
                    "gold": None,
                }
            )
        return items

    raise ValueError(f"Unsupported task: {task}")


def extract_gpqa_choice(text: str) -> str:
    after = re.sub(r"^.*?</think>\s*", "", text, count=1, flags=re.DOTALL)
    if after == text:
        after = text[-500:]
    patterns = [
        r"ANSWER:\s*([A-Da-d])",
        r"[Aa]nswer\s*(?:is\s*:?\s*|:\s*)\*{0,2}\s*\(?([A-Da-d])\)?\*{0,2}",
        r"\\boxed\{[^}]*?([A-Da-d])[^A-Da-d}]*\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, after)
        if match:
            return match.group(1).upper()
    match = re.search(r"(?:\*\*\(?([A-Da-d])\)?\*\*|\(([A-Da-d])\))\s*[.\s]*$", after.strip())
    if match:
        return (match.group(1) or match.group(2)).upper()
    return "?"


async def gen_one(
    session: aiohttp.ClientSession,
    idx: int,
    prompt: str,
    url: str,
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any], dict[str, Any], float]:
    sampling_params: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": True,
    }
    if args.presence_penalty is not None:
        sampling_params["presence_penalty"] = args.presence_penalty
    body = {"text": prompt, "sampling_params": sampling_params}
    t0 = time.time()
    async with session.post(url, json=body) as resp:
        data = await resp.json()
        if resp.status >= 400:
            raise RuntimeError(f"request {idx} failed: status={resp.status}, body={data}")
    elapsed = time.time() - t0
    return idx, body, data, elapsed


async def run_batch(
    items: list[dict[str, Any]],
    prompts: dict[int, str],
    urls: list[str],
    args: argparse.Namespace,
    label: str,
) -> tuple[dict[int, tuple[dict[str, Any], dict[str, Any], float]], float]:
    results: dict[int, tuple[dict[str, Any], dict[str, Any], float]] = {}

    async def run_one(local_idx: int, item: dict[str, Any]) -> None:
        url = urls[local_idx % len(urls)]
        idx, body, data, elapsed = await gen_one(
            session, item["id"], prompts[item["id"]], url, args
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


async def main(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    all_items = load_items(
        args.task,
        measured_limit=args.limit,
        warmup=args.warmup,
        sample_offset=args.sample_offset,
    )
    prompts = {
        item["id"]: apply_chat_template(
            tokenizer,
            item["prompt"],
            args.enable_thinking,
            item.get("messages"),
        )
        for item in all_items
    }
    prompt_token_counts = {
        item["id"]: len(tokenizer.encode(prompts[item["id"]], add_special_tokens=False))
        for item in all_items
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
    latencies: list[float] = []
    completion_tokens: list[int] = []
    finish_reasons: list[str | None] = []
    samples = []
    diagnostic_correct = 0

    for item in measured_items:
        body, data, elapsed = results[item["id"]]
        text = data.get("text", "")
        meta = data.get("meta_info", {})
        tokens = meta.get("completion_tokens", 0) if isinstance(meta, dict) else 0
        finish = (meta.get("finish_reason") or {}).get("type") if isinstance(meta, dict) else None
        total_completion_tokens += tokens
        completion_tokens.append(tokens)
        latencies.append(elapsed)
        finish_reasons.append(finish)

        diagnostic_pred = None
        if item["task"] == "gpqa_diamond":
            diagnostic_pred = extract_gpqa_choice(text)
            diagnostic_correct += int(diagnostic_pred == item["gold"])

        samples.append(
            {
                "id": item["id"],
                "question": item["prompt"],
                "task": item["task"],
                "source_id": item["source_id"],
                "gold": item["gold"],
                "prompt_tokens": prompt_token_counts[item["id"]],
                "request_body": body,
                "generation": text,
                "meta_info": meta,
                "elapsed": elapsed,
                "strict_pred": diagnostic_pred,
                "flex_pred": diagnostic_pred,
                "diagnostic_pred": diagnostic_pred,
            }
        )

    yaml_config = None
    if args.algorithm_config:
        with open(args.algorithm_config) as f:
            yaml_config = yaml.safe_load(f)

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
        {"id": item["id"], "completion_tokens": tokens}
        for item, tokens in zip(measured_items, completion_tokens)
        if tokens != args.max_new_tokens
    ]
    if unexpected_lengths:
        protocol_errors.append(
            f"{len(unexpected_lengths)} measured requests did not return exactly "
            f"{args.max_new_tokens} completion tokens"
        )
    unexpected_finishes = [
        {"id": item["id"], "finish_reason": reason}
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
        "task": args.task,
        "N": len(measured_items),
        "warmup": args.warmup,
        "sample_offset": args.sample_offset,
        "strict_correct": diagnostic_correct if args.task in {"gpqa", "gpqa_diamond"} else None,
        "flex_correct": diagnostic_correct if args.task in {"gpqa", "gpqa_diamond"} else None,
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
            "presence_penalty": args.presence_penalty,
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": True,
            "stop": None,
        },
        "enable_thinking": args.enable_thinking,
        "algorithm_config_path": args.algorithm_config,
        "algorithm_config": yaml_config,
        "model_path": args.model_path,
        "ports": args.ports,
        "latency_seconds": {
            "mean": statistics.mean(latencies) if latencies else None,
            "p50": percentile(latencies, 50),
            "p90": percentile(latencies, 90),
            "p95": percentile(latencies, 95),
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "completion_tokens": {
            "mean": statistics.mean(completion_tokens) if completion_tokens else None,
            "p50": percentile(completion_tokens, 50),
            "p90": percentile(completion_tokens, 90),
            "p95": percentile(completion_tokens, 95),
            "min": min(completion_tokens) if completion_tokens else None,
            "max": max(completion_tokens) if completion_tokens else None,
        },
        "finish_reason_counts": finish_counts,
        "protocol_valid": not protocol_errors,
        "protocol_errors": protocol_errors,
        "unexpected_completion_lengths": unexpected_lengths,
        "unexpected_finish_reasons": unexpected_finishes,
        "dllm_tpf": dllm_tpf,
        "diagnostic_correct": diagnostic_correct if args.task in {"gpqa", "gpqa_diamond"} else None,
        "samples": samples,
    }

    print("\n=== FIXED OUTPUT TASK RESULTS ===")
    print(f"task: {args.task}")
    print(f"N: {result['N']}, warmup: {args.warmup}")
    print(f"wall clock: {wall:.1f}s")
    print(f"total completion tokens: {total_completion_tokens}")
    print(f"TPS: {result['aggregate_tps']:.1f}")
    print(f"finish reasons: {finish_counts}")
    print(f"latency seconds: {result['latency_seconds']}")
    print(f"completion tokens: {result['completion_tokens']}")
    if dllm_tpf is not None:
        print(
            "dLLM TPF: "
            f"decode={dllm_tpf['decode_tpf']}, "
            f"verify={dllm_tpf['verify_tpf']}, "
            f"avg_accepted={dllm_tpf['average_accepted_draft_tokens']}"
        )
    if args.task in {"gpqa", "gpqa_diamond"}:
        print(f"diagnostic GPQA exact choice: {diagnostic_correct}/{result['N']}")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
        print(f"saved: {args.save}")
    if protocol_errors:
        raise RuntimeError("fixed-output protocol validation failed: " + "; ".join(protocol_errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["humaneval", "gpqa", "gpqa_diamond", "lcb_v6"],
        required=True,
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=True)
    parser.add_argument("--ports", type=str, default="30000")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--algorithm-config", type=str, default=None)
    parser.add_argument("--require-dllm-stats", action="store_true")
    parser.add_argument("--require-verify-stats", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    asyncio.run(main(parser.parse_args()))
