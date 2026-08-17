#!/usr/bin/env python3
"""Run and compare causal/self-spec correctness probes against SGLang servers.

This harness intentionally does not launch servers. Run it once against a
causal server and once against a self-spec server using the same checkpoint,
then compare the two JSON outputs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests


PROMPTS = [
    "Question: If Janet has 16 eggs and uses 7, how many are left? Answer briefly.",
    "Question: A train travels 60 miles per hour for 3 hours. How far does it go?",
    "Question: What is the capital of France? Answer in one sentence.",
    "Question: Write a short sentence about machine learning.",
    "Question: Compute 17 plus 25. Answer briefly.",
    "Question: Name three colors in a comma-separated list.",
    "Question: Explain gravity in one short sentence.",
    "Question: What is 9 times 8? Answer briefly.",
    "Question: Write a Python function name for adding numbers.",
    "Question: Give one reason exercise is healthy.",
    "Question: What is the opposite of hot?",
    "Question: Count from one to five.",
    "Question: Mention one ocean.",
    "Question: What planet do humans live on?",
    "Question: Translate hello to Spanish.",
    "Question: Say a polite greeting.",
]

LONG_PROMPT = (
    "Read the short story and answer the final question. "
    "Mira packed three notebooks, two pencils, and a small lunch before walking "
    "to the library. At the library she met Omar, who had already reserved a "
    "quiet table near the window. They studied for exactly two hours, took a "
    "short break, and then reviewed their notes one more time. "
    "Question: Where did Mira and Omar study?"
)


def build_cases(batch_sizes: list[int]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for bs in batch_sizes:
        cases.append(
            {
                "name": f"short_unique_bs{bs}",
                "batch_size": bs,
                "prompts": PROMPTS[:bs],
            }
        )
    cases.extend(
        [
            {
                "name": "repeated_bs8",
                "batch_size": 8,
                "prompts": [PROMPTS[0]] * 8,
            },
            {
                "name": "mixed_lengths_bs4",
                "batch_size": 4,
                "prompts": [PROMPTS[0], LONG_PROMPT, PROMPTS[3], LONG_PROMPT],
            },
        ]
    )
    return cases


def post_generate(
    base_url: str,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    timeout: int,
) -> tuple[list[dict[str, Any]], float]:
    payload = {
        "text": prompts,
        "sampling_params": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "return_logprob": False,
    }
    start = time.perf_counter()
    response = requests.post(
        f"{base_url.rstrip('/')}/generate",
        json=payload,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise TypeError(f"Expected list response, got {type(data).__name__}")
    return data, elapsed


def run(args: argparse.Namespace) -> None:
    cases = build_cases(args.batch_sizes)
    result: dict[str, Any] = {
        "mode": args.mode,
        "base_url": args.base_url,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repeat": args.repeat,
        "cases": [],
    }

    for case in cases:
        samples = []
        throughputs = []
        for rep in range(args.repeat + args.warmup):
            data, elapsed = post_generate(
                args.base_url,
                case["prompts"],
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                args.top_p,
                args.timeout,
            )
            output_ids = [item.get("output_ids", []) for item in data]
            token_count = sum(len(ids) for ids in output_ids)
            throughput = token_count / elapsed if elapsed > 0 else 0.0
            sample = {
                "rep": rep - args.warmup,
                "elapsed_s": elapsed,
                "output_tokens": token_count,
                "throughput_tok_s": throughput,
                "output_ids": output_ids,
                "texts": [item.get("text", "") for item in data],
            }
            if rep >= args.warmup:
                samples.append(sample)
                throughputs.append(throughput)
            print(
                f"{args.mode} {case['name']} rep={rep} "
                f"tokens={token_count} elapsed={elapsed:.4f}s "
                f"tps={throughput:.1f}",
                flush=True,
            )
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)

        case_result = dict(case)
        case_result["samples"] = samples
        case_result["throughput_mean"] = (
            statistics.mean(throughputs) if throughputs else 0.0
        )
        case_result["throughput_stdev"] = (
            statistics.pstdev(throughputs) if len(throughputs) > 1 else 0.0
        )
        result["cases"].append(case_result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


def first_mismatch(left: list[int], right: list[int]) -> tuple[int, int | None, int | None] | None:
    max_len = max(len(left), len(right))
    for idx in range(max_len):
        l_val = left[idx] if idx < len(left) else None
        r_val = right[idx] if idx < len(right) else None
        if l_val != r_val:
            return idx, l_val, r_val
    return None


def load_trace(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    trace_path = Path(path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    records = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def assign_trace_records(
    self_spec_cases: dict[str, Any],
    trace_records: list[dict[str, Any]],
    sample_index: int,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Map (case_name, prompt_idx, token_idx) to self-spec trace records.

    The trace is emitted in request order by the single-threaded benchmark
    harness. We consume records by batch index while walking the saved outputs
    in the same order.
    """
    queues: dict[int, list[dict[str, Any]]] = {}
    for rec in trace_records:
        queues.setdefault(int(rec.get("batch_idx", -1)), []).append(rec)

    mapping: dict[tuple[str, int, int], dict[str, Any]] = {}
    for name, case in self_spec_cases.items():
        sample = case["samples"][sample_index]
        for prompt_idx, output_ids in enumerate(sample["output_ids"]):
            queue = queues.get(prompt_idx, [])
            for token_idx, _ in enumerate(output_ids):
                if queue:
                    mapping[(name, prompt_idx, token_idx)] = queue.pop(0)
    return mapping


def load_cases(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {case["name"]: case for case in data["cases"]}


def compare(args: argparse.Namespace) -> None:
    causal_cases = load_cases(args.causal)
    self_spec_cases = load_cases(args.self_spec)
    trace_map = assign_trace_records(
        self_spec_cases,
        load_trace(args.self_spec_trace),
        args.sample_index,
    )
    failures = []
    diagnostics = []

    for name, causal_case in causal_cases.items():
        self_spec_case = self_spec_cases.get(name)
        if self_spec_case is None:
            failures.append(f"{name}: missing in self-spec results")
            continue
        causal_sample = causal_case["samples"][args.sample_index]
        self_spec_sample = self_spec_case["samples"][args.sample_index]
        for prompt_idx, (causal_ids, self_spec_ids) in enumerate(
            zip(causal_sample["output_ids"], self_spec_sample["output_ids"])
        ):
            mismatch = first_mismatch(causal_ids, self_spec_ids)
            if mismatch is not None:
                idx, causal_val, self_spec_val = mismatch
                failures.append(
                    f"{name}[{prompt_idx}] token {idx}: "
                    f"causal={causal_val} self-spec={self_spec_val}"
                )
                trace = trace_map.get((name, prompt_idx, idx))
                diagnostics.append(
                    {
                        "case": name,
                        "prompt_idx": prompt_idx,
                        "token_idx": idx,
                        "causal_token": causal_val,
                        "self_spec_token": self_spec_val,
                        "self_spec_trace": trace,
                    }
                )
        if len(causal_sample["output_ids"]) != len(
            self_spec_sample["output_ids"]
        ):
            failures.append(
                f"{name}: batch size mismatch "
                f"causal={len(causal_sample['output_ids'])} "
                f"self-spec={len(self_spec_sample['output_ids'])}"
            )

    if failures:
        print("FAILED")
        for failure in failures:
            print(failure)
        if args.diagnostics_output:
            out = Path(args.diagnostics_output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            print(f"Wrote diagnostics {out}")
        raise SystemExit(1)

    print("PASSED: all compared causal/self-spec output_ids match exactly")
    for name in causal_cases:
        causal_tps = causal_cases[name]["throughput_mean"]
        self_spec_tps = self_spec_cases[name]["throughput_mean"]
        ratio = self_spec_tps / causal_tps if causal_tps else 0.0
        print(
            f"{name}: causal={causal_tps:.1f} tok/s "
            f"self-spec={self_spec_tps:.1f} tok/s ratio={ratio:.2f}x"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--base-url", required=True)
    run_parser.add_argument(
        "--mode", required=True, choices=["causal", "self-spec"]
    )
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    run_parser.add_argument("--max-new-tokens", type=int, default=64)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--top-k", type=int, default=1)
    run_parser.add_argument("--top-p", type=float, default=1.0)
    run_parser.add_argument("--warmup", type=int, default=1)
    run_parser.add_argument("--repeat", type=int, default=3)
    run_parser.add_argument("--timeout", type=int, default=600)
    run_parser.add_argument("--sleep-s", type=float, default=0.0)
    run_parser.set_defaults(func=run)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--causal", required=True)
    compare_parser.add_argument("--self-spec", required=True)
    compare_parser.add_argument("--sample-index", type=int, default=0)
    compare_parser.add_argument("--self-spec-trace")
    compare_parser.add_argument("--diagnostics-output")
    compare_parser.set_defaults(func=compare)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
