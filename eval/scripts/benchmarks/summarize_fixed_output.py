#!/usr/bin/env python3
"""Validate and summarize a fixed-output throughput sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_MODELS = [
    "sdar-1.7b",
    "sdar-4b",
    "sdar-8b",
    "sdar-30b-a3b",
    "llada2.0-mini",
    "llada2.1-mini",
    "hybrid-diffusion-2b-causal",
    "hybrid-diffusion-2b-exact-truncated",
    "hybrid-diffusion-2b-softmax-argmax",
    "hybrid-diffusion-2b-truncated-argmax",
    "hybrid-diffusion-2b-diffusion",
    "hybrid-diffusion-4b-causal",
    "hybrid-diffusion-4b-exact-truncated",
    "hybrid-diffusion-4b-softmax-argmax",
    "hybrid-diffusion-4b-truncated-argmax",
    "hybrid-diffusion-4b-diffusion",
    "hybrid-diffusion-9b-causal",
    "hybrid-diffusion-9b-exact-truncated",
    "hybrid-diffusion-9b-softmax-argmax",
    "hybrid-diffusion-9b-truncated-argmax",
    "hybrid-diffusion-9b-diffusion",
]
DEFAULT_TASKS = ["gsm8k", "humaneval", "gpqa"]
DEFAULT_CONCURRENCIES = [1, 4, 8]
DEFAULT_LENGTHS = [2048, 16384]
DEFAULT_MEASURE_MULTIPLIER = 15
EXPECTED_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 50,
    "ignore_eos": True,
    "stop": None,
}


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def parse_strings(value: str) -> list[str]:
    return value.replace(",", " ").split()


def load_rows(
    run_root: Path,
    measure_multiplier: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted(run_root.glob("*/c*/results/*/len*/result.json")):
        relative = path.relative_to(run_root)
        model = relative.parts[0]
        concurrency = int(relative.parts[1][1:])
        task = relative.parts[3]
        output_length = int(relative.parts[4][3:])
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            errors.append(f"{relative}: cannot load JSON: {exc}")
            continue

        expected_n = measure_multiplier * concurrency
        problems = []
        if data.get("protocol_valid") is not True:
            problems.append(f"protocol_valid={data.get('protocol_valid')!r}")
        if data.get("N") != expected_n:
            problems.append(f"N={data.get('N')!r}, expected {expected_n}")
        if data.get("warmup") != concurrency:
            problems.append(
                f"warmup={data.get('warmup')!r}, expected {concurrency}"
            )
        request_schedule = data.get("request_schedule") or {}
        schedule_policy = request_schedule.get("policy")
        if concurrency == 1:
            if schedule_policy not in (None, "strict_waves"):
                problems.append(
                    f"request_schedule.policy={schedule_policy!r}, "
                    "expected strict_waves"
                )
        else:
            strict_schedule = (
                schedule_policy == "strict_waves"
                and request_schedule.get("wave_size") == concurrency
                and request_schedule.get("warmup_waves") == 1
                and request_schedule.get("measured_waves") == measure_multiplier
            )
            shard_waves = request_schedule.get("shard_measured_waves")
            sharded_schedule = (
                schedule_policy == "strict_waves_sharded"
                and request_schedule.get("wave_size") == concurrency
                and request_schedule.get("warmup_waves_per_replica") == 1
                and request_schedule.get("measured_waves") == measure_multiplier
                and isinstance(request_schedule.get("replicas"), int)
                and request_schedule["replicas"] >= 2
                and isinstance(shard_waves, list)
                and len(shard_waves) == request_schedule["replicas"]
                and sum(shard_waves) == measure_multiplier
            )
            if not strict_schedule and not sharded_schedule:
                problems.append(
                    f"request_schedule={request_schedule!r}, expected "
                    f"strict_waves or audited strict_waves_sharded at "
                    f"C={concurrency} with {measure_multiplier} measured waves"
                )
        if data.get("max_new_tokens") != output_length:
            problems.append(
                f"max_new_tokens={data.get('max_new_tokens')!r}, "
                f"expected {output_length}"
            )
        sampling = data.get("request_sampling_params") or {}
        for key, expected in EXPECTED_SAMPLING.items():
            if sampling.get(key) != expected:
                problems.append(
                    f"request_sampling_params.{key}={sampling.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if sampling.get("max_new_tokens") != output_length:
            problems.append(
                "request_sampling_params.max_new_tokens="
                f"{sampling.get('max_new_tokens')!r}, expected {output_length}"
            )
        if task != "gsm8k" and sampling.get("presence_penalty") != 1.5:
            problems.append(
                "request_sampling_params.presence_penalty="
                f"{sampling.get('presence_penalty')!r}, expected 1.5"
            )
        token_summary = data.get("completion_tokens") or {}
        if (
            token_summary.get("min") != output_length
            or token_summary.get("max") != output_length
        ):
            problems.append(f"completion_tokens={token_summary!r}")
        if problems:
            errors.append(f"{relative}: " + "; ".join(problems))

        dllm_tpf = data.get("dllm_tpf") or {}
        is_causal_hybrid_diffusion = model.startswith("hybrid-diffusion-") and model.endswith("-causal")
        decode_tpf = 1.0 if is_causal_hybrid_diffusion else dllm_tpf.get("decode_tpf")
        rows.append(
            {
                "model": model,
                "task": task,
                "output_length": output_length,
                "concurrency": concurrency,
                "measured_requests": data.get("N"),
                "warmup_requests": data.get("warmup"),
                "request_schedule": (
                    schedule_policy
                    if schedule_policy is not None
                    else "strict_waves_equivalent_c1"
                ),
                "aggregate_tps": data.get("aggregate_tps"),
                "wall_seconds": data.get("wall_seconds"),
                "total_completion_tokens": data.get("total_completion_tokens"),
                "decode_tpf": decode_tpf,
                "verify_tpf": dllm_tpf.get("verify_tpf"),
                "tpf_source": (
                    "causal_definition" if is_causal_hybrid_diffusion else "measured_counters"
                ),
                "average_accepted_draft_tokens": dllm_tpf.get(
                    "average_accepted_draft_tokens"
                ),
                "decode_forwards": dllm_tpf.get("decode_forwards"),
                "prefill_forwards": dllm_tpf.get("prefill_forwards"),
                "verify_decisions": dllm_tpf.get("verify_decisions"),
                "accept_token_histogram": json.dumps(
                    dllm_tpf.get("accept_token_histogram")
                ),
                "latency_mean_seconds": (data.get("latency_seconds") or {}).get(
                    "mean"
                ),
                "latency_p50_seconds": (data.get("latency_seconds") or {}).get(
                    "p50"
                ),
                "latency_p95_seconds": (data.get("latency_seconds") or {}).get(
                    "p95"
                ),
                "finish_reason_counts": json.dumps(
                    data.get("finish_reason_counts") or {}, sort_keys=True
                ),
                "protocol_valid": data.get("protocol_valid"),
                "result_path": str(relative),
            }
        )
    rows.sort(
        key=lambda row: (
            row["task"],
            row["output_length"],
            row["model"],
            row["concurrency"],
        )
    )
    return rows, errors


def expected_missing(
    rows: list[dict[str, Any]],
    models: list[str],
    tasks: list[str],
    concurrencies: list[int],
    lengths: list[int],
) -> list[str]:
    present = {
        (
            row["model"],
            row["task"],
            row["concurrency"],
            row["output_length"],
        )
        for row in rows
    }
    return [
        f"{model}\t{task}\tC={concurrency}\tlength={length}"
        for model in models
        for task in tasks
        for concurrency in concurrencies
        for length in lengths
        if (model, task, concurrency, length) not in present
    ]


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    errors: list[str],
    missing: list[str],
) -> None:
    lines = [
        "# Fixed-output throughput",
        "",
        "All TPS values are aggregate completion tokens divided by measured wall "
        "time. Every valid request uses `ignore_eos=true` and finishes at the "
        "configured output length.",
        "",
        f"Completed cases: **{len(rows)}**",
        "",
    ]
    for task in sorted({row["task"] for row in rows}):
        for length in sorted(
            {row["output_length"] for row in rows if row["task"] == task}
        ):
            lines.extend(
                [
                    f"## {task} — {length} output tokens",
                    "",
                    "| Model | C=1 | C=4 | C=8 |",
                    "|---|---:|---:|---:|",
                ]
            )
            models = sorted(
                {
                    row["model"]
                    for row in rows
                    if row["task"] == task and row["output_length"] == length
                }
            )
            for model in models:
                by_c = {
                    row["concurrency"]: row["aggregate_tps"]
                    for row in rows
                    if row["task"] == task
                    and row["output_length"] == length
                    and row["model"] == model
                }
                values = [
                    f"{by_c[c]:.1f}" if isinstance(by_c.get(c), (int, float)) else "—"
                    for c in (1, 4, 8)
                ]
                lines.append(f"| {model} | {' | '.join(values)} |")
            lines.append("")
            tpf_rows = [
                row
                for row in rows
                if row["task"] == task
                and row["output_length"] == length
                and row["concurrency"] == 1
                and isinstance(row.get("decode_tpf"), (int, float))
            ]
            if tpf_rows:
                lines.extend(
                    [
                        "C=1 dLLM tokens per forward:",
                        "",
                        "| Model | Decode TPF | Verify TPF | Avg. accepted drafts |",
                        "|---|---:|---:|---:|",
                    ]
                )
                for row in sorted(tpf_rows, key=lambda item: item["model"]):
                    verify_tpf = row.get("verify_tpf")
                    avg_accepted = row.get("average_accepted_draft_tokens")
                    verify_text = (
                        f"{verify_tpf:.3f}"
                        if isinstance(verify_tpf, (int, float))
                        else "—"
                    )
                    accepted_text = (
                        f"{avg_accepted:.3f}"
                        if isinstance(avg_accepted, (int, float))
                        else "—"
                    )
                    lines.append(
                        f"| {row['model']} | {row['decode_tpf']:.3f} | "
                        f"{verify_text} | {accepted_text} |"
                    )
                lines.append("")
    if errors:
        lines.extend(["## Invalid results", ""])
        lines.extend(f"- {error}" for error in errors)
        lines.append("")
    if missing:
        lines.extend(["## Missing results", ""])
        lines.extend(f"- {item}" for item in missing)
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--models", default=" ".join(DEFAULT_MODELS))
    parser.add_argument("--tasks", default=" ".join(DEFAULT_TASKS))
    parser.add_argument(
        "--concurrencies",
        default=" ".join(str(value) for value in DEFAULT_CONCURRENCIES),
    )
    parser.add_argument(
        "--lengths", default=" ".join(str(value) for value in DEFAULT_LENGTHS)
    )
    parser.add_argument(
        "--measure-multiplier",
        type=int,
        default=DEFAULT_MEASURE_MULTIPLIER,
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    models = parse_strings(args.models)
    tasks = parse_strings(args.tasks)
    concurrencies = parse_ints(args.concurrencies)
    lengths = parse_ints(args.lengths)
    rows, errors = load_rows(args.run_root, args.measure_multiplier)
    rows = [
        row
        for row in rows
        if row["model"] in models
        and row["task"] in tasks
        and row["concurrency"] in concurrencies
        and row["output_length"] in lengths
    ]
    missing = expected_missing(
        rows,
        models,
        tasks,
        concurrencies,
        lengths,
    )

    csv_path = args.run_root / "summary.csv"
    fieldnames = list(rows[0]) if rows else [
        "model",
        "task",
        "output_length",
        "concurrency",
        "aggregate_tps",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    validation_path = args.run_root / "validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "completed_cases": len(rows),
                "expected_cases": (
                    len(models)
                    * len(tasks)
                    * len(concurrencies)
                    * len(lengths)
                ),
                "invalid_results": errors,
                "missing_results": missing,
                "complete": not errors and not missing,
            },
            indent=2,
        )
        + "\n"
    )
    write_markdown(args.run_root / "summary.md", rows, errors, missing)
    print(
        f"completed={len(rows)} missing={len(missing)} invalid={len(errors)} "
        f"summary={csv_path}"
    )
    if errors or (missing and not args.allow_partial):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
