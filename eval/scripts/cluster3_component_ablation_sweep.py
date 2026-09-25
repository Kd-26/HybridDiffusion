#!/usr/bin/env python3
"""Launch isolated Cluster-3 component-ablation jobs and preserve every result."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "eval/scripts/cluster3_region_dag_validation.py"
GRID = ROOT / "eval/manifests/cluster3_component_ablation_v1.json"
FIXED_POLICIES = (
    "full_replay",
    "kv_only",
    "gdn_only",
    "kv_gdn_conservative",
    "kv_gdn_oracle",
)
SMOKE_CASE_ID = "p256-a16-s2-b1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("smoke", "sweep"))
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--correctness-artifact", required=True, type=Path)
    parser.add_argument("--preflight-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smoke-summary", type=Path)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--seed", default=20260825, type=int)
    return parser


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _grid_cases() -> list[Mapping[str, Any]]:
    value = _read_json(GRID)
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != 12:
        raise RuntimeError("component-ablation grid must contain exactly 12 cases")
    return cases


def _capacity(case: Mapping[str, Any]) -> int:
    prefix = int(case["prefix_tokens"])
    return 16384 if prefix >= 4096 else 8192


def _write_case_manifest(path: Path, case: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "cases": [dict(case)]}, indent=2) + "\n",
        encoding="utf-8",
    )


def _measurement_records(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _job_status(
    *,
    policy: str,
    returncode: int,
    summary_path: Path,
    records_path: Path,
) -> dict[str, Any]:
    summary = _read_json(summary_path) if summary_path.exists() else {}
    records = _measurement_records(records_path)
    policy_records = [
        record for record in records if record.get("ablation_policy") == policy
    ]
    hashes = {str(record.get("output_hash", "")) for record in policy_records}
    tokens = {
        tuple(int(value) for value in record.get("generated_top1_token_ids", ()))
        for record in policy_records
    }
    completed = returncode == 0 and bool(policy_records)
    unsupported = bool(summary.get("unsupported_policies"))
    correctness = bool(
        completed
        and len(hashes) == 1
        and len(tokens) == 1
        and all(
            int(record.get("fallback_count", -1)) == 0
            and int(record.get("recovery_replay_count", -1)) == 0
            for record in policy_records
        )
    )
    return {
        "policy": policy,
        "returncode": returncode,
        "status": (
            "completed" if completed else "unsupported" if unsupported else "failed"
        ),
        "correctness_pass": correctness,
        "summary_json": str(summary_path),
        "records_jsonl": str(records_path),
        "output_hashes": sorted(hashes),
    }


def _run_job(args: argparse.Namespace, case: Mapping[str, Any], policy: str):
    case_id = str(case["case_id"])
    job_dir = args.output_dir / "jobs" / case_id / policy
    job_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = job_dir / "case.json"
    records_path = job_dir / "measurements.jsonl"
    summary_path = job_dir / "summary.json"
    log_path = job_dir / "process.log"
    _write_case_manifest(manifest_path, case)
    command = [
        sys.executable,
        str(BENCHMARK),
        "--model-path",
        str(args.model_path),
        "--output-jsonl",
        str(records_path),
        "--summary-json",
        str(summary_path),
        "--profile",
        "production_efficiency",
        "--case-manifest",
        str(manifest_path),
        "--ablation-policy",
        policy,
        "--correctness-artifact",
        str(args.correctness_artifact),
        "--preflight-json",
        str(args.preflight_json),
        "--max-total-tokens",
        str(_capacity(case)),
        "--warmups",
        "3",
        "--timed-repetitions",
        "10",
        "--tp-size",
        "1",
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
    ]
    (job_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    with log_path.open("w", encoding="utf-8") as log:
        environment = os.environ.copy()
        python_path = [str(ROOT / "eval")]
        if environment.get("PYTHONPATH"):
            python_path.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_path)
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return _job_status(
        policy=policy,
        returncode=completed.returncode,
        summary_path=summary_path,
        records_path=records_path,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = _grid_cases()
    if args.mode == "smoke":
        cases = [case for case in cases if case["case_id"] == SMOKE_CASE_ID]
        if len(cases) != 1:
            raise RuntimeError("smoke case is missing from the fixed grid")
    else:
        if args.smoke_summary is None:
            raise RuntimeError("sweep mode requires --smoke-summary")
        smoke = _read_json(args.smoke_summary)
        if smoke.get("smoke_gate_pass") is not True:
            raise RuntimeError(
                "refusing the 60-job sweep because the component smoke gate did not pass"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [_run_job(args, case, policy) for case in cases for policy in FIXED_POLICIES]
    completed = []
    failed = []
    unsupported = []
    for policy in FIXED_POLICIES:
        policy_jobs = [job for job in jobs if job["policy"] == policy]
        statuses = {job["status"] for job in policy_jobs}
        if len(policy_jobs) == len(cases) and statuses == {"completed"}:
            completed.append(policy)
        elif "unsupported" in statuses:
            unsupported.append(policy)
        else:
            failed.append(policy)
    all_complete = len(jobs) == len(cases) * len(FIXED_POLICIES) and all(
        job["status"] == "completed" for job in jobs
    )
    correctness = all_complete and all(job["correctness_pass"] for job in jobs)
    result = {
        "schema_version": 1,
        "artifact_type": "cluster3_component_ablation_isolated_sweep",
        "mode": args.mode,
        "case_count": len(cases),
        "job_count": len(jobs),
        "completed_policies": completed,
        "failed_policies": failed,
        "unsupported_policies": unsupported,
        "correctness_pass": correctness,
        "publication_gate_pass": correctness,
        "all_publication_gates_pass": correctness,
        "smoke_gate_pass": correctness if args.mode == "smoke" else None,
        "jobs": jobs,
    }
    destination = args.output_dir / f"{args.mode}_summary.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_publication_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
