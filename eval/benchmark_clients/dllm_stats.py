"""Helpers for collecting measured dLLM tokens-per-forward statistics."""

from __future__ import annotations

import numbers
from typing import Any

import aiohttp


_DERIVED_COUNTERS = {"tpf", "decode_tpf"}


def _raw_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in stats.items()
        if key not in _DERIVED_COUNTERS
        and (
            isinstance(value, numbers.Number)
            or (
                isinstance(value, list)
                and all(isinstance(item, numbers.Number) for item in value)
            )
        )
    }


def _add_stats(total: dict[str, Any], stats: dict[str, Any]) -> None:
    for key, value in _raw_stats(stats).items():
        if isinstance(value, list):
            current = total.setdefault(key, [0] * len(value))
            if len(current) < len(value):
                current.extend([0] * (len(value) - len(current)))
            for index, item in enumerate(value):
                current[index] += item
        else:
            total[key] = total.get(key, 0) + value


async def fetch_dllm_stats(
    ports: list[int],
    *,
    timeout: float = 120.0,
) -> dict[str, Any] | None:
    """Fetch and aggregate one dLLM counter snapshot per server endpoint."""

    aggregate: dict[str, Any] = {}
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        for port in ports:
            async with session.get(f"http://localhost:{port}/server_info") as response:
                payload = await response.json()
                if response.status >= 400:
                    raise RuntimeError(
                        f"server_info failed on port {port}: "
                        f"status={response.status}, body={payload}"
                    )
            if "decode" in payload:
                decode_states = payload.get("decode") or []
                payload = decode_states[0] if decode_states else {}
            internal_states = payload.get("internal_states") or []
            if not internal_states:
                return None
            stats = internal_states[0].get("dllm_stats")
            if not isinstance(stats, dict):
                return None
            _add_stats(aggregate, stats)
    return aggregate


def subtract_stats(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Subtract two cumulative counter snapshots."""

    delta: dict[str, Any] = {}
    for key in set(before) | set(after):
        before_value = before.get(key, 0)
        after_value = after.get(key, 0)
        if isinstance(after_value, list) or isinstance(before_value, list):
            before_list = before_value if isinstance(before_value, list) else []
            after_list = after_value if isinstance(after_value, list) else []
            size = max(len(before_list), len(after_list))
            delta[key] = [
                (after_list[index] if index < len(after_list) else 0)
                - (before_list[index] if index < len(before_list) else 0)
                for index in range(size)
            ]
        elif isinstance(after_value, numbers.Number) and isinstance(
            before_value, numbers.Number
        ):
            delta[key] = after_value - before_value
    return delta


def build_tpf_report(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    *,
    measured_completion_tokens: int,
) -> dict[str, Any] | None:
    """Build exact measured TPF plus AR-Trust verification statistics."""

    if before is None or after is None:
        return None

    delta = subtract_stats(before, after)
    decode_forwards = int(delta.get("decode_forwards", 0))
    prefill_forwards = int(delta.get("prefill_forwards", 0))
    verify_decisions = int(delta.get("verify_decisions", 0))
    accepted_spec_tokens = int(delta.get("accepted_spec_tokens", 0))

    decode_tpf = (
        measured_completion_tokens / decode_forwards
        if decode_forwards > 0
        else None
    )
    avg_accepted = (
        accepted_spec_tokens / verify_decisions
        if verify_decisions > 0
        else None
    )

    return {
        "scope": "measured_requests_after_warmup",
        "before": before,
        "after": after,
        "delta": delta,
        "measured_completion_tokens": measured_completion_tokens,
        "prefill_forwards": prefill_forwards,
        "decode_forwards": decode_forwards,
        "decode_tpf": decode_tpf,
        "verify_decisions": verify_decisions,
        "accepted_spec_tokens": accepted_spec_tokens,
        "average_accepted_draft_tokens": avg_accepted,
        "verify_tpf": avg_accepted + 1.0 if avg_accepted is not None else None,
        "accept_token_histogram": delta.get("accept_token_hist"),
        "definitions": {
            "decode_tpf": "measured_completion_tokens / decode_forwards",
            "verify_tpf": "accepted_spec_tokens / verify_decisions + 1",
        },
    }
