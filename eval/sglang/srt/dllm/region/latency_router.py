"""Deterministic, fail-closed latency routing for Region-DAG evaluation.

The router consumes request metadata and an inspectable JSON linear policy.  It
does not inspect tensors, logits, or generated answers and is not installed in
the serving scheduler.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


POLICY_SCHEMA_VERSION = 1
DEFAULT_SAFETY_MARGIN = 0.02
ROUTES = ("full_replay", "cold_handoff_build", "warm_cached_suffix")
DEFAULT_FEATURE_SCHEMA = (
    "total_tokens",
    "stable_tokens",
    "active_tokens",
    "stable_fraction",
    "diffusion_steps",
    "active_region_count",
    "batch_size",
    "active_tokens_x_diffusion_steps",
    "total_tokens_x_diffusion_steps",
    "stable_tokens_x_diffusion_steps",
    "batch_size_x_active_tokens",
)
COMPATIBILITY_FIELDS = (
    "contract_compatible",
    "version_compatible",
    "position_hash_compatible",
    "parent_versions_compatible",
)


@dataclass(frozen=True)
class LatencyRouteDecision:
    selected_route: str
    candidate_routes: tuple[str, ...]
    predicted_latency_ms: Mapping[str, float]
    safety_margin: float
    fallback_reason: Optional[str]
    policy_version: str
    model_provenance: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["candidate_routes"] = list(self.candidate_routes)
        value["predicted_latency_ms"] = dict(self.predicted_latency_ms)
        return value


def feature_values(inputs: Mapping[str, Any]) -> dict[str, float]:
    required = (
        "total_tokens",
        "stable_tokens",
        "active_tokens",
        "diffusion_steps",
        "active_region_count",
        "batch_size",
    )
    missing = [name for name in required if name not in inputs]
    if missing:
        raise ValueError(f"missing router features: {missing}")
    values = {name: float(inputs[name]) for name in required}
    if "stable_fraction" in inputs:
        values["stable_fraction"] = float(inputs["stable_fraction"])
    elif values["total_tokens"] > 0:
        values["stable_fraction"] = values["stable_tokens"] / values["total_tokens"]
    else:
        raise ValueError("total_tokens must be positive")
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("router features must be finite")
    if (
        values["total_tokens"] <= 0
        or values["stable_tokens"] < 0
        or values["active_tokens"] <= 0
        or values["diffusion_steps"] <= 0
        or values["active_region_count"] <= 0
        or values["batch_size"] <= 0
        or values["stable_tokens"] + values["active_tokens"] > values["total_tokens"]
        or not 0.0 <= values["stable_fraction"] <= 1.0
    ):
        raise ValueError("router features are outside their physical domain")
    values.update(
        {
            "active_tokens_x_diffusion_steps": values["active_tokens"]
            * values["diffusion_steps"],
            "total_tokens_x_diffusion_steps": values["total_tokens"]
            * values["diffusion_steps"],
            "stable_tokens_x_diffusion_steps": values["stable_tokens"]
            * values["diffusion_steps"],
            "batch_size_x_active_tokens": values["batch_size"]
            * values["active_tokens"],
        }
    )
    return values


def router_inputs_from_normalized_case(
    case: Mapping[str, Any],
    *,
    cache_available: bool,
    cache_state: str,
    cache_constructible: bool = True,
    compatible: bool = True,
) -> dict[str, Any]:
    total = int(case["total_tokens_per_request"])
    active = sum(int(end) - int(start) for start, end in case["active_spans"])
    stable = total - active
    result = {
        "total_tokens": total,
        "stable_tokens": stable,
        "active_tokens": active,
        "stable_fraction": stable / total,
        "diffusion_steps": int(case["diffusion_steps"]),
        "active_region_count": len(case["active_spans"]),
        "batch_size": int(case["batch_size"]),
        "cache_available": bool(cache_available),
        "cache_state": str(cache_state),
        "cache_constructible": bool(cache_constructible),
    }
    result.update({field: bool(compatible) for field in COMPATIBILITY_FIELDS})
    return result


class ConservativeLatencyRouter:
    """Evaluate one versioned linear policy with conservative route gating."""

    def __init__(
        self,
        policy: Optional[Mapping[str, Any]],
        *,
        unavailable_reason: Optional[str] = None,
    ) -> None:
        self.policy = dict(policy) if policy is not None else None
        self.unavailable_reason = unavailable_reason
        self._policy_error = self._validate_policy()

    @classmethod
    def from_path(cls, path: Optional[Path]) -> "ConservativeLatencyRouter":
        if path is None:
            return cls(None, unavailable_reason="policy_missing")
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls(None, unavailable_reason="policy_missing")
        except (OSError, json.JSONDecodeError):
            return cls(None, unavailable_reason="policy_unreadable")
        if not isinstance(value, Mapping):
            return cls(None, unavailable_reason="policy_not_an_object")
        return cls(value)

    def _validate_policy(self) -> Optional[str]:
        if self.policy is None:
            return self.unavailable_reason or "policy_missing"
        if self.policy.get("schema_version") != POLICY_SCHEMA_VERSION:
            return "unsupported_policy_schema"
        features = self.policy.get("feature_schema")
        if (
            not isinstance(features, Sequence)
            or isinstance(features, (str, bytes))
            or not features
            or any(not isinstance(feature, str) or not feature for feature in features)
            or len(set(features)) != len(features)
        ):
            return "invalid_feature_schema"
        models = self.policy.get("route_models")
        if not isinstance(models, Mapping) or "full_replay" not in models:
            return "missing_full_replay_model"
        domain = self.policy.get("support_domain")
        if not isinstance(domain, Mapping):
            return "missing_support_domain"
        margin = self.policy.get("safety_margin", DEFAULT_SAFETY_MARGIN)
        if (
            isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(float(margin))
            or not 0.0 <= float(margin) < 1.0
        ):
            return "invalid_safety_margin"
        for feature in features:
            bounds = domain.get(feature)
            if (
                not isinstance(bounds, Sequence)
                or isinstance(bounds, (str, bytes))
                or len(bounds) != 2
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in bounds
                )
                or float(bounds[0]) > float(bounds[1])
            ):
                return "invalid_support_domain"
        for route, model in models.items():
            if route not in ROUTES or not isinstance(model, Mapping):
                return "invalid_route_model"
            coefficients = model.get("coefficients")
            means = model.get("feature_means")
            scales = model.get("feature_scales")
            if not all(
                isinstance(values, Sequence)
                and not isinstance(values, (str, bytes))
                and len(values) == len(features)
                for values in (coefficients, means, scales)
            ):
                return "invalid_route_model_shape"
            numeric = [model.get("intercept"), *coefficients, *means, *scales]
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in numeric
            ) or any(float(scale) <= 0 for scale in scales):
                return "invalid_route_model_values"
        return None

    def _fallback(
        self,
        reason: str,
        *,
        candidates: Sequence[str] = ("full_replay",),
        predictions: Optional[Mapping[str, float]] = None,
    ) -> LatencyRouteDecision:
        policy = self.policy or {}
        return LatencyRouteDecision(
            selected_route="full_replay",
            candidate_routes=tuple(candidates),
            predicted_latency_ms=dict(predictions or {}),
            safety_margin=float(policy.get("safety_margin", DEFAULT_SAFETY_MARGIN)),
            fallback_reason=reason,
            policy_version=str(policy.get("policy_version", "unavailable")),
            model_provenance=str(policy.get("model_provenance", "unavailable")),
        )

    def decide(self, inputs: Mapping[str, Any]) -> LatencyRouteDecision:
        if self._policy_error:
            return self._fallback(self._policy_error)
        assert self.policy is not None
        try:
            values = feature_values(inputs)
        except (TypeError, ValueError):
            return self._fallback("missing_or_nonfinite_feature")
        feature_schema = tuple(self.policy["feature_schema"])
        if any(feature not in values for feature in feature_schema):
            return self._fallback("missing_or_nonfinite_feature")
        for feature in feature_schema:
            lower, upper = self.policy["support_domain"][feature]
            if not float(lower) <= values[feature] <= float(upper):
                return self._fallback(f"out_of_domain:{feature}")

        compatible = all(inputs.get(field) is True for field in COMPATIBILITY_FIELDS)
        candidates = ["full_replay"]
        if compatible and inputs.get("cache_constructible") is True:
            candidates.append("cold_handoff_build")
        if (
            compatible
            and inputs.get("cache_available") is True
            and inputs.get("cache_state") == "warm"
        ):
            candidates.append("warm_cached_suffix")

        models = self.policy["route_models"]
        predictions: dict[str, float] = {}
        vector = [values[feature] for feature in feature_schema]
        for route in candidates:
            model = models.get(route)
            if model is None:
                if route == "full_replay":
                    return self._fallback("missing_full_replay_model")
                continue
            normalized = [
                (value - float(mean)) / float(scale)
                for value, mean, scale in zip(
                    vector, model["feature_means"], model["feature_scales"]
                )
            ]
            prediction = float(model["intercept"]) + sum(
                float(coefficient) * value
                for coefficient, value in zip(model["coefficients"], normalized)
            )
            if not math.isfinite(prediction) or prediction <= 0:
                return self._fallback(
                    "nonpositive_or_nonfinite_prediction",
                    candidates=candidates,
                    predictions=predictions,
                )
            predictions[route] = prediction
        full_latency = predictions.get("full_replay")
        if full_latency is None:
            return self._fallback("missing_full_replay_prediction")
        cached = [route for route in candidates[1:] if route in predictions]
        if not cached:
            reason = "cache_incompatible" if not compatible else "cache_unavailable"
            return self._fallback(
                reason, candidates=candidates, predictions=predictions
            )
        cached.sort(key=lambda route: (predictions[route], route))
        best = cached[0]
        margin = float(self.policy.get("safety_margin", DEFAULT_SAFETY_MARGIN))
        if len(cached) > 1 and predictions[cached[1]] <= predictions[best] * (
            1.0 + margin
        ):
            return self._fallback(
                "cached_candidates_tied_within_safety_margin",
                candidates=candidates,
                predictions=predictions,
            )
        if predictions[best] > full_latency * (1.0 - margin):
            return self._fallback(
                "cached_prediction_inside_safety_margin",
                candidates=candidates,
                predictions=predictions,
            )
        return LatencyRouteDecision(
            selected_route=best,
            candidate_routes=tuple(candidates),
            predicted_latency_ms=predictions,
            safety_margin=margin,
            fallback_reason=None,
            policy_version=str(self.policy.get("policy_version", "unknown")),
            model_provenance=str(self.policy.get("model_provenance", "unspecified")),
        )
