"""Pure validation and construction helpers for dLLM paged attention masks."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch


DLLM_BIDIR_MASK_BACKENDS = frozenset(("auto", "native", "custom", "full"))
DLLM_SELECTED_MASK_BACKENDS = frozenset(
    ("native_structured", "custom_paged", "full_paged")
)


def validate_bidir_mask_backend(value: str) -> str:
    if value not in DLLM_BIDIR_MASK_BACKENDS:
        allowed = ", ".join(sorted(DLLM_BIDIR_MASK_BACKENDS))
        raise ValueError(
            f"invalid dllm_bidir_mask_backend={value!r}; expected one of {allowed}"
        )
    return value


def validate_selected_mask_backend(value: str) -> str:
    if value not in DLLM_SELECTED_MASK_BACKENDS:
        allowed = ", ".join(sorted(DLLM_SELECTED_MASK_BACKENDS))
        raise ValueError(
            f"invalid dllm_selected_mask_backend={value!r}; expected one of {allowed}"
        )
    return value


def validate_bidir_block_mask(
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_mask: torch.Tensor,
) -> list[tuple[int, int, int]]:
    """Validate one active-region mask against every paged request shape."""
    if not torch.is_tensor(block_mask):
        raise TypeError("dLLM bidirectional block mask must be a tensor")
    if block_mask.dtype is not torch.bool:
        raise TypeError("dLLM bidirectional block mask must have dtype bool")
    if block_mask.ndim != 2:
        raise ValueError("dLLM bidirectional block mask must be 2-D")
    if seq_lens.ndim != 1 or prefix_lens.ndim != 1:
        raise ValueError("sequence and prefix lengths must be 1-D")
    if len(seq_lens) != len(prefix_lens):
        raise ValueError("sequence and prefix length counts differ")

    shapes: list[tuple[int, int, int]] = []
    for request_index in range(len(seq_lens)):
        seq_len = int(seq_lens[request_index].item())
        prefix_len = int(prefix_lens[request_index].item())
        query_len = seq_len - prefix_len
        if seq_len <= 0:
            raise ValueError(
                f"request {request_index} sequence length must be positive"
            )
        if prefix_len < 0 or prefix_len >= seq_len:
            raise ValueError(
                f"request {request_index} has invalid prefix length {prefix_len} "
                f"for sequence length {seq_len}"
            )
        if query_len <= 0:
            raise ValueError(f"request {request_index} query length must be positive")
        if block_mask.shape[0] < query_len:
            raise ValueError(
                f"dLLM block-mask rows {block_mask.shape[0]} do not cover "
                f"request {request_index} query length {query_len}"
            )
        active_suffix_len = seq_len - prefix_len
        if block_mask.shape[1] < active_suffix_len:
            raise ValueError(
                f"dLLM block-mask columns {block_mask.shape[1]} do not cover "
                f"request {request_index} active suffix length {active_suffix_len}"
            )
        shapes.append((seq_len, prefix_len, query_len))
    if not shapes:
        raise ValueError("dLLM bidirectional mask batch is empty")
    return shapes


def select_bidir_mask_backend(
    requested: str,
    *,
    native_available: bool,
    configured_block_size: int,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_mask: torch.Tensor,
    structured_mask: torch.Tensor,
) -> str:
    """Select an exact runtime backend without broadening the supplied mask."""
    requested = validate_bidir_mask_backend(requested)
    shapes = validate_bidir_block_mask(seq_lens, prefix_lens, block_mask)
    native_lengths_match = all(
        query_len == configured_block_size for _, _, query_len in shapes
    )
    native_mask_matches = (
        structured_mask is not None
        and tuple(structured_mask.shape)
        == (configured_block_size, configured_block_size)
        and block_mask.shape[0] >= configured_block_size
        and block_mask.shape[1] >= configured_block_size
        and torch.equal(
            block_mask[:configured_block_size, :configured_block_size],
            structured_mask.to(device=block_mask.device),
        )
    )
    native_compatible = (
        native_available and native_lengths_match and native_mask_matches
    )
    full_compatible = all(
        bool(block_mask[:query_len, : seq_len - prefix_len].all().item())
        for seq_len, prefix_len, query_len in shapes
    )

    if requested == "custom":
        return "custom_paged"
    if requested == "full":
        if not full_compatible:
            raise ValueError(
                "full dLLM paged attention requires every active-mask entry "
                "in block_mask[:query_len, :active_suffix_len] to be True"
            )
        return "full_paged"
    if requested == "native":
        if not native_available:
            raise RuntimeError("native dLLM bidirectional mask support is unavailable")
        if not native_lengths_match:
            actual = [query_len for _, _, query_len in shapes]
            raise RuntimeError(
                "native dLLM bidirectional mask requires every active query "
                f"length to equal configured block_size={configured_block_size}; "
                f"got {actual}"
            )
        if not native_mask_matches:
            raise RuntimeError(
                "supplied dLLM block mask does not match the configured native "
                "structured mask"
            )
        return "native_structured"
    if native_compatible:
        return "native_structured"
    if full_compatible:
        return "full_paged"
    return "custom_paged"


def paged_mask_planner_arguments(
    selected_backend: str,
    custom_mask: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], bool, bool]:
    """Return planner custom-mask, native-mask, and explicit-full settings."""
    selected_backend = validate_selected_mask_backend(selected_backend)
    if selected_backend == "custom_paged":
        if custom_mask is None:
            raise ValueError("custom_paged requires a materialized custom mask")
        return custom_mask, False, False
    if custom_mask is not None:
        raise ValueError(f"{selected_backend} must not receive a custom mask")
    if selected_backend == "native_structured":
        return None, True, False
    return None, False, True


def paged_attention_is_causal(
    *,
    force_causal: bool,
    selected_backend: Optional[str],
    force_noncausal_full_attention: bool,
    is_cross_attention: bool,
) -> bool:
    """Resolve paged execution causality from selected semantics."""
    if force_causal:
        return True
    if force_noncausal_full_attention:
        if selected_backend != "full_paged":
            raise ValueError(
                "explicit noncausal full attention requires selected backend "
                "full_paged"
            )
        return False
    if selected_backend is not None:
        validate_selected_mask_backend(selected_backend)
        if selected_backend == "full_paged":
            raise ValueError(
                "full_paged is missing explicit noncausal full-attention metadata"
            )
        return False
    return not is_cross_attention


def invoke_paged_attention_plan(
    planner: Callable[..., Any],
    *planner_args: Any,
    selected_backend: str,
    custom_mask: Optional[torch.Tensor],
    **planner_kwargs: Any,
) -> tuple[Optional[torch.Tensor], bool, bool]:
    """Invoke a paged planner with arguments fixed by the selected contract."""
    planned_mask, native_mask, force_full = paged_mask_planner_arguments(
        selected_backend, custom_mask
    )
    planner(
        *planner_args,
        custom_mask=planned_mask,
        dllm_native_bidir_mask=native_mask,
        **planner_kwargs,
    )
    return planned_mask, native_mask, force_full


def invoke_paged_attention_forward(
    forward: Callable[..., Any],
    *forward_args: Any,
    force_causal: bool,
    selected_backend: Optional[str],
    force_noncausal_full_attention: bool,
    is_cross_attention: bool,
    **forward_kwargs: Any,
) -> Any:
    """Invoke paged attention with causality fixed by selected semantics."""
    causal = paged_attention_is_causal(
        force_causal=force_causal,
        selected_backend=selected_backend,
        force_noncausal_full_attention=force_noncausal_full_attention,
        is_cross_attention=is_cross_attention,
    )
    return forward(*forward_args, causal=causal, **forward_kwargs)


def build_paged_custom_mask(
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_mask: torch.Tensor,
    mask_types: Optional[torch.Tensor],
    *,
    bidir_mask_type: int,
) -> torch.Tensor:
    """Build flattened ``query_len x seq_len`` masks for paged prefill."""
    shapes = validate_bidir_block_mask(seq_lens, prefix_lens, block_mask)
    device = seq_lens.device
    block_mask = block_mask.to(device=device, non_blocking=True)
    if mask_types is not None:
        if mask_types.ndim != 1 or len(mask_types) != len(shapes):
            raise ValueError("dLLM attention-mask type count differs from batch size")
        mask_types = mask_types.to(device=device, dtype=torch.int32, non_blocking=True)

    total_numel = sum(query_len * seq_len for seq_len, _, query_len in shapes)
    custom_mask = torch.empty((total_numel,), dtype=torch.bool, device=device)
    offset = 0
    for request_index, (seq_len, prefix_len, query_len) in enumerate(shapes):
        request_numel = query_len * seq_len
        request_mask = custom_mask[offset : offset + request_numel].view(
            query_len, seq_len
        )
        request_mask[:, :prefix_len] = True
        suffix_len = seq_len - prefix_len
        mask_type = (
            int(mask_types[request_index].item())
            if mask_types is not None
            else bidir_mask_type
        )
        if mask_type == bidir_mask_type:
            request_mask[:, prefix_len:] = block_mask[:query_len, :suffix_len]
        else:
            rows = torch.arange(query_len, device=device).unsqueeze(1)
            cols = torch.arange(suffix_len, device=device).unsqueeze(0)
            request_mask[:, prefix_len:] = cols <= rows
        offset += request_numel

    if offset != total_numel or custom_mask.numel() != total_numel:
        raise RuntimeError(
            "constructed dLLM paged mask length does not match "
            "sum(query_len * seq_len)"
        )
    return custom_mask
