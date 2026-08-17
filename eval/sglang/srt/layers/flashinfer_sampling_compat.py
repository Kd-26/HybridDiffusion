from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_TOP_K_TOP_P_CALL_STYLE: Optional[str] = None
_LEGACY_RAW_SAMPLING_MODULE = None


def _is_flashinfer_abi_type_error(exc: TypeError) -> bool:
    msg = str(exc).lower()
    return "argument" in msg or "keyword" in msg or "mismatch" in msg


def flashinfer_top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    *,
    filter_apply_order: str = "joint",
    check_nan: Optional[bool] = None,
) -> torch.Tensor:
    """Call FlashInfer top-k/top-p sampling across minor ABI differences."""
    from flashinfer.sampling import top_k_top_p_sampling_from_probs

    global _TOP_K_TOP_P_CALL_STYLE

    if _TOP_K_TOP_P_CALL_STYLE == "legacy_compiled":
        return _legacy_compiled_top_k_top_p_sampling_from_probs(
            probs, top_ks, top_ps, filter_apply_order=filter_apply_order
        )

    def call_with_style(style: str) -> torch.Tensor:
        if style == "full":
            return top_k_top_p_sampling_from_probs(
                probs,
                top_ks,
                top_ps,
                filter_apply_order=filter_apply_order,
                check_nan=check_nan,
            )
        if style == "filter_only":
            return top_k_top_p_sampling_from_probs(
                probs,
                top_ks,
                top_ps,
                filter_apply_order=filter_apply_order,
            )
        return top_k_top_p_sampling_from_probs(probs, top_ks, top_ps)

    styles = (
        [_TOP_K_TOP_P_CALL_STYLE]
        if _TOP_K_TOP_P_CALL_STYLE is not None
        else ["full", "filter_only", "plain"]
    )
    if check_nan is None:
        styles = [style for style in styles if style != "full"]
        if not styles:
            styles = ["filter_only", "plain"]

    last_error: Optional[TypeError] = None
    for style in styles:
        try:
            out = call_with_style(style)
            if _TOP_K_TOP_P_CALL_STYLE != style:
                logger.info("[FlashInfer sampling] using top_k_top_p ABI style=%s", style)
                _TOP_K_TOP_P_CALL_STYLE = style
            return out
        except TypeError as exc:
            if not _is_flashinfer_abi_type_error(exc):
                raise
            last_error = exc
            if _TOP_K_TOP_P_CALL_STYLE is not None:
                _TOP_K_TOP_P_CALL_STYLE = None
                return flashinfer_top_k_top_p_sampling_from_probs(
                    probs,
                    top_ks,
                    top_ps,
                    filter_apply_order=filter_apply_order,
                    check_nan=check_nan,
                )

    try:
        out = _legacy_compiled_top_k_top_p_sampling_from_probs(
            probs, top_ks, top_ps, filter_apply_order=filter_apply_order
        )
        if _TOP_K_TOP_P_CALL_STYLE != "legacy_compiled":
            logger.info(
                "[FlashInfer sampling] using legacy compiled top_k_top_p ABI"
            )
            _TOP_K_TOP_P_CALL_STYLE = "legacy_compiled"
        return out
    except TypeError:
        assert last_error is not None
        raise last_error


def _legacy_compiled_top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    *,
    filter_apply_order: str,
) -> torch.Tensor:
    if filter_apply_order != "joint":
        raise TypeError("legacy compiled FlashInfer fallback only supports joint order")

    import flashinfer.sampling as flashinfer_sampling

    global _LEGACY_RAW_SAMPLING_MODULE
    if _LEGACY_RAW_SAMPLING_MODULE is None:
        _LEGACY_RAW_SAMPLING_MODULE = (
            flashinfer_sampling.gen_sampling_module().build_and_load()
        )
    module = _LEGACY_RAW_SAMPLING_MODULE
    probs = probs.float()
    top_ks = top_ks.int() if isinstance(top_ks, torch.Tensor) else top_ks
    top_ps = top_ps.float() if isinstance(top_ps, torch.Tensor) else top_ps
    maybe_top_k_arr, top_k_val = flashinfer_sampling._to_tensor_scalar_tuple(top_ks)
    maybe_top_p_arr, top_p_val = flashinfer_sampling._to_tensor_scalar_tuple(top_ps)
    batch_size = probs.size(0)
    samples = torch.empty(batch_size, dtype=torch.int32, device=probs.device)
    seed, offset = flashinfer_sampling.get_seed_and_offset(
        batch_size * 32, None, probs.device
    )
    module.top_k_top_p_sampling_from_probs(
        probs,
        samples,
        None,
        maybe_top_k_arr,
        int(top_k_val),
        maybe_top_p_arr,
        float(top_p_val),
        True,
        seed,
        offset,
    )
    return samples
