from functools import partial

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

try:
    from liger_kernel.transformers.functional import (
        liger_fused_linear_cross_entropy,
    )
except ImportError:
    liger_fused_linear_cross_entropy = None

from torchtitan.models.attention import FlexAttentionWrapper
from torchtitan.protocols.model import AttentionMasksType

from .block_gated_delta_rule import chunk_block_causal_gated_delta_rule
from .block_gated_delta_rule.convolution import block_train_conv
from fla.modules.convolution import causal_conv1d as _causal_conv1d
from .dllm_args import Qwen3_5DLLMModelArgs
from .model import (
    GatedAttention,
    GatedDeltaNet,
    Qwen3_5Model,
    TransformerBlock,
    _l2norm,
    _to_local_if_dtensor,
    apply_partial_rotary_emb,
)


# ---------------------------------------------------------------------------
# Attention mask for the [x0; xt] two-stream layout
# ---------------------------------------------------------------------------

def _block_diff_mask_x0_xt(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    *,
    block_size: int,
    n: int,
    causal_x0: bool = False,
    causal_xt: bool = False,
    doc_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Two-stream DLLM mask for `[x0, xt]` (clean first, noisy second).

    Three components:
      1. x0 self-attention (first half): block-causal or token-causal
      2. xt self-attention (second half): bidirectional or causal within blocks
      3. xt cross-attends to earlier x0 blocks (offset block causal)
    """
    q_is_xt = q_idx >= n
    kv_is_xt = kv_idx >= n

    q_pos = torch.where(q_is_xt, q_idx - n, q_idx)
    kv_pos = torch.where(kv_is_xt, kv_idx - n, kv_idx)
    q_block = q_pos // block_size
    kv_block = kv_pos // block_size

    if causal_x0:
        x0_mask = (~q_is_xt) & (~kv_is_xt) & (q_pos >= kv_pos)
    else:
        x0_mask = (~q_is_xt) & (~kv_is_xt) & (q_block >= kv_block)

    if causal_xt:
        xt_mask = q_is_xt & kv_is_xt & (q_block == kv_block) & (q_pos >= kv_pos)
    else:
        xt_mask = q_is_xt & kv_is_xt & (q_block == kv_block)

    cross_mask = q_is_xt & (~kv_is_xt) & (q_block > kv_block)

    mask = x0_mask | xt_mask | cross_mask

    if doc_ids is not None:
        same_doc = doc_ids[b, q_idx] == doc_ids[b, kv_idx]
        mask = mask & same_doc

    return mask


def create_block_diff_attention_mask(
    seq_len: int,
    block_size: int,
    device: torch.device,
    causal_x0: bool = False,
    causal_xt: bool = False,
    doc_ids: torch.Tensor | None = None,
    attention_block_size: int | tuple[int, int] = 128,
) -> BlockMask:
    total_len = 2 * seq_len
    extended_doc_ids = None
    B = None
    if doc_ids is not None:
        extended_doc_ids = torch.cat([doc_ids, doc_ids], dim=1)
        B = doc_ids.shape[0]
    mask_mod = partial(
        _block_diff_mask_x0_xt,
        block_size=block_size,
        n=seq_len,
        causal_x0=causal_x0,
        causal_xt=causal_xt,
        doc_ids=extended_doc_ids,
    )
    return create_block_mask(
        mask_mod,
        B=B,
        H=None,
        Q_LEN=total_len,
        KV_LEN=total_len,
        device=device,
        BLOCK_SIZE=attention_block_size,
    )


def create_block_diff_4d_mask(
    seq_len: int,
    block_size: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    causal_x0: bool = False,
    causal_xt: bool = False,
    doc_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    total_len = 2 * seq_len
    q_idx = torch.arange(total_len, device=device)
    kv_idx = torch.arange(total_len, device=device)
    q_mesh, kv_mesh = torch.meshgrid(q_idx, kv_idx, indexing="ij")

    mask = _block_diff_mask_x0_xt(
        torch.zeros_like(q_mesh),
        torch.zeros_like(q_mesh),
        q_mesh,
        kv_mesh,
        block_size=block_size,
        n=seq_len,
        causal_x0=causal_x0,
        causal_xt=causal_xt,
    )
    attention_mask = torch.zeros((total_len, total_len), dtype=dtype, device=device)
    attention_mask.masked_fill_(~mask, torch.finfo(dtype).min)
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)

    if doc_ids is not None:
        extended = torch.cat([doc_ids, doc_ids], dim=1)
        same_doc = extended.unsqueeze(2) == extended.unsqueeze(1)
        attention_mask = attention_mask.clone()
        attention_mask.masked_fill_(~same_doc.unsqueeze(1), torch.finfo(dtype).min)

    return attention_mask


# ---------------------------------------------------------------------------
# Pure Region-DAG reference masks (not wired into the training forward path)
# ---------------------------------------------------------------------------

REGION_DAG_CONSERVATIVE_GDN_V1 = "region_dag_conservative_gdn_v1"


def _region_dag_reference_value(value, name):
    return value[name] if isinstance(value, dict) else getattr(value, name)


def create_region_dag_attention_mask(
    region_contract,
    query_positions=None,
    device=None,
):
    """Create the canonical boolean Region-DAG reference mask.

    Parent visibility is the complete transitive ancestor closure. This helper
    is intentionally independent of the `[x0; xt]` training forward and exists
    only as a pure reference oracle.
    """
    contract_id = str(
        _region_dag_reference_value(region_contract, "attention_contract_id")
    )
    if contract_id != REGION_DAG_CONSERVATIVE_GDN_V1:
        raise ValueError(f"unsupported Region-DAG contract {contract_id!r}")
    sequence_length = int(
        _region_dag_reference_value(region_contract, "sequence_length")
    )
    regions = tuple(_region_dag_reference_value(region_contract, "regions"))
    by_id = {
        str(_region_dag_reference_value(region, "region_id")): region
        for region in regions
    }
    if sequence_length <= 0 or not regions or len(by_id) != len(regions):
        raise ValueError("invalid Region-DAG reference partition")

    membership = [""] * sequence_length
    cursor = 0
    for region in sorted(
        regions,
        key=lambda value: int(_region_dag_reference_value(value, "start")),
    ):
        region_id = str(_region_dag_reference_value(region, "region_id"))
        start = int(_region_dag_reference_value(region, "start"))
        end = int(_region_dag_reference_value(region, "end"))
        if start != cursor or end <= start:
            raise ValueError("invalid Region-DAG reference interval partition")
        parents = tuple(_region_dag_reference_value(region, "parent_region_ids"))
        if any(parent not in by_id for parent in parents):
            raise ValueError(f"region {region_id!r} has an unknown parent")
        membership[start:end] = [region_id] * (end - start)
        cursor = end
    if cursor != sequence_length:
        raise ValueError("Region-DAG reference partition does not cover the sequence")

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    if query_positions is None:
        query_positions = torch.arange(
            sequence_length, dtype=torch.int64, device=target_device
        )
    elif not torch.is_tensor(query_positions):
        query_positions = torch.tensor(
            query_positions, dtype=torch.int64, device=target_device
        )
    else:
        if query_positions.dtype is not torch.int64 or query_positions.ndim != 1:
            raise ValueError("Region-DAG reference query positions must be 1-D int64")
        query_positions = query_positions.to(device=target_device)
    values = query_positions.detach().cpu().tolist()
    if (
        not values
        or values != sorted(values)
        or len(set(values)) != len(values)
        or values[0] < 0
        or values[-1] >= sequence_length
    ):
        raise ValueError("invalid Region-DAG reference query positions")

    def ancestors(region_id):
        found = set()
        visiting = set()

        def visit(current):
            if current in visiting:
                raise ValueError("Region-DAG reference graph contains a cycle")
            visiting.add(current)
            for parent in _region_dag_reference_value(
                by_id[current], "parent_region_ids"
            ):
                if parent not in found:
                    visit(parent)
                    found.add(parent)
            visiting.remove(current)

        visit(region_id)
        return found

    mask = torch.zeros(
        (len(values), sequence_length), dtype=torch.bool, device=target_device
    )
    for row, query_position in enumerate(values):
        region_id = membership[query_position]
        region = by_id[region_id]
        status = str(_region_dag_reference_value(region, "status"))
        if status.startswith("RegionStatus."):
            status = status.rsplit(".", 1)[-1].lower()
        for ancestor_id in ancestors(region_id):
            ancestor = by_id[ancestor_id]
            ancestor_status = str(_region_dag_reference_value(ancestor, "status"))
            if ancestor_status.startswith("RegionStatus."):
                ancestor_status = ancestor_status.rsplit(".", 1)[-1].lower()
            if status == "stable" and ancestor_status != "stable":
                raise ValueError("stable Region-DAG query has an active ancestor")
            mask[
                row,
                int(_region_dag_reference_value(ancestor, "start")) : int(
                    _region_dag_reference_value(ancestor, "end")
                ),
            ] = True
        start = int(_region_dag_reference_value(region, "start"))
        end = int(_region_dag_reference_value(region, "end"))
        if status == "stable":
            mask[row, start : query_position + 1] = True
        elif status == "active":
            mask[row, start:end] = True
        else:
            raise ValueError(f"invalid Region-DAG status {status!r}")
    return mask.contiguous()


def create_region_dag_4d_mask(
    region_contract,
    batch_size,
    dtype,
    device,
    query_positions=None,
):
    """Convert the pure Region-DAG boolean oracle to an additive 4-D mask."""
    if int(batch_size) <= 0:
        raise ValueError("Region-DAG reference batch size must be positive")
    boolean_mask = create_region_dag_attention_mask(
        region_contract,
        query_positions=query_positions,
        device=device,
    )
    attention_mask = torch.zeros(
        boolean_mask.shape, dtype=dtype, device=boolean_mask.device
    )
    attention_mask.masked_fill_(~boolean_mask, torch.finfo(dtype).min)
    return attention_mask.unsqueeze(0).unsqueeze(0).expand(int(batch_size), -1, -1, -1)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_flex_attention_block_size(
    model_args: Qwen3_5DLLMModelArgs,
    device: torch.device,
) -> int | tuple[int, int]:
    q_block_size = model_args.flex_attention_q_block_size
    kv_block_size = model_args.flex_attention_kv_block_size
    if q_block_size > 0:
        if kv_block_size <= 0:
            kv_block_size = q_block_size
        return (
            q_block_size
            if q_block_size == kv_block_size
            else (q_block_size, kv_block_size)
        )
    if (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device) >= (9, 0)
        and model_args.head_dim >= 256
    ):
        return 64
    return 128


def _doc_ids_to_cu_seqlens(doc_ids: torch.Tensor) -> torch.Tensor:
    """Convert per-token document IDs ``[B, L]`` to merged cumulative sequence lengths.

    When ``B > 1``, batch elements are treated as consecutive segments in a
    single packed sequence so that Triton kernels (which require ``B = 1``)
    can process them in one launch.

    Returns:
        ``int32`` tensor of shape ``[total_docs + 1]`` with values in
        ``[0, B * L]``.
    """
    B, L = doc_ids.shape
    parts = []
    for i in range(B):
        ids = doc_ids[i]
        bounds = torch.where(ids[1:] != ids[:-1])[0] + 1
        parts.append(torch.cat([ids.new_zeros(1), bounds]) + i * L)
    parts.append(doc_ids.new_tensor([B * L]))
    return torch.cat(parts).to(torch.int32)


def _remap_cu_seqlens_for_row_padding(
    cu_seqlens: torch.Tensor,
    seq_per_row: int,
    padded_seq_per_row: int,
) -> torch.Tensor:
    """Map flat row-local document boundaries after adding per-row padding."""
    if padded_seq_per_row == seq_per_row:
        return cu_seqlens
    rows = torch.div(cu_seqlens, seq_per_row, rounding_mode="floor")
    local = torch.remainder(cu_seqlens, seq_per_row)
    return (rows * padded_seq_per_row + local).to(cu_seqlens.dtype)


def _doc_ids_to_positions(doc_ids: torch.Tensor) -> torch.Tensor:
    """Convert per-token document IDs to per-document position indices.

    Example: doc_ids = [0,0,0, 1,1,1,1] -> positions = [0,1,2, 0,1,2,3]
    """
    B, L = doc_ids.shape
    device = doc_ids.device
    boundary = torch.cat(
        [torch.ones(B, 1, dtype=torch.long, device=device),
         (doc_ids[:, 1:] != doc_ids[:, :-1]).long()],
        dim=1,
    )
    global_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    first_occurrence = boundary * global_idx
    doc_start = torch.cummax(first_occurrence, dim=1).values
    return global_idx - doc_start


def _shift_labels(labels: torch.Tensor) -> torch.Tensor:
    """Dream-style shift: labels[i] = original_labels[i+1]."""
    return F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()


def _fix_doc_boundary(shifted: torch.Tensor, doc_ids: torch.Tensor | None) -> torch.Tensor:
    """Set shifted labels to -100 at document boundaries."""
    if doc_ids is None:
        return shifted
    boundary = doc_ids[:, :-1] != doc_ids[:, 1:]
    shifted[:, :-1][boundary] = -100
    return shifted


# ---------------------------------------------------------------------------
# Attention layers
# ---------------------------------------------------------------------------

class DLLMGatedAttention(GatedAttention):
    """Qwen3.5 full attention with block-diffusion mask support."""

    def __init__(self, model_args: Qwen3_5DLLMModelArgs):
        super().__init__(model_args)
        self.block_diff_use_flex_attention = model_args.use_flex_attention
        if self.block_diff_use_flex_attention:
            self.block_diff_attention = FlexAttentionWrapper()

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | torch.Tensor | None,
        positions: torch.Tensor | None = None,
    ):
        bs, seqlen, _ = x.shape
        xq_full = self.wq(x)
        xk, xv = self.wk(x), self.wv(x)

        xq_full = xq_full.view(bs, seqlen, self.n_heads, self.head_dim * 2)
        xq = xq_full[..., : self.head_dim].contiguous()
        gate = xq_full[..., self.head_dim :]

        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if self.q_norm:
            xq = self.q_norm(xq)
        if self.k_norm:
            xk = self.k_norm(xk)

        xq, xk = apply_partial_rotary_emb(
            xq, xk, rope_cache, self.partial_rotary_factor, positions
        )

        keys = xk.repeat_interleave(self.n_rep, dim=2)
        values = xv.repeat_interleave(self.n_rep, dim=2)

        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        if isinstance(attention_masks, BlockMask):
            output = self.block_diff_attention(
                xq, xk, xv, block_mask=attention_masks, scale=self.scaling
            )
        elif attention_masks is not None:
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=attention_masks,
                scale=self.scaling,
                is_causal=False,
            )
        else:
            match self.attn_type:
                case "flex":
                    output = self.inner_attention(
                        xq, xk, xv, block_mask=attention_masks, scale=self.scaling
                    )
                case "varlen":
                    output = self.inner_attention(
                        xq, xk, xv, self.head_dim, attention_masks, scale=self.scaling
                    )
                case "sdpa":
                    output = self.inner_attention(xq, xk, xv, scale=self.scaling)
                case _:
                    raise ValueError(f"Unknown attention type: {self.attn_type}")

        output = output.transpose(1, 2).contiguous()
        output_gated = output * torch.sigmoid(gate)
        output_gated = output_gated.view(bs, seqlen, -1)
        return self.wo(output_gated)


# ---------------------------------------------------------------------------
# Linear attention (block-train GDA kernel)
# ---------------------------------------------------------------------------

class BlockGatedDeltaNet(GatedDeltaNet):
    """Qwen3.5 linear attention with block-train GDA kernel support."""

    def __init__(self, model_args: Qwen3_5DLLMModelArgs, layer_idx: int):
        super().__init__(model_args, layer_idx)
        self.block_size = model_args.block_size
        self.block_train_method = model_args.block_train_method
        self.block_train_conv_method = model_args.block_train_conv_method
        self.chunk_wy_bwd_split_enabled = model_args.chunk_wy_bwd_split_enabled
        self.chunk_wy_bwd_parallel_groups = model_args.chunk_wy_bwd_parallel_groups
        self.chunk_wy_bwd_checkpoint_stride = model_args.chunk_wy_bwd_checkpoint_stride
        self.chunk_wy_bwd_bv = model_args.chunk_wy_bwd_bv
        self.chunk_wy_bwd_store_b = model_args.chunk_wy_bwd_store_b
        self.causal_mode_clean = int(model_args.causal_x0)
        self.causal_mode_noisy = int(model_args.causal_xt)

    def forward(
        self,
        x: torch.Tensor,
        block_train: bool = False,
        block_causal: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not block_train and not block_causal and cu_seqlens is None:
            return super().forward(x)

        input_mesh = None
        if hasattr(torch.distributed, "tensor"):
            from torch.distributed.tensor import DTensor, Replicate
            if isinstance(x, DTensor):
                input_mesh = x.device_mesh
                x = x.to_local()
        else:
            Replicate = None
            DTensor = None

        B_orig, T_orig, D = x.shape
        need_unflatten = cu_seqlens is not None and B_orig > 1
        padded_seq_per_row = None

        if need_unflatten:
            seq_per_row = T_orig // 2 if block_train else T_orig
            pad_len = (-seq_per_row) % self.block_size
            padded_seq_per_row = seq_per_row + pad_len
            cu_seqlens = _remap_cu_seqlens_for_row_padding(
                cu_seqlens,
                seq_per_row,
                padded_seq_per_row,
            )
            if block_train:
                half_T = T_orig // 2
                clean = x[:, :half_T]
                noisy = x[:, half_T:]
                if pad_len:
                    clean = F.pad(clean, (0, 0, 0, pad_len))
                    noisy = F.pad(noisy, (0, 0, 0, pad_len))
                x = torch.cat([
                    clean.reshape(1, B_orig * padded_seq_per_row, D),
                    noisy.reshape(1, B_orig * padded_seq_per_row, D),
                ], dim=1)
            else:
                if pad_len:
                    x = F.pad(x, (0, 0, 0, pad_len))
                x = x.reshape(1, B_orig * padded_seq_per_row, D)

        output = self._forward_block_standard(x, block_train, cu_seqlens=cu_seqlens)

        if need_unflatten:
            assert padded_seq_per_row is not None
            if block_train:
                half_T = T_orig // 2
                out_clean_flat = output[:, :B_orig * padded_seq_per_row]
                out_noisy_flat = output[:, B_orig * padded_seq_per_row:]
                out_clean = out_clean_flat.reshape(B_orig, padded_seq_per_row, -1)[:, :half_T]
                out_noisy = out_noisy_flat.reshape(B_orig, padded_seq_per_row, -1)[:, :half_T]
                output = torch.cat([out_clean, out_noisy], dim=1)
            else:
                output = output.reshape(B_orig, padded_seq_per_row, -1)[:, :T_orig]

        if input_mesh is not None and DTensor is not None and Replicate is not None:
            replicated = tuple(Replicate() for _ in range(input_mesh.ndim))
            output = DTensor.from_local(output, input_mesh, replicated, run_check=False)
        return output

    def _forward_block_standard(
        self,
        x: torch.Tensor,
        block_train: bool,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        mixed_qkv = F.linear(x, _to_local_if_dtensor(self.in_proj_qkv.weight))
        z = F.linear(x, _to_local_if_dtensor(self.in_proj_z.weight))
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = F.linear(x, _to_local_if_dtensor(self.in_proj_b.weight))
        a = F.linear(x, _to_local_if_dtensor(self.in_proj_a.weight))

        if block_train:
            conv_weight = _to_local_if_dtensor(self.conv1d.weight).squeeze(1)
            conv_bias = (
                _to_local_if_dtensor(self.conv1d.bias)
                if self.conv1d.bias is not None
                else None
            )
            mixed_qkv = block_train_conv(
                x=mixed_qkv,
                weight=conv_weight,
                bias=conv_bias,
                block_size=self.block_size,
                activation="silu",
                cu_seqlens=cu_seqlens,
                method=self.block_train_conv_method,
            )
        else:
            conv_weight = _to_local_if_dtensor(self.conv1d.weight)
            conv_bias = (
                _to_local_if_dtensor(self.conv1d.bias)
                if self.conv1d.bias is not None
                else None
            )
            if cu_seqlens is not None:
                conv_out = _causal_conv1d(
                    x=mixed_qkv,
                    weight=conv_weight.squeeze(1),
                    bias=conv_bias,
                    activation='silu',
                    cu_seqlens=cu_seqlens,
                )
                mixed_qkv = conv_out[0] if isinstance(conv_out, tuple) else conv_out
            else:
                mixed_qkv = mixed_qkv.transpose(1, 2)
                mixed_qkv = F.silu(
                    F.conv1d(
                        mixed_qkv,
                        conv_weight,
                        bias=conv_bias,
                        padding=self.conv1d.kernel_size[0] - 1,
                        groups=self.key_dim * 2 + self.value_dim,
                    )[:, :, :seq_len]
                )
                mixed_qkv = mixed_qkv.transpose(1, 2)

        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        a_log = _to_local_if_dtensor(self.A_log).float()
        dt_bias = _to_local_if_dtensor(self.dt_bias)
        g = -a_log.exp() * F.softplus(a.float() + dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            repeat_factor = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

        output, _ = chunk_block_causal_gated_delta_rule(
            q=query, k=key, v=value, g=g, beta=beta,
            scale=self.head_k_dim ** -0.5,
            block_size=self.block_size,
            initial_state=None,
            output_final_state=False,
            block_train=block_train,
            block_train_method=self.block_train_method,
            causal_mode_clean=self.causal_mode_clean,
            causal_mode_noisy=self.causal_mode_noisy,
            cu_seqlens=cu_seqlens,
            chunk_wy_bwd_split_enabled=self.chunk_wy_bwd_split_enabled,
            chunk_wy_bwd_parallel_groups=self.chunk_wy_bwd_parallel_groups,
            chunk_wy_bwd_checkpoint_stride=self.chunk_wy_bwd_checkpoint_stride,
            chunk_wy_bwd_bv=self.chunk_wy_bwd_bv,
            chunk_wy_bwd_store_b=self.chunk_wy_bwd_store_b,
        )

        output = output.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        output = self.norm(output, z)
        output = output.reshape(batch_size, seq_len, -1)
        output = F.linear(output, _to_local_if_dtensor(self.out_proj.weight))
        return output



# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class DLLMTransformerBlock(TransformerBlock):
    """Qwen3.5 transformer block wired for block diffusion training."""

    def __init__(self, layer_id: int, model_args: Qwen3_5DLLMModelArgs):
        super().__init__(layer_id, model_args)
        if self.layer_type == "linear_attention":
            self.linear_attn = BlockGatedDeltaNet(model_args, layer_id)
        else:
            self.self_attn = DLLMGatedAttention(model_args)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | torch.Tensor | None,
        positions: torch.Tensor | None = None,
        block_train: bool = False,
        block_causal: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ):
        residual = x
        h = self.input_layernorm(x)

        if self.layer_type == "linear_attention":
            h = self.linear_attn(
                h, block_train=block_train, block_causal=block_causal,
                cu_seqlens=cu_seqlens,
            )
        else:
            h = self.self_attn(h, rope_cache, attention_masks, positions)

        x = residual + h

        residual = x
        h = self.post_attention_layernorm(x)
        if self.moe_enabled:
            h = self.moe(h)
        else:
            h = self.feed_forward(h)
        return residual + h


# ---------------------------------------------------------------------------
# Main DLLM model
# ---------------------------------------------------------------------------

class Qwen3_5DLLMModel(Qwen3_5Model):
    """Qwen3.5 DLLM model with full training mode support.

    Supports: causal_x0/xt, all_masked, logit_shift, doc_ids (packing),
    AR loss, complementary masking, antithetic sampling, fused CE.
    Layout: [x0; xt] (clean first, noisy second).
    """

    def __init__(self, model_args: Qwen3_5DLLMModelArgs):
        super().__init__(model_args)
        self.model_args = model_args
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = DLLMTransformerBlock(layer_id, model_args)

    # -- Mask sampling --

    def _sample_block_mask_probabilities(
        self, batch_size: int, num_blocks: int, device: torch.device,
    ) -> torch.Tensor:
        block_probs = torch.rand((batch_size, num_blocks), device=device)
        if not self.model_args.antithetic_sampling:
            return block_probs
        flat_probs = block_probs.reshape(-1)
        num_samples = flat_probs.numel()
        strata_offsets = torch.arange(num_samples, device=device, dtype=flat_probs.dtype)
        flat_probs = (flat_probs + strata_offsets) / num_samples
        permutation = torch.randperm(num_samples, device=device)
        return flat_probs[permutation].view(batch_size, num_blocks)

    def _sample_block_diffusion_mask(
        self, batch_size: int, seq_len: int, device: torch.device,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.model_args.all_masked:
            masked = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
            if labels is not None:
                masked[labels == -100] = False
            return masked

        block_size = self.model_args.block_size
        num_blocks = (seq_len + block_size - 1) // block_size
        probs = self._sample_block_mask_probabilities(batch_size, num_blocks, device)
        probs = probs.repeat_interleave(block_size, dim=-1)[:, :seq_len]
        masked_indices = torch.rand(batch_size, seq_len, device=device) <= probs
        if labels is not None:
            masked_indices[labels == -100] = False
        return masked_indices

    # -- Build diffusion inputs --

    def _build_block_diffusion_inputs(
        self,
        x0_embeds: torch.Tensor,
        masked_indices: torch.Tensor,
        doc_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, BlockMask | torch.Tensor]:
        batch_size, seq_len, _ = x0_embeds.shape
        device = x0_embeds.device
        dtype = x0_embeds.dtype

        mask_embed = self.tok_embeddings(
            torch.tensor([self.model_args.mask_token_id], device=device)
        ).unsqueeze(0)
        xt_embeds = torch.where(
            masked_indices.unsqueeze(-1),
            mask_embed.expand(batch_size, seq_len, -1),
            x0_embeds,
        )
        bd_inputs = torch.cat([x0_embeds, xt_embeds], dim=1)

        if doc_ids is not None:
            pos_ids = _doc_ids_to_positions(doc_ids)
        else:
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        positions = torch.cat([pos_ids, pos_ids], dim=1)

        if self.model_args.use_flex_attention:
            attention_mask = create_block_diff_attention_mask(
                seq_len=seq_len,
                block_size=self.model_args.block_size,
                device=device,
                causal_x0=self.model_args.causal_x0,
                causal_xt=self.model_args.causal_xt,
                doc_ids=doc_ids,
                attention_block_size=_resolve_flex_attention_block_size(
                    self.model_args, device
                ),
            )
        else:
            attention_mask = create_block_diff_4d_mask(
                seq_len=seq_len,
                block_size=self.model_args.block_size,
                batch_size=batch_size,
                dtype=dtype,
                device=device,
                causal_x0=self.model_args.causal_x0,
                causal_xt=self.model_args.causal_xt,
                doc_ids=doc_ids,
            )

        return bd_inputs, positions, attention_mask

    # -- Forward diffusion --

    def forward_diffusion(
        self,
        x0_embeds: torch.Tensor,
        labels: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BlockMask | torch.Tensor]:
        batch_size, seq_len, _ = x0_embeds.shape
        device = x0_embeds.device

        masked_indices = self._sample_block_diffusion_mask(
            batch_size, seq_len, device, labels,
        )

        if self.model_args.complementary_mask:
            comp = ~masked_indices
            if labels is not None:
                comp[labels == -100] = False
            masked_indices = torch.cat([masked_indices, comp], dim=0)
            x0_embeds = torch.cat([x0_embeds, x0_embeds], dim=0)
            if doc_ids is not None:
                doc_ids = torch.cat([doc_ids, doc_ids], dim=0)

        bd_inputs, positions, attention_mask = self._build_block_diffusion_inputs(
            x0_embeds, masked_indices, doc_ids,
        )
        return bd_inputs, masked_indices, positions, attention_mask

    # -- Transformer stack --

    def _run_block_diffusion_stack(
        self,
        hidden: torch.Tensor,
        attention_mask: BlockMask | torch.Tensor,
        positions: torch.Tensor,
        *,
        block_train: bool,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cu_seqlens = None
        if doc_ids is not None:
            actual_B = hidden.shape[0]
            if doc_ids.shape[0] != actual_B:
                doc_ids = doc_ids.repeat(actual_B // doc_ids.shape[0], 1)
            cu_seqlens = _doc_ids_to_cu_seqlens(doc_ids)
        for layer in self.layers.values():
            hidden = layer(
                hidden, self.rope_cache, attention_mask, positions,
                block_train=block_train, cu_seqlens=cu_seqlens,
            )
        return self.norm(hidden) if self.norm else hidden

    # -- Loss computation --

    def _compute_training_loss(
        self,
        h_x0: torch.Tensor,
        h_xt: torch.Tensor,
        labels: torch.Tensor,
        masked_indices: torch.Tensor,
        doc_ids: torch.Tensor | None,
        skip_ar_loss: bool = False,
    ) -> dict:
        """Compute training loss and return result dict.

        Args:
            skip_ar_loss: If True, compute only the diffusion loss (L_mask)
                and skip the AR loss (L_clean). Used by serial complementary
                mode which computes AR loss separately once.
        """
        logit_shift = self.model_args.logit_shift
        ar_weight = self.model_args.ar_loss_weight
        loss_auto_balance = self.model_args.loss_auto_balance
        loss_normalize = self.model_args.loss_normalize
        use_fused = (
            self.model_args.use_fused_ce
            and h_xt.is_cuda
            and liger_fused_linear_cross_entropy is not None
        )

        expanded_labels = labels
        expanded_doc_ids = doc_ids
        if labels.shape[0] != masked_indices.shape[0]:
            repeat_factor = masked_indices.shape[0] // labels.shape[0]
            expanded_labels = labels.repeat(repeat_factor, 1)
            if doc_ids is not None:
                expanded_doc_ids = doc_ids.repeat(repeat_factor, 1)

        if logit_shift:
            diff_labels = _shift_labels(expanded_labels)
            diff_labels = _fix_doc_boundary(diff_labels, expanded_doc_ids)
            diff_labels[~masked_indices] = -100
        else:
            diff_labels = expanded_labels.clone()
            diff_labels[~masked_indices] = -100

        if use_fused:
            diff_loss = liger_fused_linear_cross_entropy(
                h_xt.reshape(-1, h_xt.shape[-1]),
                self.output.weight,
                diff_labels.reshape(-1),
                bias=self.output.bias,
                ignore_index=-100,
                reduction="mean",
            )

            ar_loss = None
            if ar_weight > 0 and not skip_ar_loss:
                B_orig = labels.shape[0]
                h_x0_orig = h_x0[:B_orig]
                ar_labels = _shift_labels(labels)
                ar_labels = _fix_doc_boundary(ar_labels, doc_ids)
                ar_loss = liger_fused_linear_cross_entropy(
                    h_x0_orig.reshape(-1, h_x0_orig.shape[-1]),
                    self.output.weight,
                    ar_labels.reshape(-1),
                    bias=self.output.bias,
                    ignore_index=-100,
                    reduction="mean",
                )

            if ar_loss is not None:
                if loss_auto_balance:
                    delta = diff_loss.detach() / (ar_loss.detach() + 1e-8)
                    total_loss = diff_loss + delta * ar_loss
                elif loss_normalize:
                    total_loss = (diff_loss + ar_weight * ar_loss) / (1 + ar_weight)
                else:
                    total_loss = diff_loss + ar_weight * ar_loss
            else:
                total_loss = diff_loss

            return {"loss": total_loss, "masked_indices": masked_indices}

        # Non-fused fallback
        logits = self.output(h_xt)

        ar_loss = None
        if ar_weight > 0 and not skip_ar_loss:
            B_orig = labels.shape[0]
            h_x0_orig = h_x0[:B_orig]
            ar_labels = _shift_labels(labels)
            ar_labels = _fix_doc_boundary(ar_labels, doc_ids)
            valid_ar = ar_labels != -100
            if valid_ar.any():
                logits_ar = self.output(h_x0_orig)
                ar_loss = F.cross_entropy(
                    logits_ar[valid_ar].float(), ar_labels[valid_ar], reduction="mean"
                )

        return {
            "logits": logits,
            "diff_labels": diff_labels,
            "masked_indices": masked_indices,
            "ar_loss": ar_loss,
            "ar_loss_weight": ar_weight,
            "loss_auto_balance": loss_auto_balance,
            "loss_normalize": loss_normalize,
        }

    # -- Main forward --

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        block_train: bool = False,
        doc_ids: torch.Tensor | None = None,
    ):
        hidden = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        if self.training and block_train:
            seq_len = tokens.shape[1]

            # Serial complementary: process primary + comp masks sequentially.
            # Diff loss (L_mask) is computed per branch and averaged.
            # AR loss (L_clean) is computed once on the primary branch's h_x0,
            # since x_0 output is independent of which tokens are masked.
            if (
                self.model_args.complementary_mask
                and self.model_args.serial_complementary_mask
                and labels is not None
            ):
                logit_shift = self.model_args.logit_shift
                ar_weight = self.model_args.ar_loss_weight
                loss_auto_balance = self.model_args.loss_auto_balance
                loss_normalize = self.model_args.loss_normalize
                use_fused = (
                    self.model_args.use_fused_ce
                    and hidden.is_cuda
                    and liger_fused_linear_cross_entropy is not None
                )

                primary_mask = self._sample_block_diffusion_mask(
                    hidden.shape[0], seq_len, hidden.device, labels,
                )
                comp_mask = ~primary_mask
                comp_mask[labels == -100] = False

                diff_losses = []
                primary_h_x0 = None

                for branch_mask in (primary_mask, comp_mask):
                    branch_inputs, branch_positions, branch_attn_mask = (
                        self._build_block_diffusion_inputs(hidden, branch_mask, doc_ids)
                    )
                    branch_hidden = self._run_block_diffusion_stack(
                        branch_inputs, branch_attn_mask, branch_positions,
                        block_train=True, doc_ids=doc_ids,
                    )
                    branch_h_x0 = branch_hidden[:, :seq_len]
                    branch_h_xt = branch_hidden[:, seq_len:]

                    if primary_h_x0 is None:
                        primary_h_x0 = branch_h_x0

                    branch_result = self._compute_training_loss(
                        branch_h_x0, branch_h_xt, labels, branch_mask, doc_ids,
                        skip_ar_loss=True,
                    )
                    if "loss" in branch_result:
                        diff_losses.append(branch_result["loss"])
                    else:
                        valid = branch_result["diff_labels"] != -100
                        if valid.any():
                            diff_losses.append(F.cross_entropy(
                                branch_result["logits"][valid].float(),
                                branch_result["diff_labels"][valid],
                                reduction="mean",
                            ))

                avg_diff_loss = sum(diff_losses) / max(len(diff_losses), 1)

                # AR loss once on primary branch h_x0
                ar_loss = None
                if ar_weight > 0 and primary_h_x0 is not None:
                    ar_labels = _shift_labels(labels)
                    ar_labels = _fix_doc_boundary(ar_labels, doc_ids)
                    if use_fused:
                        ar_loss = liger_fused_linear_cross_entropy(
                            primary_h_x0.reshape(-1, primary_h_x0.shape[-1]),
                            self.output.weight,
                            ar_labels.reshape(-1),
                            bias=self.output.bias,
                            ignore_index=-100,
                            reduction="mean",
                        )
                    else:
                        valid_ar = ar_labels != -100
                        if valid_ar.any():
                            logits_ar = self.output(primary_h_x0)
                            ar_loss = F.cross_entropy(
                                logits_ar[valid_ar].float(),
                                ar_labels[valid_ar],
                                reduction="mean",
                            )

                # Combine losses (same logic as _compute_training_loss)
                if ar_loss is not None:
                    if loss_auto_balance:
                        delta = avg_diff_loss.detach() / (ar_loss.detach() + 1e-8)
                        total_loss = avg_diff_loss + delta * ar_loss
                    elif loss_normalize:
                        total_loss = (avg_diff_loss + ar_weight * ar_loss) / (1 + ar_weight)
                    else:
                        total_loss = avg_diff_loss + ar_weight * ar_loss
                else:
                    total_loss = avg_diff_loss

                return {"loss": total_loss, "masked_indices": primary_mask}

            # Standard path (with optional batch-doubled complementary)
            bd_inputs, masked_indices, bd_positions, bd_attention_mask = (
                self.forward_diffusion(hidden, labels=labels, doc_ids=doc_ids)
            )

            hidden = self._run_block_diffusion_stack(
                bd_inputs, bd_attention_mask, bd_positions,
                block_train=True, doc_ids=doc_ids,
            )

            h_x0 = hidden[:, :seq_len]
            h_xt = hidden[:, seq_len:]

            return self._compute_training_loss(
                h_x0, h_xt, labels, masked_indices, doc_ids,
            )

        # Inference
        for layer in self.layers.values():
            hidden = layer(
                hidden, self.rope_cache, attention_masks, positions,
            )
        hidden = self.norm(hidden) if self.norm else hidden
        return self.output(hidden) if self.output else hidden

    # -- External loss functions (passthrough for fused CE) --

    @staticmethod
    def compute_loss(pred, labels) -> torch.Tensor:
        if isinstance(pred, dict) and "loss" in pred:
            return pred["loss"]

        logits = pred["logits"] if isinstance(pred, dict) else pred
        diff_labels = pred.get("diff_labels") if isinstance(pred, dict) else None

        if logits.shape[0] != labels.shape[0]:
            labels = labels.repeat(logits.shape[0] // labels.shape[0], 1)

        if diff_labels is None:
            masked_indices = pred.get("masked_indices") if isinstance(pred, dict) else None
            diff_labels = labels.clone()
            if masked_indices is not None:
                diff_labels[~masked_indices] = -100

        valid = diff_labels != -100
        if valid.sum() == 0:
            return (logits * 0).sum()

        diff_loss = F.cross_entropy(logits[valid].float(), diff_labels[valid], reduction="mean")

        if isinstance(pred, dict):
            ar_loss = pred.get("ar_loss")
            ar_weight = pred.get("ar_loss_weight", 0.0)
            loss_auto_balance = pred.get("loss_auto_balance", False)
            loss_normalize = pred.get("loss_normalize", True)

            if ar_loss is not None:
                if loss_auto_balance:
                    delta = diff_loss.detach() / (ar_loss.detach() + 1e-8)
                    return diff_loss + delta * ar_loss
                elif loss_normalize:
                    return (diff_loss + ar_weight * ar_loss) / (1 + ar_weight)
                else:
                    return diff_loss + ar_weight * ar_loss

        return diff_loss
