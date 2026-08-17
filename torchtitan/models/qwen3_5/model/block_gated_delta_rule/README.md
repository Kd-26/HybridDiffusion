<p align="right">
  <a href="../../../../../README.md">HybridDiffusion</a> ·
  <a href="https://arxiv.org/abs/2606.01774">Paper</a> ·
  <a href="#automatic-dispatch">Dispatch</a> ·
  <a href="#source-map">Source Map</a>
</p>

# Block-causal Gated DeltaNet kernels

This directory contains the recurrent-attention and short-convolution kernels
used by the Qwen3.5 HybridDiffusion training path. The implementation extends Flash
Linear Attention (FLA) primitives with HybridDiffusion's clean/noisy stream semantics,
block-causal readout, packed-document boundaries, and custom Triton kernels.

`chunk_wy_triton_fla_style` implements HybridDiffusion's fused **two-stream** path
(Route II in the paper). The clean stream uses an FLA-style chunkwise WY
representation, while custom Triton forward/backward kernels consume the
clean-prefix state for the noisy stream and return its gradients to the clean
recurrence.

The HybridDiffusion training entry point is `qwen3_5_hybrid_diffusion_sft`; training users do not
call these kernels directly. See the [repository guide](../../../../../README.md)
for installation, configurations, and launch commands.

| Release path | Implementation |
|---|---|
| Gated DeltaNet attention | Fused two-stream WY (`chunk_wy_triton_fla_style`) |
| Short convolution | Two-stream Triton (`twostream`) |
| Configuration | Automatic dispatch with internal `block_size = 3` |
| Packing | Document-safe state resets through `cu_seqlens` |

## Provenance and copyright

HybridDiffusion-specific kernel implementations and modifications are copyright 2026
Yuchen Zhu. Files that contain code copied or adapted from FLA retain the
upstream copyright in their header and identify themselves as modified from
FLA. The repository `NOTICE` includes the complete FLA MIT notice.

Importing an FLA helper does not by itself make a file an FLA copy. The origin
column below distinguishes HybridDiffusion implementations from files that contain a
substantial FLA-derived portion.

## Execution semantics

Gated DeltaNet maintains a recurrent key-value state. In simplified form,

$$
h_t = e^{g_t} h_{t-1} + k_t \otimes
      \beta_t\left(v_t-k_t^\top e^{g_t}h_{t-1}\right).
$$

The state update remains token-causal. HybridDiffusion changes the output readout so
tokens in one training block can share the state at the appropriate block
boundary. During block training the first half of a row is the clean stream
and the second half is the complementary noisy stream. State extracted from
the clean stream conditions the corresponding noisy blocks.

Packed examples pass `cu_seqlens`. The attention and convolution entry points
validate document boundaries before dispatch so state cannot silently cross
from one packed conversation into another.

## Public API

`__init__.py` exports:

- `chunk_block_causal_gated_delta_rule`: training forward/backward, including
  HybridDiffusion block training;
- `fused_recurrent_block_causal_gated_delta_rule`: forward-only recurrent
  execution with optional initial/final state.

The Qwen3.5 model calls the training kernel and `block_train_conv` from
`model/dllm_model.py`.

## Automatic dispatch

Production configs should keep `block_train_method = "auto"`. The current
attention dispatch in `chunk.py` is:

```python
method = (
    "chunk_wy_triton_fla_style"
    if block_size < 16
    else "chunk_refine"
)
```

The three release configs use internal `block_size = 3`, so their attention
path is `chunk_wy_triton_fla_style`.

| Paper route | Code route | Automatic selection |
|---|---|---|
| Route II: fused two-stream | `chunk_wy_triton_fla_style` | internal `block_size < 16` |
| Route I: chunk-then-refine | `chunk_refine` | internal `block_size >= 16` |

For the release configs, the aligned WY chunk is 63 tokens and the default
clean-state checkpoint stride is 3.

Short convolution has a separate automatic dispatch in `convolution.py`:

- CUDA, convolution width at most 4, and no/SiLU/Swish activation:
  `twostream`;
- otherwise: `fla_batched`.

Explicit attention methods remain available for kernel development:

- `sequential`;
- `fused`;
- `chunk_refine`;
- `chunk_refine_reuse`;
- `chunk_wy_triton`;
- `chunk_wy_triton_improved`;
- `chunk_wy_triton_fla_style`.

Release configurations should use automatic dispatch. Explicit overrides are
intended for kernel experiments and should be validated for the exact shape,
dtype, device, packing mode, and backward pass.

## Source map

| File | Role | Origin |
|---|---|---|
| `__init__.py` | Public exports | HybridDiffusion |
| `_profiling.py` | Optional kernel timing and checkpoint controls | HybridDiffusion |
| `block_short_conv_ops.py` | Two-stream short-convolution Triton kernels | HybridDiffusion |
| `chunk.py` | Core autograd functions and attention dispatcher | Modified from FLA |
| `chunk_bwd_dqkwg.py` | Grid-safe local backward kernel | Modified from FLA |
| `chunk_delta_h.py` | Grid-safe chunk state recurrence | Modified from FLA |
| `chunk_fla_style_wy.py` | HybridDiffusion two-stream WY implementation using FLA primitives | HybridDiffusion |
| `chunk_local_refine.py` | Refines chunk states to HybridDiffusion block boundaries | HybridDiffusion |
| `chunk_o.py` | Block-causal output and gradient kernels | Modified from FLA |
| `chunk_scaled_dot_kkt.py` | Grid-safe local KKT kernel | Modified from FLA |
| `chunk_two_stream_wy.py` | Two-stream WY forward/backward implementation | HybridDiffusion |
| `convolution.py` | ShortConvolution integration and dispatch | Modified from FLA |
| `convolution_ops.py` | Grid-safe causal-convolution training kernels | Modified from FLA |
| `cumsum.py` | Grid-safe local cumulative-sum kernels | Modified from FLA |
| `fused_recurrent.py` | Forward-only block-causal recurrence | Modified from FLA |
| `fused_recurrent_state.py` | Boundary-state extraction and backward | Modified from FLA |
| `solve_tril.py` | Grid-safe triangular solve, including small blocks | Modified from FLA |
| `wy_fast.py` | Grid-safe WY representation kernels | Modified from FLA |

## Main computational paths

### Chunked training

The standard chunked path builds the WY representation, propagates recurrent
state across chunks, applies block-causal output kernels, and mirrors those
operations in backward. Local variants of FLA utilities keep the potentially
large batch/head product on CUDA grid dimension x.

### HybridDiffusion two-stream training

For the release block size, the two-stream path is concrete rather than a
naming convention:

1. `_block_train_chunk_wy_triton_fla_style` splits every tensor into clean and
   noisy halves.
2. `ChunkBlockCausalGDAWithHFunctionV2Improved` computes clean outputs, WY-
   corrected values, cumulative gates, and chunk-boundary clean states.
3. `TwoStreamFLAStyleFullFunction` calls
   `two_stream_fla_style_full_fwd_triton` for the noisy stream.
4. Its backward calls `two_stream_fla_style_full_bwd_triton`, which computes
   noisy gradients plus clean-transition and chunk-state gradients consumed by
   the clean WY backward.
5. The clean and noisy outputs are concatenated only after their separate
   computations finish.

The fused backward reconstructs intermediate clean states from strided
checkpoints in registers instead of materializing every block-boundary state
in HBM. The implementation supports packed-document resets and exposes tuning
knobs only for explicit kernel experiments; release configs use the defaults.

### Chunk refinement

For block sizes of at least 16, automatic dispatch uses `chunk_refine`. It
computes states at chunk boundaries with the WY pipeline and then refines them
to the requested block boundaries. `chunk_refine_reuse` is available as an
explicit alternative but is not selected by automatic dispatch.

### Short convolution

The clean stream follows causal convolution. The noisy stream receives the
correct clean prefix for its block while remaining isolated from unrelated
noisy blocks and packed documents. `twostream` is the CUDA release path for
the Qwen3.5 convolution shape; `fla_batched` remains the fallback.

## Licensing

HybridDiffusion modifications are subject to the repository [License](../../../../../LICENSE),
including its non-commercial term. FLA-derived portions remain subject to the
FLA MIT notice reproduced in `NOTICE`; retain the per-file provenance headers
when redistributing or modifying this directory.
