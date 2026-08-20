"""HybridDiffusion Diffusion-Trust decoding with confidence-based block denoising.

The paper's logical block size B=4 uses three runtime query positions:

    [current seed, MASK, MASK]

The current seed is predicted by the previous block (or prompt prefill) and is
injected at position 0. Under logit-shift training, the first two logits fill
the two masks; the final shifted logit samples the next block's seed and holds
it outside the active query block. The carried seed is why runtime
``block_size: 3`` corresponds to paper block size B=4.
"""

from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.low_confidence import LowConfidence
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class LowConfidenceShiftHybridDiffusion(LowConfidence):
    """Diffusion-Trust decoder with one seed plus bidirectional mask rows."""

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.gen_block_size = int(config.algorithm_config.get("gen_block_size", 1))
        if self.gen_block_size != 1:
            raise ValueError(
                "LowConfidenceShiftHybridDiffusion requires gen_block_size=1 "
                f"(seed-only clean prefix), got {self.gen_block_size}"
            )
        if self.block_size < 2:
            raise ValueError(
                "LowConfidenceShiftHybridDiffusion expects block_size>=2 "
                f"([seed, one-or-more MASKs]), got {self.block_size}"
            )

        self.maximum_unroll = int(config.algorithm_config.get("maximum_unroll", 0))
        self.temperature = float(config.algorithm_config.get("temperature", 0.0))
        self.top_k = int(config.algorithm_config.get("top_k", 50))
        self.top_p = float(config.algorithm_config.get("top_p", 0.95))
        self._seed_tokens: Dict[int, int] = {}

        mask = torch.zeros(self.block_size, self.block_size, dtype=torch.bool)
        mask[0, 0] = True
        mask[1:, :] = True
        self._hybrid_diffusion_mask = mask

    def cleanup_request(self, req_pool_idx: int) -> None:
        self._seed_tokens.pop(req_pool_idx, None)

    def _sample_tokens(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.temperature <= 0.0:
            ids = torch.argmax(logits, dim=-1)
            probs = torch.gather(
                F.softmax(logits, dim=-1), dim=-1, index=ids.unsqueeze(-1)
            ).squeeze(-1)
            return ids, probs

        scaled = logits.float() / self.temperature
        if self.top_k and self.top_k > 0:
            k = min(self.top_k, scaled.shape[-1])
            topk_vals, topk_idx = torch.topk(scaled, k=k, dim=-1)
            scaled = torch.full_like(scaled, float("-inf"))
            scaled.scatter_(-1, topk_idx, topk_vals)

        if self.top_p and self.top_p < 1.0:
            sorted_vals, sorted_idx = torch.sort(scaled, dim=-1, descending=True)
            sorted_probs = F.softmax(sorted_vals, dim=-1)
            keep = sorted_probs.cumsum(dim=-1) <= self.top_p
            keep[..., 0] = True
            sorted_vals = torch.where(
                keep, sorted_vals, torch.full_like(sorted_vals, float("-inf"))
            )
            scaled = torch.full_like(scaled, float("-inf"))
            scaled.scatter_(-1, sorted_idx, sorted_vals)

        probs_final = F.softmax(scaled, dim=-1)
        ids = torch.multinomial(probs_final, num_samples=1).squeeze(-1)
        orig_probs = F.softmax(logits, dim=-1)
        picked_probs = torch.gather(
            orig_probs, dim=-1, index=ids.unsqueeze(-1)
        ).squeeze(-1)
        return ids, picked_probs

    def _set_denoise_flags(self, forward_batch: ForwardBatch) -> None:
        forward_batch.dllm_force_causal = False
        forward_batch.dllm_force_bidir_mask = True
        forward_batch.dllm_bidir_custom_mask = self._hybrid_diffusion_mask.to(
            device=forward_batch.input_ids.device
        )
        forward_batch.dllm_gdn_persist_state = False
        forward_batch.dllm_gdn_causal_mode = 2
        forward_batch.dllm_gdn_num_clean = 1
        forward_batch.dllm_gdn_cache_intermediate_for_commit = True
        forward_batch.dllm_gdn_save_for_commit = False
        forward_batch.dllm_gdn_block_size = self.block_size
        forward_batch.dllm_return_argmax_only = False
        forward_batch.dllm_return_topk_probs = False

    def _set_commit_flags(self, forward_batch: ForwardBatch) -> None:
        forward_batch.dllm_force_causal = True
        forward_batch.dllm_force_bidir_mask = False
        forward_batch.dllm_bidir_custom_mask = None
        forward_batch.dllm_gdn_persist_state = True
        forward_batch.dllm_gdn_causal_mode = 1
        forward_batch.dllm_gdn_num_clean = 0
        forward_batch.dllm_gdn_cache_intermediate_for_commit = False
        forward_batch.dllm_gdn_save_for_commit = False
        forward_batch.dllm_gdn_block_size = self.block_size
        forward_batch.dllm_return_argmax_only = False
        forward_batch.dllm_return_topk_probs = False

    def _clear_flags(self, forward_batch: ForwardBatch) -> None:
        forward_batch.dllm_force_causal = False
        forward_batch.dllm_force_bidir_mask = False
        forward_batch.dllm_bidir_custom_mask = None
        forward_batch.dllm_gdn_persist_state = None
        forward_batch.dllm_gdn_cache_intermediate_for_commit = False
        forward_batch.dllm_gdn_save_for_commit = False
        forward_batch.dllm_return_argmax_only = False
        forward_batch.dllm_return_topk_probs = False

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        overlap_fn=None,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        mask_index = forward_batch.input_ids == self.mask_id
        rpx_list = self._req_pool_indices_cpu(forward_batch)

        if torch.sum(mask_index).item() == 0:
            out = self._forward_with_metrics(
                model_runner,
                forward_batch,
                modes="prefill",
            )
            self._stats["total_forwards"] += 1
            self._stats["prefill_forwards"] += 1
            logits_output = out.logits_output
            if logits_output is not None:
                self._capture_prefill_seeds(logits_output, forward_batch, rpx_list)
            self._flush_forward_timings()
            return logits_output, [], out.can_run_graph

        expected_tokens = batch_size * self.block_size
        if forward_batch.input_ids.numel() != expected_tokens:
            raise RuntimeError(
                "LowConfidenceShiftHybridDiffusion only supports pure decode batches with "
                f"{self.block_size} tokens per request. Got "
                f"{forward_batch.input_ids.numel()} tokens for batch_size={batch_size}."
            )
        if getattr(forward_batch, "dllm_attn_mask_types_cpu", None) is not None:
            raise RuntimeError(
                "LowConfidenceShiftHybridDiffusion decode must not be mixed with inline "
                "prefill requests."
            )

        start_list: List[int] = []
        for bid in range(batch_size):
            block_start = bid * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask_index = block_input_ids == self.mask_id
            start = self.block_size - torch.sum(block_mask_index).item()
            start_list.append(start)

            if start == 0 and bool(block_mask_index[0].item()):
                rpx = int(rpx_list[bid])
                seed = self._seed_tokens.get(rpx)
                if seed is None:
                    raise RuntimeError(
                        "LowConfidenceShiftHybridDiffusion is missing the decode seed for "
                        f"req_pool_idx={rpx}. Ensure prompt prefill runs before "
                        "the first decode block."
                    )
                forward_batch.input_ids[block_start] = seed

        step_budget = (
            self.maximum_unroll if self.maximum_unroll > 0 else self.block_size
        )
        logits_output = None
        can_run_cuda_graph = False
        for step in range(step_budget):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            self._set_denoise_flags(forward_batch)
            out = self._forward_with_metrics(
                model_runner,
                forward_batch,
                modes="diffusion_denoise",
                diffusion_steps=True,
                gdn_restores=True,
                recomputed=step > 0,
            )
            self._stats["total_forwards"] += 1
            self._stats["decode_forwards"] += 1
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            full_logits = logits_output.full_logits
            if full_logits is None:
                raise RuntimeError(
                    "LowConfidenceShiftHybridDiffusion requires full logits during denoise. "
                    "Disable dLLM argmax/top-k logits shortcuts for this algorithm."
                )
            is_last_step = step == step_budget - 1

            for bid in range(batch_size):
                block_start = bid * self.block_size
                block_end = block_start + self.block_size
                block_input_ids = forward_batch.input_ids[block_start:block_end]
                block_mask_index = block_input_ids == self.mask_id
                if not bool(block_mask_index.any().item()):
                    continue

                shifted_logits = full_logits[block_start : block_end - 1]
                dummy = torch.zeros(
                    (1, shifted_logits.shape[-1]),
                    dtype=shifted_logits.dtype,
                    device=shifted_logits.device,
                )
                curr_logits = torch.cat([dummy, shifted_logits], dim=0)
                sampled, probs = self._sample_tokens(curr_logits)

                block_mask_index[0] = False
                sampled = torch.where(block_mask_index, sampled, block_input_ids)
                confidence = torch.where(
                    block_mask_index,
                    probs,
                    torch.tensor(float("-inf"), dtype=probs.dtype, device=probs.device),
                )

                if is_last_step:
                    transfer_index = block_mask_index
                else:
                    transfer_index = confidence > self.threshold
                    if transfer_index.sum().item() == 0:
                        _, select_index = torch.topk(confidence, k=1)
                        transfer_index[select_index] = True

                block_input_ids[transfer_index] = sampled[transfer_index]

        self._set_commit_flags(forward_batch)
        out = self._forward_with_metrics(
            model_runner,
            forward_batch,
            modes="diffusion_commit",
            recomputed=True,
        )
        self._stats["total_forwards"] += 1
        self._stats["decode_forwards"] += 1
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        if logits_output.full_logits is None:
            raise RuntimeError(
                "LowConfidenceShiftHybridDiffusion requires full logits from the causal "
                "commit forward to capture the next seed."
            )
        self._capture_next_seeds(logits_output.full_logits, rpx_list, batch_size)

        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]
        self._stats["total_tokens"] += sum(len(t) for t in next_token_ids_list)
        self._set_output_token_modes(
            forward_batch,
            token_is_ar=[
                [start_list[bid] + index == 0 for index in range(len(tokens))]
                for bid, tokens in enumerate(next_token_ids_list)
            ],
        )
        self._clear_flags(forward_batch)
        self._flush_forward_timings()
        return logits_output, next_token_ids_list, can_run_cuda_graph

    @staticmethod
    def _req_pool_indices_cpu(forward_batch: ForwardBatch) -> List[int]:
        req_pool_indices = forward_batch.req_pool_indices
        if torch.is_tensor(req_pool_indices):
            return [int(x) for x in req_pool_indices.detach().cpu().tolist()]
        return [int(x) for x in req_pool_indices]

    def _capture_prefill_seeds(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
        rpx_list: List[int],
    ) -> None:
        full_logits = getattr(logits_output, "full_logits", None)
        if full_logits is None:
            next_logits = getattr(logits_output, "next_token_logits", None)
            if next_logits is None:
                return
            seed_ids, _ = self._sample_tokens(next_logits)
            for bid, seed in enumerate(seed_ids.detach().cpu().tolist()):
                self._seed_tokens[int(rpx_list[bid])] = int(seed)
            return

        extend_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_lens is None:
            return
        if torch.is_tensor(extend_lens):
            extend_lens_list = [int(x) for x in extend_lens.detach().cpu().tolist()]
        else:
            extend_lens_list = [int(x) for x in extend_lens]

        last_indices: List[int] = []
        valid_bids: List[int] = []
        offset = 0
        for bid, n_new in enumerate(extend_lens_list):
            if n_new > 0:
                last_indices.append(offset + n_new - 1)
                valid_bids.append(bid)
            offset += n_new
        if not last_indices:
            return

        idx = torch.tensor(last_indices, dtype=torch.long, device=full_logits.device)
        seed_ids, _ = self._sample_tokens(full_logits[idx])
        for out_bid, seed in zip(valid_bids, seed_ids.detach().cpu().tolist()):
            self._seed_tokens[int(rpx_list[out_bid])] = int(seed)

    def _capture_next_seeds(
        self, full_logits: torch.Tensor, rpx_list: List[int], batch_size: int
    ) -> None:
        last_indices = [
            (bid + 1) * self.block_size - 1 for bid in range(batch_size)
        ]
        idx = torch.tensor(last_indices, dtype=torch.long, device=full_logits.device)
        seed_ids, _ = self._sample_tokens(full_logits[idx])
        for bid, seed in enumerate(seed_ids.detach().cpu().tolist()):
            self._seed_tokens[int(rpx_list[bid])] = int(seed)


Algorithm = LowConfidenceShiftHybridDiffusion
