from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


def sample_tokens_with_confidence(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample tokens and return their probabilities under the base model.

    The confidence used by denoising thresholds is always measured from the
    unfiltered model distribution. Temperature, top-k, and top-p only affect
    which token is proposed.
    """
    if temperature <= 0.0:
        token_ids = torch.argmax(logits, dim=-1)
    else:
        scaled_logits = logits.float() / temperature

        if top_k > 0:
            k = min(top_k, scaled_logits.shape[-1])
            topk_values, topk_indices = torch.topk(
                scaled_logits, k=k, dim=-1
            )
            scaled_logits = torch.full_like(scaled_logits, float("-inf"))
            scaled_logits.scatter_(-1, topk_indices, topk_values)

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                scaled_logits, dim=-1, descending=True
            )
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            remove = sorted_probs.cumsum(dim=-1) > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits.masked_fill_(remove, float("-inf"))
            scaled_logits = torch.full_like(scaled_logits, float("-inf"))
            scaled_logits.scatter_(-1, sorted_indices, sorted_logits)

        sampling_probs = F.softmax(scaled_logits, dim=-1)
        token_ids = torch.multinomial(sampling_probs, num_samples=1).squeeze(-1)

    base_probs = F.softmax(logits.float(), dim=-1)
    confidence = torch.gather(
        base_probs, dim=-1, index=token_ids.unsqueeze(-1)
    ).squeeze(-1)
    return token_ids, confidence


class LowConfidence(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.causal_xt = config.algorithm_config.get("causal_xt", False)
        self.temperature = float(config.algorithm_config.get("temperature", 0.0))
        self.top_k = int(config.algorithm_config.get("top_k", 0))
        self.top_p = float(config.algorithm_config.get("top_p", 1.0))
        if self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        overlap_fn=None,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        # Here, the forward_batch full logits contains all the blocks
        # such as [dllm_block_size * batch_size, hidden_size]
        start_list = []
        mask_index = forward_batch.input_ids == self.mask_id

        # Fast path: if there is no mask token, forward and save kv cache
        if torch.sum(mask_index).item() == 0:
            out = self._forward_with_metrics(
                model_runner,
                forward_batch,
                modes="prefill",
            )
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            next_token_ids = []
            self._flush_forward_timings()
            return logits_output, next_token_ids, can_run_cuda_graph

        # Calculate start positions for each block
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask_index = block_input_ids == self.mask_id
            start = self.block_size - torch.sum(block_mask_index).item()
            start_list.append(start)

        for _iter in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            # GDN dLLM: don't persist state during intermediate iterations
            forward_batch.dllm_gdn_persist_state = False
            forward_batch.dllm_gdn_causal_mode = int(self.causal_xt)
            forward_batch.dllm_gdn_block_size = self.block_size
            out = self._forward_with_metrics(
                model_runner,
                forward_batch,
                modes="diffusion_denoise",
                diffusion_steps=True,
            )
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            assert batch_size == forward_batch.input_ids.shape[0] // self.block_size
            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end,
                ]
                block_mask_index = block_input_ids == self.mask_id
                if torch.sum(block_mask_index).item() == 0:
                    continue
                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end,
                ]

                x, p = sample_tokens_with_confidence(
                    curr_logits,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                )
                x = torch.where(block_mask_index, x, block_input_ids)
                confidence = torch.where(block_mask_index, p, -np.inf)

                transfer_index = confidence > self.threshold

                if transfer_index.sum().item() == 0:
                    _, select_index = torch.topk(confidence, k=1)
                    transfer_index[select_index] = True

                block_input_ids[transfer_index] = x[transfer_index]

        # Final commit forward: persist GDN state
        forward_batch.dllm_gdn_persist_state = True
        forward_batch.dllm_gdn_causal_mode = int(self.causal_xt)
        forward_batch.dllm_gdn_block_size = self.block_size
        out = self._forward_with_metrics(
            model_runner,
            forward_batch,
            modes=[()] * batch_size,
        )
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        # Here next token ids is tricky to implement the dynamic lengths,
        # so we return a list of tensors
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]

        self._set_output_token_modes(
            forward_batch,
            token_is_ar=[[False] * len(tokens) for tokens in next_token_ids_list],
        )
        self._flush_forward_timings()

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = LowConfidence
