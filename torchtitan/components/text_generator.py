# Copyright (c) HybridDiffusion contributors.
#
# Licensed under the repository License; see LICENSE in the repository root.

"""
Text generation module for periodic sampling during training.
Similar to Arceus's inference_in_train functionality.
"""

import torch
import torch.nn as nn
from torch.distributed import distributed_c10d as dist

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger


class TextGenerator:
    """Generate text samples during training for visualization."""

    def __init__(
        self,
        job_config: JobConfig,
        tokenizer: BaseTokenizer,
        device: torch.device,
    ):
        self.job_config = job_config
        self.tokenizer = tokenizer
        self.device = device
        self.inference_config = job_config.inference_in_train

    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        step: int,
    ) -> list[dict[str, str]]:
        """
        Generate text samples from prompts.

        Args:
            model: The model to use for generation
            step: Current training step

        Returns:
            List of dicts with 'prompt' and 'generated_text' keys
        """
        if not self.inference_config.enable:
            return []

        # Only generate on rank 0
        if dist.is_initialized() and dist.get_rank() != 0:
            return []

        # Set model to eval mode
        was_training = model.training
        model.eval()

        results = []

        try:
            for prompt in self.inference_config.prompts:
                # Tokenize prompt
                tokens = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
                tokens = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

                # Generate
                generated_tokens = self._generate_tokens(
                    model,
                    tokens,
                    max_new_tokens=self.inference_config.max_new_tokens,
                    temperature=self.inference_config.temperature,
                    top_k=self.inference_config.top_k,
                )

                # Decode (skip special tokens for cleaner output)
                generated_text = self.tokenizer.decode(generated_tokens[0].tolist(), skip_special_tokens=True)

                results.append({
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "step": step,
                })

                logger.info(f"Generated sample at step {step}:")
                logger.info(f"  Prompt: {prompt}")
                logger.info(f"  Generated: {generated_text[:100]}...")

        except Exception as e:
            logger.error(f"Error during text generation: {e}")

        finally:
            # Restore training mode
            if was_training:
                model.train()

        return results

    def _generate_tokens(
        self,
        model: nn.Module,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            model: The model
            prompt_tokens: Input prompt tokens [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            Generated tokens including prompt [batch_size, seq_len + max_new_tokens]
        """
        tokens = prompt_tokens.clone()

        max_seq_len = self.job_config.training.seq_len
        
        for _ in range(max_new_tokens):
            # Check sequence length limit
            if tokens.size(1) >= max_seq_len:
                logger.warning(f"Reached max sequence length {max_seq_len}, stopping generation")
                break
            
            # Forward pass
            logits = model(tokens)

            # Get logits for the last token
            logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # Apply temperature
            if temperature > 0 and temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0 and top_k < logits.size(-1):
                # Get top-k values and indices
                top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
                # Find the k-th largest value (minimum of top-k)
                kth_value = top_k_values[:, -1].unsqueeze(-1)  # [batch_size, 1]
                # Set all values below kth_value to -inf
                logits = torch.where(
                    logits < kth_value,
                    torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype),
                    logits
                )

            # Sample with numerical stability checks
            probs = torch.softmax(logits, dim=-1)
            
            # Check for invalid probabilities (inf, nan, or negative)
            if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)) or torch.any(probs < 0):
                logger.warning("Invalid probabilities detected, using uniform distribution as fallback")
                probs = torch.ones_like(probs) / probs.size(-1)
            
            # Ensure probabilities sum to 1 (with numerical stability)
            probs_sum = probs.sum(dim=-1, keepdim=True)
            if torch.any(probs_sum <= 0):
                logger.warning("Zero probability sum detected, using uniform distribution")
                probs = torch.ones_like(probs) / probs.size(-1)
            else:
                probs = probs / probs_sum
            
            # Clamp probabilities to valid range [0, 1]
            probs = torch.clamp(probs, min=0.0, max=1.0)
            
            try:
                next_token = torch.multinomial(probs, num_samples=1)
            except RuntimeError as e:
                logger.error(f"multinomial failed: {e}, probs stats: min={probs.min()}, max={probs.max()}, sum={probs.sum()}")
                # Fallback: use argmax
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)

            # Check for EOS token
            if hasattr(self.tokenizer, 'eos_id') and next_token.item() == self.tokenizer.eos_id:
                break

        return tokens

    def should_generate(self, step: int) -> bool:
        """Check if we should generate at this step."""
        if not self.inference_config.enable:
            return False
        return step > 0 and step % self.inference_config.freq == 0
