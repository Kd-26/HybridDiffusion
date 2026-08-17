"""Tests for DLLM block diffusion mask sampling behavior."""

import unittest
from unittest.mock import patch

import torch

from torchtitan.models.qwen3.model.model_dllm import DLLM, DLLMModelArgs


class TestDLLMMaskSampling(unittest.TestCase):
    """Verify DLLM uses Fast-dLLM v2 style iid per-block sampling."""

    def _make_model(
        self,
        *,
        block_size: int = 2,
        complementary_mask: bool = False,
        antithetic_sampling: bool = True,
    ) -> DLLM:
        args = DLLMModelArgs(
            dim=8,
            n_layers=1,
            n_heads=1,
            n_kv_heads=1,
            vocab_size=32,
            head_dim=8,
            hidden_dim=16,
            max_seq_len=8,
            block_size=block_size,
            complementary_mask=complementary_mask,
            antithetic_sampling=antithetic_sampling,
            use_flex_attention=False,
        )
        return DLLM(args)

    @staticmethod
    def _sample_block_probs_iid(trials: int, batch_size: int, num_blocks: int) -> torch.Tensor:
        return torch.rand(trials, batch_size, num_blocks)

    @staticmethod
    def _sample_block_probs_stratified(
        trials: int,
        batch_size: int,
        num_blocks: int,
    ) -> torch.Tensor:
        num_slots = batch_size * num_blocks
        base = torch.rand(trials, num_slots)
        strata = (base + torch.arange(num_slots, dtype=base.dtype)) / num_slots
        permutation = torch.argsort(torch.rand(trials, num_slots), dim=-1)
        return torch.gather(strata, 1, permutation).view(trials, batch_size, num_blocks)

    @staticmethod
    def _sample_block_probs_global_seq(
        trials: int,
        batch_size: int,
        num_blocks: int,
    ) -> torch.Tensor:
        return torch.rand(trials, batch_size, 1).expand(-1, -1, num_blocks)

    @classmethod
    def _sample_block_rates(
        cls,
        method: str,
        *,
        trials: int,
        batch_size: int,
        num_blocks: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if method == "iid":
            block_probs = cls._sample_block_probs_iid(trials, batch_size, num_blocks)
        elif method == "stratified":
            block_probs = cls._sample_block_probs_stratified(
                trials,
                batch_size,
                num_blocks,
            )
        elif method == "global_seq":
            block_probs = cls._sample_block_probs_global_seq(
                trials,
                batch_size,
                num_blocks,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        token_probs = block_probs.unsqueeze(-1).expand(-1, -1, -1, block_size)
        token_masks = torch.rand(trials, batch_size, num_blocks, block_size) <= token_probs
        block_counts = token_masks.sum(dim=-1)
        block_rates = block_counts.float() / block_size
        return block_counts, block_rates

    def test_forward_diffusion_uses_iid_per_block_sampling(self):
        model = self._make_model(antithetic_sampling=False)
        x0_embeds = torch.randn(2, 4, 8)
        labels = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        block_probs = torch.tensor([[0.10, 0.20], [0.30, 0.40]])
        move_probs = torch.tensor(
            [[0.15, 0.05, 0.25, 0.10], [0.35, 0.25, 0.45, 0.35]]
        )
        expected_probs = block_probs.repeat_interleave(2, dim=-1)
        expected_masked = move_probs <= expected_probs

        with patch(
            "torchtitan.models.qwen3.model.model_dllm.torch.rand",
            side_effect=[block_probs, move_probs],
        ):
            _, masked_indices, _, _ = model.forward_diffusion(x0_embeds, labels=labels)

        self.assertTrue(torch.equal(masked_indices, expected_masked))

    def test_forward_diffusion_antithetic_sampling_permuted_strata(self):
        model = self._make_model(antithetic_sampling=True)
        x0_embeds = torch.randn(2, 4, 8)
        labels = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        base_probs = torch.tensor([[0.20, 0.40], [0.60, 0.80]])
        permutation = torch.tensor([2, 0, 3, 1])
        move_probs = torch.tensor(
            [[0.60, 0.70, 0.02, 0.08], [0.90, 0.98, 0.30, 0.40]]
        )

        flat = (base_probs.reshape(-1) + torch.arange(4, dtype=base_probs.dtype)) / 4
        expected_block_probs = flat[permutation].view(2, 2)
        expected_probs = expected_block_probs.repeat_interleave(2, dim=-1)
        expected_masked = move_probs <= expected_probs

        with patch(
            "torchtitan.models.qwen3.model.model_dllm.torch.rand",
            side_effect=[base_probs, move_probs],
        ), patch(
            "torchtitan.models.qwen3.model.model_dllm.torch.randperm",
            return_value=permutation,
        ):
            _, masked_indices, _, _ = model.forward_diffusion(x0_embeds, labels=labels)

        self.assertTrue(torch.equal(masked_indices, expected_masked))

    def test_forward_diffusion_never_masks_ignored_labels(self):
        model = self._make_model()
        x0_embeds = torch.randn(1, 4, 8)
        labels = torch.tensor([[11, -100, 13, -100]])

        with patch(
            "torchtitan.models.qwen3.model.model_dllm.torch.rand",
            side_effect=[torch.ones((1, 2)), torch.zeros((1, 4))],
        ):
            _, masked_indices, _, _ = model.forward_diffusion(x0_embeds, labels=labels)

        expected = torch.tensor([[True, False, True, False]])
        self.assertTrue(torch.equal(masked_indices, expected))

    def test_complementary_mask_remains_complement_on_valid_positions(self):
        model = self._make_model(
            complementary_mask=True,
            antithetic_sampling=False,
        )
        x0_embeds = torch.randn(1, 4, 8)
        labels = torch.tensor([[21, -100, 23, 24]])

        block_probs = torch.tensor([[0.30, 0.60]])
        move_probs = torch.tensor([[0.20, 0.10, 0.70, 0.50]])

        with patch(
            "torchtitan.models.qwen3.model.model_dllm.torch.rand",
            side_effect=[block_probs, move_probs],
        ):
            _, masked_indices, _, _ = model.forward_diffusion(x0_embeds, labels=labels)

        original = torch.tensor([[True, False, False, True]])
        complementary = torch.tensor([[False, False, True, False]])

        self.assertEqual(masked_indices.shape, (2, 4))
        self.assertTrue(torch.equal(masked_indices[:1], original))
        self.assertTrue(torch.equal(masked_indices[1:], complementary))

    def test_block_count_marginal_matches_uniform_for_all_three_samplers(self):
        trials = 12000
        batch_size = 3
        num_blocks = 4
        block_size = 4
        uniform_pmf = torch.full((block_size + 1,), 1.0 / (block_size + 1))

        for method in ("iid", "stratified", "global_seq"):
            block_counts, _ = self._sample_block_rates(
                method,
                trials=trials,
                batch_size=batch_size,
                num_blocks=num_blocks,
                block_size=block_size,
            )
            slot_counts = block_counts[:, 0, 0]
            pmf = torch.bincount(slot_counts, minlength=block_size + 1).float() / trials
            self.assertLess(torch.max(torch.abs(pmf - uniform_pmf)).item(), 0.025)

    def test_all_three_samplers_have_no_slot_mean_bias(self):
        trials = 12000
        batch_size = 3
        num_blocks = 4
        block_size = 4

        for method in ("iid", "stratified", "global_seq"):
            _, block_rates = self._sample_block_rates(
                method,
                trials=trials,
                batch_size=batch_size,
                num_blocks=num_blocks,
                block_size=block_size,
            )
            slot_means = block_rates.mean(dim=0)
            self.assertLess(torch.max(torch.abs(slot_means - 0.5)).item(), 0.02)
            self.assertLess((slot_means.max() - slot_means.min()).item(), 0.03)

    def test_global_seq_reference_has_same_block_variance_but_larger_batch_variance(self):
        trials = 12000
        batch_size = 3
        num_blocks = 4
        block_size = 4

        all_block_vars = {}
        batch_mean_vars = {}
        for method in ("iid", "stratified", "global_seq"):
            _, block_rates = self._sample_block_rates(
                method,
                trials=trials,
                batch_size=batch_size,
                num_blocks=num_blocks,
                block_size=block_size,
            )
            all_block_vars[method] = block_rates[:, 0, 0].var(unbiased=False).item()
            batch_mean_vars[method] = block_rates.mean(dim=(1, 2)).var(unbiased=False).item()

        expected_block_var = (block_size + 2) / (12 * block_size)
        expected_global_seq_batch_var = (
            1 / (12 * batch_size) + 1 / (6 * block_size * batch_size * num_blocks)
        )

        for method in ("iid", "stratified", "global_seq"):
            self.assertAlmostEqual(all_block_vars[method], expected_block_var, delta=0.01)

        self.assertLess(batch_mean_vars["stratified"], batch_mean_vars["iid"])
        self.assertLess(batch_mean_vars["iid"], batch_mean_vars["global_seq"])
        self.assertAlmostEqual(
            batch_mean_vars["global_seq"],
            expected_global_seq_batch_var,
            delta=0.003,
        )


if __name__ == "__main__":
    unittest.main()
