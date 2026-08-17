"""Regression tests for Qwen3 diffusion-training token IDs.

The unit tests use a tokenizer stub and run without model assets. The three
tokenizer-metadata checks are optional integration tests enabled by setting
``QWEN3_TOKENIZER_PATH`` to a local Qwen3 tokenizer directory.
"""

import os
import unittest
from unittest.mock import MagicMock


def _get_qwen3_tokenizer_path() -> str | None:
    path = os.environ.get("QWEN3_TOKENIZER_PATH")
    if path and os.path.isfile(os.path.join(path, "tokenizer_config.json")):
        return path
    return None


QWEN3_TOKENIZER_PATH = _get_qwen3_tokenizer_path()


class TestDLLMTokenIds(unittest.TestCase):
    """Verify config resolution without requiring external model assets."""

    def setUp(self):
        self.tokenizer = MagicMock()
        self.tokenizer.get_vocab_size.return_value = 151936

    def _make_job_config(self, mask_token_id=-1, pad_token_id=-1):
        job_config = MagicMock()
        job_config.dllm.mask_token_id = mask_token_id
        job_config.dllm.pad_token_id = pad_token_id
        job_config.dllm.block_size = 16
        job_config.training.seq_len = 4096
        return job_config

    def test_vocab_size_covers_mask_slot(self):
        from torchtitan.models.qwen3.model.model_dllm import DLLMModelArgs

        self.assertGreater(151936, DLLMModelArgs.QWEN3_MASK_TOKEN_ID)

    def test_model_args_default_is_vocab_size_minus_1(self):
        from torchtitan.models.qwen3.model.model_dllm import DLLMModelArgs

        args = DLLMModelArgs(vocab_size=151936)
        self.assertEqual(args.mask_token_id, 151935)

    def test_model_args_debug_model(self):
        from torchtitan.models.qwen3.model.model_dllm import DLLMModelArgs

        args = DLLMModelArgs(vocab_size=2048)
        self.assertEqual(args.mask_token_id, 2047)

    def test_model_args_explicit_override(self):
        from torchtitan.models.qwen3.model.model_dllm import DLLMModelArgs

        args = DLLMModelArgs(vocab_size=151936, mask_token_id=151669)
        self.assertEqual(args.mask_token_id, 151669)

    def test_default_backward_compat(self):
        from torchtitan.hf_datasets.text_datasets import _get_dllm_common_kwargs

        kwargs = _get_dllm_common_kwargs(self.tokenizer, self._make_job_config())
        self.assertEqual(kwargs["mask_token_id"], 151935)
        self.assertEqual(kwargs["pad_token_id"], 151935)

    def test_explicit_experiment_ids(self):
        from torchtitan.hf_datasets.text_datasets import _get_dllm_common_kwargs

        kwargs = _get_dllm_common_kwargs(
            self.tokenizer,
            self._make_job_config(mask_token_id=151669, pad_token_id=151643),
        )
        self.assertEqual(kwargs["mask_token_id"], 151669)
        self.assertEqual(kwargs["pad_token_id"], 151643)

    def test_explicit_overrides_always_win(self):
        from torchtitan.hf_datasets.text_datasets import _get_dllm_common_kwargs

        kwargs = _get_dllm_common_kwargs(
            self.tokenizer,
            self._make_job_config(mask_token_id=99999, pad_token_id=88888),
        )
        self.assertEqual(kwargs["mask_token_id"], 99999)
        self.assertEqual(kwargs["pad_token_id"], 88888)

    def test_only_mask_set_pad_follows(self):
        from torchtitan.hf_datasets.text_datasets import _get_dllm_common_kwargs

        kwargs = _get_dllm_common_kwargs(
            self.tokenizer,
            self._make_job_config(mask_token_id=151669, pad_token_id=-1),
        )
        self.assertEqual(kwargs["mask_token_id"], 151669)
        self.assertEqual(kwargs["pad_token_id"], 151669)

    def test_constants_defined(self):
        from torchtitan.hf_datasets.text_datasets import DLLMTextDataset
        from torchtitan.models.qwen3.model.model_dllm import DLLMModelArgs

        self.assertEqual(DLLMModelArgs.QWEN3_MASK_TOKEN_ID, 151669)
        self.assertEqual(DLLMModelArgs.QWEN3_PAD_TOKEN_ID, 151643)
        self.assertEqual(DLLMTextDataset.QWEN3_PAD_TOKEN_ID, 151643)


@unittest.skipIf(
    QWEN3_TOKENIZER_PATH is None,
    "set QWEN3_TOKENIZER_PATH to run tokenizer metadata checks",
)
class TestQwen3TokenizerMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from torchtitan.components.tokenizer import HuggingFaceTokenizer

        cls.tokenizer = HuggingFaceTokenizer(QWEN3_TOKENIZER_PATH)

    def test_pad_id_is_endoftext(self):
        self.assertEqual(self.tokenizer.pad_id, 151643)
        self.assertEqual(self.tokenizer.pad_token, "<|endoftext|>")

    def test_mask_id_is_none_for_stock_qwen3(self):
        self.assertIsNone(self.tokenizer.mask_id)
        self.assertIsNone(self.tokenizer.mask_token)

    def test_eos_id_is_im_end(self):
        self.assertEqual(self.tokenizer.eos_id, 151645)


if __name__ == "__main__":
    unittest.main()
