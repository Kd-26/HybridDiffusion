# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
import sys
import types
import unittest

import torch

# `text_datasets.py` imports ParallelAwareDataloader -> torchdata.StatefulDataLoader.
# The training dependency is optional in some test environments, so provide a
# small stub to keep these unit tests focused on label-mode behavior.
if "torchdata.stateful_dataloader" not in sys.modules:
    torchdata_module = types.ModuleType("torchdata")
    stateful_module = types.ModuleType("torchdata.stateful_dataloader")

    class _DummyStatefulDataLoader:
        pass

    stateful_module.StatefulDataLoader = _DummyStatefulDataLoader
    sys.modules.setdefault("torchdata", torchdata_module)
    sys.modules["torchdata.stateful_dataloader"] = stateful_module

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets.text_datasets import (
    DLLMSFTDataset,
    build_ar_sft_dataloader,
    build_dllm_dataloader,
)


class DummyTokenizer(BaseTokenizer):
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        tokens = [ord(ch) for ch in text]
        if add_bos:
            tokens = [100001] + tokens
        if add_eos:
            tokens = tokens + [100002]
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(tok) for tok in token_ids if tok < 256)

    def get_vocab_size(self) -> int:
        return 200000


def _make_dataset(
    mode: str,
    messages: list[dict[str, str]],
    *,
    seq_len: int = 256,
    pad_token_id: int = 199999,
) -> DLLMSFTDataset:
    dataset = DLLMSFTDataset.__new__(DLLMSFTDataset)
    dataset._tokenizer = DummyTokenizer()
    dataset._text_processor = lambda sample: sample
    dataset._data = [{"messages": messages}]
    dataset._sample_idx = 0
    dataset.messages_field = "messages"
    dataset.seq_len = seq_len
    dataset.pad_token_id = pad_token_id
    dataset.infinite = False
    dataset.dataset_name = "test_sft"
    dataset.sft_label_mode = mode
    dataset.enable_packing = False
    return dataset


def _expected_tokens_and_labels(
    tokenizer: BaseTokenizer,
    messages: list[dict[str, str]],
    mode: str,
) -> tuple[list[int], list[int]]:
    tokens: list[int] = []
    labels: list[int] = []

    last_query_idx = max(
        (idx for idx, message in enumerate(messages) if message["role"] == "user"),
        default=-1,
    )
    for idx, message in enumerate(messages):
        role = message["role"]
        content = message["content"]
        if (
            role == "assistant"
            and idx > last_query_idx >= 0
            and idx == len(messages) - 1
            and "<think>" not in content
        ):
            content = "<think>\n\n</think>\n\n" + content
        header_tokens = tokenizer.encode(
            f"<|im_start|>{role}\n", add_bos=False, add_eos=False
        )
        content_tokens = tokenizer.encode(content, add_bos=False, add_eos=False)
        end_tokens = tokenizer.encode("<|im_end|>\n", add_bos=False, add_eos=False)

        if mode == "response":
            train_header = False
            train_content = role == "assistant"
            train_end = role == "assistant"
        elif mode == "full":
            train_header = True
            train_content = True
            train_end = True
        else:
            train_header = False
            train_content = True
            train_end = False

        for span_tokens, train_span in (
            (header_tokens, train_header),
            (content_tokens, train_content),
            (end_tokens, train_end),
        ):
            tokens.extend(span_tokens)
            labels.extend(list(span_tokens) if train_span else [-100] * len(span_tokens))

    return tokens, labels


class TestDLLMSFTLabelMode(unittest.TestCase):
    def setUp(self):
        self.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "ask"},
            {"role": "assistant", "content": "answer"},
        ]
        self.tokenizer = DummyTokenizer()

    def test_response_mode_matches_current_behavior(self):
        dataset = _make_dataset("response", self.messages)
        tokens, labels = dataset._tokenize_chat_messages(self.messages)
        expected_tokens, expected_labels = _expected_tokens_and_labels(
            self.tokenizer, self.messages, "response"
        )

        self.assertEqual(tokens, expected_tokens)
        self.assertEqual(labels, expected_labels)

    def test_full_mode_trains_all_non_padding_tokens(self):
        dataset = _make_dataset("full", self.messages)
        tokens, labels = dataset._tokenize_chat_messages(self.messages)
        expected_tokens, expected_labels = _expected_tokens_and_labels(
            self.tokenizer, self.messages, "full"
        )

        self.assertEqual(tokens, expected_tokens)
        self.assertEqual(labels, expected_labels)
        self.assertNotIn(-100, labels)

    def test_full_content_mode_trains_content_only(self):
        dataset = _make_dataset("full_content", self.messages)
        tokens, labels = dataset._tokenize_chat_messages(self.messages)
        expected_tokens, expected_labels = _expected_tokens_and_labels(
            self.tokenizer, self.messages, "full_content"
        )

        self.assertEqual(tokens, expected_tokens)
        self.assertEqual(labels, expected_labels)

    def test_padding_stays_ignored_in_all_modes(self):
        messages = [{"role": "assistant", "content": "ok"}]

        for mode in ("response", "full", "full_content"):
            dataset = _make_dataset(mode, messages, seq_len=64)
            batch_inputs, batch_labels = next(iter(dataset))

            self.assertIsInstance(batch_inputs["input"], torch.Tensor)
            self.assertIsInstance(batch_labels, torch.Tensor)

            real_tokens, real_labels = _expected_tokens_and_labels(
                self.tokenizer, messages, mode
            )
            pad_len = 64 - len(real_tokens)

            self.assertEqual(batch_inputs["input"][: len(real_tokens)].tolist(), real_tokens)
            self.assertEqual(batch_labels[: len(real_labels)].tolist(), real_labels)
            self.assertEqual(
                batch_inputs["input"][len(real_tokens) :].tolist(),
                [dataset.pad_token_id] * pad_len,
            )
            self.assertEqual(
                batch_labels[len(real_labels) :].tolist(),
                [-100] * pad_len,
            )

    def test_non_sft_dllm_builder_rejects_non_default_label_mode(self):
        job_config = SimpleNamespace(
            dllm=SimpleNamespace(sft_label_mode="full"),
            training=SimpleNamespace(dataset_mix=[]),
        )

        with self.assertRaisesRegex(ValueError, "only applies to DLLM SFT"):
            build_dllm_dataloader(
                dp_world_size=1,
                dp_rank=0,
                tokenizer=self.tokenizer,
                job_config=job_config,
            )

    def test_ar_sft_builder_rejects_non_default_label_mode(self):
        job_config = SimpleNamespace(
            dllm=SimpleNamespace(sft_label_mode="full_content"),
        )

        with self.assertRaisesRegex(ValueError, "only applies to DLLM SFT"):
            build_ar_sft_dataloader(
                dp_world_size=1,
                dp_rank=0,
                tokenizer=self.tokenizer,
                job_config=job_config,
            )
