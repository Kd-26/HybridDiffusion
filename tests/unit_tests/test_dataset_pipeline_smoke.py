"""Synthetic smoke tests for the retained dataset build and mixture paths.

These tests intentionally create only tiny local Arrow datasets. They must
never download or materialize the production datasets.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from datasets import Dataset, DatasetDict, load_from_disk

from scripts import build_domain_dataset as domain_builder
from scripts import build_if_dataset as if_builder
from scripts import build_long_sft_dataset as long_builder
from torchtitan.config.job_config import DatasetMixEntry, JobConfig
from torchtitan.hf_datasets.text_datasets import build_dllm_sft_dataloader


class TinyTokenizer:
    eos_id = 2

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        ids = [3 + (ord(char) % 251) for char in text]
        if add_bos:
            ids.insert(0, 1)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def __call__(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return SimpleNamespace(input_ids=[self.encode(text) for text in texts])

    def get_vocab_size(self) -> int:
        return 512


def _patch_tokenizer(monkeypatch, module) -> None:
    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: TinyTokenizer(),
    )


def _messages(answer: str, reasoning: str | None = None) -> list[dict]:
    assistant = {"role": "assistant", "content": answer}
    if reasoning is not None:
        assistant["reasoning_content"] = reasoning
    return [
        {"role": "user", "content": "Please follow this synthetic instruction."},
        assistant,
    ]


def test_long_sft_all_source_families_and_think_filter(tmp_path, monkeypatch):
    _patch_tokenizer(monkeypatch, long_builder)
    root = tmp_path / "sources"
    root.mkdir()

    source_paths = long_builder.resolve_source_paths(str(root))
    reasoning_rows = Dataset.from_list(
        [
            {
                "messages": _messages(
                    "A sufficiently long synthetic final answer.",
                    "A sufficiently long synthetic reasoning trace.",
                )
            },
            {"messages": _messages("<think>unfinished synthetic trace")},
        ]
    )
    for cfg in long_builder.DATASETS_WITH_REASONING.values():
        reasoning_rows.save_to_disk(source_paths[cfg["path_key"]])

    Dataset.from_list(
        [
            {"category": "math", "messages": _messages("<think></think>\n\nvalid")},
            {"category": "math", "messages": _messages("<think>unfinished")},
        ]
    ).save_to_disk(source_paths["post_training_v2"])

    llama_splits = {}
    for split in long_builder.LLAMA_NEMOTRON_SPLITS:
        llama_splits[split] = Dataset.from_list(
            [
                {
                    "input": [{"role": "user", "content": "synthetic prompt"}],
                    "output": "<think></think>\n\nA long enough synthetic answer.",
                },
                {
                    "input": [{"role": "user", "content": "synthetic prompt"}],
                    "output": "<think>unfinished",
                },
            ]
        )
    DatasetDict(llama_splits).save_to_disk(source_paths["llama_nemotron"])

    output = tmp_path / "long-output"
    long_builder.build_dataset(
        threshold=10,
        output_dir=str(output),
        model_name="synthetic-tokenizer",
        num_proc=1,
        data_root=str(root),
    )

    built = load_from_disk(output)
    assert len(built) == 10
    assert set(built.column_names) == {
        "messages_json",
        "category",
        "source",
        "total_tokens",
        "assistant_tokens",
    }
    assert all(
        long_builder._messages_have_balanced_think(messages_json)
        for messages_json in built["messages_json"]
    )


def test_math_domain_full_builder_filters_unclosed_think(tmp_path, monkeypatch):
    _patch_tokenizer(monkeypatch, domain_builder)
    root = tmp_path / "sources"
    root.mkdir()
    source_paths = domain_builder.resolve_source_paths(str(root))

    DatasetDict(
        {
            "math": Dataset.from_list(
                [
                    {
                        "reasoning": "on",
                        "input": [{"role": "user", "content": "synthetic math"}],
                        "output": "<think></think>\n\n42",
                    },
                    {
                        "reasoning": "off",
                        "input": [{"role": "user", "content": "excluded"}],
                        "output": "excluded",
                    },
                ]
            )
        }
    ).save_to_disk(source_paths["llama_nemotron"])
    Dataset.from_list(
        [
            {"messages": _messages("42", "synthetic proof")},
            {"messages": _messages("<think>unfinished")},
        ]
    ).save_to_disk(source_paths["math_proofs_v1"])

    output = tmp_path / "math-output"
    domain_builder.build_domain_dataset(
        domain="math",
        output_dir=str(output),
        model_name="synthetic-tokenizer",
        num_proc=1,
        data_root=str(root),
    )

    built = load_from_disk(output)
    assert len(built) == 2
    assert set(built["source"]) == {
        "llama_nemotron_math",
        "nemotron_math_proofs_v1",
    }
    assert all(
        not domain_builder._has_unclosed_think(messages_json)
        for messages_json in built["messages_json"]
    )


def test_if_full_builder_selection_balance_and_dedup(tmp_path, monkeypatch):
    _patch_tokenizer(monkeypatch, if_builder)
    root = tmp_path / "sources"
    root.mkdir()
    source_paths = if_builder.resolve_source_paths(str(root))

    duplicate = _messages("shared answer")
    Dataset.from_list(
        [
            {"messages": duplicate},
            {"messages": _messages("<think>unfinished")},
        ]
    ).save_to_disk(source_paths["cascade1_if"])
    Dataset.from_list(
        [
            {"messages": duplicate},
            {"messages": _messages("cascade two answer")},
        ]
    ).save_to_disk(source_paths["cascade2_if"])
    Dataset.from_list(
        [
            {
                "_split": "chat_if",
                "capability_target": "instruction_following",
                "messages": _messages("chat IF answer"),
            },
            {
                "_split": "structured_outputs",
                "capability_target": "structured_output",
                "messages": _messages('{"answer": 42}'),
            },
            {
                "_split": "chat_if",
                "capability_target": "conversation",
                "messages": _messages("must be excluded"),
            },
        ]
    ).save_to_disk(source_paths["chat_v1"])

    output = tmp_path / "if-output"
    if_builder.build_if_dataset(
        data_root=str(root),
        output_dir=str(output),
        model_name="synthetic-tokenizer",
        num_proc=1,
        cascade2_mode="raw_open",
    )

    built = load_from_disk(output)
    assert len(built) == 4
    assert built.column_names == if_builder.OUTPUT_COLUMNS
    assert len(set(built["messages_json"])) == len(built)
    for messages_json in built["messages_json"]:
        for message in json.loads(messages_json):
            if message["role"] == "assistant":
                assert if_builder._think_tags_balanced(message["content"])


def _write_mixture_component(path, label: str) -> None:
    Dataset.from_list(
        [
            {
                "messages_json": json.dumps(
                    _messages(f"{label} synthetic answer number {index}"),
                    ensure_ascii=False,
                ),
                "category": label,
                "source": label,
                "total_tokens": 32,
                "assistant_tokens": 16,
            }
            for index in range(12)
        ]
    ).save_to_disk(path)


def _mixture_config(paths) -> JobConfig:
    config = JobConfig()
    config.training.local_batch_size = 1
    config.training.seq_len = 96
    config.training.dataloader.num_workers = 0
    config.training.dataset_mix = [
        DatasetMixEntry(
            name=name,
            path=str(path),
            weight=weight,
        )
        for name, path, weight in zip(
            [
                "nemotron_long_sft_3k",
                "nemotron_math_domain",
                "nemotron_if_domain_cascade",
            ],
            paths,
            [0.4, 0.4, 0.2],
            strict=True,
        )
    ]
    config.dllm.block_size = 3
    config.dllm.mask_token_id = 510
    config.dllm.pad_token_id = 2
    config.dllm.sft_label_mode = "response"
    config.dllm.enable_packing = True
    return config


def _assert_batch_equal(left, right) -> None:
    left_inputs, left_labels = left
    right_inputs, right_labels = right
    assert left_inputs.keys() == right_inputs.keys()
    for key in left_inputs:
        assert torch.equal(left_inputs[key], right_inputs[key])
    assert torch.equal(left_labels, right_labels)


def test_local_three_way_mixture_packing_and_resume(tmp_path):
    paths = [tmp_path / name for name in ("long", "math", "if")]
    for path, label in zip(paths, ("long", "math", "if"), strict=True):
        _write_mixture_component(path, label)

    tokenizer = TinyTokenizer()
    config = _mixture_config(paths)
    dataloader = build_dllm_sft_dataloader(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=tokenizer,
        job_config=config,
        infinite=True,
    )
    iterator = iter(dataloader)
    first = next(iterator)
    assert first[0]["input"].shape == (1, 96)
    assert first[0]["doc_ids"].shape == (1, 96)
    state = dataloader.state_dict()
    expected = [next(iterator) for _ in range(4)]

    resumed = build_dllm_sft_dataloader(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=tokenizer,
        job_config=_mixture_config(paths),
        infinite=True,
    )
    resumed.load_state_dict(state)
    resumed_iterator = iter(resumed)
    actual = [next(resumed_iterator) for _ in range(4)]
    for expected_batch, actual_batch in zip(expected, actual, strict=True):
        _assert_batch_equal(expected_batch, actual_batch)
