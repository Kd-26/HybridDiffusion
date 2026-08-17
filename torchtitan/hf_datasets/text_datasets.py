# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict
from functools import partial
from typing import Any, Callable
import json
import os

import torch

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import cache_s3_path_if_needed


def _load_c4_dataset(dataset_path: str, split: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, name="en", split=split, streaming=True)


def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]


def _load_llama_nemotron_sft(path: str):
    """Load Llama-Nemotron SFT data (lazy -- no upfront .map()).

    The raw arrow data has ``input`` (list of {role, content} dicts for
    system/user turns) and ``output`` (assistant response string).  The
    conversion to ``messages`` format is handled lazily in the
    sample_processor so we don't block on a full-dataset .map() at startup.
    """
    if os.path.exists(os.path.join(path, "dataset_dict.json")):
        return concatenate_datasets(list(load_from_disk(path).values()))
    elif os.path.exists(os.path.join(path, "dataset_info.json")):
        return load_from_disk(path)
    else:
        return concatenate_datasets(list(load_dataset(path).values()))


def _llama_nemotron_sample_processor(sample):
    """Convert Llama-Nemotron input/output format to unified messages format."""
    msgs = list(sample["input"])
    msgs.append({"role": "assistant", "content": sample["output"]})
    return {"messages": msgs}


def _long_sft_sample_processor(sample):
    """Decode ``messages_json`` back to a ``messages`` list for Long-SFT datasets."""
    sample["messages"] = json.loads(sample["messages_json"])
    return sample


def _merge_reasoning_content(sample):
    """Merge ``reasoning_content`` into ``content`` with ``<think>`` tags.

    For assistant messages that carry a separate ``reasoning_content`` field
    (e.g. Nemotron Science-v1, Math-Proofs-v1), prepend the reasoning chain
    wrapped in ``<think>...</think>`` so the model learns chain-of-thought.
    Non-assistant messages are left unchanged.
    """
    msgs = sample.get("messages", [])
    for m in msgs:
        reasoning = m.get("reasoning_content")
        if m.get("role") == "assistant" and reasoning:
            r = reasoning.strip("\n")
            c = (m.get("content", "") or "").lstrip("\n")
            m["content"] = f"<think>\n{r}\n</think>\n\n{c}"
    return sample


_TOOL_SCHEMA_TEMPLATE = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n{tool_json}\n</tools>\n\n"
    "For each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n'
    "</tool_call>"
)


def _format_tool_messages(sample):
    """Reformat tool-related fields to match Qwen3's chat template conventions.

    Applies three transformations in order:

    1. **Merge reasoning_content** — same as ``_merge_reasoning_content``.
    2. **Inject tool schema into system prompt** — if the sample has a
       ``tools`` column, append the Qwen3-style ``# Tools`` block (with
       ``<tools>`` XML tags) to the first system message's content.
    3. **Convert ``role=tool`` → ``role=user`` with ``<tool_response>`` tags**
       — consecutive tool messages are collapsed into a single user message,
       matching Qwen3's ``apply_chat_template`` output.
    4. **Flatten assistant ``tool_calls``** — if an assistant message has
       ``tool_calls`` but empty ``content``, render each call as a
       ``<tool_call>`` block inside the content so it is not skipped.
    """
    msgs = sample.get("messages", [])

    # --- Step 1: merge reasoning_content (same logic as _merge_reasoning_content) ---
    for m in msgs:
        reasoning = m.get("reasoning_content")
        if m.get("role") == "assistant" and reasoning:
            r = reasoning.strip("\n")
            c = (m.get("content", "") or "").lstrip("\n")
            m["content"] = f"<think>\n{r}\n</think>\n\n{c}"

    # --- Step 2: inject tools schema into first system message ---
    tools = sample.get("tools")
    if tools:
        tool_json_lines = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
        tools_block = _TOOL_SCHEMA_TEMPLATE.format(tool_json=tool_json_lines)
        for m in msgs:
            if m.get("role") == "system":
                m["content"] = m.get("content", "") + "\n\n" + tools_block
                break

    # --- Step 3 & 4: rewrite tool messages and flatten tool_calls ---
    new_msgs = []
    pending_tool_responses: list[str] = []

    def _flush_tool_responses():
        if pending_tool_responses:
            body = "\n".join(
                f"<tool_response>\n{resp}\n</tool_response>"
                for resp in pending_tool_responses
            )
            new_msgs.append({"role": "user", "content": body})
            pending_tool_responses.clear()

    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "") or ""

        if role == "tool":
            pending_tool_responses.append(content)
            continue

        # Non-tool message: flush any accumulated tool responses first
        _flush_tool_responses()

        if role == "assistant":
            tc = m.get("tool_calls")
            if tc:
                tc_blocks = []
                for call in tc:
                    func = call.get("function", call) if isinstance(call, dict) else call
                    name = func.get("name", "") if isinstance(func, dict) else ""
                    args = func.get("arguments", "{}") if isinstance(func, dict) else "{}"
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    tc_blocks.append(
                        f"<tool_call>\n"
                        f'{{"name": "{name}", "arguments": {args}}}\n'
                        f"</tool_call>"
                    )
                tc_text = "\n".join(tc_blocks)
                if content:
                    content = content + "\n" + tc_text
                else:
                    content = tc_text
            new_msgs.append({"role": role, "content": content})
        else:
            new_msgs.append({"role": role, "content": content})

    _flush_tool_responses()

    sample["messages"] = new_msgs
    return sample


def _find_last_query_index(messages: list[dict]) -> int:
    """Return the index of the last real user query in *messages*.

    Mirrors Qwen3's ``apply_chat_template`` logic: scan backwards for
    the last ``role=user`` message whose content is NOT a pure
    ``<tool_response>`` block.  Returns ``-1`` if no real user query
    is found (which suppresses ``<think>`` injection).
    """
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content", "")
        if isinstance(c, str) and not (
            c.startswith("<tool_response>") and c.endswith("</tool_response>")
        ):
            return i
    return -1


_SHUFFLE_ROW_THRESHOLD = 10_000_000


def _load_arrow_flat(path: str, seed: int = 42):
    """Load an Arrow dataset from *path*.

    Handles three directory layouts:
    1. ``dataset_dict.json`` present  -> DatasetDict, concatenate all splits
    2. No ``.arrow`` files but has subdirectories -> load each valid subdir,
       concatenate, and shuffle (deterministic)
    3. Otherwise -> single flat ``Dataset`` directory
    """
    if os.path.exists(os.path.join(path, "dataset_dict.json")):
        return concatenate_datasets(list(load_from_disk(path).values()))

    entries = os.listdir(path)
    subdirs = sorted(d for d in entries if os.path.isdir(os.path.join(path, d)))
    has_arrow_files = any(f.endswith(".arrow") for f in entries)

    if subdirs and not has_arrow_files:
        valid_subdirs = [
            d for d in subdirs
            if os.path.isfile(os.path.join(path, d, "dataset_info.json"))
            or os.path.isfile(os.path.join(path, d, "state.json"))
        ]
        if not valid_subdirs:
            raise FileNotFoundError(
                f"No valid Arrow dataset subdirectories found in {path}. "
                f"Subdirs present: {subdirs}. Each subdir must contain "
                f"dataset_info.json or state.json."
            )
        if len(valid_subdirs) < len(subdirs):
            skipped = sorted(set(subdirs) - set(valid_subdirs))
            logger.warning(f"Skipping non-dataset subdirs in {path}: {skipped}")
        logger.info(
            f"Auto-concatenating {len(valid_subdirs)} subdirectories from {path}: {valid_subdirs}"
        )
        parts = [load_from_disk(os.path.join(path, d)) for d in valid_subdirs]
        combined = concatenate_datasets(parts)
        if len(combined) > _SHUFFLE_ROW_THRESHOLD:
            logger.warning(
                f"Concatenated dataset has {len(combined)} rows "
                f"(>{_SHUFFLE_ROW_THRESHOLD}), skipping global shuffle to "
                f"avoid OOM. Consider using streaming."
            )
            return combined
        logger.info(f"Shuffling concatenated dataset ({len(combined)} rows)")
        return combined.shuffle(seed=seed)

    return load_from_disk(path)


def _data_root_path(relative_path: str) -> str:
    """Resolve a built dataset below the optional public ``DATA_ROOT``."""
    data_root = os.environ.get("DATA_ROOT", "").strip()
    if not data_root:
        return ""
    return os.path.join(os.path.expanduser(data_root), relative_path)


# Dataset registry; paper-facing build details live in scripts/dataset.md.
DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="train"),
        sample_processor=_process_c4_text,
    ),
    "synthetic_test": DatasetConfig(
        path="tests/assets/synthetic_text",
        loader=lambda path: load_dataset(path, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_validation": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="validation"),
        sample_processor=_process_c4_text,
    ),
    "nemotron_post_training_v2": DatasetConfig(
        path="nvidia/Nemotron-Post-Training-Dataset-v2",
        loader=lambda path: load_from_disk(path) if os.path.exists(os.path.join(path, "dataset_info.json")) else concatenate_datasets(list(load_dataset(path).values())),
        sample_processor=lambda sample: "\n".join(
            m.get("content", "") for m in sample.get("messages", []) if m.get("content")
        ),
    ),
    # SFT variant: same data but preserves the raw messages list so
    # DLLMSFTDataset can apply the ChatML template and mask prompts.
    # sample_processor is unused by DLLMSFTDataset (it reads sample["messages"]
    # directly), but is provided as a pass-through for compatibility.
    "nemotron_post_training_v2_sft": DatasetConfig(
        path="nvidia/Nemotron-Post-Training-Dataset-v2",
        loader=lambda path: load_from_disk(path) if os.path.exists(os.path.join(path, "dataset_info.json")) else concatenate_datasets(list(load_dataset(path).values())),
        sample_processor=lambda sample: sample,
    ),
    "llama_nemotron_sft": DatasetConfig(
        path="nvidia/Llama-Nemotron-Post-Training-Dataset",
        loader=_load_llama_nemotron_sft,
        sample_processor=_llama_nemotron_sample_processor,
    ),
    # ---- CPT: Swallow datasets ----
    # Parent-level (auto-concats all subdirs)
    "swallow_code_v2": DatasetConfig(
        path=_data_root_path("swallow-code-v2"),
        loader=_load_arrow_flat,
        sample_processor=lambda sample: sample["text"],
    ),
    "swallow_math_v2": DatasetConfig(
        path=_data_root_path("swallow-math-v2"),
        loader=_load_arrow_flat,
        sample_processor=lambda sample: sample["text"],
    ),
    # Individual subdirs (for when you only want one split)
    "swallow_math_v2_qa": DatasetConfig(
        path=_data_root_path("swallow-math-v2/swallow-math-v2-qa"),
        loader=_load_arrow_flat,
        sample_processor=lambda sample: sample["text"],
    ),
    "swallow_math_v2_textbook": DatasetConfig(
        path=_data_root_path("swallow-math-v2/swallow-math-v2-textbook"),
        loader=_load_arrow_flat,
        sample_processor=lambda sample: sample["text"],
    ),
    # ---- CPT: Nemotron-Pretraining-SFT-v1 ----
    # Parent-level (auto-concats Code + General + MATH)
    "nemotron_pretraining_sft": DatasetConfig(
        path=_data_root_path("Nemotron-Pretraining-SFT-v1"),
        loader=_load_arrow_flat,
        sample_processor=lambda sample: sample["text"],
    ),
    # Individual subdirs
    "nemotron_pretraining_sft_code": DatasetConfig(
        path=_data_root_path("Nemotron-Pretraining-SFT-v1/Nemotron-SFT-Code"),
        loader=_load_arrow_flat,
        sample_processor=lambda sample: sample["text"],
    ),
    "nemotron_pretraining_sft_general": DatasetConfig(
        path=_data_root_path("Nemotron-Pretraining-SFT-v1/Nemotron-SFT-General"),
        loader=_load_arrow_flat,
        sample_processor=lambda sample: sample["text"],
    ),
    "nemotron_pretraining_sft_math": DatasetConfig(
        path=_data_root_path("Nemotron-Pretraining-SFT-v1/Nemotron-SFT-MATH"),
        loader=_load_arrow_flat,
        sample_processor=lambda sample: sample["text"],
    ),
    # ---- SFT: Nemotron datasets (flat Arrow dirs, messages column) ----
    "nemotron_chat_v1_sft": DatasetConfig(
        path=_data_root_path("Nemotron-Instruction-Following-Chat-v1"),
        loader=_load_arrow_flat,
        sample_processor=_merge_reasoning_content,
    ),
    "nemotron_science_v1_sft": DatasetConfig(
        path=_data_root_path("Nemotron-Science-v1"),
        loader=_load_arrow_flat,
        sample_processor=_merge_reasoning_content,
    ),
    "nemotron_math_proofs_v1_sft": DatasetConfig(
        path=_data_root_path("Nemotron-Math-Proofs-v1"),
        loader=_load_arrow_flat,
        sample_processor=_merge_reasoning_content,
    ),
    "nemotron_chat_v2_sft": DatasetConfig(
        path=_data_root_path("Nemotron-SFT-Instruction-Following-Chat-v2"),
        loader=_load_arrow_flat,
        sample_processor=_merge_reasoning_content,
    ),
    "nemotron_safety_v1_sft": DatasetConfig(
        path=_data_root_path("Nemotron-SFT-Safety-v1"),
        loader=_load_arrow_flat,
        sample_processor=_merge_reasoning_content,
    ),
    "nemotron_math_v3_sft": DatasetConfig(
        path=_data_root_path("Nemotron-SFT-Math-v3"),
        loader=_load_arrow_flat,
        sample_processor=_format_tool_messages,
    ),
    "nemotron_swe_v2_sft": DatasetConfig(
        path=_data_root_path("Nemotron-SFT-SWE-v2"),
        loader=_load_arrow_flat,
        sample_processor=_format_tool_messages,
    ),
    "nemotron_multilingual_v1_sft": DatasetConfig(
        path=_data_root_path("Nemotron-SFT-Multilingual-v1"),
        loader=_load_arrow_flat,
        sample_processor=_merge_reasoning_content,
    ),
    "nemotron_finance_v1_sft": DatasetConfig(
        path=_data_root_path("Nemotron-SpecializedDomains-Finance-v1"),
        loader=_load_arrow_flat,
        sample_processor=_merge_reasoning_content,
    ),
    "nemotron_compprog_v1_sft": DatasetConfig(
        path=_data_root_path("Nemotron-Competitive-Programming-v1"),
        loader=_load_arrow_flat,
        sample_processor=_merge_reasoning_content,
    ),
    "nemotron_compprog_v2_sft": DatasetConfig(
        path=_data_root_path("Nemotron-SFT-Competitive-Programming-v2"),
        loader=_load_arrow_flat,
        sample_processor=_merge_reasoning_content,
    ),
    # ---- Curated Long-SFT datasets (pre-merged, unified {role, content} format) ----
    "nemotron_long_sft_3k": DatasetConfig(
        path=_data_root_path("Long-SFT-3K"),
        loader=_load_arrow_flat,
        sample_processor=_long_sft_sample_processor,
    ),
    "nemotron_long_sft_7k": DatasetConfig(
        path=_data_root_path("Long-SFT-7K"),
        loader=_load_arrow_flat,
        sample_processor=_long_sft_sample_processor,
    ),
    # ---- Per-domain datasets (same format as Long-SFT, no length filter) ----
    "nemotron_math_domain": DatasetConfig(
        path=_data_root_path("Nemotron-Math-Domain"),
        loader=_load_arrow_flat,
        sample_processor=_long_sft_sample_processor,
    ),
    "nemotron_code_domain": DatasetConfig(
        path=_data_root_path("Nemotron-Code-Domain"),
        loader=_load_arrow_flat,
        sample_processor=_long_sft_sample_processor,
    ),
    "nemotron_code_domain_cascade": DatasetConfig(
        path=_data_root_path("Nemotron-Code-Domain-Cascade"),
        loader=_load_arrow_flat,
        sample_processor=_long_sft_sample_processor,
    ),
    "nemotron_math_domain_cascade": DatasetConfig(
        path=_data_root_path("Nemotron-Math-Domain-Cascade"),
        loader=_load_arrow_flat,
        sample_processor=_long_sft_sample_processor,
    ),
    "nemotron_science_domain": DatasetConfig(
        path=_data_root_path("Nemotron-Science-Domain"),
        loader=_load_arrow_flat,
        sample_processor=_long_sft_sample_processor,
    ),
    "nemotron_science_domain_cascade": DatasetConfig(
        path=_data_root_path("Nemotron-Science-Domain-Cascade"),
        loader=_load_arrow_flat,
        sample_processor=_long_sft_sample_processor,
    ),
    "nemotron_if_domain_cascade": DatasetConfig(
        path=_data_root_path("Nemotron-IF-Domain-Cascade"),
        loader=_load_arrow_flat,
        sample_processor=_long_sft_sample_processor,
    ),
    # Post-Training-v2 English-only (no multilingual splits)
    "nemotron_post_training_v2_en_sft": DatasetConfig(
        path=_data_root_path("Nemotron-Post-Training-Dataset-v2-EN-arrow"),
        loader=_load_arrow_flat,
        sample_processor=lambda sample: sample,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path.

    If dataset_path is an S3 path, it will be automatically downloaded to
    local SSD cache via cache_s3_path_if_needed (multi-node safe, parallel).
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    if not path:
        raise ValueError(
            f"Dataset {dataset_name} requires an explicit training.dataset_path "
            "or the DATA_ROOT environment variable."
        )
    # Unified S3 -> local cache for all datasets (parallel download, multi-node safe)
    path = cache_s3_path_if_needed(path) or path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class HuggingFaceTextDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        shift_labels: bool = True,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.shift_labels = shift_labels
        self._text_processor = text_processor

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        # For shifted labels (next token prediction), need seq_len + 1 tokens
        # For non-shifted labels (e.g., diffusion LLM), only need seq_len tokens
        max_buffer_token_len = (1 + self.seq_len) if self.shift_labels else self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)
                
                # Skip empty samples to avoid training on meaningless data
                if not sample_text or not sample_text.strip():
                    self._sample_idx += 1
                    continue
                
                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=True, add_eos=True
                )
                self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    
                    if self.shift_labels:
                        # Standard language modeling: input=x[:-1], label=x[1:]
                        input = x[:-1]
                        label = x[1:]
                    else:
                        # No shift for diffusion LLM or other models
                        input = x
                        label = x
                    
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.debug(f"Dataset {self.dataset_name} is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict: dict[str, Any] = {"token_buffer": self._token_buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


def build_text_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets.

    Args:
        dp_world_size: Data parallelism world size.
        dp_rank: Data parallelism rank.
        tokenizer: Tokenizer to use for encoding text.
        job_config: Job configuration containing dataset and DataLoader settings.
        infinite: Whether to loop the dataset infinitely.
    """
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len
    shift_labels = job_config.training.shift_labels

    hf_ds = HuggingFaceTextDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        shift_labels=shift_labels,
    )

    dataloader_kwargs = {
        **asdict(job_config.training.dataloader),
        "batch_size": batch_size,
    }

    return ParallelAwareDataloader(
        hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )


def build_text_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = False,
) -> ParallelAwareDataloader:
    """Build a validation data loader for HuggingFace datasets.

    Args:
        dp_world_size: Data parallelism world size.
        dp_rank: Data parallelism rank.
        tokenizer: Tokenizer to use for encoding text.
        job_config: Job configuration containing dataset and DataLoader settings.
        infinite: Whether to loop the dataset infinitely.
    """
    dataset_name = job_config.validation.dataset
    dataset_path = job_config.validation.dataset_path
    batch_size = job_config.validation.local_batch_size
    seq_len = job_config.validation.seq_len
    shift_labels = job_config.validation.shift_labels

    hf_ds = HuggingFaceTextDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        shift_labels=shift_labels,
    )

    dataloader_kwargs = {
        **asdict(job_config.validation.dataloader),
        "batch_size": batch_size,
    }

    return ParallelAwareDataloader(
        hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )


# ===========================================================================
# DLLM (block diffusion) datasets and dataloaders
# ===========================================================================


class DLLMTextDataset(HuggingFaceTextDataset):
    """Text dataset with block-size-aligned document padding for DLLM training.

    Pads each document to a multiple of ``block_size`` before concatenation so
    that block boundaries in the final training chunks align with document
    boundaries.  Maintains dual buffers (tokens + labels) so that padding
    positions get ``labels = -100``, which the loss function can ignore.

    Yields ``({"input": input, "labels": labels}, labels)`` instead of
    deriving labels from shifted inputs.
    """

    # Qwen3 <|endoftext|> token ID — used for padding positions.
    QWEN3_PAD_TOKEN_ID: int = 151643

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        block_size: int = 16,
        mask_token_id: int = -1,
        pad_token_id: int = -1,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
        )
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        # pad_token_id: explicit value from config, or fall back to mask_token_id (old behavior)
        self.pad_token_id = pad_token_id if pad_token_id >= 0 else mask_token_id
        self._label_buffer: list[int] = []

    def __iter__(self):
        max_buffer_token_len = self.seq_len

        while True:
            for sample in self._get_data_iter():
                sample_text = self._text_processor(sample)

                if not sample_text or not sample_text.strip():
                    self._sample_idx += 1
                    continue

                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=True, add_eos=True
                )

                remainder = len(sample_tokens) % self.block_size
                if remainder != 0:
                    pad_len = self.block_size - remainder
                    sample_labels = list(sample_tokens) + [-100] * pad_len
                    sample_tokens = sample_tokens + [self.pad_token_id] * pad_len
                else:
                    sample_labels = list(sample_tokens)

                self._token_buffer.extend(sample_tokens)
                self._label_buffer.extend(sample_labels)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    x_labels = self._label_buffer[:max_buffer_token_len]
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    self._label_buffer = self._label_buffer[max_buffer_token_len:]

                    input_tokens = x
                    labels = torch.LongTensor(x_labels)

                    yield {"input": input_tokens, "labels": labels}, labels

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                self._sample_idx = 0
                logger.debug(f"Dataset {self.dataset_name} is being re-looped")
                if isinstance(self._data, IterableDataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._label_buffer = state_dict.get("label_buffer", [])

    def state_dict(self):
        _state_dict = super().state_dict()
        _state_dict["label_buffer"] = self._label_buffer
        return _state_dict


class DLLMSFTDataset(DLLMTextDataset):
    """DLLM SFT dataset with ChatML template formatting and prompt masking.

    Applies the ChatML template to message-style data and sets
    labels according to ``sft_label_mode``:

    * ``response``: current behavior, only assistant response tokens train
    * ``full``: all non-padding ChatML tokens train
    * ``full_content``: content tokens train for all roles, but not headers/end

    Unlike ``DLLMTextDataset``, conversations are **not** packed: each
    conversation is independently truncated to ``seq_len`` and padded to
    ``seq_len`` with ``pad_token_id`` (``labels = -100``).
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        block_size: int = 16,
        mask_token_id: int = -1,
        pad_token_id: int = -1,
        messages_field: str = "messages",
        sft_label_mode: str = "response",
        enable_packing: bool = False,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
            block_size=block_size,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
        )
        self.messages_field = messages_field
        self.sft_label_mode = _validate_sft_label_mode(sft_label_mode)
        self.enable_packing = enable_packing
        if enable_packing and seq_len % block_size != 0:
            logger.warning(
                f"DLLMSFTDataset: seq_len ({seq_len}) is not divisible by "
                f"block_size ({block_size}). Packed samples remain doc-aligned "
                f"to block boundaries where possible; model-side row padding "
                f"handles the partial final row segment."
            )
        # Packing state
        self._token_buffer: list[int] = []
        self._label_buffer: list[int] = []
        self._doc_id_buffer: list[int] = []
        self._next_doc_id: int = 0

    def _tokenize_chat_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[int], list[int]]:
        """Apply ChatML template to *messages* and produce SFT-masked labels.

        Returns ``(tokens, labels)`` where ``labels[i] = -100`` for
        positions the model should NOT be trained on.
        """
        if not isinstance(messages, list):
            return [], []

        last_query_idx = _find_last_query_index(messages)
        num_messages = len(messages)

        all_tokens: list[int] = []
        all_labels: list[int] = []

        for idx, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role", "")
            content = message.get("content", "")

            if (
                role == "assistant"
                and idx > last_query_idx >= 0
                and idx == num_messages - 1
                and "<think>" not in content
            ):
                content = "<think>\n\n</think>\n\n" + content

            if not role:
                continue

            header_tokens = self._tokenizer.encode(
                f"<|im_start|>{role}\n", add_bos=False, add_eos=False
            )
            content_tokens = (
                self._tokenizer.encode(
                    content, add_bos=False, add_eos=False
                )
                if content
                else []
            )
            end_tokens = self._tokenizer.encode(
                "<|im_end|>\n", add_bos=False, add_eos=False
            )

            if self.sft_label_mode == "response":
                train_header = False
                train_content = role == "assistant"
                train_end = role == "assistant"
            elif self.sft_label_mode == "full":
                train_header = True
                train_content = True
                train_end = True
            else:
                train_header = False
                train_content = True
                train_end = False

            all_tokens.extend(header_tokens)
            all_labels.extend(
                list(header_tokens) if train_header else [-100] * len(header_tokens)
            )
            all_tokens.extend(content_tokens)
            all_labels.extend(
                list(content_tokens)
                if train_content
                else [-100] * len(content_tokens)
            )
            all_tokens.extend(end_tokens)
            all_labels.extend(
                list(end_tokens) if train_end else [-100] * len(end_tokens)
            )

        return all_tokens, all_labels

    def _tokenize_sample(self, sample):
        """Tokenize a single sample, returning (tokens, labels) or None."""
        sample = self._text_processor(sample)
        messages = sample.get(self.messages_field, [])
        if not messages:
            return None
        tokens, labels = self._tokenize_chat_messages(messages)
        if not tokens or all(lab == -100 for lab in labels):
            return None
        return tokens, labels

    def _align_to_block_size(self, tokens, labels):
        """Pad tokens/labels so length is a multiple of block_size."""
        remainder = len(tokens) % self.block_size
        if remainder != 0:
            pad_len = self.block_size - remainder
            tokens = tokens + [self.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
        return tokens, labels

    def _flush_pack_buffer(self):
        """Pad buffer to seq_len and yield one packed training example."""
        if not self._token_buffer:
            return None

        pad_len = self.seq_len - len(self._token_buffer)
        if pad_len > 0:
            last_doc_id = self._doc_id_buffer[-1] if self._doc_id_buffer else 0
            self._token_buffer.extend([self.pad_token_id] * pad_len)
            self._label_buffer.extend([-100] * pad_len)
            self._doc_id_buffer.extend([last_doc_id] * pad_len)

        input_tokens = torch.LongTensor(self._token_buffer[: self.seq_len])
        labels = torch.LongTensor(self._label_buffer[: self.seq_len])
        doc_ids = torch.LongTensor(self._doc_id_buffer[: self.seq_len])

        self._token_buffer = self._token_buffer[self.seq_len :]
        self._label_buffer = self._label_buffer[self.seq_len :]
        self._doc_id_buffer = self._doc_id_buffer[self.seq_len :]

        if self._doc_id_buffer:
            self._next_doc_id = max(self._doc_id_buffer) + 1
        else:
            self._next_doc_id = 0

        if all(lab == -100 for lab in labels.tolist()):
            return None
        return {"input": input_tokens, "labels": labels, "doc_ids": doc_ids}, labels

    def _iter_packed(self):
        """Greedy bin-packing: accumulate block-aligned samples into seq_len rows."""
        while True:
            for sample in self._get_data_iter():
                result = self._tokenize_sample(sample)
                self._sample_idx += 1
                if result is None:
                    continue

                sample_tokens, sample_labels = result
                # Truncate to largest block-aligned length <= seq_len so that
                # _align_to_block_size never produces a sample exceeding seq_len.
                max_aligned = (self.seq_len // self.block_size) * self.block_size
                if len(sample_tokens) > max_aligned:
                    sample_tokens = sample_tokens[:max_aligned]
                    sample_labels = sample_labels[:max_aligned]
                # Block-align (pads up to next multiple of block_size; after
                # the truncation above the result is guaranteed <= seq_len)
                sample_tokens, sample_labels = self._align_to_block_size(
                    sample_tokens, sample_labels
                )

                # If sample doesn't fit, flush current buffer first
                if len(self._token_buffer) + len(sample_tokens) > self.seq_len:
                    out = self._flush_pack_buffer()
                    if out is not None:
                        yield out

                # Add sample to buffer with unique doc_id
                doc_id = self._next_doc_id
                self._next_doc_id += 1
                self._token_buffer.extend(sample_tokens)
                self._label_buffer.extend(sample_labels)
                self._doc_id_buffer.extend([doc_id] * len(sample_tokens))

                # Flush if exactly full
                if len(self._token_buffer) >= self.seq_len:
                    out = self._flush_pack_buffer()
                    if out is not None:
                        yield out

            # Flush remaining buffer at end of epoch
            if self._token_buffer:
                out = self._flush_pack_buffer()
                if out is not None:
                    yield out

            if not self.infinite:
                logger.warning(
                    f"Dataset {self.dataset_name} has run out of data"
                )
                break
            else:
                self._sample_idx = 0
                logger.debug(
                    f"Dataset {self.dataset_name} is being re-looped"
                )
                if isinstance(self._data, IterableDataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def _iter_single(self):
        """Original non-packed iteration: one sample per row."""
        while True:
            for sample in self._get_data_iter():
                result = self._tokenize_sample(sample)
                self._sample_idx += 1
                if result is None:
                    continue

                sample_tokens, sample_labels = result
                if len(sample_tokens) > self.seq_len:
                    sample_tokens = sample_tokens[: self.seq_len]
                    sample_labels = sample_labels[: self.seq_len]

                pad_len = self.seq_len - len(sample_tokens)
                if pad_len > 0:
                    sample_tokens = sample_tokens + [self.pad_token_id] * pad_len
                    sample_labels = sample_labels + [-100] * pad_len

                input_tokens = torch.LongTensor(sample_tokens)
                labels = torch.LongTensor(sample_labels)
                yield {"input": input_tokens, "labels": labels}, labels

            if not self.infinite:
                logger.warning(
                    f"Dataset {self.dataset_name} has run out of data"
                )
                break
            else:
                self._sample_idx = 0
                logger.debug(
                    f"Dataset {self.dataset_name} is being re-looped"
                )
                if isinstance(self._data, IterableDataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._doc_id_buffer = state_dict.get("doc_id_buffer", [])
        self._next_doc_id = state_dict.get("next_doc_id", 0)
        if self._token_buffer and not self._doc_id_buffer:
            self._doc_id_buffer = [0] * len(self._token_buffer)

    def state_dict(self):
        _state_dict = super().state_dict()
        _state_dict["doc_id_buffer"] = self._doc_id_buffer
        _state_dict["next_doc_id"] = self._next_doc_id
        return _state_dict

    def __iter__(self):
        if self.enable_packing:
            yield from self._iter_packed()
        else:
            yield from self._iter_single()


class ARSFTDataset(HuggingFaceTextDataset):
    """Autoregressive SFT dataset with ChatML template and prompt masking.

    Applies the same ChatML formatting as ``DLLMSFTDataset`` but uses
    standard autoregressive training conventions:

    * Shifted labels: ``label[t] = token[t+1]`` (next-token prediction).
    * Padding uses ``pad_token_id`` instead of a noise/mask token.
    * Non-assistant positions in labels are set to ``-100``.

    Supports packing (``enable_packing=True``): multiple conversations
    are greedily bin-packed into each ``seq_len`` row.  Shifted labels
    at document boundaries are set to ``-100`` so the model never
    predicts the first token of the next document.  When packing is
    enabled, ``doc_ids`` are generated so the model can use
    document-aware attention masking (e.g. FlexAttention with
    ``attn_mask_type="doc_causal"``) to prevent cross-document attention.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        pad_token_id: int | None = None,
        messages_field: str = "messages",
        enable_packing: bool = False,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
            shift_labels=False,
        )
        self.pad_token_id = (
            pad_token_id if pad_token_id is not None else tokenizer.get_vocab_size() - 1
        )
        self.messages_field = messages_field
        self.enable_packing = enable_packing
        self._token_buffer: list[int] = []
        self._label_buffer: list[int] = []
        self._doc_id_buffer: list[int] = []
        self._next_doc_id: int = 0

    @staticmethod
    def _tokenize_chatml(
        messages: list[dict[str, str]],
        tokenizer: BaseTokenizer,
    ) -> tuple[list[int], list[int]]:
        """Apply ChatML template and return ``(tokens, labels)``.

        ``labels[i] = -100`` for all non-assistant positions.  This is a
        static method so it can be shared across dataset classes.
        """
        if not isinstance(messages, list):
            return [], []

        last_query_idx = _find_last_query_index(messages)
        num_messages = len(messages)

        all_tokens: list[int] = []
        all_labels: list[int] = []

        for idx, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role", "")
            content = message.get("content", "")

            if (
                role == "assistant"
                and idx > last_query_idx >= 0
                and idx == num_messages - 1
                and "<think>" not in content
            ):
                content = "<think>\n\n</think>\n\n" + content

            if not role:
                continue

            header = tokenizer.encode(
                f"<|im_start|>{role}\n", add_bos=False, add_eos=False
            )
            body = (
                tokenizer.encode(content, add_bos=False, add_eos=False)
                if content
                else []
            )
            end = tokenizer.encode(
                "<|im_end|>\n", add_bos=False, add_eos=False
            )

            if role == "assistant":
                all_tokens.extend(header)
                all_labels.extend([-100] * len(header))
                all_tokens.extend(body)
                all_labels.extend(list(body))
                all_tokens.extend(end)
                all_labels.extend(list(end))
            else:
                turn = header + body + end
                all_tokens.extend(turn)
                all_labels.extend([-100] * len(turn))

        return all_tokens, all_labels

    def _tokenize_sample(self, sample):
        """Tokenize a single sample, returning (tokens, labels) or None."""
        sample = self._text_processor(sample)
        messages = sample.get(self.messages_field, [])
        if not messages:
            return None
        tokens, labels = self._tokenize_chatml(messages, self._tokenizer)
        if not tokens or all(lab == -100 for lab in labels):
            return None
        return tokens, labels

    def _flush_pack_buffer(self):
        """Shift labels, fix doc boundaries, pad to seq_len, yield."""
        if not self._token_buffer:
            return None

        # Need seq_len+1 tokens for shift
        max_len = self.seq_len + 1
        pad_len = max_len - len(self._token_buffer)
        if pad_len > 0:
            last_doc_id = self._doc_id_buffer[-1] if self._doc_id_buffer else 0
            self._token_buffer.extend([self.pad_token_id] * pad_len)
            self._label_buffer.extend([-100] * pad_len)
            self._doc_id_buffer.extend([last_doc_id] * pad_len)

        tokens = self._token_buffer[:max_len]
        labels = self._label_buffer[:max_len]
        doc_ids = self._doc_id_buffer[:self.seq_len]
        self._token_buffer = self._token_buffer[max_len:]
        self._label_buffer = self._label_buffer[max_len:]
        self._doc_id_buffer = self._doc_id_buffer[max_len:]
        self._next_doc_id = 0

        input_tokens = torch.LongTensor(tokens[:-1])
        shifted_labels = torch.LongTensor(labels[1:])
        doc_ids = torch.LongTensor(doc_ids)

        if all(lab == -100 for lab in shifted_labels.tolist()):
            return None
        return {"input": input_tokens, "labels": shifted_labels, "doc_ids": doc_ids}, shifted_labels

    def _iter_packed(self):
        """Greedy bin-packing with AR-style shifted labels."""
        while True:
            for sample in self._get_data_iter():
                result = self._tokenize_sample(sample)
                self._sample_idx += 1
                if result is None:
                    continue

                sample_tokens, sample_labels = result
                if len(sample_tokens) > self.seq_len:
                    sample_tokens = sample_tokens[:self.seq_len]
                    sample_labels = sample_labels[:self.seq_len]

                # At doc boundary: after shift (shifted_labels = labels[1:]),
                # the last input token of doc A would predict the first token
                # of doc B. To prevent this, mark the first label of doc B as
                # -100 so shifted_labels masks out that cross-doc prediction.
                if self._token_buffer and sample_labels:
                    sample_labels[0] = -100

                # Flush if won't fit
                if len(self._token_buffer) + len(sample_tokens) > self.seq_len + 1:
                    out = self._flush_pack_buffer()
                    if out is not None:
                        yield out

                doc_id = self._next_doc_id
                self._next_doc_id += 1
                self._token_buffer.extend(sample_tokens)
                self._label_buffer.extend(sample_labels)
                self._doc_id_buffer.extend([doc_id] * len(sample_tokens))

                if len(self._token_buffer) >= self.seq_len + 1:
                    out = self._flush_pack_buffer()
                    if out is not None:
                        yield out

            if self._token_buffer:
                out = self._flush_pack_buffer()
                if out is not None:
                    yield out

            if not self.infinite:
                logger.warning(
                    f"Dataset {self.dataset_name} has run out of data"
                )
                break
            else:
                self._sample_idx = 0
                logger.debug(
                    f"Dataset {self.dataset_name} is being re-looped"
                )
                if isinstance(self._data, IterableDataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def _iter_single(self):
        """Original non-packed iteration: one sample per row."""
        while True:
            for sample in self._get_data_iter():
                result = self._tokenize_sample(sample)
                self._sample_idx += 1
                if result is None:
                    continue

                tokens, labels = result
                max_len = self.seq_len + 1
                if len(tokens) > max_len:
                    tokens = tokens[:max_len]
                    labels = labels[:max_len]

                if all(l == -100 for l in labels):
                    continue

                pad_len = max_len - len(tokens)
                if pad_len > 0:
                    tokens = tokens + [self.pad_token_id] * pad_len
                    labels = labels + [-100] * pad_len

                input_tokens = torch.LongTensor(tokens[:-1])
                shifted_labels = torch.LongTensor(labels[1:])
                yield {"input": input_tokens, "labels": shifted_labels}, shifted_labels

            if not self.infinite:
                logger.warning(
                    f"Dataset {self.dataset_name} has run out of data"
                )
                break
            else:
                self._sample_idx = 0
                logger.debug(
                    f"Dataset {self.dataset_name} is being re-looped"
                )
                if isinstance(self._data, IterableDataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._label_buffer = state_dict.get("label_buffer", [])
        self._doc_id_buffer = state_dict.get("doc_id_buffer", [])
        self._next_doc_id = state_dict.get("next_doc_id", 0)
        if self._token_buffer and not self._doc_id_buffer:
            self._doc_id_buffer = [0] * len(self._token_buffer)
        if self._token_buffer and not self._label_buffer:
            self._label_buffer = [-100] * len(self._token_buffer)

    def state_dict(self):
        _state_dict = super().state_dict()
        _state_dict["label_buffer"] = self._label_buffer
        _state_dict["doc_id_buffer"] = self._doc_id_buffer
        _state_dict["next_doc_id"] = self._next_doc_id
        return _state_dict

    def __iter__(self):
        if self.enable_packing:
            yield from self._iter_packed()
        else:
            yield from self._iter_single()


class MixedDataset(IterableDataset, Stateful):
    """Wraps multiple iterable datasets and samples from them according to weights.

    Each child dataset is a fully independent ``IterableDataset`` with its
    own internal state.  On each ``__next__`` call the mixer draws from a
    child according to the weight distribution and yields that child's
    next sample.

    Supports stateful checkpointing: ``state_dict`` / ``load_state_dict``
    save and restore every child's state plus the sampling RNG.
    """

    def __init__(
        self,
        children: list[IterableDataset],
        weights: list[float],
        seed: int = 42,
    ) -> None:
        if len(children) != len(weights):
            raise ValueError(
                f"Number of children ({len(children)}) must match "
                f"number of weights ({len(weights)})"
            )
        if not children:
            raise ValueError("At least one child dataset is required")
        if any(w < 0 for w in weights):
            raise ValueError("MixedDataset weights must be non-negative")
        if sum(weights) == 0:
            raise ValueError("MixedDataset weights must sum to > 0")

        self._children = children
        weight_t = torch.tensor(weights, dtype=torch.float64)
        self._weights = weight_t / weight_t.sum()
        self._seed = seed
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)
        self._resume_rng_state: torch.Tensor | None = None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        effective_seed = self._seed + (worker_info.id if worker_info else 0)
        self._rng = torch.Generator()
        if self._resume_rng_state is not None:
            self._rng.set_state(self._resume_rng_state)
            self._resume_rng_state = None
        else:
            self._rng.manual_seed(effective_seed)

        child_iters = [iter(c) for c in self._children]
        active_mask = torch.ones(len(self._children), dtype=torch.bool)
        weights = self._weights.clone()

        while active_mask.any():
            idx = int(torch.multinomial(weights, 1, generator=self._rng).item())
            try:
                yield next(child_iters[idx])
            except StopIteration:
                logger.warning(
                    f"Mixed dataset child {idx} exhausted, removing from mix"
                )
                active_mask[idx] = False
                if not active_mask.any():
                    return
                weights = self._weights * active_mask.float()
                weights = weights / weights.sum()

    def state_dict(self) -> dict[str, Any]:
        sd: dict[str, Any] = {"rng_state": self._rng.get_state()}
        for i, child in enumerate(self._children):
            if isinstance(child, Stateful):
                sd[f"child_{i}"] = child.state_dict()
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # __iter__ creates a fresh Generator. Defer restoration until then so
        # the loaded sampling position is not overwritten by the base seed.
        self._resume_rng_state = state_dict["rng_state"].clone()
        for i, child in enumerate(self._children):
            key = f"child_{i}"
            if key in state_dict and isinstance(child, Stateful):
                child.load_state_dict(state_dict[key])


def _get_dllm_common_kwargs(
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
) -> dict:
    """Extract DLLM keyword arguments that are shared across all datasets.

    Default (backward compatible): mask_token_id = pad_token_id = vocab_size - 1.
    New experiments should set both explicitly in [dllm] config:
        mask_token_id = 151669   # <|MASK|>
        pad_token_id  = 151643   # <|endoftext|>
    """
    dllm_config = getattr(job_config, "dllm", None)
    block_size = getattr(dllm_config, "block_size", 16) if dllm_config else 16

    # -- mask_token_id: default = vocab_size - 1 (old behavior) --
    mask_token_id = (
        getattr(dllm_config, "mask_token_id", -1) if dllm_config else -1
    )
    if mask_token_id < 0:
        mask_token_id = tokenizer.get_vocab_size() - 1

    # -- pad_token_id: default = mask_token_id (old behavior: pad = mask) --
    pad_token_id = (
        getattr(dllm_config, "pad_token_id", -1) if dllm_config else -1
    )
    if pad_token_id < 0:
        pad_token_id = mask_token_id

    return {
        "tokenizer": tokenizer,
        "seq_len": job_config.training.seq_len,
        "block_size": block_size,
        "mask_token_id": mask_token_id,
        "pad_token_id": pad_token_id,
    }


def _validate_sft_label_mode(sft_label_mode: str) -> str:
    valid_modes = {"response", "full", "full_content"}
    if sft_label_mode not in valid_modes:
        raise ValueError(
            f"dllm.sft_label_mode must be one of {sorted(valid_modes)}, "
            f"got {sft_label_mode!r}"
        )
    return sft_label_mode


def _get_sft_label_mode(job_config: JobConfig) -> str:
    dllm_config = getattr(job_config, "dllm", None)
    sft_label_mode = (
        getattr(dllm_config, "sft_label_mode", "response")
        if dllm_config
        else "response"
    )
    return _validate_sft_label_mode(sft_label_mode)


def _validate_sft_label_mode_usage(
    job_config: JobConfig,
    *,
    allow_dllm_sft: bool,
) -> None:
    sft_label_mode = _get_sft_label_mode(job_config)
    if not allow_dllm_sft and sft_label_mode != "response":
        raise ValueError(
            "dllm.sft_label_mode only applies to DLLM SFT dataloaders; "
            f"got {sft_label_mode!r} in a non-SFT dataloader path"
        )


def _get_dllm_dataset_kwargs(
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
) -> dict:
    """Extract common DLLM dataset keyword arguments from job config."""
    kwargs = _get_dllm_common_kwargs(tokenizer, job_config)
    kwargs["dataset_name"] = job_config.training.dataset
    kwargs["dataset_path"] = job_config.training.dataset_path
    return kwargs


def _get_dllm_sft_dataset_kwargs(
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
) -> dict:
    kwargs = _get_dllm_dataset_kwargs(tokenizer, job_config)
    kwargs["sft_label_mode"] = _get_sft_label_mode(job_config)
    if hasattr(job_config, "dllm") and hasattr(job_config.dllm, "enable_packing"):
        kwargs["enable_packing"] = job_config.dllm.enable_packing
    return kwargs


def _validate_dataset_mix(job_config: JobConfig) -> None:
    """Validate that every dataset_mix entry has the required fields."""
    for i, entry in enumerate(job_config.training.dataset_mix):
        if not entry.name:
            raise ValueError(f"dataset_mix[{i}] is missing 'name'")
        if not entry.path:
            raise ValueError(f"dataset_mix[{i}] is missing 'path'")


def build_dllm_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build dataloader for DLLM training with block-size document padding.

    Supports multi-dataset mixing when ``training.dataset_mix`` is set.
    Otherwise falls back to the single-dataset path using
    ``training.dataset``.
    """
    _validate_sft_label_mode_usage(job_config, allow_dllm_sft=False)
    mix_entries = job_config.training.dataset_mix
    if mix_entries:
        _validate_dataset_mix(job_config)
        common = _get_dllm_common_kwargs(tokenizer, job_config)
        children = [
            DLLMTextDataset(
                dataset_name=e.name,
                dataset_path=e.path,
                **common,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                infinite=infinite,
            )
            for e in mix_entries
        ]
        weights = [e.weight for e in mix_entries]
        logger.info(
            f"Building mixed DLLM dataloader with {len(children)} datasets: "
            + ", ".join(
                f"{e.name} (weight={e.weight})" for e in mix_entries
            )
        )
        hf_ds = MixedDataset(children, weights)
    else:
        ds_kwargs = _get_dllm_dataset_kwargs(tokenizer, job_config)
        hf_ds = DLLMTextDataset(
            **ds_kwargs,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
        )

    dataloader_kwargs = {
        **asdict(job_config.training.dataloader),
        "batch_size": job_config.training.local_batch_size,
    }

    return ParallelAwareDataloader(
        hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )


def build_dllm_sft_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build dataloader for DLLM SFT training with ChatML template and prompt masking.

    Supports multi-dataset mixing when ``training.dataset_mix`` is set.
    Otherwise falls back to the single-dataset path using
    ``training.dataset``.
    """
    sft_label_mode = _get_sft_label_mode(job_config)
    mix_entries = job_config.training.dataset_mix
    if mix_entries:
        _validate_dataset_mix(job_config)
        common = _get_dllm_common_kwargs(tokenizer, job_config)
        enable_packing = getattr(job_config.dllm, "enable_packing", False) if hasattr(job_config, "dllm") else False
        children = [
            DLLMSFTDataset(
                dataset_name=e.name,
                dataset_path=e.path,
                **common,
                sft_label_mode=sft_label_mode,
                enable_packing=enable_packing,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                infinite=infinite,
            )
            for e in mix_entries
        ]
        weights = [e.weight for e in mix_entries]
        logger.info(
            f"Building mixed DLLM SFT dataloader with {len(children)} datasets: "
            + ", ".join(
                f"{e.name} (weight={e.weight})" for e in mix_entries
            )
        )
        hf_ds = MixedDataset(children, weights)
    else:
        ds_kwargs = _get_dllm_sft_dataset_kwargs(tokenizer, job_config)
        hf_ds = DLLMSFTDataset(
            **ds_kwargs,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
        )
    logger.info(f"DLLM SFT label mode: {sft_label_mode}")

    dataloader_kwargs = {
        **asdict(job_config.training.dataloader),
        "batch_size": job_config.training.local_batch_size,
    }

    return ParallelAwareDataloader(
        hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )


def build_ar_sft_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build dataloader for standard autoregressive SFT with ChatML template.

    Supports multi-dataset mixing when ``training.dataset_mix`` is set.
    Otherwise falls back to the single-dataset path using
    ``training.dataset``.
    """
    _validate_sft_label_mode_usage(job_config, allow_dllm_sft=False)
    pad_token_id = getattr(tokenizer, "pad_id", None)
    if pad_token_id is None:
        pad_token_id = tokenizer.get_vocab_size() - 1
    enable_packing = (
        getattr(job_config.dllm, "enable_packing", False)
        if hasattr(job_config, "dllm")
        else False
    )

    mix_entries = job_config.training.dataset_mix
    if mix_entries:
        _validate_dataset_mix(job_config)
        children = [
            ARSFTDataset(
                dataset_name=e.name,
                dataset_path=e.path,
                tokenizer=tokenizer,
                seq_len=job_config.training.seq_len,
                pad_token_id=pad_token_id,
                enable_packing=enable_packing,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                infinite=infinite,
            )
            for e in mix_entries
        ]
        weights = [e.weight for e in mix_entries]
        logger.info(
            f"Building mixed AR SFT dataloader with {len(children)} datasets: "
            + ", ".join(
                f"{e.name} (weight={e.weight})" for e in mix_entries
            )
        )
        hf_ds = MixedDataset(children, weights)
    else:
        hf_ds = ARSFTDataset(
            dataset_name=job_config.training.dataset,
            dataset_path=job_config.training.dataset_path,
            tokenizer=tokenizer,
            seq_len=job_config.training.seq_len,
            pad_token_id=pad_token_id,
            enable_packing=enable_packing,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
        )

    dataloader_kwargs = {
        **asdict(job_config.training.dataloader),
        "batch_size": job_config.training.local_batch_size,
    }

    return ParallelAwareDataloader(
        hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )
