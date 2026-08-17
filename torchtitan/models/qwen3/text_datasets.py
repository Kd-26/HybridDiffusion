# Backward-compatibility re-exports.
# The canonical location is now torchtitan.hf_datasets.text_datasets.
from torchtitan.hf_datasets.text_datasets import (  # noqa: F401
    ARSFTDataset,
    DLLMSFTDataset,
    DLLMTextDataset,
    MixedDataset,
    build_ar_sft_dataloader,
    build_dllm_dataloader,
    build_dllm_sft_dataloader,
)
