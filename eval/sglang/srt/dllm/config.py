from enum import Enum
from typing import Any

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs


DLLM_ATTN_MASK_CAUSAL_PREFILL = 0
DLLM_ATTN_MASK_BIDIR_BLOCK = 1


class SelfSpecVariant(str, Enum):
    CAUSAL_SHIFT = "causal_shift"     # shift everywhere, causal, blk=2N-1
    BD_CAUSAL = "bd_causal"           # V2: shift clean only, causal, blk=2N
    BD_BIDIR = "bd_bidir"             # V3: shift clean only, bidir MASK, blk=2N
    BD_BIDIR_SHIFT = "bd_bidir_shift" # V4: shift everywhere, bidir MASK, blk=2N-1
    HYBRID_DIFFUSION_SHIFT = "hybrid_diffusion_shift"       # HybridDiffusion: [seed, M, M, M], mask rows bidir


class DllmConfig:
    def __init__(
        self,
        algorithm: str,
        algorithm_config: dict[str, Any],
        block_size: int,
        mask_id: int,
        max_running_requests: int,
        causal_prefill: bool = False,
        variant: str = "causal_shift",
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.block_size = block_size
        self.mask_id = mask_id
        self.max_running_requests = max_running_requests
        self.causal_prefill = causal_prefill
        self.variant = SelfSpecVariant(variant)

    @staticmethod
    def from_server_args(
        server_args: ServerArgs,
    ):
        if server_args.dllm_algorithm is None:
            return None

        algorithm_config = {}
        if server_args.dllm_algorithm_config is not None:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Please install PyYAML to use YAML config files. "
                    "`pip install pyyaml`"
                )
            with open(server_args.dllm_algorithm_config, "r") as f:
                algorithm_config = yaml.safe_load(f) or {}

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
            model_revision=server_args.revision,
        )
        DLLM_PARAMS = {
            "LLaDA2MoeModelLM": {"block_size": 32, "mask_id": 156895},
            # Qwen3/SDAR checkpoints use the tokenizer MASK token. Qwen3.5
            # hybrid training configs use a tokenizer-specific mask id unless
            # config/YAML overrides it.
            "SDARForCausalLM": {"block_size": 4, "mask_id": 151669},
            "Qwen3DLLMForCausalLM": {"block_size": 4, "mask_id": 151669},
            "SDARMoeForCausalLM": {"block_size": 4, "mask_id": 151669},
            "Qwen3ForCausalLM": {"block_size": 4, "mask_id": 151669},
            # Qwen3.5 hybrid (softmax + GDN) DLLM
            "Qwen3_5DLLMForCausalLM": {"block_size": 4, "mask_id": 248077},
            "Qwen3_5DLLMForConditionalGeneration": {
                "block_size": 4,
                "mask_id": 248077,
            },
        }

        arch = model_config.hf_config.architectures[0]
        if arch in DLLM_PARAMS:
            params = DLLM_PARAMS[arch]
            block_size = params["block_size"]
            mask_id = params["mask_id"]
        elif "block_size" in algorithm_config and "mask_id" in algorithm_config:
            # HybridDiffusion checkpoints may export a new architecture name before this
            # serving fork knows about it. Allow explicit YAML to define the
            # dLLM shape/token contract instead of rejecting the model.
            block_size = algorithm_config["block_size"]
            mask_id = algorithm_config["mask_id"]
        else:
            raise RuntimeError(f"Unknown diffusion LLM: {arch}")

        # Prefer block_size from model config if available (e.g. block_size=1
        # models vs the default block_size=4 in DLLM_PARAMS).
        hf_block_size = getattr(model_config.hf_config, "block_size", None)
        if hf_block_size is not None:
            block_size = hf_block_size

        # Prefer mask_id from model config if available. Different ckpts
        # under the same arch may use different mask tokens.
        hf_mask_id = getattr(model_config.hf_config, "mask_token_id", None)
        if hf_mask_id is None:
            hf_mask_id = getattr(model_config.hf_config, "mask_id", None)
        if hf_mask_id is not None:
            mask_id = hf_mask_id

        # Models with use_regular_causal=True were trained with causal
        # attention for prompt (x0) tokens; prefill must use causal attention.
        hf_causal_prefill = getattr(
            model_config.hf_config, "use_regular_causal", None
        )
        if hf_causal_prefill is None and arch in {
            "Qwen3DLLMForCausalLM",
            "Qwen3ForCausalLM",
            "Qwen3_5DLLMForCausalLM",
            "Qwen3_5DLLMForConditionalGeneration",
        }:
            # Our Qwen3 DLLM checkpoints were trained with causal x0/prompt
            # attention, but older exports do not carry use_regular_causal in
            # config.json. Default them to causal prefill so hybrid-mask
            # decoding remains numerically sane.
            causal_prefill = True
        else:
            causal_prefill = bool(hf_causal_prefill)

        max_running_requests = (
            1
            if server_args.max_running_requests is None
            else server_args.max_running_requests
        )

        if algorithm_config:
            block_size = algorithm_config.get("block_size", block_size)
            mask_id = algorithm_config.get("mask_id", mask_id)

        # Allow YAML to override causal_prefill (some models have
        # use_regular_causal=None but still need causal prefill).
        if algorithm_config.get("causal_prefill") is not None:
            causal_prefill = algorithm_config["causal_prefill"]

        variant = algorithm_config.get("variant", "causal_shift")
        if variant == SelfSpecVariant.HYBRID_DIFFUSION_SHIFT.value:
            algorithm_config.setdefault("gen_block_size", 1)
            block_size = algorithm_config.get("block_size", block_size)
            if block_size < 2:
                raise RuntimeError(
                    "variant=hybrid_diffusion_shift requires block_size>=2 "
                    f"([seed, one-or-more MASKs]), got {block_size}"
                )
            causal_prefill = algorithm_config.get("causal_prefill", True)

        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=mask_id,
            max_running_requests=max_running_requests,
            causal_prefill=causal_prefill,
            variant=variant,
        )
