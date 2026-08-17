# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import torch.nn as nn


def canonicalize_fqn(fqn: str) -> str:
    """Normalize wrapper-specific path segments out of an FQN."""
    return ".".join(part for part in fqn.split(".") if part and part != "_orig_mod")


def build_parameter_alias_maps(
    model: nn.Module,
) -> tuple[dict[int, list[str]], dict[str, list[nn.Parameter]]]:
    """Build alias maps keyed by parameter identity and canonical FQN."""
    alias_names_by_id: dict[int, set[str]] = defaultdict(set)
    params_by_canonical_name: dict[str, dict[int, nn.Parameter]] = defaultdict(dict)

    for raw_name, param in model.named_parameters(remove_duplicate=False):
        canonical_name = canonicalize_fqn(raw_name)
        alias_names_by_id[id(param)].add(canonical_name)
        params_by_canonical_name[canonical_name][id(param)] = param

    alias_lists = {
        param_id: sorted(names) for param_id, names in alias_names_by_id.items()
    }
    param_lists = {
        name: list(params.values())
        for name, params in params_by_canonical_name.items()
    }
    return alias_lists, param_lists


def summarize_parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable

