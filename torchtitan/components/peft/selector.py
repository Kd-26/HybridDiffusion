# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from fnmatch import fnmatchcase

import torch.nn as nn


@dataclass(frozen=True)
class ModuleSelection:
    fqn: str
    parent_fqn: str
    child_name: str
    module: nn.Module


def _matches_any(fqn: str, globs: list[str]) -> bool:
    return any(fnmatchcase(fqn, pattern) for pattern in globs)


def select_target_modules(
    model: nn.Module,
    target_globs: list[str],
    exclude_globs: list[str],
) -> tuple[list[ModuleSelection], list[str]]:
    """Select modules by FQN with exact glob matching."""
    selections: list[ModuleSelection] = []
    unsupported_fqns: list[str] = []

    for fqn, module in model.named_modules():
        if not fqn:
            continue
        if not _matches_any(fqn, target_globs):
            continue
        if _matches_any(fqn, exclude_globs):
            continue

        if not isinstance(module, nn.Linear):
            unsupported_fqns.append(fqn)
            continue

        parent_fqn, _, child_name = fqn.rpartition(".")
        selections.append(
            ModuleSelection(
                fqn=fqn,
                parent_fqn=parent_fqn,
                child_name=child_name,
                module=module,
            )
        )

    selections.sort(key=lambda item: item.fqn)
    unsupported_fqns.sort()
    return selections, unsupported_fqns


def select_parameter_fqns(model: nn.Module, target_globs: list[str]) -> list[str]:
    """Select parameter FQNs by exact glob matching."""
    matched = {
        fqn
        for fqn, _ in model.named_parameters(remove_duplicate=False)
        if _matches_any(fqn, target_globs)
    }
    return sorted(matched)

