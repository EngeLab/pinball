# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
"""Hardware-aware autocast dtype selection.

Native (fast, tensor-core) bf16 needs Ampere (sm_80) or newer. GPUs like the T4
(Turing, sm_75) have no native bf16 — autocasting to bf16 there is emulated and slow.

NOTE: ``torch.cuda.is_bf16_supported()`` is NOT a reliable native check — in recent
PyTorch it returns True on a T4 because it counts bf16 *emulation*. We gate on the
device compute capability (major >= 8) instead.
"""
import torch


def bf16_supported() -> bool:
    """True only if the current CUDA device has native (sm_80+) bf16."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return bool(torch.cuda.is_bf16_supported())


def amp_dtype(device_type: str = "cuda") -> torch.dtype:
    """Return the AMP autocast dtype to use for the given device type."""
    if str(device_type) == "cuda" and bf16_supported():
        return torch.bfloat16
    return torch.float16
