# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = float(eps)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        if self.weight is not None:
            x = x * self.weight
        if x.dtype != input_dtype:
            x = x.to(dtype=input_dtype)
        return x


def make_norm(
    normalized_shape,
    norm_type: str = "rmsnorm",
    eps: float = 1e-6,
    elementwise_affine: bool = True,
):
    kind = "rmsnorm" if norm_type is None else str(norm_type).replace("_", "").lower()
    if kind in {"layernorm", "ln"}:
        return nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    if kind in {"rmsnorm", "rms"}:
        if hasattr(nn, "RMSNorm"):
            return nn.RMSNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        return RMSNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    raise ValueError(f"Unknown norm_type '{norm_type}'")
