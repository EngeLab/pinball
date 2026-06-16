# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
from collections import OrderedDict
from typing import Dict, Tuple


def tied_output_head(model) -> bool:
    token_embedding = getattr(model, "token_embedding", None)
    output_projection = getattr(model, "output_projection", None)
    if token_embedding is None or output_projection is None:
        return False
    token_weight = getattr(token_embedding, "weight", None)
    output_weight = getattr(output_projection, "weight", None)
    if token_weight is None or output_weight is None:
        return False
    return int(token_weight.data_ptr()) == int(output_weight.data_ptr())


def bucket_for_param(name: str, model_type: str) -> str:
    if name.startswith("token_embedding"):
        return "token_embedding"
    if name.startswith(("position_embedding", "position_embeddings")):
        return "position_embedding"
    if name.startswith("output_projection"):
        return "output_head"
    if model_type == "transformer":
        if name.startswith("blocks"):
            return "transformer_blocks"
        if name.startswith("ln_f"):
            return "final_norm"
        return "other"
    if model_type == "pinball_dynamic":
        if name.startswith("level_blocks"):
            return "level_blocks"
        if name.startswith("message_bus"):
            return "message_bus"
        if name.startswith("universal_graph"):
            return "universal_graph"
        if name.startswith("native_graph"):
            return "native_graph"
        if name.startswith(("sparse_q", "sparse_k", "sparse_norms")):
            return "sparse_retrieval"
        if name.startswith("lap_pe_proj"):
            return "lap_pe"
        if name.startswith("upper_init"):
            return "upper_initializers"
        if name.startswith("final_norm"):
            return "final_norm"
        return "other"
    if name.startswith(("level_transformers", "refinement_transformers")):
        return "transformer_blocks"
    if name.startswith(("pinball_mem_read", "pinball_mem_write", "pinball_mem_gates", "pinball_work_in", "pinball_work_out")):
        return "hetero_adapters"
    if name.startswith(("pinball_upper_refiner", "pinball_upper_cross_refiner", "pinball_cross_query_refiners", "pinball_top_refiner")):
        return "multirate_refiners"
    if name.startswith(("upward_projections", "downward_projections", "level_projections")):
        return "hierarchy_projections"
    return "other"


def parameter_breakdown(model, model_type: str) -> Tuple[OrderedDict, int, int]:
    buckets = OrderedDict()
    for key in (
        "token_embedding",
        "position_embedding",
        "output_head",
        "transformer_blocks",
        "level_blocks",
        "message_bus",
        "universal_graph",
        "native_graph",
        "sparse_retrieval",
        "lap_pe",
        "upper_initializers",
        "hetero_adapters",
        "multirate_refiners",
        "hierarchy_projections",
        "final_norm",
        "other",
    ):
        buckets[key] = 0

    total = 0
    trainable = 0
    seen = set()
    for name, param in model.named_parameters():
        ptr = int(param.data_ptr())
        if ptr in seen:
            continue
        seen.add(ptr)
        count = int(param.numel())
        total += count
        if param.requires_grad:
            trainable += count
        buckets[bucket_for_param(name, model_type)] += count
    return buckets, total, trainable


def module_param_count(module) -> int:
    if module is None:
        return 0
    seen = set()
    total = 0
    for param in module.parameters(recurse=True):
        ptr = int(param.data_ptr())
        if ptr in seen:
            continue
        seen.add(ptr)
        total += int(param.numel())
    return total


def params_to_millions(value: int) -> float:
    return float(value) / 1_000_000.0


def fmt_millions(value: int) -> str:
    return f"{params_to_millions(value):.2f}M"


def parameter_summary(model, model_type: str) -> Dict[str, object]:
    buckets, total, trainable = parameter_breakdown(model, model_type=model_type)
    tied = tied_output_head(model)
    token_embedding_params = module_param_count(getattr(model, "token_embedding", None))
    output_projection_params = module_param_count(getattr(model, "output_projection", None))
    bucket_params = {key: int(value) for key, value in buckets.items() if int(value) > 0}
    bucket_params_m = {key: params_to_millions(value) for key, value in bucket_params.items()}
    return {
        "total": int(total),
        "trainable": int(trainable),
        "total_m": params_to_millions(total),
        "trainable_m": params_to_millions(trainable),
        "buckets": bucket_params,
        "buckets_m": bucket_params_m,
        "tied_output_head": bool(tied),
        "token_embedding_params": int(token_embedding_params),
        "output_projection_physical_params": int(output_projection_params),
    }


def format_parameter_summary_lines(summary: Dict[str, object]) -> list:
    lines = [
        f"total={fmt_millions(int(summary['total']))} ({int(summary['total']):,})",
        f"trainable={fmt_millions(int(summary['trainable']))} ({int(summary['trainable']):,})",
    ]
    buckets = summary.get("buckets", {}) or {}
    tied = bool(summary.get("tied_output_head", False))
    for key, value in buckets.items():
        suffix = " tied" if key == "output_head" and tied else ""
        lines.append(f"{key}={fmt_millions(int(value))}{suffix}")
    if tied and "output_head" not in buckets:
        physical = int(summary.get("output_projection_physical_params", 0))
        lines.append(f"output_head=0.00M tied (physical={fmt_millions(physical)})")
    return lines
