# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
"""End-to-end smoke test for the extracted Pinball package.

Builds a tiny Pinball model, runs a few autoregressive training steps through the real
trainer path (`train_with_hybrid_masking`), and an eval forward. Asserts the loss is finite
and trends down. This is the Phase-A gate: it must pass before any trimming.

Run directly:   python tests/test_smoke.py
Run via pytest: pytest tests/test_smoke.py -s
"""
import os
import sys
import pathlib

import torch
import torch.nn as nn

# Make `pinball` importable without installation.
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from transformers import AutoTokenizer
from pinball import build_model, PinballConfig, count_parameters
from pinball.train import train_with_hybrid_masking


TINY_CFG = dict(
    model_type="pinball",
    modality="text",
    tokenizer_name="gpt2",
    block_size=64,
    hidden_dim=64,
    num_heads=4,
    num_refinement_layers=2,
    num_layers=[0, 0, 0, 0],
    internal_cycles=[0, 0, 0, 0],
    refinement_style="unified",
    unified_refinement_cycles=1,
    compression_ratios=[8, 4, 2],
    overlap_ratios=[0.1, 0.2, 0.4],
    local_attn_windows=[16, 8, 8, 8],
    local_attn_levels=[0, 1, 2, 3],
    local_attn_causal_levels=[0, 1, 2, 3],
    learn_edge_from_attention=True,
    dropout=0.0,
    norm_type="layernorm",
    lap_pe_k=0,
    l0_cycles=0,
    iterative_refinement_cycles=0,
    local_connectivity_window_size=0,
    attn_backend="sdpa",          # friendly toggle -> l0_local_backend
    l0_local_window=16,
    use_hqd=False,                # friendly toggle -> hierarchical_query_descent_enable
    train_mode="ar",              # friendly toggle -> train_objective_mode="ar"
)


def _build_tiny():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = PinballConfig(**TINY_CFG)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    vocab_size = len(tokenizer)
    model = build_model(
        cfg, tokenizer=tokenizer, vocab_size=vocab_size,
        input_mode="tokens", tie_weights=True, max_seq_len=cfg.block_size,
    ).to(device)
    model.emit_features_only = True
    return model, tokenizer, cfg, device


def test_pinball_trains_and_evals():
    torch.manual_seed(0)
    model, tokenizer, cfg, device = _build_tiny()
    n_params = count_parameters(model)
    print(f"[smoke] device={device} params={n_params:,}")
    assert n_params > 0

    B, T = 2, cfg.block_size
    vocab = len(tokenizer)
    torch.manual_seed(1)
    input_ids = torch.randint(0, tokenizer.vocab_size, (B, T), device=device)
    batch = {"input_ids": input_ids}

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    model.train()
    losses = []
    for step in range(12):
        # train_with_hybrid_masking does the forward + .backward() (grad accumulation);
        # the caller is responsible for zero_grad / optimizer.step (see trainer class).
        optimizer.zero_grad()
        loss = train_with_hybrid_masking(
            model, batch, criterion, optimizer, tokenizer,
            gradient_accumulation_steps=1, mixed_precision=False, device=device,
            objective_mode="ar", lambda_ar=1.0, lambda_base_ce=1.0, lambda_masked=0.0,
        )
        optimizer.step()
        print(f"[smoke] step {step} ar_loss={loss:.4f}")
        losses.append(float(loss))

    assert all(l == l for l in losses), f"NaN in losses: {losses}"  # NaN check
    assert all(l != float("inf") for l in losses), f"inf in losses: {losses}"
    # Loss should trend down over a handful of steps on a fixed batch (overfit).
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"

    # Eval forward.
    model.eval()
    with torch.no_grad():
        out = model(input_ids)
    feats = out[0] if isinstance(out, (tuple, list)) else out
    assert torch.isfinite(feats).all(), "non-finite eval output"
    print(f"[smoke] eval output shape={tuple(feats.shape)} OK")


if __name__ == "__main__":
    test_pinball_trains_and_evals()
    print("SMOKE OK")
