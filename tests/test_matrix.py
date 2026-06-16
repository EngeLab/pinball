# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
"""Coverage matrix: build + train-a-few-steps across the kept feature combinations, so the
elegance-pass scaffolding removal has a real safety net (not just the single-path smoke test).

Covers: attn_backend {sdpa, pyg} x qkv_sharing {shared, separate}, the masked-diffusion
objective, HQD, and the NeighborLoader/blockdiag sampled path. Each combo asserts loss is
finite and decreasing on a fixed batch.

Run: python tests/test_matrix.py   (CUDA recommended; flash combos auto-skip on CPU)
"""
import sys, pathlib, itertools
import torch
import torch.nn as nn

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from transformers import AutoTokenizer
from pinball import build_model, PinballConfig
from pinball.train import train_with_hybrid_masking

_DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_TOK = AutoTokenizer.from_pretrained("gpt2")
_TOK.pad_token = _TOK.eos_token
if _TOK.mask_token is None:
    _TOK.add_special_tokens({"mask_token": "<mask>"})

BASE = dict(
    model_type="pinball", modality="text", tokenizer_name="gpt2",
    block_size=64, hidden_dim=64, num_heads=4, num_refinement_layers=2,
    num_layers=[0, 0, 0, 0], internal_cycles=[0, 0, 0, 0], refinement_style="unified",
    unified_refinement_cycles=1, l0_cycles=0, iterative_refinement_cycles=0,
    local_connectivity_window_size=0, compression_ratios=[8, 4, 2],
    overlap_ratios=[0.1, 0.2, 0.4], norm_type="layernorm", dropout=0.0,
    local_attn_levels=[0, 1, 2, 3], local_attn_windows=[16, 8, 8, 8], l0_local_window=16,
    learn_edge_from_attention=True,
)


def _run(name, overrides, objective="ar", steps=6):
    torch.manual_seed(0)
    cfg = PinballConfig(**{**BASE, **overrides})
    model = build_model(cfg, tokenizer=_TOK, vocab_size=len(_TOK), input_mode="tokens",
                        tie_weights=True, max_seq_len=cfg.block_size).to(_DEV)
    model.emit_features_only = True
    torch.manual_seed(1)
    batch = {"input_ids": torch.randint(0, _TOK.vocab_size, (2, cfg.block_size), device=_DEV)}
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    use_amp = _DEV.type == "cuda"
    losses = []
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = train_with_hybrid_masking(
            model, batch, crit, opt, _TOK, device=_DEV, objective_mode=objective,
            mixed_precision=use_amp, use_bf16=True,
            lambda_ar=1.0 if objective == "ar" else 0.1,
            lambda_masked=1.0 if objective != "ar" else 0.0, lambda_base_ce=1.0,
        )
        opt.step(); losses.append(float(loss))
    assert all(l == l for l in losses), f"{name}: NaN {losses}"
    assert losses[-1] < losses[0] + 1e-3, f"{name}: not decreasing {losses[0]:.3f}->{losses[-1]:.3f}"
    print(f"OK  {name:42s} {losses[0]:.3f} -> {losses[-1]:.3f}")


def test_matrix():
    for backend, qkv in itertools.product(["sdpa", "pyg"], ["shared", "separate"]):
        _run(f"{backend}/{qkv}/ar", {"attn_backend": backend, "qkv_sharing": qkv})
    _run("sdpa/separate/masked_diffusion",
         {"attn_backend": "sdpa", "qkv_sharing": "separate", "use_hybrid_masking": True},
         objective="masked")
    _run("sdpa/separate/hqd", {"attn_backend": "sdpa", "qkv_sharing": "separate", "use_hqd": True})
    _run("sdpa/sampled-blockdiag",
         {"attn_backend": "sdpa", "use_neighbor_sampling": True, "num_neighbors": [8],
          "refinement_batch_mode": "blockdiag"})


if __name__ == "__main__":
    test_matrix()
    print("MATRIX OK")
