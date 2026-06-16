# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
"""Coverage for EnhancedHierarchicalTrainer.train_epoch + validate (the trainer-class path,
which the smoke/matrix tests don't exercise). Used as the safety net for trainer cleanup.

Run: python tests/test_trainer_loop.py
"""
import sys, pathlib
import torch
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from transformers import AutoTokenizer
from pinball import build_model, PinballConfig
from pinball.train import EnhancedHierarchicalTrainer

TINY = dict(
    model_type="pinball", modality="text", tokenizer_name="gpt2",
    block_size=64, hidden_dim=64, num_heads=4, num_refinement_layers=2,
    num_layers=[0, 0, 0, 0], internal_cycles=[0, 0, 0, 0], refinement_style="unified",
    unified_refinement_cycles=1, l0_cycles=0, iterative_refinement_cycles=0,
    local_connectivity_window_size=0, compression_ratios=[8, 4, 2],
    overlap_ratios=[0.1, 0.2, 0.4], norm_type="layernorm", dropout=0.0,
    local_attn_levels=[0, 1, 2, 3], local_attn_windows=[16, 8, 8, 8], l0_local_window=16,
    learn_edge_from_attention=True, attn_backend="sdpa",
)


def test_train_epoch_and_validate():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = PinballConfig(**TINY)
    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<mask>"})
    model = build_model(cfg, tokenizer=tok, vocab_size=len(tok), input_mode="tokens",
                        tie_weights=True, max_seq_len=cfg.block_size).to(dev)
    model.emit_features_only = True
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    trainer = EnhancedHierarchicalTrainer(
        model, None, optimizer=opt, tokenizer=tok, device=dev,
        train_objective_mode="ar", mixed_precision=(dev.type == "cuda"),
        log_interval=1, eval_interval=1,
    )

    def get_batch(device):
        ids = torch.randint(0, tok.vocab_size, (2, cfg.block_size), device=device)
        return {"input_ids": ids, "labels": ids.clone()}

    data = {"get_batch": get_batch, "steps_per_epoch": 3}
    loss = trainer.train_epoch(data, 0)
    print(f"[trainer] train_epoch loss={loss}")
    assert loss == loss, "NaN train loss"
    out = trainer.validate(data)
    print(f"[trainer] validate -> {out!r}")


if __name__ == "__main__":
    test_train_epoch_and_validate()
    print("TRAINER LOOP OK")
