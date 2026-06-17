# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
# model/batched_layer_executor.py
# import torch
# from torch_geometric.loader import NeighborLoader

# class BatchedLayerExecutor:
#     """
#     Runs a single GNN/Transformer layer across a large graph by sweeping seed nodes
#     (e.g., L0/L1) in batches, with exact 1-hop subgraphs (num_neighbors=[-1]).
#     Writes results back into a single x_global buffer.
#     """

#     def __init__(self, device="cuda", batch_size=8192, pin_memory=True, num_workers=0):
#         self.device = torch.device(device)
#         self.batch_size = batch_size
#         self.pin_memory = pin_memory
#         self.num_workers = num_workers

#     def _mask_to_indices(self, mask_or_idx):
#         if isinstance(mask_or_idx, torch.Tensor) and mask_or_idx.dtype == torch.bool:
#             return mask_or_idx.nonzero(as_tuple=False).view(-1)
#         return mask_or_idx  # already indices

#     @torch.no_grad()
#     def run_one_layer(
#         self,
#         layer,                # your HierarchicalTransformerLayer (unified)
#         data,                 # PyG Data on CPU (edge_index, node_level, edge_type/attr)
#         x_global,             # torch.Tensor [N, H] on CPU (pinned recommended) or GPU
#         seed_nodes,           # indices or bool mask of nodes to sweep (e.g., L0, then L1)
#         extra_kwargs=None,    # dict with fixed inputs: level_offsets, etc.
#     ):
#         """
#         Executes `layer` across seed_nodes in batches. For each batch, it:
#           1) samples exact 1-hop subgraph of the seeds (neighbors included)
#           2) copies subgraph & x_sub to GPU
#           3) runs the layer
#           4) writes back to x_global for those subgraph nodes
#         """
#         extra_kwargs = extra_kwargs or {}

#         seeds = self._mask_to_indices(seed_nodes)
#         if seeds.numel() == 0:
#             return x_global  # nothing to do

#         # Always exact 1-hop neighbors (induced subgraph of the batch frontier)
#         loader = NeighborLoader(
#             data,
#             num_neighbors=[-1],         # exact 1-hop
#             input_nodes=seeds,          # seeds to sweep
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             persistent_workers=False,
#             pin_memory=self.pin_memory,
#         )

#         # We’ll keep one pre-allocated CUDA buffer for speed (optional)
#         for sub in loader:
#             # sub is an induced Data: sub.n_id maps back to global node ids
#             n_id = sub.n_id  # [n_sub]
#             # Gather features for subgraph
#             x_sub_cpu = x_global.index_select(0, n_id)

#             # Move to GPU
#             x_sub = x_sub_cpu.to(self.device, non_blocking=True)
#             edge_index = sub.edge_index.to(self.device, non_blocking=True)
#             node_level = sub.node_level.to(self.device, non_blocking=True)

#             edge_attr = getattr(sub, "edge_attr", None)
#             edge_type = getattr(sub, "edge_type", None)
#             if edge_attr is not None:
#                 edge_attr = edge_attr.to(self.device, non_blocking=True)
#             if edge_type is not None:
#                 edge_type = edge_type.to(self.device, non_blocking=True)

#             # Call your unified layer
#             out = layer(
#                 x_sub,
#                 edge_index,
#                 node_level,
#                 # Your layer signature may be (positions=None, level_offsets=..., edge_attr=...)
#                 positions=None,
#                 level_offsets=extra_kwargs.get("level_offsets", None),
#                 edge_attr=edge_attr
#             )

#             # Write back to x_global
#             out_cpu = out.to(x_global.device, non_blocking=True)
#             x_global.index_copy_(0, n_id, out_cpu)

#         return x_global
    
# model/batched_layer_executor.py
# model/batched_layer_executor.py
import torch
from torch_geometric.loader import NeighborLoader


class BatchedLayerExecutor_old:
    def __init__(self, batch_size=8192, pin_memory=True, num_workers=0):
        self.batch_size = int(batch_size)
        self.pin_memory = bool(pin_memory)
        self.num_workers = int(num_workers)

    #@torch.no_grad()
    def run_one_layer(self, layer, data, x_global, seed_nodes, extra_kwargs=None):
        extra_kwargs = extra_kwargs or {}

        loader = NeighborLoader(
            data,
            num_neighbors=[-1],      # exact 1-hop
            input_nodes=seed_nodes,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            subgraph_type="induced",  # replaces deprecated `directed`
        )

        def _layer_device(l):
            try:
                return next(l.parameters()).device
            except StopIteration:
                for b in l.buffers(recurse=True):
                    return b.device
                return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

        for batch in loader:
            target_device = _layer_device(layer)
            n_id = batch.n_id

            # move slice of global features to correct device
            x_sub = x_global[n_id].to(target_device, non_blocking=True)

            edge_index = batch.edge_index.to(target_device, non_blocking=True)
            node_level = (
                batch.node_level.to(target_device, non_blocking=True)
                if hasattr(batch, "node_level") and batch.node_level is not None
                else None
            )
            positions = n_id.to(target_device, non_blocking=True)
            level_offsets = None  # ensures your RoPE code uses positions path

            out = layer(
                x_sub,
                edge_index,
                node_level,
                positions=positions,
                level_offsets=level_offsets,
            )

            # write back to CPU global store
            x_global[n_id] = out.to(x_global.device, non_blocking=True)

            del x_sub, edge_index, node_level, positions, out
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)

        return x_global
    
import torch
from torch.utils.checkpoint import checkpoint
from torch_geometric.loader import NeighborLoader
import contextlib
from ..utils.amp import amp_dtype

class BatchedLayerExecutor_v1:
    def __init__(self, default_batch_size=1, pin_memory=True, num_workers=0, use_ckpt=False):
        self.default_batch_size = int(default_batch_size)
        self.pin_memory = bool(pin_memory)
        self.num_workers = int(num_workers)
        self.use_ckpt = bool(use_ckpt)

    def run_one_layer_long(self, layer, data, x_global, seed_nodes, batch_size=None, extra_kwargs=None):
        extra_kwargs = extra_kwargs or {}
        bs = int(batch_size) if batch_size is not None else self.default_batch_size

        loader = NeighborLoader(
            data,
            num_neighbors=[-1],      # exact 1-hop
            input_nodes=seed_nodes,
            batch_size=bs,
            shuffle=False,
            prefetch_factor=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
            subgraph_type="induced",
        )

        def _layer_device(l):
            for p in l.parameters(recurse=True):
                return p.device
            for b in l.buffers(recurse=True):
                return b.device
            return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

        # tiny wrapper so checkpoint can call it with tensors only
        def _call_layer(x_sub, edge_index, node_level, positions, edge_attr=None):
            return layer(
                x_sub,
                edge_index,
                node_level,
                positions=positions,
                edge_attr=None,
                level_offsets=None,  # positions path (RoPE uses positions)
            )

        def _param_device(mod: torch.nn.Module) -> torch.device:
            for p in mod.parameters(recurse=True):
                return p.device
            for b in mod.buffers(recurse=True):
                return b.device
            return torch.device("cpu")
            
        for batch in loader:
            target_device = _layer_device(layer)
            n_id = batch.n_id

            # slice *previous* layer state for this mini-batch
            x_sub = x_global[n_id].to(target_device, non_blocking=True)
            # LapPE slice & add
            if hasattr(data, "lap_pe_proj_cpu") and data.lap_pe_proj_cpu is not None:
                pe = data.lap_pe_proj_cpu[n_id].to(target_device, non_blocking=True)                           # CPU [B,H]
                x_sub = x_sub + pe#.to(target_device, non_blocking=True)
            # elif hasattr(data, "lap_pe_raw_cpu") and data.lap_pe_raw_cpu is not None:
            #     pe_raw = data.lap_pe_raw_cpu[n_id].to(target_device, non_blocking=True)  # [B,k]
            #     pe_proj = self_owner.lap_pe_proj.to(target_device)(pe_raw)                # [B,H]
            #     x_sub = x_sub + pe_proj
            edge_index = batch.edge_index.to(target_device, non_blocking=True)
            node_level = (
                batch.node_level.to(target_device, non_blocking=True)
                if hasattr(batch, "node_level") and batch.node_level is not None
                else None
            )
            positions = n_id.to(target_device, non_blocking=True)

            # --- activation checkpoint per mini-batch ---
            if self.use_ckpt and torch.is_grad_enabled():
                # use_reentrant=False avoids some engine overhead on new PyTorch
                out = checkpoint(_call_layer, x_sub, edge_index, node_level, positions, use_reentrant=False)
            else:
                out = _call_layer(x_sub, edge_index, node_level, positions)

            # write back OUT to the global store (CPU or GPU — wherever x_global lives)
            x_global[n_id] = out.to(x_global.device, non_blocking=True)

            # free per-batch temporaries asap
            del x_sub, edge_index, node_level, positions, out
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)

        return x_global
    
    def run_one_layer(self, layer, data, x_global, seed_nodes, batch_size=None, extra_kwargs=None):
        extra_kwargs = extra_kwargs or {}
        bs = int(batch_size) if batch_size is not None else self.default_batch_size

        loader = NeighborLoader(
            data,
            num_neighbors=[-1],         # exact 1-hop
            input_nodes=seed_nodes,
            batch_size=bs,
            shuffle=False,
            num_workers=self.num_workers,   # start with 0 for stability
            pin_memory=self.pin_memory,     # start False, then test True
            subgraph_type="induced",
            prefetch_factor=None,
            persistent_workers=False,
        )

        # snapshot read (break cross-batch graph) + write buffer
        x_prev   = x_global#.detach()
        x_next   = torch.empty_like(x_prev, device=x_prev.device)
        touched  = torch.zeros((x_prev.size(0), 1), dtype=torch.bool, device=x_prev.device)

        def _layer_device(l):
            try:
                return next(l.parameters()).device
            except StopIteration:
                for b in l.buffers(recurse=True):
                    return b.device
                return x_prev.device

        target_device = _layer_device(layer)

        # tiny callable to allow checkpoint with tensor-only signature
        def _call_layer(x_sub, edge_index, node_level, positions, edge_attr=None):
            return layer(
                x_sub, edge_index, node_level,
                positions=positions, level_offsets=None,
                edge_attr=edge_attr
            )

        use_amp = torch.is_grad_enabled()
        amp_ctx = torch.autocast(device_type=target_device.type, dtype=amp_dtype(target_device.type)) if use_amp else contextlib.nullcontext()

        # cache level_offsets tensor on device for fast per-batch indexing
        level_offsets_dev = torch.as_tensor(getattr(data, "level_offsets", []), device=target_device)

        for batch in loader:
            n_id = batch.n_id
            x_sub = x_prev[n_id].to(target_device, non_blocking=True)

            # LapPE slice & add
            if hasattr(data, "lap_pe_proj_cpu") and data.lap_pe_proj_cpu is not None:
                pe = data.lap_pe_proj_cpu[n_id].to(target_device, non_blocking=True)
                x_sub = x_sub + pe

            edge_index = batch.edge_index.to(target_device, non_blocking=True)
            node_level = batch.node_level.to(target_device, non_blocking=True) if hasattr(batch, "node_level") else None

            # level-aware RoPE positions
            if node_level is not None and level_offsets_dev.numel() > 0:
                pos_in_level = n_id.to(target_device, non_blocking=True) - level_offsets_dev[node_level]
            else:
                pos_in_level = n_id.to(target_device, non_blocking=True)

            # (optional) edge_attr, if used
            edge_attr = None
            # if getattr(self, "use_edge_attr", False) and getattr(self, "edge_feature_generator", None) is not None and hasattr(batch, "edge_type"):
            #     edge_attr = self.edge_feature_generator(
            #         x_sub, edge_index, batch.edge_type.to(target_device, non_blocking=True)
            #     )

            with amp_ctx:
                if self.use_ckpt and use_amp:
                    out = torch.utils.checkpoint.checkpoint(
                        _call_layer, x_sub, edge_index, node_level, pos_in_level,
                        use_reentrant=False, edge_attr=edge_attr
                    )
                else:
                    out = _call_layer(x_sub, edge_index, node_level, pos_in_level, edge_attr=edge_attr)

            # write into x_next (replace semantics; avoids shrink)
            x_next[n_id] = out.to(x_next.device, non_blocking=True)
            touched[n_id] = True

            # free temps aggressively
            del x_sub, edge_index, node_level, pos_in_level, out, batch, edge_attr

        # keep-old-where-not-updated
        x_next = torch.where(touched, x_next, x_prev)
        return x_next

import contextlib
from typing import Iterable, Optional, Tuple, List
import torch
from torch_geometric.loader import NeighborLoader

def _split_levels(node_level_cpu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # node_level_cpu must be on CPU
    L0 = (node_level_cpu == 0).nonzero(as_tuple=False).view(-1)
    L1 = (node_level_cpu == 1).nonzero(as_tuple=False).view(-1)
    L2 = (node_level_cpu == 2).nonzero(as_tuple=False).view(-1)
    L3 = (node_level_cpu == 3).nonzero(as_tuple=False).view(-1)
    return L0, L1, L2, L3

def _iter_l0_windows(l0_idx: torch.Tensor, window_size: int, stride: int, halo: int) -> Iterable[torch.Tensor]:
    """
    Yields L0 windows with halo (overlap) as CPU index tensors into the *global node index space*.

    l0_idx should be sorted ascending and is a CPU index tensor of all L0 nodes (global ids).
    """
    assert l0_idx.device.type == "cpu"
    if l0_idx.numel() == 0:
        return
    # l0_idx is sorted by construction in most pipelines; sort defensively
    l0_sorted = torch.sort(l0_idx).values
    n = l0_sorted.numel()
    # window boundaries refer to *positions in l0_sorted*, not raw node ids
    start = 0
    while start < n:
        end = min(start + window_size, n)
        # halo as extra neighbors in index space
        h0 = max(0, start - halo)
        h1 = min(n, end + halo)
        # slice and map back to global node ids
        yield l0_sorted[h0:h1]
        if end == n:
            break
        start += stride

class BatchedLayerExecutor:
    """
    Stream-optimized exact 1-hop minibatch executor with two modes:

    - mode="full":   preserves autograd graph across the whole layer pass.
    - mode="windowed": breaks cross-batch autograd with a detached snapshot,
                       while still learning layer parameters.

    Extras:
      • LapPE: assumes CPU-cached `data.lap_pe_proj_cpu` [N,H] (optional).
      • Level-aware RoPE positions via `level_offsets`.
      • Overlap-safe writeback (average when nodes appear in multiple minibatches).
      • Non-picklable CUDA stream is dropped/recreated to play nice with EMA deepcopy.
    """

    def __init__(
        self,
        default_batch_size: int = 8192,
        pin_memory: bool = True,
        num_workers: int = 0,
        use_ckpt: bool = True,
        use_amp: bool = True,
        prefetch_batches: int = 0,
    ):
        self.default_batch_size = int(default_batch_size)
        self.pin_memory = bool(pin_memory)
        self.num_workers = int(num_workers)
        self.use_ckpt = bool(use_ckpt)
        self.use_amp = bool(use_amp)
        self.prefetch_batches = int(prefetch_batches)

        # Not picklable: we drop it in __getstate__/__setstate__
        self._h2d_stream: Optional[torch.cuda.Stream] = None
        if torch.cuda.is_available():
            try:
                self._h2d_stream = torch.cuda.Stream()
            except Exception:
                self._h2d_stream = None

    # --- EMA / deepcopy compatibility ---
    def __getstate__(self):
        st = self.__dict__.copy()
        st["_h2d_stream"] = None
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self._h2d_stream = None
        if torch.cuda.is_available():
            try:
                self._h2d_stream = torch.cuda.Stream()
            except Exception:
                self._h2d_stream = None

    def _layer_device(self, layer: torch.nn.Module, fallback: torch.device) -> torch.device:
        for p in layer.parameters(recurse=True):
            return p.device
        for b in layer.buffers(recurse=True):
            return b.device
        return fallback

    def _loader(self, data, seed_nodes, batch_size: int):
        return NeighborLoader(
            data,
            num_neighbors=[-1],         # exact 1-hop
            input_nodes=seed_nodes,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            subgraph_type="induced",
            prefetch_factor=None if self.num_workers == 0 else max(2, self.prefetch_batches),
            persistent_workers=(self.num_workers > 0),
        )

    def run_one_layer_v1(
        self,
        *,
        layer: torch.nn.Module,
        data,                       # CPU PyG Data: edge_index, node_level, level_offsets, (lap_pe_proj_cpu optional)
        x_global: torch.Tensor,     # [N,H] global state tensor (CPU or GPU)
        seed_nodes: torch.Tensor,   # indices to sweep (CPU)
        batch_size: Optional[int] = None,
        mode: str = "full",     # "windowed" | "full"
        extra_kwargs=None,
    ) -> torch.Tensor:
        """
        Returns x_next (same shape/device as x_global).
        - "full":   x_prev = x_global (no detach), graph can chain across layer.
        - "windowed": x_prev = x_global.detach(), breaks cross-batch autograd, keeps grads to params.
        """

        assert data.edge_index.device.type == "cpu", "data.edge_index must be on CPU"
        bs = int(batch_size) if batch_size is not None else self.default_batch_size
        loader = self._loader(data, seed_nodes, bs)

        # snapshot read (controls cross-batch autograd)
        x_prev = x_global if mode == "full" else x_global.detach()
        x_next = torch.empty_like(x_prev, device=x_prev.device)
        cnts   = torch.zeros((x_prev.size(0), 1), dtype=x_prev.dtype, device=x_prev.device)

        target_device = self._layer_device(layer, fallback=x_prev.device)

        # cache offsets (for level-aware RoPE positions)
        level_offsets = getattr(data, "level_offsets", [])
        level_offsets_dev = torch.as_tensor(level_offsets, device=target_device) if level_offsets else torch.empty(0, device=target_device)

        # AMP context
        use_amp = self.use_amp and torch.is_grad_enabled()
        amp_ctx = torch.autocast(device_type=target_device.type, dtype=amp_dtype(target_device.type)) if use_amp else contextlib.nullcontext()

        # tiny wrapper (plays nice with checkpoint)
        def _call_layer(x_sub, edge_index, node_level, positions, edge_attr=None):
            return layer(
                x_sub, edge_index, node_level,
                positions=positions, level_offsets=None, edge_attr=edge_attr
            )

        h2d_stream = self._h2d_stream if (self._h2d_stream is not None and target_device.type == "cuda") else None

        for batch in loader:
            n_id = batch.n_id                     # CPU indices into global tensors

            # --- H2D copies in a separate stream (when possible) ---
            if h2d_stream is not None:
                with torch.cuda.stream(h2d_stream):
                    x_sub = x_prev[n_id].to(target_device, non_blocking=True)
                    edge_index = batch.edge_index.to(target_device, non_blocking=True)
                    node_level = batch.node_level.to(target_device, non_blocking=True) if hasattr(batch, "node_level") else None

                    # LapPE add (projected & cached on CPU)
                    if hasattr(data, "lap_pe_proj_cpu") and data.lap_pe_proj_cpu is not None:
                        pe = data.lap_pe_proj_cpu[n_id].to(target_device, non_blocking=True)
                        x_sub = x_sub + pe

                    # level-aware positions for RoPE
                    if node_level is not None and level_offsets_dev.numel() > 0:
                        pos_in_level = n_id.to(target_device, non_blocking=True) - level_offsets_dev[node_level]
                    else:
                        pos_in_level = n_id.to(target_device, non_blocking=True)
                # ensure copies done before using them on default stream
                torch.cuda.current_stream().wait_stream(h2d_stream)
            else:
                x_sub = x_prev[n_id].to(target_device, non_blocking=True)
                edge_index = batch.edge_index.to(target_device, non_blocking=True)
                node_level = batch.node_level.to(target_device, non_blocking=True) if hasattr(batch, "node_level") else None

                if hasattr(data, "lap_pe_proj_cpu") and data.lap_pe_proj_cpu is not None:
                    pe = data.lap_pe_proj_cpu[n_id].to(target_device, non_blocking=True)
                    x_sub = x_sub + pe

                if node_level is not None and level_offsets_dev.numel() > 0:
                    pos_in_level = n_id.to(target_device, non_blocking=True) - level_offsets_dev[node_level]
                else:
                    pos_in_level = n_id.to(target_device, non_blocking=True)

            # (optional) edge_attr regen if you use it (kept off by default)
            edge_attr = None
            # if getattr(layer, "use_edge_attr", False) and hasattr(batch, "edge_type"):
            #     edge_attr = ...  # generate on the fly

            # compute
            with amp_ctx:
                if self.use_ckpt and torch.is_grad_enabled():
                    out = torch.utils.checkpoint.checkpoint(
                        _call_layer, x_sub, edge_index, node_level, pos_in_level,
                        use_reentrant=False, edge_attr=edge_attr
                    )
                else:
                    out = _call_layer(x_sub, edge_index, node_level, pos_in_level, edge_attr=edge_attr)

            # write back + overlap count
            out_dev = out.to(x_next.device, non_blocking=True)
            x_next[n_id] = out_dev
            cnts[n_id]  += 1

            # free temps
            del x_sub, edge_index, node_level, pos_in_level, out, out_dev, batch, edge_attr

        # average overlaps (or switch to keep-old semantics if you prefer)
        mask = (cnts > 0).to(x_prev.dtype)
        x_next = (x_next / cnts.clamp_min(1)) * mask + x_prev * (1 - mask)
        return x_next
    
    def run_one_layer_v2(
        self,
        *,
        layer: torch.nn.Module,
        data,                       # CPU PyG Data (edge_index, node_level, level_offsets, optional lap_pe_proj_cpu)
        x_global: torch.Tensor,     # [N,H] global state (CPU or GPU)
        seed_nodes: torch.Tensor,   # indices to sweep (CPU)
        window_size: int = 8192,
        stride: int = 4096,
        halo: int = 1024,
        include_upper: Tuple[str, ...] = ("L1","L2","L3"),
        batch_size: Optional[int] = None,  # per-Loader batch_size (usually larger than “full sweep” since each window is smaller)
        mode: str = "full",     # "windowed" or "full"
        extra_kwargs=None,
    ) -> torch.Tensor:
        """
        Slides windows over L0, and for each L0-window makes an induced exact 1-hop subgraph
        using seeds = L0_window (+ selected upper levels). Returns x_next like run_one_layer.
        """
        assert data.edge_index.device.type == "cpu", "data.edge_index must be on CPU"
        assert hasattr(data, "node_level") and data.node_level is not None, "node_level required"
        node_level_cpu: torch.Tensor = data.node_level
        l0_idx, l1_idx, l2_idx, l3_idx = _split_levels(node_level_cpu)

        # decide seeds to *always* include (upper levels are usually small)
        always_upper = []
        if "L1" in include_upper and l1_idx.numel() > 0: always_upper.append(l1_idx)
        if "L2" in include_upper and l2_idx.numel() > 0: always_upper.append(l2_idx)
        if "L3" in include_upper and l3_idx.numel() > 0: always_upper.append(l3_idx)

        # snapshot read
        x_prev = x_global if mode == "full" else x_global.detach()
        x_next = torch.empty_like(x_prev, device=x_prev.device)
        cnts   = torch.zeros((x_prev.size(0), 1), dtype=x_prev.dtype, device=x_prev.device)

        target_device = self._layer_device(layer, fallback=x_prev.device)
        bs = int(batch_size) if batch_size is not None else self.default_batch_size

        # AMP context
        use_amp = self.use_amp and torch.is_grad_enabled()
        amp_ctx = torch.autocast(device_type=target_device.type, dtype=amp_dtype(target_device.type)) if use_amp else contextlib.nullcontext()

        # level-aware RoPE positions cache
        level_offsets = getattr(data, "level_offsets", [])
        level_offsets_dev = torch.as_tensor(level_offsets, device=target_device) if level_offsets else torch.empty(0, device=target_device)

        # wrapper for checkpoint
        def _call_layer(x_sub, edge_index, node_level, positions, edge_attr=None):
            return layer(
                x_sub, edge_index, node_level,
                positions=positions, level_offsets=None, edge_attr=edge_attr
            )

        # iterate L0 windows
        for l0_window in _iter_l0_windows(l0_idx, window_size=window_size, stride=stride, halo=halo):
            # build seeds = L0 window ∪ selected uppers
            if always_upper:
                seeds = torch.cat([l0_window, *always_upper], dim=0)
            else:
                seeds = l0_window
            # NOTE: seeds are *global node ids on CPU*
            loader = NeighborLoader(
                data,
                num_neighbors=[-1],
                input_nodes=seeds,
                batch_size=bs,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                subgraph_type="induced",
                prefetch_factor=None if self.num_workers == 0 else max(2, self.prefetch_batches),
                persistent_workers=(self.num_workers > 0),
            )

            for batch in loader:
                n_id = batch.n_id  # CPU global ids that appear in this mini-batch’s induced subgraph

                # slice previous layer’s state for this mini-batch
                x_sub = x_prev[n_id].to(target_device, non_blocking=True)

                # LapPE add
                if hasattr(data, "lap_pe_proj_cpu") and data.lap_pe_proj_cpu is not None:
                    pe = data.lap_pe_proj_cpu[n_id].to(target_device, non_blocking=True)
                    x_sub = x_sub + pe

                edge_index = batch.edge_index.to(target_device, non_blocking=True)
                node_level = batch.node_level.to(target_device, non_blocking=True) if hasattr(batch, "node_level") else None

                # level-aware positions for RoPE (intra-level)
                if node_level is not None and level_offsets_dev.numel() > 0:
                    pos_in_level = n_id.to(target_device, non_blocking=True) - level_offsets_dev[node_level]
                else:
                    pos_in_level = n_id.to(target_device, non_blocking=True)

                # (optional) edge_attr regen — off by default
                edge_attr = None

                with amp_ctx:
                    if self.use_ckpt and torch.is_grad_enabled():
                        out = torch.utils.checkpoint.checkpoint(
                            _call_layer, x_sub, edge_index, node_level, pos_in_level,
                            use_reentrant=False, edge_attr=edge_attr
                        )
                    else:
                        out = _call_layer(x_sub, edge_index, node_level, pos_in_level, edge_attr=edge_attr)

                # write-back + overlap count
                out_dev = out.to(x_next.device, non_blocking=True)
                x_next[n_id] = out_dev
                cnts[n_id]  += 1

                # free temps
                del x_sub, edge_index, node_level, pos_in_level, edge_attr, out, out_dev, batch

        # average overlaps and keep old where untouched
        mask = (cnts > 0).to(x_prev.dtype)
        x_next = (x_next / cnts.clamp_min(1)) * mask + x_prev * (1 - mask)
        return x_next
    
    def _loader_sage(
        self,
        data,
        seed_nodes: torch.Tensor,      # CPU global ids of the seed set to update
        fanouts: List[int],            # e.g., [25, 10] for 2-hop sampling
        batch_size: int,
    ):
        """
        k-hop sampled neighborhood. For each mini-batch of seeds, sample `fanouts[h]` neighbors at hop h.
        Writes only to the seed rows later via `batch.batch_size`.
        """
        #assert data.edge_index.device.type == "cpu", "data.edge_index must be on CPU"
        # NOTE: For sampling, just pass num_neighbors=fanouts (no -1).
        return NeighborLoader(
            data,
            num_neighbors=fanouts,          # <-- enables GraphSAGE-style sampling
            input_nodes=seed_nodes,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # subgraph_type can stay default; sampling is controlled by num_neighbors
            prefetch_factor=None if self.num_workers == 0 else max(2, self.prefetch_batches),
            persistent_workers=(self.num_workers > 0),
        )

    # ---------- NEW: run_one_layer_sage (GraphSAGE semantics) ----------
    def run_one_layer_v3(
        self,
        *,
        layer: torch.nn.Module,
        data,                          # CPU PyG Data (edge_index, node_level, level_offsets, lap_pe_proj_cpu optional)
        x_global: torch.Tensor,        # [N,H] global state (CPU or GPU)
        seed_nodes: torch.Tensor,      # CPU global ids to UPDATE (L0 seeds, or any set you choose)
        fanouts: List[int] = [5],#[5, 10],            # e.g., [25, 10] (k hops)
        batch_size: Optional[int] = None,
        mode: str = "full",            # "full" keeps autograd across batches; "windowed" detaches x_prev, "cpu_fullgrad" keeps full autograd but forces CPU grads
        extra_kwargs=None,
    ) -> torch.Tensor:
        """
        GraphSAGE-style layer pass:
          • Builds a sampled k-hop subgraph per mini-batch (fanouts).
          • Computes on all nodes in sampled subgraph.
          • Writes back ONLY the seed rows (first `batch.batch_size` nodes in n_id) to x_global.
        """
        bs = int(batch_size) if batch_size is not None else self.default_batch_size
        loader = self._loader_sage(data, seed_nodes, fanouts, bs)
        cpu_fullgrad = (mode == "cpu_fullgrad")
        # snapshot read
        #x_prev = x_global if mode == "full" else x_global.detach()
        #x_next = torch.empty_like(x_prev, device=x_prev.device)

        # snapshot
        if mode == "full":
            x_prev = x_global                  # GPU, keeps graph
            x_next = torch.empty_like(x_prev)
        elif mode == "windowed":
            x_prev = x_global.detach()         # GPU, breaks cross-batch graph
            x_next = torch.empty_like(x_prev)
        elif cpu_fullgrad:
            # keep the big buffers on CPU; grads still flow through CPU Tensors
            x_prev = x_global.to("cpu") if x_global.device.type != "cpu" else x_global
            x_next = torch.empty_like(x_prev, device="cpu")
        else:
            raise ValueError(f"Unknown mode {mode}")

        cnts = torch.zeros((x_prev.size(0),1), dtype=x_prev.dtype, device=x_prev.device)

        # We will mark touched seeds; neighbors are context and never written back.
        touched = torch.zeros((x_prev.size(0), 1), dtype=torch.bool, device=x_prev.device)

        target_device = self._layer_device(layer, fallback=x_prev.device)

        # AMP context
        use_amp = self.use_amp and torch.is_grad_enabled()
        amp_ctx = torch.autocast(device_type=target_device.type, dtype=amp_dtype(target_device.type)) if use_amp else contextlib.nullcontext()

        # RoPE level-aware positions
        level_offsets = getattr(data, "level_offsets", [])
        level_offsets_dev = torch.as_tensor(level_offsets, device=target_device) if level_offsets else torch.empty(0, device=target_device)

        # tiny wrapper (checkpoint-friendly)
        def _call_layer(x_sub, edge_index, node_level, positions, edge_attr=None):
            return layer(
                x_sub, edge_index, node_level,
                positions=positions, level_offsets=None, edge_attr=edge_attr
            )

        h2d_stream = self._h2d_stream if (self._h2d_stream is not None and target_device.type == "cuda") else None

        for batch in loader:
            # n_id lists ALL nodes in sampled subgraph (seeds FIRST)
            n_id = batch.n_id                                # CPU global ids
            seed_bs = getattr(batch, "batch_size", None)
            if seed_bs is None:
                # Fallback heuristic if PyG version lacks .batch_size:
                # assume seeds are the nodes with node_level == 0 and in the first contiguous block
                # but better require a modern PyG. We try best-effort:
                seed_bs = int((batch.node_level == 0).sum().item()) if hasattr(batch, "node_level") else n_id.size(0)

            # --- staged H2D on a side stream (optional) ---
            if h2d_stream is not None:
                with torch.cuda.stream(h2d_stream):
                    x_sub = x_prev[n_id].to(target_device, non_blocking=True)
                    edge_index = batch.edge_index.to(target_device, non_blocking=True)
                    node_level = batch.node_level.to(target_device, non_blocking=True) if hasattr(batch, "node_level") else None

                    # LapPE add
                    if hasattr(data, "lap_pe_proj_cpu") and data.lap_pe_proj_cpu is not None:
                        pe = data.lap_pe_proj_cpu[n_id].to(target_device, non_blocking=True)
                        x_sub = x_sub + pe

                    # positions (level-aware RoPE if possible)
                    if node_level is not None and level_offsets_dev.numel() > 0:
                        pos_in_level = n_id.to(target_device, non_blocking=True) - level_offsets_dev[node_level]
                    else:
                        pos_in_level = n_id.to(target_device, non_blocking=True)
                torch.cuda.current_stream().wait_stream(h2d_stream)
            else:
                x_sub = x_prev[n_id].to(target_device, non_blocking=True)
                edge_index = batch.edge_index.to(target_device, non_blocking=True)
                node_level = batch.node_level.to(target_device, non_blocking=True) if hasattr(batch, "node_level") else None

                if hasattr(data, "lap_pe_proj_cpu") and data.lap_pe_proj_cpu is not None:
                    pe = data.lap_pe_proj_cpu[n_id].to(target_device, non_blocking=True)
                    x_sub = x_sub + pe

                if node_level is not None and level_offsets_dev.numel() > 0:
                    pos_in_level = n_id.to(target_device, non_blocking=True) - level_offsets_dev[node_level]
                else:
                    pos_in_level = n_id.to(target_device, non_blocking=True)

            # (optional) edge_attr regen (usually off)
            edge_attr = None

            with amp_ctx:
                if self.use_ckpt and torch.is_grad_enabled():
                    out = torch.utils.checkpoint.checkpoint(
                        _call_layer, x_sub, edge_index, node_level, pos_in_level,
                        use_reentrant=False, edge_attr=edge_attr
                    )
                else:
                    out = _call_layer(x_sub, edge_index, node_level, pos_in_level, edge_attr=edge_attr)

            if cpu_fullgrad:
                x_next[n_id] = out.to("cpu", non_blocking=True)
                cnts[n_id]  += 1
            else:
                out_dev = out.to(x_next.device, non_blocking=True)
                x_next[n_id] = out_dev
                cnts[n_id]  += 1
                del out_dev

            del x_sub, edge_index, node_level, pos_in_level, out, batch, edge_attr

            # average overlaps + keep-old
            mask = (cnts > 0).to(x_prev.dtype)
            x_next = (x_next / cnts.clamp_min(1)) * mask + x_prev * (1 - mask)

            # hand back on the same device as input x_global
            if cpu_fullgrad and (x_global.device.type != "cpu"):
                x_next = x_next.to(x_global.device, non_blocking=True)
            # ---------- seed-only write-back ----------
            # seeds are the FIRST `seed_bs` nodes in the sampled n_id:
            #seed_bs = batch.batch_size if hasattr(batch, "batch_size") else seed_bs
            # n_id_seed = n_id[:seed_bs]                       # CPU
            # out_seed  = out[:seed_bs].to(x_next.device, non_blocking=True)
            # x_next[n_id_seed] = out_seed
            # touched[n_id_seed] = True

            # free temps
            #del x_sub, edge_index, node_level, pos_in_level, out, out_seed, batch, edge_attr, n_id_seed

        # keep old values for non-updated rows (neighbors)
        #x_next = torch.where(touched, x_next, x_prev)
        return x_next
    
    def run_one_layer_v4(
        self,
        *,
        layer: torch.nn.Module,
        data,                          # CPU PyG Data (edge_index, node_level, level_offsets, lap_pe_proj_cpu optional)
        x_global: torch.Tensor,        # [N,H] global state (CPU or GPU)
        seed_nodes: torch.Tensor,      # CPU global ids to UPDATE (L0 seeds, or any set you choose)
        fanouts: List[int] = (-1,),     # e.g. (25,10) for 2-hop; (5,) == 1-hop SAGE
        batch_size: Optional[int] = None,
        mode: str = "full",            # "full" | "windowed" | "cpu_fullgrad"
        extra_kwargs=None,
    ) -> torch.Tensor:
        """
        GraphSAGE-style layer pass:
        • Samples a k-hop subgraph per mini-batch (fanouts).
        • Computes on ALL nodes in sampled subgraph (seeds + neighbors).
        • Writes back ONLY the seed rows (first `batch.batch_size` nodes in n_id) to x_global.
        • Accumulate/average overlaps across mini-batches at the END (not inside the loop).
        """
        import contextlib
        from torch_geometric.loader import NeighborLoader

        extra_kwargs = extra_kwargs or {}
        bs = int(batch_size) if batch_size is not None else self.default_batch_size

        # -------- loader: SAGE sampling --------
        loader = self._loader_sage(data, seed_nodes, fanouts, bs)  # your existing helper

        # -------- snapshot policy + buffers --------
        if mode == "full":
            # keep cross-batch autograd graph
            x_prev = x_global
            device_buf = x_prev.device
        elif mode == "windowed":
            # break cross-batch graph; grads still flow to params
            x_prev = x_global.detach()
            device_buf = x_prev.device
        elif mode == "cpu_fullgrad":
            # grads flow through CPU tensors (slower but tiny GPU footprint)
            x_prev = x_global.to("cpu") if x_global.device.type != "cpu" else x_global
            device_buf = torch.device("cpu")
        else:
            raise ValueError(f"Unknown mode {mode}")

        N, H = x_prev.size(0), x_prev.size(1)

        # We only UPDATE seeds; neighbors are context. Accumulate seed updates in separate buffers:
        # accum/cnts live on the SAME device as x_prev to avoid device mismatches.
        accum = torch.zeros((N, H), dtype=x_prev.dtype, device=device_buf)  # holds sum of outputs for seeds
        cnts  = torch.zeros((N, 1), dtype=x_prev.dtype, device=device_buf)  # how many times a seed was updated

        # target (compute) device for the layer
        target_device = self._layer_device(layer, fallback=device_buf)

        # AMP context
        use_amp = self.use_amp and torch.is_grad_enabled()
        amp_ctx = torch.autocast(device_type=target_device.type, dtype=amp_dtype(target_device.type)) if use_amp else contextlib.nullcontext()

        # level-aware RoPE positions (optional, safe fallback is positions=n_id)
        #level_offsets = getattr(data, "level_offsets", [])
        #level_offsets_dev = torch.as_tensor(level_offsets, device=target_device) if level_offsets else torch.empty(0, device=target_device)

        level_offsets_dev = None
        if getattr(data, "level_offsets", None) is not None:
            # data.level_offsets can be list or tensor; normalize to tensor
            level_offsets_dev = torch.as_tensor(data.level_offsets, device=target_device)

        # checkpoint wrapper
        def _call_layer(x_sub, edge_index, node_level, positions, edge_attr=None):
            return layer(
                x_sub, edge_index, node_level,
                positions=positions, level_offsets=None, edge_attr=edge_attr
            )

        h2d_stream = self._h2d_stream if (self._h2d_stream is not None and target_device.type == "cuda") else None

        for batch in loader:
            # In PyG NeighborLoader, seeds are FIRST in n_id; batch.batch_size tells how many.
            n_id = batch.n_id  # CPU global ids (seeds first)
            if hasattr(batch, "batch_size") and batch.batch_size is not None:
                seed_bs = int(batch.batch_size)
            else:
                # fallback: assume we asked for bs seeds; clip to actual size
                seed_bs = min(bs, n_id.size(0))

            

            # ------- stage H2D copies (optional stream) -------
            if h2d_stream is not None:
                with torch.cuda.stream(h2d_stream):
                    x_sub = x_prev[n_id].to(target_device, non_blocking=True)
                    edge_index = batch.edge_index.to(target_device, non_blocking=True)
                    node_level = batch.node_level.to(target_device, non_blocking=True) if hasattr(batch, "node_level") else None

                    # ---- SAFE node_level retrieval ----
                    node_level = None
                    if hasattr(batch, "node_level") and batch.node_level is not None:
                        node_level = batch.node_level.to(target_device, non_blocking=True)
                    elif getattr(data, "node_level", None) is not None:
                        # index from the full graph with n_id
                        node_level = data.node_level[n_id].to(target_device, non_blocking=True)

                    # LapPE add (if cached)
                    if getattr(data, "lap_pe_proj_cpu", None) is not None:
                        pe = data.lap_pe_proj_cpu[n_id].to(target_device, non_blocking=True)
                        x_sub = x_sub + pe

                    # ---- positions (RoPE) ----
                    # If we have both node_level and merged level_offsets, do level-aware positions; else fallback to global ids.
                    if (node_level is not None) and (level_offsets_dev is not None) and (level_offsets_dev.numel() > 0):
                        pos_in_level = n_id.to(target_device, non_blocking=True) - level_offsets_dev[node_level]
                    else:
                        pos_in_level = n_id.to(target_device, non_blocking=True)

                
                torch.cuda.current_stream().wait_stream(h2d_stream)
            else:
                x_sub = x_prev[n_id].to(target_device, non_blocking=True)
                if x_sub.dtype != next(layer.parameters()).dtype:
                    x_sub = x_sub.to(next(layer.parameters()).dtype)
                edge_index = batch.edge_index.to(target_device, non_blocking=True)
                node_level = batch.node_level.to(target_device, non_blocking=True) if hasattr(batch, "node_level") else None

                if hasattr(data, "lap_pe_proj_cpu") and data.lap_pe_proj_cpu is not None:
                    pe = data.lap_pe_proj_cpu[n_id].to(target_device, non_blocking=True)
                    x_sub = x_sub + pe

                if node_level is not None and level_offsets_dev.numel() > 0:
                    pos_in_level = n_id.to(target_device, non_blocking=True) - level_offsets_dev[node_level]
                else:
                    pos_in_level = n_id.to(target_device, non_blocking=True)

            # ------- forward on sampled subgraph -------
            edge_attr = None  # regen if you actually use it

            with amp_ctx:
                if self.use_ckpt and torch.is_grad_enabled():
                    out = torch.utils.checkpoint.checkpoint(
                        _call_layer, x_sub, edge_index, node_level, pos_in_level, use_reentrant=False
                    )
                else:
                    out = _call_layer(x_sub, edge_index, node_level, pos_in_level)

            # ------- seed-only writeback (accumulate; no averaging yet) -------
            # Seeds are the FIRST seed_bs rows in both n_id and out:
            n_id_seed = n_id[:seed_bs]                      # CPU ids
            out_seed  = out[:seed_bs]                      # [seed_bs, H] on target_device

            if device_buf.type != out_seed.device.type:
                out_seed = out_seed.to(device_buf, non_blocking=True)

            # Accumulate into seed rows only (index_add_ is overlap-safe and differentiable)
            accum.index_add_(0, n_id_seed.to(device_buf), out_seed)
            cnts.index_add_(0, n_id_seed.to(device_buf), torch.ones((seed_bs, 1), dtype=cnts.dtype, device=device_buf))

            # free per-batch temporaries
            del x_sub, edge_index, node_level, pos_in_level, out, out_seed, n_id_seed, batch

        # -------- finalize: average where updated; keep old elsewhere --------
        # NOTE: average only SEEDS we touched (cnts>0); neighbors keep x_prev.
        # Build result tensor on same device as x_prev to keep autograd/device consistent:
        x_next = x_prev  # default: keep old values for all rows
        updated_mask = (cnts > 0)
        if updated_mask.any():
            averaged = accum / cnts.clamp_min(1.0)
            # We need a *copy* before in-place masked assignment if x_prev is needed elsewhere in autograd.
            x_next = x_prev.clone()
            x_next[updated_mask.squeeze(1)] = averaged[updated_mask.squeeze(1)]

        # Hand back on the same device as input x_global (for cpu_fullgrad, move if needed)
        if mode == "cpu_fullgrad" and (x_global.device.type != "cpu"):
            x_next = x_next.to(x_global.device, non_blocking=True)

        return x_next
    
    def run_one_layer_prebatch(
        self,
        *,
        layer: torch.nn.Module,
        data,                          # CPU PyG Data (edge_index, node_level, level_offsets, lap_pe_proj_cpu optional)
        x_global: torch.Tensor,        # [N,H]
        seed_nodes: torch.Tensor,      # CPU global ids to UPDATE
        fanouts: List[int] = (-1,),
        batch_size: Optional[int] = None,
        mode: str = "full",            # "full" | "windowed" | "cpu_fullgrad"
        extra_kwargs=None,
    ) -> torch.Tensor:
        import contextlib

        extra_kwargs = extra_kwargs or {}
        bs = int(batch_size) if batch_size is not None else self.default_batch_size

        loader = self._loader_sage(data, seed_nodes, fanouts, bs)

        # snapshot policy
        if mode == "full":
            x_prev = x_global
            dev_buf = x_prev.device
        elif mode == "windowed":
            x_prev = x_global.detach()
            dev_buf = x_prev.device
        elif mode == "cpu_fullgrad":
            x_prev = x_global.to("cpu") if x_global.device.type != "cpu" else x_global
            dev_buf = torch.device("cpu")
        else:
            raise ValueError(f"Unknown mode {mode}")

        N, H = x_prev.size(0), x_prev.size(1)

        # # target compute device & dtype
        target_device = self._layer_device(layer, fallback=dev_buf)
        layer_dtype   = next(layer.parameters()).dtype if any(True for _ in layer.parameters()) else x_prev.dtype
        
        # target compute device & dtype
        #target_device = self._layer_device(layer, fallback=x_prev.device)
        #layer_dtype   = next(layer.parameters()).dtype if any(True for _ in layer.parameters()) else x_prev.dtype

        # accumulators (overwrites seeds only)
        accum = torch.zeros((N, H), dtype=layer_dtype, device=dev_buf)     # sum of seed outputs
        cnts  = torch.zeros((N, 1), dtype=torch.float32, device=dev_buf)   # counts for averaging
        

        # AMP
        use_amp = self.use_amp and torch.is_grad_enabled()
        amp_ctx = torch.autocast(device_type=target_device.type, dtype=amp_dtype(target_device.type)) if use_amp else contextlib.nullcontext()

        # level_offsets (for level-aware RoPE)
        level_offsets_dev = None
        if getattr(data, "level_offsets", None) is not None:
            level_offsets_dev = torch.as_tensor(data.level_offsets, device=target_device)

        def _call_layer(x_sub, edge_index, node_level, positions, edge_attr=None):
            return layer(
                x_sub, edge_index, node_level,
                positions=positions, level_offsets=None, edge_attr=edge_attr
            )

        h2d_stream = self._h2d_stream if (self._h2d_stream is not None and target_device.type == "cuda") else None

        for batch in loader:
            n_id = batch.n_id  # CPU global ids; seeds come first
            seed_bs = int(getattr(batch, "batch_size", bs))
            seed_bs = min(seed_bs, n_id.size(0))

            # ---- stage H2D (optional stream) ----
            if h2d_stream is not None:
                with torch.cuda.stream(h2d_stream):
                    x_sub = x_prev[n_id].to(target_device, non_blocking=True).to(layer_dtype)
                    edge_index = batch.edge_index.to(target_device, non_blocking=True)

                    # node_level with robust fallback
                    if hasattr(batch, "node_level") and batch.node_level is not None:
                        node_level = batch.node_level.to(target_device, non_blocking=True)
                    elif getattr(data, "node_level", None) is not None:
                        node_level = data.node_level[n_id].to(target_device, non_blocking=True)
                    else:
                        node_level = None

                    ## LapPE add (cast to math dtype)
                    #if getattr(data, "lap_pe_proj_cpu", None) is not None:
                    #    pe = data.lap_pe_proj_cpu[n_id].to(target_device, non_blocking=True).to(layer_dtype)
                    #    x_sub = x_sub + pe

                    lap_proj = None
                    if extra_kwargs and "lap_pe_proj" in extra_kwargs and extra_kwargs["lap_pe_proj"] is not None:
                        lap_proj = extra_kwargs["lap_pe_proj"]
                    elif hasattr(layer, "lap_pe_proj"):
                        lap_proj = layer.lap_pe_proj
                    elif hasattr(self, "lap_pe_proj"):
                        lap_proj = self.lap_pe_proj
                    #print(getattr(data, "lap_pe_raw_cpu", None))
                    pe = None
                    if getattr(data, "lap_pe_raw_cpu", None) is not None:
                        pe_in = data.lap_pe_raw_cpu[n_id].to(target_device, non_blocking=True)
                        with amp_ctx:
                            pe = lap_proj(pe_in)                           # grads flow to lap_proj
                        if pe.dtype != layer_dtype:
                            pe = pe.to(layer_dtype)
                        #print(pe,"pe matrix")
                        x_sub = x_sub + pe
                        #print("Added LapPE projection on device", target_device)
                        # don't store pe back to data; let it die this iteration

                    # positions (precomputed per-node if available)
                    if getattr(data, "pos_in_level", None) is not None:
                        pos_in_level = data.pos_in_level[n_id].to(target_device, non_blocking=True)
                        #print("Using precomputed pos_in_level")
                    else:
                        # fallback (single-sample or legacy)
                        # positions (long)
                        if (node_level is not None) and (level_offsets_dev is not None) and (level_offsets_dev.numel() > 0):
                            pos_in_level = (n_id.to(target_device, non_blocking=True) - level_offsets_dev[node_level]).long()
                        else:
                            pos_in_level = n_id.to(target_device, non_blocking=True).long()

                    # Resolve the generator in this priority order:
                    gen = None
                    if extra_kwargs and "edge_feature_generator" in extra_kwargs:
                        gen = extra_kwargs["edge_feature_generator"]
                    if gen is None:
                        # If the *layer* carries the generator (recommended)
                        gen = getattr(layer, "edge_feature_generator", None)
                    if gen is None:
                        # If you really want to put one on the executor itself
                        gen = getattr(self, "edge_feature_generator", None)

                    # Compute edge_attr only if a generator exists
                    edge_attr = None
                    if gen is not None:
                        # Make sure kwargs exist if your generator needs them
                        edge_type = batch.edge_type.to(target_device, non_blocking=True) if hasattr(batch, "edge_type") else None
                        # Match dtype/device to x_sub (don’t move the module; inputs decide the device)
                        #with amp_ctx:
                        edge_attr = gen(x_sub, edge_index, edge_type)
                        # If your layer expects a specific dtype:
                        if edge_attr.dtype != x_sub.dtype:
                            edge_attr = edge_attr.to(x_sub.dtype)
                    #print("Generated edge_attr with shape", edge_attr.shape)
                torch.cuda.current_stream().wait_stream(h2d_stream)
            else:
                x_sub = x_prev[n_id].to(target_device, non_blocking=True).to(layer_dtype)
                edge_index = batch.edge_index.to(target_device, non_blocking=True)

                if hasattr(batch, "node_level") and batch.node_level is not None:
                    node_level = batch.node_level.to(target_device, non_blocking=True)
                elif getattr(data, "node_level", None) is not None:
                    node_level = data.node_level[n_id].to(target_device, non_blocking=True)
                else:
                    node_level = None

                #if getattr(data, "lap_pe_proj_cpu", None) is not None:
                #    pe = data.lap_pe_proj_cpu[n_id].to(target_device, non_blocking=True).to(layer_dtype)
                #    x_sub = x_sub + pe

                lap_proj = None
                if extra_kwargs and "lap_pe_proj" in extra_kwargs and extra_kwargs["lap_pe_proj"] is not None:
                    lap_proj = extra_kwargs["lap_pe_proj"]
                elif hasattr(layer, "lap_pe_proj"):
                    lap_proj = layer.lap_pe_proj
                elif hasattr(self, "lap_pe_proj"):
                    lap_proj = self.lap_pe_proj
                #print(getattr(data, "lap_pe_raw_cpu", None))
                pe = None
                if getattr(data, "lap_pe_raw_cpu", None) is not None:
                    pe_in = data.lap_pe_raw_cpu[n_id].to(target_device, non_blocking=True)
                    with amp_ctx:
                        pe = lap_proj(pe_in)                           # grads flow to lap_proj
                    if pe.dtype != layer_dtype:
                        pe = pe.to(layer_dtype)
                    x_sub = x_sub + pe
                    #print("Added LapPE projection on device", target_device)
                    # don't store pe back to data; let it die this iteration

                # positions (precomputed per-node if available)
                    if getattr(data, "pos_in_level", None) is not None:
                        pos_in_level = data.pos_in_level[n_id].to(target_device, non_blocking=True)
                    else:
                        # fallback (single-sample or legacy)
                        # positions (long)
                        if (node_level is not None) and (level_offsets_dev is not None) and (level_offsets_dev.numel() > 0):
                            pos_in_level = (n_id.to(target_device, non_blocking=True) - level_offsets_dev[node_level]).long()
                        else:
                            pos_in_level = n_id.to(target_device, non_blocking=True).long()

                # Resolve the generator in this priority order:
                gen = None
                if extra_kwargs and "edge_feature_generator" in extra_kwargs:
                    gen = extra_kwargs["edge_feature_generator"]
                if gen is None:
                    # If the *layer* carries the generator (recommended)
                    gen = getattr(layer, "edge_feature_generator", None)
                if gen is None:
                    # If you really want to put one on the executor itself
                    gen = getattr(self, "edge_feature_generator", None)

                # Compute edge_attr only if a generator exists
                edge_attr = None
                if gen is not None:
                    # Make sure kwargs exist if your generator needs them
                    edge_type = batch.edge_type.to(target_device, non_blocking=True) if hasattr(batch, "edge_type") else None
                    # Match dtype/device to x_sub (don’t move the module; inputs decide the device)
                    #with amp_ctx:
                    edge_attr = gen(x_sub, edge_index, edge_type)
                    # If your layer expects a specific dtype:
                    if edge_attr.dtype != x_sub.dtype:
                        edge_attr = edge_attr.to(x_sub.dtype)
                #print("Generated edge_attr with shape", edge_attr.shape)
            
            # ---- forward ----
            with amp_ctx:
                if self.use_ckpt and torch.is_grad_enabled():
                    out = torch.utils.checkpoint.checkpoint(
                        _call_layer, x_sub, edge_index, node_level, pos_in_level, edge_attr,  use_reentrant=False
                    )
                else:
                    out = _call_layer(x_sub, edge_index, node_level, pos_in_level, edge_attr)

            # ---- seed-only writeback (accumulate) ----
            # first seed_bs rows are seeds by NeighborLoader convention
            out_seed = out[:seed_bs].to(dev_buf, non_blocking=True).to(accum.dtype)
            n_id_seed = n_id[:seed_bs].to(dev_buf, non_blocking=True)

            accum.index_add_(0, n_id_seed, out_seed)
            cnts.index_add_(0, n_id_seed, torch.ones((seed_bs, 1), dtype=cnts.dtype, device=dev_buf))

            # # ---- seed-only writeback (accumulate on compute device) ----
            # if seed_bs > 0:
            #     n_id_seed = n_id[:seed_bs].to(target_device, non_blocking=True)     # indices on target_device
            #     out_seed  = out[:seed_bs]                                           # already on target_device
            #     if out_seed.dtype != accum.dtype:
            #         out_seed = out_seed.to(accum.dtype)

            #     accum.index_add_(0, n_id_seed, out_seed)
            #     cnts.index_add_(
            #         0, n_id_seed,
            #         torch.ones((seed_bs, 1), dtype=cnts.dtype, device=target_device)
            #     )

            # free temps
            del x_sub, edge_index, node_level, pos_in_level, out, out_seed, n_id_seed, batch
            if edge_attr is not None:
                del edge_attr, gen, edge_type
            if pe is not None:
                del pe
            

        # ---- finalize: average where updated; keep old elsewhere ----
        updated = (cnts > 0).squeeze(1)
        if updated.any():
            averaged = (accum / cnts.clamp_min(1.0)).to(x_prev.dtype)
            x_next = x_prev.clone()
            x_next[updated] = averaged[updated]
        else:
            x_next = x_prev  # nothing updated, keep as-is

        # updated = (cnts > 0).squeeze(1)
        # if updated.any():
        #     averaged = (accum / cnts.clamp_min(1.0)).to(x_prev.dtype)  # still on target_device
        #     x_next = x_prev.clone()                                    # on x_prev.device (often CPU)

        #     upd_idx = updated.nonzero(as_tuple=False).view(-1)
        #     x_next[upd_idx.to(x_prev.device, non_blocking=True)] = \
        #         averaged[upd_idx].to(x_prev.device, non_blocking=True)
        # else:
        #     x_next = x_prev


        if mode == "cpu_fullgrad" and (x_global.device.type != "cpu"):
            x_next = x_next.to(x_global.device, non_blocking=True)

        return x_next
    
    def run_one_layer(
        self,
        *,
        layer: torch.nn.Module,
        data,                          # CPU PyG Data (edge_index, node_level, level_offsets, lap_pe_raw_cpu optional)
        x_global: torch.Tensor,        # [N,H] or [B,N,H]
        seed_nodes: torch.Tensor,      # CPU global ids to UPDATE
        fanouts: List[int] = (-1,),
        batch_size: Optional[int] = None,
        mode: str = "full",            # "full" | "windowed" | "cpu_fullgrad"
        extra_kwargs=None,
    ) -> torch.Tensor:
        import contextlib

        extra_kwargs = extra_kwargs or {}
        bs = int(batch_size) if batch_size is not None else self.default_batch_size

        # --- Normalize to always have [B, N, H] ---
        single_batch = (x_global.dim() == 2)
        if single_batch:
            x_global = x_global.unsqueeze(0)   # [1, N, H]

        B, N, H = x_global.shape

        loader = self._loader_sage(data, seed_nodes, fanouts, bs)

        # snapshot policy
        if mode == "full":
            x_prev = x_global
            dev_buf = x_prev.device
        elif mode == "windowed":
            x_prev = x_global.detach()
            dev_buf = x_prev.device
        elif mode == "cpu_fullgrad":
            x_prev = x_global.to("cpu") if x_global.device.type != "cpu" else x_global
            dev_buf = torch.device("cpu")
        else:
            raise ValueError(f"Unknown mode {mode}")

        # target compute device & dtype
        #target_device = self._layer_device(layer, fallback=dev_buf)
        #layer_dtype   = next(layer.parameters()).dtype if any(True for _ in layer.parameters()) else x_prev.dtype

        # --- accumulators (overwrites seeds only), keep them FP32 for stability ---
        #accum = torch.zeros((B, N, H), dtype=torch.float32, device=dev_buf)
        #cnts  = torch.zeros((B, N, 1), dtype=torch.float32, device=dev_buf)

        # target compute device & dtype
        target_device = self._layer_device(layer, fallback=x_prev.device)
        layer_dtype   = next(layer.parameters()).dtype if any(True for _ in layer.parameters()) else x_prev.dtype

        # --- accumulators (overwrites seeds only), keep them FP32 for stability ---
        accum = torch.zeros((B, N, H), dtype=torch.float32, device=target_device)
        cnts  = torch.zeros((B, N, 1), dtype=torch.float32, device=target_device)

        # AMP
        use_amp = self.use_amp and torch.is_grad_enabled()
        amp_ctx = torch.autocast(device_type=target_device.type, dtype=amp_dtype(target_device.type)) if use_amp else contextlib.nullcontext()

        # level_offsets (for level-aware RoPE)
        level_offsets_dev = None
        if getattr(data, "level_offsets", None) is not None:
            level_offsets_dev = torch.as_tensor(data.level_offsets, device=target_device)

        def _call_layer(x_sub, edge_index, node_level, positions, edge_attr=None):
            return layer(
                x_sub, edge_index, node_level,
                positions=positions, level_offsets=None, edge_attr=edge_attr
            )

        # We drop the h2d stream branching for simplicity & correctness in batch mode
        for batch in loader:
            n_id = batch.n_id  # CPU global ids; seeds come first
            seed_bs = int(getattr(batch, "batch_size", bs))
            seed_bs = min(seed_bs, n_id.size(0))

            # ---- graph bits to device (shared across batch items) ----
            edge_index = batch.edge_index.to(target_device, non_blocking=True)

            # node_level with robust fallback
            if hasattr(batch, "node_level") and batch.node_level is not None:
                node_level = batch.node_level.to(target_device, non_blocking=True)
            elif getattr(data, "node_level", None) is not None:
                node_level = data.node_level[n_id].to(target_device, non_blocking=True)
            else:
                node_level = None

            # positions (precomputed per-node if available)
            if getattr(data, "pos_in_level", None) is not None:
                pos_in_level = data.pos_in_level[n_id].to(target_device, non_blocking=True)
            else:
                # fallback: compute from node_level + level_offsets
                if (node_level is not None) and (level_offsets_dev is not None) and (level_offsets_dev.numel() > 0):
                    pos_in_level = (n_id.to(target_device, non_blocking=True) - level_offsets_dev[node_level]).long()
                else:
                    pos_in_level = n_id.to(target_device, non_blocking=True).long()

            # ---- LapPE projection: compute once per subgraph, reuse for all B ----
            pe = None
            lap_proj = None
            if getattr(data, "lap_pe_raw_cpu", None) is not None:
                if extra_kwargs and "lap_pe_proj" in extra_kwargs and extra_kwargs["lap_pe_proj"] is not None:
                    lap_proj = extra_kwargs["lap_pe_proj"]
                elif hasattr(layer, "lap_pe_proj"):
                    lap_proj = layer.lap_pe_proj
                elif hasattr(self, "lap_pe_proj"):
                    lap_proj = self.lap_pe_proj

                if lap_proj is not None:
                    pe_in = data.lap_pe_raw_cpu[n_id].to(target_device, non_blocking=True)  # [n_sub, k]
                    with amp_ctx:
                        pe = lap_proj(pe_in)                                              # [n_sub, H]
                    if pe.dtype != layer_dtype:
                        pe = pe.to(layer_dtype)

            # ---- Edge feature generator: resolve once; we’ll call per batch item ----
            edge_gen = None
            if extra_kwargs and "edge_feature_generator" in extra_kwargs:
                edge_gen = extra_kwargs["edge_feature_generator"]
            if edge_gen is None:
                edge_gen = getattr(layer, "edge_feature_generator", None)
            if edge_gen is None:
                edge_gen = getattr(self, "edge_feature_generator", None)

            # ---- forward per batch item (simple, no block-diagonal tricks) ----
            for b in range(B):
                # slice features for this subgraph: [n_sub, H]
                x_sub = x_prev[b, n_id, :].to(target_device, non_blocking=True).to(layer_dtype)
                if pe is not None:
                    x_sub = x_sub + pe

                # edge_attr per batch item (if any)
                edge_attr = None
                memory_edge_attr = None
                if (
                    extra_kwargs
                    and "memory_cross_edge_logit_gate" in extra_kwargs
                    and hasattr(batch, "edge_attr")
                    and batch.edge_attr is not None
                ):
                    flags = batch.edge_attr.to(target_device, non_blocking=True).view(-1).to(dtype=x_sub.dtype)
                    gate = extra_kwargs["memory_cross_edge_logit_gate"].to(device=target_device, dtype=x_sub.dtype)
                    bias = float(extra_kwargs.get("memory_cross_edge_logit_bias", -20.0))
                    memory_edge_attr = flags * (gate + bias)
                if memory_edge_attr is not None:
                    edge_attr = memory_edge_attr
                elif edge_gen is not None:
                    edge_type = batch.edge_type.to(target_device, non_blocking=True) if hasattr(batch, "edge_type") else None
                    edge_attr = edge_gen(x_sub, edge_index, edge_type)
                    if edge_attr.dtype != x_sub.dtype:
                        edge_attr = edge_attr.to(x_sub.dtype)

                # ---- forward ----
                with amp_ctx:
                    if self.use_ckpt and torch.is_grad_enabled():
                        out = torch.utils.checkpoint.checkpoint(
                            _call_layer, x_sub, edge_index, node_level, pos_in_level, edge_attr, use_reentrant=False
                        )
                    else:
                        out = _call_layer(x_sub, edge_index, node_level, pos_in_level, edge_attr)
                if isinstance(out, tuple):
                    out = out[0]

                # ---- seed-only writeback (accumulate) ----
                #out_seed = out[:seed_bs].to(dev_buf, non_blocking=True).to(torch.float32)
                #n_id_seed = n_id[:seed_bs].to(dev_buf, non_blocking=True)

                #accum[b].index_add_(0, n_id_seed, out_seed)
                #cnts[b].index_add_(0, n_id_seed, torch.ones((seed_bs, 1), dtype=torch.float32, device=dev_buf))

                # # ---- seed-only writeback (accumulate on compute device) ----
                if seed_bs > 0:
                    n_id_seed = n_id[:seed_bs].to(target_device, non_blocking=True)     # indices on target_device
                    out_seed  = out[:seed_bs]                                           # already on target_device
                    if out_seed.dtype != accum.dtype:
                        out_seed = out_seed.to(accum.dtype)

                    accum[b].index_add_(0, n_id_seed, out_seed)
                    cnts[b].index_add_(
                        0, n_id_seed,
                        torch.ones((seed_bs, 1), dtype=cnts.dtype, device=target_device)
                    )

                # free per-batch-item temps
                del x_sub, out, out_seed, n_id_seed
                if edge_attr is not None:
                    del edge_attr

            # free per-subgraph temps
            del batch, edge_index, node_level, pos_in_level
            if pe is not None:
                del pe, pe_in
            if lap_proj is not None:
                del lap_proj
            if edge_gen is not None:
                del edge_gen

        # ---- finalize: average where updated; keep old elsewhere ----
        #updated = (cnts > 0).squeeze(2)          # [B, N]
        #x_next = x_prev.clone()

        #avg = (accum / cnts.clamp_min(1.0)).to(x_prev.dtype)  # [B, N, H]
        #for b in range(B):
        #    if updated[b].any():
        #        x_next[b, updated[b]] = avg[b, updated[b]]

        

        
        updated = (cnts > 0).squeeze(2).to(x_prev.device, non_blocking=True)
        
        averaged = (accum / cnts.clamp_min(1.0)).to(x_prev.dtype).to(x_prev.device, non_blocking=True)  # still on target_device
        x_next = x_prev.clone()                                    # on x_prev.device (often CPU)
        #print(updated.shape, averaged.shape, x_next.shape)
        #for b in range(B):
        #    if updated[b].any():
        #        upd_idx = updated[b].nonzero(as_tuple=False).reshape(-1)
        #        x_next[b, upd_idx.to(x_prev.device, non_blocking=True)] = \
        #            averaged[b, upd_idx].to(x_prev.device, non_blocking=True)
        # updated: [B, N], bool
        mask = updated.unsqueeze(-1).expand_as(x_next)  # [B, N, H]
        x_next = torch.where(mask, averaged, x_prev)
       

        if mode == "cpu_fullgrad" and (x_global.device.type != "cpu"):
            x_next = x_next.to(x_global.device, non_blocking=True)

        # drop back to [N,H] if we started single-batch
        if single_batch:
            x_next = x_next.squeeze(0)

        return x_next
