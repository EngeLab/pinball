# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
import torch
from torch_geometric.utils import add_self_loops, remove_self_loops
import logging

logger = logging.getLogger(__name__)


def is_pyg_available():
    """Check if PyTorch Geometric is available."""
    try:
        import torch_geometric
        return True
    except ImportError:
        return False


def check_pyg_compatibility():
    """
    Check if PyTorch Geometric is installed and compatible.
    
    Returns:
        bool: True if PyG is available and compatible, False otherwise
    """
    if not is_pyg_available():
        logger.warning("torch_geometric not found. Install with: pip install torch-geometric")
        return False
        




def make_pyg_compatible_for_device(device):
    """
    Configure PyG to work with the specified device (especially MPS).
    
    Args:
        device: PyTorch device
    """
    if str(device) == "mps":
        # Set environment variable for MPS fallback
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        logger.info("Set MPS fallback for PyTorch Geometric")


def add_level_offsets_to_unified_graph(graph):
    """
    Calculate and add level offsets to a unified graph.
    
    Args:
        graph: PyG Data object
        
    Returns:
        graph: Updated graph with level_offsets
    """
    if not hasattr(graph, 'node_level'):
        logger.warning("Graph does not have node_level attribute")
        return graph
    
    # Get unique levels and their counts
    levels, counts = torch.unique(graph.node_level, return_counts=True)
    
    # Calculate offsets
    offsets = [0]
    cumulative = 0
    for count in counts:
        cumulative += count
        offsets.append(cumulative)
    
    graph.level_offsets = offsets
    return graph


def extract_level_features(graph, level_idx):
    """
    Extract features for a specific level from a unified graph.
    
    Args:
        graph: PyG Data object with unified graph
        level_idx: Level index to extract
        
    Returns:
        features: Node features for the specified level
    """
    if not hasattr(graph, 'node_level'):
        logger.warning("Graph does not have node_level attribute")
        return None
    
    # Get mask for the specific level
    level_mask = (graph.node_level == level_idx)
    
    # Extract features
    return graph.x[level_mask]


def extract_level_subgraph(graph, level_idx):
    """
    Extract a subgraph for a specific level from a unified graph.
    
    Args:
        graph: PyG Data object with unified graph
        level_idx: Level index to extract
        
    Returns:
        subgraph: PyG Data object for the specified level
    """
    from torch_geometric.utils import subgraph
    
    if not hasattr(graph, 'node_level'):
        logger.warning("Graph does not have node_level attribute")
        return None
    
    # Get indices for the specific level
    level_mask = (graph.node_level == level_idx)
    level_indices = torch.where(level_mask)[0]
    
    # Extract subgraph
    edge_index, edge_attr = subgraph(
        level_indices, 
        graph.edge_index, 
        edge_attr=graph.edge_type if hasattr(graph, 'edge_type') else None,
        relabel_nodes=True
    )
    
    # Create new subgraph
    import torch_geometric.data as data
    subgraph = data.Data(
        x=graph.x[level_mask],
        edge_index=edge_index,
        edge_type=edge_attr
    )
    
    return subgraph
