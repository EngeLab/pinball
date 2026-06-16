# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
import torch
from torch_geometric.data import Data
import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

class EdgeFeatureGenerator(nn.Module):
    """
    Generates edge features based on edge type and connected nodes.
    """
    def __init__(self, hidden_dim, num_edge_types=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        
        # Edge type embeddings
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_dim // 4)
        
        # Projection for node feature combination
        self.node_combination_proj = nn.Linear(hidden_dim * 2, hidden_dim // 2)
        
        # Final edge feature projection
        self.edge_projection = nn.Linear(hidden_dim // 4 + hidden_dim // 2, hidden_dim)
        
        # Layer norm for edge features
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, node_features, edge_index, edge_type):
        """
        Generate edge features based on edge type and connected nodes.
        
        Args:
            node_features: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge type for each edge [num_edges]
            
        Returns:
            edge_attr: Edge features [num_edges, hidden_dim]
        """
        # Get source and target node indices
        src_idx, dst_idx = edge_index
        
        # Get source and target node features
        src_features = node_features[src_idx]  # [num_edges, hidden_dim]
        dst_features = node_features[dst_idx]  # [num_edges, hidden_dim]
        
        # Combine node features (concatenate and project)
        node_combined = torch.cat([src_features, dst_features], dim=-1)  # [num_edges, hidden_dim*2]
        node_component = self.node_combination_proj(node_combined)  # [num_edges, hidden_dim//2]
        
        # Get edge type embeddings
        edge_type_component = self.edge_type_embedding(edge_type)  # [num_edges, hidden_dim//4]
        
        # Combine node and edge type components
        edge_features = torch.cat([node_component, edge_type_component], dim=-1)  # [num_edges, hidden_dim//2 + hidden_dim//4]
        edge_features = self.edge_projection(edge_features)  # [num_edges, hidden_dim]
        
        # Apply layer normalization
        edge_features = self.layer_norm(edge_features)
        
        return edge_features

class UnifiedHierarchyBuilder:
    """
    Builder for creating unified hierarchical graph structures.
    
    This class handles the construction of a unified graph representation
    where all hierarchy levels are connected in a single graph structure,
    allowing direct message passing between levels.
    """
    
    def __init__(
        self,
        compression_ratios=[128, 16, 8],
        overlap_ratios=[0.5, 0.5, 0.5],
        add_self_loops=True,
        add_long_range_edges=True,
        long_range_distance=3,
    ):
        """
        Initialize the Unified Hierarchy Builder.
        
        Args:
            compression_ratios: Compression ratios between hierarchy levels
            overlap_ratios: Overlap ratios between adjacent summaries
            add_self_loops: Whether to add self-loops to nodes
            add_long_range_edges: Whether to add edges between non-adjacent nodes
            long_range_distance: Maximum distance for long-range connections
        """
        self.compression_ratios = compression_ratios
        self.overlap_ratios = overlap_ratios
        self.add_self_loops = add_self_loops
        self.add_long_range_edges = add_long_range_edges
        self.long_range_distance = long_range_distance
    
    def build(self, token_features, token_edge_index):
        """
        Build a unified hierarchical graph from token features.
        
        Args:
            token_features: Token node features [num_tokens, hidden_dim]
            token_edge_index: Token edge indices [2, num_edges]
            
        Returns:
            unified_graph: PyG Data object with unified hierarchical graph
        """
        device = token_features.device
        num_tokens = token_features.size(0)
        hidden_dim = token_features.size(1)
        
        # Create a PyG Data object
        unified_graph = Data()
        
        # Initialize lists for collecting all nodes and edges
        all_node_features = [token_features]  # Start with L0 (token) features
        all_node_levels = [torch.zeros(num_tokens, dtype=torch.long, device=device)]  # L0 level indicator
        
        # Initialize within-level and cross-level edge lists
        within_level_edges = [token_edge_index]
        cross_level_edges = []
        
        # Initialize edge types
        within_level_edge_types = [torch.zeros(token_edge_index.size(1), dtype=torch.long, device=device)]
        cross_level_edge_types = []
        
        # Track node offsets for creating edge indices
        node_offsets = [0, num_tokens]
        
        # Create each level of the hierarchy
        current_features = token_features
        current_num_nodes = num_tokens
        
        logger.info(f"Building unified hierarchical graph with {len(self.compression_ratios)} levels")
        
        for level_idx, (compression_ratio, overlap_ratio) in enumerate(zip(self.compression_ratios, self.overlap_ratios)):
            level_name = f"L{level_idx + 1}"
            logger.info(f"Creating {level_name} with compression ratio {compression_ratio}:1 and overlap {overlap_ratio}")
            
            # Create next level
            (
                next_level_features,
                next_level_edge_index,
                cross_edges_up,
                cross_edges_down,
                lower_to_higher,
                higher_to_lower,
            ) = self._create_level(
                current_features,
                compression_ratio,
                overlap_ratio,
                node_offset=node_offsets[-1],
                current_level=level_idx,
            )
            
            next_num_nodes = next_level_features.size(0)
            logger.info(f"Created {level_name} with {next_num_nodes} nodes")
            
            # Add next level features and level indicators
            all_node_features.append(next_level_features)
            all_node_levels.append(torch.full((next_num_nodes,), level_idx + 1, dtype=torch.long, device=device))
            
            # Add within-level edges
            within_level_edges.append(next_level_edge_index)
            # Within-level edge type is the level index
            within_level_edge_types.append(
                torch.full((next_level_edge_index.size(1),), level_idx + 1, dtype=torch.long, device=device)
            )
            
            # Add cross-level edges and their types
            if cross_edges_up.size(1) > 0:
                cross_level_edges.append(cross_edges_up)
                # Cross-level up edge type: 2*level_idx + 4 (starting at 4 to distinguish from within-level)
                cross_level_edge_types.append(
                    torch.full((cross_edges_up.size(1),), 2*level_idx + 4, dtype=torch.long, device=device)
                )
            
            if cross_edges_down.size(1) > 0:
                cross_level_edges.append(cross_edges_down)
                # Cross-level down edge type: 2*level_idx + 5
                cross_level_edge_types.append(
                    torch.full((cross_edges_down.size(1),), 2*level_idx + 5, dtype=torch.long, device=device)
                )
            
            # Store mappings for later use
            unified_graph[f"l{level_idx}_to_l{level_idx+1}"] = lower_to_higher
            unified_graph[f"l{level_idx+1}_to_l{level_idx}"] = higher_to_lower
            
            # Update for next iteration
            current_features = next_level_features
            current_num_nodes = next_num_nodes
            node_offsets.append(node_offsets[-1] + current_num_nodes)
        
        # Combine all features
        unified_graph.x = torch.cat(all_node_features, dim=0)
        unified_graph.node_level = torch.cat(all_node_levels, dim=0)
        
        # Store level offsets for easy access to specific levels
        unified_graph.level_offsets = node_offsets
        
        # Combine all edges and edge types
        all_edges = []
        all_edge_types = []
        
        # Add within-level edges
        for edges, edge_types in zip(within_level_edges, within_level_edge_types):
            all_edges.append(edges)
            all_edge_types.append(edge_types)
        
        # Add cross-level edges
        for edges, edge_types in zip(cross_level_edges, cross_level_edge_types):
            all_edges.append(edges)
            all_edge_types.append(edge_types)
        
        # Combine all edges and edge types
        if all_edges:
            unified_graph.edge_index = torch.cat(all_edges, dim=1)
            unified_graph.edge_type = torch.cat(all_edge_types, dim=0)
        else:
            # Handle empty graph case
            unified_graph.edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            unified_graph.edge_type = torch.zeros((0,), dtype=torch.long, device=device)
        
        logger.info(f"Built unified graph with {unified_graph.x.size(0)} nodes and {unified_graph.edge_index.size(1)} edges")
        
        # Generate edge features based on node features and edge types
        if not hasattr(self, 'edge_feature_generator'):
            self.edge_feature_generator = EdgeFeatureGenerator(token_features.size(1)).to(token_features.device)
        
        # Generate edge features
        if unified_graph.edge_index.size(1) > 0:  # Only if we have edges
            unified_graph.edge_attr = self.edge_feature_generator(
                unified_graph.x, 
                unified_graph.edge_index, 
                unified_graph.edge_type
            )
        else:
            unified_graph.edge_attr = torch.zeros((0, token_features.size(1)), 
                                                dtype=torch.float, 
                                                device=token_features.device)
        
        return unified_graph
    
    def _create_level(
        self, 
        node_features, 
        compression_ratio, 
        overlap_ratio, 
        node_offset=0,
        current_level=0,
    ):
        """
        Create a single level in the hierarchy with connections to lower level.
        
        Args:
            node_features: Node features at current level [num_nodes, hidden_dim]
            compression_ratio: Compression ratio from current to next level
            overlap_ratio: Overlap ratio between adjacent summaries
            node_offset: Offset for node indices in the unified graph
            current_level: Current level index (0 for token level)
            
        Returns:
            new_features: Features for the new level [num_new_nodes, hidden_dim]
            new_edge_index: Edge indices for the new level [2, num_new_edges]
            cross_edges_up: Edge indices for lower->higher connections [2, num_cross_edges]
            cross_edges_down: Edge indices for higher->lower connections [2, num_cross_edges]
            lower_to_higher: Mapping from lower level nodes to higher level nodes
            higher_to_lower: Mapping from higher level nodes to lower level nodes
        """
        num_nodes = node_features.size(0)
        device = node_features.device
        
        # Calculate stride based on compression and overlap
        stride = max(1, int(compression_ratio * (1 - overlap_ratio)))
        
        # Calculate number of nodes in the new level
        num_new_nodes = max(1, (num_nodes - 1) // stride + 1)
        
        # Create new features through mean pooling
        new_features = torch.zeros_like(node_features[:num_new_nodes])
        
        # Initialize cross-level connections
        cross_edges_up_list = []    # Lower to higher
        cross_edges_down_list = []  # Higher to lower
        
        # Initialize mappings
        lower_to_higher = {}  # Maps from lower level nodes to higher level nodes
        higher_to_lower = {}  # Maps from higher level nodes to lower level nodes
        
        # Create summary nodes with overlap
        for i in range(num_new_nodes):
            # Define range for this summary node
            start_idx = min(i * stride, num_nodes - 1)
            end_idx = min(start_idx + compression_ratio, num_nodes)
            
            # Create summary through mean pooling
            if start_idx < end_idx:
                new_features[i] = torch.mean(node_features[start_idx:end_idx], dim=0)
            else:
                new_features[i] = node_features[start_idx]  # Handle edge case
            
            # Store mappings
            higher_to_lower[i] = list(range(start_idx, end_idx))
            
            # Add cross-level connections
            for j in range(start_idx, end_idx):
                # Lower to higher (e.g., L0->L1)
                cross_edges_up_list.append((j, i + node_offset))
                
                # Higher to lower (e.g., L1->L0)
                cross_edges_down_list.append((i + node_offset, j))
                
                # Update lower to higher mapping
                if j not in lower_to_higher:
                    lower_to_higher[j] = [i]
                else:
                    lower_to_higher[j].append(i)
        
        # Create within-level edges
        edges = []
        
        if num_new_nodes <= 1:
            # Self-loop for single node
            if self.add_self_loops:
                edges = [[0, 0]]
        else:
            # Create sequential connections (bidirectional)
            for i in range(num_new_nodes - 1):
                edges.append([i, i+1])
                edges.append([i+1, i])
            
            # Add self-loops
            if self.add_self_loops:
                for i in range(num_new_nodes):
                    edges.append([i, i])
            
            # Add long-range connections
            if self.add_long_range_edges:
                for i in range(num_new_nodes):
                    for j in range(i + 2, min(i + self.long_range_distance, num_new_nodes)):
                        edges.append([i, j])
                        edges.append([j, i])
        
        # Convert edges to tensor with offset
        new_edge_index = torch.tensor(edges, device=device).t() if edges else torch.zeros((2, 0), dtype=torch.long, device=device)
        if node_offset > 0 and new_edge_index.numel() > 0:
            new_edge_index += node_offset
        
        # Convert cross-level edges to tensors
        cross_edges_up = torch.tensor(cross_edges_up_list, device=device).t() if cross_edges_up_list else torch.zeros((2, 0), dtype=torch.long, device=device)
        cross_edges_down = torch.tensor(cross_edges_down_list, device=device).t() if cross_edges_down_list else torch.zeros((2, 0), dtype=torch.long, device=device)
        
        return new_features, new_edge_index, cross_edges_up, cross_edges_down, lower_to_higher, higher_to_lower