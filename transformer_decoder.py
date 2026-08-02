"""
Transformer Decoder Module for DETR-style Detection

This module implements a standard Transformer Decoder with:
- Self-attention layers (query-to-query)
- Cross-attention layers (query-to-image features)
- Feed-forward networks (FFN)
- Layer normalization
- Residual connections
"""

import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder for DETR-style object detection.
    
    Architecture:
    - Multiple decoder layers
    - Each layer: Self-Attention -> LayerNorm -> Cross-Attention -> LayerNorm -> FFN -> LayerNorm
    - Residual connections around each sub-layer
    """
    
    def __init__(self, embed_dim=256, num_heads=8, num_layers=6,
                 dim_feedforward=1024, dropout=0.0):
        """
        Args:
            embed_dim: Embedding dimension (default: 256)
            num_heads: Number of attention heads (default: 8)
            num_layers: Number of decoder layers (default: 6)
            dim_feedforward: FFN hidden dimension (default: 1024)
            dropout: Dropout rate (default: 0.0)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Self-attention layers (query-to-query)
        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                batch_first=True, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Cross-attention layers (query-to-image features)
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                batch_first=True, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, embed_dim)
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization (3 per layer: after self-attn, cross-attn, and FFN)
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers * 3)
        ])
    
    def forward(self, query, query_pos, image_features_key, image_features_value):
        """
        Forward pass through the decoder.
        
        Args:
            query: Query embeddings [batch_size, num_queries, embed_dim]
            query_pos: Query positional embeddings [batch_size, num_queries, embed_dim]
            image_features_key: Image features for attention keys [batch_size, seq_len, embed_dim]
            image_features_value: Image features for attention values [batch_size, seq_len, embed_dim]
        
        Returns:
            output: Decoded query features [batch_size, num_queries, embed_dim]
            intermediate: List of intermediate outputs from each layer (for auxiliary loss)
        """
        current_query = query
        intermediate_outputs = []  # Collect intermediate outputs for auxiliary loss
        
        for i in range(self.num_layers):
            # Self-attention: query-to-query
            q_in = current_query + query_pos
            # need_weights=False enables the fused attention path (see encoder)
            q_self_out, _ = self.self_attns[i](
                query=q_in,
                key=q_in,
                value=current_query,
                need_weights=False
            )
            # Residual connection and normalization
            q_self_out = self.norms[i * 3](current_query + q_self_out)
            
            # Cross-attention: query-to-image features
            q_cross_in = q_self_out + query_pos
            q_cross_out, _ = self.cross_attns[i](
                query=q_cross_in,
                key=image_features_key,
                value=image_features_value,
                need_weights=False
            )
            # Residual connection and normalization
            q_cross_out = self.norms[i * 3 + 1](q_self_out + q_cross_out)
            
            # Feed-forward network
            ffn_out = self.ffns[i](q_cross_out)
            # Residual connection and normalization
            current_query = self.norms[i * 3 + 2](q_cross_out + ffn_out)
            
            # Save intermediate output for auxiliary loss
            intermediate_outputs.append(current_query)
        
        return current_query, intermediate_outputs

