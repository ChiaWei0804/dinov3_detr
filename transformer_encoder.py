"""
Transformer Encoder Module for DETR-style Detection

This module implements a standard Transformer Encoder with:
- Self-attention layers
- Feed-forward networks (FFN)
- Layer normalization
- Residual connections
"""

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for DETR-style object detection.
    
    Architecture:
    - Multiple encoder layers with self-attention
    - Each layer: Self-Attention -> LayerNorm -> FFN -> LayerNorm
    - Residual connections around each sub-layer
    """
    
    def __init__(self, embed_dim=256, num_heads=8, num_layers=2, 
                 dim_feedforward=1024, dropout=0.1):
        """
        Args:
            embed_dim: Embedding dimension (default: 256)
            num_heads: Number of attention heads (default: 8)
            num_layers: Number of encoder layers (default: 2)
            dim_feedforward: FFN hidden dimension (default: 1024)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Self-attention layers
        self.self_attns = nn.ModuleList([
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
        
        # Layer normalization (2 per layer: after attention and after FFN)
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers * 2)
        ])
    
    def forward(self, src, pos_embedding=None):
        """
        Forward pass through the encoder.
        
        Args:
            src: Input features [batch_size, seq_len, embed_dim]
            pos_embedding: Positional embeddings [batch_size, seq_len, embed_dim]
        
        Returns:
            output: Encoded features [batch_size, seq_len, embed_dim]
        """
        # If no positional embedding provided, use zero
        if pos_embedding is None:
            pos_embedding = torch.zeros_like(src)
        
        current_feat = src
        
        for i in range(self.num_layers):
            # Self-attention with positional encoding
            feat_with_pos = current_feat + pos_embedding
            # need_weights=False lets nn.MultiheadAttention dispatch to
            # scaled_dot_product_attention instead of materialising the full
            # [B*heads, L, L] attention matrix. At 800px that matrix is
            # 8*8*2500*2500 entries per layer - several GB of activations that
            # were only being discarded. Output is mathematically identical.
            attn_out, _ = self.self_attns[i](
                query=feat_with_pos,
                key=feat_with_pos,
                value=current_feat,
                need_weights=False
            )
            
            # Residual connection and normalization
            current_feat = current_feat + attn_out
            current_feat = self.norms[i * 2](current_feat)
            
            # Feed-forward network
            ffn_out = self.ffns[i](current_feat)
            
            # Residual connection and normalization
            current_feat = current_feat + ffn_out
            current_feat = self.norms[i * 2 + 1](current_feat)
        
        return current_feat

