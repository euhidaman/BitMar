"""
BitMar Model Architecture
BitNet-quantized Vision-Language Episodic Memory Transformer
Combines 1.58-bit quantization, DiNOv2 vision, Larimar episodic memory, and QFormer Quadrangle Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer
import math
import logging

# Import our new Quadrangle Attention implementation
from src.quadrangle_attention import QuadrangleAttention, QuadrangleTransformerBlock, EpisodicQuadrangleProcessor

logger = logging.getLogger(__name__)


class BitNetLinear(nn.Module):
    """1.58-bit Linear layer following BitNet b1.58 architecture"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters with proper initialization to prevent NaN
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Initialize weights with Xavier uniform for stability
        with torch.no_grad():
            nn.init.xavier_uniform_(self.weight, gain=0.1)  # Small gain to prevent saturation
            self.weight.clamp_(-1.0, 1.0)  # Clamp to prevent extreme values
            if self.bias is not None:
                self.bias.zero_()

        # Quantization scaling factors with safe initialization
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('input_scale', torch.ones(1))

    def quantize_weights_1_58_bit(self, weight: torch.Tensor) -> torch.Tensor:
        """BitNet b1.58 weight quantization: {-1, 0, +1} with numerical stability"""
        # Clamp input weights to prevent extreme values
        weight_clamped = torch.clamp(weight, min=-5.0, max=5.0)
        
        # Compute scaling factor with numerical stability
        scale = weight_clamped.abs().mean()
        scale = scale.clamp(min=1e-8, max=10.0)  # Prevent extreme scales
        self.weight_scale.data = scale

        # Normalize weights with gradient clipping
        weight_norm = weight_clamped / scale
        weight_norm = torch.clamp(weight_norm, min=-3.0, max=3.0)

        # 1.58-bit quantization with threshold
        threshold = 2.0 / 3.0  # Optimal threshold for ternary quantization

        # Create ternary weights
        quantized = torch.zeros_like(weight_norm)
        quantized[weight_norm > threshold] = 1.0
        quantized[weight_norm < -threshold] = -1.0
        # Values between -threshold and threshold remain 0

        return quantized

    def quantize_activations_8bit(self, x: torch.Tensor) -> torch.Tensor:
        """8-bit activation quantization with enhanced numerical stability"""
        # Clamp extreme values to prevent overflow
        x_clamped = torch.clamp(x, min=-100.0, max=100.0)
        
        # Simplified finite check that's torch.compile friendly
        if torch.any(torch.isnan(x_clamped)):
            logger.warning("NaN values in activation quantization, using fallback")
            return torch.clamp(x, min=-1.0, max=1.0)

        # Compute quantization parameters with stability checks
        x_min, x_max = x_clamped.min(), x_clamped.max()

        # Prevent division by zero and handle edge cases
        range_val = x_max - x_min
        if range_val < 1e-6:
            return x_clamped

        scale = range_val / 255.0
        scale = scale.clamp(min=1e-6, max=100.0)
        self.input_scale.data = scale

        # Quantize to 8-bit with bounds checking
        zero_point = (-x_min / scale).round().clamp(0, 255)
        quantized = ((x_clamped / scale) + zero_point).round().clamp(0, 255)

        # Dequantize with final stability check
        dequantized = scale * (quantized - zero_point)
        dequantized = torch.clamp(dequantized, min=-100.0, max=100.0)
        
        return dequantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified finite check that's torch.compile friendly
        # Only check during training and use simpler logic
        if self.training and torch.any(torch.isnan(x)):
            logger.warning("NaN detected in BitNetLinear input, clamping...")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            x = torch.clamp(x, min=-10.0, max=10.0)
        
        if self.training:
            # Full precision training with straight-through estimator
            # Forward pass with quantized weights but gradients flow through original weights
            weight_q = self.quantize_weights_1_58_bit(self.weight)
            weight_forward = weight_q * self.weight_scale

            # Use original weight for gradient computation (straight-through estimator)
            weight_forward = weight_forward + \
                (self.weight - self.weight.detach())

            # Clamp weights to prevent extreme outputs
            weight_forward = torch.clamp(weight_forward, min=-5.0, max=5.0)
            
            output = F.linear(x, weight_forward, self.bias)
            
            # Final output clamping for stability
            output = torch.clamp(output, min=-50.0, max=50.0)
            return output
        else:
            # Inference with full quantization
            weight_q = self.quantize_weights_1_58_bit(
                self.weight) * self.weight_scale
            x_q = self.quantize_activations_8bit(x)
            
            output = F.linear(x_q, weight_q, self.bias)
            output = torch.clamp(output, min=-50.0, max=50.0)
            return output


class BitNetMLP(nn.Module):
    """BitNet MLP block with 1.58-bit quantization"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = BitNetLinear(dim, hidden_dim)
        self.fc2 = BitNetLinear(hidden_dim, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm(x + residual)


class BitNetAttention(nn.Module):
    """Multi-head attention with BitNet quantization"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # BitNet quantized projections
        self.q_proj = BitNetLinear(dim, dim, bias=bias)
        self.k_proj = BitNetLinear(dim, dim, bias=bias)
        self.v_proj = BitNetLinear(dim, dim, bias=bias)
        self.out_proj = BitNetLinear(dim, dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = query.shape[:2]

        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads,
                   self.head_dim).transpose(1, 2)

        # Attention computation
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Handle mask shape: expand to match attention scores shape
            if mask.dim() == 2:  # [batch_size, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(
                    1)  # [batch_size, 1, 1, seq_len]
            elif mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]

            # Expand mask to match attention scores shape
            mask = mask.expand(batch_size, self.num_heads, seq_len, -1)
            attention_scores.masked_fill_(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, v)

        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.dim
        )
        output = self.out_proj(attended)

        return output, attention_weights.mean(dim=1)  # Average across heads


class BitNetTransformerBlock(nn.Module):
    """BitNet Transformer block with quantized components"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = BitNetAttention(dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = BitNetMLP(dim, int(dim * mlp_ratio), dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual connection
        normed_x = self.norm1(x)
        attn_out, attn_weights = self.attn(normed_x, normed_x, normed_x, mask)
        x = x + attn_out

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights


class BitNetTextEncoder(nn.Module):
    """BitNet-based text encoder"""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Token embeddings (kept full precision)
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        # BitNet transformer layers
        self.layers = nn.ModuleList([
            BitNetTransformerBlock(dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size, seq_len = input_ids.shape

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + \
            self.position_embedding(positions)
        x = self.dropout(x)

        # Transform through BitNet layers - attention tracking removed for performance
        for layer in self.layers:
            # Convert attention mask to the right format for the layer
            layer_mask = None
            if attention_mask is not None:
                # Create a mask where 1 means attend, 0 means don't attend
                layer_mask = attention_mask.unsqueeze(
                    1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]

            x, _ = layer(x, layer_mask)  # Ignore attention weights for performance

        x = self.norm(x)
        return x, []  # Return empty list instead of attention patterns


class BitNetTextDecoder(nn.Module):
    """BitNet-based text decoder with causal masking"""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        # BitNet transformer layers
        self.layers = nn.ModuleList([
            BitNetTransformerBlock(dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        # Output projection to vocabulary
        self.lm_head = BitNetLinear(dim, vocab_size, bias=False)

        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

        # Register causal mask
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)
                       ).unsqueeze(0).unsqueeze(0)
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:

        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(
                seq_len, device=input_ids.device).unsqueeze(0)
            x = self.token_embedding(input_ids) + \
                self.position_embedding(positions)
        elif inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
            positions = torch.arange(
                seq_len, device=inputs_embeds.device).unsqueeze(0)
            x = inputs_embeds + self.position_embedding(positions)
        else:
            raise ValueError(
                "Either input_ids or inputs_embeds must be provided")

        x = self.dropout(x)

        # Create causal mask
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        if attention_mask is not None:
            # Combine causal mask with padding mask
            mask = attention_mask.unsqueeze(1).unsqueeze(2) * causal_mask
        else:
            mask = causal_mask

        # Transform through BitNet layers - attention tracking removed for performance
        for layer in self.layers:
            x, _ = layer(x, mask)  # Ignore attention weights for performance

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return {
            'logits': logits,
            'loss': loss,
            'attention_patterns': []  # Empty list for performance
        }


class EpisodicMemory(nn.Module):
    """Episodic Memory mechanism inspired by Larimar"""

    def __init__(
        self,
        memory_size: int,
        episode_dim: int,
        alpha: float = 0.1,
        direct_writing: bool = True,
        observation_noise_std: float = 1e-6
    ):
        super().__init__()
        self.memory_size = memory_size
        self.episode_dim = episode_dim
        self.alpha = alpha
        self.direct_writing = direct_writing
        self.observation_noise_std = observation_noise_std

        # Memory storage
        self.register_buffer('memory', torch.zeros(memory_size, episode_dim))
        self.register_buffer('memory_age', torch.zeros(memory_size))
        self.register_buffer('memory_usage', torch.zeros(memory_size))

        # Memory access networks
        self.query_net = BitNetLinear(episode_dim, episode_dim)
        self.key_net = BitNetLinear(episode_dim, episode_dim)
        self.value_net = BitNetLinear(episode_dim, episode_dim)

    def write_memory(self, episode: torch.Tensor) -> torch.Tensor:
        """Write episode to memory"""
        batch_size = episode.size(0)

        if self.direct_writing:
            # Direct writing: find least recently used slots
            # Ensure we don't request more indices than available memory slots
            k = min(batch_size, self.memory_size)
            _, lru_indices = self.memory_age.topk(k, largest=False)

            # If batch_size > memory_size, we need to handle multiple batches
            if batch_size > self.memory_size:
                # Process in chunks of memory_size
                for i in range(0, batch_size, self.memory_size):
                    end_idx = min(i + self.memory_size, batch_size)
                    chunk_size = end_idx - i

                    # Get LRU indices for this chunk
                    _, chunk_lru_indices = self.memory_age.topk(
                        chunk_size, largest=False)

                    # Update memory slots with proper dtype conversion
                    episode_chunk = episode[i:end_idx].detach()
                    # Ensure dtype compatibility for mixed precision training
                    if episode_chunk.dtype != self.memory.dtype:
                        episode_chunk = episode_chunk.to(self.memory.dtype)
                    self.memory[chunk_lru_indices] = episode_chunk
                    self.memory_age[chunk_lru_indices] = self.memory_age.max(
                    ) + 1 + i
                    self.memory_usage[chunk_lru_indices] += 1
            else:
                # Normal case: batch_size <= memory_size
                # Update memory slots with proper dtype conversion
                episode_to_store = episode[:k].detach()
                # Ensure dtype compatibility for mixed precision training
                if episode_to_store.dtype != self.memory.dtype:
                    episode_to_store = episode_to_store.to(self.memory.dtype)
                self.memory[lru_indices] = episode_to_store
                self.memory_age[lru_indices] = self.memory_age.max() + 1
                self.memory_usage[lru_indices] += 1

        return episode

    def read_memory(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read from memory using attention mechanism"""
        batch_size = query.size(0)

        # Compute attention weights
        q = self.query_net(query)  # [batch_size, episode_dim]
        k = self.key_net(self.memory)  # [memory_size, episode_dim]
        v = self.value_net(self.memory)  # [memory_size, episode_dim]

        # Attention scores
        attention_scores = torch.matmul(
            q, k.transpose(0, 1)) / math.sqrt(self.episode_dim)
        # [batch_size, memory_size]
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Weighted memory retrieval
        # [batch_size, episode_dim]
        retrieved = torch.matmul(attention_weights, v)

        # Update memory access statistics with dtype safety
        access_counts = attention_weights.sum(0).detach()
        # Ensure dtype compatibility for mixed precision training
        if access_counts.dtype != self.memory_usage.dtype:
            access_counts = access_counts.to(self.memory_usage.dtype)
        self.memory_usage += access_counts

        return retrieved, attention_weights

    def forward(self, episode: torch.Tensor, mode: str = "read_write") -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through episodic memory"""
        if mode == "write":
            return self.write_memory(episode), None
        elif mode == "read":
            return self.read_memory(episode)
        else:  # read_write
            # Write episode to memory
            self.write_memory(episode)
            # Read from memory
            retrieved, attention_weights = self.read_memory(episode)
            return retrieved, attention_weights


class LearnableQueryFusion(nn.Module):
    """
    Enhanced QFormer-inspired Cross-modal fusion using Quadrangle Attention
    Bridges text and vision modalities through quadrangle attention patterns
    Integrates with episodic memory for human-like learning
    """

    def __init__(
        self,
        text_dim: int,
        vision_dim: int,
        hidden_dim: int,
        num_queries: int = 32,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_quadrangle: bool = True,
        episodic_memory_size: int = 1024
    ):
        super().__init__()
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_quadrangle = use_quadrangle

        # Learnable query tokens - these bridge text and vision
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        # Projection layers to common dimension
        self.text_proj = BitNetLinear(text_dim, hidden_dim)
        self.vision_proj = BitNetLinear(vision_dim, hidden_dim)

        if self.use_quadrangle:
            # Use Quadrangle Attention with episodic memory integration
            self.quadrangle_processor = EpisodicQuadrangleProcessor(
                dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                memory_size=episodic_memory_size,
                episode_dim=hidden_dim // 2,
                memory_alpha=0.1
            )
            
            # Query integration layers for quadrangle output
            self.query_integration = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim)
                ) for _ in range(num_layers)
            ])
            
        else:
            # Fallback to original query-based attention layers
            self._init_original_attention_layers(dropout)

        
        # Query-based output projection
        self.output_proj = BitNetLinear(hidden_dim, hidden_dim)

        # Learnable position embeddings for queries
        self.query_pos_embed = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim))
        nn.init.trunc_normal_(self.query_pos_embed, std=0.02)

    def _init_original_attention_layers(self, dropout: float):
        """Initialize original query-based attention layers for fallback"""
        # Query-based attention layers
        self.query_layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = nn.ModuleDict({
                # Query-to-Text attention (queries attend to text)
                'q2t_attention': BitNetAttention(
                    dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout=dropout
                ),
                'q2t_norm': nn.LayerNorm(self.hidden_dim),

                # Query-to-Vision attention (queries attend to vision)
                'q2v_attention': BitNetAttention(
                    dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout=dropout
                ),
                'q2v_norm': nn.LayerNorm(self.hidden_dim),

                # Query self-attention (queries attend to each other)
                'self_attention': BitNetAttention(
                    dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout=dropout
                ),
                'self_norm': nn.LayerNorm(self.hidden_dim),

                # MLP for query refinement
                'mlp': BitNetMLP(self.hidden_dim, self.hidden_dim * 4, dropout),
                'mlp_norm': nn.LayerNorm(self.hidden_dim)
            })
            self.query_layers.append(layer)

        # Text-to-Query attention (text tokens attend to learned queries)
        self.text2query_layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = nn.ModuleDict({
                'attention': BitNetAttention(
                    dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout=dropout
                ),
                'norm': nn.LayerNorm(self.hidden_dim),
                'mlp': BitNetMLP(self.hidden_dim, self.hidden_dim * 4, dropout),
                'mlp_norm': nn.LayerNorm(self.hidden_dim)
            })
            self.text2query_layers.append(layer)

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        mode: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            text_features: [batch_size, seq_len, text_dim]
            vision_features: [batch_size, vision_dim]
            mode: Training mode for episodic memory ("episodic_capture", "consolidation", "integration", "train")

        Returns:
            fused_features: [batch_size, seq_len, hidden_dim]
            attention_weights: Dict of attention patterns
        """
        batch_size, seq_len = text_features.shape[:2]

        # Project to common dimension
        text_proj = self.text_proj(text_features)  # [B, seq_len, hidden_dim]
        vision_proj = self.vision_proj(vision_features)  # [B, hidden_dim]
        
        # Expand vision features to match text sequence length
        vision_expanded = vision_proj.unsqueeze(1).expand(-1, seq_len, -1)  # [B, seq_len, hidden_dim]

        if self.use_quadrangle:
            # Use Quadrangle Attention for enhanced multimodal understanding
            result = self.quadrangle_processor(
                text_features=text_proj,
                image_features=vision_expanded,
                mode=mode
            )
            
            enhanced_text = result['text_features']
            enhanced_image = result['image_features']
            attention_maps = result['attention_maps']
            
            # Initialize learnable queries
            queries = self.query_tokens.expand(batch_size, -1, -1)
            queries = queries + self.query_pos_embed
            
            # Integrate queries with enhanced features through attention
            for i, integration_layer in enumerate(self.query_integration):
                # Attend to enhanced text and image features
                query_context = torch.cat([
                    enhanced_text.mean(dim=1, keepdim=True),  # Global text context
                    enhanced_image.mean(dim=1, keepdim=True)  # Global image context
                ], dim=1)  # [B, 2, hidden_dim]
                
                # Apply integration layer
                integrated_queries = integration_layer(queries)
                
                # Query attention to multimodal context
                query_attn_weights = F.softmax(
                    torch.matmul(integrated_queries, query_context.transpose(1, 2)) / math.sqrt(self.hidden_dim),
                    dim=-1
                )
                attended_context = torch.matmul(query_attn_weights, query_context)
                queries = queries + attended_context
            
            # Project queries back to text sequence
            output = self.output_proj(enhanced_text + queries.mean(dim=1, keepdim=True))
            
            # Add episodic information to attention maps
            attention_maps.update({
                'episodic_episode': result.get('episode'),
                'memory_usage': result.get('memory_usage'),
                'memory_age': result.get('memory_age')
            })
            
            return output, attention_maps
            
        else:
            # Fallback to original query-based attention
            return self._forward_original(text_proj, vision_proj)

    def _forward_original(
        self, 
        text_proj: torch.Tensor, 
        vision_proj: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Original forward implementation for fallback"""
        batch_size, seq_len = text_proj.shape[:2]
        vision_proj = vision_proj.unsqueeze(1)  # [B, 1, hidden_dim]

        # Initialize learnable queries
        queries = self.query_tokens.expand(batch_size, -1, -1)
        queries = queries + self.query_pos_embed

        # Phase 1: Query learning - queries extract information from both modalities
        for i, layer in enumerate(self.query_layers):
            # Query-to-Text: queries attend to text tokens
            q2t_out, _ = layer['q2t_attention'](
                query=queries,
                key=text_proj,
                value=text_proj
            )
            queries = layer['q2t_norm'](queries + q2t_out)

            # Query-to-Vision: queries attend to vision features
            q2v_out, _ = layer['q2v_attention'](
                query=queries,
                key=vision_proj,
                value=vision_proj
            )
            queries = layer['q2v_norm'](queries + q2v_out)

            # Query self-attention: queries refine themselves
            self_out, _ = layer['self_attention'](
                query=queries,
                key=queries,
                value=queries
            )
            queries = layer['self_norm'](queries + self_out)

            # MLP refinement
            mlp_out = layer['mlp'](queries)
            queries = layer['mlp_norm'](queries + mlp_out)

        # Phase 2: Text enhancement - text tokens attend to learned queries
        enhanced_text = text_proj
        for i, layer in enumerate(self.text2query_layers):
            # Text-to-Query: text tokens attend to learned queries
            t2q_out, _ = layer['attention'](
                query=enhanced_text,
                key=queries,
                value=queries
            )
            enhanced_text = layer['norm'](enhanced_text + t2q_out)

            # MLP refinement
            mlp_out = layer['mlp'](enhanced_text)
            enhanced_text = layer['mlp_norm'](enhanced_text + mlp_out)

        # Final output projection
        output = self.output_proj(enhanced_text)

        return output, {}  # Return empty dict instead of attention weights


# Keep the old CrossModalFusion for backward compatibility, but replace it
class CrossModalFusion(LearnableQueryFusion):
    """Alias for backward compatibility"""
    pass


class VisionEncoder(nn.Module):
    """Quantized Vision Encoder for DiNOv2 features"""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 2
    ):
        super().__init__()

        # Quantized layers
        self.layers = nn.ModuleList([
            BitNetLinear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        # Output projection
        self.output_proj = BitNetLinear(hidden_dim, output_dim)

        # Activation and normalization
        self.activation = nn.GELU()
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.1)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [batch_size, input_dim] - DiNOv2 features

        Returns:
            encoded_features: [batch_size, output_dim]
        """
        x = vision_features

        for layer, norm in zip(self.layers, self.layer_norms):
            x = layer(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)

        # Output projection
        output = self.output_proj(x)

        return output


class BitMarModel(nn.Module):
    """
    BitMar: Vision-Language Episodic Memory Transformer
    Combines BitNet quantization, DiNOv2 vision, and Larimar episodic memory
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # BitNet text encoder/decoder
        self.text_encoder = BitNetTextEncoder(
            vocab_size=config['vocab_size'],
            dim=config['text_encoder_dim'],
            num_layers=config['text_encoder_layers'],
            num_heads=config['text_encoder_heads'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout']
        )

        self.text_decoder = BitNetTextDecoder(
            vocab_size=config['vocab_size'],
            dim=config['text_decoder_dim'],
            num_layers=config['text_decoder_layers'],
            num_heads=config['text_decoder_heads'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout']
        )

        # Vision processing with BitNet quantization
        self.vision_encoder = VisionEncoder(
            input_dim=config['vision_encoder_dim'],
            hidden_dim=config['vision_hidden_size'],
            output_dim=config['vision_latent_size']
        )

        # Cross-modal fusion with Quadrangle Attention
        self.fusion = CrossModalFusion(
            text_dim=config['text_encoder_dim'],
            vision_dim=config['vision_latent_size'],
            hidden_dim=config['fusion_hidden_size'],
            # NEW: Use config parameter
            num_queries=config.get('fusion_num_queries', 32),
            num_heads=config['fusion_num_heads'],
            num_layers=config['fusion_num_layers'],
            # Enable Quadrangle Attention with episodic memory
            use_quadrangle=config.get('use_quadrangle_attention', True),
            episodic_memory_size=config.get('quadrangle_memory_size', config['memory_size'])
        )

        # Episodic memory with BitNet quantization
        self.memory = EpisodicMemory(
            memory_size=config['memory_size'],
            episode_dim=config['episode_dim'],
            alpha=config['memory_alpha'],
            direct_writing=config['direct_writing']
        )

        # Additional BitNet projection layers
        self.text_to_episode = BitNetLinear(
            config['text_encoder_dim'],
            config['episode_dim']
        )

        self.memory_to_decoder = BitNetLinear(
            config['episode_dim'],
            config['fusion_hidden_size']
        )

        # Projection to decoder dimension
        self.decoder_input_proj = BitNetLinear(
            config['fusion_hidden_size'],
            config['text_decoder_dim']
        )

        # CRITICAL FIX: Pre-initialize dynamic projection layers to prevent NaN
        # These layers were previously created on-the-fly during forward pass
        
        # Vision projection for compressed features (64 -> vision_encoder_dim)
        self.compressed_vision_proj = BitNetLinear(
            64, config['vision_encoder_dim']
        )
        
        # Vision to episode projection (vision_latent_size -> episode_dim)
        self.vision_to_episode = BitNetLinear(
            config['vision_latent_size'], 
            config['episode_dim']
        )
        
        # Enhanced initialization for numerical stability
        with torch.no_grad():
            # Initialize compressed vision projection
            nn.init.xavier_uniform_(self.compressed_vision_proj.weight, gain=0.1)
            if self.compressed_vision_proj.bias is not None:
                self.compressed_vision_proj.bias.zero_()
            self.compressed_vision_proj.weight.clamp_(-1.0, 1.0)
            
            # Initialize vision to episode projection
            nn.init.xavier_uniform_(self.vision_to_episode.weight, gain=0.1)
            if self.vision_to_episode.bias is not None:
                self.vision_to_episode.bias.zero_()
            self.vision_to_episode.weight.clamp_(-1.0, 1.0)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Track training steps for consolidation logic
        self.global_step = 0

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode text using BitNet encoder"""
        text_features, attention_patterns = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask)
        return text_features, attention_patterns

    def encode_vision(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Encode vision features"""
        # Check if we're using compressed features (64 dims) vs original (768 dims)
        if vision_features.shape[-1] == 64:
            # We're using compressed features - use pre-initialized projection layer
            vision_features = self.compressed_vision_proj(vision_features)
            logger.debug(f"Using compressed vision projection: 64 → {self.config['vision_encoder_dim']}")

        # Now process with normal vision encoder
        vision_latent = self.vision_encoder(
            vision_features
        )
        return vision_latent

    def create_episode(
        self,
        text_features: torch.Tensor,
        vision_latent: torch.Tensor,
        attention_weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Create multimodal episode for memory storage"""
        # Pool text features (mean pooling)
        # [batch_size, text_encoder_dim]
        text_pooled = text_features.mean(dim=1)

        # Project text to episode dimension
        text_projected = self.text_to_episode(text_pooled)

        # Handle dimension mismatch between text and vision features
        # Vision features might be compressed (e.g., 64D) while text is projected to episode_dim (e.g., 96D)
        if text_projected.shape[-1] != vision_latent.shape[-1]:
            # Use pre-initialized vision projection layer to match text dimensions
            try:
                vision_projected = self.vision_to_episode(vision_latent)
                # Simplified finite check that's torch.compile friendly
                if torch.any(torch.isnan(vision_projected)):
                    logger.warning("NaN values in vision projection, clamping...")
                    vision_projected = torch.clamp(vision_projected, -10.0, 10.0)
            except Exception as e:
                logger.error(f"Vision projection failed: {e}")
                # Fallback: use zero projection
                vision_projected = torch.zeros_like(text_projected)
        else:
            vision_projected = vision_latent

        # Combine text and vision features (now with matching dimensions)
        episode = text_projected + vision_projected  # Simple addition fusion

        return episode

    def _consolidation_fusion(
        self,
        text_features: torch.Tensor,
        vision_latent: torch.Tensor,
        mode: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        🧠 EPISODIC MEMORY CONSOLIDATION: Enhanced QFormer-based fusion
        
        Different processing strategies for each consolidation phase:
        - episodic_capture: Fast, high-capacity QFormer processing
        - consolidation: Pattern extraction and replay-aware fusion
        - integration: Semantic integration with refined attention
        """
        
        if mode == "episodic_capture":
            # Phase 1: Fast episodic capture with enhanced QFormer attention
            # Use full QFormer capacity for rich multimodal encoding
            fused_features, attention_weights = self.fusion(text_features, vision_latent, mode)
            
            # Enhance attention patterns for better episodic encoding
            for key, attn in attention_weights.items():
                if isinstance(attn, torch.Tensor) and 'q2v' in key or 'q2t' in key:
                    # Sharpen cross-modal attention for clearer episodic traces
                    attention_weights[key] = torch.softmax(attn * 1.5, dim=-1)
            
        elif mode == "consolidation":
            # Phase 2: Pattern extraction and memory replay-aware fusion
            fused_features, attention_weights = self.fusion(text_features, vision_latent, mode)
            
            # Add consolidation-specific processing
            # Encourage pattern extraction by emphasizing consistent attention patterns
            batch_size = text_features.size(0)
            
            # Pattern consistency regularization (encourage stable patterns)
            for key, attn in attention_weights.items():
                if isinstance(attn, torch.Tensor) and 'query_self' in key:
                    # Encourage self-consistency in query patterns
                    attention_weights[key] = torch.softmax(attn * 1.2, dim=-1)
            
        elif mode == "integration":
            # Phase 3: Semantic integration with refined processing
            fused_features, attention_weights = self.fusion(text_features, vision_latent, mode)
            
            # Integration-specific refinement
            # Emphasize semantic coherence by smoothing attention patterns
            for key, attn in attention_weights.items():
                if isinstance(attn, torch.Tensor) and 't2q' in key:
                    # Smoother text-to-query attention for semantic integration
                    attention_weights[key] = torch.softmax(attn * 0.8, dim=-1)
        
        else:
            # Fallback to standard fusion
            fused_features, attention_weights = self.fusion(text_features, vision_latent, mode)
        
        # Add consolidation mode information to attention weights
        attention_weights['consolidation_mode'] = mode
        
        return fused_features, attention_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mode: str = "train"
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with episodic memory consolidation support

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            vision_features: [batch_size, vision_dim]
            labels: [batch_size, seq_len] - for training
            mode: "train", "inference", "episodic_capture", "consolidation", "integration"

        Returns:
            Dictionary containing outputs and analysis
        """
        batch_size, seq_len = input_ids.shape

        # Encode text and vision
        text_features, text_attention = self.encode_text(
            input_ids, attention_mask)
        vision_latent = self.encode_vision(vision_features)

        # 🧠 EPISODIC MEMORY CONSOLIDATION: QFormer-based multimodal fusion
        if mode in ["episodic_capture", "consolidation", "integration"]:
            # Enhanced QFormer processing for consolidation training
            fused_features, cross_attention = self._consolidation_fusion(
                text_features, vision_latent, mode)
        else:
            # Standard fusion for regular training - pass mode for Quadrangle Attention
            fused_features, cross_attention = self.fusion(
                text_features, vision_latent, mode)

        # Create multimodal episode
        episode = self.create_episode(
            text_features, vision_latent, cross_attention)

        # 🧠 CONSOLIDATION-AWARE MEMORY INTERACTION
        if mode == "episodic_capture":
            # Rapid episodic storage - prioritize writing
            self.memory.write_memory(episode)
            retrieved_memory, memory_attention = self.memory.read_memory(episode)
        elif mode == "consolidation":
            # Balanced read/write with replay emphasis
            retrieved_memory, memory_attention = self.memory(episode, mode="read_write")
            # Additional replay mechanism handled in trainer
        elif mode == "integration":
            # Enhanced retrieval for semantic integration
            retrieved_memory, memory_attention = self.memory.read_memory(episode)
            # Write less frequently during integration
            if hasattr(self, 'global_step') and self.global_step % 5 == 0:  # Write every 5 steps
                self.memory.write_memory(episode)
        else:
            # Standard memory interaction
            if mode == "train":
                # Write and read from memory
                retrieved_memory, memory_attention = self.memory(
                    episode, mode="read_write")
            else:
                # Only read from memory during inference
                retrieved_memory, memory_attention = self.memory(
                    episode, mode="read")

        # Prepare decoder input
        memory_context = self.memory_to_decoder(
            retrieved_memory)  # [batch_size, fusion_hidden_size]

        # Add memory context to fused features
        # Broadcast memory context to sequence length
        memory_context_expanded = memory_context.unsqueeze(
            1).expand(-1, seq_len, -1)
        fused_with_memory = fused_features + memory_context_expanded

        # Project to decoder dimension
        # [batch_size, seq_len, text_decoder_dim]
        decoder_input = self.decoder_input_proj(fused_with_memory)

        # Generate text using BitNet decoder
        decoder_outputs = self.text_decoder(
            inputs_embeds=decoder_input,
            attention_mask=attention_mask,
            labels=labels
        )

        return {
            'loss': decoder_outputs['loss'],
            'logits': decoder_outputs['logits'],
            'text_features': text_features,
            'vision_latent': vision_latent,
            'fused_features': fused_features,
            'episode': episode,
            'retrieved_memory': retrieved_memory,
            'cross_attention': cross_attention,
            'memory_attention': memory_attention,
            'text_attention': text_attention,
            'decoder_attention': decoder_outputs['attention_patterns'],
            'memory_usage': self.memory.memory_usage.clone(),
            'consolidation_mode': mode
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_features: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, torch.Tensor]:
        """Generate text given input text and vision features"""
        self.eval()

        with torch.no_grad():
            # Encode inputs
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_features=vision_features,
                mode="inference"
            )

            # Start with input sequence
            generated_ids = input_ids.clone()

            for _ in range(max_length - input_ids.size(1)):
                # Get next token logits
                # [batch_size, vocab_size]
                next_logits = outputs['logits'][:, -1, :]

                # Apply temperature
                next_logits = next_logits / temperature

                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[...,
                                             1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones_like(next_token)
                ], dim=1)

                # Check for EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Update outputs for next iteration
                outputs = self.forward(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    vision_features=vision_features,
                    mode="inference"
                )

        return {
            'generated_ids': generated_ids,
            'generated_text': self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True),
            'attention_patterns': outputs['cross_attention'],
            'memory_patterns': outputs['memory_attention']
        }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def create_bitmar_model(config: Dict) -> BitMarModel:
    """Create BitMar model from configuration"""
    model = BitMarModel(config)

    # Print model statistics
    param_count = count_parameters(model)
    logger.info(
        f"BitMar Model created with {param_count['total_parameters']:,} total parameters")
    logger.info(
        f"Trainable parameters: {param_count['trainable_parameters']:,}")

    return model
