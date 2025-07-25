"""
BitMar Model Architecture
BitNet-quantized Vision-Language Episodic Memory Transformer
Combines 1.58-bit quantization, DiNOv2 vision, and Larimar episodic memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer
import math
import logging

logger = logging.getLogger(__name__)


class BitNetLinear(nn.Module):
    """1.58-bit Linear layer following BitNet b1.58 architecture"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters (full precision for training)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Quantization scaling factors
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('input_scale', torch.ones(1))

    def quantize_weights_1_58_bit(self, weight: torch.Tensor) -> torch.Tensor:
        """BitNet b1.58 weight quantization: {-1, 0, +1}"""
        # Compute scaling factor with numerical stability
        scale = weight.abs().mean()
        self.weight_scale.data = scale.clamp(min=1e-5, max=1e3)  # Prevent extreme scales

        # Normalize weights with gradient clipping
        weight_norm = torch.clamp(weight / self.weight_scale, min=-10.0, max=10.0)

        # 1.58-bit quantization with threshold
        threshold = 2.0 / 3.0  # Optimal threshold for ternary quantization

        # Create ternary weights
        quantized = torch.zeros_like(weight_norm)
        quantized[weight_norm > threshold] = 1.0
        quantized[weight_norm < -threshold] = -1.0
        # Values between -threshold and threshold remain 0

        return quantized

    def quantize_activations_8bit(self, x: torch.Tensor) -> torch.Tensor:
        """8-bit activation quantization with numerical stability"""
        # Clamp extreme values to prevent overflow
        x_clamped = torch.clamp(x, min=-1e6, max=1e6)

        # Compute quantization parameters
        x_min, x_max = x_clamped.min(), x_clamped.max()

        # Prevent division by zero
        range_val = x_max - x_min
        if range_val < 1e-8:
            return x_clamped

        scale = range_val / 255.0
        self.input_scale.data = scale.clamp(min=1e-8, max=1e3)

        # Quantize to 8-bit
        zero_point = (-x_min / scale).round().clamp(0, 255)
        quantized = ((x_clamped / scale) + zero_point).round().clamp(0, 255)

        # Dequantize
        dequantized = scale * (quantized - zero_point)
        return dequantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Full precision training with straight-through estimator
            # Forward pass with quantized weights but gradients flow through original weights
            weight_q = self.quantize_weights_1_58_bit(self.weight)
            weight_forward = weight_q * self.weight_scale

            # Use original weight for gradient computation
            weight_forward = weight_forward + \
                (self.weight - self.weight.detach())

            return F.linear(x, weight_forward, self.bias)
        else:
            # Inference with full quantization
            weight_q = self.quantize_weights_1_58_bit(
                self.weight) * self.weight_scale
            x_q = self.quantize_activations_8bit(x)
            return F.linear(x_q, weight_q, self.bias)


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

        # Transform through BitNet layers
        attention_patterns = []
        for layer in self.layers:
            # Convert attention mask to the right format for the layer
            layer_mask = None
            if attention_mask is not None:
                # Create a mask where 1 means attend, 0 means don't attend
                layer_mask = attention_mask.unsqueeze(
                    1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]

            x, attn_weights = layer(x, layer_mask)
            attention_patterns.append(attn_weights)

        x = self.norm(x)
        return x, attention_patterns


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

        # Transform through BitNet layers
        attention_patterns = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_patterns.append(attn_weights)

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
            'attention_patterns': attention_patterns
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
                    _, chunk_lru_indices = self.memory_age.topk(chunk_size, largest=False)

                    # Update memory slots
                    self.memory[chunk_lru_indices] = episode[i:end_idx].detach()
                    self.memory_age[chunk_lru_indices] = self.memory_age.max() + 1 + i
                    self.memory_usage[chunk_lru_indices] += 1
            else:
                # Normal case: batch_size <= memory_size
                # Update memory slots
                self.memory[lru_indices] = episode[:k].detach()
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

        # Update memory access statistics
        access_counts = attention_weights.sum(0)
        self.memory_usage += access_counts.detach()

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


class CrossModalFusion(nn.Module):
    """Cross-modal fusion module for text and vision features"""

    def __init__(
        self,
        text_dim: int,
        vision_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 2
    ):
        super().__init__()
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim

        # Projection layers
        self.text_proj = BitNetLinear(text_dim, hidden_dim)
        self.vision_proj = BitNetLinear(vision_dim, hidden_dim)

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            BitNetAttention(
                dim=hidden_dim,
                num_heads=num_heads
            ) for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = BitNetLinear(hidden_dim, hidden_dim)

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            text_features: [batch_size, seq_len, text_dim]
            vision_features: [batch_size, vision_dim]

        Returns:
            fused_features: [batch_size, seq_len, hidden_dim]
            attention_weights: Dict of attention patterns
        """
        batch_size, seq_len = text_features.shape[:2]

        # Project to common dimension
        # [batch_size, seq_len, hidden_dim]
        text_proj = self.text_proj(text_features)
        vision_proj = self.vision_proj(vision_features).unsqueeze(
            1)  # [batch_size, 1, hidden_dim]

        # Cross-attention fusion
        fused = text_proj
        attention_weights = {}

        for i, (attn_layer, norm_layer) in enumerate(zip(self.cross_attention_layers, self.layer_norms)):
            # Text-to-vision cross-attention
            attn_output, attn_weights = attn_layer(
                query=fused,
                key=vision_proj,
                value=vision_proj
            )

            # Residual connection and normalization
            fused = norm_layer(fused + attn_output)
            attention_weights[f'layer_{i}'] = attn_weights

        # Output projection
        output = self.output_proj(fused)

        return output, attention_weights


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
    BitMar: BitNet-quantized Vision-Language Episodic Memory Transformer
    Combines 1.58-bit quantization, DiNOv2 vision features, and Larimar episodic memory
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Loss balancing parameters
        self.cross_modal_loss_weight = config.get('cross_modal_loss_weight', 0.1)
        self.text_loss_weight = config.get('text_loss_weight', 1.0)
        self.vision_loss_weight = config.get('vision_loss_weight', 0.1)
        self.memory_loss_weight = config.get('memory_loss_weight', 0.05)

        # Dynamic loss scaling
        self.adaptive_loss_scaling = config.get('adaptive_loss_scaling', True)
        self.loss_scale_temperature = config.get('loss_scale_temperature', 0.07)

        # Encoder freezing parameters
        self.freeze_text_encoder_steps = config.get('freeze_text_encoder_steps', 0)
        self.freeze_vision_encoder_steps = config.get('freeze_vision_encoder_steps', 0)
        self.current_step = 0

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

        # Cross-modal fusion with BitNet
        self.fusion = CrossModalFusion(
            text_dim=config['text_encoder_dim'],
            vision_dim=config['vision_latent_size'],
            hidden_dim=config['fusion_hidden_size'],
            num_heads=config['fusion_num_heads'],
            num_layers=config['fusion_num_layers']
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

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode text using BitNet encoder"""
        text_features, attention_patterns = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask)
        return text_features, attention_patterns

    def encode_vision(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Encode vision features using quantized vision encoder"""
        vision_latent = self.vision_encoder(
            vision_features)  # [batch_size, vision_latent_size]
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

        # Combine text and vision features
        episode = text_projected + vision_latent  # Simple addition fusion

        return episode

    def compute_cross_modal_contrastive_loss(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Compute cross-modal contrastive loss similar to CLIP
        """
        batch_size = text_features.shape[0]

        # Normalize features
        text_features = F.normalize(text_features, dim=-1)
        vision_features = F.normalize(vision_features, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(text_features, vision_features.T) / temperature

        # Create labels (diagonal should be positive pairs)
        labels = torch.arange(batch_size, device=logits.device)

        # Compute cross-entropy loss for both directions
        text_to_vision_loss = F.cross_entropy(logits, labels)
        vision_to_text_loss = F.cross_entropy(logits.T, labels)

        return (text_to_vision_loss + vision_to_text_loss) / 2

    def compute_vision_reconstruction_loss(
        self,
        original_vision: torch.Tensor,
        reconstructed_vision: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute vision reconstruction loss to prevent vision encoder collapse
        """
        return F.mse_loss(reconstructed_vision, original_vision)

    def compute_memory_consistency_loss(
        self,
        episode: torch.Tensor,
        retrieved_memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute memory consistency loss to encourage meaningful memory usage
        """
        # L2 regularization on memory difference
        memory_diff = episode - retrieved_memory
        return torch.mean(torch.norm(memory_diff, dim=-1))

    def compute_balanced_loss(
        self,
        decoder_loss: torch.Tensor,
        cross_modal_loss: torch.Tensor,
        vision_loss: Optional[torch.Tensor] = None,
        memory_loss: Optional[torch.Tensor] = None,
        step: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute balanced multi-objective loss with adaptive scaling
        """
        losses = {'decoder_loss': decoder_loss, 'cross_modal_loss': cross_modal_loss}

        if vision_loss is not None:
            losses['vision_loss'] = vision_loss
        if memory_loss is not None:
            losses['memory_loss'] = memory_loss

        if self.adaptive_loss_scaling:
            # Adaptive scaling based on loss magnitudes
            with torch.no_grad():
                # Compute relative loss scales
                decoder_scale = decoder_loss.detach()
                cross_modal_scale = cross_modal_loss.detach()

                # Prevent division by zero
                if decoder_scale > 1e-8:
                    adaptive_cross_modal_weight = (decoder_scale / cross_modal_scale.clamp(min=1e-8)) * self.cross_modal_loss_weight
                else:
                    adaptive_cross_modal_weight = self.cross_modal_loss_weight

                # Clamp adaptive weights
                adaptive_cross_modal_weight = torch.clamp(adaptive_cross_modal_weight, 0.01, 1.0)
        else:
            adaptive_cross_modal_weight = self.cross_modal_loss_weight

        # Apply loss scheduling (increase cross-modal importance over time)
        cross_modal_schedule = min(1.0, step / 50000)  # Ramp up over 50k steps
        scheduled_cross_modal_weight = adaptive_cross_modal_weight * cross_modal_schedule

        # Compute weighted total loss
        total_loss = (
            self.text_loss_weight * decoder_loss +
            scheduled_cross_modal_weight * cross_modal_loss
        )

        if vision_loss is not None:
            total_loss += self.vision_loss_weight * vision_loss
        if memory_loss is not None:
            total_loss += self.memory_loss_weight * memory_loss

        losses.update({
            'total_loss': total_loss,
            'cross_modal_weight': scheduled_cross_modal_weight,
            'adaptive_weight': adaptive_cross_modal_weight if self.adaptive_loss_scaling else torch.tensor(0.0)
        })

        return losses

    def apply_encoder_freezing(self, step: int):
        """
        Apply temporary encoder freezing based on training step
        """
        self.current_step = step

        # Freeze text encoder if within freezing window
        freeze_text = step < self.freeze_text_encoder_steps
        for param in self.text_encoder.parameters():
            param.requires_grad = not freeze_text

        # Freeze vision encoder if within freezing window
        freeze_vision = step < self.freeze_vision_encoder_steps
        for param in self.vision_encoder.parameters():
            param.requires_grad = not freeze_vision

        return {
            'text_encoder_frozen': freeze_text,
            'vision_encoder_frozen': freeze_vision
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mode: str = "train",
        step: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through BitMar model with balanced losses
        """
        batch_size, seq_len = input_ids.shape

        # Apply encoder freezing if in training mode
        freezing_status = {}
        if mode == "train":
            freezing_status = self.apply_encoder_freezing(step)

        # Encode text and vision
        text_features, text_attention = self.encode_text(input_ids, attention_mask)
        vision_latent = self.encode_vision(vision_features)

        # Cross-modal fusion
        fused_features, cross_attention = self.fusion(text_features, vision_latent)

        # Create multimodal episode
        episode = self.create_episode(text_features, vision_latent, cross_attention)

        # Episodic memory interaction
        if mode == "train":
            retrieved_memory, memory_attention = self.memory(episode, mode="read_write")
        else:
            retrieved_memory, memory_attention = self.memory(episode, mode="read")

        # Prepare decoder input
        memory_context = self.memory_to_decoder(retrieved_memory)
        memory_context_expanded = memory_context.unsqueeze(1).expand(-1, seq_len, -1)
        fused_with_memory = fused_features + memory_context_expanded
        decoder_input = self.decoder_input_proj(fused_with_memory)

        # Generate text using BitNet decoder
        decoder_outputs = self.text_decoder(
            inputs_embeds=decoder_input,
            attention_mask=attention_mask,
            labels=labels
        )

        # Compute losses if in training mode
        if mode == "train" and labels is not None:
            # Primary decoder loss
            decoder_loss = decoder_outputs['loss']

            # Cross-modal contrastive loss
            text_pooled = text_features.mean(dim=1)  # Pool text features
            cross_modal_loss = self.compute_cross_modal_contrastive_loss(
                text_pooled, vision_latent, temperature=self.loss_scale_temperature
            )

            # Optional additional losses
            vision_loss = None
            if hasattr(self, 'vision_reconstruction') and self.config.get('use_vision_reconstruction', False):
                reconstructed_vision = self.vision_reconstruction(vision_latent)
                vision_loss = self.compute_vision_reconstruction_loss(vision_features, reconstructed_vision)

            memory_loss = None
            if self.config.get('use_memory_consistency_loss', True):
                memory_loss = self.compute_memory_consistency_loss(episode, retrieved_memory)

            # Compute balanced loss
            loss_dict = self.compute_balanced_loss(
                decoder_loss, cross_modal_loss, vision_loss, memory_loss, step
            )

            final_loss = loss_dict['total_loss']
        else:
            final_loss = decoder_outputs['loss'] if 'loss' in decoder_outputs else None
            loss_dict = {}

        result = {
            'loss': final_loss,
            'logits': decoder_outputs['logits'],
            'text_features': text_features,
            'vision_latent': vision_latent,
            'fused_features': fused_features,
            'episode': episode,
            'retrieved_memory': retrieved_memory,
            'cross_attention': cross_attention,
            'memory_attention': memory_attention,
            'text_attention': text_attention,
            'decoder_attention': decoder_outputs.get('attention_patterns', None),
            'memory_usage': self.memory.memory_usage.clone(),
        }

        # Add loss breakdown and freezing status
        if mode == "train":
            result.update(loss_dict)
            result.update(freezing_status)

        return result

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
