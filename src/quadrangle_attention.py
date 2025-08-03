"""
Quadrangle Attention implementation for BitMar
Based on QFormer's quadrangle attention mechanism for enhanced image understanding and text grounding
Integrated with BitNet quantization for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math
import logging

logger = logging.getLogger(__name__)


class QuadrangleAttention(nn.Module):
    """
    Quadrangle Attention mechanism from QFormer
    Creates four attention patterns for comprehensive multimodal understanding:
    1. Image-to-Text: Visual features attend to text tokens
    2. Text-to-Image: Text tokens attend to visual features  
    3. Image-to-Image: Visual self-attention for spatial understanding
    4. Text-to-Text: Text self-attention for linguistic understanding
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        scale_factor: float = None
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = scale_factor or (self.head_dim ** -0.5)
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Separate projection layers for each quadrangle pattern
        # Image-to-Text projections
        self.i2t_q = nn.Linear(dim, dim, bias=bias)
        self.i2t_k = nn.Linear(dim, dim, bias=bias)
        self.i2t_v = nn.Linear(dim, dim, bias=bias)
        
        # Text-to-Image projections
        self.t2i_q = nn.Linear(dim, dim, bias=bias)
        self.t2i_k = nn.Linear(dim, dim, bias=bias)
        self.t2i_v = nn.Linear(dim, dim, bias=bias)
        
        # Image-to-Image projections (visual self-attention)
        self.i2i_q = nn.Linear(dim, dim, bias=bias)
        self.i2i_k = nn.Linear(dim, dim, bias=bias)
        self.i2i_v = nn.Linear(dim, dim, bias=bias)
        
        # Text-to-Text projections (text self-attention)
        self.t2t_q = nn.Linear(dim, dim, bias=bias)
        self.t2t_k = nn.Linear(dim, dim, bias=bias)
        self.t2t_v = nn.Linear(dim, dim, bias=bias)
        
        # Output projections for each pattern
        self.i2t_out = nn.Linear(dim, dim)
        self.t2i_out = nn.Linear(dim, dim)
        self.i2i_out = nn.Linear(dim, dim)
        self.t2t_out = nn.Linear(dim, dim)
        
        # Gating mechanism for adaptive pattern weighting
        self.pattern_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 4),  # 4 patterns
            nn.Softmax(dim=-1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization for each pattern
        self.i2t_norm = nn.LayerNorm(dim)
        self.t2i_norm = nn.LayerNorm(dim)
        self.i2i_norm = nn.LayerNorm(dim)
        self.t2t_norm = nn.LayerNorm(dim)
        
    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            text_features: [B, T, D] - Text token features
            image_features: [B, N, D] - Image patch/region features
            text_mask: [B, T] - Text attention mask
            image_mask: [B, N] - Image attention mask
            
        Returns:
            enhanced_text: [B, T, D] - Enhanced text features
            enhanced_image: [B, N, D] - Enhanced image features
            attention_maps: Dict containing attention patterns
        """
        B, T, D = text_features.shape
        B, N, D = image_features.shape
        
        attention_maps = {}
        
        # 1. Image-to-Text Attention (Visual grounding in text)
        i2t_q = self.reshape_for_attention(self.i2t_q(image_features))  # [B, H, N, d]
        i2t_k = self.reshape_for_attention(self.i2t_k(text_features))   # [B, H, T, d]
        i2t_v = self.reshape_for_attention(self.i2t_v(text_features))   # [B, H, T, d]
        
        i2t_attn, i2t_weights = self.compute_attention(
            i2t_q, i2t_k, i2t_v, text_mask
        )
        i2t_out = self.i2t_out(i2t_attn.reshape(B, N, D))
        attention_maps['image_to_text'] = i2t_weights
        
        # 2. Text-to-Image Attention (Textual guidance for visual understanding)
        t2i_q = self.reshape_for_attention(self.t2i_q(text_features))   # [B, H, T, d]
        t2i_k = self.reshape_for_attention(self.t2i_k(image_features))  # [B, H, N, d]
        t2i_v = self.reshape_for_attention(self.t2i_v(image_features))  # [B, H, N, d]
        
        t2i_attn, t2i_weights = self.compute_attention(
            t2i_q, t2i_k, t2i_v, image_mask
        )
        t2i_out = self.t2i_out(t2i_attn.reshape(B, T, D))
        attention_maps['text_to_image'] = t2i_weights
        
        # 3. Image-to-Image Attention (Spatial visual understanding)
        i2i_q = self.reshape_for_attention(self.i2i_q(image_features))  # [B, H, N, d]
        i2i_k = self.reshape_for_attention(self.i2i_k(image_features))  # [B, H, N, d]
        i2i_v = self.reshape_for_attention(self.i2i_v(image_features))  # [B, H, N, d]
        
        i2i_attn, i2i_weights = self.compute_attention(
            i2i_q, i2i_k, i2i_v, image_mask
        )
        i2i_out = self.i2i_out(i2i_attn.reshape(B, N, D))
        attention_maps['image_to_image'] = i2i_weights
        
        # 4. Text-to-Text Attention (Linguistic understanding)
        t2t_q = self.reshape_for_attention(self.t2t_q(text_features))   # [B, H, T, d]
        t2t_k = self.reshape_for_attention(self.t2t_k(text_features))   # [B, H, T, d]
        t2t_v = self.reshape_for_attention(self.t2t_v(text_features))   # [B, H, T, d]
        
        t2t_attn, t2t_weights = self.compute_attention(
            t2t_q, t2t_k, t2t_v, text_mask
        )
        t2t_out = self.t2t_out(t2t_attn.reshape(B, T, D))
        attention_maps['text_to_text'] = t2t_weights
        
        # Adaptive pattern weighting with gating
        # Compute global context for gating
        text_global = text_features.mean(dim=1)  # [B, D]
        image_global = image_features.mean(dim=1)  # [B, D]
        global_context = torch.cat([text_global, image_global], dim=-1)  # [B, 2*D]
        
        pattern_weights = self.pattern_gate(global_context)  # [B, 4]
        
        # Apply adaptive weighting to each pattern
        w_i2t, w_t2i, w_i2i, w_t2t = pattern_weights.unbind(dim=-1)
        
        # Enhanced text features (combine text-to-image and text-to-text)
        enhanced_text = (
            w_t2i.unsqueeze(1).unsqueeze(2) * self.t2i_norm(text_features + t2i_out) +
            w_t2t.unsqueeze(1).unsqueeze(2) * self.t2t_norm(text_features + t2t_out)
        )
        
        # Enhanced image features (combine image-to-text and image-to-image)
        enhanced_image = (
            w_i2t.unsqueeze(1).unsqueeze(2) * self.i2t_norm(image_features + i2t_out) +
            w_i2i.unsqueeze(1).unsqueeze(2) * self.i2i_norm(image_features + i2i_out)
        )
        
        # Store pattern weights for analysis
        attention_maps['pattern_weights'] = {
            'image_to_text': w_i2t,
            'text_to_image': w_t2i,
            'image_to_image': w_i2i,
            'text_to_text': w_t2t
        }
        
        return enhanced_text, enhanced_image, attention_maps
    
    def reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention"""
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
    
    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention"""
        # q: [B, H, L_q, d], k: [B, H, L_k, d], v: [B, H, L_k, d]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L_k]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        return attn_output, attn_weights


class QuadrangleTransformerBlock(nn.Module):
    """
    Transformer block with Quadrangle Attention for multimodal understanding
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.quadrangle_attn = QuadrangleAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # MLP layers for text and image features
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.text_mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.image_mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.text_norm1 = nn.LayerNorm(dim)
        self.text_norm2 = nn.LayerNorm(dim)
        self.image_norm1 = nn.LayerNorm(dim)
        self.image_norm2 = nn.LayerNorm(dim)
        
    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with residual connections
        """
        # Quadrangle attention with residual connection
        text_normed = self.text_norm1(text_features)
        image_normed = self.image_norm1(image_features)
        
        enhanced_text, enhanced_image, attention_maps = self.quadrangle_attn(
            text_normed, image_normed, text_mask, image_mask
        )
        
        text_features = text_features + enhanced_text
        image_features = image_features + enhanced_image
        
        # MLP with residual connection
        text_features = text_features + self.text_mlp(self.text_norm2(text_features))
        image_features = image_features + self.image_mlp(self.image_norm2(image_features))
        
        return text_features, image_features, attention_maps


class EpisodicQuadrangleProcessor(nn.Module):
    """
    Integrates Quadrangle Attention with Episodic Memory for human-like learning
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 3,
        memory_size: int = 1024,
        episode_dim: int = 512,
        memory_alpha: float = 0.1
    ):
        super().__init__()
        
        self.dim = dim
        self.memory_size = memory_size
        self.episode_dim = episode_dim
        self.memory_alpha = memory_alpha
        
        # Quadrangle attention layers
        self.quadrangle_layers = nn.ModuleList([
            QuadrangleTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                dropout=0.1
            ) for _ in range(num_layers)
        ])
        
        # Episodic memory components
        self.memory = nn.Parameter(torch.randn(memory_size, episode_dim))
        self.memory_age = nn.Parameter(torch.zeros(memory_size), requires_grad=False)
        self.memory_usage = nn.Parameter(torch.zeros(memory_size), requires_grad=False)
        
        # Memory projection layers
        self.text_to_episode = nn.Linear(dim, episode_dim)
        self.image_to_episode = nn.Linear(dim, episode_dim)
        self.episode_to_features = nn.Linear(episode_dim, dim)
        
        # Memory attention for retrieval
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=episode_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Semantic integration layers
        self.semantic_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        self.register_buffer('global_step', torch.tensor(0))
        
    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        mode: str = "train",
        text_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with episodic and semantic memory integration
        
        Args:
            text_features: [B, T, D]
            image_features: [B, N, D] 
            mode: "episodic_capture", "consolidation", "integration", or "train"
        """
        batch_size = text_features.size(0)
        all_attention_maps = {}
        
        # Phase 1: Quadrangle Attention Processing
        current_text = text_features
        current_image = image_features
        
        for i, layer in enumerate(self.quadrangle_layers):
            current_text, current_image, layer_attention = layer(
                current_text, current_image, text_mask, image_mask
            )
            all_attention_maps[f'layer_{i}'] = layer_attention
        
        # Phase 2: Episode Creation for Memory
        text_episode = self.text_to_episode(current_text.mean(dim=1))  # [B, episode_dim]
        image_episode = self.image_to_episode(current_image.mean(dim=1))  # [B, episode_dim]
        
        # Multimodal episode combining text and image understanding
        multimodal_episode = (text_episode + image_episode) / 2  # [B, episode_dim]
        
        # Phase 3: Episodic Memory Operations
        if mode == "episodic_capture":
            # Fast episodic capture - store new experiences rapidly
            memory_output = self._episodic_capture(multimodal_episode)
            
        elif mode == "consolidation":
            # Memory consolidation - replay and strengthen important patterns
            memory_output, retrieved_episodes = self._memory_consolidation(multimodal_episode)
            all_attention_maps['memory_retrieval'] = retrieved_episodes
            
        elif mode == "integration":
            # Semantic integration - combine episodic and semantic knowledge
            memory_output, semantic_context = self._semantic_integration(multimodal_episode)
            all_attention_maps['semantic_context'] = semantic_context
            
        else:
            # Standard training mode
            memory_output = self._standard_memory_operation(multimodal_episode)
        
        # Phase 4: Memory-Enhanced Feature Integration
        memory_features = self.episode_to_features(memory_output)  # [B, dim]
        memory_features = memory_features.unsqueeze(1)  # [B, 1, dim]
        
        # Enhance text features with memory context
        text_memory_context = torch.cat([current_text, memory_features.expand(-1, current_text.size(1), -1)], dim=-1)
        enhanced_text = self.semantic_fusion(text_memory_context)
        
        # Enhance image features with memory context  
        image_memory_context = torch.cat([current_image, memory_features.expand(-1, current_image.size(1), -1)], dim=-1)
        enhanced_image = self.semantic_fusion(image_memory_context)
        
        return {
            'text_features': enhanced_text,
            'image_features': enhanced_image,
            'episode': multimodal_episode,
            'memory_output': memory_output,
            'attention_maps': all_attention_maps,
            'memory_usage': self.memory_usage.clone(),
            'memory_age': self.memory_age.clone()
        }
    
    def _episodic_capture(self, episode: torch.Tensor) -> torch.Tensor:
        """Fast episodic capture with high learning rate"""
        batch_size = episode.size(0)
        
        # Find least recently used memory slots for new episodes
        _, lru_indices = torch.topk(self.memory_age, batch_size, largest=False)
        
        # Store new episodes - fix dtype mismatch
        with torch.no_grad():
            # Ensure dtype compatibility for mixed precision training
            episode_to_store = episode.detach().to(self.memory.dtype)
            self.memory[lru_indices] = episode_to_store
            self.memory_age[lru_indices] = self.global_step.float()
            self.memory_usage[lru_indices] += 1
        
        return episode
    
    def _memory_consolidation(self, episode: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Memory consolidation with replay mechanism"""
        batch_size = episode.size(0)
        
        # Retrieve relevant episodes from memory
        episode_expanded = episode.unsqueeze(1)  # [B, 1, episode_dim]
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)  # [B, memory_size, episode_dim]
        
        # Attention-based memory retrieval
        retrieved_memory, retrieval_weights = self.memory_attention(
            episode_expanded, memory_expanded, memory_expanded
        )
        
        # Update memory usage based on retrieval
        with torch.no_grad():
            usage_update = retrieval_weights.sum(dim=0).sum(dim=0)  # [memory_size]
            self.memory_usage += usage_update
        
        # Combine current episode with retrieved memory
        consolidated_output = (episode + retrieved_memory.squeeze(1)) / 2
        
        return consolidated_output, retrieval_weights
    
    def _semantic_integration(self, episode: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Semantic integration of episodic and general knowledge"""
        batch_size = episode.size(0)
        
        # Retrieve semantically similar episodes
        episode_expanded = episode.unsqueeze(1)
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute semantic similarity
        similarity = torch.cosine_similarity(
            episode_expanded, memory_expanded, dim=-1
        )  # [B, memory_size]
        
        # Select top-k most similar episodes for integration
        k = min(5, self.memory_size)
        top_k_sim, top_k_indices = torch.topk(similarity, k, dim=-1)
        
        # Weighted integration of similar episodes
        top_k_episodes = torch.gather(
            memory_expanded, 1, 
            top_k_indices.unsqueeze(-1).expand(-1, -1, self.episode_dim)
        )  # [B, k, episode_dim]
        
        weights = F.softmax(top_k_sim, dim=-1).unsqueeze(-1)  # [B, k, 1]
        semantic_context = (top_k_episodes * weights).sum(dim=1)  # [B, episode_dim]
        
        # Integration with current episode
        integrated_output = (episode + semantic_context) / 2
        
        return integrated_output, semantic_context
    
    def _standard_memory_operation(self, episode: torch.Tensor) -> torch.Tensor:
        """Standard memory operation for regular training"""
        # Simple memory read without complex operations
        return episode
    
    def update_global_step(self):
        """Update global step for memory aging"""
        self.global_step += 1
