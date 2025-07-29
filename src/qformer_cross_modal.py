"""
QFormer-style Cross-Modal Alignment for BitMar
Implements learnable query tokens for better text-image alignment
Based on BLIP-2 QFormer architecture but adapted for BitNet quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union
from src.model import BitNetLinear, BitNetAttention


class BitNetQFormerBlock(nn.Module):
    """QFormer transformer block with BitNet quantization"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        has_cross_attention: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.has_cross_attention = has_cross_attention

        # Self-attention
        self.self_attention = BitNetAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.self_attn_norm = nn.LayerNorm(dim)

        # Cross-attention (for image features)
        if has_cross_attention:
            self.cross_attention = BitNetAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout
            )
            self.cross_attn_norm = nn.LayerNorm(dim)

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            BitNetLinear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            BitNetLinear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.mlp_norm = nn.LayerNorm(dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        # Self-attention
        self_attn_output, self_attn_weights = self.self_attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            mask=attention_mask
        )
        hidden_states = self.self_attn_norm(hidden_states + self_attn_output)

        # Cross-attention
        cross_attn_weights = None
        if self.has_cross_attention and encoder_hidden_states is not None:
            cross_attn_output, cross_attn_weights = self.cross_attention(
                query=hidden_states,
                key=encoder_hidden_states,
                value=encoder_hidden_states,
                mask=encoder_attention_mask
            )
            hidden_states = self.cross_attn_norm(hidden_states + cross_attn_output)

        # MLP
        mlp_output = self.mlp(hidden_states)
        hidden_states = self.mlp_norm(hidden_states + mlp_output)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


class BitNetQFormer(nn.Module):
    """
    QFormer for cross-modal alignment between text and vision
    Uses learnable query tokens to bridge modalities
    """

    def __init__(
        self,
        num_query_tokens: int = 32,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0.1,
        vision_feature_size: int = 768,
        max_position_embeddings: int = 512
    ):
        super().__init__()

        self.num_query_tokens = num_query_tokens
        self.hidden_size = hidden_size

        # Learnable query tokens (key innovation of QFormer)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, hidden_size)
        )
        nn.init.normal_(self.query_tokens, std=0.02)

        # Position embeddings
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_size
        )

        # Vision feature projection
        self.vision_projection = BitNetLinear(vision_feature_size, hidden_size)

        # QFormer layers
        self.layers = nn.ModuleList([
            BitNetQFormerBlock(
                dim=hidden_size,
                num_heads=num_attention_heads,
                mlp_ratio=intermediate_size / hidden_size,
                dropout=dropout,
                has_cross_attention=True
            )
            for _ in range(num_hidden_layers)
        ])

        # Output projections for different tasks
        self.text_projection = BitNetLinear(hidden_size, hidden_size)
        self.vision_projection_out = BitNetLinear(hidden_size, hidden_size)

        # Cross-modal contrastive head
        self.contrastive_head = BitNetLinear(hidden_size, hidden_size)

        # Language modeling head for generative tasks
        self.lm_head = BitNetLinear(hidden_size, hidden_size)

    def forward(
        self,
        text_embeds: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        vision_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        mode: str = "multimodal"  # "multimodal", "text", "vision"
    ) -> Dict[str, torch.Tensor]:

        batch_size = text_embeds.shape[0] if text_embeds is not None else vision_embeds.shape[0]
        device = text_embeds.device if text_embeds is not None else vision_embeds.device

        # Prepare query tokens
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        if mode == "multimodal":
            # Multimodal mode: queries attend to both text and vision

            # Prepare vision features
            if vision_embeds is not None:
                vision_embeds = self.vision_projection(vision_embeds)

                # Handle different vision input shapes
                if vision_embeds.dim() == 2:  # [batch_size, vision_dim]
                    vision_embeds = vision_embeds.unsqueeze(1)  # [batch_size, 1, hidden_size]
                elif vision_embeds.dim() == 4:  # [batch_size, channels, height, width]
                    # Flatten spatial dimensions
                    b, c, h, w = vision_embeds.shape
                    vision_embeds = vision_embeds.view(b, c, h*w).transpose(1, 2)
                    vision_embeds = self.vision_projection(vision_embeds)

            # Combine query tokens with text (if available)
            if text_embeds is not None:
                # Concatenate query tokens with text embeddings
                embeddings = torch.cat([query_tokens, text_embeds], dim=1)

                # Create attention mask
                query_mask = torch.ones(
                    batch_size, self.num_query_tokens,
                    dtype=torch.long, device=device
                )
                if text_attention_mask is not None:
                    attention_mask = torch.cat([query_mask, text_attention_mask], dim=1)
                else:
                    attention_mask = torch.ones_like(embeddings[..., 0])
            else:
                embeddings = query_tokens
                attention_mask = torch.ones(
                    batch_size, self.num_query_tokens,
                    dtype=torch.long, device=device
                )

            # Add position embeddings
            seq_length = embeddings.shape[1]
            position_ids = torch.arange(seq_length, device=device).expand(batch_size, -1)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings

            # Pass through QFormer layers
            hidden_states = embeddings
            all_attentions = []

            for layer in self.layers:
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=vision_embeds,
                    encoder_attention_mask=vision_attention_mask,
                    output_attentions=True
                )
                hidden_states = layer_outputs[0]
                if len(layer_outputs) > 1:
                    all_attentions.append(layer_outputs[1:])

            # Extract query representations
            query_output = hidden_states[:, :self.num_query_tokens, :]

            # Generate different output representations
            outputs = {
                'last_hidden_state': hidden_states,
                'query_output': query_output,
                'text_embeds': self.text_projection(query_output.mean(dim=1)),
                'vision_embeds': self.vision_projection_out(query_output.mean(dim=1)),
                'multimodal_embeds': query_output,
                'attentions': all_attentions
            }

        elif mode == "text":
            # Text-only mode for language modeling
            embeddings = torch.cat([query_tokens, text_embeds], dim=1)

            query_mask = torch.ones(
                batch_size, self.num_query_tokens,
                dtype=torch.long, device=device
            )
            attention_mask = torch.cat([query_mask, text_attention_mask], dim=1)

            # Add position embeddings
            seq_length = embeddings.shape[1]
            position_ids = torch.arange(seq_length, device=device).expand(batch_size, -1)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings

            # Pass through layers without cross-attention
            hidden_states = embeddings
            for layer in self.layers:
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=False
                )
                hidden_states = layer_outputs[0]

            outputs = {
                'last_hidden_state': hidden_states,
                'query_output': hidden_states[:, :self.num_query_tokens, :],
                'text_embeds': self.text_projection(hidden_states[:, self.num_query_tokens:, :])
            }

        else:  # vision mode
            # Vision-only mode
            vision_embeds = self.vision_projection(vision_embeds)

            # Process queries with vision cross-attention
            hidden_states = query_tokens
            attention_mask = torch.ones(
                batch_size, self.num_query_tokens,
                dtype=torch.long, device=device
            )

            for layer in self.layers:
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=vision_embeds,
                    encoder_attention_mask=vision_attention_mask,
                    output_attentions=False
                )
                hidden_states = layer_outputs[0]

            outputs = {
                'last_hidden_state': hidden_states,
                'query_output': hidden_states,
                'vision_embeds': self.vision_projection_out(hidden_states.mean(dim=1))
            }

        return outputs

    def compute_contrastive_loss(
        self,
        text_embeds: torch.Tensor,
        vision_embeds: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """Compute cross-modal contrastive loss"""

        # Normalize embeddings
        text_embeds = F.normalize(text_embeds, dim=-1)
        vision_embeds = F.normalize(vision_embeds, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(text_embeds, vision_embeds.T) / temperature

        # Create labels
        batch_size = text_embeds.shape[0]
        labels = torch.arange(batch_size, device=text_embeds.device)

        # Compute loss in both directions
        text_to_vision_loss = F.cross_entropy(logits, labels)
        vision_to_text_loss = F.cross_entropy(logits.T, labels)

        return (text_to_vision_loss + vision_to_text_loss) / 2


class EnhancedCrossModalFusion(nn.Module):
    """
    Enhanced cross-modal fusion using QFormer
    Replaces the simple fusion in BitMar with more sophisticated alignment
    """

    def __init__(
        self,
        text_dim: int,
        vision_dim: int,
        hidden_dim: int = 768,
        num_query_tokens: int = 32,
        num_qformer_layers: int = 6,
        num_heads: int = 12
    ):
        super().__init__()

        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim

        # Input projections to common dimension
        self.text_input_projection = BitNetLinear(text_dim, hidden_dim)
        self.vision_input_projection = BitNetLinear(vision_dim, hidden_dim)

        # QFormer for cross-modal alignment
        self.qformer = BitNetQFormer(
            num_query_tokens=num_query_tokens,
            hidden_size=hidden_dim,
            num_hidden_layers=num_qformer_layers,
            num_attention_heads=num_heads,
            vision_feature_size=hidden_dim
        )

        # Output projections
        self.text_output_projection = BitNetLinear(hidden_dim, text_dim)
        self.vision_output_projection = BitNetLinear(hidden_dim, vision_dim)
        self.multimodal_projection = BitNetLinear(hidden_dim, hidden_dim)

    def forward(
        self,
        text_features: torch.Tensor,  # [batch_size, seq_len, text_dim]
        vision_features: torch.Tensor,  # [batch_size, vision_dim] or [batch_size, num_patches, vision_dim]
        text_attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = False
    ) -> Dict[str, torch.Tensor]:

        # Project to common dimension
        text_projected = self.text_input_projection(text_features)

        # Handle different vision feature shapes
        if vision_features.dim() == 2:  # [batch_size, vision_dim]
            vision_projected = self.vision_input_projection(vision_features).unsqueeze(1)
        else:  # [batch_size, num_patches, vision_dim]
            vision_projected = self.vision_input_projection(vision_features)

        # Apply QFormer for cross-modal alignment
        qformer_outputs = self.qformer(
            text_embeds=text_projected,
            text_attention_mask=text_attention_mask,
            vision_embeds=vision_projected,
            return_dict=True,
            mode="multimodal"
        )

        # Extract aligned representations
        multimodal_embeds = qformer_outputs['multimodal_embeds']  # Query tokens
        text_embeds = qformer_outputs['text_embeds']
        vision_embeds = qformer_outputs['vision_embeds']

        # Project back to original dimensions if needed
        aligned_text = self.text_output_projection(text_embeds)
        aligned_vision = self.vision_output_projection(vision_embeds)
        multimodal_features = self.multimodal_projection(multimodal_embeds.mean(dim=1))

        outputs = {
            'aligned_text_features': aligned_text,
            'aligned_vision_features': aligned_vision,
            'multimodal_features': multimodal_features,
            'query_tokens': multimodal_embeds,
            'attention_weights': qformer_outputs.get('attentions', [])
        }

        # Compute contrastive loss if requested
        if return_loss:
            contrastive_loss = self.qformer.compute_contrastive_loss(
                text_embeds, vision_embeds
            )
            outputs['contrastive_loss'] = contrastive_loss

        return outputs
