"""
Comprehensive Modality Tracking System for BitMar
Tracks individual modality performance, cross-modal interactions, and attention patterns
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import os
from pathlib import Path
import json

class ModalityTracker:
    """Track individual modality performance and cross-modal interactions"""

    def __init__(self, save_dir: str = "./modality_analysis", wandb_logger=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.wandb_logger = wandb_logger

        # Initialize tracking metrics
        self.reset_metrics()

        # Create subdirectories
        (self.save_dir / "graphs").mkdir(exist_ok=True)
        (self.save_dir / "data").mkdir(exist_ok=True)
        (self.save_dir / "attention_maps").mkdir(exist_ok=True)

    def reset_metrics(self):
        """Reset all tracking metrics"""
        self.metrics_history = {
            'text_modality': {
                'loss': [],
                'perplexity': [],
                'attention_entropy': [],
                'gradient_norm': [],
                'learning_rate': [],
                'steps': []
            },
            'vision_modality': {
                'loss': [],
                'feature_variance': [],
                'compression_ratio': [],
                'attention_coverage': [],
                'gradient_norm': [],
                'steps': []
            },
            'cross_modal': {
                'similarity': [],
                'alignment_score': [],
                'fusion_entropy': [],
                'attention_balance': [],
                'information_flow': [],
                'steps': []
            },
            'attention_heads': {
                'total_heads': [],
                'active_heads': [],
                'specialized_heads': [],  # Heads that focus on specific modalities
                'generalist_heads': [],   # Heads that process both modalities
                'head_efficiency': [],    # How well heads are being utilized
                'steps': []
            },
            'memory_system': {
                'memory_usage': [],
                'memory_entropy': [],
                'retrieval_accuracy': [],
                'storage_efficiency': [],
                'steps': []
            }
        }

    def track_step(self, step: int, outputs: Dict[str, torch.Tensor],
                   model: nn.Module, batch: Dict[str, torch.Tensor]):
        """Track metrics for a single training step"""

        # Track text modality
        self._track_text_modality(step, outputs, model, batch)

        # Track vision modality
        self._track_vision_modality(step, outputs, model, batch)

        # Track cross-modal interactions
        self._track_cross_modal(step, outputs, model, batch)

        # Track attention heads
        self._track_attention_heads(step, outputs, model)

        # Track memory system
        self._track_memory_system(step, outputs, model)

    def _track_text_modality(self, step: int, outputs: Dict, model: nn.Module, batch: Dict):
        """Track text modality specific metrics"""
        try:
            # Text loss (if available separately)
            if 'text_loss' in outputs:
                text_loss = outputs['text_loss'].item()
            else:
                # Approximate from total loss
                text_loss = outputs.get('loss', torch.tensor(0.0)).item()

            self.metrics_history['text_modality']['loss'].append(text_loss)
            self.metrics_history['text_modality']['perplexity'].append(np.exp(min(text_loss, 10)))
            self.metrics_history['text_modality']['steps'].append(step)

            # Calculate text attention entropy
            text_attention_entropy = 0.0
            if hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'layers'):
                total_entropy = 0.0
                layer_count = 0

                for layer in model.text_encoder.layers:
                    if hasattr(layer, 'attn') and hasattr(layer.attn, 'attention_weights'):
                        attn_weights = layer.attn.attention_weights
                        if attn_weights is not None:
                            # Compute entropy
                            entropy = self._compute_attention_entropy(attn_weights)
                            total_entropy += entropy
                            layer_count += 1

                if layer_count > 0:
                    text_attention_entropy = total_entropy / layer_count

            self.metrics_history['text_modality']['attention_entropy'].append(text_attention_entropy)

            # Track gradient norms for text parameters
            text_grad_norm = 0.0
            text_param_count = 0

            for name, param in model.named_parameters():
                if 'text' in name.lower() and param.grad is not None:
                    text_grad_norm += param.grad.norm().item() ** 2
                    text_param_count += 1

            if text_param_count > 0:
                text_grad_norm = np.sqrt(text_grad_norm / text_param_count)

            self.metrics_history['text_modality']['gradient_norm'].append(text_grad_norm)

        except Exception as e:
            print(f"Error tracking text modality: {e}")

    def _track_vision_modality(self, step: int, outputs: Dict, model: nn.Module, batch: Dict):
        """Track vision modality specific metrics"""
        try:
            # Vision loss (if available separately)
            if 'vision_loss' in outputs:
                vision_loss = outputs['vision_loss'].item()
            else:
                # Approximate from reconstruction or feature matching
                vision_loss = 0.0

            self.metrics_history['vision_modality']['loss'].append(vision_loss)
            self.metrics_history['vision_modality']['steps'].append(step)

            # Feature variance (diversity of vision features)
            if 'vision_features' in outputs:
                vision_features = outputs['vision_features']
                feature_variance = torch.var(vision_features, dim=0).mean().item()
            elif 'vision_latent' in outputs:
                vision_latent = outputs['vision_latent']
                feature_variance = torch.var(vision_latent, dim=0).mean().item()
            else:
                feature_variance = 0.0

            self.metrics_history['vision_modality']['feature_variance'].append(feature_variance)

            # Compression ratio (how much vision info is compressed)
            compression_ratio = 1.0  # Default
            if 'vision_features' in batch and 'vision_latent' in outputs:
                original_dim = batch['vision_features'].shape[-1]
                compressed_dim = outputs['vision_latent'].shape[-1]
                compression_ratio = compressed_dim / original_dim

            self.metrics_history['vision_modality']['compression_ratio'].append(compression_ratio)

            # Vision attention coverage (how much of vision is attended to)
            attention_coverage = 0.0
            if 'cross_modal_attention' in outputs:
                cross_attn = outputs['cross_modal_attention']
                if isinstance(cross_attn, dict):
                    # Average attention coverage across layers
                    total_coverage = 0.0
                    layer_count = 0
                    for layer_name, attn_weights in cross_attn.items():
                        if attn_weights is not None and 'vision' in layer_name:
                            # Compute coverage as entropy of attention distribution
                            coverage = self._compute_attention_coverage(attn_weights)
                            total_coverage += coverage
                            layer_count += 1

                    if layer_count > 0:
                        attention_coverage = total_coverage / layer_count

            self.metrics_history['vision_modality']['attention_coverage'].append(attention_coverage)

            # Track gradient norms for vision parameters
            vision_grad_norm = 0.0
            vision_param_count = 0

            for name, param in model.named_parameters():
                if 'vision' in name.lower() and param.grad is not None:
                    vision_grad_norm += param.grad.norm().item() ** 2
                    vision_param_count += 1

            if vision_param_count > 0:
                vision_grad_norm = np.sqrt(vision_grad_norm / vision_param_count)

            self.metrics_history['vision_modality']['gradient_norm'].append(vision_grad_norm)

        except Exception as e:
            print(f"Error tracking vision modality: {e}")

    def _track_cross_modal(self, step: int, outputs: Dict, model: nn.Module, batch: Dict):
        """Track cross-modal interaction metrics"""
        try:
            self.metrics_history['cross_modal']['steps'].append(step)

            # Cross-modal similarity
            similarity = 0.0
            if 'text_features' in outputs and 'vision_latent' in outputs:
                text_feats = outputs['text_features'].mean(dim=1)  # Pool over sequence
                vision_feats = outputs['vision_latent']

                # Handle dimension mismatch
                if text_feats.shape[-1] != vision_feats.shape[-1]:
                    min_dim = min(text_feats.shape[-1], vision_feats.shape[-1])
                    text_feats = text_feats[:, :min_dim]
                    vision_feats = vision_feats[:, :min_dim]

                similarity = torch.cosine_similarity(text_feats, vision_feats, dim=1).mean().item()

            self.metrics_history['cross_modal']['similarity'].append(similarity)

            # Alignment score (how well modalities are aligned)
            alignment_score = abs(similarity)  # Higher absolute similarity = better alignment
            self.metrics_history['cross_modal']['alignment_score'].append(alignment_score)

            # Fusion entropy (diversity in cross-modal fusion)
            fusion_entropy = 0.0
            if 'cross_modal_attention' in outputs:
                cross_attn = outputs['cross_modal_attention']
                if isinstance(cross_attn, dict):
                    entropies = []
                    for layer_name, attn_weights in cross_attn.items():
                        if attn_weights is not None:
                            entropy = self._compute_attention_entropy(attn_weights)
                            entropies.append(entropy)

                    if entropies:
                        fusion_entropy = np.mean(entropies)

            self.metrics_history['cross_modal']['fusion_entropy'].append(fusion_entropy)

            # Attention balance (how balanced is attention between modalities)
            attention_balance = 0.5  # Perfect balance
            if 'cross_modal_attention' in outputs:
                cross_attn = outputs['cross_modal_attention']
                if isinstance(cross_attn, dict):
                    text_attention = 0.0
                    vision_attention = 0.0

                    for layer_name, attn_weights in cross_attn.items():
                        if attn_weights is not None:
                            if 'text' in layer_name or 't2v' in layer_name:
                                text_attention += attn_weights.mean().item()
                            elif 'vision' in layer_name or 'v2t' in layer_name:
                                vision_attention += attn_weights.mean().item()

                    total_attention = text_attention + vision_attention
                    if total_attention > 0:
                        attention_balance = min(text_attention, vision_attention) / total_attention

            self.metrics_history['cross_modal']['attention_balance'].append(attention_balance)

            # Information flow (bidirectional information transfer)
            information_flow = similarity * fusion_entropy  # Combine similarity and diversity
            self.metrics_history['cross_modal']['information_flow'].append(information_flow)

        except Exception as e:
            print(f"Error tracking cross-modal metrics: {e}")

    def _track_attention_heads(self, step: int, outputs: Dict, model: nn.Module):
        """Track attention head utilization and specialization"""
        try:
            self.metrics_history['attention_heads']['steps'].append(step)

            total_heads = 0
            active_heads = 0
            specialized_heads = 0
            generalist_heads = 0

            # Count heads in encoder
            if hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'layers'):
                for layer in model.text_encoder.layers:
                    if hasattr(layer, 'attn') and hasattr(layer.attn, 'num_heads'):
                        total_heads += layer.attn.num_heads

                        # Check head activity (simplified)
                        if hasattr(layer.attn, 'attention_weights'):
                            attn_weights = layer.attn.attention_weights
                            if attn_weights is not None:
                                # Count active heads (those with significant attention variance)
                                head_variances = torch.var(attn_weights, dim=-1).mean(dim=1)  # [batch, heads]
                                active_heads += (head_variances > 0.01).sum().item()

                                # Simplified specialization detection
                                # Heads with high variance are considered specialized
                                specialized_heads += (head_variances > 0.1).sum().item()
                                generalist_heads += (head_variances <= 0.1).sum().item()

            # Count heads in decoder
            if hasattr(model, 'text_decoder') and hasattr(model.text_decoder, 'layers'):
                for layer in model.text_decoder.layers:
                    if hasattr(layer, 'attn') and hasattr(layer.attn, 'num_heads'):
                        total_heads += layer.attn.num_heads

            # Count heads in fusion layers
            if hasattr(model, 'fusion'):
                if hasattr(model.fusion, 'query_layers'):
                    for layer in model.fusion.query_layers:
                        if hasattr(layer, 'q2t_attention') and hasattr(layer.q2t_attention, 'num_heads'):
                            total_heads += layer.q2t_attention.num_heads
                        if hasattr(layer, 'q2v_attention') and hasattr(layer.q2v_attention, 'num_heads'):
                            total_heads += layer.q2v_attention.num_heads
                        if hasattr(layer, 'self_attention') and hasattr(layer.self_attention, 'num_heads'):
                            total_heads += layer.self_attention.num_heads

                if hasattr(model.fusion, 'text2query_layers'):
                    for layer in model.fusion.text2query_layers:
                        if hasattr(layer, 'attention') and hasattr(layer.attention, 'num_heads'):
                            total_heads += layer.attention.num_heads

            self.metrics_history['attention_heads']['total_heads'].append(total_heads)
            self.metrics_history['attention_heads']['active_heads'].append(active_heads)
            self.metrics_history['attention_heads']['specialized_heads'].append(specialized_heads)
            self.metrics_history['attention_heads']['generalist_heads'].append(generalist_heads)

            # Head efficiency (ratio of active to total heads)
            head_efficiency = active_heads / max(total_heads, 1)
            self.metrics_history['attention_heads']['head_efficiency'].append(head_efficiency)

        except Exception as e:
            print(f"Error tracking attention heads: {e}")

    def _track_memory_system(self, step: int, outputs: Dict, model: nn.Module):
        """Track episodic memory system performance"""
        try:
            self.metrics_history['memory_system']['steps'].append(step)

            memory_usage = 0.0
            memory_entropy = 0.0
            retrieval_accuracy = 0.0
            storage_efficiency = 0.0

            if hasattr(model, 'memory') and outputs.get('memory_usage') is not None:
                memory_usage_tensor = outputs['memory_usage']
                memory_usage = memory_usage_tensor.mean().item()

                # Memory entropy (how evenly memory is used)
                if memory_usage_tensor.numel() > 0:
                    probs = memory_usage_tensor / (memory_usage_tensor.sum() + 1e-8)
                    log_probs = torch.log(probs + 1e-8)
                    memory_entropy = -(probs * log_probs).sum().item()

                # Storage efficiency (how much unique information is stored)
                if hasattr(model.memory, 'memory'):
                    memory_states = model.memory.memory
                    if memory_states.numel() > 0:
                        # Compute pairwise similarities to measure uniqueness
                        similarities = torch.cosine_similarity(
                            memory_states.unsqueeze(1),
                            memory_states.unsqueeze(0),
                            dim=2
                        )
                        # Storage efficiency = 1 - average similarity (excluding diagonal)
                        mask = ~torch.eye(similarities.size(0), dtype=torch.bool, device=similarities.device)
                        avg_similarity = similarities[mask].mean().item()
                        storage_efficiency = 1.0 - avg_similarity

                # Retrieval accuracy (simplified - based on attention weights)
                if 'memory_attention' in outputs:
                    memory_attn = outputs['memory_attention']
                    if memory_attn is not None:
                        # Higher attention variance indicates better retrieval precision
                        retrieval_accuracy = torch.var(memory_attn, dim=-1).mean().item()

            self.metrics_history['memory_system']['memory_usage'].append(memory_usage)
            self.metrics_history['memory_system']['memory_entropy'].append(memory_entropy)
            self.metrics_history['memory_system']['retrieval_accuracy'].append(retrieval_accuracy)
            self.metrics_history['memory_system']['storage_efficiency'].append(storage_efficiency)

        except Exception as e:
            print(f"Error tracking memory system: {e}")

    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention weights"""
        try:
            if attention_weights.numel() == 0:
                return 0.0

            # Flatten and normalize
            attn_flat = attention_weights.flatten()
            attn_probs = torch.softmax(attn_flat, dim=0)

            # Compute entropy
            log_probs = torch.log(attn_probs + 1e-8)
            entropy = -(attn_probs * log_probs).sum().item()

            return entropy if np.isfinite(entropy) else 0.0

        except Exception:
            return 0.0

    def _compute_attention_coverage(self, attention_weights: torch.Tensor) -> float:
        """Compute attention coverage (how spread out attention is)"""
        try:
            if attention_weights.numel() == 0:
                return 0.0

            # Compute variance as a measure of coverage
            coverage = torch.var(attention_weights, dim=-1).mean().item()
            return coverage if np.isfinite(coverage) else 0.0

        except Exception:
            return 0.0

    def create_modality_graphs(self, step: int):
        """Create comprehensive modality tracking graphs"""

        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'BitMar Modality Analysis - Step {step}', fontsize=16, fontweight='bold')

        # 1. Text Modality Performance
        ax = axes[0, 0]
        if self.metrics_history['text_modality']['steps']:
            steps = self.metrics_history['text_modality']['steps']
            losses = self.metrics_history['text_modality']['loss']
            perplexities = self.metrics_history['text_modality']['perplexity']

            ax.plot(steps, losses, 'b-', label='Text Loss', linewidth=2)
            ax2 = ax.twinx()
            ax2.plot(steps, perplexities, 'r--', label='Perplexity', linewidth=2)

            ax.set_xlabel('Training Step')
            ax.set_ylabel('Text Loss', color='b')
            ax2.set_ylabel('Perplexity', color='r')
            ax.set_title('Text Modality Performance')
            ax.grid(True, alpha=0.3)

            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # 2. Vision Modality Performance
        ax = axes[0, 1]
        if self.metrics_history['vision_modality']['steps']:
            steps = self.metrics_history['vision_modality']['steps']
            feature_var = self.metrics_history['vision_modality']['feature_variance']
            compression = self.metrics_history['vision_modality']['compression_ratio']

            ax.plot(steps, feature_var, 'g-', label='Feature Variance', linewidth=2)
            ax2 = ax.twinx()
            ax2.plot(steps, compression, 'm--', label='Compression Ratio', linewidth=2)

            ax.set_xlabel('Training Step')
            ax.set_ylabel('Feature Variance', color='g')
            ax2.set_ylabel('Compression Ratio', color='m')
            ax.set_title('Vision Modality Performance')
            ax.grid(True, alpha=0.3)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # 3. Cross-Modal Similarity
        ax = axes[0, 2]
        if self.metrics_history['cross_modal']['steps']:
            steps = self.metrics_history['cross_modal']['steps']
            similarity = self.metrics_history['cross_modal']['similarity']
            alignment = self.metrics_history['cross_modal']['alignment_score']

            ax.plot(steps, similarity, 'purple', label='Cross-Modal Similarity', linewidth=2)
            ax.plot(steps, alignment, 'orange', label='Alignment Score', linewidth=2)

            ax.set_xlabel('Training Step')
            ax.set_ylabel('Similarity/Alignment')
            ax.set_title('Cross-Modal Interaction')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1.1, 1.1)

        # 4. Attention Head Utilization
        ax = axes[1, 0]
        if self.metrics_history['attention_heads']['steps']:
            steps = self.metrics_history['attention_heads']['steps']
            total_heads = self.metrics_history['attention_heads']['total_heads']
            active_heads = self.metrics_history['attention_heads']['active_heads']
            specialized = self.metrics_history['attention_heads']['specialized_heads']

            ax.plot(steps, total_heads, 'k-', label='Total Heads', linewidth=2)
            ax.plot(steps, active_heads, 'b-', label='Active Heads', linewidth=2)
            ax.plot(steps, specialized, 'r-', label='Specialized Heads', linewidth=2)

            ax.set_xlabel('Training Step')
            ax.set_ylabel('Number of Heads')
            ax.set_title('Attention Head Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 5. Memory System Performance
        ax = axes[1, 1]
        if self.metrics_history['memory_system']['steps']:
            steps = self.metrics_history['memory_system']['steps']
            memory_usage = self.metrics_history['memory_system']['memory_usage']
            memory_entropy = self.metrics_history['memory_system']['memory_entropy']

            ax.plot(steps, memory_usage, 'cyan', label='Memory Usage', linewidth=2)
            ax2 = ax.twinx()
            ax2.plot(steps, memory_entropy, 'brown', label='Memory Entropy', linewidth=2)

            ax.set_xlabel('Training Step')
            ax.set_ylabel('Memory Usage', color='cyan')
            ax2.set_ylabel('Memory Entropy', color='brown')
            ax.set_title('Episodic Memory System')
            ax.grid(True, alpha=0.3)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # 6. Gradient Norms Comparison
        ax = axes[1, 2]
        if (self.metrics_history['text_modality']['steps'] and
            self.metrics_history['vision_modality']['steps']):

            text_steps = self.metrics_history['text_modality']['steps']
            text_grads = self.metrics_history['text_modality']['gradient_norm']
            vision_steps = self.metrics_history['vision_modality']['steps']
            vision_grads = self.metrics_history['vision_modality']['gradient_norm']

            ax.plot(text_steps, text_grads, 'b-', label='Text Gradients', linewidth=2)
            ax.plot(vision_steps, vision_grads, 'g-', label='Vision Gradients', linewidth=2)

            ax.set_xlabel('Training Step')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Modality Gradient Norms')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        # 7. Cross-Modal Information Flow
        ax = axes[2, 0]
        if self.metrics_history['cross_modal']['steps']:
            steps = self.metrics_history['cross_modal']['steps']
            info_flow = self.metrics_history['cross_modal']['information_flow']
            fusion_entropy = self.metrics_history['cross_modal']['fusion_entropy']

            ax.plot(steps, info_flow, 'red', label='Information Flow', linewidth=2)
            ax.plot(steps, fusion_entropy, 'blue', label='Fusion Entropy', linewidth=2)

            ax.set_xlabel('Training Step')
            ax.set_ylabel('Information Flow')
            ax.set_title('Cross-Modal Information Transfer')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 8. Attention Balance
        ax = axes[2, 1]
        if self.metrics_history['cross_modal']['steps']:
            steps = self.metrics_history['cross_modal']['steps']
            balance = self.metrics_history['cross_modal']['attention_balance']

            ax.plot(steps, balance, 'purple', linewidth=2)
            ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Perfect Balance')

            ax.set_xlabel('Training Step')
            ax.set_ylabel('Attention Balance')
            ax.set_title('Cross-Modal Attention Balance')
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 9. Head Efficiency Over Time
        ax = axes[2, 2]
        if self.metrics_history['attention_heads']['steps']:
            steps = self.metrics_history['attention_heads']['steps']
            efficiency = self.metrics_history['attention_heads']['head_efficiency']

            ax.plot(steps, efficiency, 'orange', linewidth=2)
            ax.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Good Efficiency')
            ax.axhline(y=0.6, color='y', linestyle='--', alpha=0.5, label='Fair Efficiency')

            ax.set_xlabel('Training Step')
            ax.set_ylabel('Head Efficiency')
            ax.set_title('Attention Head Efficiency')
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save graph
        save_path = self.save_dir / "graphs" / f"modality_analysis_step_{step}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # Log to wandb if available
        if self.wandb_logger:
            wandb.log({f"Modality_Analysis/Comprehensive_Tracking": wandb.Image(fig)}, step=step)

        plt.close(fig)

        return save_path

    def save_metrics_data(self, step: int):
        """Save metrics data to JSON for later analysis"""
        data_path = self.save_dir / "data" / f"metrics_step_{step}.json"

        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for modality, metrics in self.metrics_history.items():
            serializable_metrics[modality] = {}
            for metric_name, values in metrics.items():
                if isinstance(values, list):
                    serializable_metrics[modality][metric_name] = values
                else:
                    serializable_metrics[modality][metric_name] = list(values)

        with open(data_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        return data_path

    def generate_summary_report(self, step: int) -> Dict[str, Any]:
        """Generate a summary report of all modality metrics"""
        report = {
            'step': step,
            'timestamp': str(step),
            'summary': {}
        }

        # Text modality summary
        if self.metrics_history['text_modality']['loss']:
            report['summary']['text_modality'] = {
                'latest_loss': self.metrics_history['text_modality']['loss'][-1],
                'latest_perplexity': self.metrics_history['text_modality']['perplexity'][-1],
                'avg_loss': np.mean(self.metrics_history['text_modality']['loss'][-10:]),
                'loss_trend': 'decreasing' if len(self.metrics_history['text_modality']['loss']) > 1 and
                             self.metrics_history['text_modality']['loss'][-1] < self.metrics_history['text_modality']['loss'][-2]
                             else 'increasing'
            }

        # Vision modality summary
        if self.metrics_history['vision_modality']['feature_variance']:
            report['summary']['vision_modality'] = {
                'latest_feature_variance': self.metrics_history['vision_modality']['feature_variance'][-1],
                'latest_compression_ratio': self.metrics_history['vision_modality']['compression_ratio'][-1],
                'avg_coverage': np.mean(self.metrics_history['vision_modality']['attention_coverage'][-10:])
                if self.metrics_history['vision_modality']['attention_coverage'] else 0.0
            }

        # Cross-modal summary
        if self.metrics_history['cross_modal']['similarity']:
            report['summary']['cross_modal'] = {
                'latest_similarity': self.metrics_history['cross_modal']['similarity'][-1],
                'latest_alignment': self.metrics_history['cross_modal']['alignment_score'][-1],
                'avg_information_flow': np.mean(self.metrics_history['cross_modal']['information_flow'][-10:])
                if self.metrics_history['cross_modal']['information_flow'] else 0.0
            }

        # Attention heads summary
        if self.metrics_history['attention_heads']['total_heads']:
            report['summary']['attention_heads'] = {
                'total_heads': self.metrics_history['attention_heads']['total_heads'][-1],
                'active_heads': self.metrics_history['attention_heads']['active_heads'][-1],
                'head_efficiency': self.metrics_history['attention_heads']['head_efficiency'][-1],
                'specialization_ratio': (self.metrics_history['attention_heads']['specialized_heads'][-1] /
                                       max(self.metrics_history['attention_heads']['total_heads'][-1], 1))
            }

        # Memory system summary
        if self.metrics_history['memory_system']['memory_usage']:
            report['summary']['memory_system'] = {
                'memory_usage': self.metrics_history['memory_system']['memory_usage'][-1],
                'memory_entropy': self.metrics_history['memory_system']['memory_entropy'][-1],
                'storage_efficiency': self.metrics_history['memory_system']['storage_efficiency'][-1]
            }

        return report
