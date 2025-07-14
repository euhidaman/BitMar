"""
Enhanced Wandb Logger for BitMar Model
Comprehensive logging with proper axis labels and visualization
"""

import wandb
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import io
import base64
from PIL import Image

class BitMarWandbLogger:
    """Comprehensive wandb logging for BitMar model with detailed visualizations"""
    
    def __init__(self, project_name: str = "bitmar-babylm", config: Dict = None, run_name: str = None):
        self.project_name = project_name
        self.step = 0
        self.config = config or {}
        
        # Initialize wandb
        wandb.init(
            project=self.project_name,
            config=self.config,
            name=run_name or f"bitmar_{wandb.util.generate_id()}"
        )
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def log_training_metrics(self, outputs: Dict[str, torch.Tensor], epoch: int, step: int):
        """Log comprehensive training metrics from model outputs"""
        metrics = {}
        
        # Basic training metrics
        if 'loss' in outputs and outputs['loss'] is not None:
            metrics['Training/Loss'] = outputs['loss'].item()
        
        # Memory metrics with proper categorization
        if 'memory_usage' in outputs:
            memory_usage = outputs['memory_usage']
            metrics['Memory/Usage_Mean'] = memory_usage.mean().item()
            metrics['Memory/Usage_Max'] = memory_usage.max().item()
            metrics['Memory/Usage_Min'] = memory_usage.min().item()
            metrics['Memory/Usage_Std'] = memory_usage.std().item()
            
            # Memory utilization percentage
            active_slots = (memory_usage > 0).float().mean().item()
            metrics['Memory/Active_Slots_Percentage'] = active_slots * 100
            
        # Attention pattern analysis
        if 'cross_attention' in outputs:
            for layer_name, attention_weights in outputs['cross_attention'].items():
                avg_attention = attention_weights.mean().item()
                max_attention = attention_weights.max().item()
                entropy = self._compute_attention_entropy(attention_weights)
                
                metrics[f'Attention/CrossModal_{layer_name}_Mean'] = avg_attention
                metrics[f'Attention/CrossModal_{layer_name}_Max'] = max_attention
                metrics[f'Attention/CrossModal_{layer_name}_Entropy'] = entropy
                
        if 'memory_attention' in outputs and outputs['memory_attention'] is not None:
            memory_attn = outputs['memory_attention']
            metrics['Attention/Memory_Mean'] = memory_attn.mean().item()
            metrics['Attention/Memory_Max'] = memory_attn.max().item()
            metrics['Attention/Memory_Entropy'] = self._compute_attention_entropy(memory_attn)
            
            # Top-k memory slots being accessed
            top_k_indices = torch.topk(memory_attn.sum(0), k=5)[1]
            for i, idx in enumerate(top_k_indices):
                metrics[f'Memory/Top_{i+1}_Slot_Access'] = memory_attn[:, idx].mean().item()
            
        # Feature analysis
        if 'text_features' in outputs and outputs['text_features'] is not None:
            text_feat = outputs['text_features']
            metrics['Features/Text_Mean'] = text_feat.mean().item()
            metrics['Features/Text_Std'] = text_feat.std().item()
            metrics['Features/Text_Norm'] = torch.norm(text_feat, dim=-1).mean().item()
            
        if 'vision_latent' in outputs and outputs['vision_latent'] is not None:
            vision_feat = outputs['vision_latent']
            metrics['Features/Vision_Mean'] = vision_feat.mean().item()
            metrics['Features/Vision_Std'] = vision_feat.std().item()
            metrics['Features/Vision_Norm'] = torch.norm(vision_feat, dim=-1).mean().item()
            
        if 'episode' in outputs and outputs['episode'] is not None:
            episode = outputs['episode']
            metrics['Features/Episode_Mean'] = episode.mean().item()
            metrics['Features/Episode_Std'] = episode.std().item()
            metrics['Features/Episode_Norm'] = torch.norm(episode, dim=-1).mean().item()
            
        # Cross-modal similarity
        if 'text_features' in outputs and 'vision_latent' in outputs:
            if outputs['text_features'] is not None and outputs['vision_latent'] is not None:
                similarity = self._compute_cross_modal_similarity(
                    outputs['text_features'], outputs['vision_latent']
                )
                metrics['Features/CrossModal_Similarity'] = similarity
            
        # Add epoch and step info
        metrics['Training/Epoch'] = epoch
        metrics['Training/Step'] = step
        
        # Log to wandb
        wandb.log(metrics, step=step)
        self.step = step
        
    def log_quantization_metrics(self, model: nn.Module, step: int):
        """Log BitNet quantization statistics with proper categorization"""
        metrics = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'quantize_weights_1_58_bit'):  # BitNet layer
                module_name = name.replace('.', '_')
                
                # Weight scale statistics
                if hasattr(module, 'weight_scale'):
                    metrics[f'Quantization/WeightScale_{module_name}'] = module.weight_scale.item()
                    
                if hasattr(module, 'input_scale'):
                    metrics[f'Quantization/InputScale_{module_name}'] = module.input_scale.item()
                    
                # Weight distribution after quantization
                if hasattr(module, 'weight'):
                    weight = module.weight.data
                    quantized_weight = module.quantize_weights_1_58_bit(weight)
                    
                    # Count ternary values
                    total_weights = quantized_weight.numel()
                    zeros = (quantized_weight == 0).float().sum().item() / total_weights
                    ones = (quantized_weight == 1).float().sum().item() / total_weights
                    neg_ones = (quantized_weight == -1).float().sum().item() / total_weights
                    
                    metrics[f'Quantization/Zeros_Ratio_{module_name}'] = zeros
                    metrics[f'Quantization/Ones_Ratio_{module_name}'] = ones
                    metrics[f'Quantization/NegOnes_Ratio_{module_name}'] = neg_ones
                    
                    # Sparsity (zeros percentage)
                    metrics[f'Quantization/Sparsity_{module_name}'] = zeros * 100
        
        wandb.log(metrics, step=step)
        
    def log_memory_analysis(self, memory_module, step: int):
        """Log detailed episodic memory analysis"""
        metrics = {}
        
        if hasattr(memory_module, 'memory'):
            memory = memory_module.memory
            memory_age = memory_module.memory_age
            memory_usage = memory_module.memory_usage
            
            # Memory utilization
            active_slots = (memory_usage > 0).float().mean().item()
            metrics['Memory/Analysis_Active_Slots_Ratio'] = active_slots
            
            # Memory age distribution
            metrics['Memory/Analysis_Avg_Age'] = memory_age.mean().item()
            metrics['Memory/Analysis_Max_Age'] = memory_age.max().item()
            metrics['Memory/Analysis_Age_Std'] = memory_age.std().item()
            
            # Memory usage distribution
            metrics['Memory/Analysis_Usage_Mean'] = memory_usage.mean().item()
            metrics['Memory/Analysis_Usage_Max'] = memory_usage.max().item()
            
            # Memory similarity analysis
            if memory.numel() > 0:
                # Compute pairwise similarities for active slots
                active_memory = memory[memory_usage > 0]
                if active_memory.size(0) > 1:
                    normalized_memory = nn.functional.normalize(active_memory, dim=1)
                    similarity_matrix = torch.mm(normalized_memory, normalized_memory.t())
                    
                    # Remove diagonal (self-similarity)
                    mask = ~torch.eye(similarity_matrix.size(0), dtype=bool, device=memory.device)
                    if mask.any():
                        similarities = similarity_matrix[mask]
                        
                        metrics['Memory/Analysis_Avg_Similarity'] = similarities.mean().item()
                        metrics['Memory/Analysis_Max_Similarity'] = similarities.max().item()
                        metrics['Memory/Analysis_Similarity_Std'] = similarities.std().item()
        
        wandb.log(metrics, step=step)
        
    def log_learning_rate(self, lr: float, step: int):
        """Log learning rate with proper categorization"""
        wandb.log({'Training/Learning_Rate': lr}, step=step)
        
    def log_gradient_metrics(self, model: nn.Module, step: int):
        """Log gradient statistics with proper categorization"""
        metrics = {}
        
        total_norm = 0
        param_count = 0
        
        # Track gradients by component
        component_norms = {
            'encoder': 0,
            'decoder': 0,
            'fusion': 0,
            'memory': 0,
            'vision': 0,
            'projection': 0,
            'other': 0
        }
        component_counts = {k: 0 for k in component_norms.keys()}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
                
                # Categorize by component
                component = 'other'
                if 'text_encoder' in name:
                    component = 'encoder'
                elif 'text_decoder' in name:
                    component = 'decoder'
                elif 'fusion' in name:
                    component = 'fusion'
                elif 'memory' in name:
                    component = 'memory'
                elif 'vision' in name:
                    component = 'vision'
                elif any(proj in name for proj in ['proj', 'to_episode', 'to_decoder']):
                    component = 'projection'
                
                component_norms[component] += param_norm ** 2
                component_counts[component] += 1
        
        total_norm = total_norm ** 0.5
        metrics['Gradients/Total_Norm'] = total_norm
        metrics['Gradients/Avg_Norm'] = total_norm / max(param_count, 1)
        
        # Log component-wise gradients
        for component, norm in component_norms.items():
            if component_counts[component] > 0:
                component_norm = (norm ** 0.5) / component_counts[component]
                metrics[f'Gradients/{component.title()}_Norm'] = component_norm
        
        wandb.log(metrics, step=step)
        
    def log_validation_metrics(self, val_loss: float, perplexity: float, step: int, **kwargs):
        """Log validation metrics with proper categorization"""
        metrics = {
            'Validation/Loss': val_loss,
            'Validation/Perplexity': perplexity,
        }
        
        # Add any additional validation metrics
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                metrics[f'Validation/{key}'] = value
                
        wandb.log(metrics, step=step)
        
    def log_model_size_metrics(self, model: nn.Module):
        """Log model size and parameter statistics"""
        from model import count_parameters
        
        param_stats = count_parameters(model)
        
        metrics = {
            'Model/Total_Parameters': param_stats['total_parameters'],
            'Model/Trainable_Parameters': param_stats['trainable_parameters'],
            'Model/NonTrainable_Parameters': param_stats['non_trainable_parameters'],
        }
        
        # Estimate model size in MB
        param_size_mb = param_stats['total_parameters'] * 4 / (1024 * 1024)  # Assuming float32
        quantized_size_mb = self._estimate_quantized_size(model) / (1024 * 1024)
        
        metrics['Model/Size_FP32_MB'] = param_size_mb
        metrics['Model/Size_Quantized_MB'] = quantized_size_mb
        metrics['Model/Compression_Ratio'] = param_size_mb / (quantized_size_mb + 1e-8)
        
        wandb.log(metrics)
        
    def log_epoch_summary(self, epoch: int, train_loss: float, val_loss: float, 
                         memory_efficiency: float, step: int, **kwargs):
        """Log epoch summary metrics"""
        metrics = {
            'Epoch_Summary/Epoch': epoch,
            'Epoch_Summary/Train_Loss': train_loss,
            'Epoch_Summary/Val_Loss': val_loss,
            'Epoch_Summary/Memory_Efficiency': memory_efficiency,
        }
        
        # Add any additional epoch metrics
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                metrics[f'Epoch_Summary/{key}'] = value
                
        wandb.log(metrics, step=step)
        
    def create_memory_heatmap(self, memory_usage: torch.Tensor, memory_age: torch.Tensor, step: int):
        """Create and log memory usage heatmap"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Memory usage heatmap
        memory_2d = memory_usage.cpu().numpy().reshape(-1, int(np.sqrt(len(memory_usage))))
        im1 = ax1.imshow(memory_2d, cmap='viridis', aspect='auto')
        ax1.set_title('Memory Slot Usage')
        ax1.set_xlabel('Memory Slot (X)')
        ax1.set_ylabel('Memory Slot (Y)')
        plt.colorbar(im1, ax=ax1, label='Usage Count')
        
        # Memory age heatmap
        age_2d = memory_age.cpu().numpy().reshape(-1, int(np.sqrt(len(memory_age))))
        im2 = ax2.imshow(age_2d, cmap='plasma', aspect='auto')
        ax2.set_title('Memory Slot Age')
        ax2.set_xlabel('Memory Slot (X)')
        ax2.set_ylabel('Memory Slot (Y)')
        plt.colorbar(im2, ax=ax2, label='Age (Steps)')
        
        plt.tight_layout()
        wandb.log({"Memory/Usage_Age_Heatmap": wandb.Image(fig)}, step=step)
        plt.close(fig)
        
    def create_attention_distribution_plot(self, attention_weights: Dict[str, torch.Tensor], step: int):
        """Create attention distribution plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        for layer_name, weights in attention_weights.items():
            if plot_idx >= 4:
                break
                
            ax = axes[plot_idx]
            weights_np = weights[0].cpu().numpy().flatten()  # Take first batch item
            
            ax.hist(weights_np, bins=50, alpha=0.7, density=True)
            ax.set_title(f'Attention Distribution - {layer_name}')
            ax.set_xlabel('Attention Weight')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 4):
            axes[i].axis('off')
            
        plt.tight_layout()
        wandb.log({"Attention/Distribution_Plot": wandb.Image(fig)}, step=step)
        plt.close(fig)
        
    def create_quantization_plot(self, model: nn.Module, step: int):
        """Create quantization distribution plot"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        for name, module in model.named_modules():
            if hasattr(module, 'quantize_weights_1_58_bit') and plot_idx < 4:
                if hasattr(module, 'weight'):
                    weight = module.weight.data
                    quantized_weight = module.quantize_weights_1_58_bit(weight)
                    
                    ax = axes[plot_idx]
                    weights_np = quantized_weight.cpu().numpy().flatten()
                    
                    # Count occurrences
                    unique, counts = np.unique(weights_np, return_counts=True)
                    ax.bar(unique, counts, alpha=0.7)
                    ax.set_title(f'Quantized Weights - {name.split(".")[-2]}')
                    ax.set_xlabel('Weight Value')
                    ax.set_ylabel('Count')
                    ax.set_xticks([-1, 0, 1])
                    ax.grid(True, alpha=0.3)
                    
                    plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 4):
            axes[i].axis('off')
            
        plt.tight_layout()
        wandb.log({"Quantization/Weight_Distribution": wandb.Image(fig)}, step=step)
        plt.close(fig)
        
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention weights"""
        # Add small epsilon to avoid log(0)
        entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()
        return entropy.item()
        
    def _compute_cross_modal_similarity(self, text_features: torch.Tensor, vision_features: torch.Tensor) -> float:
        """Compute cosine similarity between text and vision features"""
        # Pool text features (mean over sequence)
        text_pooled = text_features.mean(dim=1)  # [batch_size, feature_dim]
        
        # Compute cosine similarity
        cos_sim = torch.cosine_similarity(text_pooled, vision_features, dim=1)
        return cos_sim.mean().item()
        
    def _estimate_quantized_size(self, model: nn.Module) -> float:
        """Estimate model size after quantization in bytes"""
        total_size = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                weight_numel = module.weight.numel()
                
                # Check if it's a BitNet layer
                if hasattr(module, 'quantize_weights_1_58_bit'):
                    # 1.58 bits per weight + scaling factors
                    total_size += weight_numel * 1.58 / 8  # Convert to bytes
                    total_size += 4  # 32-bit scaling factor
                else:
                    # Full precision
                    total_size += weight_numel * 4  # 32-bit floats
                    
        return total_size
        
    def finish(self):
        """Finish wandb run"""
        wandb.finish()
