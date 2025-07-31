"""
Enhanced Wandb Logger for BitMar Model
Comprehensive logging with proper axis labels and visualization
"""

import wandb
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BitMarWandbLogger:
    """Enhanced Wandb logger for BitMar training with comprehensive metrics tracking"""

    def __init__(self, project_name: str, config: Dict[str, Any], run_name: Optional[str] = None):
        """Initialize wandb logger with project configuration

        Args:
            project_name: Name of the wandb project
            config: Training configuration dictionary
            run_name: Optional run name
        """
        self.project_name = project_name
        self.config = config
        self.step = 0

        # Initialize wandb run
        wandb.init(
            project=project_name,
            config=config,
            name=run_name,
            tags=["bitmar", "multimodal", "babylm"]
        )

        logger.info(f"Wandb logger initialized for project: {project_name}")

    def log_model_size_metrics(self, model: nn.Module):
        """Log model size and parameter count metrics"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Component-wise parameter counting
        component_params = {
            'text_encoder': 0,
            'vision_encoder': 0,
            'fusion': 0,
            'memory': 0,
            'text_decoder': 0,
            'projection': 0
        }

        for name, param in model.named_parameters():
            param_count = param.numel()

            if 'text_encoder' in name:
                component_params['text_encoder'] += param_count
            elif 'vision_encoder' in name:
                component_params['vision_encoder'] += param_count
            elif 'fusion' in name:
                component_params['fusion'] += param_count
            elif 'memory' in name:
                component_params['memory'] += param_count
            elif 'text_decoder' in name:
                component_params['text_decoder'] += param_count
            elif any(proj in name for proj in ['proj', 'to_episode', 'to_decoder']):
                component_params['projection'] += param_count

        metrics = {
            'Model/Total_Parameters': total_params,
            'Model/Trainable_Parameters': trainable_params,
            'Model/Non_Trainable_Parameters': total_params - trainable_params,
            'Model/Parameter_Efficiency': trainable_params / total_params if total_params > 0 else 0
        }

        # Log component-wise parameters
        for component, count in component_params.items():
            metrics[f'Model/{component.title()}_Parameters'] = count

        wandb.log(metrics)
        logger.info(f"Model size metrics logged: {total_params:,} total parameters")

    def log_training_metrics(self, loss: float, lr: float, step: int, epoch: int, **kwargs):
        """Log training metrics with proper categorization"""
        metrics = {
            'Training/Loss': loss,
            'Training/Learning_Rate': lr,
            'Training/Epoch': epoch,
            'step': step
        }

        # Add any additional training metrics
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                metrics[f'Training/{key}'] = value

        wandb.log(metrics, step=step)
        self.step = max(self.step, step)

    def log_gradient_metrics(self, model: nn.Module, step: int):
        """Log gradient norms and statistics"""
        total_norm = 0.0
        param_count = 0

        # Component-wise gradient tracking
        component_norms = {
            'text_encoder': 0.0,
            'vision_encoder': 0.0,
            'fusion': 0.0,
            'memory': 0.0,
            'text_decoder': 0.0,
            'projection': 0.0
        }
        component_counts = {key: 0 for key in component_norms.keys()}

        metrics = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1

                # Categorize by component
                component = 'projection'  # default
                if 'text_encoder' in name:
                    component = 'text_encoder'
                elif 'vision_encoder' in name:
                    component = 'vision_encoder'
                elif 'fusion' in name:
                    component = 'fusion'
                elif 'memory' in name:
                    component = 'memory'
                elif 'text_decoder' in name:
                    component = 'text_decoder'
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

        # Add step for consistency
        if step > self.step:
            metrics['step'] = step
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

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

        # Add step for consistency
        if step > self.step:
            metrics['step'] = step
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

    def log_memory_metrics(self, memory_usage: torch.Tensor, memory_entropy: float, step: int):
        """Log episodic memory usage metrics"""
        metrics = {
            'Memory/Usage_Mean': memory_usage.mean().item(),
            'Memory/Usage_Std': memory_usage.std().item(),
            'Memory/Usage_Max': memory_usage.max().item(),
            'Memory/Usage_Min': memory_usage.min().item(),
            'Memory/Entropy': memory_entropy,
        }

        if step > self.step:
            metrics['step'] = step
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

    def log_attention_metrics(self, attention_weights: torch.Tensor, step: int):
        """Log attention pattern metrics"""
        # Compute attention statistics
        attention_mean = attention_weights.mean().item()
        attention_std = attention_weights.std().item()
        attention_max = attention_weights.max().item()
        attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8)).item()

        metrics = {
            'Attention/Mean_Weight': attention_mean,
            'Attention/Std_Weight': attention_std,
            'Attention/Max_Weight': attention_max,
            'Attention/Entropy': attention_entropy,
        }

        if step > self.step:
            metrics['step'] = step
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

    def log_cross_modal_metrics(self, text_features: torch.Tensor, vision_features: torch.Tensor,
                               similarity: float, step: int):
        """Log cross-modal alignment metrics"""
        metrics = {
            'CrossModal/Similarity': similarity,
            'CrossModal/Text_Feature_Norm': torch.norm(text_features, dim=-1).mean().item(),
            'CrossModal/Vision_Feature_Norm': torch.norm(vision_features, dim=-1).mean().item(),
        }

        if step > self.step:
            metrics['step'] = step
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

    def log_consolidated_metrics(self, outputs: Dict[str, Any], epoch: int, step: int, lr: float,
                               model: nn.Module, memory_module: Optional[nn.Module] = None,
                               log_quantization: bool = False):
        """Log consolidated metrics from model outputs"""
        metrics = {
            'Training/Epoch': epoch,
            'Training/Learning_Rate': lr,
            'step': step
        }

        # Extract and log loss
        if 'loss' in outputs:
            metrics['Training/Loss'] = outputs['loss'].item()

        # Log memory metrics if available
        if memory_module is not None and hasattr(memory_module, 'memory_usage'):
            try:
                memory_usage = memory_module.memory_usage
                if memory_usage is not None:
                    metrics['Memory/Usage_Mean'] = memory_usage.mean().item()
                    metrics['Memory/Usage_Std'] = memory_usage.std().item()
            except Exception as e:
                logger.debug(f"Memory metrics logging failed: {e}")

        # Log cross-modal similarity if features are available
        if 'text_features' in outputs and 'vision_latent' in outputs:
            try:
                text_features = outputs['text_features']
                vision_features = outputs['vision_latent']

                if text_features is not None and vision_features is not None:
                    # Compute cosine similarity
                    text_pooled = text_features.mean(dim=1)
                    cos_sim = torch.cosine_similarity(text_pooled, vision_features, dim=1)
                    similarity = cos_sim.mean().item()

                    metrics['CrossModal/Similarity'] = similarity
                    metrics['CrossModal/Text_Feature_Norm'] = torch.norm(text_features, dim=-1).mean().item()
                    metrics['CrossModal/Vision_Feature_Norm'] = torch.norm(vision_features, dim=-1).mean().item()
            except Exception as e:
                logger.debug(f"Cross-modal metrics logging failed: {e}")

        # Log quantization metrics if requested
        if log_quantization and model is not None:
            try:
                self.log_quantization_info(model, step)
            except Exception as e:
                logger.debug(f"Quantization metrics logging failed: {e}")

        # Log all metrics
        wandb.log(metrics, step=step)
        self.step = max(self.step, step)

    def log_quantization_info(self, model: nn.Module, step: int):
        """Log quantization information for BitNet layers"""
        metrics = {}

        for name, module in model.named_modules():
            if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
                if 'int8' in str(module.weight.dtype) or 'int4' in str(module.weight.dtype):
                    # This is a quantized layer
                    weight_min = module.weight.min().item()
                    weight_max = module.weight.max().item()
                    weight_mean = module.weight.float().mean().item()

                    layer_type = name.split('.')[-1] if '.' in name else name
                    metrics[f'Quantization/{layer_type}_Min'] = weight_min
                    metrics[f'Quantization/{layer_type}_Max'] = weight_max
                    metrics[f'Quantization/{layer_type}_Mean'] = weight_mean

        if metrics:
            metrics['step'] = step
            wandb.log(metrics, step=step)

    def finish(self):
        """Finish the wandb run"""
        wandb.finish()
        logger.info("Wandb run finished")
