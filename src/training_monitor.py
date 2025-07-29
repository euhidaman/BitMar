"""
Training Monitor for BitMar
Provides comprehensive monitoring of BitNet quantization, episodic memory, and QFormer alignment
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import time

logger = logging.getLogger(__name__)


class BitMarTrainingMonitor:
    """Comprehensive training monitor for BitMar model"""

    def __init__(self, config: Dict, log_dir: str = "logs"):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Initialize tracking dictionaries
        self.metrics_history = defaultdict(list)
        self.quantization_stats = defaultdict(list)
        self.memory_stats = defaultdict(list)
        self.loss_components = defaultdict(list)
        self.attention_patterns = defaultdict(list)

        # Moving averages for stability
        self.moving_averages = defaultdict(lambda: deque(maxlen=100))

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_metrics = {}

        # Performance tracking
        self.timing_stats = defaultdict(list)

    def log_step(self, model, outputs: Dict[str, Any], batch_time: float = 0.0):
        """Log metrics for a single training step"""
        self.step += 1

        # Basic loss metrics
        if 'loss' in outputs and outputs['loss'] is not None:
            loss_val = outputs['loss'].item()
            self.metrics_history['total_loss'].append(loss_val)
            self.moving_averages['total_loss'].append(loss_val)

        # Loss component breakdown
        loss_components = ['decoder_loss', 'cross_modal_loss', 'vision_loss',
                          'memory_loss', 'quantization_loss']
        for component in loss_components:
            if component in outputs and outputs[component] is not None:
                val = outputs[component].item() if torch.is_tensor(outputs[component]) else outputs[component]
                self.loss_components[component].append(val)
                self.moving_averages[component].append(val)

        # Loss weights and scheduling
        weight_components = ['cross_modal_weight', 'memory_weight', 'adaptive_weight',
                           'cross_modal_schedule', 'memory_schedule']
        for component in weight_components:
            if component in outputs and outputs[component] is not None:
                val = outputs[component].item() if torch.is_tensor(outputs[component]) else outputs[component]
                self.metrics_history[component].append(val)

        # BitNet quantization statistics
        self._log_quantization_stats(model)

        # Episodic memory statistics
        self._log_memory_stats(model, outputs)

        # Attention pattern analysis
        self._log_attention_patterns(outputs)

        # Performance metrics
        if batch_time > 0:
            self.timing_stats['batch_time'].append(batch_time)
            self.timing_stats['samples_per_second'].append(
                self.config.get('batch_size', 1) / batch_time
            )

        # Periodic logging
        if self.step % 100 == 0:
            self._print_status()

        # Periodic saving
        if self.step % 1000 == 0:
            self.save_metrics()

    def _log_quantization_stats(self, model):
        """Log BitNet quantization statistics"""
        total_stats = {
            'weight_scale': [],
            'input_scale': [],
            'gradient_norm': [],
            'weight_neg1_ratio': [],
            'weight_zero_ratio': [],
            'weight_pos1_ratio': []
        }

        # Collect stats from all BitNet layers
        for name, module in model.named_modules():
            if hasattr(module, 'get_quantization_stats'):
                stats = module.get_quantization_stats()
                for key, value in stats.items():
                    if key in total_stats:
                        total_stats[key].append(value)

        # Aggregate statistics
        for key, values in total_stats.items():
            if values:
                avg_val = np.mean(values)
                self.quantization_stats[f'{key}_mean'].append(avg_val)
                self.quantization_stats[f'{key}_std'].append(np.std(values))
                self.moving_averages[f'quant_{key}'].append(avg_val)

    def _log_memory_stats(self, model, outputs):
        """Log episodic memory statistics"""
        if hasattr(model, 'memory') and hasattr(model.memory, 'memory_usage'):
            memory_usage = model.memory.memory_usage.cpu().numpy()

            # Memory utilization metrics
            self.memory_stats['memory_utilization_mean'].append(memory_usage.mean())
            self.memory_stats['memory_utilization_max'].append(memory_usage.max())
            self.memory_stats['memory_utilization_std'].append(memory_usage.std())

            # Memory distribution
            used_slots = (memory_usage > 0).sum()
            total_slots = len(memory_usage)
            utilization_ratio = used_slots / total_slots
            self.memory_stats['memory_slots_used'].append(used_slots)
            self.memory_stats['memory_utilization_ratio'].append(utilization_ratio)

            # Memory attention patterns
            if 'memory_attention' in outputs and outputs['memory_attention'] is not None:
                attn_weights = outputs['memory_attention'].cpu().numpy()
                self.memory_stats['memory_attention_entropy'].append(
                    self._compute_attention_entropy(attn_weights)
                )

    def _log_attention_patterns(self, outputs):
        """Log attention pattern statistics"""
        attention_keys = ['cross_attention', 'text_attention', 'memory_attention']

        for key in attention_keys:
            if key in outputs and outputs[key] is not None:
                if isinstance(outputs[key], dict):
                    # Multiple attention layers
                    for layer_name, attn_weights in outputs[key].items():
                        if torch.is_tensor(attn_weights):
                            entropy = self._compute_attention_entropy(attn_weights.cpu().numpy())
                            self.attention_patterns[f'{key}_{layer_name}_entropy'].append(entropy)
                elif torch.is_tensor(outputs[key]):
                    # Single attention matrix
                    entropy = self._compute_attention_entropy(outputs[key].cpu().numpy())
                    self.attention_patterns[f'{key}_entropy'].append(entropy)

    def _compute_attention_entropy(self, attention_weights: np.ndarray) -> float:
        """Compute entropy of attention weights"""
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        attention_weights = attention_weights + eps

        # Normalize to ensure proper probability distribution
        attention_weights = attention_weights / attention_weights.sum(axis=-1, keepdims=True)

        # Compute entropy
        entropy = -np.sum(attention_weights * np.log(attention_weights), axis=-1)
        return float(entropy.mean())

    def _print_status(self):
        """Print current training status"""
        if not self.moving_averages['total_loss']:
            return

        avg_loss = np.mean(list(self.moving_averages['total_loss']))

        status_parts = [f"Step {self.step:,}"]
        status_parts.append(f"Loss: {avg_loss:.4f}")

        # Add component losses if available
        if self.moving_averages['decoder_loss']:
            decoder_loss = np.mean(list(self.moving_averages['decoder_loss']))
            status_parts.append(f"Decoder: {decoder_loss:.4f}")

        if self.moving_averages['cross_modal_loss']:
            cm_loss = np.mean(list(self.moving_averages['cross_modal_loss']))
            status_parts.append(f"CrossModal: {cm_loss:.4f}")

        # Add quantization stats
        if self.moving_averages['quant_weight_zero_ratio']:
            zero_ratio = np.mean(list(self.moving_averages['quant_weight_zero_ratio']))
            status_parts.append(f"ZeroWeights: {zero_ratio:.2f}")

        # Add memory utilization
        if self.memory_stats['memory_utilization_ratio']:
            mem_util = self.memory_stats['memory_utilization_ratio'][-1]
            status_parts.append(f"MemUtil: {mem_util:.2f}")

        # Add timing info
        if self.timing_stats['samples_per_second']:
            sps = np.mean(self.timing_stats['samples_per_second'][-10:])  # Last 10 batches
            status_parts.append(f"SPS: {sps:.1f}")

        logger.info(" | ".join(status_parts))

    def log_epoch(self, epoch: int, validation_metrics: Optional[Dict] = None):
        """Log epoch-level metrics"""
        self.epoch = epoch

        if validation_metrics:
            for key, value in validation_metrics.items():
                self.metrics_history[f'val_{key}'].append(value)

                # Track best metrics
                if key not in self.best_metrics or value > self.best_metrics[key]:
                    self.best_metrics[key] = value
                    logger.info(f"New best {key}: {value:.4f}")

    def save_metrics(self):
        """Save all metrics to disk"""
        metrics_file = self.log_dir / f"training_metrics_step_{self.step}.json"

        # Prepare data for JSON serialization
        save_data = {
            'step': self.step,
            'epoch': self.epoch,
            'metrics_history': dict(self.metrics_history),
            'quantization_stats': dict(self.quantization_stats),
            'memory_stats': dict(self.memory_stats),
            'loss_components': dict(self.loss_components),
            'attention_patterns': dict(self.attention_patterns),
            'timing_stats': dict(self.timing_stats),
            'best_metrics': self.best_metrics
        }

        with open(metrics_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        logger.info(f"Saved metrics to {metrics_file}")

    def load_metrics(self, metrics_file: str):
        """Load metrics from disk"""
        with open(metrics_file, 'r') as f:
            data = json.load(f)

        self.step = data.get('step', 0)
        self.epoch = data.get('epoch', 0)
        self.metrics_history = defaultdict(list, data.get('metrics_history', {}))
        self.quantization_stats = defaultdict(list, data.get('quantization_stats', {}))
        self.memory_stats = defaultdict(list, data.get('memory_stats', {}))
        self.loss_components = defaultdict(list, data.get('loss_components', {}))
        self.attention_patterns = defaultdict(list, data.get('attention_patterns', {}))
        self.timing_stats = defaultdict(list, data.get('timing_stats', {}))
        self.best_metrics = data.get('best_metrics', {})

        logger.info(f"Loaded metrics from {metrics_file}")

    def plot_training_curves(self, save_path: Optional[str] = None):
        """Generate training curve plots"""
        if not save_path:
            save_path = self.log_dir / "training_curves.png"

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('BitMar Training Progress', fontsize=16)

        # Loss curves
        ax = axes[0, 0]
        if self.metrics_history['total_loss']:
            ax.plot(self.metrics_history['total_loss'], label='Total Loss', alpha=0.7)
        if self.loss_components['decoder_loss']:
            ax.plot(self.loss_components['decoder_loss'], label='Decoder Loss', alpha=0.7)
        if self.loss_components['cross_modal_loss']:
            ax.plot(self.loss_components['cross_modal_loss'], label='Cross-Modal Loss', alpha=0.7)
        ax.set_title('Loss Components')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Quantization statistics
        ax = axes[0, 1]
        if self.quantization_stats['weight_zero_ratio_mean']:
            ax.plot(self.quantization_stats['weight_zero_ratio_mean'], label='Zero Weights', alpha=0.7)
        if self.quantization_stats['weight_pos1_ratio_mean']:
            ax.plot(self.quantization_stats['weight_pos1_ratio_mean'], label='+1 Weights', alpha=0.7)
        if self.quantization_stats['weight_neg1_ratio_mean']:
            ax.plot(self.quantization_stats['weight_neg1_ratio_mean'], label='-1 Weights', alpha=0.7)
        ax.set_title('BitNet Weight Distribution')
        ax.set_xlabel('Step')
        ax.set_ylabel('Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Memory utilization
        ax = axes[0, 2]
        if self.memory_stats['memory_utilization_ratio']:
            ax.plot(self.memory_stats['memory_utilization_ratio'], label='Memory Utilization', alpha=0.7)
        if self.memory_stats['memory_utilization_mean']:
            ax.plot(self.memory_stats['memory_utilization_mean'], label='Avg Usage', alpha=0.7)
        ax.set_title('Episodic Memory Usage')
        ax.set_xlabel('Step')
        ax.set_ylabel('Utilization')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Loss weights evolution
        ax = axes[1, 0]
        if self.metrics_history['cross_modal_weight']:
            ax.plot(self.metrics_history['cross_modal_weight'], label='Cross-Modal Weight', alpha=0.7)
        if self.metrics_history['memory_weight']:
            ax.plot(self.metrics_history['memory_weight'], label='Memory Weight', alpha=0.7)
        if self.metrics_history['adaptive_weight']:
            ax.plot(self.metrics_history['adaptive_weight'], label='Adaptive Weight', alpha=0.7)
        ax.set_title('Loss Weight Scheduling')
        ax.set_xlabel('Step')
        ax.set_ylabel('Weight')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Attention entropy
        ax = axes[1, 1]
        for key, values in self.attention_patterns.items():
            if 'entropy' in key and values:
                ax.plot(values, label=key.replace('_entropy', ''), alpha=0.7)
        ax.set_title('Attention Entropy')
        ax.set_xlabel('Step')
        ax.set_ylabel('Entropy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Performance metrics
        ax = axes[1, 2]
        if self.timing_stats['samples_per_second']:
            ax.plot(self.timing_stats['samples_per_second'], label='Samples/Second', alpha=0.7)
        ax.set_title('Training Performance')
        ax.set_xlabel('Step')
        ax.set_ylabel('Samples/Second')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved training curves to {save_path}")

    def generate_report(self) -> str:
        """Generate a comprehensive training report"""
        report_lines = []
        report_lines.append("=== BitMar Training Report ===")
        report_lines.append(f"Training Step: {self.step:,}")
        report_lines.append(f"Epoch: {self.epoch}")
        report_lines.append("")

        # Loss summary
        if self.moving_averages['total_loss']:
            avg_loss = np.mean(list(self.moving_averages['total_loss']))
            report_lines.append(f"Current Average Loss: {avg_loss:.4f}")

        # Component losses
        if self.moving_averages['decoder_loss']:
            decoder_loss = np.mean(list(self.moving_averages['decoder_loss']))
            report_lines.append(f"  - Decoder Loss: {decoder_loss:.4f}")

        if self.moving_averages['cross_modal_loss']:
            cm_loss = np.mean(list(self.moving_averages['cross_modal_loss']))
            report_lines.append(f"  - Cross-Modal Loss: {cm_loss:.4f}")

        report_lines.append("")

        # BitNet quantization status
        report_lines.append("BitNet Quantization Status:")
        if self.quantization_stats['weight_zero_ratio_mean']:
            zero_ratio = self.quantization_stats['weight_zero_ratio_mean'][-1]
            pos_ratio = self.quantization_stats['weight_pos1_ratio_mean'][-1]
            neg_ratio = self.quantization_stats['weight_neg1_ratio_mean'][-1]

            report_lines.append(f"  - Zero weights: {zero_ratio:.3f}")
            report_lines.append(f"  - Positive weights: {pos_ratio:.3f}")
            report_lines.append(f"  - Negative weights: {neg_ratio:.3f}")

        report_lines.append("")

        # Memory utilization
        report_lines.append("Episodic Memory Status:")
        if self.memory_stats['memory_utilization_ratio']:
            mem_util = self.memory_stats['memory_utilization_ratio'][-1]
            mem_slots = self.memory_stats['memory_slots_used'][-1]
            report_lines.append(f"  - Memory utilization: {mem_util:.3f}")
            report_lines.append(f"  - Active memory slots: {mem_slots}")

        report_lines.append("")

        # Performance summary
        if self.timing_stats['samples_per_second']:
            avg_sps = np.mean(self.timing_stats['samples_per_second'][-100:])
            report_lines.append(f"Training Speed: {avg_sps:.1f} samples/second")

        # Best metrics
        if self.best_metrics:
            report_lines.append("")
            report_lines.append("Best Validation Metrics:")
            for metric, value in self.best_metrics.items():
                report_lines.append(f"  - {metric}: {value:.4f}")

        return "\n".join(report_lines)

    def get_current_stats(self) -> Dict[str, float]:
        """Get current training statistics"""
        stats = {}

        # Current losses
        if self.moving_averages['total_loss']:
            stats['current_loss'] = np.mean(list(self.moving_averages['total_loss']))

        # Quantization stats
        if self.quantization_stats['weight_zero_ratio_mean']:
            stats['weight_zero_ratio'] = self.quantization_stats['weight_zero_ratio_mean'][-1]

        # Memory stats
        if self.memory_stats['memory_utilization_ratio']:
            stats['memory_utilization'] = self.memory_stats['memory_utilization_ratio'][-1]

        # Performance
        if self.timing_stats['samples_per_second']:
            stats['samples_per_second'] = np.mean(self.timing_stats['samples_per_second'][-10:])

        return stats
