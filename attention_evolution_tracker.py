"""
Enhanced Token-to-Image Attention Evolution Tracker for BitMar
Tracks how text tokens attend to image regions across training epochs
Shows learning progression and attention pattern changes over time
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple, Optional
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
from datetime import datetime

class AttentionEvolutionTracker:
    """Track and visualize attention evolution across training epochs"""
    
    def __init__(self, save_dir: str = "./attention_evolution"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for organized storage
        (self.save_dir / "epoch_snapshots").mkdir(exist_ok=True)
        (self.save_dir / "evolution_plots").mkdir(exist_ok=True)
        (self.save_dir / "attention_matrices").mkdir(exist_ok=True)
        (self.save_dir / "token_heatmaps").mkdir(exist_ok=True)
        (self.save_dir / "comparison_grids").mkdir(exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        
        # Storage for attention evolution
        self.attention_history = {}  # {epoch: {sample_id: attention_data}}
        self.token_evolution = {}    # {token: {epoch: attention_values}}
        
    def save_epoch_attention(self, 
                           epoch: int,
                           sample_id: str,
                           caption: str,
                           attention_weights: torch.Tensor,
                           image_features: torch.Tensor,
                           compressed_features: Optional[torch.Tensor] = None):
        """Save attention data for a specific epoch and sample"""
        
        if epoch not in self.attention_history:
            self.attention_history[epoch] = {}
            
        # Process attention data
        attention_data = {
            'caption': caption,
            'attention_weights': attention_weights.detach().cpu().numpy(),
            'image_features_shape': image_features.shape,
            'timestamp': datetime.now().isoformat(),
            'compressed_features_shape': compressed_features.shape if compressed_features is not None else None
        }
        
        # Store in memory
        self.attention_history[epoch][sample_id] = attention_data
        
        # Save to disk for persistence
        epoch_file = self.save_dir / "epoch_snapshots" / f"epoch_{epoch:03d}.pkl"
        with open(epoch_file, 'wb') as f:
            pickle.dump(self.attention_history[epoch], f)
            
        # Update token evolution tracking
        self._update_token_evolution(epoch, caption, attention_weights)
        
    def _update_token_evolution(self, epoch: int, caption: str, attention_weights: torch.Tensor):
        """Track how individual tokens' attention patterns evolve"""
        
        tokens = self.tokenizer.encode(caption)
        token_texts = [self.tokenizer.decode([token]) for token in tokens]
        
        # attention_weights shape: [batch_size, seq_len, vision_features]
        attention_np = attention_weights[0].detach().cpu().numpy()  # Take first batch
        
        for i, (token_id, token_text) in enumerate(zip(tokens, token_texts)):
            if i >= attention_np.shape[0]:
                break
                
            token_key = f"{token_text}_{token_id}"
            
            if token_key not in self.token_evolution:
                self.token_evolution[token_key] = {}
                
            # Store attention statistics for this token
            token_attention = attention_np[i]  # [vision_features]
            
            self.token_evolution[token_key][epoch] = {
                'mean_attention': float(np.mean(token_attention)),
                'max_attention': float(np.max(token_attention)),
                'attention_entropy': float(self._calculate_entropy(token_attention)),
                'top_k_indices': np.argsort(token_attention)[-10:].tolist(),  # Top 10 attended features
                'attention_pattern': token_attention.tolist()
            }
    
    def _calculate_entropy(self, attention: np.ndarray) -> float:
        """Calculate entropy of attention distribution"""
        # Normalize to probability distribution
        prob = attention - np.min(attention)
        prob = prob / (np.sum(prob) + 1e-8)
        
        # Calculate entropy
        entropy = -np.sum(prob * np.log(prob + 1e-8))
        return entropy
    
    def create_token_evolution_plot(self, token_text: str, token_id: int, epochs_to_show: List[int] = None):
        """Create plot showing how a specific token's attention evolves over epochs"""
        
        token_key = f"{token_text}_{token_id}"
        
        if token_key not in self.token_evolution:
            print(f"No evolution data found for token: {token_text}")
            return
            
        token_data = self.token_evolution[token_key]
        
        if epochs_to_show is None:
            epochs_to_show = sorted(token_data.keys())
            
        # Extract metrics over epochs
        epochs = []
        mean_attentions = []
        max_attentions = []
        entropies = []
        
        for epoch in epochs_to_show:
            if epoch in token_data:
                epochs.append(epoch)
                mean_attentions.append(token_data[epoch]['mean_attention'])
                max_attentions.append(token_data[epoch]['max_attention'])
                entropies.append(token_data[epoch]['attention_entropy'])
        
        # Create evolution plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Attention Evolution for Token: "{token_text}"', fontsize=16)
        
        # Plot 1: Mean attention over epochs
        axes[0, 0].plot(epochs, mean_attentions, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Mean Attention Strength')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Mean Attention')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Max attention over epochs
        axes[0, 1].plot(epochs, max_attentions, 'r-o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Peak Attention Strength')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Max Attention')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Attention entropy over epochs
        axes[1, 0].plot(epochs, entropies, 'g-o', linewidth=2, markersize=6)
        axes[1, 0].set_title('Attention Entropy (Focus vs Spread)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Attention pattern heatmap across epochs
        if len(epochs) > 1:
            attention_matrix = []
            for epoch in epochs:
                if epoch in token_data:
                    pattern = token_data[epoch]['attention_pattern']
                    # Truncate or pad to consistent length for visualization
                    pattern = pattern[:100] if len(pattern) > 100 else pattern + [0] * (100 - len(pattern))
                    attention_matrix.append(pattern)
            
            attention_matrix = np.array(attention_matrix)
            im = axes[1, 1].imshow(attention_matrix, aspect='auto', cmap='viridis')
            axes[1, 1].set_title('Attention Pattern Evolution')
            axes[1, 1].set_xlabel('Vision Feature Index')
            axes[1, 1].set_ylabel('Epoch')
            axes[1, 1].set_yticks(range(len(epochs)))
            axes[1, 1].set_yticklabels(epochs)
            plt.colorbar(im, ax=axes[1, 1], label='Attention Weight')
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_dir / "evolution_plots" / f"token_{token_text}_{token_id}_evolution.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved token evolution plot to {save_path}")
        plt.close()
    
    def create_epoch_comparison_grid(self, 
                                   sample_id: str, 
                                   epochs: List[int],
                                   image_path: Optional[str] = None):
        """Create grid comparing attention across multiple epochs for same sample"""
        
        # Load attention data for specified epochs
        epoch_data = {}
        for epoch in epochs:
            if epoch in self.attention_history and sample_id in self.attention_history[epoch]:
                epoch_data[epoch] = self.attention_history[epoch][sample_id]
        
        if not epoch_data:
            print(f"No attention data found for sample {sample_id} across epochs {epochs}")
            return
            
        # Get caption from first available epoch
        first_epoch = min(epoch_data.keys())
        caption = epoch_data[first_epoch]['caption']
        tokens = self.tokenizer.encode(caption)
        token_texts = [self.tokenizer.decode([token]) for token in tokens]
        
        # Create comparison grid
        num_epochs = len(epoch_data)
        num_tokens = min(len(tokens), 8)  # Limit to 8 tokens for visualization
        
        fig, axes = plt.subplots(num_epochs, num_tokens, figsize=(num_tokens * 3, num_epochs * 3))
        fig.suptitle(f'Token Attention Evolution\nCaption: "{caption[:50]}..."', fontsize=14)
        
        if num_epochs == 1:
            axes = axes.reshape(1, -1)
        if num_tokens == 1:
            axes = axes.reshape(-1, 1)
            
        for row, epoch in enumerate(sorted(epoch_data.keys())):
            attention_weights = epoch_data[epoch]['attention_weights']
            
            for col, token_idx in enumerate(range(num_tokens)):
                if token_idx < len(token_texts):
                    # Create attention heatmap for this token
                    token_attention = attention_weights[token_idx]  # [vision_features]
                    
                    # Reshape for visualization (assume spatial structure)
                    grid_size = int(np.sqrt(len(token_attention)))
                    if grid_size * grid_size == len(token_attention):
                        attention_map = token_attention.reshape(grid_size, grid_size)
                    else:
                        # Truncate or pad to nearest square
                        target_size = 14  # Common DiNOv2 spatial size
                        if len(token_attention) >= target_size * target_size:
                            attention_map = token_attention[:target_size*target_size].reshape(target_size, target_size)
                        else:
                            # Pad with zeros
                            padded = np.zeros(target_size * target_size)
                            padded[:len(token_attention)] = token_attention
                            attention_map = padded.reshape(target_size, target_size)
                    
                    # Plot heatmap
                    im = axes[row, col].imshow(attention_map, cmap='hot', interpolation='bilinear')
                    
                    # Add title and labels
                    if row == 0:  # First row
                        axes[row, col].set_title(f'"{token_texts[token_idx]}"', fontsize=10)
                    if col == 0:  # First column
                        axes[row, col].set_ylabel(f'Epoch {epoch}', fontsize=10)
                    
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
                    
                    # Add colorbar for last column
                    if col == num_tokens - 1:
                        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save comparison grid
        save_path = self.save_dir / "comparison_grids" / f"sample_{sample_id}_epochs_{'_'.join(map(str, epochs))}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved epoch comparison grid to {save_path}")
        plt.close()
    
    def create_attention_learning_summary(self, max_epochs: int = None):
        """Create comprehensive summary of attention learning progress"""
        
        if max_epochs is None:
            max_epochs = max(self.attention_history.keys()) if self.attention_history else 0
            
        epochs = sorted([e for e in self.attention_history.keys() if e <= max_epochs])
        
        if len(epochs) < 2:
            print("Need at least 2 epochs to show learning progress")
            return
            
        # Analyze overall attention patterns
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('BitMar Attention Learning Progress', fontsize=16)
        
        # 1. Token diversity over epochs
        token_diversity = []
        for epoch in epochs:
            unique_tokens = set()
            for sample_data in self.attention_history[epoch].values():
                tokens = self.tokenizer.encode(sample_data['caption'])
                unique_tokens.update(tokens)
            token_diversity.append(len(unique_tokens))
            
        axes[0, 0].plot(epochs, token_diversity, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Token Vocabulary Diversity')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Unique Tokens')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Average attention entropy over epochs
        avg_entropies = []
        for epoch in epochs:
            entropies = []
            for sample_data in self.attention_history[epoch].values():
                attention = sample_data['attention_weights']
                for token_att in attention:
                    entropies.append(self._calculate_entropy(token_att))
            avg_entropies.append(np.mean(entropies))
            
        axes[0, 1].plot(epochs, avg_entropies, 'g-o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Average Attention Entropy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Entropy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature compression effectiveness (if available)
        compression_ratios = []
        for epoch in epochs:
            ratios = []
            for sample_data in self.attention_history[epoch].values():
                if sample_data['compressed_features_shape'] is not None:
                    original_size = np.prod(sample_data['image_features_shape'])
                    compressed_size = np.prod(sample_data['compressed_features_shape'])
                    ratios.append(original_size / compressed_size)
            if ratios:
                compression_ratios.append(np.mean(ratios))
            else:
                compression_ratios.append(1.0)  # No compression
                
        axes[0, 2].plot(epochs, compression_ratios, 'r-o', linewidth=2, markersize=6)
        axes[0, 2].set_title('DiNOv2 Compression Ratio')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Compression Ratio')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Most attended vision features heatmap
        feature_attention_matrix = []
        for epoch in epochs:
            epoch_attention = np.zeros(196)  # Assume 14x14 spatial features
            count = 0
            for sample_data in self.attention_history[epoch].values():
                attention = sample_data['attention_weights']
                avg_attention = np.mean(attention, axis=0)  # Average across tokens
                if len(avg_attention) <= 196:
                    epoch_attention[:len(avg_attention)] += avg_attention
                    count += 1
            if count > 0:
                feature_attention_matrix.append(epoch_attention / count)
            else:
                feature_attention_matrix.append(epoch_attention)
                
        feature_attention_matrix = np.array(feature_attention_matrix)
        im = axes[1, 0].imshow(feature_attention_matrix, aspect='auto', cmap='viridis')
        axes[1, 0].set_title('Vision Feature Attention Over Time')
        axes[1, 0].set_xlabel('Vision Feature Index')
        axes[1, 0].set_ylabel('Epoch')
        axes[1, 0].set_yticks(range(len(epochs)))
        axes[1, 0].set_yticklabels(epochs)
        plt.colorbar(im, ax=axes[1, 0], label='Average Attention')
        
        # 5. Top token evolution
        top_tokens = {}
        for epoch_data in self.token_evolution.values():
            for epoch, data in epoch_data.items():
                if epoch in epochs:
                    mean_att = data['mean_attention']
                    token = epoch_data  # This would need to be fixed to get actual token
                    # Simplified for now - would need better token tracking
        
        # 6. Learning stability metrics
        stability_scores = []
        for i in range(1, len(epochs)):
            prev_epoch = epochs[i-1]
            curr_epoch = epochs[i]
            
            # Compare attention patterns between consecutive epochs
            similarities = []
            for sample_id in self.attention_history[curr_epoch]:
                if sample_id in self.attention_history[prev_epoch]:
                    prev_att = self.attention_history[prev_epoch][sample_id]['attention_weights']
                    curr_att = self.attention_history[curr_epoch][sample_id]['attention_weights']
                    
                    # Calculate cosine similarity
                    prev_flat = prev_att.flatten()
                    curr_flat = curr_att.flatten()
                    
                    cos_sim = np.dot(prev_flat, curr_flat) / (np.linalg.norm(prev_flat) * np.linalg.norm(curr_flat) + 1e-8)
                    similarities.append(cos_sim)
            
            stability_scores.append(np.mean(similarities) if similarities else 0)
            
        axes[1, 1].plot(epochs[1:], stability_scores, 'purple', linewidth=2, marker='o', markersize=6)
        axes[1, 1].set_title('Attention Pattern Stability')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Cosine Similarity to Previous Epoch')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save summary
        save_path = self.save_dir / "attention_learning_summary.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention learning summary to {save_path}")
        plt.close()
        
        # Save data for further analysis
        summary_data = {
            'epochs': epochs,
            'token_diversity': token_diversity,
            'avg_entropies': avg_entropies,
            'compression_ratios': compression_ratios,
            'stability_scores': stability_scores,
            'feature_attention_matrix': feature_attention_matrix.tolist()
        }
        
        with open(self.save_dir / "learning_summary_data.json", 'w') as f:
            json.dump(summary_data, f, indent=2)
