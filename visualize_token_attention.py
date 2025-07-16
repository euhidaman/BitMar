"""
Token-to-Pixel Attention Visualization for BitMar
Visualizes how text tokens attend to image regions/pixels during training
Shows the evolution of cross-modal attention alignment
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Optional
from transformers import GPT2TokenizerFast
import torch.nn.functional as F

class TokenPixelAttentionVisualizer:
    """Visualize token-to-pixel attention maps for BitMar"""
    
    def __init__(self, save_dir: str = "./attention_maps"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.save_dir / "token_maps").mkdir(exist_ok=True)
        (self.save_dir / "evolution").mkdir(exist_ok=True)
        (self.save_dir / "heatmaps").mkdir(exist_ok=True)
        (self.save_dir / "animations").mkdir(exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        
    def extract_cross_modal_attention(self, 
                                    model_outputs: Dict,
                                    input_ids: torch.Tensor,
                                    vision_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract cross-modal attention weights from model outputs"""
        
        # Get cross-modal attention from model outputs
        # This assumes the model returns cross_modal_attention weights
        cross_attention = model_outputs.get('cross_modal_attention', None)
        
        if cross_attention is None:
            print("Warning: No cross-modal attention found in model outputs")
            return {}
            
        # cross_attention shape: [batch_size, num_heads, seq_len, vision_dim]
        batch_size, num_heads, seq_len, vision_dim = cross_attention.shape
        
        # Average across heads (or keep separate for head-specific analysis)
        avg_attention = cross_attention.mean(dim=1)  # [batch_size, seq_len, vision_dim]
        
        return {
            'attention_weights': avg_attention,
            'per_head_attention': cross_attention,
            'tokens': input_ids,
            'vision_features': vision_features,
            'seq_len': seq_len,
            'vision_dim': vision_dim
        }
    
    def vision_features_to_spatial_map(self, 
                                     vision_features: torch.Tensor,
                                     target_height: int = 14,
                                     target_width: int = 14) -> torch.Tensor:
        """Convert 1D vision features back to 2D spatial map"""
        
        # DiNOv2 features are typically from 14x14 patches for 224x224 images
        batch_size, feature_dim = vision_features.shape
        
        # Reshape to spatial grid
        if feature_dim == target_height * target_width * 768:
            # Full spatial features
            spatial_features = vision_features.reshape(batch_size, target_height, target_width, 768)
        else:
            # If features are pre-pooled, we need to estimate spatial layout
            # This is an approximation - ideally we'd have the actual spatial features
            spatial_size = int(np.sqrt(feature_dim // 768)) if feature_dim > 768 else 1
            if spatial_size * spatial_size * 768 == feature_dim:
                spatial_features = vision_features.reshape(batch_size, spatial_size, spatial_size, 768)
                # Resize to target dimensions
                spatial_features = F.interpolate(
                    spatial_features.permute(0, 3, 1, 2), 
                    size=(target_height, target_width), 
                    mode='bilinear'
                ).permute(0, 2, 3, 1)
            else:
                # Fallback: create uniform spatial map
                spatial_features = vision_features.unsqueeze(1).unsqueeze(1).expand(
                    batch_size, target_height, target_width, -1
                )
        
        return spatial_features
    
    def create_token_attention_map(self,
                                 attention_weights: torch.Tensor,
                                 tokens: torch.Tensor,
                                 token_idx: int,
                                 image_size: Tuple[int, int] = (224, 224),
                                 spatial_size: Tuple[int, int] = (14, 14)) -> np.ndarray:
        """Create attention heatmap for a specific token"""
        
        # Get attention for specific token
        token_attention = attention_weights[0, token_idx]  # [vision_dim]
        
        # Reshape to spatial dimensions
        if len(token_attention.shape) == 1:
            # If 1D, reshape to spatial grid
            grid_size = int(np.sqrt(token_attention.shape[0]))
            if grid_size * grid_size == token_attention.shape[0]:
                spatial_attention = token_attention.reshape(grid_size, grid_size)
            else:
                # Approximate spatial layout
                spatial_attention = token_attention[:spatial_size[0]*spatial_size[1]].reshape(spatial_size)
        else:
            spatial_attention = token_attention
            
        # Convert to numpy and normalize
        attention_map = spatial_attention.detach().cpu().numpy()
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Resize to image dimensions
        attention_resized = cv2.resize(attention_map, image_size, interpolation=cv2.INTER_CUBIC)
        
        return attention_resized
    
    def visualize_caption_attention(self,
                                  caption: str,
                                  attention_data: Dict,
                                  image: Optional[np.ndarray] = None,
                                  epoch: int = 0,
                                  step: int = 0,
                                  save_name: str = "attention_map") -> None:
        """Create comprehensive attention visualization for caption"""
        
        # Tokenize caption
        tokens = self.tokenizer.encode(caption)
        token_texts = [self.tokenizer.decode([token]) for token in tokens]
        
        attention_weights = attention_data['attention_weights']
        seq_len = min(len(tokens), attention_weights.shape[1])
        
        # Create grid of subplots
        fig_width = min(20, seq_len * 3)
        fig_height = 8
        
        fig, axes = plt.subplots(2, seq_len, figsize=(fig_width, fig_height))
        if seq_len == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(f'Token-to-Image Attention Maps\\nEpoch {epoch}, Step {step}\\nCaption: "{caption}"', 
                     fontsize=14, y=0.95)
        
        for token_idx in range(seq_len):
            token_text = token_texts[token_idx]
            
            # Create attention heatmap for this token
            attention_map = self.create_token_attention_map(
                attention_weights, tokens, token_idx
            )
            
            # Top row: Raw attention heatmap
            im1 = axes[0, token_idx].imshow(attention_map, cmap='hot', interpolation='nearest')
            axes[0, token_idx].set_title(f'Token: "{token_text}"', fontsize=10)
            axes[0, token_idx].axis('off')
            
            # Bottom row: Attention overlay on image (if available)
            if image is not None:
                # Overlay attention on image
                overlay = self.create_attention_overlay(image, attention_map)
                axes[1, token_idx].imshow(overlay)
            else:
                # Just show attention map with different colormap
                im2 = axes[1, token_idx].imshow(attention_map, cmap='viridis', interpolation='nearest')
                
            axes[1, token_idx].set_title(f'Attention: {attention_map.max():.3f}', fontsize=10)
            axes[1, token_idx].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        save_path = self.save_dir / "token_maps" / f"{save_name}_epoch{epoch}_step{step}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved token attention map to {save_path}")
        plt.close()
    
    def create_attention_overlay(self, image: np.ndarray, attention_map: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """Create overlay of attention heatmap on image"""
        
        # Ensure image is RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            base_image = image.copy()
        else:
            base_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Normalize image to 0-1
        if base_image.max() > 1:
            base_image = base_image.astype(np.float32) / 255.0
        
        # Create colored attention map
        attention_colored = plt.cm.hot(attention_map)[:, :, :3]  # Remove alpha channel
        
        # Blend with original image
        overlay = (1 - alpha) * base_image + alpha * attention_colored
        
        return np.clip(overlay, 0, 1)
    
    def visualize_attention_evolution(self,
                                    attention_history: List[Dict],
                                    caption: str,
                                    token_idx: int,
                                    save_name: str = "evolution") -> None:
        """Visualize how attention for a specific token evolves over training"""
        
        num_steps = len(attention_history)
        cols = min(6, num_steps)
        rows = (num_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
            
        token_text = self.tokenizer.decode([self.tokenizer.encode(caption)[token_idx]])
        fig.suptitle(f'Attention Evolution for Token: "{token_text}"\\nCaption: "{caption}"', 
                     fontsize=14)
        
        for i, attention_data in enumerate(attention_history):
            row = i // cols
            col = i % cols
            
            if row < rows and col < cols:
                attention_map = self.create_token_attention_map(
                    attention_data['attention_weights'], 
                    attention_data['tokens'], 
                    token_idx
                )
                
                im = axes[row, col].imshow(attention_map, cmap='hot', interpolation='nearest')
                axes[row, col].set_title(f'Step {attention_data.get("step", i)}', fontsize=10)
                axes[row, col].axis('off')
                
                # Add colorbar for first subplot
                if i == 0:
                    plt.colorbar(im, ax=axes[row, col], shrink=0.8)
        
        # Hide empty subplots
        for i in range(num_steps, rows * cols):
            row = i // cols
            col = i % cols
            if row < rows and col < cols:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save evolution plot
        save_path = self.save_dir / "evolution" / f"{save_name}_token{token_idx}_evolution.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention evolution to {save_path}")
        plt.close()
    
    def create_attention_statistics(self,
                                  attention_history: List[Dict],
                                  caption: str,
                                  save_name: str = "stats") -> Dict:
        """Analyze attention statistics over training"""
        
        tokens = self.tokenizer.encode(caption)
        token_texts = [self.tokenizer.decode([token]) for token in tokens]
        
        stats = {
            'caption': caption,
            'tokens': token_texts,
            'attention_entropy': [],
            'attention_max': [],
            'attention_focus': [],
            'steps': []
        }
        
        for attention_data in attention_history:
            attention_weights = attention_data['attention_weights'][0]  # First batch item
            step = attention_data.get('step', 0)
            
            # Calculate statistics for each token
            token_entropies = []
            token_maxes = []
            token_focus = []
            
            for token_idx in range(min(len(tokens), attention_weights.shape[0])):
                token_attention = attention_weights[token_idx].detach().cpu().numpy()
                
                # Normalize
                token_attention = token_attention / (token_attention.sum() + 1e-8)
                
                # Calculate entropy (measure of attention spread)
                entropy = -np.sum(token_attention * np.log(token_attention + 1e-8))
                
                # Calculate max attention (measure of focus strength)
                max_attention = token_attention.max()
                
                # Calculate focus index (top 10% attention mass)
                sorted_attention = np.sort(token_attention)[::-1]
                top_10_percent = int(len(sorted_attention) * 0.1) + 1
                focus = sorted_attention[:top_10_percent].sum()
                
                token_entropies.append(entropy)
                token_maxes.append(max_attention)
                token_focus.append(focus)
            
            stats['attention_entropy'].append(token_entropies)
            stats['attention_max'].append(token_maxes)
            stats['attention_focus'].append(token_focus)
            stats['steps'].append(step)
        
        # Save statistics
        stats_path = self.save_dir / f"{save_name}_attention_stats.json"
        with open(stats_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_stats = {k: v if k != 'steps' else v for k, v in stats.items()}
            json.dump(json_stats, f, indent=2)
        
        print(f"Saved attention statistics to {stats_path}")
        
        return stats
    
    def plot_attention_trends(self,
                            stats: Dict,
                            save_name: str = "trends") -> None:
        """Plot attention trends over training"""
        
        steps = stats['steps']
        tokens = stats['tokens']
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot entropy trends
        entropies = np.array(stats['attention_entropy'])
        for token_idx, token_text in enumerate(tokens):
            if token_idx < entropies.shape[1]:
                axes[0].plot(steps, entropies[:, token_idx], 
                           marker='o', label=f'"{token_text}"', linewidth=2)
        axes[0].set_title('Attention Entropy Over Training (Higher = More Spread Out)')
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Entropy')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Plot max attention trends
        max_attentions = np.array(stats['attention_max'])
        for token_idx, token_text in enumerate(tokens):
            if token_idx < max_attentions.shape[1]:
                axes[1].plot(steps, max_attentions[:, token_idx], 
                           marker='s', label=f'"{token_text}"', linewidth=2)
        axes[1].set_title('Maximum Attention Over Training (Higher = More Focused)')
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Max Attention')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        # Plot focus trends
        focus_scores = np.array(stats['attention_focus'])
        for token_idx, token_text in enumerate(tokens):
            if token_idx < focus_scores.shape[1]:
                axes[2].plot(steps, focus_scores[:, token_idx], 
                           marker='^', label=f'"{token_text}"', linewidth=2)
        axes[2].set_title('Attention Focus Score Over Training (Top 10% Mass)')
        axes[2].set_xlabel('Training Step')
        axes[2].set_ylabel('Focus Score')
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save trends plot
        save_path = self.save_dir / f"{save_name}_attention_trends.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention trends to {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize BitMar token-to-pixel attention")
    parser.add_argument("--save_dir", type=str, default="./attention_maps", 
                       help="Directory to save attention visualizations")
    parser.add_argument("--caption", type=str, default="a cat sitting on a table",
                       help="Example caption to analyze")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = TokenPixelAttentionVisualizer(args.save_dir)
    
    print(f"Token-pixel attention visualizer initialized")
    print(f"Save directory: {args.save_dir}")
    print(f"Example caption: {args.caption}")
    print("\\nTo use this visualizer:")
    print("1. Extract attention weights during BitMar training")
    print("2. Call visualizer.visualize_caption_attention() with your data")
    print("3. Use visualizer.visualize_attention_evolution() to show training progression")

if __name__ == "__main__":
    main()
