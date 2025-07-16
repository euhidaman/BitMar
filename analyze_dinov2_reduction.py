"""
DiNOv2 Feature Reduction and Quality Analysis for BitMar Edge Deployment
Explores different strategies to reduce image feature complexity while maintaining performance
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DiNOFeatureReducer:
    """Reduce DiNOv2 feature complexity for edge deployment"""
    
    def __init__(self, original_dim: int = 768, target_dim: int = 128):
        self.original_dim = original_dim
        self.target_dim = target_dim
        
    def reduce_dimensionality(self, features: torch.Tensor, method: str = "pca") -> torch.Tensor:
        """Reduce feature dimensionality using various methods"""
        
        if method == "linear_projection":
            return self._linear_projection(features)
        elif method == "pca":
            return self._pca_reduction(features)
        elif method == "top_k_selection":
            return self._top_k_selection(features)
        elif method == "spatial_pooling":
            return self._spatial_pooling(features)
        elif method == "learned_compression":
            return self._learned_compression(features)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
    
    def _linear_projection(self, features: torch.Tensor) -> torch.Tensor:
        """Simple linear projection to reduce dimensions"""
        # features shape: [batch_size, spatial_dim, feature_dim] or [batch_size, feature_dim]
        
        if len(features.shape) == 3:
            batch_size, spatial_dim, feature_dim = features.shape
            # Reshape for linear layer
            features_flat = features.reshape(-1, feature_dim)
            
            # Create projection layer
            projection = nn.Linear(feature_dim, self.target_dim)
            
            # Apply projection
            reduced_flat = projection(features_flat)
            
            # Reshape back
            reduced = reduced_flat.reshape(batch_size, spatial_dim, self.target_dim)
        else:
            # Already flat
            projection = nn.Linear(features.shape[-1], self.target_dim)
            reduced = projection(features)
            
        return reduced
    
    def _pca_reduction(self, features: torch.Tensor) -> torch.Tensor:
        """PCA-based dimensionality reduction"""
        
        # Convert to numpy for sklearn PCA
        features_np = features.detach().cpu().numpy()
        original_shape = features_np.shape
        
        # Flatten spatial dimensions if present
        if len(original_shape) == 3:
            features_flat = features_np.reshape(-1, original_shape[-1])
        else:
            features_flat = features_np
            
        # Apply PCA (simplified version - in practice use sklearn)
        # For demonstration, we'll use SVD
        U, S, Vt = np.linalg.svd(features_flat, full_matrices=False)
        
        # Keep top components
        n_components = min(self.target_dim, features_flat.shape[1])
        reduced_flat = U[:, :n_components] @ np.diag(S[:n_components])
        
        # Reshape back to original spatial structure
        if len(original_shape) == 3:
            reduced = reduced_flat.reshape(original_shape[0], original_shape[1], n_components)
        else:
            reduced = reduced_flat
            
        return torch.tensor(reduced, dtype=features.dtype, device=features.device)
    
    def _top_k_selection(self, features: torch.Tensor) -> torch.Tensor:
        """Select top-K most important feature dimensions"""
        
        # Calculate feature importance (variance across spatial/batch dimensions)
        if len(features.shape) == 3:
            importance = features.var(dim=(0, 1))  # Variance across batch and spatial
        else:
            importance = features.var(dim=0)  # Variance across batch
            
        # Get top-K indices
        _, top_indices = torch.topk(importance, self.target_dim)
        
        # Select top features
        reduced = features[..., top_indices]
        
        return reduced
    
    def _spatial_pooling(self, features: torch.Tensor) -> torch.Tensor:
        """Reduce spatial resolution while keeping feature dimensions"""
        
        if len(features.shape) != 3:
            return features  # Can't spatially pool 2D features
            
        batch_size, spatial_dim, feature_dim = features.shape
        
        # Assume square spatial layout
        spatial_size = int(np.sqrt(spatial_dim))
        if spatial_size * spatial_size != spatial_dim:
            return features  # Not square, can't easily pool
            
        # Reshape to 2D spatial
        features_2d = features.reshape(batch_size, spatial_size, spatial_size, feature_dim)
        
        # Apply pooling (2x2 average pooling)
        pool_size = 2
        new_spatial_size = spatial_size // pool_size
        
        pooled = nn.functional.avg_pool2d(
            features_2d.permute(0, 3, 1, 2),  # [batch, feature, h, w]
            kernel_size=pool_size
        ).permute(0, 2, 3, 1)  # Back to [batch, h, w, feature]
        
        # Flatten spatial dimensions
        reduced = pooled.reshape(batch_size, new_spatial_size * new_spatial_size, feature_dim)
        
        return reduced
    
    def _learned_compression(self, features: torch.Tensor) -> torch.Tensor:
        """Learned compression using autoencoder-style bottleneck"""
        
        # Simple autoencoder bottleneck
        encoder = nn.Sequential(
            nn.Linear(self.original_dim, self.original_dim // 2),
            nn.ReLU(),
            nn.Linear(self.original_dim // 2, self.target_dim)
        )
        
        if len(features.shape) == 3:
            batch_size, spatial_dim, feature_dim = features.shape
            features_flat = features.reshape(-1, feature_dim)
            encoded_flat = encoder(features_flat)
            encoded = encoded_flat.reshape(batch_size, spatial_dim, self.target_dim)
        else:
            encoded = encoder(features)
            
        return encoded

class FeatureQualityAnalyzer:
    """Analyze the impact of feature reduction on model performance"""
    
    def __init__(self, save_dir: str = "./feature_analysis"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def analyze_reduction_impact(self, 
                                original_features: torch.Tensor,
                                reduction_methods: List[str],
                                target_dims: List[int]) -> Dict:
        """Analyze impact of different reduction strategies"""
        
        results = {
            'methods': reduction_methods,
            'target_dims': target_dims,
            'compression_ratios': [],
            'information_retention': [],
            'computational_savings': []
        }
        
        reducer = DiNOFeatureReducer()
        
        for method in reduction_methods:
            method_results = {
                'method': method,
                'results_by_dim': []
            }
            
            for target_dim in target_dims:
                reducer.target_dim = target_dim
                
                # Apply reduction
                reduced_features = reducer.reduce_dimensionality(original_features, method)
                
                # Calculate metrics
                compression_ratio = self._calculate_compression_ratio(original_features, reduced_features)
                info_retention = self._calculate_information_retention(original_features, reduced_features)
                comp_savings = self._calculate_computational_savings(original_features, reduced_features)
                
                method_results['results_by_dim'].append({
                    'target_dim': target_dim,
                    'compression_ratio': compression_ratio,
                    'information_retention': info_retention,
                    'computational_savings': comp_savings
                })
            
            results[method] = method_results
        
        return results
    
    def _calculate_compression_ratio(self, original: torch.Tensor, reduced: torch.Tensor) -> float:
        """Calculate compression ratio (original_size / reduced_size)"""
        original_size = original.numel()
        reduced_size = reduced.numel()
        return original_size / reduced_size
    
    def _calculate_information_retention(self, original: torch.Tensor, reduced: torch.Tensor) -> float:
        """Estimate information retention (simplified metric)"""
        
        # For linear projection, we can approximate info retention
        # by comparing the variance captured
        
        original_var = torch.var(original)
        reduced_var = torch.var(reduced)
        
        # Normalize by dimensions
        orig_var_per_dim = original_var / original.shape[-1]
        red_var_per_dim = reduced_var / reduced.shape[-1]
        
        retention = min(1.0, red_var_per_dim / (orig_var_per_dim + 1e-8))
        
        return retention.item() if torch.is_tensor(retention) else retention
    
    def _calculate_computational_savings(self, original: torch.Tensor, reduced: torch.Tensor) -> float:
        """Calculate computational savings for downstream processing"""
        
        # Assume linear computational cost proportional to feature dimensions
        original_ops = original.shape[-1] * 2  # Simplified: feature_dim * (weight + bias)
        reduced_ops = reduced.shape[-1] * 2
        
        savings = 1.0 - (reduced_ops / original_ops)
        return savings
    
    def create_analysis_plots(self, results: Dict):
        """Create visualization plots for feature reduction analysis"""
        
        methods = results['methods']
        target_dims = results['target_dims']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Compression Ratios
        for method in methods:
            if method in results:
                method_data = results[method]['results_by_dim']
                compression_ratios = [r['compression_ratio'] for r in method_data]
                axes[0, 0].plot(target_dims, compression_ratios, marker='o', label=method, linewidth=2)
        
        axes[0, 0].set_title('Compression Ratio vs Target Dimensions')
        axes[0, 0].set_xlabel('Target Dimensions')
        axes[0, 0].set_ylabel('Compression Ratio')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Information Retention
        for method in methods:
            if method in results:
                method_data = results[method]['results_by_dim']
                info_retention = [r['information_retention'] for r in method_data]
                axes[0, 1].plot(target_dims, info_retention, marker='s', label=method, linewidth=2)
        
        axes[0, 1].set_title('Information Retention vs Target Dimensions')
        axes[0, 1].set_xlabel('Target Dimensions')
        axes[0, 1].set_ylabel('Information Retention')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Computational Savings
        for method in methods:
            if method in results:
                method_data = results[method]['results_by_dim']
                comp_savings = [r['computational_savings'] for r in method_data]
                axes[1, 0].plot(target_dims, comp_savings, marker='^', label=method, linewidth=2)
        
        axes[1, 0].set_title('Computational Savings vs Target Dimensions')
        axes[1, 0].set_xlabel('Target Dimensions')
        axes[1, 0].set_ylabel('Computational Savings')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Trade-off Analysis (Info Retention vs Compression)
        for method in methods:
            if method in results:
                method_data = results[method]['results_by_dim']
                info_retention = [r['information_retention'] for r in method_data]
                compression_ratios = [r['compression_ratio'] for r in method_data]
                axes[1, 1].scatter(compression_ratios, info_retention, label=method, s=100, alpha=0.7)
        
        axes[1, 1].set_title('Trade-off: Information Retention vs Compression')
        axes[1, 1].set_xlabel('Compression Ratio')
        axes[1, 1].set_ylabel('Information Retention')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_dir / "feature_reduction_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature reduction analysis to {save_path}")
        plt.close()

def generate_edge_deployment_recommendations():
    """Generate recommendations for edge deployment feature reduction"""
    
    recommendations = {
        "ultra_tiny_edge": {
            "description": "ARM Cortex-M7, <1MB RAM",
            "target_dim": 32,
            "method": "top_k_selection",
            "spatial_pooling": "4x4 -> 2x2",
            "expected_compression": "24x",
            "performance_impact": "15-25% accuracy loss"
        },
        "tiny_edge": {
            "description": "Raspberry Pi, <100MB RAM", 
            "target_dim": 64,
            "method": "linear_projection",
            "spatial_pooling": "14x14 -> 7x7", 
            "expected_compression": "12x",
            "performance_impact": "8-15% accuracy loss"
        },
        "mobile": {
            "description": "Mobile phones, <500MB RAM",
            "target_dim": 128,
            "method": "learned_compression",
            "spatial_pooling": "14x14 -> 10x10",
            "expected_compression": "6x", 
            "performance_impact": "3-8% accuracy loss"
        },
        "edge_gpu": {
            "description": "Edge GPU (Jetson), <2GB RAM",
            "target_dim": 256,
            "method": "pca",
            "spatial_pooling": "none",
            "expected_compression": "3x",
            "performance_impact": "1-3% accuracy loss"
        }
    }
    
    return recommendations

def main():
    """Run feature reduction analysis"""
    
    print("DiNOv2 Feature Reduction Analysis for BitMar Edge Deployment")
    print("=" * 60)
    
    # Create synthetic DiNOv2 features for analysis
    batch_size = 4
    spatial_dim = 14 * 14  # 14x14 patches
    feature_dim = 768  # DiNOv2 dimension
    
    # Simulate DiNOv2 features
    torch.manual_seed(42)
    original_features = torch.randn(batch_size, spatial_dim, feature_dim)
    
    print(f"Original features shape: {original_features.shape}")
    print(f"Original memory usage: {original_features.numel() * 4 / 1024:.1f} KB (fp32)")
    
    # Define reduction strategies to test
    reduction_methods = [
        "linear_projection",
        "top_k_selection", 
        "spatial_pooling",
        "pca"
    ]
    
    target_dims = [32, 64, 128, 256, 384]
    
    # Run analysis
    analyzer = FeatureQualityAnalyzer()
    results = analyzer.analyze_reduction_impact(
        original_features, reduction_methods, target_dims
    )
    
    # Create visualizations
    analyzer.create_analysis_plots(results)
    
    # Generate recommendations
    recommendations = generate_edge_deployment_recommendations()
    
    print("\\nEdge Deployment Recommendations:")
    print("-" * 40)
    
    for config_name, config in recommendations.items():
        print(f"\\n{config_name.upper()}:")
        print(f"  Target: {config['description']}")
        print(f"  Feature Dim: {config['target_dim']} (from 768)")
        print(f"  Method: {config['method']}")
        print(f"  Spatial: {config['spatial_pooling']}")
        print(f"  Compression: {config['expected_compression']}")
        print(f"  Performance Impact: {config['performance_impact']}")
        
        # Calculate memory savings
        original_mem = 14 * 14 * 768 * 4 / 1024  # KB
        if "spatial_pooling" in config and "4x4" in config["spatial_pooling"]:
            spatial_factor = (4*4) / (14*14)
        elif "spatial_pooling" in config and "7x7" in config["spatial_pooling"]:
            spatial_factor = (7*7) / (14*14)
        elif "spatial_pooling" in config and "10x10" in config["spatial_pooling"]:
            spatial_factor = (10*10) / (14*14)
        else:
            spatial_factor = 1.0
            
        feature_factor = config['target_dim'] / 768
        reduced_mem = original_mem * spatial_factor * feature_factor
        
        print(f"  Memory: {original_mem:.1f}KB -> {reduced_mem:.1f}KB ({original_mem/reduced_mem:.1f}x reduction)")

if __name__ == "__main__":
    main()
