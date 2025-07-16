"""
Training Integration for BitMar with Attention Evolution Tracking and DiNOv2 Compression
Shows how to integrate attention tracking and image compression during training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import yaml
import argparse
import wandb
from dotenv import load_dotenv
import os
from typing import Dict, Optional, Tuple

# Import our custom modules
from attention_evolution_tracker import AttentionEvolutionTracker
from analyze_dinov2_reduction import DiNOFeatureReducer

class BitMarTrainingWithTracking:
    """Enhanced BitMar training with attention evolution tracking and DiNOv2 compression"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict,
                 save_dir: str = "./training_outputs"):
        
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize attention tracker
        self.attention_tracker = AttentionEvolutionTracker(
            save_dir=str(self.save_dir / "attention_evolution")
        )
        
        # Initialize DiNOv2 compression
        self.vision_compressor = DiNOFeatureReducer(
            original_dim=768,  # DiNOv2 feature dimension
            target_dim=config.get('compressed_vision_dim', 128)
        )
        
        # Tracking settings
        self.track_every_n_steps = config.get('track_attention_every_n_steps', 100)
        self.save_attention_every_n_epochs = config.get('save_attention_every_n_epochs', 1)
        self.compression_method = config.get('vision_compression_method', 'linear_projection')
        
        print(f"✅ Initialized training with:")
        print(f"   - Attention tracking every {self.track_every_n_steps} steps")
        print(f"   - DiNOv2 compression: {768} → {config.get('compressed_vision_dim', 128)} dims")
        print(f"   - Compression method: {self.compression_method}")
    
    def compress_vision_features(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Apply DiNOv2 compression during training"""
        
        # vision_features shape: [batch_size, spatial_patches, feature_dim]
        original_shape = vision_features.shape
        
        # Apply compression
        compressed_features = self.vision_compressor.reduce_dimensionality(
            vision_features, 
            method=self.compression_method
        )
        
        # Log compression ratio
        original_size = np.prod(original_shape)
        compressed_size = np.prod(compressed_features.shape)
        compression_ratio = original_size / compressed_size
        
        if hasattr(self, '_log_compression_ratio'):
            self._compression_ratios.append(compression_ratio)
        else:
            self._compression_ratios = [compression_ratio]
            self._log_compression_ratio = True
        
        return compressed_features
    
    def extract_cross_modal_attention(self, model_outputs: Dict) -> Optional[torch.Tensor]:
        """Extract cross-modal attention weights from model outputs"""
        
        # This depends on your model architecture
        # Adjust based on how BitMar returns attention weights
        
        if 'cross_modal_attention' in model_outputs:
            return model_outputs['cross_modal_attention']
        elif 'attentions' in model_outputs:
            # Look for cross-modal attention in the attention stack
            attentions = model_outputs['attentions']
            if isinstance(attentions, (list, tuple)) and len(attentions) > 0:
                # Typically cross-modal attention is in the last layer
                return attentions[-1]
        
        # Fallback: try to extract from intermediate representations
        if hasattr(self.model, 'get_cross_modal_attention'):
            return self.model.get_cross_modal_attention()
            
        return None
    
    def training_step(self, 
                     batch: Dict, 
                     epoch: int, 
                     step: int, 
                     global_step: int) -> Dict:
        """Enhanced training step with attention tracking and compression"""
        
        # Extract batch data
        input_ids = batch['input_ids']  # Text tokens
        vision_features = batch['vision_features']  # Original DiNOv2 features
        captions = batch.get('captions', [''] * len(input_ids))
        
        # Apply DiNOv2 compression
        compressed_vision = self.compress_vision_features(vision_features)
        
        # Forward pass with compressed features
        batch_compressed = batch.copy()
        batch_compressed['vision_features'] = compressed_vision
        
        model_outputs = self.model(**batch_compressed)
        
        # Extract attention weights for tracking
        cross_modal_attention = self.extract_cross_modal_attention(model_outputs)
        
        # Track attention evolution (periodic)
        if (global_step % self.track_every_n_steps == 0 and 
            cross_modal_attention is not None):
            
            # Save attention data for first sample in batch
            sample_id = f"step_{global_step}_sample_0"
            
            self.attention_tracker.save_epoch_attention(
                epoch=epoch,
                sample_id=sample_id,
                caption=captions[0] if captions else "No caption",
                attention_weights=cross_modal_attention[0:1],  # First sample only
                image_features=vision_features[0:1],
                compressed_features=compressed_vision[0:1]
            )
            
            print(f"📊 Tracked attention at epoch {epoch}, step {step} (global {global_step})")
        
        # Calculate loss
        loss = model_outputs.get('loss', model_outputs.get('logits', torch.tensor(0.0)))
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, requires_grad=True)
        
        # Prepare return dict
        return {
            'loss': loss,
            'model_outputs': model_outputs,
            'compression_ratio': self._compression_ratios[-1] if self._compression_ratios else 1.0,
            'attention_tracked': cross_modal_attention is not None
        }
    
    def end_of_epoch_processing(self, epoch: int):
        """Process attention data at the end of each epoch"""
        
        if epoch % self.save_attention_every_n_epochs == 0:
            print(f"🎯 Processing attention evolution for epoch {epoch}...")
            
            # Create evolution visualizations for this epoch
            if epoch > 0:  # Need at least 2 epochs for comparison
                # Create comparison grid for a sample
                sample_ids = []
                if epoch in self.attention_tracker.attention_history:
                    sample_ids = list(self.attention_tracker.attention_history[epoch].keys())
                
                if sample_ids and epoch > 1:
                    # Compare this epoch with previous epoch
                    self.attention_tracker.create_epoch_comparison_grid(
                        sample_id=sample_ids[0],
                        epochs=[epoch-1, epoch]
                    )
            
            # Generate learning summary
            if epoch >= 2:  # Need at least 3 epochs for meaningful summary
                self.attention_tracker.create_attention_learning_summary(max_epochs=epoch)
            
            # Log compression statistics
            if hasattr(self, '_compression_ratios') and self._compression_ratios:
                avg_compression = np.mean(self._compression_ratios)
                print(f"📉 Average DiNOv2 compression ratio this epoch: {avg_compression:.2f}x")
                print(f"🔥 Memory savings: {(1 - 1/avg_compression)*100:.1f}%")
    
    def generate_final_analysis(self):
        """Generate comprehensive analysis at the end of training"""
        
        print("🎨 Generating final attention evolution analysis...")
        
        # Create learning summary for all epochs
        self.attention_tracker.create_attention_learning_summary()
        
        # Analyze token evolution for important tokens
        common_tokens = ['the', 'a', 'an', 'is', 'are', 'dog', 'cat', 'person', 'man', 'woman']
        
        for token_text in common_tokens:
            # Find token ID
            token_ids = self.attention_tracker.tokenizer.encode(token_text)
            if token_ids:
                token_id = token_ids[0]  # Take first encoding
                
                try:
                    self.attention_tracker.create_token_evolution_plot(
                        token_text=token_text,
                        token_id=token_id
                    )
                except Exception as e:
                    print(f"⚠️  Could not create evolution plot for '{token_text}': {e}")
        
        # Save compression analysis
        if hasattr(self, '_compression_ratios') and self._compression_ratios:
            compression_stats = {
                'mean_ratio': float(np.mean(self._compression_ratios)),
                'std_ratio': float(np.std(self._compression_ratios)),
                'min_ratio': float(np.min(self._compression_ratios)),
                'max_ratio': float(np.max(self._compression_ratios)),
                'total_samples': len(self._compression_ratios),
                'compression_method': self.compression_method
            }
            
            import json
            with open(self.save_dir / "compression_stats.json", 'w') as f:
                json.dump(compression_stats, f, indent=2)
            
            print(f"💾 Saved compression statistics to {self.save_dir / 'compression_stats.json'}")
        
        print(f"✅ Final analysis complete! Check {self.save_dir} for all outputs")

# Integration example for your training loop
def integrate_with_existing_training():
    """Example of how to integrate with existing BitMar training"""
    
    # Example usage in your training loop:
    """
    # Initialize enhanced trainer
    config = {
        'compressed_vision_dim': 128,
        'vision_compression_method': 'linear_projection',
        'track_attention_every_n_steps': 50,
        'save_attention_every_n_epochs': 1
    }
    
    enhanced_trainer = BitMarTrainingWithTracking(
        model=your_bitmar_model,
        config=config,
        save_dir="./bitmar_training_analysis"
    )
    
    # In your training loop:
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            
            # Enhanced training step
            results = enhanced_trainer.training_step(
                batch=batch,
                epoch=epoch,
                step=step,
                global_step=epoch * len(dataloader) + step
            )
            
            # Your existing training code
            loss = results['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Log compression info
            if step % 100 == 0:
                print(f"Compression ratio: {results['compression_ratio']:.2f}x")
                print(f"Attention tracked: {results['attention_tracked']}")
        
        # End of epoch processing
        enhanced_trainer.end_of_epoch_processing(epoch)
    
    # Final analysis
    enhanced_trainer.generate_final_analysis()
    """
    
    print("Integration example ready! See function docstring for usage.")

def main():
    """Main training function with command line interface"""
    import argparse
    import yaml
    import wandb
    from dotenv import load_dotenv
    import os
    
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="BitMar Training with Attention Tracking")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--wandb_project", type=str, default="bitmar-training", help="W&B project name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--track_attention_every_n_steps", type=int, default=50, help="Save attention every N steps")
    parser.add_argument("--save_attention_every_n_epochs", type=int, default=1, help="Generate visualizations every N epochs")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args
    config['training']['max_epochs'] = args.epochs
    config['track_attention_every_n_steps'] = args.track_attention_every_n_steps
    config['save_attention_every_n_epochs'] = args.save_attention_every_n_epochs
    
    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        config=config,
        name=f"bitmar-ultra-tiny-{args.epochs}epochs"
    )
    
    print(f"🚀 Starting BitMar training with:")
    print(f"   - Config: {args.config}")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - W&B Project: {args.wandb_project}")
    print(f"   - Attention tracking every {args.track_attention_every_n_steps} steps")
    
    # TODO: Import and initialize your actual BitMar model here
    # from src.models.bitmar import BitMarModel
    # model = BitMarModel(config)
    
    # Initialize enhanced trainer
    enhanced_trainer = BitMarTrainingWithTracking(
        model=None,  # Replace with actual model
        config=config,
        save_dir=config.get('output', {}).get('checkpoint_dir', './training_outputs')
    )
    
    print("⚠️  Note: Model initialization needs to be implemented.")
    print("   Please add your BitMar model import and initialization.")
    print("   See the integrate_with_existing_training() function for the complete training loop.")
    
    # For now, show what would happen
    integrate_with_existing_training()

if __name__ == "__main__":
    main()
