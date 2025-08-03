#!/usr/bin/env python3
"""
Fast training script for BitMar - optimized for maximum speed
Removes all expensive analytics and focuses purely on training performance
"""

import os
import sys
import logging
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from train_bitmar import BitMarTrainer

def create_fast_config():
    """Create optimized configuration for fast training"""
    return {
        'model': {
            'vocab_size': 50257,
            'text_encoder_dim': 256,
            'text_encoder_layers': 3,  # Smaller model for speed
            'text_encoder_heads': 4,
            'text_decoder_dim': 256,
            'text_decoder_layers': 3,  # Smaller model for speed
            'text_decoder_heads': 4,
            'max_seq_len': 256,  # Shorter sequences for speed
            'dropout': 0.1,
            'vision_encoder_dim': 768,
            'vision_hidden_size': 256,
            'vision_latent_size': 128,
            'fusion_hidden_size': 256,
            'fusion_num_queries': 16,  # Fewer queries for speed
            'fusion_num_heads': 4,
            'fusion_num_layers': 2,  # Fewer layers for speed
            'memory_size': 512,  # Smaller memory for speed
            'episode_dim': 96,
            'memory_alpha': 0.1,
            'direct_writing': True
        },
        'training': {
            'max_epochs': 3,  # Fewer epochs for testing
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'gradient_clip_val': 1.0,
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
        },
        'data': {
            'dataset_dir': '../babylm_dataset',
            'batch_size': 64,  # Larger batch for efficiency
            'max_seq_length': 256,  # Shorter sequences
            'num_workers': 4,  # Moderate workers
            'pin_memory': True,
            'text_encoder_name': 'gpt2',
            'persistent_workers': True,
            'validation_datasets': ['glue/sst2'],
            'prefetch_factor': 4,  # Moderate prefetch
            'drop_last': True,
            'non_blocking': True,
            'text_ratio': 0.5
        },
        'output': {
            'checkpoint_dir': './checkpoints_fast',
            'log_dir': './logs_fast',
            'attention_dir': './attention_fast',
            'memory_dir': './memory_fast',
            'results_dir': './results_fast'
        },
        'wandb': {
            'log_every_n_steps': 1000  # Very infrequent logging
        },
        'quick_training_mode': {
            'enabled': True,
            'optimizations': {
                'aggressive_image_compression': True,
                'mixed_precision_training': True,
                'compiled_model': False,  # Skip compilation overhead
                'cached_vision_features': True
            }
        },
        # Disable all expensive analytics
        'track_attention_every_n_steps': 0,  # Disabled
        'attention_analysis': {
            'log_every_n_steps': 0  # Disabled
        }
    }

def main():
    """Fast training main function"""
    print("🚀 Starting FAST BitMar training...")
    
    # Create fast config
    config = create_fast_config()
    
    # Initialize trainer with optimized settings
    trainer = BitMarTrainer(config)
    
    # Setup with minimal samples for speed testing
    trainer.setup_directories()
    trainer.setup_logging_systems()
    trainer.setup_model_and_data(max_samples=10000)  # Limited samples for speed
    
    print("✅ Fast trainer setup complete")
    print(f"Device: {trainer.device}")
    print(f"Model parameters: {trainer.model.numel() if hasattr(trainer.model, 'numel') else 'Unknown'}")
    
    # Training loop with minimal overhead
    try:
        for epoch in range(config['training']['max_epochs']):
            print(f"\n🏃‍♂️ Fast training epoch {epoch+1}/{config['training']['max_epochs']}")
            
            # Train epoch
            train_metrics = trainer.train_epoch(epoch)
            print(f"Epoch {epoch+1} - Train Loss: {train_metrics['train_loss']:.4f}")
            
            # Simple validation (no expensive metrics)
            val_metrics = trainer.validate_epoch(epoch)
            print(f"Epoch {epoch+1} - Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < trainer.best_val_loss
            if is_best:
                trainer.best_val_loss = val_metrics['val_loss']
            trainer.save_checkpoint(epoch, is_best)
            
            # Update scheduler
            if trainer.scheduler:
                trainer.scheduler.step()
        
        print("✅ Fast training completed successfully!")
        
    except KeyboardInterrupt:
        print("🛑 Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
