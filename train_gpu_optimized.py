#!/usr/bin/env python3
"""
GPU-Optimized BitMar Training Script
Fixes dimension issues and ensures proper GPU utilization
"""

import logging
import torch
import yaml
from train_bitmar import BitMarTrainer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_gpu_optimized_config():
    """Create a GPU-optimized configuration"""
    config = {
        'model': {
            # Simplified model architecture for stability
            'vocab_size': 50257,
            'text_encoder_dim': 256,
            'text_encoder_layers': 2,  # Reduced for stability
            'text_encoder_heads': 4,

            'text_decoder_dim': 256,
            'text_decoder_layers': 2,  # Reduced for stability
            'text_decoder_heads': 4,

            # Simplified vision processing
            'vision_encoder_dim': 768,
            'vision_latent_size': 128,
            'vision_hidden_size': 64,
            'vision_compression_method': "simple_linear",
            'vision_spatial_pooling': True,
            'vision_pool_size': 2,

            # Minimal fusion
            'fusion_hidden_size': 128,
            'fusion_num_heads': 4,
            'fusion_num_layers': 1,
            'fusion_num_queries': 8,

            # Minimal memory
            'memory_size': 8,
            'episode_dim': 64,
            'memory_alpha': 0.1,
            'direct_writing': True,
            'memory_compression': True,

            'max_seq_len': 256,
            'dropout': 0.1,
            'text_encoder_name': "gpt2"
        },

        'training': {
            'max_epochs': 3,  # Start with fewer epochs
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'gradient_clip_val': 1.0,
            'device': 'cuda:0'
        },

        'data': {
            'dataset_dir': "../babylm_dataset",
            'max_seq_length': 256,
            'batch_size': 8,  # Conservative batch size
            'num_workers': 2,  # Conservative workers
            'pin_memory': True,
            'text_encoder_name': 'gpt2',
            'persistent_workers': True,
            'validation_datasets': ['glue/sst2'],
            'prefetch_factor': 2,
            'drop_last': True,
            'memory_efficient_loading': False,
            'non_blocking': True,
            'shuffle': True,
            'timeout': 30,
            'text_ratio': 0.5
        },

        'output': {
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
            'attention_dir': 'attention_analysis',
            'memory_dir': 'memory_analysis',
            'results_dir': 'results'
        },

        'wandb': {
            'project': 'bitmar-gpu-optimized',
            'log_every_n_steps': 10
        },

        'quick_training_mode': {
            'enabled': True,
            'max_samples_per_epoch': 1000,  # Very limited for testing
            'optimizations': {
                'aggressive_image_compression': True,
                'mixed_precision_training': True,
                'compiled_model': False,  # Disabled for stability
                'cached_vision_features': False
            }
        }
    }

    return config


def main():
    """Main training function"""
    logger.info("🚀 Starting GPU-Optimized BitMar Training")

    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available! This script requires GPU.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"✅ Using GPU: {gpu_name}")

    # Create optimized config
    config = create_gpu_optimized_config()
    logger.info("✅ Created GPU-optimized configuration")

    try:
        # Create trainer
        trainer = BitMarTrainer(config)
        logger.info("✅ BitMar trainer initialized")

        # Start training with limited samples for testing
        trainer.train(max_samples=5000)  # Very limited for initial test
        logger.info("🎉 Training completed successfully!")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
