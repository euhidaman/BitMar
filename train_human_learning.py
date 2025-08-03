#!/usr/bin/env python3
"""
Human-Like Learning BitMar Training Script
Implements progressive learning like humans: Vision → Vision+Text → Text
Optimized for maximum speed to complete 10 epochs in reasonable time
"""

import os
import sys
import logging
import torch
import yaml
import gc
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from train_bitmar import BitMarTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HumanLikeLearningTrainer(BitMarTrainer):
    """BitMar trainer with human-like learning progression"""
    
    def __init__(self, config, device: Optional[str] = None):
        super().__init__(config, device)
        
        # Human-like learning schedule
        self.learning_schedule = self._create_learning_schedule()
        logger.info("🧠 Human-like learning schedule initialized")
        
    def _create_learning_schedule(self):
        """Define the human-like learning progression"""
        total_epochs = self.config['training']['max_epochs']
        
        schedule = {
            # Phase 1: Vision-only learning (like human babies focusing on visual world)
            'phase_1_vision': {
                'epochs': list(range(0, 3)),  # Epochs 0, 1, 2
                'data_type': 'vision_only',
                'description': '👁️ Vision-only learning (like babies learning to see)',
                'vision_ratio': 1.0,
                'text_ratio': 0.0
            },
            
            # Phase 2: Vision+Text integration (like children learning language with visual context)
            'phase_2_multimodal': {
                'epochs': list(range(3, 8)),  # Epochs 3, 4, 5, 6, 7
                'data_type': 'vision_text',
                'description': '🧠 Vision+Text integration (like children learning language)',
                'vision_ratio': 0.6,  # More vision to maintain visual grounding
                'text_ratio': 0.4
            },
            
            # Phase 3: Text mastery (like adults perfecting language skills)
            'phase_3_text': {
                'epochs': list(range(8, 10)),  # Epochs 8, 9
                'data_type': 'text_only',
                'description': '📚 Text mastery (like adults perfecting language)',
                'vision_ratio': 0.0,
                'text_ratio': 1.0
            }
        }
        
        return schedule
    
    def get_current_learning_phase(self, epoch: int):
        """Get the current learning phase for the epoch"""
        for phase_name, phase_info in self.learning_schedule.items():
            if epoch in phase_info['epochs']:
                return phase_name, phase_info
        
        # Fallback to multimodal if epoch not found
        return 'phase_2_multimodal', self.learning_schedule['phase_2_multimodal']
    
    def setup_data_for_phase(self, phase_info: Dict):
        """Setup data loader for the current learning phase"""
        logger.info(f"🔄 Setting up data for: {phase_info['description']}")
        
        # Update data config for this phase
        data_config = self.config['data'].copy()
        
        # Set the data ratios for this phase
        data_config['vision_ratio'] = phase_info['vision_ratio']
        data_config['text_ratio'] = phase_info['text_ratio']
        
        # 🚀 ULTRA-AGGRESSIVE OPTIMIZATION for maximum speed
        data_config.update({
            'batch_size': 256,  # Very large batches for maximum GPU utilization
            'max_seq_length': 64,  # Very short sequences for speed
            'num_workers': 12,  # Maximum workers
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 32,  # Extremely aggressive prefetching
            'drop_last': True,
            'non_blocking': True,
            'shuffle': True,
            'timeout': 300,  # Longer timeout
            # CRITICAL: Disable all expensive preprocessing
            'cache_preprocessed': False,
            'use_memory_mapping': False,
            'memory_efficient_loading': False,
            'use_compressed_cache': False,  # Disable caching that slows things down
            'precompute_features': False,  # Disable precomputation
        })
        
        # Phase-specific optimizations with LIMITED SAMPLES for speed
        if phase_info['data_type'] == 'vision_only':
            data_config['text_ratio'] = 0.0
            data_config['vision_ratio'] = 1.0
            data_config['max_samples_per_epoch'] = 25000  # MUCH smaller for speed
            logger.info("👁️ Phase 1: Vision-only learning (25K samples for speed)")
            
        elif phase_info['data_type'] == 'vision_text':
            data_config['text_ratio'] = phase_info['text_ratio']
            data_config['vision_ratio'] = phase_info['vision_ratio']
            data_config['max_samples_per_epoch'] = 35000  # MUCH smaller for speed
            logger.info("🧠 Phase 2: Vision+Text integration (35K samples for speed)")
            
        elif phase_info['data_type'] == 'text_only':
            data_config['text_ratio'] = 1.0
            data_config['vision_ratio'] = 0.0
            data_config['max_samples_per_epoch'] = 50000  # MUCH smaller for speed
            logger.info("📚 Phase 3: Text mastery (50K samples for speed)")
        
        # Create new data module for this phase
        from src.dataset import create_data_module
        self.data_module = create_data_module(data_config)
        self.data_module.setup(max_samples=data_config.get('max_samples_per_epoch'))
        
        logger.info(f"✅ Data setup complete for {phase_info['description']}")
        logger.info(f"📊 Using {data_config.get('max_samples_per_epoch')} samples for maximum speed")
    
    def train_epoch_with_human_learning(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with human-like learning approach"""
        
        # Get current learning phase
        phase_name, phase_info = self.get_current_learning_phase(epoch)
        
        # Print learning phase info
        print(f"\n🎓 EPOCH {epoch+1}/10 - {phase_info['description']}")
        print(f"📊 Data Mix: Vision {phase_info['vision_ratio']:.0%}, Text {phase_info['text_ratio']:.0%}")
        
        # Setup data for this phase (only if changed)
        if not hasattr(self, '_last_phase') or self._last_phase != phase_name:
            self.setup_data_for_phase(phase_info)
            self._last_phase = phase_name
        
        # Adjust learning rate for phase
        self._adjust_lr_for_phase(phase_info, epoch)
        
        # Train the epoch
        return self.train_epoch(epoch)
    
    def _adjust_lr_for_phase(self, phase_info: Dict, epoch: int):
        """Adjust learning rate based on learning phase"""
        base_lr = self.config['training']['learning_rate']
        
        if phase_info['data_type'] == 'vision_only':
            # Higher LR for initial vision learning
            lr_multiplier = 1.5
        elif phase_info['data_type'] == 'vision_text':
            # Standard LR for multimodal learning
            lr_multiplier = 1.0
        elif phase_info['data_type'] == 'text_only':
            # Lower LR for fine-tuning text
            lr_multiplier = 0.5
        else:
            lr_multiplier = 1.0
        
        new_lr = base_lr * lr_multiplier
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        logger.info(f"📈 Learning rate adjusted to {new_lr:.2e} for {phase_info['description']}")
    
    def train_human_like(self, max_samples: Optional[int] = None):
        """Main training loop with human-like learning progression"""
        
        print("🧠 Starting Human-Like Learning Training")
        print("=" * 60)
        print("Phase 1 (Epochs 0-2): Vision-only learning 👁️")
        print("Phase 2 (Epochs 3-7): Vision+Text integration 🧠")
        print("Phase 3 (Epochs 8-9): Text mastery 📚")
        print("=" * 60)
        
        # Setup initial components
        self.setup_directories()
        self.setup_logging_systems()
        self.setup_model_and_data(max_samples)
        
        # Initialize tracking
        self._last_phase = None
        best_val_loss = float('inf')
        
        try:
            for epoch in range(self.config['training']['max_epochs']):
                
                # Train epoch with human-like learning
                train_metrics = self.train_epoch_with_human_learning(epoch)
                
                # Validate
                val_metrics = self.validate_epoch(epoch)
                
                # Save checkpoint
                is_best = val_metrics['val_loss'] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics['val_loss']
                
                self.save_checkpoint(epoch, is_best)
                
                # Update scheduler
                if self.scheduler:
                    self.scheduler.step()
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Print epoch summary
                phase_name, phase_info = self.get_current_learning_phase(epoch)
                print(f"✅ Epoch {epoch+1} completed:")
                print(f"   📊 Phase: {phase_info['description']}")
                print(f"   📉 Train Loss: {train_metrics['train_loss']:.4f}")
                print(f"   📈 Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"   🏆 Best: {'✨ NEW BEST!' if is_best else f'{best_val_loss:.4f}'}")
                print("-" * 50)
            
            print("🎉 Human-Like Learning Training Completed Successfully!")
            return best_val_loss
            
        except KeyboardInterrupt:
            print("🛑 Training interrupted by user")
            return best_val_loss
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def create_fast_human_learning_config():
    """Create optimized configuration for fast human-like learning"""
    return {
        'model': {
            # Optimized model size for speed
            'vocab_size': 50257,
            'text_encoder_dim': 384,  # Balanced size
            'text_encoder_layers': 4,  # Reasonable depth
            'text_encoder_heads': 6,
            'text_decoder_dim': 384,
            'text_decoder_layers': 4,
            'text_decoder_heads': 6,
            'max_seq_len': 128,  # Shorter for speed
            'dropout': 0.1,
            
            # Efficient vision processing
            'vision_encoder_dim': 768,
            'vision_hidden_size': 256,
            'vision_latent_size': 128,
            
            # Streamlined fusion
            'fusion_hidden_size': 256,
            'fusion_num_queries': 32,
            'fusion_num_heads': 8,
            'fusion_num_layers': 2,
            
            # Efficient memory
            'memory_size': 256,
            'episode_dim': 128,
            'memory_alpha': 0.1,
            'direct_writing': True
        },
        
        'training': {
            'max_epochs': 10,  # Human-like learning progression
            'learning_rate': 2e-4,  # Good starting rate
            'weight_decay': 0.01,
            'gradient_clip_val': 1.0,
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
        },
        
        'data': {
            'dataset_dir': '../babylm_dataset',
            'batch_size': 128,  # Large batch for efficiency
            'max_seq_length': 128,  # Short sequences for speed
            'num_workers': 8,  # Maximum workers
            'pin_memory': True,
            'text_encoder_name': 'gpt2',
            'persistent_workers': True,
            'validation_datasets': ['glue/sst2'],
            'prefetch_factor': 16,  # Aggressive prefetching
            'drop_last': True,
            'non_blocking': True,
            'text_ratio': 0.5,  # Will be adjusted per phase
            'vision_ratio': 0.5  # Will be adjusted per phase
        },
        
        'output': {
            'checkpoint_dir': './checkpoints_human_learning',
            'log_dir': './logs_human_learning',
            'attention_dir': './attention_human_learning',
            'memory_dir': './memory_human_learning',
            'results_dir': './results_human_learning'
        },
        
        'wandb': {
            'project': 'bitmar-human-learning',
            'log_every_n_steps': 500  # Less frequent logging for speed
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
        
        # Disable expensive analytics for speed
        'track_attention_every_n_steps': 0,  # Disabled
        'attention_analysis': {
            'log_every_n_steps': 0  # Disabled
        }
    }


def main():
    """Main function for human-like learning"""
    print("🧠 Starting Human-Like Learning BitMar Training")
    print("Mimicking human learning progression: Vision → Vision+Text → Text")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("⚠️ Warning: CUDA not available, training will be slow on CPU")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ Using GPU: {gpu_name}")
    
    # Create optimized config
    config = create_fast_human_learning_config()
    
    # Initialize human-like learning trainer
    trainer = HumanLikeLearningTrainer(config)
    
    # Start human-like training
    best_loss = trainer.train_human_like(max_samples=None)  # Use full dataset
    
    print(f"🏆 Training completed! Best validation loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
