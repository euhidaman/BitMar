#!/usr/bin/env python3
"""
Speed Test Script for BitMar Training
Tests the fixes for 160+ hour per epoch issue
"""

import time
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from train_human_learning import HumanLikeLearningTrainer, create_fast_human_learning_config

def test_training_speed():
    """Test training speed with optimizations"""
    print("🚀 Testing BitMar Training Speed")
    print("=" * 50)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This test requires GPU.")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ Using GPU: {gpu_name}")
    
    # Create ultra-fast config for testing
    config = create_fast_human_learning_config()
    
    # Override with even more aggressive settings for speed test
    config['model'].update({
        'text_encoder_layers': 2,  # Minimal layers
        'text_decoder_layers': 2,
        'fusion_num_layers': 1,
        'memory_size': 32,  # Tiny memory
        'episode_dim': 32
    })
    
    config['data'].update({
        'batch_size': 256,  # Large batch
        'max_seq_length': 32,  # Very short sequences
        'num_workers': 8,
        'max_samples_per_epoch': 1000  # Very limited for speed test
    })
    
    config['training']['max_epochs'] = 1  # Single epoch test
    
    print("📊 Speed Test Configuration:")
    print(f"   Model: {config['model']['text_encoder_layers']} layers")
    print(f"   Batch Size: {config['data']['batch_size']}")
    print(f"   Sequence Length: {config['data']['max_seq_length']}")
    print(f"   Samples: {config['data']['max_samples_per_epoch']}")
    print()
    
    try:
        # Initialize trainer
        print("🔧 Initializing trainer...")
        start_time = time.time()
        
        trainer = HumanLikeLearningTrainer(config)
        trainer.setup_directories()
        trainer.setup_logging_systems()
        trainer.setup_model_and_data(max_samples=1000)
        
        init_time = time.time() - start_time
        print(f"✅ Initialization took: {init_time:.2f} seconds")
        
        # Test one epoch
        print("🏃‍♂️ Testing one epoch...")
        epoch_start = time.time()
        
        train_metrics = trainer.train_epoch_with_human_learning(0)
        
        epoch_time = time.time() - epoch_start
        
        print(f"✅ Epoch completed in: {epoch_time:.2f} seconds")
        print(f"📉 Train Loss: {train_metrics['train_loss']:.4f}")
        
        # Calculate estimated full training time
        full_training_estimate = epoch_time * 10  # 10 epochs
        hours_estimate = full_training_estimate / 3600
        
        print()
        print("📈 SPEED TEST RESULTS:")
        print(f"   Single Epoch: {epoch_time:.2f} seconds")
        print(f"   Estimated 10 Epochs: {full_training_estimate:.2f} seconds ({hours_estimate:.2f} hours)")
        
        if hours_estimate < 5:
            print("🎉 SUCCESS! Training should complete in reasonable time!")
        elif hours_estimate < 24:
            print("⚠️  MODERATE: Training will take several hours but manageable")
        else:
            print("❌ STILL TOO SLOW: Need more optimizations")
        
        return epoch_time, hours_estimate
        
    except Exception as e:
        print(f"❌ Speed test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Run speed test"""
    print("🧪 BitMar Training Speed Test")
    print("Testing fixes for 160+ hour per epoch issue")
    print()
    
    epoch_time, total_estimate = test_training_speed()
    
    if epoch_time is not None:
        print()
        print("🎯 RECOMMENDATION:")
        if total_estimate < 2:
            print("   Use the human learning script with full dataset")
        elif total_estimate < 12:
            print("   Use medium-sized batches and dataset")
        else:
            print("   Use smaller model or further optimizations needed")
    
    print("\n✅ Speed test completed!")

if __name__ == "__main__":
    main()
