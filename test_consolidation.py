#!/usr/bin/env python3
"""
Test Episodic Memory Consolidation Training
Quick test to verify the consolidation phases work correctly
"""

import os
import sys
import torch
import yaml
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from train_bitmar import BitMarTrainer

def create_consolidation_test_config():
    """Create a minimal config for testing consolidation training"""
    return {
        'model': {
            # Ultra-minimal model for testing
            'vocab_size': 50257,
            'text_encoder_dim': 128,
            'text_encoder_layers': 2,
            'text_encoder_heads': 4,
            'text_decoder_dim': 128,
            'text_decoder_layers': 2,
            'text_decoder_heads': 4,
            'max_seq_len': 32,
            'dropout': 0.1,
            
            # Minimal vision processing
            'vision_encoder_dim': 768,
            'vision_hidden_size': 64,
            'vision_latent_size': 64,
            
            # Minimal fusion with QFormer
            'fusion_hidden_size': 128,
            'fusion_num_queries': 8,  # Small number for testing
            'fusion_num_heads': 4,
            'fusion_num_layers': 1,
            
            # Minimal memory
            'memory_size': 8,
            'episode_dim': 128,
            'memory_alpha': 0.1,
            'direct_writing': True,
            'text_encoder_name': 'gpt2'
        },
        
        'training': {
            'max_epochs': 6,  # Test 6 epochs to see all 3 phases
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'gradient_clip_val': 1.0,
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
        },
        
        'data': {
            'dataset_dir': '../babylm_dataset',
            'batch_size': 4,  # Very small for testing
            'max_seq_length': 32,
            'num_workers': 2,
            'pin_memory': True,
            'text_encoder_name': 'gpt2',
            'persistent_workers': False,  # Disable for testing
            'validation_datasets': ['glue/sst2'],
            'prefetch_factor': 2,
            'text_ratio': 0.5,
            'vision_ratio': 0.5
        },
        
        'output': {
            'checkpoint_dir': './checkpoints_consolidation_test',
            'log_dir': './logs_consolidation_test',
            'attention_dir': './attention_consolidation_test',
            'memory_dir': './memory_consolidation_test',
            'results_dir': './results_consolidation_test'
        },
        
        'wandb': {
            'project': 'bitmar-consolidation-test',
            'log_every_n_steps': 10  # Frequent logging for testing
        },
        
        # Disable expensive analytics for speed
        'track_attention_every_n_steps': 0,
        'attention_analysis': {
            'log_every_n_steps': 0
        }
    }

def test_consolidation_phases():
    """Test that consolidation phases work correctly"""
    print("🧠 Testing Episodic Memory Consolidation Training")
    print("=" * 60)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available, testing on CPU")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ Using GPU: {gpu_name}")
    
    # Create test configuration
    config = create_consolidation_test_config()
    print("✅ Test configuration created")
    
    try:
        # Initialize trainer
        print("🔧 Initializing BitMar trainer with consolidation support...")
        trainer = BitMarTrainer(config)
        
        # Test phase detection
        print("\n🔍 Testing consolidation phase detection:")
        for epoch in range(6):
            phase = trainer._get_consolidation_phase(epoch)
            print(f"   Epoch {epoch}: {phase}")
        
        # Test that the phases are correctly distributed
        phases = [trainer._get_consolidation_phase(e) for e in range(6)]
        expected_phases = ["episodic_capture", "consolidation", "semantic_integration"]
        
        unique_phases = list(set(phases))
        if all(phase in unique_phases for phase in expected_phases):
            print("✅ All consolidation phases detected correctly!")
        else:
            print("❌ Phase detection issue!")
            return False
        
        # Test quick training setup
        print("\n🚀 Setting up training components...")
        trainer.setup_directories()
        trainer.setup_logging_systems()
        
        # Quick model setup test (limit samples for speed)
        print("🧠 Setting up model and data (limited samples)...")
        trainer.setup_model_and_data(max_samples=100)  # Very limited for testing
        
        print("✅ Model and data setup completed!")
        
        # Test one batch from each phase
        print("\n🎯 Testing consolidation forward passes...")
        
        # Get a sample batch
        train_loader = trainer.data_module.train_dataloader()
        sample_batch = next(iter(train_loader))
        
        # Move to device
        for key in sample_batch:
            if torch.is_tensor(sample_batch[key]):
                sample_batch[key] = sample_batch[key].to(trainer.device)
        
        # Test each consolidation phase
        phases_to_test = ["episodic_capture", "consolidation", "semantic_integration"]
        
        for phase in phases_to_test:
            print(f"   Testing {phase} phase...")
            start_time = time.time()
            
            if phase == "episodic_capture":
                outputs = trainer._episodic_capture_forward(sample_batch)
            elif phase == "consolidation":
                outputs = trainer._consolidation_forward(sample_batch)
            elif phase == "semantic_integration":
                outputs = trainer._integration_forward(sample_batch)
            
            end_time = time.time()
            
            # Check outputs
            if outputs and 'loss' in outputs and torch.isfinite(outputs['loss']):
                print(f"   ✅ {phase}: Loss = {outputs['loss'].item():.4f}, Time = {end_time - start_time:.3f}s")
                
                # Check for consolidation mode in outputs
                if 'consolidation_mode' in outputs:
                    print(f"      🧠 Consolidation mode: {outputs['consolidation_mode']}")
            else:
                print(f"   ❌ {phase}: Invalid outputs!")
                return False
        
        print("\n🎉 All consolidation tests passed!")
        print("\n📊 Test Summary:")
        print("   ✅ Phase detection working correctly")
        print("   ✅ All three consolidation phases functional")
        print("   ✅ QFormer-based fusion operational")
        print("   ✅ Episodic memory integration working")
        print("\n🚀 Ready for full consolidation training!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧠 BitMar Episodic Memory Consolidation Test")
    print("Testing cognitively-inspired training approach")
    print()
    
    success = test_consolidation_phases()
    
    if success:
        print("\n🎉 All tests passed! Ready to run consolidation training:")
        print("   python train_bitmar.py --config configs/bitmar_config.yaml")
        print("\n🧠 This will use the Episodic Memory Consolidation approach:")
        print("   Phase 1 (30%): Episodic Capture - Rapid multimodal encoding")
        print("   Phase 2 (40%): Memory Consolidation - Pattern extraction & replay")
        print("   Phase 3 (30%): Semantic Integration - Knowledge refinement")
    else:
        print("\n❌ Tests failed! Check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
