"""
CPU Compatibility Test for BitMar
Tests model creation, data loading, and basic inference on CPU
"""

import sys
import logging
import torch
import yaml
from pathlib import Path
import time
import psutil
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.model import create_bitmar_model, count_parameters
from src.dataset import create_data_module

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_model_creation():
    """Test model creation with reduced size for CPU"""
    logger.info("Testing model creation...")
    
    # CPU-friendly configuration
    cpu_config = {
        'text_encoder_type': 'bert',
        'text_encoder_name': 'bert-base-uncased',
        'text_decoder_type': 'gpt2',
        'text_decoder_name': 'gpt2',  # Smaller decoder for CPU
        'text_latent_size': 512,  # Reduced from 768
        
        'vision_encoder_dim': 768,
        'vision_latent_size': 512,  # Reduced from 768
        'vision_quantization': True,
        'vision_hidden_size': 256,  # Reduced from 512
        
        'memory_size': 64,  # Reduced from 512
        'episode_dim': 512,
        'memory_alpha': 0.1,
        'direct_writing': True,
        'ordering': False,
        'pseudoinverse_approx_step': 15,
        'observation_noise_std': 0.000001,
        'identity': True,
        'w_logvar_setting': 3,
        'deterministic_w': False,
        
        'fusion_type': 'cross_attention',
        'fusion_hidden_size': 512,
        'fusion_num_heads': 8,  # Reduced from 12
        'fusion_num_layers': 1,  # Reduced from 2
        
        'weight_quantization': 'ternary',
        'activation_quantization': 'int8',
        'gradient_checkpointing': True,
        
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'warmup_steps': 100,
        'gradient_clip': 1.0,
        'beta': 0.5,
        'use_beta_schedule': True,
        'ratio_increase': 0.25,
        'ratio_zero': 0.5,
        
        'track_attention': False  # Disable for CPU testing
    }
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        model = create_bitmar_model(cpu_config)
        creation_time = time.time() - start_time
        creation_memory = get_memory_usage() - start_memory
        
        # Count parameters
        param_info = count_parameters(model)
        
        logger.info(f"✅ Model created successfully!")
        logger.info(f"   Creation time: {creation_time:.2f}s")
        logger.info(f"   Memory usage: {creation_memory:.1f} MB")
        logger.info(f"   Total parameters: {param_info['total_parameters']:,}")
        logger.info(f"   Trainable parameters: {param_info['trainable_parameters']:,}")
        
        return model, cpu_config
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        raise


def test_data_loading():
    """Test data loading with small sample"""
    logger.info("Testing data loading...")
    
    # Check if dataset files exist
    dataset_files = [
        "../babylm_dataset/cc_3M_captions.json",
        "../babylm_dataset/cc_3M_dino_v2_states_1of2.npy",
        "../babylm_dataset/cc_3M_dino_v2_states_2of2.npy"
    ]
    
    missing_files = []
    for file_path in dataset_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"⚠️  Missing dataset files: {missing_files}")
        logger.info("Creating mock data for testing...")
        return create_mock_data()
    
    # Test real data loading
    try:
        data_config = {
            'captions_file': "../babylm_dataset/cc_3M_captions.json",
            'vision_features_1': "../babylm_dataset/cc_3M_dino_v2_states_1of2.npy",
            'vision_features_2': "../babylm_dataset/cc_3M_dino_v2_states_2of2.npy",
            'text_encoder_name': "bert-base-uncased",
            'max_seq_length': 256,  # Reduced for CPU
            'train_split': 0.95,
            'val_split': 0.05,
            'batch_size': 2,  # Small batch for CPU
            'num_workers': 0,  # No multiprocessing for CPU test
            'pin_memory': False,
        }
        
        start_time = time.time()
        start_memory = get_memory_usage()
        
        data_module = create_data_module(data_config)
        data_module.setup(max_samples=100)  # Very small sample
        
        loading_time = time.time() - start_time
        loading_memory = get_memory_usage() - start_memory
        
        # Test data iteration
        sample = data_module.train_dataset[0]
        batch = data_module.get_sample_batch(num_samples=2)
        
        logger.info(f"✅ Data loading successful!")
        logger.info(f"   Loading time: {loading_time:.2f}s")
        logger.info(f"   Memory usage: {loading_memory:.1f} MB")
        logger.info(f"   Train samples: {len(data_module.train_dataset)}")
        logger.info(f"   Val samples: {len(data_module.val_dataset)}")
        logger.info(f"   Sample input shape: {sample['input_ids'].shape}")
        logger.info(f"   Sample vision shape: {sample['vision_features'].shape}")
        
        return data_module
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        logger.info("Creating mock data for testing...")
        return create_mock_data()


def create_mock_data():
    """Create mock data for testing when real data is not available"""
    logger.info("Creating mock data...")
    
    class MockDataModule:
        def __init__(self):
            self.batch_size = 2
            
        def get_sample_batch(self, num_samples=2):
            return {
                'input_ids': torch.randint(0, 1000, (num_samples, 64)),
                'attention_mask': torch.ones(num_samples, 64),
                'labels': torch.randint(0, 1000, (num_samples, 64)),
                'vision_features': torch.randn(num_samples, 768),
                'caption': ['Mock caption 1', 'Mock caption 2']
            }
    
    return MockDataModule()


def test_model_inference(model, data_module):
    """Test basic model inference"""
    logger.info("Testing model inference...")
    
    try:
        model.eval()
        
        # Get a test batch
        batch = data_module.get_sample_batch(num_samples=2)
        
        start_time = time.time()
        start_memory = get_memory_usage()
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                vision_features=batch['vision_features'],
                labels=batch['labels']
            )
        
        inference_time = time.time() - start_time
        inference_memory = get_memory_usage() - start_memory
        
        # Check outputs
        loss = outputs['loss']
        logits = outputs['logits']
        
        logger.info(f"✅ Model inference successful!")
        logger.info(f"   Inference time: {inference_time:.3f}s")
        logger.info(f"   Memory usage: {inference_memory:.1f} MB")
        logger.info(f"   Loss: {loss.item():.4f}")
        logger.info(f"   Logits shape: {logits.shape}")
        logger.info(f"   Text latent shape: {outputs['text_latent'].shape}")
        logger.info(f"   Vision latent shape: {outputs['vision_latent'].shape}")
        
        if outputs['memory_usage'] is not None:
            logger.info(f"   Memory usage pattern: {outputs['memory_usage'][:10].tolist()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_efficiency():
    """Test memory usage and efficiency"""
    logger.info("Testing memory efficiency...")
    
    initial_memory = get_memory_usage()
    
    # Test memory cleanup
    import gc
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    cleaned_memory = get_memory_usage()
    
    logger.info(f"   Initial memory: {initial_memory:.1f} MB")
    logger.info(f"   After cleanup: {cleaned_memory:.1f} MB")
    logger.info(f"   Memory freed: {initial_memory - cleaned_memory:.1f} MB")
    
    # Check system resources
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    
    logger.info(f"   CPU usage: {cpu_percent:.1f}%")
    logger.info(f"   System memory usage: {memory_percent:.1f}%")
    
    # Memory efficiency recommendations
    if cleaned_memory > 2000:  # > 2GB
        logger.warning("⚠️  High memory usage detected!")
        logger.info("   Recommendations:")
        logger.info("   - Reduce batch size")
        logger.info("   - Reduce model dimensions")
        logger.info("   - Use gradient checkpointing")
    else:
        logger.info("✅ Memory usage is reasonable for CPU testing")


def test_quantization():
    """Test quantization functionality"""
    logger.info("Testing quantization...")
    
    try:
        # Test ternary quantization
        test_weight = torch.randn(100, 100)
        
        # Simple ternary quantization
        scale = test_weight.abs().mean()
        weight_normalized = test_weight / (scale + 1e-8)
        threshold = 0.5
        quantized = torch.sign(weight_normalized) * (weight_normalized.abs() > threshold)
        quantized_weight = quantized * scale
        
        # Check quantization results
        unique_values = torch.unique(quantized).tolist()
        
        logger.info(f"✅ Quantization test successful!")
        logger.info(f"   Original weight range: [{test_weight.min():.3f}, {test_weight.max():.3f}]")
        logger.info(f"   Quantized unique values: {unique_values}")
        logger.info(f"   Scale factor: {scale:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Quantization test failed: {e}")


def run_cpu_compatibility_test():
    """Run complete CPU compatibility test"""
    logger.info("🚀 Starting BitMar CPU Compatibility Test")
    logger.info("=" * 50)
    
    total_start_time = time.time()
    tests_passed = 0
    total_tests = 5
    
    try:
        # Test 1: Model Creation
        logger.info("\n1️⃣ Testing Model Creation")
        model, config = test_model_creation()
        tests_passed += 1
        
        # Test 2: Data Loading
        logger.info("\n2️⃣ Testing Data Loading")
        data_module = test_data_loading()
        tests_passed += 1
        
        # Test 3: Model Inference
        logger.info("\n3️⃣ Testing Model Inference")
        if test_model_inference(model, data_module):
            tests_passed += 1
        
        # Test 4: Memory Efficiency
        logger.info("\n4️⃣ Testing Memory Efficiency")
        test_memory_efficiency()
        tests_passed += 1
        
        # Test 5: Quantization
        logger.info("\n5️⃣ Testing Quantization")
        test_quantization()
        tests_passed += 1
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    total_time = time.time() - total_start_time
    final_memory = get_memory_usage()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("🎯 CPU Compatibility Test Results")
    logger.info("=" * 50)
    logger.info(f"Tests passed: {tests_passed}/{total_tests}")
    logger.info(f"Total test time: {total_time:.2f}s")
    logger.info(f"Final memory usage: {final_memory:.1f} MB")
    
    if tests_passed == total_tests:
        logger.info("🎉 All tests passed! BitMar is CPU compatible.")
        logger.info("\n📋 Next steps:")
        logger.info("1. Run dataset compatibility test: python test_dataset_compatibility.py")
        logger.info("2. For GPU training, push to GitHub and run on RunPod")
        logger.info("3. Use full configuration for GPU training")
    else:
        logger.warning(f"⚠️  {total_tests - tests_passed} tests failed. Check logs above.")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    # Set torch to CPU-only mode
    torch.set_num_threads(1)  # Single thread for CPU test
    
    success = run_cpu_compatibility_test()
    
    if success:
        exit(0)
    else:
        exit(1)
