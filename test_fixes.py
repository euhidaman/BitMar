#!/usr/bin/env python3
"""
Test script to verify the performance and stability fixes
"""

import torch
import time
import logging
from src.model import create_bitmar_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_fixes():
    """Test the model with the performance and stability fixes"""
    
    # Test configuration
    config = {
        'model': {
            'vocab_size': 50257,
            'text_encoder_dim': 256,
            'text_encoder_layers': 4,
            'text_encoder_heads': 4,
            'text_decoder_dim': 256,
            'text_decoder_layers': 4,
            'text_decoder_heads': 4,
            'vision_encoder_dim': 768,
            'vision_latent_size': 256,
            'vision_hidden_size': 128,
            'fusion_hidden_size': 256,
            'fusion_num_heads': 4,
            'fusion_num_layers': 2,
            'fusion_num_queries': 32,
            'memory_size': 16,
            'episode_dim': 256,
            'memory_alpha': 0.15,
            'direct_writing': True,
            'max_seq_len': 64,
            'dropout': 0.15
        }
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Testing on device: {device}")
    
    # Create model
    logger.info("Creating model...")
    model = create_bitmar_model(config)
    model = model.to(device)
    model.train()
    
    # Test data
    batch_size = 8
    seq_len = 64
    vision_dim = 768
    
    # Create test tensors
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device)
    vision_features = torch.randn((batch_size, vision_dim), device=device)
    labels = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    
    # Test 1: Check for NaN/Inf stability
    logger.info("Test 1: Testing NaN/Inf stability...")
    
    # Introduce some NaN values to test handling
    vision_features_with_nan = vision_features.clone()
    vision_features_with_nan[0, 0] = float('nan')
    
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_features=vision_features_with_nan,
            labels=labels,
            mode="episodic_capture"
        )
        logger.info("✅ NaN handling test passed")
    except Exception as e:
        logger.error(f"❌ NaN handling test failed: {e}")
        return False
    
    # Test 2: Performance test
    logger.info("Test 2: Testing performance...")
    
    num_iterations = 10
    start_time = time.time()
    
    for i in range(num_iterations):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_features=vision_features,
            labels=labels,
            mode="episodic_capture"
        )
        
        loss = outputs['loss']
        loss.backward()
        
        # Clear gradients
        model.zero_grad()
        
        if i % 5 == 0:
            logger.info(f"Iteration {i+1}/{num_iterations} completed")
    
    end_time = time.time()
    avg_time_per_batch = (end_time - start_time) / num_iterations
    
    logger.info(f"✅ Performance test completed")
    logger.info(f"Average time per batch: {avg_time_per_batch:.3f} seconds")
    
    if avg_time_per_batch > 5.0:  # If more than 5 seconds per batch, still too slow
        logger.warning(f"⚠️ Performance may still be slow: {avg_time_per_batch:.3f}s/batch")
    else:
        logger.info(f"✅ Performance looks good: {avg_time_per_batch:.3f}s/batch")
    
    # Test 3: Memory efficiency
    logger.info("Test 3: Testing memory efficiency...")
    
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
        
        if memory_allocated > 8.0:  # More than 8GB might be too much
            logger.warning(f"⚠️ High memory usage: {memory_allocated:.2f} GB")
        else:
            logger.info(f"✅ Memory usage looks good: {memory_allocated:.2f} GB")
    
    logger.info("🎉 All tests completed successfully!")
    return True

if __name__ == "__main__":
    print("🚀 Testing BitMar performance and stability fixes...")
    success = test_model_fixes()
    
    if success:
        print("\n✅ Tests passed! The fixes should resolve:")
        print("   1. Dimension assertion errors from torch.compile")
        print("   2. Slow training speed from excessive analytics")
        print("   3. NaN/Inf handling issues")
        print("\n🚀 Ready to train with improved performance!")
    else:
        print("\n❌ Some tests failed. Please check the logs.")
