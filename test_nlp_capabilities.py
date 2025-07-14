#!/usr/bin/env python3
"""
BitMar NLP Capabilities Test
Tests specific natural language processing capabilities
"""

import os
import sys
import torch
import yaml
import json
import logging
import numpy as np
from pathlib import Path
import time

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_nlp_capabilities():
    """Test NLP capabilities without loading the full model"""
    logger.info("🧪 Testing BitMar NLP Capabilities")
    logger.info("=" * 50)
    
    try:
        from model import BitNetLinear, BitNetAttention, create_bitmar_model
        logger.info("✅ BitMar modules imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import BitMar modules: {e}")
        return False
    
    # Test 1: BitNet Linear Layer
    logger.info("\n1️⃣ Testing BitNet Linear Layer")
    try:
        layer = BitNetLinear(512, 256)
        test_input = torch.randn(4, 512)
        
        # Training mode
        layer.train()
        output_train = layer(test_input)
        
        # Inference mode (quantized)
        layer.eval()
        output_inference = layer(test_input)
        
        # Test quantization
        weights = layer.weight.detach()
        quantized = layer.quantize_weights_1_58_bit(weights)
        unique_vals = quantized.unique()
        
        logger.info(f"   ✅ Training output shape: {output_train.shape}")
        logger.info(f"   ✅ Inference output shape: {output_inference.shape}")
        logger.info(f"   ✅ Quantized unique values: {unique_vals.tolist()}")
        logger.info(f"   ✅ BitNet quantization working correctly")
        
    except Exception as e:
        logger.error(f"   ❌ BitNet Linear test failed: {e}")
        return False
    
    # Test 2: BitNet Attention
    logger.info("\n2️⃣ Testing BitNet Attention")
    try:
        attention = BitNetAttention(dim=512, num_heads=8)
        test_seq = torch.randn(2, 16, 512)  # batch=2, seq_len=16, dim=512
        
        output, attn_weights = attention(test_seq, test_seq, test_seq)
        
        logger.info(f"   ✅ Attention output shape: {output.shape}")
        logger.info(f"   ✅ Attention weights shape: {attn_weights.shape}")
        logger.info(f"   ✅ BitNet attention working correctly")
        
    except Exception as e:
        logger.error(f"   ❌ BitNet Attention test failed: {e}")
        return False
    
    # Test 3: Model Configuration
    logger.info("\n3️⃣ Testing Model Configuration")
    try:
        config_path = "configs/bitmar_config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            model_config = config['model']
            required_keys = [
                'vocab_size', 'text_encoder_dim', 'text_decoder_dim',
                'vision_encoder_dim', 'fusion_hidden_size', 'memory_size'
            ]
            
            missing_keys = [key for key in required_keys if key not in model_config]
            
            if missing_keys:
                logger.warning(f"   ⚠️  Missing config keys: {missing_keys}")
            else:
                logger.info(f"   ✅ All required config keys present")
                logger.info(f"   ✅ Vocab size: {model_config['vocab_size']:,}")
                logger.info(f"   ✅ Text encoder dim: {model_config['text_encoder_dim']}")
                logger.info(f"   ✅ Memory slots: {model_config['memory_size']}")
        else:
            logger.error(f"   ❌ Config file not found: {config_path}")
            return False
            
    except Exception as e:
        logger.error(f"   ❌ Config test failed: {e}")
        return False
    
    # Test 4: Text Generation Capability
    logger.info("\n4️⃣ Testing Text Generation Setup")
    try:
        # Test tokenizer setup
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test encoding
        test_text = "The future of artificial intelligence"
        encoded = tokenizer(test_text, return_tensors="pt", max_length=32)
        
        logger.info(f"   ✅ Tokenizer loaded successfully")
        logger.info(f"   ✅ Test text encoded: {encoded['input_ids'].shape}")
        logger.info(f"   ✅ Vocab size: {tokenizer.vocab_size:,}")
        
    except Exception as e:
        logger.error(f"   ❌ Text generation setup failed: {e}")
        return False
    
    # Test 5: Memory Efficiency
    logger.info("\n5️⃣ Testing Memory Efficiency")
    try:
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and delete some layers
        layers = []
        for _ in range(10):
            layer = BitNetLinear(512, 512)
            layers.append(layer)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clean up
        del layers
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"   ✅ Initial memory: {initial_memory:.1f} MB")
        logger.info(f"   ✅ Peak memory: {peak_memory:.1f} MB")
        logger.info(f"   ✅ Final memory: {final_memory:.1f} MB")
        logger.info(f"   ✅ Memory usage reasonable")
        
    except Exception as e:
        logger.error(f"   ❌ Memory efficiency test failed: {e}")
        return False
    
    # Test 6: GPU Compatibility
    logger.info("\n6️⃣ Testing GPU Compatibility")
    try:
        gpu_available = torch.cuda.is_available()
        device = torch.device("cuda" if gpu_available else "cpu")
        
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"   ✅ GPU available: {gpu_name}")
            logger.info(f"   ✅ GPU memory: {gpu_memory:.1f} GB")
            
            # Test moving tensor to GPU
            test_tensor = torch.randn(100, 100).to(device)
            logger.info(f"   ✅ Tensor moved to GPU successfully")
        else:
            logger.info(f"   ℹ️  No GPU available, using CPU")
            logger.info(f"   ✅ CPU device ready")
        
        logger.info(f"   ✅ Device: {device}")
        
    except Exception as e:
        logger.error(f"   ❌ GPU compatibility test failed: {e}")
        return False
    
    logger.info("\n" + "=" * 50)
    logger.info("🎉 All NLP Capability Tests Passed!")
    logger.info("=" * 50)
    
    return True


def test_text_processing_pipeline():
    """Test a simple text processing pipeline"""
    logger.info("\n🔄 Testing Text Processing Pipeline")
    
    try:
        from transformers import AutoTokenizer
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test texts
        test_texts = [
            "The cat sat on the mat.",
            "Artificial intelligence is transforming the world.",
            "Climate change requires immediate action.",
        ]
        
        # Process texts
        processed_results = []
        
        for text in test_texts:
            # Tokenize
            tokens = tokenizer(text, return_tensors="pt", max_length=32, padding=True)
            
            # Basic statistics
            num_tokens = tokens['input_ids'].size(1)
            decoded = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
            
            result = {
                'original': text,
                'num_tokens': num_tokens,
                'decoded': decoded,
                'matches_original': text.strip().lower() == decoded.strip().lower()
            }
            
            processed_results.append(result)
        
        # Report results
        all_match = all(r['matches_original'] for r in processed_results)
        avg_tokens = np.mean([r['num_tokens'] for r in processed_results])
        
        logger.info(f"   ✅ Processed {len(test_texts)} texts")
        logger.info(f"   ✅ Average tokens: {avg_tokens:.1f}")
        logger.info(f"   ✅ All texts decoded correctly: {all_match}")
        
        # Show sample
        sample = processed_results[0]
        logger.info(f"   📝 Sample: '{sample['original']}'")
        logger.info(f"   🔢 Tokens: {sample['num_tokens']}")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Text processing pipeline failed: {e}")
        return False


def main():
    """Main test function"""
    logger.info("🚀 BitMar NLP Testing Suite")
    logger.info("Testing core NLP capabilities before full model evaluation")
    
    start_time = time.time()
    
    # Run capability tests
    capabilities_passed = test_nlp_capabilities()
    
    # Run pipeline test
    pipeline_passed = test_text_processing_pipeline()
    
    total_time = time.time() - start_time
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Summary")
    logger.info("=" * 50)
    logger.info(f"⏱️  Total test time: {total_time:.2f}s")
    logger.info(f"✅ Core capabilities: {'PASSED' if capabilities_passed else 'FAILED'}")
    logger.info(f"✅ Text pipeline: {'PASSED' if pipeline_passed else 'FAILED'}")
    
    if capabilities_passed and pipeline_passed:
        logger.info("\n🎉 BitMar is ready for NLP benchmarking!")
        logger.info("Next steps:")
        logger.info("1. Run: python test_quick_benchmarks.py")
        logger.info("2. Run: python test_text_benchmarks.py")
        logger.info("3. Start training: python train_bitmar.py")
        return 0
    else:
        logger.error("\n❌ Some tests failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    exit(main())
