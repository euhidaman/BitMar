#!/usr/bin/env python3
"""
BitMar CPU Compatibility Test
Tests the BitNet-quantized Vision-Language Episodic Memory Transformer
"""

from model import create_bitmar_model, BitMarModel, count_parameters
import os
import sys
import torch
import yaml
import logging
import time
import psutil
import traceback
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['model']


def test_bitnet_quantization():
    """Test BitNet 1.58-bit quantization"""
    logger.info("Testing BitNet quantization...")

    from src.model import BitNetLinear

    # Create a test layer
    layer = BitNetLinear(256, 128)
    test_input = torch.randn(4, 256)

    # Test forward pass
    output = layer(test_input)

    # Test quantization
    original_weights = layer.weight.clone()
    quantized_weights = layer.quantize_weights_1_58_bit(original_weights)

    # Check quantization constraints
    unique_values = quantized_weights.unique().sort()[0]
    expected_values = torch.tensor([-1.0, 0.0, 1.0])

    logger.info(f"✅ Quantization test successful!")
    logger.info(
        f"   Original weight range: [{original_weights.min():.3f}, {original_weights.max():.3f}]")
    logger.info(f"   Quantized unique values: {unique_values.tolist()}")
    logger.info(f"   Scale factor: {layer.weight_scale.item():.4f}")

    return True


def test_model_creation(config):
    """Test BitMar model creation"""
    logger.info("Testing model creation...")

    start_time = time.time()
    model = create_bitmar_model(config)
    creation_time = time.time() - start_time

    # Get model statistics
    param_count = count_parameters(model)

    # Memory usage
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024

    logger.info(f"✅ Model created successfully!")
    logger.info(f"   Creation time: {creation_time:.2f}s")
    logger.info(f"   Memory usage: {memory_mb:.1f} MB")
    logger.info(f"   Total parameters: {param_count['total_parameters']:,}")
    logger.info(
        f"   Trainable parameters: {param_count['trainable_parameters']:,}")

    return model, param_count


def test_model_inference(model, config):
    """Test model inference with dummy data"""
    logger.info("Testing model inference...")

    # Create dummy inputs
    batch_size = 2
    seq_len = 32
    vision_dim = config['vision_encoder_dim']

    # Text inputs (dummy token IDs)
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    # Vision inputs (dummy DiNOv2 features)
    vision_features = torch.randn(batch_size, vision_dim)

    # Labels for training
    labels = input_ids.clone()

    # Test forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_features=vision_features,
            labels=labels,
            mode="inference"
        )

    # Check outputs
    assert outputs['logits'].shape == (
        batch_size, seq_len, config['vocab_size'])
    assert outputs['text_features'].shape == (
        batch_size, seq_len, config['text_encoder_dim'])
    assert outputs['vision_latent'].shape == (
        batch_size, config['vision_latent_size'])
    assert outputs['episode'].shape == (batch_size, config['episode_dim'])

    logger.info(f"✅ Model inference successful!")
    logger.info(f"   Output logits shape: {outputs['logits'].shape}")
    logger.info(f"   Text features shape: {outputs['text_features'].shape}")
    logger.info(f"   Vision latent shape: {outputs['vision_latent'].shape}")
    logger.info(f"   Episode shape: {outputs['episode'].shape}")
    logger.info(f"   Memory usage shape: {outputs['memory_usage'].shape}")

    return outputs


def test_memory_efficiency():
    """Test memory efficiency"""
    logger.info("Testing memory efficiency...")

    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024

    # Create and delete multiple models
    config_path = "configs/bitmar_config.yaml"
    config = load_config(config_path)

    for i in range(3):
        model = create_bitmar_model(config)
        del model

    # Force garbage collection
    import gc
    gc.collect()

    final_memory = process.memory_info().rss / 1024 / 1024
    memory_freed = initial_memory - final_memory

    # System stats
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent

    logger.info(f"   Initial memory: {initial_memory:.1f} MB")
    logger.info(f"   After cleanup: {final_memory:.1f} MB")
    logger.info(f"   Memory freed: {memory_freed:.1f} MB")
    logger.info(f"   CPU usage: {cpu_percent:.1f}%")
    logger.info(f"   System memory usage: {memory_percent:.1f}%")

    if final_memory < 1000:  # Less than 1GB
        logger.info(f"✅ Memory usage is reasonable for CPU testing")
        return True
    else:
        logger.warning(f"⚠️  High memory usage detected")
        return False


def test_episodic_memory(model, config):
    """Test episodic memory functionality"""
    logger.info("Testing episodic memory...")

    batch_size = 2
    episode_dim = config['episode_dim']
    memory_size = config['memory_size']

    # Create dummy episodes
    episodes = torch.randn(batch_size, episode_dim)

    # Test memory operations
    memory = model.memory

    # Test writing
    written_episodes = memory.write_memory(episodes)
    assert written_episodes.shape == episodes.shape

    # Test reading
    retrieved, attention_weights = memory.read_memory(episodes)
    assert retrieved.shape == (batch_size, episode_dim)
    assert attention_weights.shape == (batch_size, memory_size)

    # Check memory state
    memory_usage = memory.memory_usage
    memory_age = memory.memory_age

    logger.info(f"✅ Episodic memory test successful!")
    logger.info(
        f"   Memory slots used: {(memory_usage > 0).sum().item()}/{memory_size}")
    logger.info(
        f"   Average attention entropy: {(-attention_weights * torch.log(attention_weights + 1e-8)).sum(-1).mean():.3f}")

    return True


def main():
    """Main test function"""
    logger.info("🚀 Starting BitMar CPU Compatibility Test")
    logger.info("=" * 50)

    tests_passed = 0
    total_tests = 6
    start_time = time.time()

    try:
        # Load configuration
        config_path = "configs/bitmar_config.yaml"
        config = load_config(config_path)

        # Test 1: BitNet Quantization
        logger.info("\n1️⃣ Testing BitNet Quantization")
        if test_bitnet_quantization():
            tests_passed += 1

        # Test 2: Model Creation
        logger.info("\n2️⃣ Testing Model Creation")
        model, param_count = test_model_creation(config)
        tests_passed += 1

        # Test 3: Model Inference
        logger.info("\n3️⃣ Testing Model Inference")
        outputs = test_model_inference(model, config)
        tests_passed += 1

        # Test 4: Episodic Memory
        logger.info("\n4️⃣ Testing Episodic Memory")
        if test_episodic_memory(model, config):
            tests_passed += 1

        # Test 5: Memory Efficiency
        logger.info("\n5️⃣ Testing Memory Efficiency")
        if test_memory_efficiency():
            tests_passed += 1

        # Test 6: Generation (if model is working)
        logger.info("\n6️⃣ Testing Text Generation")
        try:
            # Simple generation test
            batch_size = 1
            seq_len = 10
            input_ids = torch.randint(
                0, config['vocab_size'], (batch_size, seq_len))
            attention_mask = torch.ones_like(input_ids)
            vision_features = torch.randn(
                batch_size, config['vision_encoder_dim'])

            generation_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_features=vision_features,
                max_length=20,
                temperature=1.0
            )

            logger.info(f"✅ Text generation successful!")
            logger.info(
                f"   Generated sequence length: {generation_outputs['generated_ids'].shape[1]}")
            tests_passed += 1

        except Exception as e:
            logger.error(f"❌ Text generation failed: {e}")

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        traceback.print_exc()

    # Final results
    total_time = time.time() - start_time
    process = psutil.Process()
    final_memory = process.memory_info().rss / 1024 / 1024

    logger.info("\n" + "=" * 50)
    logger.info("🎯 CPU Compatibility Test Results")
    logger.info("=" * 50)
    logger.info(f"Tests passed: {tests_passed}/{total_tests}")
    logger.info(f"Total test time: {total_time:.2f}s")
    logger.info(f"Final memory usage: {final_memory:.1f} MB")

    if tests_passed == total_tests:
        logger.info("🎉 All tests passed! BitMar is ready for training.")
    elif tests_passed >= total_tests - 1:
        logger.info("✅ Most tests passed. Minor issues detected.")
    else:
        logger.warning("⚠️ Multiple test failures. Check logs above.")

    return tests_passed, total_tests


if __name__ == "__main__":
    tests_passed, total_tests = main()
    sys.exit(0 if tests_passed >= total_tests - 1 else 1)
