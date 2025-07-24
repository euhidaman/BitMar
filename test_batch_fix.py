#!/usr/bin/env python3
"""
Test script with smaller batch size to verify the fix
"""

import torch
from transformers import AutoTokenizer
from src.dataset import create_data_module
import yaml

def test_with_small_batch():
    """Test training with smaller batch size"""
    print("🔍 Testing with smaller batch size...")

    # Load config
    with open('configs/bitmar_ultra_tiny.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Temporarily override batch size for testing
    original_batch_size = config['data']['batch_size']
    config['data']['batch_size'] = 4  # Smaller than our 10 samples

    print(f"Original batch_size: {original_batch_size}")
    print(f"Test batch_size: {config['data']['batch_size']}")
    print(f"Max samples: 10")

    try:
        # Create data module with small sample
        data_module = create_data_module(config['data'])
        data_module.setup(max_samples=10)

        # Get train dataloader
        train_loader = data_module.train_dataloader()
        print(f"✅ Train loader created with {len(train_loader)} batches")

        # Test first batch
        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            print(f"✅ Successfully got batch!")
            print(f"  Batch size: {batch['input_ids'].shape[0]}")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            print(f"  Vision features shape: {batch['vision_features'].shape}")
            return True
        else:
            print("❌ Still 0 batches")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_with_small_batch()
