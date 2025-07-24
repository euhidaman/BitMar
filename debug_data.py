#!/usr/bin/env python3
"""
Enhanced debug script to identify data loading issues
"""

import torch
from transformers import AutoTokenizer
from src.dataset import create_data_module
import yaml
import os
from pathlib import Path

def debug_data_loading():
    """Debug data loading issues"""
    print("🔍 Debugging data loading issues...")

    # Load config
    with open('configs/bitmar_ultra_tiny.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print(f"Config dataset_dir: {config['data']['dataset_dir']}")
    print(f"Config batch_size: {config['data']['batch_size']}")
    print(f"Config max_seq_length: {config['data']['max_seq_length']}")

    # Check if dataset directory exists
    dataset_dir = Path(config['data']['dataset_dir'])
    print(f"\n📂 Checking dataset directory: {dataset_dir}")

    if not dataset_dir.exists():
        print(f"❌ Dataset directory does not exist: {dataset_dir}")
        return False
    else:
        print(f"✅ Dataset directory exists")

    # Check for required files
    required_files = [
        "cc_3M_captions.json",
        "cc_3M_dino_v2_states_1of2.npy",
        "cc_3M_dino_v2_states_2of2.npy",
        "local_narr_captions.json",
        "local_narr_dino_v2_states.npy"
    ]

    print("\n📋 Checking required dataset files:")
    missing_files = []
    for file_name in required_files:
        file_path = dataset_dir / file_name
        if file_path.exists():
            file_size = file_path.stat().st_size / (1024**2)  # MB
            print(f"  ✅ {file_name} ({file_size:.1f} MB)")
        else:
            print(f"  ❌ {file_name} - MISSING")
            missing_files.append(file_name)

    if missing_files:
        print(f"\n🚨 Missing {len(missing_files)} required files!")
        print("The dataset appears to be incomplete.")
        return False

    # Try to load the dataset directly
    try:
        print("\n🔄 Testing direct dataset loading...")
        from src.dataset import CompleteBabyLMDataset

        dataset = CompleteBabyLMDataset(
            dataset_dir=str(dataset_dir),
            tokenizer_name="gpt2",
            max_seq_length=config['data']['max_seq_length'],
            split="train",
            max_samples=10
        )

        print(f"✅ Dataset loaded successfully with {len(dataset)} samples")

        if len(dataset) == 0:
            print("❌ Dataset is empty!")
            return False

        # Try to get a sample
        print("\n🔍 Testing sample retrieval...")
        sample = dataset[0]
        print(f"  Sample keys: {list(sample.keys())}")
        for key, value in sample.items():
            if torch.is_tensor(value):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)}")

        return True

    except Exception as e:
        print(f"❌ Error loading dataset directly: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_data_loading()
