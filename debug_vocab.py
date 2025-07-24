#!/usr/bin/env python3
"""
Debug script to identify vocabulary and token ID issues
"""

import torch
from transformers import AutoTokenizer
from src.dataset import create_data_module
import yaml

def debug_vocabulary_issues():
    """Debug vocabulary and token ID issues"""
    print("🔍 Debugging vocabulary issues...")

    # Load config
    with open('configs/bitmar_ultra_tiny.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print(f"Config vocab_size: {config['model']['vocab_size']}")
    print(f"Config max_seq_length: {config['data']['max_seq_length']}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")
    print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")
    print(f"Tokenizer bos_token_id: {tokenizer.bos_token_id}")

    # Create data module with small sample
    try:
        print("\n🔄 Creating data module...")
        data_module = create_data_module(config['data'])
        print("✅ Data module created successfully")

        print("🔄 Setting up data module...")
        data_module.setup(max_samples=10)
        print("✅ Data module setup complete")

        # Get a few batches and check token IDs
        print("🔄 Getting train dataloader...")
        train_loader = data_module.train_dataloader()
        print(f"✅ Train loader created with {len(train_loader)} batches")

        print("\n📊 Checking token IDs in batches...")
        batch_count = 0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 3:  # Check first 3 batches
                break

            batch_count += 1
            print(f"\n🔍 Processing batch {batch_idx}...")

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            print(f"  Input IDs shape: {input_ids.shape}")
            print(f"  Input IDs min: {input_ids.min().item()}")
            print(f"  Input IDs max: {input_ids.max().item()}")
            print(f"  Attention mask shape: {attention_mask.shape}")

            # Check for out-of-bounds tokens
            vocab_size = config['model']['vocab_size']
            invalid_tokens = (input_ids >= vocab_size) | (input_ids < 0)
            if invalid_tokens.any():
                print(f"  ❌ FOUND INVALID TOKENS!")
                print(f"  Invalid token positions: {invalid_tokens.sum().item()}")
                invalid_ids = input_ids[invalid_tokens]
                print(f"  Invalid token IDs: {invalid_ids.unique().tolist()}")

                # Show where invalid tokens occur
                invalid_positions = torch.where(invalid_tokens)
                print(f"  Invalid positions (batch, seq): {list(zip(invalid_positions[0].tolist()[:5], invalid_positions[1].tolist()[:5]))}")
            else:
                print(f"  ✅ All tokens within vocab range [0, {vocab_size-1}]")

            # Show some example tokens
            print(f"  Sample tokens: {input_ids[0, :10].tolist()}")

            # Check for specific problematic tokens
            if input_ids.max().item() >= vocab_size:
                print(f"  🚨 MAX TOKEN ID {input_ids.max().item()} >= VOCAB SIZE {vocab_size}")
                return False

        if batch_count == 0:
            print("❌ No batches were processed!")
            return False

        print(f"\n✅ Successfully processed {batch_count} batches")

    except Exception as e:
        print(f"❌ Error during data processing: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    debug_vocabulary_issues()
