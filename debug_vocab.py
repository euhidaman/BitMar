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
        data_module = create_data_module(config['data'])
        data_module.setup(max_samples=10)

        # Get a few batches and check token IDs
        train_loader = data_module.train_dataloader()

        print("\n📊 Checking token IDs in batches...")
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 3:  # Check first 3 batches
                break

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            print(f"\nBatch {batch_idx}:")
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
            else:
                print(f"  ✅ All tokens within vocab range [0, {vocab_size-1}]")

            # Show some example tokens
            print(f"  Sample tokens: {input_ids[0, :10].tolist()}")

    except Exception as e:
        print(f"❌ Error creating data module: {e}")
        return False

    return True

if __name__ == "__main__":
    debug_vocabulary_issues()
