#!/usr/bin/env python3
"""
Test script to verify BitMar model compiles with PyTorch 2.0
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import yaml
from model import create_bitmar_model

def test_model_compilation():
    """Test if the BitMar model compiles successfully"""
    print("Testing BitMar model compilation...")
    
    # Load configuration
    config_path = "configs/bitmar_100M_tokens.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded config from {config_path}")
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_path}")
        return False
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Create model
        model = create_bitmar_model(config)
        model = model.to(device)
        print("✓ Model created successfully")
        
        # Test model compilation
        if config.get('compile_model', False):
            print("Testing PyTorch 2.0 compilation...")
            compiled_model = torch.compile(model, mode='default')
            print("✓ Model compiled successfully with PyTorch 2.0")
        else:
            print("Model compilation disabled in config")
        
        # Test forward pass with small batch
        print("Testing forward pass...")
        batch_size = 2
        seq_len = 32
        vision_dim = config['vision_encoder_dim']
        
        # Create dummy inputs
        input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        vision_features = torch.randn(batch_size, vision_dim, device=device)
        labels = input_ids.clone()
        
        # Forward pass
        with torch.no_grad():
            if config.get('compile_model', False):
                outputs = compiled_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    vision_features=vision_features,
                    labels=labels,
                    mode="train"
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    vision_features=vision_features,
                    labels=labels,
                    mode="train"
                )
        
        print("✓ Forward pass successful")
        print(f"✓ Loss: {outputs['loss'].item():.4f}")
        
        # Check for NaN values in outputs
        nan_checks = {
            'loss': torch.isnan(outputs['loss']).any(),
            'logits': torch.isnan(outputs['logits']).any(),
            'text_features': torch.isnan(outputs['text_features']).any(),
            'vision_latent': torch.isnan(outputs['vision_latent']).any(),
            'retrieved_memory': torch.isnan(outputs['retrieved_memory']).any(),
        }
        
        nan_found = any(nan_checks.values())
        if nan_found:
            print("✗ NaN values detected:")
            for key, has_nan in nan_checks.items():
                if has_nan:
                    print(f"  - {key}: contains NaN")
        else:
            print("✓ No NaN values detected in outputs")
        
        return not nan_found
        
    except Exception as e:
        print(f"✗ Error during model testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_compilation()
    sys.exit(0 if success else 1)
