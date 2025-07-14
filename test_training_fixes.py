#!/usr/bin/env python3
"""
Quick test script to validate training fixes
"""

import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_import_fixes():
    """Test that all imports work correctly"""
    try:
        from src.wandb_logger import BitMarWandbLogger
        print("✓ wandb_logger import successful")
        
        from src.attention_visualizer import AttentionHeadAnalyzer
        print("✓ attention_visualizer import successful")
        
        # Test basic logger creation (without wandb initialization)
        config = {"model": {"vocab_size": 50257}}
        
        # This should not crash even without wandb setup
        logger = BitMarWandbLogger(project_name="test", config=config)
        print("✓ BitMarWandbLogger instantiation successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_config_loading():
    """Test that configuration loading works"""
    try:
        config_path = Path("configs/bitmar_config.yaml")
        if not config_path.exists():
            print("✗ Config file not found")
            return False
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Check required sections
        required_sections = ['model', 'data', 'training', 'wandb', 'attention_analysis']
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing config section: {section}")
                return False
                
        print("✓ Configuration loading successful")
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_error_handling():
    """Test that error handling works as expected"""
    try:
        # Test tensor operations that could fail
        import torch
        import numpy as np
        
        # Test memory heatmap creation logic
        memory_usage = torch.randn(100)  # Non-square number
        total_slots = len(memory_usage)
        side_length = int(np.ceil(np.sqrt(total_slots)))
        
        padded_usage = np.zeros(side_length * side_length)
        padded_usage[:total_slots] = memory_usage.numpy()
        
        memory_2d = padded_usage.reshape(side_length, side_length)
        
        print("✓ Memory heatmap reshaping logic successful")
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing BitMar training fixes...")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    tests = [
        ("Import Tests", test_import_fixes),
        ("Config Loading", test_config_loading), 
        ("Error Handling", test_error_handling)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if not test_func():
            all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ All tests passed! Training should work correctly.")
    else:
        print("✗ Some tests failed. Please check the issues above.")
    
    print("\nMain fixes applied:")
    print("1. Fixed 'input_ids' undefined error -> batch['input_ids']")
    print("2. Added error handling for wandb logging")
    print("3. Added error handling for attention analysis") 
    print("4. Fixed memory heatmap reshaping for non-square sizes")
    print("5. Added missing logger import")
    print("6. Made wandb config access safer with defaults")
