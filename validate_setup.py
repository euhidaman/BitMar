#!/usr/bin/env python3
"""
Pre-training validation script for BitMar 100M Token Training
Validates all dependencies, files, and configurations before training
"""

import sys
import os
import importlib
import torch
from pathlib import Path

def check_imports():
    """Check all required imports"""
    print("🔍 Checking imports...")
    
    required_modules = [
        'torch', 'torchvision', 'numpy', 'yaml', 'tqdm', 
        'transformers', 'wandb', 'matplotlib', 'seaborn', 'PIL'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module} - MISSING")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️  Missing modules: {missing_modules}")
        print("Install with: pip install " + " ".join(missing_modules))
        return False
    return True

def check_files():
    """Check all required source files"""
    print("\n📁 Checking source files...")
    
    required_files = [
        'src/model.py',
        'src/dataset.py', 
        'src/wandb_logger.py',
        'src/attention_visualizer.py',
        'src/token_constrained_dataset.py',
        'src/adaptive_training_controller.py',
        'configs/bitmar_100M_tokens.yaml',
        'train_100M_tokens.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        return False
    return True

def check_config():
    """Check configuration file"""
    print("\n⚙️  Checking configuration...")
    
    try:
        import yaml
        with open('configs/bitmar_100M_tokens.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['token_constraints', 'model', 'data', 'training', 'output']
        for section in required_sections:
            if section in config:
                print(f"  ✅ {section}")
            else:
                print(f"  ❌ {section} - MISSING")
                return False
        
        # Check token constraints
        if config['token_constraints']['total_tokens'] == 100000000:
            print(f"  ✅ Token target: {config['token_constraints']['total_tokens']:,}")
        else:
            print(f"  ⚠️  Unexpected token target: {config['token_constraints']['total_tokens']:,}")
        
        return True
    except Exception as e:
        print(f"  ❌ Config validation failed: {e}")
        return False

def check_gpu():
    """Check GPU availability"""
    print("\n🖥️  Checking GPU...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"  ✅ CUDA available with {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    - GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        return True
    else:
        print("  ⚠️  CUDA not available - will use CPU")
        return False

def check_dataset():
    """Check dataset availability"""
    print("\n📊 Checking dataset...")
    
    dataset_dir = Path("../babylm_dataset")
    if not dataset_dir.exists():
        print(f"  ❌ Dataset directory not found: {dataset_dir}")
        return False
    
    required_files = [
        'cc_3M_captions.json',
        'cc_3M_dino_v2_states_1of2.npy',
        'cc_3M_dino_v2_states_2of2.npy',
        'local_narr_captions.json',
        'local_narr_dino_v2_states.npy',
        'train_50M.zip'
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = dataset_dir / file_name
        if file_path.exists():
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name} - MISSING")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n⚠️  Missing dataset files: {missing_files}")
        return False
    return True

def main():
    """Main validation function"""
    print("🚀 BitMar 100M Token Training - Pre-Training Validation")
    print("=" * 60)
    
    checks = [
        ("Imports", check_imports),
        ("Files", check_files), 
        ("Config", check_config),
        ("GPU", check_gpu),
        ("Dataset", check_dataset)
    ]
    
    all_passed = True
    results = {}
    
    for check_name, check_func in checks:
        results[check_name] = check_func()
        all_passed = all_passed and results[check_name]
    
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:15} | {status}")
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Ready to start training with:")
        print("   python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --device cuda:0")
        return 0
    else:
        print("❌ SOME CHECKS FAILED!")
        print("⚠️  Please fix the issues above before training")
        return 1

if __name__ == "__main__":
    sys.exit(main())
