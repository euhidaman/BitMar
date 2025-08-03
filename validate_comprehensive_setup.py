#!/usr/bin/env python3
"""
Comprehensive Validation Script for BitMar Project
Validates all components: BabyLM compliance, GPU optimization, episodic memory, 
Quadrangle Attention, and wandb logging
"""

import os
import sys
import torch
import yaml
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_cuda_setup():
    """Validate CUDA availability and configuration"""
    logger.info("=== CUDA VALIDATION ===")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"✅ CUDA available with {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        logger.info(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        
        if "A6000" in props.name or memory_gb > 40:
            logger.info(f"   ✅ RTX A6000 or similar detected - excellent for training")
    
    return True

def validate_dataset_structure():
    """Validate BabyLM dataset structure"""
    logger.info("=== DATASET VALIDATION ===")
    
    dataset_dir = Path("../babylm_dataset")
    if not dataset_dir.exists():
        logger.error(f"❌ Dataset directory not found: {dataset_dir}")
        return False
    
    required_files = [
        "cc_3M_captions.json",
        "local_narr_captions.json", 
        "cc_3M_dino_v2_states_1of2.npy",
        "cc_3M_dino_v2_states_2of2.npy",
        "local_narr_dino_v2_states.npy"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = dataset_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / 1024**2
            logger.info(f"   ✅ {file} ({size_mb:.1f}MB)")
        else:
            missing_files.append(file)
            logger.error(f"   ❌ Missing: {file}")
    
    # Check train_50M directory
    train_dir = dataset_dir / "train_50M"
    if train_dir.exists():
        text_files = list(train_dir.glob("*.train"))
        logger.info(f"   ✅ train_50M directory with {len(text_files)} files")
    else:
        logger.warning(f"   ⚠️ train_50M directory not found")
    
    return len(missing_files) == 0

def validate_config():
    """Validate configuration file"""
    logger.info("=== CONFIG VALIDATION ===")
    
    config_path = Path("configs/bitmar_config.yaml")
    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check critical settings
        model_config = config.get('model', {})
        data_config = config.get('data', {})
        training_config = config.get('training', {})
        
        # Validate BabyLM compliance settings
        if model_config.get('use_quadrangle_attention', False):
            logger.info("   ✅ Quadrangle Attention enabled")
        else:
            logger.warning("   ⚠️ Quadrangle Attention disabled")
        
        # Validate memory settings
        memory_size = model_config.get('memory_size', 0)
        if memory_size > 0:
            logger.info(f"   ✅ Episodic memory enabled ({memory_size} slots)")
        else:
            logger.warning("   ⚠️ Episodic memory disabled")
        
        # Validate batch size configuration
        batch_size = data_config.get('batch_size', 0)
        grad_accum = training_config.get('gradient_accumulation_steps', 1)
        effective_batch = batch_size * grad_accum
        logger.info(f"   ✅ Batch configuration: {batch_size} × {grad_accum} = {effective_batch}")
        
        if batch_size <= 4 and grad_accum >= 8:
            logger.info("   ✅ Memory-efficient batch configuration")
        else:
            logger.warning("   ⚠️ Batch configuration may cause GPU OOM")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Config validation failed: {e}")
        return False

def validate_model_components():
    """Validate model components can be imported and initialized"""
    logger.info("=== MODEL COMPONENT VALIDATION ===")
    
    try:
        # Test model imports
        from src.model import create_bitmar_model, count_parameters
        from src.quadrangle_attention import QuadrangleAttention
        from src.dataset import create_data_module
        from src.wandb_logger import BitMarWandbLogger
        
        logger.info("   ✅ All core imports successful")
        
        # Test model creation with minimal config
        test_config = {
            'vocab_size': 1000,
            'text_encoder_dim': 128,
            'text_encoder_layers': 2,
            'text_encoder_heads': 4,  # Missing key added
            'text_decoder_dim': 128,
            'text_decoder_layers': 2,
            'text_decoder_heads': 4,  # Missing key added
            'vision_encoder_dim': 768,
            'vision_latent_size': 128,
            'fusion_hidden_size': 128,
            'fusion_num_layers': 1,
            'fusion_num_heads': 4,
            'memory_size': 8,
            'episode_dim': 128,
            'use_quadrangle_attention': True,
            'quadrangle_memory_size': 256,
            'max_seq_len': 64,
            'dropout': 0.1  # Also missing dropout key
        }
        
        # Test model creation
        model = create_bitmar_model(test_config)
        param_count = count_parameters(model)
        logger.info(f"   ✅ Model created successfully ({param_count:,} parameters)")
        
        # Test GPU transfer
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("   ✅ Model transferred to GPU")
        
        # Test QuadrangleAttention separately
        qa = QuadrangleAttention(dim=128, num_heads=4)
        logger.info("   ✅ QuadrangleAttention module created")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model component validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_token_compliance():
    """Validate BabyLM token compliance implementation"""
    logger.info("=== TOKEN COMPLIANCE VALIDATION ===")
    
    try:
        from src.dataset import MixedMultimodalTextDataset
        
        # Test dataset with minimal configuration
        test_config = {
            'dataset_dir': '../babylm_dataset',
            'max_seq_length': 64,
            'batch_size': 2,
            'text_ratio': 0.4
        }
        
        dataset = MixedMultimodalTextDataset(
            dataset_dir=test_config['dataset_dir'],
            max_seq_length=test_config['max_seq_length'],
            text_ratio=test_config['text_ratio'],
            config=test_config
        )
        
        # Check token usage stats
        if hasattr(dataset, 'get_token_usage_stats'):
            stats = dataset.get_token_usage_stats()
            logger.info("   ✅ Token usage tracking available:")
            logger.info(f"      Text tokens: {stats['text_tokens_used']:,}/{stats['text_tokens_limit']:,}")
            logger.info(f"      Image tokens: {stats['image_tokens_used']:,}/{stats['image_tokens_limit']:,}")
            
            # Validate limits
            if stats['text_tokens_used'] <= stats['text_tokens_limit']:
                logger.info("      ✅ Text token limit respected")
            else:
                logger.error("      ❌ Text token limit exceeded")
                
            if stats['image_tokens_used'] <= stats['image_tokens_limit']:
                logger.info("      ✅ Image token limit respected")
            else:
                logger.error("      ❌ Image token limit exceeded")
                
        else:
            logger.error("   ❌ Token usage tracking not available")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Token compliance validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_memory_optimization():
    """Validate GPU memory optimization settings"""
    logger.info("=== MEMORY OPTIMIZATION VALIDATION ===")
    
    if not torch.cuda.is_available():
        logger.warning("   ⚠️ CUDA not available, skipping GPU memory validation")
        return True
    
    try:
        # Test memory fraction setting
        torch.cuda.set_per_process_memory_fraction(0.85)
        logger.info("   ✅ GPU memory fraction set to 85%")
        
        # Test mixed precision
        if hasattr(torch.cuda, 'amp'):
            scaler = torch.amp.GradScaler('cuda')
            logger.info("   ✅ Mixed precision training available")
        else:
            logger.warning("   ⚠️ Mixed precision not available")
        
        # Test CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.info("   ✅ CUDA optimizations enabled")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory optimization validation failed: {e}")
        return False

def main():
    """Run comprehensive validation"""
    logger.info("🚀 Starting Comprehensive BitMar Validation")
    logger.info("=" * 60)
    
    validations = [
        ("CUDA Setup", validate_cuda_setup),
        ("Dataset Structure", validate_dataset_structure),
        ("Configuration", validate_config),
        ("Model Components", validate_model_components),
        ("Token Compliance", validate_token_compliance),
        ("Memory Optimization", validate_memory_optimization)
    ]
    
    results = {}
    
    for name, validation_func in validations:
        try:
            results[name] = validation_func()
        except Exception as e:
            logger.error(f"❌ {name} validation crashed: {e}")
            results[name] = False
        
        logger.info("")  # Add spacing
    
    # Summary
    logger.info("=" * 60)
    logger.info("🎯 VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {status} - {name}")
    
    logger.info(f"\nOverall: {passed}/{total} validations passed")
    
    if passed == total:
        logger.info("🎉 ALL VALIDATIONS PASSED - Ready for training!")
        logger.info("\nRecommended training command:")
        logger.info("python train_bitmar.py configs/bitmar_config.yaml")
    else:
        logger.error(f"⚠️ {total - passed} validation(s) failed - Fix issues before training")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
