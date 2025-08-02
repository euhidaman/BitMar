#!/usr/bin/env python3
"""
Pre-training validation script for BitMar
Validates configuration and dependencies before training
"""

import os
import sys
import yaml
import torch
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are available"""

    logger.info("Checking dependencies...")

    required_packages = [
        'torch',
        'torchvision',
        'transformers',
        'datasets',
        'wandb',
        'numpy',
        'scipy',
        'tqdm',
        'psutil',
        'codecarbon'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package}")

    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Install with: pip install " + " ".join(missing_packages))
        return False

    return True


def check_gpu_setup():
    """Check GPU availability and memory"""

    logger.info("Checking GPU setup...")

    if not torch.cuda.is_available():
        logger.warning(
            "⚠️ CUDA not available. Training will be very slow on CPU.")
        return False

    device_count = torch.cuda.device_count()
    logger.info(f"✅ CUDA available with {device_count} GPU(s)")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        logger.info(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")

        if memory_gb < 16:
            logger.warning(
                f"⚠️ GPU {i} has only {memory_gb:.1f} GB memory. May run into OOM issues.")

    return True


def check_data_paths(config):
    """Check if data paths exist"""

    logger.info("Checking data paths...")

    data_config = config.get('data', {})

    # Check training data
    train_path = data_config.get(
        'train_data_path', '../babylm_dataset/train_50M')
    train_path = Path(train_path)

    if train_path.exists():
        logger.info(f"✅ Training data found: {train_path}")
    else:
        logger.error(f"❌ Training data not found: {train_path}")
        return False

    # Check vision data
    vision_path = data_config.get('vision_data_path', '../babylm_dataset')
    vision_path = Path(vision_path)

    if vision_path.exists():
        logger.info(f"✅ Vision data path found: {vision_path}")
    else:
        logger.warning(f"⚠️ Vision data path not found: {vision_path}")
        logger.warning("This may cause issues with multimodal training")

    return True


def check_src_modules():
    """Check if all required src modules are available"""

    logger.info("Checking src modules...")

    required_modules = [
        'src.model',
        'src.dataset',
        'src.wandb_logger',
        'src.attention_visualizer',
        'src.modality_tracker',
        'src.dataset_optimizer',
        'src.hf_compatibility'
    ]

    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module}")
        except ImportError as e:
            missing_modules.append(module)
            logger.error(f"❌ {module}: {e}")

    if missing_modules:
        logger.error(
            "Some modules are missing. Check src/ directory structure.")
        return False

    return True


def validate_config(config_path):
    """Validate the training configuration"""

    logger.info(f"Validating config: {config_path}")

    if not Path(config_path).exists():
        logger.error(f"❌ Config file not found: {config_path}")
        return False, None

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("✅ Config file loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return False, None

    # Check required sections
    required_sections = ['model', 'training', 'data']
    for section in required_sections:
        if section not in config:
            logger.error(f"❌ Missing config section: {section}")
            return False, config

    # Check model config
    model_config = config['model']
    required_model_params = [
        'vocab_size', 'text_encoder_dim', 'text_decoder_dim',
        'vision_latent_size', 'memory_size', 'episode_dim'
    ]

    for param in required_model_params:
        if param not in model_config:
            logger.error(f"❌ Missing model parameter: {param}")
            return False, config

    logger.info("✅ Config validation passed")
    return True, config


def test_model_creation(config):
    """Test creating a BitMar model with the given config"""

    logger.info("Testing model creation...")

    try:
        from src.model import create_bitmar_model

        # Create model
        model = create_bitmar_model(config['model'])
        logger.info("✅ Model creation successful")

        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)

        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Test moving to GPU
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            model.to(device)
            logger.info("✅ Model moved to GPU successfully")

            # Test memory usage
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)
            logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")

            if memory_allocated > 10:
                logger.warning(
                    f"⚠️ High GPU memory usage: {memory_allocated:.2f} GB")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return True

    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_hf_compatibility():
    """Test HuggingFace compatibility components"""

    logger.info("Testing HuggingFace compatibility...")

    try:
        from src.hf_compatibility import BitMarConfig, BitMarForCausalLM

        # Create minimal config
        config = BitMarConfig(
            vocab_size=50257,
            text_encoder_dim=128,
            text_encoder_layers=2,
            text_encoder_heads=2,
            text_decoder_dim=128,
            text_decoder_layers=2,
            text_decoder_heads=2,
            max_seq_len=128
        )

        # Create model
        model = BitMarForCausalLM(config)
        logger.info("✅ HuggingFace compatibility test passed")

        return True

    except Exception as e:
        logger.error(f"❌ HuggingFace compatibility test failed: {e}")
        return False


def main():
    """Run all validation checks"""

    logger.info("🚀 BitMar Pre-training Validation")
    logger.info("=" * 50)

    all_checks_passed = True

    # Check dependencies
    if not check_dependencies():
        all_checks_passed = False

    print()

    # Check GPU setup
    if not check_gpu_setup():
        logger.warning(
            "GPU setup issues detected. Training may be slow or fail.")

    print()

    # Check src modules
    if not check_src_modules():
        all_checks_passed = False

    print()

    # Validate config
    config_path = "configs/bitmar_10epoch_memory_optimized.yaml"
    config_valid, config = validate_config(config_path)
    if not config_valid:
        all_checks_passed = False

    print()

    # Check data paths
    if config and not check_data_paths(config):
        logger.warning("Data path issues detected. Training may fail.")

    print()

    # Test model creation
    if config and not test_model_creation(config):
        all_checks_passed = False

    print()

    # Test HuggingFace compatibility
    if not test_hf_compatibility():
        all_checks_passed = False

    print()
    logger.info("=" * 50)

    if all_checks_passed:
        logger.info("🎉 All validation checks passed!")
        logger.info("✅ Ready for training")

        print("\nTo start training, run:")
        print("python train_bitmar.py --config configs/bitmar_10epoch_memory_optimized.yaml --optimizer adamw --epochs 10 --wandb_project bitmar-10epoch-memory-safe")

    else:
        logger.error("❌ Some validation checks failed")
        logger.error("Please fix the issues before training")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
