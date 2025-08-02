#!/usr/bin/env python3
"""
BitMar Evaluation Script
Tests compatibility with BabyLM 2025 evaluation pipeline
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.hf_compatibility import load_bitmar_from_hf_format, BitMarForCausalLM
import os
import sys
import argparse
from pathlib import Path

# Add BitMar src to path
sys.path.append(str(Path(__file__).parent / "src"))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_hf_compatibility(model_path: str):
    """Test if BitMar model is compatible with HuggingFace AutoModel"""

    logger.info(f"Testing HuggingFace compatibility for model: {model_path}")

    try:
        # Test 1: Load with our custom loader
        logger.info("Test 1: Loading with BitMar custom loader...")
        model = load_bitmar_from_hf_format(model_path)
        logger.info("✅ Custom loader successful")

        # Test 2: Load with AutoModel (should work if properly registered)
        logger.info("Test 2: Loading with AutoModelForCausalLM...")
        try:
            auto_model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True)
            logger.info("✅ AutoModelForCausalLM successful")
        except Exception as e:
            logger.warning(f"⚠️ AutoModelForCausalLM failed: {e}")
            logger.info(
                "This is expected if the model hasn't been trained yet")

        # Test 3: Load tokenizer
        logger.info("Test 3: Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("✅ Tokenizer loading successful")
        except Exception as e:
            logger.warning(f"⚠️ Tokenizer loading failed: {e}")
            # Use default tokenizer
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info("Using default GPT-2 tokenizer")

        # Test 4: Simple forward pass
        logger.info("Test 4: Testing forward pass...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Create dummy inputs
        test_text = "The cat sat on the mat"
        inputs = tokenizer(test_text, return_tensors="pt",
                           padding=True, truncation=True, max_length=50)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logger.info(
                f"✅ Forward pass successful. Output shape: {outputs.logits.shape}")

        # Test 5: Generation (if model supports it)
        logger.info("Test 5: Testing text generation...")
        try:
            with torch.no_grad():
                generated = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=60,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_text = tokenizer.decode(
                    generated[0], skip_special_tokens=True)
                logger.info(f"✅ Generation successful: {generated_text}")
        except Exception as e:
            logger.warning(f"⚠️ Generation failed: {e}")

        logger.info("🎉 BitMar HuggingFace compatibility test completed!")
        return True

    except Exception as e:
        logger.error(f"❌ Compatibility test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_evaluation_pipeline_compatibility():
    """Test compatibility with evaluation pipeline structure"""

    logger.info("Testing evaluation pipeline compatibility...")

    # Check if evaluation pipeline exists
    eval_pipeline_path = Path("../evaluation-pipeline-2025")
    if not eval_pipeline_path.exists():
        logger.warning("⚠️ Evaluation pipeline not found at expected location")
        logger.info("Expected location: ../evaluation-pipeline-2025")
        return False

    # Check required evaluation scripts
    required_scripts = [
        "eval_zero_shot.sh",
        "eval_zero_shot_fast.sh",
        "eval_finetune.sh"
    ]

    for script in required_scripts:
        script_path = eval_pipeline_path / script
        if script_path.exists():
            logger.info(f"✅ Found evaluation script: {script}")
        else:
            logger.warning(f"⚠️ Missing evaluation script: {script}")

    # Check evaluation pipeline structure
    eval_module_path = eval_pipeline_path / "evaluation_pipeline"
    if eval_module_path.exists():
        logger.info("✅ Found evaluation_pipeline module")

        # Check key modules
        key_modules = [
            "sentence_zero_shot",
            "finetune",
            "reading"
        ]

        for module in key_modules:
            module_path = eval_module_path / module
            if module_path.exists():
                logger.info(f"✅ Found module: {module}")
            else:
                logger.warning(f"⚠️ Missing module: {module}")
    else:
        logger.warning("⚠️ evaluation_pipeline module not found")

    logger.info("📋 Evaluation pipeline compatibility check completed")
    return True


def create_test_model():
    """Create a minimal test model for evaluation pipeline testing"""

    logger.info("Creating test model for evaluation...")

    try:
        from src.hf_compatibility import BitMarConfig, BitMarForCausalLM
        from transformers import AutoTokenizer

        # Create minimal config
        config = BitMarConfig(
            vocab_size=50257,
            text_encoder_dim=128,
            text_encoder_layers=2,
            text_encoder_heads=2,
            text_decoder_dim=128,
            text_decoder_layers=2,
            text_decoder_heads=2,
            vision_latent_size=32,
            vision_hidden_size=16,
            fusion_hidden_size=64,
            fusion_num_heads=2,
            fusion_num_layers=1,
            memory_size=8,
            episode_dim=64,
            max_seq_len=128
        )

        # Create model
        model = BitMarForCausalLM(config)

        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Save test model
        test_model_dir = Path("./test_model")
        test_model_dir.mkdir(exist_ok=True)

        model.save_pretrained(test_model_dir)
        tokenizer.save_pretrained(test_model_dir)

        logger.info(f"✅ Test model created at: {test_model_dir}")
        return str(test_model_dir)

    except Exception as e:
        logger.error(f"❌ Failed to create test model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Test BitMar evaluation compatibility")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained BitMar model in HF format"
    )
    parser.add_argument(
        "--create_test_model",
        action="store_true",
        help="Create a test model for evaluation"
    )

    args = parser.parse_args()

    logger.info("🚀 BitMar Evaluation Compatibility Test")
    logger.info("=" * 50)

    # Test evaluation pipeline structure
    test_evaluation_pipeline_compatibility()

    print()

    if args.create_test_model:
        # Create test model
        test_model_path = create_test_model()
        if test_model_path:
            args.model_path = test_model_path

    if args.model_path:
        if Path(args.model_path).exists():
            # Test HuggingFace compatibility
            test_hf_compatibility(args.model_path)
        else:
            logger.warning(f"Model path does not exist: {args.model_path}")
            logger.info("Use --create_test_model to create a test model")
    else:
        logger.info(
            "No model path specified. Use --model_path or --create_test_model")

    print()
    logger.info("🏁 Evaluation compatibility test completed!")


if __name__ == "__main__":
    main()
