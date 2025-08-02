#!/usr/bin/env python3
"""
Prepare BitMar model for BabyLM 2025 evaluation pipeline
Run this after training to convert model to evaluation-ready format
"""

import yaml
from transformers import AutoTokenizer
from src.model import BitMarModel
from src.hf_compatibility import save_bitmar_as_hf_model
import os
import sys
import argparse
import torch
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_checkpoint_to_hf(checkpoint_path: str, config_path: str, output_dir: str):
    """
    Convert BitMar checkpoint to HuggingFace format for evaluation

    Args:
        checkpoint_path: Path to BitMar checkpoint (.pt file)
        config_path: Path to model configuration (.yaml file)
        output_dir: Directory to save HuggingFace model
    """

    logger.info("🔄 Converting BitMar checkpoint to HuggingFace format...")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']

    # Create BitMar model
    logger.info("Creating BitMar model...")
    from src.model import create_bitmar_model
    model = create_bitmar_model(model_config)

    # Load checkpoint
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✅ Loaded model state from checkpoint")
    else:
        model.load_state_dict(checkpoint)
        logger.info("✅ Loaded model state directly")

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Save in HuggingFace format
    logger.info(f"Saving HuggingFace model to: {output_dir}")
    hf_model = save_bitmar_as_hf_model(
        bitmar_model=model,
        config_dict=model_config,
        save_directory=output_dir,
        tokenizer=tokenizer
    )

    logger.info("✅ Conversion completed!")
    return hf_model


def find_latest_checkpoint(checkpoint_dir: str):
    """Find the latest or best checkpoint in the directory"""

    checkpoint_dir = Path(checkpoint_dir)

    # Look for best checkpoint first
    best_checkpoint = checkpoint_dir / "best_checkpoint.pt"
    if best_checkpoint.exists():
        logger.info(f"Found best checkpoint: {best_checkpoint}")
        return str(best_checkpoint)

    # Look for latest checkpoint
    latest_checkpoint = checkpoint_dir / "latest_checkpoint.pt"
    if latest_checkpoint.exists():
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        return str(latest_checkpoint)

    # Look for any checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if checkpoint_files:
        # Sort by epoch number and take the latest
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        latest = checkpoint_files[-1]
        logger.info(f"Found latest epoch checkpoint: {latest}")
        return str(latest)

    raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")


def create_evaluation_commands(model_path: str):
    """Create example evaluation commands for the BabyLM pipeline"""

    model_path = Path(model_path).absolute()

    commands = f"""
# BitMar Model Evaluation Commands for BabyLM 2025 Pipeline

## Model Location
Model saved at: {model_path}

## Zero-shot Evaluation (Fast)
cd ../evaluation-pipeline-2025
./eval_zero_shot_fast.sh "{model_path}" "checkpoint_1M" "causal"

## Zero-shot Evaluation (Full)
./eval_zero_shot.sh "{model_path}" "causal"

## Fine-tuning Evaluation
./eval_finetune.sh "{model_path}"

## Manual Python Commands

# BLiMP evaluation
python -m evaluation_pipeline.sentence_zero_shot.run \\
    --model_path_or_name "{model_path}" \\
    --backend causal \\
    --task blimp \\
    --data_path "evaluation_data/fast_eval/blimp_filtered" \\
    --save_predictions

# EWoK evaluation  
python -m evaluation_pipeline.sentence_zero_shot.run \\
    --model_path_or_name "{model_path}" \\
    --backend causal \\
    --task ewok \\
    --data_path "evaluation_data/fast_eval/ewok_filtered" \\
    --save_predictions

# GLUE fine-tuning (single task example)
python -m evaluation_pipeline.finetune.run \\
    --model_path_or_name "{model_path}" \\
    --task_name "sst2" \\
    --learning_rate 3e-5 \\
    --batch_size 32 \\
    --max_epochs 10

## Notes
- Make sure evaluation_data directory exists in evaluation-pipeline-2025
- Download required datasets (BLiMP, EWoK, GLUE) as per evaluation pipeline README
- BitMar is configured as a causal language model for evaluation
"""

    return commands


def main():
    parser = argparse.ArgumentParser(
        description="Prepare BitMar for evaluation")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint file (will auto-find if not specified)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoints"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/bitmar_10epoch_memory_optimized.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="final_model",
        help="Output directory for HuggingFace model"
    )
    parser.add_argument(
        "--generate_commands",
        action="store_true",
        help="Generate evaluation commands"
    )

    args = parser.parse_args()

    logger.info("🚀 BitMar Evaluation Preparation")
    logger.info("=" * 50)

    # Find checkpoint if not specified
    if args.checkpoint_path is None:
        try:
            args.checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        except FileNotFoundError as e:
            logger.error(f"❌ {e}")
            return

    # Verify files exist
    if not Path(args.checkpoint_path).exists():
        logger.error(f"❌ Checkpoint not found: {args.checkpoint_path}")
        return

    if not Path(args.config_path).exists():
        logger.error(f"❌ Config not found: {args.config_path}")
        return

    try:
        # Convert to HuggingFace format
        convert_checkpoint_to_hf(
            checkpoint_path=args.checkpoint_path,
            config_path=args.config_path,
            output_dir=args.output_dir
        )

        logger.info(f"🎉 Model ready for evaluation at: {args.output_dir}")

        # Generate evaluation commands
        if args.generate_commands:
            commands = create_evaluation_commands(args.output_dir)

            commands_file = Path(args.output_dir) / "evaluation_commands.txt"
            with open(commands_file, 'w') as f:
                f.write(commands)

            logger.info(f"📋 Evaluation commands saved to: {commands_file}")
            print("\n" + "=" * 60)
            print("EVALUATION COMMANDS:")
            print("=" * 60)
            print(commands)

    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return


if __name__ == "__main__":
    main()
