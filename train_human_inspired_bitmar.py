"""
Human-Inspired BitMar Training Script
=====================================

This script implements the 3-stage human-inspired learning approach:
1. Visual Understanding First (babies learn to see)
2. Visual-Language Grounding (connect words to images) 
3. Abstract Language Learning (pure text reasoning)

Usage:
python train_human_inspired_bitmar.py --config configs/bitmar_adaptive.yaml
"""

import argparse
import logging
import torch
import torch.optim as optim
from pathlib import Path
import yaml
import wandb
from torch.utils.data import DataLoader
import zipfile
import os

from src.model import create_bitmar_model
from src.human_inspired_training import (
    HumanInspiredTrainingPipeline,
    create_human_inspired_dataloaders
)
from src.wandb_logger import WandBLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_train_50M_if_needed(dataset_dir: Path):
    """Extract train_50M.zip if not already extracted"""
    zip_path = dataset_dir / "train_50M.zip"
    extract_path = dataset_dir / "train_50M"

    if not zip_path.exists():
        raise FileNotFoundError(f"train_50M.zip not found at {zip_path}")

    if extract_path.exists() and any(extract_path.iterdir()):
        logger.info(f"✅ train_50M already extracted at {extract_path}")
        return extract_path

    logger.info(f"📦 Extracting train_50M.zip to {extract_path}...")
    extract_path.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path.parent)

    # Verify extraction
    expected_files = [
        'train_50M/childes.train',
        'train_50M/gutenberg.train',
        'train_50M/open_subtitles.train',
        'train_50M/simple_wiki.train',
        'train_50M/bnc_spoken.train',
        'train_50M/switchboard.train'
    ]

    missing_files = []
    for file_path in expected_files:
        full_path = dataset_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)

    if missing_files:
        raise FileNotFoundError(f"Missing files after extraction: {missing_files}")

    logger.info(f"✅ Successfully extracted train_50M.zip!")
    logger.info(f"📁 Files available:")
    for file_path in expected_files:
        full_path = dataset_dir / file_path
        size_mb = full_path.stat().st_size / (1024 * 1024)
        logger.info(f"   - {full_path.name}: {size_mb:.1f}MB")

    return extract_path


def main():
    parser = argparse.ArgumentParser(description='Human-Inspired BitMar Training')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--wandb_project', type=str, default='bitmar-human-inspired',
                       help='Weights & Biases project name')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], default=None,
                       help='Run specific stage only (default: run all stages)')
    parser.add_argument('--resume_from_stage', type=int, choices=[1, 2, 3], default=1,
                       help='Resume training from specific stage')
    parser.add_argument('--extract_only', action='store_true',
                       help='Only extract train_50M.zip and exit')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract train_50M.zip if needed
    dataset_dir = Path(config['data']['dataset_dir'])
    extract_path = extract_train_50M_if_needed(dataset_dir)

    if args.extract_only:
        logger.info("🎯 Extraction complete! Exiting as requested.")
        return

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize model
    model = create_bitmar_model(config['model'])
    model = model.to(device)

    # Initialize human-inspired training pipeline
    training_pipeline = HumanInspiredTrainingPipeline(model, config['model'])

    # Create dataloaders for all stages
    logger.info("🔄 Creating dataloaders for human-inspired learning...")
    visual_dataloader, multimodal_dataloader, text_dataloader = create_human_inspired_dataloaders(config['data'])

    # Initialize Weights & Biases
    wandb.init(
        project=args.wandb_project,
        config=config,
        name=f"human-inspired-bitmar-{config['model'].get('text_encoder_dim', 64)}d"
    )

    # Setup optimizers for each stage
    optimizers = {
        1: optim.AdamW(model.vision_encoder.parameters(), lr=0.001, weight_decay=0.01),
        2: optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0005, weight_decay=0.01),
        3: optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0003, weight_decay=0.01)
    }

    # Human-Inspired Training Pipeline
    logger.info("🧠 Starting Human-Inspired Learning Pipeline!")
    logger.info("=" * 60)

    stages_to_run = [args.stage] if args.stage else [1, 2, 3]
    if args.resume_from_stage > 1:
        stages_to_run = [s for s in stages_to_run if s >= args.resume_from_stage]

    for stage in stages_to_run:
        logger.info(f"\n🎯 STARTING STAGE {stage}")
        logger.info(f"📝 {training_pipeline.stage_configs[stage]['name']}")
        logger.info(f"💡 {training_pipeline.stage_configs[stage]['description']}")
        logger.info("-" * 60)

        # Set model to training mode
        model.train()

        # Setup optimizer for this stage
        optimizer = optimizers[stage]

        if stage == 1:
            # Stage 1: Visual Understanding (like babies learning to see)
            logger.info("👶 STAGE 1: Teaching the model to 'see' like a baby")
            logger.info("🎯 Focus: Learn visual representations without language pressure")

            # Add reconstruction head if needed
            if not hasattr(model, 'vision_reconstruction_head'):
                model.vision_reconstruction_head = torch.nn.Linear(
                    config['model']['vision_latent_size'],
                    config['model']['vision_encoder_dim']
                ).to(device)

                # Add to optimizer
                optimizer.add_param_group({
                    'params': model.vision_reconstruction_head.parameters(),
                    'lr': 0.001
                })

            training_pipeline.train_stage_1_visual_understanding(visual_dataloader)

            # Save stage 1 checkpoint
            torch.save(model.state_dict(), f'checkpoints/stage1_visual_understanding.pth')
            logger.info("💾 Stage 1 checkpoint saved!")

        elif stage == 2:
            # Stage 2: Visual-Language Grounding (connecting words to images)
            logger.info("🔗 STAGE 2: Teaching the model to connect words with images")
            logger.info("🎯 Focus: Learn that 'red car' text matches red car image")

            # Load stage 1 checkpoint if resuming
            if args.resume_from_stage <= 1 and Path('checkpoints/stage1_visual_understanding.pth').exists():
                model.load_state_dict(torch.load('checkpoints/stage1_visual_understanding.pth'))
                logger.info("📂 Loaded Stage 1 checkpoint")

            training_pipeline.train_stage_2_visual_language_grounding(multimodal_dataloader)

            # Save stage 2 checkpoint
            torch.save(model.state_dict(), f'checkpoints/stage2_visual_language_grounding.pth')
            logger.info("💾 Stage 2 checkpoint saved!")

        elif stage == 3:
            # Stage 3: Abstract Language Learning (pure text reasoning)
            logger.info("📚 STAGE 3: Teaching the model abstract language reasoning")
            logger.info("🎯 Focus: Learn from pure text like reading books")

            # Load stage 2 checkpoint if resuming
            if args.resume_from_stage <= 2 and Path('checkpoints/stage2_visual_language_grounding.pth').exists():
                model.load_state_dict(torch.load('checkpoints/stage2_visual_language_grounding.pth'))
                logger.info("📂 Loaded Stage 2 checkpoint")

            training_pipeline.train_stage_3_abstract_language(text_dataloader)

            # Save final model
            torch.save(model.state_dict(), f'checkpoints/final_human_inspired_bitmar.pth')
            logger.info("💾 Final model saved!")

        # Log stage completion to wandb
        wandb.log({
            f'stage_{stage}_completed': True,
            'current_stage': stage
        })

        logger.info(f"✅ STAGE {stage} COMPLETED!")
        logger.info("=" * 60)

    # Final evaluation
    logger.info("\n🎉 HUMAN-INSPIRED TRAINING COMPLETE!")
    logger.info("🧠 The model has learned like a human:")
    logger.info("   1. ✅ Visual understanding (like babies learning to see)")
    logger.info("   2. ✅ Visual-language grounding (connecting words to images)")
    logger.info("   3. ✅ Abstract language reasoning (reading and thinking)")

    # Test the final model
    test_human_inspired_model(model, device)

    wandb.finish()


def test_human_inspired_model(model, device):
    """Test the human-inspired trained model"""
    logger.info("\n🧪 Testing Human-Inspired Model...")

    model.eval()

    # Test 1: Visual understanding
    logger.info("Test 1: Visual Understanding")
    dummy_vision = torch.randn(1, 768).to(device)  # Dummy DiNOv2 features
    with torch.no_grad():
        vision_encoded = model.encode_vision(dummy_vision)
        logger.info(f"✅ Vision encoding successful: {vision_encoded.shape}")

    # Test 2: Visual-language grounding
    logger.info("Test 2: Visual-Language Grounding")
    dummy_text = torch.randint(0, 1000, (1, 10)).to(device)  # Dummy tokens
    dummy_mask = torch.ones(1, 10).to(device)

    with torch.no_grad():
        try:
            outputs = model(
                input_ids=dummy_text,
                attention_mask=dummy_mask,
                vision_features=dummy_vision,
                mode="inference"
            )
            logger.info(f"✅ Multimodal inference successful: {outputs['logits'].shape}")
        except Exception as e:
            logger.warning(f"⚠️ Multimodal test failed: {e}")
    
    # Test 3: Abstract language reasoning
    logger.info("Test 3: Abstract Language Reasoning") 
    with torch.no_grad():
        try:
            text_outputs = model.text_decoder(
                input_ids=dummy_text,
                attention_mask=dummy_mask
            )
            logger.info(f"✅ Text-only reasoning successful: {text_outputs['logits'].shape}")
        except Exception as e:
            logger.warning(f"⚠️ Text-only test failed: {e}")
    
    logger.info("🎯 Human-Inspired Model Testing Complete!")


if __name__ == "__main__":
    main()
