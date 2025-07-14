"""
Training script for BitMar model
Handles multimodal training with episodic memory and attention analysis
"""

from src.attention_analysis import analyze_model_attention
from src.dataset import create_data_module
from src.model import create_bitmar_model, count_parameters
import os
import sys
import argparse
import logging
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BitMarTrainer:
    """BitMar model trainer with episodic memory and attention analysis"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Setup directories
        self.setup_directories()

        # Initialize wandb if configured
        self.setup_wandb()

        # Create model and data
        self.model = None
        self.data_module = None
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def setup_directories(self):
        """Create output directories"""
        for dir_name in ['checkpoint_dir', 'log_dir', 'attention_dir', 'memory_dir', 'results_dir']:
            dir_path = Path(self.config['output'][dir_name])
            dir_path.mkdir(parents=True, exist_ok=True)
            setattr(self, dir_name, dir_path)

    def setup_wandb(self):
        """Initialize Weights & Biases logging"""
        if self.config.get('wandb', {}).get('api_key'):
            # Set API key
            os.environ['WANDB_API_KEY'] = self.config['wandb']['api_key']

            # Initialize wandb
            wandb.init(
                project=self.config['wandb']['project'],
                entity=self.config['wandb']['entity'],
                config=self.config,
                name=f"bitmar-{self.config['training']['max_epochs']}epochs",
                tags=["bitmar", "multimodal", "episodic-memory", "babylm"]
            )

            # Watch model (will be set later)
            self.use_wandb = True
            logger.info("Wandb initialized successfully")
        else:
            self.use_wandb = False
            logger.info("Wandb not configured, logging locally only")

    def setup_model_and_data(self, max_samples: Optional[int] = None):
        """Initialize model and data loaders"""
        logger.info("Setting up model and data...")

        # Create model
        self.model = create_bitmar_model(self.config['model'])
        self.model.to(self.device)

        # Log model info
        param_count = count_parameters(self.model)
        logger.info(f"Model parameters: {param_count}")

        if self.use_wandb:
            wandb.log(param_count)
            wandb.watch(self.model, log="all", log_freq=100)

        # Create data module
        self.data_module = create_data_module(self.config['data'])
        self.data_module.setup(max_samples=max_samples)

        # Setup optimizer and scheduler
        self.setup_optimizer()

    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['model']['learning_rate'],
            weight_decay=self.config['model']['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler
        if self.config['training']['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['max_epochs'],
                eta_min=self.config['training']['min_lr']
            )
        else:
            self.scheduler = None

        logger.info(
            f"Optimizer: AdamW with LR={self.config['model']['learning_rate']}")
        if self.scheduler:
            logger.info(f"Scheduler: {self.config['training']['scheduler']}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        train_loader = self.data_module.train_dataloader()

        epoch_losses = []
        epoch_metrics = {
            'train_loss': 0.0,
            'memory_usage_entropy': 0.0,
            'cross_modal_similarity': 0.0
        }

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                vision_features=batch['vision_features'],
                labels=batch['labels']
            )

            loss = outputs['loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config['model']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['model']['gradient_clip']
                )

            self.optimizer.step()

            # Update metrics
            epoch_losses.append(loss.item())

            # Compute additional metrics
            if outputs['memory_usage'] is not None:
                memory_entropy = self._compute_memory_entropy(
                    outputs['memory_usage'])
                epoch_metrics['memory_usage_entropy'] += memory_entropy

            if outputs['text_latent'] is not None and outputs['vision_latent'] is not None:
                cross_modal_sim = self._compute_cross_modal_similarity(
                    outputs['text_latent'], outputs['vision_latent']
                )
                epoch_metrics['cross_modal_similarity'] += cross_modal_sim

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{np.mean(epoch_losses):.4f}"
            })

            # Log to wandb
            if self.use_wandb and batch_idx % self.config['wandb']['log_every_n_steps'] == 0:
                wandb.log({
                    'train_loss_step': loss.item(),
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })

            self.global_step += 1

        # Average metrics over epoch
        epoch_metrics['train_loss'] = np.mean(epoch_losses)
        epoch_metrics['memory_usage_entropy'] /= len(train_loader)
        epoch_metrics['cross_modal_similarity'] /= len(train_loader)

        return epoch_metrics

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        val_loader = self.data_module.val_dataloader()

        val_losses = []
        val_metrics = {
            'val_loss': 0.0,
            'val_memory_entropy': 0.0,
            'val_cross_modal_similarity': 0.0
        }

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    vision_features=batch['vision_features'],
                    labels=batch['labels']
                )

                val_losses.append(outputs['loss'].item())

                # Compute additional metrics
                if outputs['memory_usage'] is not None:
                    memory_entropy = self._compute_memory_entropy(
                        outputs['memory_usage'])
                    val_metrics['val_memory_entropy'] += memory_entropy

                if outputs['text_latent'] is not None and outputs['vision_latent'] is not None:
                    cross_modal_sim = self._compute_cross_modal_similarity(
                        outputs['text_latent'], outputs['vision_latent']
                    )
                    val_metrics['val_cross_modal_similarity'] += cross_modal_sim

        # Average metrics
        val_metrics['val_loss'] = np.mean(val_losses)
        val_metrics['val_memory_entropy'] /= len(val_loader)
        val_metrics['val_cross_modal_similarity'] /= len(val_loader)

        return val_metrics

    def _compute_memory_entropy(self, memory_usage: torch.Tensor) -> float:
        """Compute entropy of memory usage distribution"""
        # Normalize to probabilities
        probs = memory_usage / (memory_usage.sum() + 1e-8)
        # Compute entropy
        entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
        return entropy

    def _compute_cross_modal_similarity(
        self,
        text_latent: torch.Tensor,
        vision_latent: torch.Tensor
    ) -> float:
        """Compute cosine similarity between text and vision features"""
        # Pool text features (mean over sequence)
        text_pooled = text_latent.mean(dim=1)  # [batch_size, feature_dim]

        # Compute cosine similarity
        cos_sim = torch.cosine_similarity(text_pooled, vision_latent, dim=1)
        return cos_sim.mean().item()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"New best checkpoint saved: {best_path}")

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}")

        return self.current_epoch

    def run_attention_analysis(self):
        """Run comprehensive attention analysis"""
        logger.info("Starting attention analysis...")

        # Create analysis data loader (smaller batch size for memory efficiency)
        analysis_config = self.config['data'].copy()
        analysis_config['batch_size'] = min(4, analysis_config['batch_size'])

        analysis_data_module = create_data_module(analysis_config)
        # Limit samples for analysis
        analysis_data_module.setup(max_samples=1000)

        # Run analysis
        analyzer = analyze_model_attention(
            model=self.model,
            dataloader=analysis_data_module.val_dataloader(),
            tokenizer=self.model.tokenizer,
            config=self.config['output'],
            num_analysis_batches=50
        )

        # Log analysis results to wandb
        if self.use_wandb:
            report = analyzer.generate_report()
            wandb.log({
                f"analysis/{key}": value
                for key, value in report.items()
                if isinstance(value, (int, float))
            })

        logger.info("Attention analysis completed")
        return analyzer

    def train(self, max_samples: Optional[int] = None):
        """Main training loop"""
        logger.info("Starting BitMar training...")

        # Setup model and data
        self.setup_model_and_data(max_samples=max_samples)

        # Training loop
        for epoch in range(self.current_epoch, self.config['training']['max_epochs']):
            logger.info(
                f"\nEpoch {epoch + 1}/{self.config['training']['max_epochs']}")

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate_epoch(epoch)

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics['epoch'] = epoch
            all_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']

            # Log metrics
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(
                f"Memory Entropy: {train_metrics['memory_usage_entropy']:.4f}")
            logger.info(
                f"Cross-Modal Similarity: {train_metrics['cross_modal_similarity']:.4f}")

            if self.use_wandb:
                wandb.log(all_metrics)

            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']

            self.save_checkpoint(epoch, is_best=is_best)

            # Run attention analysis periodically
            if (epoch + 1) % 2 == 0 and self.config['model']['track_attention']:
                self.run_attention_analysis()

        # Final attention analysis
        if self.config['model']['track_attention']:
            logger.info("Running final attention analysis...")
            self.run_attention_analysis()

        logger.info("Training completed!")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train BitMar model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bitmar_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Override max epochs from config"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of training samples (for testing)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if args.max_epochs:
        config['training']['max_epochs'] = args.max_epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size

    # Create trainer
    trainer = BitMarTrainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.setup_model_and_data(max_samples=args.max_samples)
        trainer.load_checkpoint(args.resume)

    # Start training
    trainer.train(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
