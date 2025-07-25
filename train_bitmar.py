"""
Training script for BitMar model
Handles multimodal training with episodic memory and attention analysis
"""

from src.attention_analysis import analyze_model_attention
from src.dataset import create_data_module
from src.model import create_bitmar_model, count_parameters
from src.wandb_logger import BitMarWandbLogger
from src.attention_visualizer import AttentionHeadAnalyzer
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

# Try to import bitsandbytes for 8-bit optimizer
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Warning: bitsandbytes not available. Install with: pip install bitsandbytes")

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

        # Initialize enhanced wandb logger and attention analyzer
        self.wandb_logger = None
        self.attention_analyzer = None
        self.setup_logging_systems()

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

    def setup_logging_systems(self):
        """Initialize enhanced wandb logger and attention analyzer"""
        wandb_config = self.config.get('wandb', {})
        
        # Check if wandb should be used
        use_wandb = (
            wandb_config.get('api_key') or 
            os.getenv('WANDB_API_KEY') or
            wandb_config.get('project')
        )
        
        if use_wandb:
            try:
                # Set API key if provided in config
                if wandb_config.get('api_key'):
                    os.environ['WANDB_API_KEY'] = wandb_config['api_key']

                # Initialize enhanced wandb logger
                run_name = f"bitmar-{self.config['training']['max_epochs']}epochs-{wandb.util.generate_id()[:8]}"
                
                self.wandb_logger = BitMarWandbLogger(
                    project_name=wandb_config.get('project', 'bitmar-babylm'),
                    config=self.config,
                    run_name=run_name
                )
                
                self.use_wandb = True
                logger.info("Enhanced Wandb logger initialized successfully")
                
            except Exception as e:
                logger.warning(f"Wandb initialization failed: {e}")
                logger.info("Continuing training without wandb logging")
                self.use_wandb = False
                self.wandb_logger = None
        else:
            self.use_wandb = False
            self.wandb_logger = None
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

        # Log model size with enhanced wandb logger
        if self.wandb_logger:
            self.wandb_logger.log_model_size_metrics(self.model)

        # Initialize attention analyzer
        self.attention_analyzer = AttentionHeadAnalyzer(
            model=self.model,
            tokenizer=self.model.tokenizer,
            save_dir=str(self.attention_dir),
            wandb_logger=self.wandb_logger,
            track_top_k=self.config.get('attention_analysis', {}).get('track_top_k', 10)
        )

        # Create data module
        self.data_module = create_data_module(self.config['data'])
        self.data_module.setup(max_samples=max_samples)

        # Setup optimizer and scheduler
        self.setup_optimizer()

    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Use AdamW8bit if bitsandbytes is available, otherwise fallback to regular AdamW
        if BITSANDBYTES_AVAILABLE:
            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
            logger.info(f"Using AdamW8bit optimizer for memory efficiency")
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
            logger.info(f"Using regular AdamW optimizer (bitsandbytes not available)")

        # Learning rate scheduler with proper step-based scheduling
        if self.config['training']['scheduler'] == 'cosine':
            # Calculate total training steps for proper cosine annealing
            train_loader = self.data_module.train_dataloader()
            steps_per_epoch = len(train_loader)
            total_steps = steps_per_epoch * self.config['training']['max_epochs']
            
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,  # Use total steps, not epochs
                eta_min=self.config['training']['min_lr']
            )
            self.scheduler_step_mode = 'step'  # Step every training step, not epoch
            logger.info(f"Cosine scheduler: {total_steps} total steps, eta_min={self.config['training']['min_lr']}")
        else:
            self.scheduler = None
            self.scheduler_step_mode = 'epoch'

        logger.info(
            f"Optimizer: {'AdamW8bit' if BITSANDBYTES_AVAILABLE else 'AdamW'} with LR={self.config['training']['learning_rate']}")
        if self.scheduler:
            logger.info(f"Scheduler: {self.config['training']['scheduler']} ({'step-based' if self.scheduler_step_mode == 'step' else 'epoch-based'})")

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
            try:
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
                
                # Check for invalid loss
                if not torch.isfinite(loss):
                    logger.warning(f"Invalid loss at step {self.global_step}: {loss.item()}")
                    continue

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.config['training']['gradient_clip_val'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip_val']
                    )

                self.optimizer.step()

                # Update metrics
                epoch_losses.append(loss.item())

                # Compute additional metrics with error handling
                if outputs['memory_usage'] is not None:
                    try:
                        memory_entropy = self._compute_memory_entropy(outputs['memory_usage'])
                        if np.isfinite(memory_entropy):
                            epoch_metrics['memory_usage_entropy'] += memory_entropy
                    except Exception as e:
                        logger.warning(f"Memory entropy computation failed at step {self.global_step}: {e}")

                if outputs['text_features'] is not None and outputs['vision_latent'] is not None:
                    try:
                        cross_modal_sim = self._compute_cross_modal_similarity(
                            outputs['text_features'], outputs['vision_latent']
                        )
                        if np.isfinite(cross_modal_sim):
                            epoch_metrics['cross_modal_similarity'] += cross_modal_sim
                    except Exception as e:
                        logger.warning(f"Cross-modal similarity computation failed at step {self.global_step}: {e}")

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{np.mean(epoch_losses):.4f}"
                })

                # Enhanced logging with wandb logger
                log_every_n_steps = self.config.get('wandb', {}).get('log_every_n_steps', 50)
                if self.wandb_logger and batch_idx % log_every_n_steps == 0:
                    try:
                        # Log all metrics in a single consolidated call
                        log_quantization = self.global_step % (log_every_n_steps * 10) == 0
                        memory_module = self.model.memory if hasattr(self.model, 'memory') else None
                        
                        self.wandb_logger.log_consolidated_metrics(
                            outputs=outputs,
                            epoch=epoch,
                            step=self.global_step,
                            lr=self.optimizer.param_groups[0]['lr'],
                            model=self.model,
                            memory_module=memory_module,
                            log_quantization=log_quantization
                        )
                            
                    except Exception as e:
                        logger.warning(f"Wandb logging failed at step {self.global_step}: {e}")

                # Attention analysis (less frequent to avoid overhead)
                if (self.attention_analyzer and 
                    self.global_step % self.config.get('attention_analysis', {}).get('log_every_n_steps', 100) == 0):
                    
                    try:
                        self.attention_analyzer.analyze_batch_attention(
                            outputs, batch['input_ids'], self.global_step
                        )
                    except Exception as e:
                        logger.warning(f"Attention analysis failed at step {self.global_step}: {e}")

                self.global_step += 1
                
                # Step learning rate scheduler if step-based
                if self.scheduler and hasattr(self, 'scheduler_step_mode') and self.scheduler_step_mode == 'step':
                    self.scheduler.step()
                
                # Memory cleanup every 100 steps to prevent OOM
                if self.global_step % 100 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                logger.error(f"Training batch {batch_idx} failed at step {self.global_step}: {e}")
                logger.error(f"Skipping batch and continuing training...")
                
                # Clear GPU cache after error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                self.global_step += 1
                continue

        # Average metrics over epoch with safety checks
        epoch_metrics['train_loss'] = np.mean(epoch_losses) if epoch_losses else float('inf')
        epoch_metrics['memory_usage_entropy'] = (epoch_metrics['memory_usage_entropy'] / len(train_loader)) if len(train_loader) > 0 else 0.0
        epoch_metrics['cross_modal_similarity'] = (epoch_metrics['cross_modal_similarity'] / len(train_loader)) if len(train_loader) > 0 else 0.0

        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return epoch_metrics

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        val_loaders = self.data_module.val_dataloader()  # Returns list of loaders

        val_losses = []
        val_metrics = {
            'val_loss': 0.0,
            'val_memory_entropy': 0.0,
            'val_cross_modal_similarity': 0.0
        }

        with torch.no_grad():
            # Handle multiple validation dataloaders
            for loader_idx, val_loader in enumerate(val_loaders):
                logger.info(f"Validating on dataset {loader_idx + 1}/{len(val_loaders)}")
                
                for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Validation-{loader_idx+1}")):
                    try:
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

                        # Safely extract loss
                        if outputs['loss'] is not None and torch.isfinite(outputs['loss']):
                            val_losses.append(outputs['loss'].item())

                        # Compute additional metrics with safety checks
                        if outputs['memory_usage'] is not None:
                            try:
                                memory_entropy = self._compute_memory_entropy(outputs['memory_usage'])
                                if np.isfinite(memory_entropy):
                                    val_metrics['val_memory_entropy'] += memory_entropy
                            except Exception as e:
                                logger.warning(f"Memory entropy computation failed: {e}")

                        if outputs['text_features'] is not None and outputs['vision_latent'] is not None:
                            try:
                                cross_modal_sim = self._compute_cross_modal_similarity(
                                    outputs['text_features'], outputs['vision_latent']
                                )
                                if np.isfinite(cross_modal_sim):
                                    val_metrics['val_cross_modal_similarity'] += cross_modal_sim
                            except Exception as e:
                                logger.warning(f"Cross-modal similarity computation failed: {e}")
                                
                    except Exception as e:
                        logger.warning(f"Validation batch {batch_idx} in loader {loader_idx} failed: {e}")
                        continue

        # Calculate total number of batches across all loaders for averaging
        total_batches = sum(len(loader) for loader in val_loaders) if val_loaders else 1

        # Average metrics with safety checks
        val_metrics['val_loss'] = np.mean(val_losses) if val_losses else float('inf')
        val_metrics['val_memory_entropy'] = (val_metrics['val_memory_entropy'] / total_batches) if total_batches > 0 else 0.0
        val_metrics['val_cross_modal_similarity'] = (val_metrics['val_cross_modal_similarity'] / total_batches) if total_batches > 0 else 0.0

        # Clear GPU cache and restore training mode
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.model.train()  # Restore training mode
        
        logger.info(f"Validation completed - Loss: {val_metrics['val_loss']:.4f}")
        
        return val_metrics

    def _compute_memory_entropy(self, memory_usage: torch.Tensor) -> float:
        """Compute entropy of memory usage distribution"""
        try:
            # Check for valid input
            if memory_usage is None or memory_usage.numel() == 0:
                return 0.0
            
            # Check for all-zero usage
            usage_sum = memory_usage.sum()
            if usage_sum <= 1e-8:
                return 0.0
            
            # Normalize to probabilities
            probs = memory_usage / usage_sum
            
            # Compute entropy with numerical stability
            log_probs = torch.log(probs + 1e-8)
            entropy = -(probs * log_probs).sum().item()
            
            # Return finite value only
            return entropy if np.isfinite(entropy) else 0.0
            
        except Exception as e:
            logger.warning(f"Memory entropy computation failed: {e}")
            return 0.0

    def _compute_cross_modal_similarity(
        self,
        text_latent: torch.Tensor,
        vision_latent: torch.Tensor
    ) -> float:
        """Compute cosine similarity between text and vision features"""
        try:
            # Check for valid inputs
            if text_latent is None or vision_latent is None:
                return 0.0
            
            if text_latent.numel() == 0 or vision_latent.numel() == 0:
                return 0.0
            
            # Pool text features (mean over sequence)
            text_pooled = text_latent.mean(dim=1)  # [batch_size, feature_dim]

            # Check dimensions match
            if text_pooled.shape[-1] != vision_latent.shape[-1]:
                logger.warning(f"Dimension mismatch: text {text_pooled.shape} vs vision {vision_latent.shape}")
                return 0.0

            # Compute cosine similarity with numerical stability
            cos_sim = torch.cosine_similarity(text_pooled, vision_latent, dim=1)
            similarity = cos_sim.mean().item()
            
            # Return finite value only
            return similarity if np.isfinite(similarity) else 0.0
            
        except Exception as e:
            logger.warning(f"Cross-modal similarity computation failed: {e}")
            return 0.0

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        try:
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
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint at epoch {epoch}: {e}")
            logger.error("Training will continue but checkpoint is not saved")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint"""
        try:
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
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            logger.error("Starting training from scratch")
            return 0

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

            # Update learning rate scheduler (only for epoch-based schedulers)
            if self.scheduler and hasattr(self, 'scheduler_step_mode') and self.scheduler_step_mode == 'epoch':
                self.scheduler.step()
                logger.info(f"Scheduler stepped (epoch-based), new LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Log validation metrics with enhanced logger
            if self.wandb_logger:
                self.wandb_logger.log_validation_metrics(
                    val_metrics['val_loss'],
                    np.exp(val_metrics['val_loss']),  # Perplexity
                    self.global_step,
                    memory_entropy=val_metrics['val_memory_entropy'],
                    cross_modal_similarity=val_metrics['val_cross_modal_similarity']
                )

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics['epoch'] = epoch
            all_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']

            # Log epoch summary
            if self.wandb_logger:
                self.wandb_logger.log_epoch_summary(
                    epoch=epoch,
                    train_loss=train_metrics['train_loss'],
                    val_loss=val_metrics['val_loss'],
                    memory_efficiency=train_metrics['memory_usage_entropy'],
                    step=self.global_step,
                    cross_modal_similarity=train_metrics['cross_modal_similarity']
                )

            # Log metrics
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(
                f"Memory Entropy: {train_metrics['memory_usage_entropy']:.4f}")
            logger.info(
                f"Cross-Modal Similarity: {train_metrics['cross_modal_similarity']:.4f}")

            # Create attention visualizations (every few epochs to avoid overhead)
            if (self.attention_analyzer and 
                (epoch + 1) % self.config.get('attention_analysis', {}).get('viz_every_n_epochs', 2) == 0):
                
                logger.info("Creating attention visualizations...")
                
                try:
                    # Create attention head heatmaps
                    for attention_type in ['encoder', 'decoder', 'cross_modal']:
                        self.attention_analyzer.create_attention_head_heatmap(
                            self.global_step, attention_type
                        )
                    
                    # Create timeline plots
                    self.attention_analyzer.create_attention_timeline_plot(self.global_step)
                    
                    # Save top attention heads
                    for attention_type in ['encoder', 'decoder', 'cross_modal']:
                        self.attention_analyzer.save_top_heads(self.global_step, attention_type)
                        
                except Exception as e:
                    logger.warning(f"Attention visualization creation failed: {e}")
                
                # Create visualizations with wandb logger
                if self.wandb_logger and hasattr(self.model, 'memory'):
                    try:
                        # Memory heatmaps
                        self.wandb_logger.create_memory_heatmap(
                            self.model.memory.memory_usage,
                            self.model.memory.memory_age,
                            self.global_step
                        )
                        
                        # Quantization plots
                        self.wandb_logger.create_quantization_plot(self.model, self.global_step)
                        
                    except Exception as e:
                        logger.warning(f"Wandb visualization creation failed: {e}")

            if self.use_wandb:
                wandb.log(all_metrics)

            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']

            self.save_checkpoint(epoch, is_best=is_best)

        # Final analysis and cleanup
        logger.info("Training completed! Running final analysis...")
        
        if self.attention_analyzer:
            # Generate final attention report
            final_report = self.attention_analyzer.generate_attention_report(self.global_step)
            logger.info(f"Final attention analysis: {final_report}")
            
            # Save final top heads
            for attention_type in ['encoder', 'decoder', 'cross_modal']:
                self.attention_analyzer.save_top_heads(self.global_step, attention_type, k=20)
        
        # Close wandb logger
        if self.wandb_logger:
            self.wandb_logger.finish()

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
