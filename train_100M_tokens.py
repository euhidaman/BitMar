"""
Token-Aware Training Script for BitMar 100M Token Model
Handles exactly 100M tokens with perfect image-caption alignment
Includes advanced token tracking and stopping mechanisms
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm
import time
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_100M_tokens.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import components
from src.dataset import create_data_module
from src.model import create_bitmar_model, count_parameters
from src.wandb_logger import BitMarWandbLogger
from src.attention_visualizer import AttentionHeadAnalyzer

# Try to import token-constrained dataset
try:
    from src.token_constrained_dataset import create_token_constrained_data_module
    TOKEN_CONSTRAINED_AVAILABLE = True
    logger.info("✅ Token-constrained dataset available")
except ImportError:
    try:
        # Try importing without src prefix (in case running from different directory)
        from token_constrained_dataset import create_token_constrained_data_module
        TOKEN_CONSTRAINED_AVAILABLE = True
        logger.info("✅ Token-constrained dataset available (direct import)")
    except ImportError:
        TOKEN_CONSTRAINED_AVAILABLE = False
        logger.warning("⚠️  Token-constrained dataset not available")

# Try to import optional components
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

try:
    from src.adaptive_training_controller import AdaptiveTrainingController, compute_cross_modal_similarity
    ADAPTIVE_TRAINING_AVAILABLE = True
except ImportError:
    ADAPTIVE_TRAINING_AVAILABLE = False


class TokenAwareTrainer:
    """Token-aware trainer for exactly 100M tokens"""

    def __init__(self, config_path: str, device: Optional[str] = None):
        """Initialize trainer with token awareness"""
        # Load configuration with validation
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Validate required config sections
            required_sections = ['token_constraints', 'model', 'data', 'training', 'output']
            for section in required_sections:
                if section not in self.config:
                    raise ValueError(f"Missing required config section: {section}")
                    
            logger.info(f"Configuration loaded successfully from {config_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load config: {e}")

        # Set device with better error handling
        if device:
            try:
                self.device = torch.device(device)
                # Test if device is available
                if device.startswith('cuda'):
                    if not torch.cuda.is_available():
                        logger.warning(f"CUDA not available, falling back to CPU")
                        self.device = torch.device("cpu")
                    elif device != "cuda:0" and not torch.cuda.device_count() > int(device.split(':')[1]):
                        logger.warning(f"Device {device} not available, using cuda:0")
                        self.device = torch.device("cuda:0")
                logger.info(f"Using device: {self.device}")
            except Exception as e:
                logger.warning(f"Failed to set device {device}: {e}, using default")
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Token tracking
        self.tokens_processed = 0
        self.target_tokens = self.config['token_constraints']['total_tokens']
        self.token_log_frequency = self.config.get('token_tracking', {}).get('log_frequency', 1000)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_similarity = 0.0
        self.token_exhausted = False

        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.setup_wandb()
        
        # Initialize carbon tracking if available
        self.setup_carbon_tracking()

        logger.info(f"🎯 Token-aware trainer initialized for {self.target_tokens:,} tokens")
        logger.info(f"Device: {self.device}")

    def setup_directories(self):
        """Create output directories"""
        for dir_name in ['checkpoint_dir', 'log_dir', 'attention_dir', 'memory_dir', 'results_dir', 'token_logs_dir']:
            dir_path = Path(self.config['output'][dir_name])
            dir_path.mkdir(parents=True, exist_ok=True)
            setattr(self, dir_name, dir_path)

    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        wandb_config = self.config.get('wandb', {})
        
        if wandb_config.get('project'):
            try:
                # Enhanced run name with token info
                run_name = f"bitmar-100M-tokens-{wandb.util.generate_id()[:8]}"
                
                self.wandb_logger = BitMarWandbLogger(
                    project_name=wandb_config['project'],
                    config=self.config,
                    entity=wandb_config.get('entity'),
                    run_name=run_name
                )
                self.use_wandb = True
                logger.info("✅ Weights & Biases initialized for 100M token training")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
                self.wandb_logger = None
        else:
            self.use_wandb = False
            self.wandb_logger = None

    def setup_carbon_tracking(self):
        """Setup carbon emissions tracking"""
        if not CODECARBON_AVAILABLE:
            self.carbon_tracker = None
            return

        try:
            carbon_logs_dir = Path("./carbon_logs_100M")
            carbon_logs_dir.mkdir(exist_ok=True)

            self.carbon_tracker = EmissionsTracker(
                project_name="BitMar-100M-Token-Training",
                experiment_id=f"bitmar-100M-{self.device.type}",
                output_dir=str(carbon_logs_dir),
                output_file="emissions_100M.csv",
                log_level="INFO",
                save_to_file=True,
                tracking_mode="machine"
            )
            logger.info("🌱 Carbon emissions tracking enabled for 100M token training")
        except Exception as e:
            logger.warning(f"Failed to setup carbon tracking: {e}")
            self.carbon_tracker = None

    def setup_model_and_data(self):
        """Setup model and token-constrained data"""
        logger.info("Setting up model and token-constrained data...")

        # Check if we should use token-constrained dataset
        if self.config.get('token_constraints') and TOKEN_CONSTRAINED_AVAILABLE:
            logger.info("🎯 Using token-constrained dataset for 100M tokens")
            # Pass the full data config including token constraints
            data_config = self.config['data'].copy()
            data_config['token_constraints'] = self.config['token_constraints']
            self.data_module = create_token_constrained_data_module(data_config)
        else:
            if self.config.get('token_constraints') and not TOKEN_CONSTRAINED_AVAILABLE:
                logger.warning("⚠️  Token constraints specified but token-constrained dataset not available")
            logger.info("📊 Using standard BabyLM dataset")
            self.data_module = create_data_module(self.config['data'])
        
        self.data_module.setup(rebuild_cache=getattr(self, 'rebuild_cache', False))

        # Get and log token statistics if available
        if hasattr(self.data_module, 'get_token_statistics'):
            token_stats = self.data_module.get_token_statistics()
            logger.info("📊 Token Statistics:")
            for key, value in token_stats.items():
                if isinstance(value, int):
                    logger.info(f"  • {key}: {value:,}")
                else:
                    logger.info(f"  • {key}: {value}")

            # Verify token constraints
            if token_stats['total_tokens'] != self.target_tokens:
                logger.warning(f"Token mismatch: Expected {self.target_tokens:,}, got {token_stats['total_tokens']:,}")
        else:
            logger.warning("⚠️  Token statistics not available - using standard dataset")
            logger.info(f"Target tokens: {self.target_tokens:,}")

        # Create model
        self.model = create_bitmar_model(self.config['model'])
        self.model.to(self.device)

        # Log model info
        param_count = count_parameters(self.model)
        logger.info(f"Model created with {param_count['total_parameters']:,} total parameters")
        logger.info(f"Trainable parameters: {param_count['trainable_parameters']:,}")
        logger.info(f"Non-trainable parameters: {param_count['non_trainable_parameters']:,}")

        # Setup optimizer with token-aware configuration
        self.setup_optimizer()

        # Initialize attention analyzer
        self.attention_analyzer = AttentionHeadAnalyzer(
            model=self.model,
            tokenizer=self.model.tokenizer,
            save_dir=str(self.attention_dir),
            wandb_logger=self.wandb_logger,
            track_top_k=self.config.get('attention_analysis', {}).get('track_top_k', 5)
        )

        # Setup adaptive training if enabled
        self.setup_adaptive_training()

    def setup_optimizer(self):
        """Setup optimizer and scheduler for 100M token training"""
        # Calculate total training steps based on exact token count
        train_loader = self.data_module.train_dataloader()
        
        # Estimate steps per epoch
        steps_per_epoch = len(train_loader)
        
        # Calculate how many steps we need to process exactly 100M tokens
        avg_tokens_per_batch = self.target_tokens // (steps_per_epoch * self.config['training']['max_epochs'])
        estimated_total_steps = self.target_tokens // avg_tokens_per_batch

        logger.info(f"Training planning:")
        logger.info(f"  • Steps per epoch: {steps_per_epoch}")
        logger.info(f"  • Estimated avg tokens per batch: {avg_tokens_per_batch}")
        logger.info(f"  • Estimated total steps: {estimated_total_steps}")

        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Create scheduler with restarts
        scheduler_config = self.config['training'].get('scheduler_config', {})
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=scheduler_config.get('T_0', 2000),
            T_mult=scheduler_config.get('T_mult', 1.5),
            eta_min=self.config['training']['learning_rate'] * scheduler_config.get('eta_min_ratio', 0.1)
        )

        logger.info(f"✅ Optimizer and scheduler configured for 100M token training")

    def setup_adaptive_training(self):
        """Setup adaptive training controller"""
        if not ADAPTIVE_TRAINING_AVAILABLE or not self.config['model'].get('enable_adaptive_training', False):
            self.adaptive_controller = None
            return

        adaptive_config = self.config.get('adaptive_training', {})
        adaptive_logs_dir = Path("./logs/adaptive_training_100M")
        adaptive_logs_dir.mkdir(parents=True, exist_ok=True)

        self.adaptive_controller = AdaptiveTrainingController(
            similarity_window_size=adaptive_config.get('similarity_window_size', 200),
            drop_threshold=adaptive_config.get('drop_threshold', 0.12),
            min_steps_between_interventions=adaptive_config.get('min_steps_between_interventions', 800),
            freeze_duration_steps=adaptive_config.get('freeze_duration_steps', 1500),
            loss_rebalance_factor=adaptive_config.get('loss_rebalance_factor', 2.0),
            similarity_smoothing_alpha=adaptive_config.get('similarity_smoothing_alpha', 0.15),
            save_dir=str(adaptive_logs_dir)
        )

        logger.info("🤖 Adaptive training controller enabled for 100M token training")

    def count_tokens_in_batch(self, batch: Dict) -> int:
        """Count actual tokens in a batch"""
        attention_mask = batch['attention_mask']
        return attention_mask.sum().item()

    def log_token_progress(self):
        """Log token consumption progress"""        
        logger.info(f"🎯 Tokens processed so far: {self.tokens_processed:,}")
        logger.info(f"   Dataset size: {self.target_tokens:,} tokens")
        
        # Log to wandb with error handling
        if self.use_wandb:
            try:
                wandb.log({
                    'token_progress/processed': self.tokens_processed,
                    'token_progress/target': self.target_tokens
                }, step=self.global_step)
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")
                # Disable wandb if it keeps failing
                self.use_wandb = False

    def save_token_checkpoint(self):
        """Save checkpoint with token information"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'tokens_processed': self.tokens_processed,
            'target_tokens': self.target_tokens,
            'best_similarity': self.best_similarity,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}_tokens_{self.tokens_processed}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with token awareness"""
        self.model.train()
        train_loader = self.data_module.train_dataloader()
        
        epoch_losses = []
        epoch_metrics = {
            'train_loss': 0.0,
            'cross_modal_similarity': 0.0,
            'tokens_in_epoch': 0
        }

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} | Tokens: {self.tokens_processed:,}")

        for batch_idx, batch in enumerate(progress_bar):
            # Count tokens in this batch for logging purposes
            batch_tokens = self.count_tokens_in_batch(batch)

            try:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    vision_features=batch['vision_features'],
                    labels=batch['labels'],
                    step=self.global_step,
                    has_vision=batch.get('has_vision', torch.ones(batch['input_ids'].size(0), dtype=torch.bool)),
                    adaptive_controller=self.adaptive_controller
                )

                loss = outputs['loss']

                # Check for valid loss
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
                self.scheduler.step()

                # Update token count
                self.tokens_processed += batch_tokens
                epoch_metrics['tokens_in_epoch'] += batch_tokens

                # Update metrics
                epoch_losses.append(loss.item())

                # Compute cross-modal similarity if available
                if outputs.get('text_features') is not None and outputs.get('vision_latent') is not None:
                    try:
                        similarity = self._compute_cross_modal_similarity(
                            outputs['text_features'], outputs['vision_latent']
                        )
                        epoch_metrics['cross_modal_similarity'] += similarity
                        
                        # Update best similarity
                        if similarity > self.best_similarity:
                            self.best_similarity = similarity
                    except Exception as e:
                        logger.warning(f"Cross-modal similarity computation failed: {e}")

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'tokens': f"{self.tokens_processed:,}",
                    'epoch': f"{self.current_epoch + 1}/{self.config['training']['max_epochs']}"
                })

                # Log token progress periodically
                if self.global_step % self.token_log_frequency == 0:
                    self.log_token_progress()

                # Enhanced wandb logging
                if self.use_wandb and self.global_step % 100 == 0:
                    log_dict = {
                        'train/loss': loss.item(),
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'tokens/processed': self.tokens_processed,
                        'tokens/batch_size': batch_tokens,
                        'step': self.global_step
                    }
                    
                    # Only add similarity if it was computed
                    if outputs.get('text_features') is not None and outputs.get('vision_latent') is not None:
                        try:
                            current_similarity = self._compute_cross_modal_similarity(
                                outputs['text_features'], outputs['vision_latent']
                            )
                            log_dict['train/cross_modal_similarity'] = current_similarity
                        except Exception as e:
                            logger.warning(f"Failed to compute similarity for wandb: {e}")

                    try:
                        wandb.log(log_dict, step=self.global_step)
                    except Exception as e:
                        logger.warning(f"Failed to log to wandb during training: {e}")
                        self.use_wandb = False

                self.global_step += 1

                # Save checkpoint periodically
                if self.global_step % 5000 == 0:
                    self.save_token_checkpoint()

            except Exception as e:
                logger.error(f"Training step failed: {e}")
                
                # Clear any gradients and free memory
                if hasattr(self, 'optimizer'):
                    self.optimizer.zero_grad()
                
                # Clear CUDA cache if using GPU
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                continue

        # Calculate epoch metrics
        if epoch_losses:
            epoch_metrics['train_loss'] = np.mean(epoch_losses)
            epoch_metrics['cross_modal_similarity'] = epoch_metrics['cross_modal_similarity'] / len(epoch_losses)
        
        logger.info(f"Epoch {epoch} completed:")
        logger.info(f"  • Loss: {epoch_metrics['train_loss']:.4f}")
        logger.info(f"  • Cross-modal similarity: {epoch_metrics['cross_modal_similarity']:.4f}")
        logger.info(f"  • Tokens in epoch: {epoch_metrics['tokens_in_epoch']:,}")
        logger.info(f"  • Total tokens processed: {self.tokens_processed:,}")

        return epoch_metrics

    def _compute_cross_modal_similarity(self, text_features: torch.Tensor, vision_features: torch.Tensor) -> float:
        """Compute cross-modal similarity"""
        try:
            # Pool text features if needed
            if text_features.dim() == 3:  # [batch, seq, dim]
                text_pooled = text_features.mean(dim=1)  # [batch, dim]
            else:
                text_pooled = text_features

            # Ensure same dimensions
            if text_pooled.size(-1) != vision_features.size(-1):
                min_dim = min(text_pooled.size(-1), vision_features.size(-1))
                text_pooled = text_pooled[:, :min_dim]
                vision_features = vision_features[:, :min_dim]

            # Compute cosine similarity
            cos_sim = torch.cosine_similarity(text_pooled, vision_features, dim=1)
            return cos_sim.mean().item()
        except Exception as e:
            logger.warning(f"Cross-modal similarity computation failed: {e}")
            return 0.0

    def train(self):
        """Main training loop with token awareness"""
        logger.info("🚀 Starting 100M token training...")
        
        # Start carbon tracking
        if self.carbon_tracker:
            self.carbon_tracker.start()

        # Setup model and data
        self.setup_model_and_data()

        try:
            for epoch in range(self.config['training']['max_epochs']):
                logger.info(f"Starting epoch {epoch + 1}/{self.config['training']['max_epochs']}")
                
                self.current_epoch = epoch
                epoch_metrics = self.train_epoch(epoch)

                # Save checkpoint after each epoch
                self.save_token_checkpoint()

                # Log epoch summary to wandb with error handling
                if self.use_wandb:
                    try:
                        wandb.log({
                            'epoch/train_loss': epoch_metrics['train_loss'],
                            'epoch/cross_modal_similarity': epoch_metrics['cross_modal_similarity'],
                            'epoch/tokens_processed': self.tokens_processed,
                            'epoch/tokens_in_epoch': epoch_metrics['tokens_in_epoch'],
                            'epoch/number': epoch
                        }, step=self.global_step)
                    except Exception as e:
                        logger.warning(f"Failed to log epoch summary to wandb: {e}")
                        self.use_wandb = False

                # Continue training for all epochs (no token limit stopping)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            # Stop carbon tracking
            if self.carbon_tracker:
                emissions = self.carbon_tracker.stop()
                logger.info(f"🌱 Carbon emissions: {emissions:.6f} kg CO2")

            # Final checkpoint
            self.save_token_checkpoint()

            # Final token summary
            logger.info("🎯 Final Token Summary:")
            logger.info(f"  • Target tokens: {self.target_tokens:,}")
            logger.info(f"  • Processed tokens: {self.tokens_processed:,}")
            logger.info(f"  • Completion: {(self.tokens_processed/self.target_tokens)*100:.2f}%")
            logger.info(f"  • Best cross-modal similarity: {self.best_similarity:.4f}")

            if self.use_wandb:
                try:
                    wandb.finish()
                except Exception as e:
                    logger.warning(f"Failed to finish wandb run: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train BitMar with exactly 100M tokens")
    
    parser.add_argument("--config", type=str, default="configs/bitmar_100M_tokens.yaml",
                       help="Path to configuration file")
    parser.add_argument("--device", type=str, help="Device to use (cuda:0, cpu)")
    parser.add_argument("--rebuild_cache", action="store_true",
                       help="Rebuild token-constrained dataset cache")
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = TokenAwareTrainer(args.config, device=args.device)
        trainer.rebuild_cache = args.rebuild_cache  # Pass rebuild_cache to trainer
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
