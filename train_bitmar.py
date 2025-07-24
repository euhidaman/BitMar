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

# Import attention evolution tracker
try:
    from attention_evolution_tracker import AttentionEvolutionTracker
    ATTENTION_TRACKING_AVAILABLE = True
except ImportError:
    ATTENTION_TRACKING_AVAILABLE = False
    print("Warning: attention_evolution_tracker not available")


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

    def __init__(self, config, device: Optional[str] = None):
        """Initialize trainer with configuration

        Args:
            config: Either a string path to config file or a loaded config dictionary
            device: Optional device specification
        """
        # Handle both config path (string) and loaded config (dict)
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError("config must be either a string path or a dictionary")

        # Set device - prioritize user specification, then config, then auto-detect
        if device:
            self.device = torch.device(device)
        elif self.config.get('training', {}).get('device'):
            self.device = torch.device(self.config['training']['device'])
        else:
            # Force CUDA device index specification when available
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Ensure CUDA is initialized if available
        if self.device.type == 'cuda':
            torch.cuda.init()
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(self.device)}")
            # Don't set default tensor type to avoid DataLoader issues
            logger.info("CUDA initialized, model will be moved to GPU explicitly")
        else:
            logger.warning("CUDA not available, using CPU. Training will be slow.")

        # Initialize tracking variables
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self._last_model_device = None
        self._device_warnings_count = 0

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

        # Force model to GPU with verification
        self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")

        # Verify model is actually on GPU
        model_device = next(self.model.parameters()).device
        logger.info(f"Model parameters are on device: {model_device}")

        # Force all model components to GPU
        for name, param in self.model.named_parameters():
            if param.device != self.device:
                logger.warning(f"Parameter {name} on wrong device {param.device}, moving to {self.device}")
                param.data = param.data.to(self.device)

        # Check GPU memory usage after model loading
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            logger.info(f"GPU memory allocated after model loading: {memory_allocated:.2f} GB")

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
            track_top_k=self.config.get(
                'attention_analysis', {}).get('track_top_k', 10)
        )

        # Initialize attention evolution tracker
        if ATTENTION_TRACKING_AVAILABLE:
            self.attention_evolution_tracker = AttentionEvolutionTracker(
                save_dir=str(self.attention_dir / "attention_evolution")
            )
            logger.info("Attention evolution tracker initialized")
        else:
            self.attention_evolution_tracker = None
            logger.warning("Attention evolution tracker not available")

        # Create data module
        self.data_module = create_data_module(self.config['data'])
        self.data_module.setup(max_samples=max_samples)

        # Setup optimizer and scheduler
        self.setup_optimizer()

    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        optimizer_type = self.config.get('optimizer', 'adamw').lower()

        # Use AdamW8bit if bitsandbytes is available and requested
        if BITSANDBYTES_AVAILABLE and optimizer_type == 'adamw8bit':
            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
            logger.info(f"Using AdamW8bit optimizer for memory efficiency")
        elif optimizer_type == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
            logger.info(f"Using AdamW optimizer")
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
            logger.info(f"Using Adam optimizer")
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['training']['weight_decay']
            )
            logger.info(f"Using SGD optimizer")
        elif optimizer_type == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                alpha=0.99,
                eps=1e-8,
                weight_decay=self.config['training']['weight_decay']
            )
            logger.info(f"Using RMSprop optimizer")
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        # Learning rate scheduler with proper step-based scheduling
        if self.config['training']['scheduler'] == 'cosine':
            # Calculate total training steps for proper cosine annealing
            train_loader = self.data_module.train_dataloader()
            steps_per_epoch = len(train_loader)
            total_steps = steps_per_epoch * \
                self.config['training']['max_epochs']

            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,  # Use total steps, not epochs
                eta_min=self.config['training']['min_lr']
            )
            self.scheduler_step_mode = 'step'  # Step every training step, not epoch
            logger.info(
                f"Cosine scheduler: {total_steps} total steps, eta_min={self.config['training']['min_lr']}")
        else:
            self.scheduler = None
            self.scheduler_step_mode = 'epoch'

        logger.info(
            f"Optimizer: {optimizer_type.upper()} with LR={self.config['training']['learning_rate']}")
        if self.scheduler:
            logger.info(
                f"Scheduler: {self.config['training']['scheduler']} ({'step-based' if self.scheduler_step_mode == 'step' else 'epoch-based'})")

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
                # Use silent device checking every 500 steps (much less frequent)
                if self.global_step % 500 == 0:
                    self._silent_device_check()

                # Use safe batch transfer method
                batch = self._safe_batch_to_device(batch)

                # Forward pass with device-aware error handling
                try:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        vision_features=batch['vision_features'],
                        labels=batch['labels']
                    )
                    loss = outputs['loss']
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "device" in str(e).lower():
                        logger.warning(f"Device/memory error in forward pass: {e}")
                        # Force device consistency and retry
                        self._force_model_device_consistency()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            vision_features=batch['vision_features'],
                            labels=batch['labels']
                        )
                        loss = outputs['loss']
                    else:
                        raise e

                # Check for invalid loss
                if not torch.isfinite(loss):
                    logger.warning(
                        f"Invalid loss at step {self.global_step}: {loss.item()}")
                    continue

                # Backward pass with device-aware error handling
                try:
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping
                    if self.config['training']['gradient_clip_val'] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['training']['gradient_clip_val']
                        )

                    self.optimizer.step()

                except RuntimeError as e:
                    if "device" in str(e).lower():
                        logger.warning(f"Device error in backward pass: {e}")
                        # Recreate optimizer and retry
                        self._create_device_pinned_optimizer()

                        self.optimizer.zero_grad()
                        loss.backward()

                        if self.config['training']['gradient_clip_val'] > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config['training']['gradient_clip_val']
                            )

                        self.optimizer.step()
                    else:
                        raise e

                # Update metrics
                epoch_losses.append(loss.item())

                # Compute additional metrics with error handling
                if outputs['memory_usage'] is not None:
                    try:
                        memory_entropy = self._compute_memory_entropy(
                            outputs['memory_usage'])
                        if np.isfinite(memory_entropy):
                            epoch_metrics['memory_usage_entropy'] += memory_entropy
                    except Exception as e:
                        logger.warning(
                            f"Memory entropy computation failed at step {self.global_step}: {e}")

                if outputs['text_features'] is not None and outputs['vision_latent'] is not None:
                    try:
                        cross_modal_sim = self._compute_cross_modal_similarity(
                            outputs['text_features'], outputs['vision_latent']
                        )
                        if np.isfinite(cross_modal_sim):
                            epoch_metrics['cross_modal_similarity'] += cross_modal_sim
                    except Exception as e:
                        logger.warning(
                            f"Cross-modal similarity computation failed at step {self.global_step}: {e}")

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{np.mean(epoch_losses):.4f}"
                })

                # Log GPU memory usage every 100 steps
                if self.global_step % 100 == 0 and torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
                    gpu_util = torch.cuda.utilization(self.device) if hasattr(torch.cuda, 'utilization') else -1
                    logger.info(f"Step {self.global_step}: GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")

                # Enhanced logging with wandb logger - fix step counting
                log_every_n_steps = self.config.get(
                    'wandb', {}).get('log_every_n_steps', 50)
                if self.wandb_logger and log_every_n_steps > 0 and batch_idx % log_every_n_steps == 0 and self.global_step > 0:
                    try:
                        # Log all metrics in a single consolidated call
                        log_quantization = self.global_step % (
                            log_every_n_steps * 10) == 0
                        memory_module = self.model.memory if hasattr(
                            self.model, 'memory') else None

                        # Only log if cross-modal similarity computation succeeded
                        log_outputs = outputs.copy() if isinstance(outputs, dict) else {}

                        self.wandb_logger.log_consolidated_metrics(
                            outputs=log_outputs,
                            epoch=epoch,
                            step=self.global_step,  # Now guaranteed to be > 0
                            lr=self.optimizer.param_groups[0]['lr'],
                            model=self.model,
                            memory_module=memory_module,
                            log_quantization=log_quantization
                        )

                    except Exception as e:
                        logger.warning(
                            f"Wandb logging failed at step {self.global_step}: {e}")
                        # Continue training without wandb logging for this step

                # Attention analysis (less frequent to avoid overhead)
                attention_log_steps = self.config.get(
                    'attention_analysis', {}).get('log_every_n_steps', 100)
                if (self.attention_analyzer and attention_log_steps > 0 and
                        self.global_step % attention_log_steps == 0):

                    try:
                        self.attention_analyzer.analyze_batch_attention(
                            outputs, batch['input_ids'], self.global_step
                        )
                    except Exception as e:
                        logger.warning(
                            f"Attention analysis failed at step {self.global_step}: {e}")

                # Attention evolution tracking (NEW!)
                track_attention_steps = self.config.get(
                    'track_attention_every_n_steps', 50)
                if (self.attention_evolution_tracker and track_attention_steps > 0 and
                        self.global_step % track_attention_steps == 0):

                    try:
                        # Extract cross-modal attention if available
                        cross_modal_attention = None
                        if 'cross_modal_attention' in outputs:
                            cross_modal_attention = outputs['cross_modal_attention']
                        elif hasattr(outputs, 'attentions') and outputs.attentions:
                            # Last layer attention
                            cross_modal_attention = outputs.attentions[-1]

                        if cross_modal_attention is not None:
                            # Get caption for first sample in batch
                            sample_caption = "Generated caption"  # TODO: Extract actual caption
                            if 'captions' in batch:
                                sample_caption = batch['captions'][0] if batch['captions'] else "No caption"

                            # Save attention evolution data
                            self.attention_evolution_tracker.save_epoch_attention(
                                epoch=epoch,
                                sample_id=f"step_{self.global_step}_sample_0",
                                caption=sample_caption,
                                # First sample
                                attention_weights=cross_modal_attention[0:1],
                                image_features=batch['vision_features'][0:1],
                                compressed_features=outputs.get(
                                    'vision_latent', None)
                            )

                            if self.global_step % 200 == 0 and self.global_step > 0:  # Less frequent logging, avoid zero
                                logger.info(
                                    f"🎯 Tracked attention evolution at step {self.global_step}")

                    except Exception as e:
                        logger.warning(
                            f"Attention evolution tracking failed at step {self.global_step}: {e}")

                self.global_step += 1

                # Step learning rate scheduler if step-based
                if self.scheduler and hasattr(self, 'scheduler_step_mode') and self.scheduler_step_mode == 'step':
                    self.scheduler.step()

                # Memory cleanup every 100 steps to prevent OOM
                if self.global_step > 0 and self.global_step % 100 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except Exception as e:
                logger.error(
                    f"Training batch {batch_idx} failed at step {self.global_step}: {e}")
                logger.error(f"Skipping batch and continuing training...")

                # Add detailed traceback for debugging
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")

                # Clear GPU cache after error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.global_step += 1
                continue

        # Average metrics over epoch with safety checks
        epoch_metrics['train_loss'] = np.mean(
            epoch_losses) if epoch_losses else float('inf')
        epoch_metrics['memory_usage_entropy'] = (
            epoch_metrics['memory_usage_entropy'] / len(train_loader)) if len(train_loader) > 0 else 0.0
        epoch_metrics['cross_modal_similarity'] = (
            epoch_metrics['cross_modal_similarity'] / len(train_loader)) if len(train_loader) > 0 else 0.0

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
                logger.info(
                    f"Validating on dataset {loader_idx + 1}/{len(val_loaders)}")

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
                                memory_entropy = self._compute_memory_entropy(
                                    outputs['memory_usage'])
                                if np.isfinite(memory_entropy):
                                    val_metrics['val_memory_entropy'] += memory_entropy
                            except Exception as e:
                                logger.warning(
                                    f"Memory entropy computation failed: {e}")

                        if outputs['text_features'] is not None and outputs['vision_latent'] is not None:
                            try:
                                cross_modal_sim = self._compute_cross_modal_similarity(
                                    outputs['text_features'], outputs['vision_latent']
                                )
                                if np.isfinite(cross_modal_sim):
                                    val_metrics['val_cross_modal_similarity'] += cross_modal_sim
                            except Exception as e:
                                logger.warning(
                                    f"Cross-modal similarity computation failed: {e}")

                    except Exception as e:
                        logger.warning(
                            f"Validation batch {batch_idx} in loader {loader_idx} failed: {e}")
                        continue

        # Calculate total number of batches across all loaders for averaging
        total_batches = sum(len(loader)
                            for loader in val_loaders) if val_loaders else 1

        # Average metrics with safety checks
        val_metrics['val_loss'] = np.mean(
            val_losses) if val_losses else float('inf')
        val_metrics['val_memory_entropy'] = (
            val_metrics['val_memory_entropy'] / total_batches) if total_batches > 0 else 0.0
        val_metrics['val_cross_modal_similarity'] = (
            val_metrics['val_cross_modal_similarity'] / total_batches) if total_batches > 0 else 0.0

        # Clear GPU cache and restore training mode
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.train()  # Restore training mode

        logger.info(
            f"Validation completed - Loss: {val_metrics['val_loss']:.4f}")

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

            # Handle dimension mismatch by projecting to smaller dimension
            if text_pooled.shape[-1] != vision_latent.shape[-1]:
                text_dim = text_pooled.shape[-1]
                vision_dim = vision_latent.shape[-1]

                if text_dim > vision_dim:
                    # Project text to vision dimension (take first N dimensions)
                    text_pooled = text_pooled[:, :vision_dim]
                    logger.debug(
                        f"Projected text features from {text_dim}D to {vision_dim}D")
                elif vision_dim > text_dim:
                    # Project vision to text dimension (take first N dimensions)
                    vision_latent = vision_latent[:, :text_dim]
                    logger.debug(
                        f"Projected vision features from {vision_dim}D to {text_dim}D")

            # Compute cosine similarity with numerical stability
            cos_sim = torch.cosine_similarity(
                text_pooled, vision_latent, dim=1)
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
            checkpoint_path = self.checkpoint_dir / \
                f'checkpoint_epoch_{epoch}.pt'
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
                self.scheduler.load_state_dict(
                    checkpoint['scheduler_state_dict'])

            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint['best_val_loss']

            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            logger.info(f"Resuming from epoch {self.current_epoch}")

            return self.current_epoch

        except Exception as e:
            logger.error(
                f"Failed to load checkpoint from {checkpoint_path}: {e}")
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

    def safe_gpu_operation(self, operation_name: str, operation_func):
        """Safely execute GPU operations with fallback handling"""
        try:
            return operation_func()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.error(f"GPU error in {operation_name}: {e}")
                logger.info("Attempting GPU memory cleanup...")

                # Aggressive memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Verify device consistency
                self.verify_device_consistency()

                # Retry once
                try:
                    return operation_func()
                except Exception as retry_e:
                    logger.error(f"Retry failed for {operation_name}: {retry_e}")
                    raise retry_e
            else:
                raise e

    def verify_device_consistency(self):
        """Verify model and optimizer are on correct device with improved stability"""
        try:
            # Check model device
            model_device = next(self.model.parameters()).device
            if model_device != self.device:
                logger.warning(f"Model moved from {self.device} to {model_device}. Moving back...")
                self.model.to(self.device)
                # Force model to stay on device with stronger pinning
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    # Force all model parameters to stay on the target device
                    for param in self.model.parameters():
                        if param.device != self.device:
                            param.data = param.data.to(self.device)
                    for buffer in self.model.buffers():
                        if buffer.device != self.device:
                            buffer.data = buffer.data.to(self.device)

            # More conservative optimizer state checking - only recreate if absolutely necessary
            if hasattr(self.optimizer, 'state') and self.optimizer.state:
                # Count how many states are on wrong device
                wrong_device_count = 0
                total_states = 0

                for param_id, state in self.optimizer.state.items():
                    if isinstance(state, dict):
                        for key, value in state.items():
                            if torch.is_tensor(value):
                                total_states += 1
                                if value.device.type != self.device.type:
                                    wrong_device_count += 1

                # Only recreate if significant portion of states are on wrong device
                wrong_device_ratio = wrong_device_count / max(total_states, 1)
                if wrong_device_ratio > 0.5 and wrong_device_count > 0:  # More than 50% of states on wrong device
                    logger.warning(f"Optimizer state moved to wrong device ({wrong_device_count}/{total_states} tensors). Recreating optimizer...")
                    self._recreate_optimizer_with_state_preservation()

        except Exception as e:
            logger.error(f"Device verification failed: {e}")
            # Minimal fallback - just ensure model is on correct device
            try:
                self.model.to(self.device)
            except Exception as fallback_e:
                logger.error(f"Fallback device move also failed: {fallback_e}")

    def _recreate_optimizer_with_state_preservation(self):
        """Recreate optimizer while preserving as much state as possible"""
        try:
            # Store current learning rate and other important state
            current_lr = self.optimizer.param_groups[0]['lr']
            current_step_count = getattr(self.optimizer, '_step_count', 0) if hasattr(self.optimizer, '_step_count') else 0

            # Store momentum and other state if available (for Adam/AdamW)
            preserved_state = {}
            if hasattr(self.optimizer, 'state') and self.optimizer.state:
                for param_id, state in self.optimizer.state.items():
                    if isinstance(state, dict):
                        # Try to preserve momentum terms on correct device
                        preserved_entry = {}
                        for key, value in state.items():
                            if torch.is_tensor(value):
                                try:
                                    preserved_entry[key] = value.to(self.device).clone()
                                except:
                                    # Skip if can't move to device
                                    pass
                            else:
                                preserved_entry[key] = value
                        if preserved_entry:
                            preserved_state[param_id] = preserved_entry

            # Recreate optimizer
            old_optimizer_type = self.config.get('optimizer', 'adamw').lower()
            self.setup_optimizer()

            # Restore learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            # Try to restore preserved state
            if preserved_state and hasattr(self.optimizer, 'state'):
                try:
                    # Only restore if the parameter structure matches
                    param_ids = list(self.optimizer.state_dict()['param_groups'][0]['params']);
                    for i, param_id in enumerate(param_ids):
                        if i < len(preserved_state):
                            old_param_id = list(preserved_state.keys())[i]
                            if old_param_id in preserved_state:
                                # Restore the state for this parameter
                                self.optimizer.state[param_id] = preserved_state[old_param_id]
                except Exception as restore_e:
                    logger.warning(f"Could not restore optimizer state: {restore_e}")

            logger.info(f"Optimizer recreated on {self.device} with LR={current_lr:.2e}")

        except Exception as e:
            logger.error(f"Optimizer recreation failed: {e}")
            # Fallback to basic setup
            self.setup_optimizer()

    def _force_model_device_consistency(self):
        """Aggressively force model to stay on target device"""
        if not torch.cuda.is_available():
            return

        try:
            # Force all model components to target device
            self.model.to(self.device)

            # Manually move all parameters and buffers
            for name, param in self.model.named_parameters():
                if param.device != self.device:
                    param.data = param.data.to(self.device)
                    if param.grad is not None:
                        param.grad.data = param.grad.data.to(self.device)

            for name, buffer in self.model.named_buffers():
                if buffer.device != self.device:
                    buffer.data = buffer.data.to(self.device)

            # Force CUDA synchronization
            torch.cuda.synchronize()

        except Exception as e:
            logger.error(f"Failed to force model device consistency: {e}")

    def _create_device_pinned_optimizer(self):
        """Create optimizer with device-pinned state"""
        try:
            # Store optimizer state before recreation
            old_state = None
            current_lr = self.config['training']['learning_rate']

            if hasattr(self, 'optimizer') and self.optimizer is not None:
                current_lr = self.optimizer.param_groups[0]['lr']
                try:
                    old_state = self.optimizer.state_dict()
                except:
                    old_state = None

            # Create new optimizer
            optimizer_type = self.config.get('optimizer', 'adamw').lower()

            if optimizer_type == 'adamw':
                self.optimizer = AdamW(
                    self.model.parameters(),
                    lr=current_lr,
                    weight_decay=self.config['training']['weight_decay'],
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
            elif optimizer_type == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=current_lr,
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
            else:
                # Fallback to AdamW
                self.optimizer = AdamW(
                    self.model.parameters(),
                    lr=current_lr,
                    weight_decay=self.config['training']['weight_decay'],
                    betas=(0.9, 0.999),
                    eps=1e-8
                )

            # Try to restore old state if available
            if old_state is not None:
                try:
                    self.optimizer.load_state_dict(old_state)
                    # Force optimizer state to correct device
                    for state in self.optimizer.state.values():
                        if isinstance(state, dict):
                            for key, value in state.items():
                                if torch.is_tensor(value):
                                    state[key] = value.to(self.device)
                except Exception as e:
                    logger.warning(f"Could not restore optimizer state: {e}")

            logger.info(f"Device-pinned optimizer created on {self.device}")

        except Exception as e:
            logger.error(f"Failed to create device-pinned optimizer: {e}")
            raise e

    def _safe_batch_to_device(self, batch):
        """Safely move batch to device with error handling"""
        try:
            device_batch = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    # Use pin_memory and non_blocking for faster transfers
                    if value.device != self.device:
                        device_batch[key] = value.to(self.device, non_blocking=True)
                    else:
                        device_batch[key] = value
                else:
                    device_batch[key] = value
            return device_batch
        except Exception as e:
            logger.error(f"Failed to move batch to device: {e}")
            # Fallback: try moving without non_blocking
            try:
                device_batch = {}
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        device_batch[key] = value.to(self.device)
                    else:
                        device_batch[key] = value
                return device_batch
            except Exception as fallback_e:
                logger.error(f"Fallback batch move also failed: {fallback_e}")
                raise fallback_e

    def _silent_device_check(self):
        """Silently check and fix device inconsistencies without warnings"""
        try:
            # Check model device silently
            model_device = next(self.model.parameters()).device
            if model_device != self.device:
                self._device_warnings_count += 1
                # Only log every 50 warnings to avoid spam
                if self._device_warnings_count % 50 == 1:
                    logger.warning(f"Device inconsistency detected ({self._device_warnings_count} times). Fixing silently...")

                # Force model back to correct device
                self._force_model_device_consistency()

                # If too many device switches, recreate optimizer
                if self._device_warnings_count > 10:
                    self._create_device_pinned_optimizer()
                    self._device_warnings_count = 0  # Reset counter

        except Exception as e:
            # Don't log device check failures - they create noise
            pass

    def train(self, max_samples: Optional[int] = None):
        """Main training loop"""
        logger.info("Starting BitMar training...")

        # Setup directories first
        self.setup_directories()

        # Setup logging systems before model setup
        self.setup_logging_systems()

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
                logger.info(
                    f"Scheduler stepped (epoch-based), new LR: {self.optimizer.param_groups[0]['lr']:.2e}")

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
                    self.attention_analyzer.create_attention_timeline_plot(
                        self.global_step)

                    # Save top attention heads
                    for attention_type in ['encoder', 'decoder', 'cross_modal']:
                        self.attention_analyzer.save_top_heads(
                            self.global_step, attention_type)

                except Exception as e:
                    logger.warning(
                        f"Attention visualization creation failed: {e}")

            # Generate attention evolution visualizations (NEW!)
            if (self.attention_evolution_tracker and
                    (epoch + 1) % self.config.get('save_attention_every_n_epochs', 1) == 0):

                logger.info(
                    "🎨 Generating attention evolution visualizations...")

                try:
                    # Create epoch comparison grids
                    if epoch > 0:  # Need at least 2 epochs
                        # Get sample IDs from this epoch
                        if epoch in self.attention_evolution_tracker.attention_history:
                            sample_ids = list(
                                self.attention_evolution_tracker.attention_history[epoch].keys())
                            if sample_ids:
                                self.attention_evolution_tracker.create_epoch_comparison_grid(
                                    sample_id=sample_ids[0],
                                    epochs=[epoch-1, epoch]
                                )

                    # Generate learning summary
                    if epoch >= 2:  # Need at least 3 epochs
                        self.attention_evolution_tracker.create_attention_learning_summary(
                            max_epochs=epoch)

                    # Create token evolution plots for common tokens
                    if epoch >= 3:  # Need several epochs for meaningful evolution
                        common_tokens = ['the', 'a', 'dog', 'cat', 'person']
                        for token_text in common_tokens:
                            try:
                                token_ids = self.attention_evolution_tracker.tokenizer.encode(
                                    token_text)
                                if token_ids:
                                    self.attention_evolution_tracker.create_token_evolution_plot(
                                        token_text=token_text,
                                        token_id=token_ids[0]
                                    )
                            except Exception as e:
                                continue  # Skip if token not found

                    logger.info(
                        f"✅ Attention evolution visualizations complete for epoch {epoch}")

                except Exception as e:
                    logger.warning(
                        f"Attention evolution visualization failed: {e}")

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
                        self.wandb_logger.create_quantization_plot(
                            self.model, self.global_step)

                    except Exception as e:
                        logger.warning(
                            f"Wandb visualization creation failed: {e}")

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
            final_report = self.attention_analyzer.generate_attention_report(
                self.global_step)
            logger.info(f"Final attention analysis: {final_report}")

            # Save final top heads
            for attention_type in ['encoder', 'decoder', 'cross_modal']:
                self.attention_analyzer.save_top_heads(
                    self.global_step, attention_type, k=20)

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
        default="configs/bitmar_ultra_tiny.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Override max epochs from config"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max epochs from config (alias for --max_epochs)"
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
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Override W&B project name"
    )
    parser.add_argument(
        "--track_attention_every_n_steps",
        type=int,
        default=50,
        help="Save attention evolution data every N steps"
    )
    parser.add_argument(
        "--save_attention_every_n_epochs",
        type=int,
        default=1,
        help="Generate attention visualizations every N epochs"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adamw8bit", "adam", "sgd", "rmsprop"],
        help="Optimizer to use for training"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    # Handle both --max_epochs and --epochs
    if args.max_epochs:
        config['training']['max_epochs'] = args.max_epochs
    elif args.epochs:
        config['training']['max_epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.wandb_project:
        config['wandb']['project'] = args.wandb_project

    # Add attention tracking config
    config['track_attention_every_n_steps'] = args.track_attention_every_n_steps
    config['save_attention_every_n_epochs'] = args.save_attention_every_n_epochs
    config['optimizer'] = args.optimizer

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
