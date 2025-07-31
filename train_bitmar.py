"""
Training script for BitMar model with QFormer Cross-Modal Alignment
Handles multimodal training with episodic memory and human-inspired learning
"""

print("🚀 Starting train_bitmar.py script...")

# CodeCarbon for carbon footprint tracking
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
    print("✅ CodeCarbon imported successfully")
except ImportError:
    CODECARBON_AVAILABLE = False
    print("Warning: CodeCarbon not available. Install with: pip install codecarbon")

print("📦 Importing core modules...")
from src.dataset import create_data_module, TextOnlyDataset, VisualOnlyDataset, extract_train_50M_if_needed
print("✅ Dataset modules imported")
from src.model import create_bitmar_model, count_parameters
print("✅ Model modules imported")
from src.wandb_logger import BitMarWandbLogger
print("✅ Wandb logger imported")
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse
import logging
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import wandb
print("✅ All core imports completed")

# Try to import bitsandbytes for 8-bit optimizer
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Warning: bitsandbytes not available. Install with: pip install bitsandbytes")

# Try to import Lion optimizer
try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False
    print("Warning: Lion optimizer not available. Install with: pip install lion-pytorch")

# Import adaptive training controller
try:
    from src.adaptive_training_controller import AdaptiveTrainingController, compute_cross_modal_similarity
    ADAPTIVE_TRAINING_AVAILABLE = True
except ImportError:
    ADAPTIVE_TRAINING_AVAILABLE = False
    print("Warning: Adaptive training controller not available")

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Disable attention tracker to prevent hanging issues
ATTENTION_TRACKING_AVAILABLE = False
print("ℹ️  Attention tracker disabled to prevent training hang")

# Simple logging setup without file handler to prevent hanging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
print("✅ Logging setup completed")


class BitMarTrainer:
    """BitMar model trainer with episodic memory, attention analysis, and human-inspired 3-stage learning"""

    def __init__(self, config, device: Optional[str] = None):
        """Initialize trainer with configuration

        Args:
            config: Either a string path to config file or a loaded config dictionary
            device: Optional device specification
        """
        print("🔧 BitMarTrainer.__init__() started...")
        sys.stdout.flush()

        # Handle both config path (string) and loaded config (dict)
        if isinstance(config, str):
            print(f"📄 Loading config from file: {config}")
            sys.stdout.flush()
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, dict):
            print("📄 Using provided config dictionary")
            sys.stdout.flush()
            self.config = config
        else:
            raise TypeError("config must be either a string path or a dictionary")

        print("✅ Configuration loaded successfully")
        sys.stdout.flush()

        # Set device - prioritize user specification, then config, then auto-detect
        print("🎯 Setting up device configuration...")
        sys.stdout.flush()
        if device:
            self.device = torch.device(device)
        elif self.config.get('training', {}).get('device'):
            self.device = torch.device(self.config['training']['device'])
        else:
            # Force CUDA device index specification when available
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"🎯 Device selected: {self.device}")
        sys.stdout.flush()

        # Ensure CUDA is initialized if available - WITH TIMEOUT PROTECTION
        if self.device.type == 'cuda':
            try:
                print("🔄 Initializing CUDA...")
                sys.stdout.flush()

                # Add timeout protection for CUDA operations
                import signal
                import threading
                import time

                def cuda_init_with_timeout():
                    """Initialize CUDA with timeout protection"""
                    try:
                        # Test CUDA availability first
                        if not torch.cuda.is_available():
                            raise RuntimeError("CUDA not available")

                        # Initialize CUDA (this can hang)
                        torch.cuda.init()

                        # Get device name (this can also hang)
                        device_name = torch.cuda.get_device_name(self.device)
                        return device_name
                    except Exception as e:
                        raise e

                # Use threading for timeout control on Windows
                result = [None]
                exception = [None]

                def cuda_worker():
                    try:
                        result[0] = cuda_init_with_timeout()
                    except Exception as e:
                        exception[0] = e

                thread = threading.Thread(target=cuda_worker)
                thread.daemon = True
                thread.start()
                thread.join(timeout=10.0)  # 10 second timeout

                if thread.is_alive():
                    logger.warning("CUDA initialization timed out (10s), continuing with CPU")
                    self.device = torch.device("cpu")
                elif exception[0]:
                    logger.warning(f"CUDA initialization failed: {exception[0]}")
                    logger.warning("Falling back to CPU")
                    self.device = torch.device("cpu")
                elif result[0]:
                    logger.info(f"Using CUDA device: {result[0]}")
                    logger.info("CUDA initialized, model will be moved to GPU explicitly")
                    print("✅ CUDA initialized successfully")
                    sys.stdout.flush()
                else:
                    logger.warning("CUDA initialization returned no result, falling back to CPU")
                    self.device = torch.device("cpu")

            except Exception as e:
                logger.warning(f"CUDA initialization failed: {e}")
                logger.warning("Falling back to CPU")
                self.device = torch.device("cpu")
        else:
            logger.warning("CUDA not available, using CPU. Training will be slow.")

        print(f"✅ Final device: {self.device}")
        sys.stdout.flush()

        # Initialize tracking variables
        print("🔢 Initializing tracking variables...")
        sys.stdout.flush()
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self._last_model_device = None
        self._device_warnings_count = 0

        # Human-inspired training stages
        print("📚 Setting up human-inspired training stages...")
        sys.stdout.flush()
        self.current_stage = 1
        self.stage_configs = {
            1: {  # Visual Understanding Stage
                'name': 'Visual Understanding',
                'description': 'Learn visual representations like babies learning to see',
                'epochs': self.config.get('training', {}).get('visual_stage_epochs', 3),
                'learning_rate': self.config.get('training', {}).get('visual_stage_lr', 0.001),
                'focus': 'vision_encoder_only',
                'loss_weights': {'vision_reconstruction': 1.0, 'vision_clustering': 0.5}
            },
            2: {  # Visual-Language Grounding Stage
                'name': 'Visual-Language Grounding',
                'description': 'Connect words to visual concepts like children learning "apple"',
                'epochs': self.config.get('training', {}).get('grounding_stage_epochs', 5),
                'learning_rate': self.config.get('training', {}).get('grounding_stage_lr', 0.0005),
                'focus': 'multimodal_alignment',
                'loss_weights': {
                    'cross_modal_contrastive': 1.0,
                    'visual_language_alignment': 0.8,
                    'caption_generation': 0.3
                }
            },
            3: {  # Abstract Language Learning Stage
                'name': 'Abstract Language Learning',
                'description': 'Learn pure language patterns like reading books',
                'epochs': self.config.get('training', {}).get('language_stage_epochs', 7),
                'learning_rate': self.config.get('training', {}).get('language_stage_lr', 0.0003),
                'focus': 'language_modeling',
                'loss_weights': {'language_modeling': 1.0, 'text_understanding': 0.5}
            }
        }

        # Stage tracking for wandb
        print("📊 Setting up stage metrics tracking...")
        sys.stdout.flush()
        self.stage_metrics = {
            'stage_1_metrics': {'vision_loss': [], 'vision_accuracy': [], 'visual_clustering_score': []},
            'stage_2_metrics': {'cross_modal_similarity': [], 'alignment_accuracy': [], 'caption_bleu': []},
            'stage_3_metrics': {'text_perplexity': [], 'language_accuracy': [], 'text_loss': []}
        }

        print("🔄 Setting up carbon tracking...")
        sys.stdout.flush()
        # Initialize CodeCarbon tracker
        self.setup_carbon_tracking()
        print("✅ Carbon tracking setup completed")
        sys.stdout.flush()

        print("🔄 Setting up adaptive controller...")
        sys.stdout.flush()
        # Initialize adaptive training controller
        self.setup_adaptive_controller()
        print("✅ Adaptive controller setup completed")
        sys.stdout.flush()

        print("✅ BitMarTrainer.__init__() completed successfully")
        sys.stdout.flush()

    def setup_carbon_tracking(self):
        """Initialize CodeCarbon emissions tracker for remote machine usage"""
        try:
            # Skip carbon tracking if not available to prevent hanging
            if not CODECARBON_AVAILABLE:
                logger.info("CodeCarbon not available, skipping carbon tracking")
                self.carbon_tracker = None
                self.carbon_tracking_enabled = False
                return

            # Create carbon logs directory
            carbon_logs_dir = Path("./carbon_logs")
            carbon_logs_dir.mkdir(exist_ok=True)

            # Configure tracker for remote/cloud environment with timeout protection
            tracker_config = {
                "project_name": f"BitMar-BabyLM-Training-{self.config.get('training', {}).get('max_epochs', 'unknown')}epochs",
                "experiment_id": f"bitmar-{self.device.type}-training",
                "output_dir": str(carbon_logs_dir),
                "output_file": "emissions.csv",
                "log_level": "INFO",
                "save_to_file": True,
                "save_to_api": False,  # Disabled for remote machines by default
                "measure_power_secs": 15,  # Measure every 15 seconds
                "tracking_mode": "machine"  # Track entire machine
            }

            # Auto-detect cloud provider for remote machines
            if torch.cuda.is_available():
                try:
                    tracker_config["gpu_ids"] = [0]  # Track primary GPU
                    logger.info(f"Carbon tracking configured for GPU: {torch.cuda.get_device_name(0)}")
                except Exception as e:
                    logger.warning(f"Could not configure GPU for carbon tracking: {e}")

            # Initialize the tracker but don't start it yet - with timeout protection
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("CodeCarbon initialization timed out")

            # Set a 10-second timeout for initialization
            if hasattr(signal, 'SIGALRM'):  # Unix-like systems
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)

            try:
                self.carbon_tracker = EmissionsTracker(**tracker_config)
                self.carbon_tracking_enabled = True
                logger.info("🌱 CodeCarbon emissions tracker initialized for remote training")
                logger.info(f"Carbon logs will be saved to: {carbon_logs_dir}/emissions.csv")
            finally:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Cancel the alarm

        except (TimeoutError, Exception) as e:
            logger.warning(f"Failed to initialize CodeCarbon tracker (timeout or error): {e}")
            logger.warning("Training will continue without carbon tracking")
            self.carbon_tracker = None
            self.carbon_tracking_enabled = False

    def start_carbon_tracking(self):
        """Start carbon emissions tracking"""
        if self.carbon_tracking_enabled and self.carbon_tracker:
            try:
                self.carbon_tracker.start()
                logger.info("🌱 Started carbon emissions tracking")
            except Exception as e:
                logger.warning(f"Failed to start carbon tracking: {e}")
                self.carbon_tracking_enabled = False

    def stop_carbon_tracking(self):
        """Stop carbon emissions tracking and log results"""
        if self.carbon_tracking_enabled and self.carbon_tracker:
            try:
                emissions = self.carbon_tracker.stop()

                # Log emissions data
                if emissions:
                    logger.info(f"🌱 Training Carbon Footprint Summary:")
                    logger.info(f"   Total CO2 emissions: {emissions:.6f} kg CO2")
                    logger.info(f"   Equivalent to: {emissions * 1000:.3f} g CO2")

                    # Log to wandb if available
                    if self.wandb_logger:
                        try:
                            carbon_metrics = {
                                "carbon_emissions_kg": emissions,
                                "carbon_emissions_g": emissions * 1000,
                                "training_duration_hours": (self.global_step * self.config.get('data', {}).get('batch_size', 1)) / 3600,  # Rough estimate
                                "emissions_per_epoch": emissions / max(self.current_epoch, 1)
                            }

                            # Log final carbon metrics
                            wandb.log(carbon_metrics)
                            logger.info("🌱 Carbon metrics logged to wandb")

                        except Exception as e:
                            logger.warning(f"Failed to log carbon metrics to wandb: {e}")

                    return emissions
                else:
                    logger.warning("No emissions data collected")
                    return 0.0

            except Exception as e:
                logger.warning(f"Failed to stop carbon tracking: {e}")
                return 0.0

        return 0.0

    def log_carbon_metrics_periodic(self, epoch: int, step: int):
        """Log periodic carbon metrics during training"""
        if not self.carbon_tracking_enabled or not self.carbon_tracker:
            return

        try:
            # Get current emissions (this doesn't stop tracking)
            current_emissions = getattr(self.carbon_tracker, '_total_energy', 0.0)

            # Log periodic metrics every 10 epochs or every 1000 steps
            if (epoch > 0 and epoch % 10 == 0) or (step > 0 and step % 1000 == 0):
                if self.wandb_logger and current_emissions > 0:
                    try:
                        periodic_metrics = {
                            "carbon_emissions_current_kg": current_emissions,
                            "carbon_emissions_per_step": current_emissions / max(step, 1),
                            "carbon_emissions_per_epoch": current_emissions / max(epoch, 1)
                        }
                        wandb.log(periodic_metrics)
                        logger.info(f"🌱 Periodic carbon update - Current emissions: {current_emissions:.6f} kg CO2")
                    except Exception as e:
                        logger.debug(f"Failed to log periodic carbon metrics: {e}")

        except Exception as e:
            logger.debug(f"Failed to get periodic carbon metrics: {e}")

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

        # Initialize attention analyzer - DISABLED to prevent performance issues
        self.attention_analyzer = None
        logger.info("Attention analyzer disabled to prevent performance issues during training")

        # Attention evolution tracker disabled to prevent training hang
        self.attention_evolution_tracker = None
        logger.info("Attention evolution tracker disabled to prevent training hang")

        # Create data module
        self.data_module = create_data_module(self.config['data'])
        self.data_module.setup(max_samples=max_samples)

        # Setup optimizer and scheduler
        self.setup_optimizer()

    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        optimizer_type = self.config['training'].get('optimizer', 'adamw').lower()

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
        elif optimizer_type == 'lion' and LION_AVAILABLE:
            # Use Lion optimizer if available and requested
            optimizer_params = self.config['training'].get('optimizer_config', {})
            self.optimizer = Lion(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                betas=optimizer_params.get('betas', [0.9, 0.99]),
                weight_decay=self.config['training']['weight_decay']
            )
            logger.info(f"✅ Using Lion optimizer with betas={optimizer_params.get('betas', [0.9, 0.99])}")
        elif optimizer_type == 'lion' and not LION_AVAILABLE:
            logger.error("❌ Lion optimizer requested but not available!")
            logger.error("📦 Install with: pip install lion-pytorch")
            logger.warning("🔄 Falling back to AdamW optimizer...")
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
            logger.info(f"Using AdamW optimizer (Lion fallback)")
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        # Learning rate scheduler with warmup and cosine restarts
        scheduler_type = self.config['training'].get('scheduler', 'cosine')
        warmup_steps = self.config['training'].get('warmup_steps', 1000)

        if scheduler_type == 'cosine_with_restarts':
            # Calculate total training steps for proper cosine annealing with restarts
            train_loader = self.data_module.train_dataloader()
            steps_per_epoch = len(train_loader)
            total_steps = steps_per_epoch * \
                self.config['training']['max_epochs']

            scheduler_config = self.config['training'].get('scheduler_config', {})
            T_0 = scheduler_config.get('T_0', 1000)
            T_mult = scheduler_config.get('T_mult', 2)
            eta_min_ratio = scheduler_config.get('eta_min_ratio', 0.1)
            eta_min = self.config['training']['learning_rate'] * eta_min_ratio

            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min
            )
            self.scheduler_step_mode = 'step'

            # Add warmup scheduler wrapper if warmup_steps > 0
            if warmup_steps > 0:
                from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
                try:
                    self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                        self.optimizer,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=total_steps,
                        num_cycles=max(1, total_steps // T_0)
                    )
                    logger.info(f"Cosine with restarts + warmup scheduler: {warmup_steps} warmup steps, T_0={T_0}, T_mult={T_mult}")
                except ImportError:
                    logger.warning("Transformers warmup scheduler not available, using basic cosine with restarts")
                    logger.info(f"Cosine with restarts scheduler: T_0={T_0}, T_mult={T_mult}, eta_min={eta_min}")
            else:
                logger.info(f"Cosine with restarts scheduler: T_0={T_0}, T_mult={T_mult}, eta_min={eta_min}")

        elif scheduler_type == 'cosine':
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

        # Dynamic logging based on actual optimizer type
        actual_optimizer_name = type(self.optimizer).__name__
        if hasattr(self.optimizer, '__module__') and 'lion' in self.optimizer.__module__.lower():
            optimizer_display_name = "LION"
        elif actual_optimizer_name == 'AdamW':
            optimizer_display_name = "ADAMW"
        elif actual_optimizer_name == 'Adam':
            optimizer_display_name = "ADAM"
        elif actual_optimizer_name == 'SGD':
            optimizer_display_name = "SGD"
        elif actual_optimizer_name == 'RMSprop':
            optimizer_display_name = "RMSPROP"
        elif 'AdamW8bit' in actual_optimizer_name:
            optimizer_display_name = "ADAMW8BIT"
        else:
            optimizer_display_name = actual_optimizer_name.upper()

        logger.info(
            f"Optimizer: {optimizer_display_name} with LR={self.config['training']['learning_rate']}")
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
                        labels=batch['labels'],
                        step=self.global_step,  # Add step parameter for loss balancing
                        adaptive_controller=self.adaptive_controller  # Pass adaptive controller
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
                            labels=batch['labels'],
                            step=self.global_step,  # Add step parameter for retry too
                            adaptive_controller=self.adaptive_controller  # Pass adaptive controller
                        )
                        loss = outputs['loss']
                    else:
                        raise e

                # Check for invalid loss
                if not torch.isfinite(loss):
                    logger.warning(
                        f"Invalid loss at step {self.global_step}: {loss.item()}")
                    continue

                # ADAPTIVE TRAINING MONITORING (NEW!)
                if self.adaptive_controller is not None and outputs['text_features'] is not None and outputs['vision_latent'] is not None:
                    try:
                        # Compute cross-modal similarity for adaptive controller
                        similarity_score = compute_cross_modal_similarity(
                            outputs['text_features'],
                            outputs['vision_latent']
                        )

                        # Update adaptive controller and get intervention decisions
                        controller_info = self.adaptive_controller.update_similarity(
                            similarity_score,
                            self.global_step
                        )

                        # Log intervention if triggered
                        if controller_info['intervention_info']['intervention_triggered']:
                            intervention = controller_info['intervention_info']
                            logger.info(f"🚨 STEP {self.global_step}: ADAPTIVE INTERVENTION TRIGGERED!")
                            logger.info(f"   Type: {intervention['intervention_type']}")
                            logger.info(f"   Reason: {intervention['trigger_reason']}")
                            logger.info(f"   Text Encoder Frozen: {controller_info['text_encoder_frozen']}")
                            logger.info(f"   Vision Encoder Frozen: {controller_info['vision_encoder_frozen']}")
                            logger.info(f"   Loss Multiplier: {controller_info['cross_modal_weight_multiplier']:.2f}")

                            # Log to wandb if available
                            if self.wandb_logger:
                                intervention_metrics = {
                                    'adaptive/intervention_triggered': 1,
                                    'adaptive/intervention_type_encoded': hash(intervention['intervention_type']) % 100,  # Simple encoding for wandb
                                    'adaptive/similarity_drop': intervention['similarity_drop'],
                                    'adaptive/text_encoder_frozen': int(controller_info['text_encoder_frozen']),
                                    'adaptive/vision_encoder_frozen': int(controller_info['vision_encoder_frozen']),
                                    'adaptive/cross_modal_weight_multiplier': controller_info['cross_modal_weight_multiplier'],
                                    'adaptive/similarity_ema': controller_info['similarity_ema'],
                                    'adaptive/similarity_current': similarity_score
                                }
                                wandb.log(intervention_metrics, step=self.global_step)

                        # Always log similarity metrics (less frequently)
                        if self.global_step % 50 == 0 and self.wandb_logger:
                            adaptive_metrics = {
                                'adaptive/cross_modal_similarity': similarity_score,
                                'adaptive/similarity_ema': controller_info['similarity_ema'],
                                'adaptive/text_frozen_status': int(controller_info['text_encoder_frozen']),
                                'adaptive/vision_frozen_status': int(controller_info['vision_encoder_frozen']),
                                'adaptive/loss_multiplier': controller_info['cross_modal_weight_multiplier']
                            }
                            wandb.log(adaptive_metrics, step=self.global_step)

                        # Update best similarity for checkpointing
                        if similarity_score > self.best_similarity:
                            self.best_similarity = similarity_score
                            # Save best similarity checkpoint
                            self.save_checkpoint(epoch, is_best=False, suffix='best_similarity')

                    except Exception as e:
                        logger.warning(f"Adaptive controller update failed at step {self.global_step}: {e}")

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

                # All attention analysis disabled to prevent performance issues
                # (Removed attention_analyzer.analyze_batch_attention call)

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
        """Validate for one epoch with improved numerical stability"""
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

                        # Forward pass with gradient checkpointing disabled for validation
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            vision_features=batch['vision_features'],
                            labels=batch['labels']
                        )

                        # Safely extract loss with numerical stability checks
                        if outputs['loss'] is not None:
                            loss_value = outputs['loss'].item()
                            # Check for numerical stability
                            if torch.isfinite(outputs['loss']) and not (torch.isnan(outputs['loss']) or torch.isinf(outputs['loss'])):
                                # Clamp extreme loss values to prevent numerical instability
                                loss_value = max(0.0, min(loss_value, 100.0))  # Clamp between 0 and 100
                                val_losses.append(loss_value)
                            else:
                                logger.warning(f"Non-finite loss detected in validation batch {batch_idx}: {loss_value}")
                                # Skip this batch but continue validation
                                continue

                        # Compute additional metrics with enhanced safety checks
                        if outputs.get('memory_usage') is not None:
                            try:
                                memory_entropy = self._compute_memory_entropy(
                                    outputs['memory_usage'])
                                if np.isfinite(memory_entropy) and memory_entropy >= 0:
                                    val_metrics['val_memory_entropy'] += memory_entropy
                            except Exception as e:
                                logger.warning(
                                    f"Memory entropy computation failed: {e}")

                        if outputs.get('text_features') is not None and outputs.get('vision_latent') is not None:
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
                        # Add detailed error logging for debugging
                        import traceback
                        logger.warning(f"Validation error traceback: {traceback.format_exc()}")
                        continue

        # Calculate total number of batches across all loaders for averaging
        total_batches = sum(len(loader)
                            for loader in val_loaders) if val_loaders else 1

        # Average metrics with enhanced safety checks and fallback values
        if val_losses:
            val_metrics['val_loss'] = float(np.mean(val_losses))
            # Additional sanity check on the mean
            if not np.isfinite(val_metrics['val_loss']) or val_metrics['val_loss'] < 0:
                logger.warning(f"Invalid mean validation loss: {val_metrics['val_loss']}, using fallback")
                val_metrics['val_loss'] = 10.0  # Reasonable fallback value
        else:
            logger.warning("No valid validation losses collected, using fallback value")
            val_metrics['val_loss'] = 10.0  # Use a reasonable fallback instead of inf

        val_metrics['val_memory_entropy'] = (
            val_metrics['val_memory_entropy'] / total_batches) if total_batches > 0 else 0.0
        val_metrics['val_cross_modal_similarity'] = (
            val_metrics['val_cross_modal_similarity'] / total_batches) if total_batches > 0 else 0.0

        # Final validation of all metrics
        for key, value in val_metrics.items():
            if not np.isfinite(value):
                logger.warning(f"Non-finite metric detected: {key}={value}, setting to 0")
                val_metrics[key] = 0.0 if 'similarity' in key or 'entropy' in key else 10.0

        # Clear GPU cache and restore training mode
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.train()  # Restore training mode

        logger.info(
            f"Validation completed - Loss: {val_metrics['val_loss']:.4f}")

        return val_metrics

    def _force_model_device_consistency(self):
        """Force all model parameters to be on the correct device"""
        target_device = self.device

        # Move model to device if not already there
        self.model.to(target_device)

        # Ensure all parameters are on the correct device
        for name, param in self.model.named_parameters():
            if param.device != target_device:
                logger.warning(f"Moving parameter {name} from {param.device} to {target_device}")
                param.data = param.data.to(target_device)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(target_device)

        # Ensure all buffers are on the correct device
        for name, buffer in self.model.named_buffers():
            if buffer.device != target_device:
                logger.warning(f"Moving buffer {name} from {buffer.device} to {target_device}")
                buffer.data = buffer.data.to(target_device)

    def _silent_device_check(self):
        """Silently check device consistency without excessive logging"""
        try:
            # Check model device
            model_device = next(self.model.parameters()).device
            if model_device != self.device:
                self._device_warnings_count += 1
                if self._device_warnings_count <= 3:  # Only warn first 3 times
                    logger.warning(f"Model device mismatch: {self.device}, got {model_device}")
                    self._force_model_device_consistency()
        except Exception as e:
            logger.debug(f"Device check failed silently: {e}")

    def _create_device_pinned_optimizer(self):
        """Recreate optimizer with device-pinned parameters"""
        try:
            # Force model to correct device first
            self._force_model_device_consistency()

            # Recreate optimizer with current parameters
            optimizer_type = self.config['training'].get('optimizer', 'adamw').lower()

            if optimizer_type == 'adamw':
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config['training']['learning_rate'],
                    weight_decay=self.config['training']['weight_decay'],
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
            elif optimizer_type == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.config['training']['learning_rate'],
                    betas=(0.9, 0.999),
                    eps=1e-8
                )

            logger.info(f"Recreated optimizer on device {self.device}")

        except Exception as e:
            logger.error(f"Failed to recreate device-pinned optimizer: {e}")

    def _safe_batch_to_device(self, batch):
        """Safely move batch tensors to the specified device

        Args:
            batch: Dictionary containing batch data with potential tensors

        Returns:
            batch: Dictionary with tensors moved to self.device
        """
        try:
            # Move each tensor in the batch to the device
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(self.device)
            return batch
        except Exception as e:
            logger.error(f"Error moving batch to device {self.device}: {e}")
            # Return original batch if device transfer fails
            return batch

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

    def save_checkpoint(self, epoch: int, is_best: bool = False, suffix: str = ''):
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

    def setup_adaptive_controller(self):
        """Initialize adaptive training controller if enabled"""
        if not ADAPTIVE_TRAINING_AVAILABLE:
            logger.info("Adaptive training controller not available")
            self.adaptive_controller = None
            return

        # Check if adaptive training is enabled in config
        model_config = self.config.get('model', {})
        adaptive_config = self.config.get('adaptive_training', {})

        if not model_config.get('enable_adaptive_training', False):
            logger.info("Adaptive training disabled in config")
            self.adaptive_controller = None
            return

        # Create adaptive logs directory
        adaptive_logs_dir = Path("./logs/adaptive_training")
        adaptive_logs_dir.mkdir(parents=True, exist_ok=True)

        self.adaptive_controller = AdaptiveTrainingController(
            similarity_window_size=adaptive_config.get('similarity_window_size', 100),
            drop_threshold=adaptive_config.get('drop_threshold', 0.15),
            min_steps_between_interventions=adaptive_config.get('min_steps_between_interventions', 1000),
            freeze_duration_steps=adaptive_config.get('freeze_duration_steps', 2000),
            loss_rebalance_factor=adaptive_config.get('loss_rebalance_factor', 2.5),
            similarity_smoothing_alpha=adaptive_config.get('similarity_smoothing_alpha', 0.1),
            save_dir=str(adaptive_logs_dir)
        )

        logger.info("🤖 Adaptive training controller initialized!")
        logger.info(f"   Drop threshold: {adaptive_config.get('drop_threshold', 0.15)}")
        logger.info(f"   Window size: {adaptive_config.get('similarity_window_size', 100)} steps")
        logger.info(f"   Intervention cooldown: {adaptive_config.get('min_steps_between_interventions', 1000)} steps")
        logger.info(f"   Logs saved to: {adaptive_logs_dir}")

        self.best_similarity = 0.0  # Track best similarity for checkpointing

    def freeze_all_except_vision(self):
        """Freeze all model components except vision encoder for Stage 1"""
        # Freeze text encoder
        for param in self.model.text_encoder.parameters():
            param.requires_grad = False

        # Freeze text decoder
        for param in self.model.text_decoder.parameters():
            param.requires_grad = False

        # Freeze fusion module
        for param in self.model.fusion.parameters():
            param.requires_grad = False

        # Freeze memory module
        for param in self.model.memory.parameters():
            param.requires_grad = False

        # Freeze all projection layers except vision
        for name, param in self.model.named_parameters():
            if any(proj in name for proj in ['text_to_episode', 'memory_to_decoder', 'decoder_input_proj']):
                param.requires_grad = False

        # Keep vision encoder unfrozen
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = True

        logger.info("🍼 Stage 1: Only vision encoder is trainable")

    def freeze_all_except_multimodal(self):
        """Freeze text decoder, keep vision, text encoder, and fusion trainable for Stage 2"""
        # Freeze text decoder (no generation yet)
        for param in self.model.text_decoder.parameters():
            param.requires_grad = False

        # Keep vision encoder trainable
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = True

        # Keep text encoder trainable
        for param in self.model.text_encoder.parameters():
            param.requires_grad = True

        # Keep fusion module trainable (most important for alignment)
        for param in self.model.fusion.parameters():
            param.requires_grad = True

        # Keep memory trainable for multimodal episodes
        for param in self.model.memory.parameters():
            param.requires_grad = True

        # Keep relevant projection layers trainable
        for name, param in self.model.named_parameters():
            if any(proj in name for proj in ['text_to_episode', 'memory_to_decoder']):
                param.requires_grad = True
            elif 'decoder_input_proj' in name:
                param.requires_grad = False  # Not needed yet

        logger.info("🔗 Stage 2: Vision, text encoder, fusion, and memory are trainable")

    def unfreeze_all_components(self):
        """Unfreeze all model components for Stage 3"""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("📚 Stage 3: All components are trainable")

    def train_stage_1_visual_understanding(self) -> Dict[str, float]:
        """Stage 1: Visual Understanding - Learn visual representations like babies"""
        logger.info("🍼 STAGE 1 STARTING: Visual Understanding (like babies learning to see)")

        # Configure for visual-only learning
        self.freeze_all_except_vision()
        stage_config = self.stage_configs[1]

        # Update learning rate for this stage
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = stage_config['learning_rate']

        stage_metrics = {'vision_loss': [], 'vision_consistency': [], 'visual_diversity_score': []}

        for epoch in range(stage_config['epochs']):
            logger.info(f"Stage 1, Epoch {epoch + 1}/{stage_config['epochs']}")

            epoch_vision_loss = 0.0
            epoch_consistency = 0.0
            epoch_diversity = 0.0
            epoch_batches = 0

            train_loader = self.data_module.train_dataloader()
            progress_bar = tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch + 1}")

            for batch_idx, batch in enumerate(progress_bar):
                batch = self._safe_batch_to_device(batch)

                # Stage 1: Focus on learning good vision representations
                vision_features = batch['vision_features']  # [batch_size, 768]

                # Forward through vision encoder
                encoded_vision = self.model.encode_vision(vision_features)  # [batch_size, 32]

                # Vision consistency loss - encourage stable encoding
                if encoded_vision.size(0) > 1:
                    # Pairwise consistency within batch
                    pairwise_dists = torch.cdist(encoded_vision, encoded_vision, p=2)
                    # Remove diagonal (self-distances)
                    mask = ~torch.eye(encoded_vision.size(0), dtype=torch.bool, device=encoded_vision.device)
                    avg_distance = pairwise_dists[mask].mean()

                    # We want some distance but not too much (encourage meaningful clustering)
                    target_distance = 2.0  # Reasonable target for normalized features
                    vision_consistency_loss = F.mse_loss(avg_distance, torch.tensor(target_distance, device=encoded_vision.device))
                else:
                    vision_consistency_loss = torch.tensor(0.0, device=encoded_vision.device)

                # Vision diversity loss - encourage diverse representations
                vision_diversity_loss = self._compute_vision_diversity_loss(encoded_vision)

                # Feature quality loss - encourage features to use full dynamic range
                feature_range_loss = self._compute_feature_range_loss(encoded_vision)

                # Combined loss for Stage 1
                loss = (0.4 * vision_consistency_loss +
                       0.4 * vision_diversity_loss +
                       0.2 * feature_range_loss)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                if self.config['training']['gradient_clip_val'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                 self.config['training']['gradient_clip_val'])

                self.optimizer.step()

                # Track metrics
                epoch_vision_loss += loss.item()
                epoch_consistency += vision_consistency_loss.item()
                epoch_diversity += vision_diversity_loss.item()
                epoch_batches += 1

                progress_bar.set_postfix({
                    'total_loss': f"{loss.item():.4f}",
                    'consistency': f"{vision_consistency_loss.item():.4f}",
                    'diversity': f"{vision_diversity_loss.item():.4f}"
                })

                # Proper wandb logging for Stage 1
                if self.wandb_logger and batch_idx % 50 == 0:
                    stage_1_metrics = {
                        'Stage_1/Total_Loss': loss.item(),
                        'Stage_1/Vision_Consistency_Loss': vision_consistency_loss.item(),
                        'Stage_1/Vision_Diversity_Loss': vision_diversity_loss.item(),
                        'Stage_1/Feature_Range_Loss': feature_range_loss.item(),
                        'Stage_1/Epoch': epoch + 1,
                        'Stage_1/Learning_Rate': self.optimizer.param_groups[0]['lr'],
                        'Stage_1/Encoded_Feature_Mean': encoded_vision.mean().item(),
                        'Stage_1/Encoded_Feature_Std': encoded_vision.std().item(),
                        'Training/Current_Stage': 1,
                        'Training/Stage_Progress': (epoch + 1) / stage_config['epochs'],
                        'step': self.global_step
                    }
                    wandb.log(stage_1_metrics, step=self.global_step)

                self.global_step += 1

            # Epoch metrics
            avg_vision_loss = epoch_vision_loss / epoch_batches if epoch_batches > 0 else 0
            avg_consistency = epoch_consistency / epoch_batches if epoch_batches > 0 else 0
            avg_diversity = epoch_diversity / epoch_batches if epoch_batches > 0 else 0

            stage_metrics['vision_loss'].append(avg_vision_loss)
            stage_metrics['vision_consistency'].append(avg_consistency)
            stage_metrics['visual_diversity_score'].append(avg_diversity)

            # Log epoch summary to wandb
            if self.wandb_logger:
                epoch_summary = {
                    'Stage_1/Epoch_Avg_Loss': avg_vision_loss,
                    'Stage_1/Epoch_Avg_Consistency': avg_consistency,
                    'Stage_1/Epoch_Avg_Diversity': avg_diversity,
                    'Stage_1/Epoch_Number': epoch + 1,
                    'Training/Current_Stage': 1,
                    'step': self.global_step
                }
                wandb.log(epoch_summary, step=self.global_step)

            logger.info(f"Stage 1, Epoch {epoch + 1} - Loss: {avg_vision_loss:.4f}, Consistency: {avg_consistency:.4f}, Diversity: {avg_diversity:.4f}")

        self.stage_metrics['stage_1_metrics'] = stage_metrics
        logger.info("🍼 STAGE 1 COMPLETED: Visual Understanding")

        return {
            'avg_loss': np.mean(stage_metrics['vision_loss']),
            'avg_consistency': np.mean(stage_metrics['vision_consistency']),
            'avg_diversity': np.mean(stage_metrics['visual_diversity_score'])
        }

    def train_stage_2_visual_language_grounding(self) -> Dict[str, float]:
        """Stage 2: Visual-Language Grounding - Connect words to visual concepts"""
        logger.info("🔗 STAGE 2 STARTING: Visual-Language Grounding (connecting words to images)")

        # Configure for multimodal alignment
        self.freeze_all_except_multimodal()
        stage_config = self.stage_configs[2]

        # Update learning rate for this stage
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = stage_config['learning_rate']

        stage_metrics = {'cross_modal_similarity': [], 'alignment_accuracy': [], 'caption_bleu': []}

        for epoch in range(stage_config['epochs']):
            logger.info(f"Stage 2, Epoch {epoch + 1}/{stage_config['epochs']}")

            epoch_cross_modal_loss = 0.0
            epoch_alignment_loss = 0.0
            epoch_similarity = 0.0
            epoch_batches = 0

            train_loader = self.data_module.train_dataloader()
            progress_bar = tqdm(train_loader, desc=f"Stage 2 - Epoch {epoch + 1}")

            for batch_idx, batch in enumerate(progress_bar):
                batch = self._safe_batch_to_device(batch)

                # Stage 2: Process both text and vision for alignment
                text_features, _ = self.model.encode_text(batch['input_ids'], batch['attention_mask'])
                vision_latent = self.model.encode_vision(batch['vision_features'])

                # Cross-modal fusion (learning alignment)
                fused_features, cross_attention = self.model.fusion(text_features, vision_latent)

                # Cross-modal contrastive loss (CLIP-style)
                text_pooled = text_features.mean(dim=1)
                cross_modal_loss = self.model.compute_cross_modal_contrastive_loss(
                    text_pooled, vision_latent, temperature=0.07
                )

                # Visual-language alignment loss (encourage good fusion)
                alignment_loss = self._compute_alignment_loss(text_pooled, vision_latent, fused_features.mean(dim=1))

                # Combined loss for Stage 2
                loss = (stage_config['loss_weights']['cross_modal_contrastive'] * cross_modal_loss +
                       stage_config['loss_weights']['visual_language_alignment'] * alignment_loss)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                if self.config['training']['gradient_clip_val'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                 self.config['training']['gradient_clip_val'])

                self.optimizer.step()

                # Track metrics
                epoch_cross_modal_loss += cross_modal_loss.item()
                epoch_alignment_loss += alignment_loss.item()

                # Compute cross-modal similarity
                similarity = self._compute_cross_modal_similarity(text_features, vision_latent)
                epoch_similarity += similarity
                epoch_batches += 1

                progress_bar.set_postfix({
                    'total_loss': f"{loss.item():.4f}",
                    'cross_modal': f"{cross_modal_loss.item():.4f}",
                    'alignment': f"{alignment_loss.item():.4f}",
                    'similarity': f"{similarity:.4f}"
                })

                # Log to wandb
                if self.wandb_logger and batch_idx % 50 == 0:
                    wandb.log({
                        'Stage_2/Total_Loss': loss.item(),
                        'Stage_2/Cross_Modal_Loss': cross_modal_loss.item(),
                        'Stage_2/Alignment_Loss': alignment_loss.item(),
                        'Stage_2/Cross_Modal_Similarity': similarity,
                        'Stage_2/Epoch': epoch,
                        'Stage_2/Learning_Rate': self.optimizer.param_groups[0]['lr'],
                        'stage': 2,
                        'step': self.global_step
                    }, step=self.global_step)

                self.global_step += 1

            # Epoch metrics
            avg_cross_modal_loss = epoch_cross_modal_loss / epoch_batches if epoch_batches > 0 else 0
            avg_alignment_loss = epoch_alignment_loss / epoch_batches if epoch_batches > 0 else 0
            avg_similarity = epoch_similarity / epoch_batches if epoch_batches > 0 else 0

            stage_metrics['cross_modal_similarity'].append(avg_similarity)
            stage_metrics['alignment_accuracy'].append(1.0 - avg_alignment_loss)  # Proxy for accuracy

            logger.info(f"Stage 2, Epoch {epoch + 1} - Cross-Modal Similarity: {avg_similarity:.4f}, Alignment Loss: {avg_alignment_loss:.4f}")

        self.stage_metrics['stage_2_metrics'] = stage_metrics
        logger.info("🔗 STAGE 2 COMPLETED: Visual-Language Grounding")
        return {
            'avg_cross_modal_similarity': np.mean(stage_metrics['cross_modal_similarity']),
            'avg_alignment_accuracy': np.mean(stage_metrics['alignment_accuracy'])
        }

    def train_stage_3_abstract_language_learning(self) -> Dict[str, float]:
        """Stage 3: Abstract Language Learning - Learn pure language patterns"""
        logger.info("📚 STAGE 3 STARTING: Abstract Language Learning (like reading books)")

        # Configure for full language modeling
        self.unfreeze_all_components()
        stage_config = self.stage_configs[3]

        # Update learning rate for this stage
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = stage_config['learning_rate']

        stage_metrics = {'text_perplexity': [], 'language_accuracy': [], 'text_loss': []}

        for epoch in range(stage_config['epochs']):
            logger.info(f"Stage 3, Epoch {epoch + 1}/{stage_config['epochs']}")

            epoch_text_loss = 0.0
            epoch_perplexity = 0.0
            epoch_batches = 0

            train_loader = self.data_module.train_dataloader()
            progress_bar = tqdm(train_loader, desc=f"Stage 3 - Epoch {epoch + 1}")

            for batch_idx, batch in enumerate(progress_bar):
                batch = self._safe_batch_to_device(batch)

                # Stage 3: Full multimodal learning with emphasis on language
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    vision_features=batch['vision_features'],
                    labels=batch['labels'],
                    step=self.global_step
                )

                # Primary language modeling loss
                language_loss = outputs['loss']

                # Text understanding loss (additional objective)
                text_understanding_loss = self._compute_text_understanding_loss(
                    outputs['text_features'], outputs['logits'], batch['labels']
                )

                # Combined loss for Stage 3
                loss = (stage_config['loss_weights']['language_modeling'] * language_loss +
                       stage_config['loss_weights']['text_understanding'] * text_understanding_loss)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                if self.config['training']['gradient_clip_val'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                 self.config['training']['gradient_clip_val'])

                self.optimizer.step()

                # Track metrics
                epoch_text_loss += language_loss.item()
                perplexity = torch.exp(language_loss).item() if language_loss.item() < 10 else 100.0  # Cap perplexity
                epoch_perplexity += perplexity
                epoch_batches += 1

                progress_bar.set_postfix({
                    'total_loss': f"{loss.item():.4f}",
                    'text_loss': f"{language_loss.item():.4f}",
                    'perplexity': f"{perplexity:.2f}",
                    'understanding': f"{text_understanding_loss.item():.4f}"
                })

                # Log to wandb
                if self.wandb_logger and batch_idx % 50 == 0:
                    wandb.log({
                        'Stage_3/Total_Loss': loss.item(),
                        'Stage_3/Language_Loss': language_loss.item(),
                        'Stage_3/Text_Understanding_Loss': text_understanding_loss.item(),
                        'Stage_3/Perplexity': perplexity,
                        'Stage_3/Epoch': epoch,
                        'Stage_3/Learning_Rate': self.optimizer.param_groups[0]['lr'],
                        'stage': 3,
                        'step': self.global_step
                    }, step=self.global_step)

                    # Also log comprehensive metrics using existing wandb logger
                    if batch_idx % 100 == 0:  # Less frequent comprehensive logging
                        try:
                            self.wandb_logger.log_consolidated_metrics(
                                outputs=outputs,
                                epoch=epoch,
                                step=self.global_step,
                                lr=self.optimizer.param_groups[0]['lr'],
                                model=self.model,
                                memory_module=self.model.memory if hasattr(self.model, 'memory') else None,
                                log_quantization=(batch_idx % 500 == 0)
                            )
                        except Exception as e:
                            logger.debug(f"Comprehensive wandb logging failed: {e}")

                self.global_step += 1

            # Epoch metrics
            avg_text_loss = epoch_text_loss / epoch_batches if epoch_batches > 0 else 0
            avg_perplexity = epoch_perplexity / epoch_batches if epoch_batches > 0 else 0

            stage_metrics['text_loss'].append(avg_text_loss)
            stage_metrics['text_perplexity'].append(avg_perplexity)
            stage_metrics['language_accuracy'].append(max(0, 1.0 - avg_text_loss / 10.0))  # Proxy accuracy

            logger.info(f"Stage 3, Epoch {epoch + 1} - Text Loss: {avg_text_loss:.4f}, Perplexity: {avg_perplexity:.2f}")

        self.stage_metrics['stage_3_metrics'] = stage_metrics
        logger.info("📚 STAGE 3 COMPLETED: Abstract Language Learning")
        return {
            'avg_text_loss': np.mean(stage_metrics['text_loss']),
            'avg_perplexity': np.mean(stage_metrics['text_perplexity']),
            'avg_language_accuracy': np.mean(stage_metrics['language_accuracy'])
        }

    def _compute_vision_clustering_loss(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Compute clustering loss to encourage diverse visual representations"""
        try:
            # Simple clustering loss using feature diversity
            batch_size = vision_features.size(0)
            if batch_size < 2:
                return torch.tensor(0.0, device=vision_features.device)

            # Compute pairwise distances
            pairwise_dists = torch.cdist(vision_features, vision_features, p=2)

            # Encourage features to be separated (maximize minimum distance)
            min_dists = torch.min(pairwise_dists + torch.eye(batch_size, device=vision_features.device) * 1e6, dim=1)[0]
            clustering_loss = -torch.mean(min_dists)  # Negative to maximize distances

            return clustering_loss
        except Exception as e:
            logger.warning(f"Vision clustering loss computation failed: {e}")
            return torch.tensor(0.0, device=vision_features.device)

    def _compute_alignment_loss(self, text_features: torch.Tensor,
                               vision_features: torch.Tensor,
                               fused_features: torch.Tensor) -> torch.Tensor:
        """Compute alignment loss between modalities and their fusion"""
        try:
            # Handle dimension mismatches by projecting to the smallest common dimension
            text_dim = text_features.shape[-1]
            vision_dim = vision_features.shape[-1]
            fused_dim = fused_features.shape[-1]

            # Find the minimum dimension to project all features to
            min_dim = min(text_dim, vision_dim, fused_dim)

            # Project all features to the same dimension
            if text_dim > min_dim:
                text_proj = text_features[..., :min_dim]
            else:
                text_proj = text_features

            if vision_dim > min_dim:
                vision_proj = vision_features[..., :min_dim]
            else:
                vision_proj = vision_features

            if fused_dim > min_dim:
                fused_proj = fused_features[..., :min_dim]
            else:
                fused_proj = fused_features

            # Ensure features are similar after fusion
            text_to_fused = F.mse_loss(text_proj, fused_proj)
            vision_to_fused = F.mse_loss(vision_proj, fused_proj)

            # Balance both modalities
            alignment_loss = 0.5 * text_to_fused + 0.5 * vision_to_fused
            return alignment_loss
        except Exception as e:
            logger.warning(f"Alignment loss computation failed: {e}")
            return torch.tensor(0.0, device=text_features.device)

    def _compute_text_understanding_loss(self, text_features: torch.Tensor,
                                       logits: torch.Tensor,
                                       labels: torch.Tensor) -> torch.Tensor:
        """Compute additional text understanding loss beyond standard language modeling"""
        try:
            # Feature consistency loss - encourage stable text representations
            if text_features.size(0) > 1:
                # Compute feature stability across batch
                feature_std = torch.std(text_features, dim=0).mean()
                stability_loss = feature_std  # Encourage stable features
            else:
                stability_loss = torch.tensor(0.0, device=text_features.device)

            return stability_loss
        except Exception as e:
            logger.warning(f"Text understanding loss computation failed: {e}")
            return torch.tensor(0.0, device=text_features.device)

    def _compute_vision_diversity_loss(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Compute diversity loss to encourage varied visual representations"""
        try:
            batch_size = vision_features.size(0)
            if batch_size < 2:
                return torch.tensor(0.0, device=vision_features.device)

            # Compute pairwise distances between all vision features
            pairwise_dists = torch.cdist(vision_features, vision_features, p=2)

            # Remove diagonal (self-distances) and get lower triangular part
            mask = torch.triu(torch.ones_like(pairwise_dists, dtype=torch.bool), diagonal=1)
            distances = pairwise_dists[mask]

            # Encourage diversity by penalizing small distances (too similar features)
            min_distance_threshold = 1.0  # Minimum desired distance between features
            diversity_loss = torch.relu(min_distance_threshold - distances).mean()

            return diversity_loss
        except Exception as e:
            logger.warning(f"Vision diversity loss computation failed: {e}")
            return torch.tensor(0.0, device=vision_features.device)

    def _compute_feature_range_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Compute feature range loss to encourage full utilization of feature space"""
        try:
            # Encourage features to use the full dynamic range
            feature_std = torch.std(features, dim=0)  # Standard deviation per feature dimension
            feature_mean_std = feature_std.mean()  # Average std across all dimensions

            # We want reasonable standard deviation (not too small, not too large)
            target_std = 1.0  # Target standard deviation
            range_loss = F.mse_loss(feature_mean_std, torch.tensor(target_std, device=features.device))

            return range_loss
        except Exception as e:
            logger.warning(f"Feature range loss computation failed: {e}")
            return torch.tensor(0.0, device=features.device)

    def train(self):
        """Main training method that orchestrates the 3-stage human-inspired training process"""
        logger.info("Starting BitMar training...")
        logger.info(f"Configuration: {self.config}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Max epochs: {self.config['training']['max_epochs']}")
        logger.info(f"Wandb project: {self.config['wandb']['project']}")

        # Human-inspired 3-stage training process
        for stage in range(1, 4):
            logger.info(f"🚀 Starting Stage {stage}: {self.stage_configs[stage]['name']}")

            # Call the appropriate training method for each stage
            if stage == 1:
                stage_results = self.train_stage_1_visual_understanding()
            elif stage == 2:
                stage_results = self.train_stage_2_visual_language_grounding()
            elif stage == 3:
                stage_results = self.train_stage_3_abstract_language_learning()

            # Log stage results to wandb
            if self.wandb_logger:
                wandb.log({
                    f"Stage_{stage}/Avg_Loss": stage_results.get('avg_loss', 0),
                    f"Stage_{stage}/Avg_Consistency": stage_results.get('avg_consistency', 0),
                    f"Stage_{stage}/Avg_Diversity": stage_results.get('avg_diversity', 0),
                    f"Stage_{stage}/Avg_Similarity": stage_results.get('avg_cross_modal_similarity', 0),
                    f"Stage_{stage}/Avg_Alignment_Accuracy": stage_results.get('avg_alignment_accuracy', 0),
                    f"Stage_{stage}/Avg_Text_Loss": stage_results.get('avg_text_loss', 0),
                    f"Stage_{stage}/Avg_Perplexity": stage_results.get('avg_perplexity', 0),
                    'step': self.global_step
                }, step=self.global_step)

            logger.info(f"✅ Stage {stage} completed: {self.stage_configs[stage]['name']}")

            # Save checkpoint at the end of each stage
            self.save_checkpoint(epoch=self.current_epoch, is_best=False, suffix=f'stage_{stage}')

        logger.info("🎉 Training completed successfully!")
        logger.info("Model is now ready for evaluation or inference.")


def main():
    """Main training function with command line interface"""
    print("🎬 Starting main() function...")
    
    parser = argparse.ArgumentParser(description="BitMar Training with Human-Inspired Learning")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--wandb_project", type=str, default="bitmar-training", help="W&B project name")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override max epochs from config")
    parser.add_argument("--optimizer", type=str, default=None, help="Override optimizer from config")
    parser.add_argument("--device", type=str, default=None, help="Force specific device (cuda:0, cpu, etc.)")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit training samples for debugging")
    
    # Legacy arguments - kept for compatibility but ignored
    parser.add_argument("--track_attention_every_n_steps", type=int, default=50000, help="[IGNORED] Attention tracking disabled")
    parser.add_argument("--save_attention_every_n_epochs", type=int, default=1, help="[IGNORED] Attention tracking disabled")

    args = parser.parse_args()
    print(f"✅ Arguments parsed: {args}")
    
    # Warn about ignored arguments
    if hasattr(args, 'track_attention_every_n_steps') and args.track_attention_every_n_steps != 50000:
        print("⚠️  --track_attention_every_n_steps is ignored (attention tracking disabled)")
    if hasattr(args, 'save_attention_every_n_epochs') and args.save_attention_every_n_epochs != 1:
        print("⚠️  --save_attention_every_n_epochs is ignored (attention tracking disabled)")

    try:
        # Load and override config
        print(f"📄 Loading config from: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override config with command line args
        if args.max_epochs is not None:
            config['training']['max_epochs'] = args.max_epochs
            print(f"🔧 Overriding max_epochs: {args.max_epochs}")
        
        if args.optimizer is not None:
            config['training']['optimizer'] = args.optimizer.lower()
            print(f"🔧 Overriding optimizer: {args.optimizer}")
        
        if args.wandb_project:
            config['wandb']['project'] = args.wandb_project
            print(f"📊 W&B project: {args.wandb_project}")
        
        print("✅ Configuration loaded and overridden successfully")
        
        # Initialize trainer
        print("🏗️ Initializing BitMar trainer...")
        trainer = BitMarTrainer(config, device=args.device)
        print("✅ Trainer initialized successfully")
        
        # Setup directories and logging
        print("📁 Setting up directories...")
        trainer.setup_directories()
        print("✅ Directories setup completed")
        
        print("📊 Setting up logging systems...")
        trainer.setup_logging_systems()
        print("✅ Logging systems setup completed")
        
        # Setup model and data
        print("🤖 Setting up model and data...")
        trainer.setup_model_and_data(max_samples=args.max_samples)
        print("✅ Model and data setup completed")
        
        # Resume from checkpoint if specified
        if args.resume_from:
            print(f"🔄 Resuming from checkpoint: {args.resume_from}")
            start_epoch = trainer.load_checkpoint(args.resume_from)
            print(f"✅ Resumed from epoch {start_epoch}")
        
        # Start carbon tracking
        print("🌱 Starting carbon tracking...")
        trainer.start_carbon_tracking()
        print("✅ Carbon tracking started")
        
        # Start training
        print("🚀 Starting BitMar training...")
        logger.info("=" * 50)
        logger.info("🚀 BITMAR TRAINING STARTED")
        logger.info("=" * 50)
        
        trainer.train()
        
        logger.info("=" * 50)
        logger.info("🎉 BITMAR TRAINING COMPLETED")
        logger.info("=" * 50)
        
        # Stop carbon tracking
        print("🌱 Stopping carbon tracking...")
        emissions = trainer.stop_carbon_tracking()
        if emissions > 0:
            print(f"🌍 Total CO2 emissions: {emissions:.6f} kg")
        
        print("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        logger.info("Training interrupted by user")
        if 'trainer' in locals():
            trainer.stop_carbon_tracking()
    
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        if 'trainer' in locals():
            trainer.stop_carbon_tracking()
        raise


if __name__ == "__main__":
    print("🎯 Script called directly, running main()...")
    main()

