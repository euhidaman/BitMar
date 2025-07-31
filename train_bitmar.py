"""
Training script for BitMar model with QFormer Cross-Modal Alignment
Handles multimodal training with episodic memory, attention analysis, and human-inspired learning
"""

# CodeCarbon for carbon footprint tracking
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("Warning: CodeCarbon not available. Install with: pip install codecarbon")

from src.attention_analysis import analyze_model_attention
from src.dataset import create_data_module, TextOnlyDataset, VisualOnlyDataset, extract_train_50M_if_needed
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
from torch.utils.data import DataLoader
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

# Import attention evolution tracker
try:
    # First try direct import
    from attention_evolution_tracker import AttentionEvolutionTracker
    ATTENTION_TRACKING_AVAILABLE = True
    print("✅ Attention evolution tracker imported successfully")
except ImportError as e:
    try:
        # Try importing from current directory with explicit path manipulation
        import sys
        import os
        from pathlib import Path

        # Get the directory where this script is located
        script_dir = Path(__file__).parent.absolute()

        # Add current directory to Python path if not already there
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))

        from attention_evolution_tracker import AttentionEvolutionTracker
        ATTENTION_TRACKING_AVAILABLE = True
        print("✅ Attention evolution tracker imported from current directory")
    except ImportError as e2:
        try:
            # Final fallback: check if file exists and provide detailed error
            tracker_file = Path(__file__).parent / "attention_evolution_tracker.py"
            if tracker_file.exists():
                print(f"⚠️  File exists at {tracker_file} but import failed:")
                print(f"   Original error: {e}")
                print(f"   Fallback error: {e2}")
            else:
                print(f"❌ File not found at {tracker_file}")

            ATTENTION_TRACKING_AVAILABLE = False
            print("Warning: attention_evolution_tracker not available - continuing without it")
        except Exception as e3:
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
    """BitMar model trainer with episodic memory, attention analysis, and human-inspired 3-stage learning"""

    def __init__(self, config, device: Optional[str] = None):
        """Initialize trainer with configuration

        Args:
            config: Either a string path to config file or a loaded config dictionary
            device: Optional device specification
        """
        print("🔧 BitMarTrainer.__init__() started...")

        # Handle both config path (string) and loaded config (dict)
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError("config must be either a string path or a dictionary")

        print("✅ Configuration loaded successfully")

        # Set device - prioritize user specification, then config, then auto-detect
        if device:
            self.device = torch.device(device)
        elif self.config.get('training', {}).get('device'):
            self.device = torch.device(self.config['training']['device'])
        else:
            # Force CUDA device index specification when available
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"🎯 Device selected: {self.device}")

        # Ensure CUDA is initialized if available - WITH TIMEOUT PROTECTION
        if self.device.type == 'cuda':
            try:
                print("🔄 Initializing CUDA...")

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

        # Initialize tracking variables
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self._last_model_device = None
        self._device_warnings_count = 0

        # Human-inspired training stages
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
        self.stage_metrics = {
            'stage_1_metrics': {'vision_loss': [], 'vision_accuracy': [], 'visual_clustering_score': []},
            'stage_2_metrics': {'cross_modal_similarity': [], 'alignment_accuracy': [], 'caption_bleu': []},
            'stage_3_metrics': {'text_perplexity': [], 'language_accuracy': [], 'text_loss': []}
        }

        print("🔄 Setting up carbon tracking...")
        # Initialize CodeCarbon tracker
        self.setup_carbon_tracking()
        print("✅ Carbon tracking setup completed")

        print("🔄 Setting up adaptive controller...")
        # Initialize adaptive training controller
        self.setup_adaptive_controller()
        print("✅ Adaptive controller setup completed")

        print("✅ BitMarTrainer.__init__() completed successfully")

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

        stage_metrics = {'vision_loss': [], 'vision_accuracy': [], 'visual_clustering_score': []}

        for epoch in range(stage_config['epochs']):
            logger.info(f"Stage 1, Epoch {epoch + 1}/{stage_config['epochs']}")

            epoch_vision_loss = 0.0
            epoch_batches = 0

            train_loader = self.data_module.train_dataloader()
            progress_bar = tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch + 1}")

            for batch_idx, batch in enumerate(progress_bar):
                batch = self._safe_batch_to_device(batch)

                # Stage 1: Only process vision features
                vision_features = batch['vision_features']

                # Forward through vision encoder
                encoded_vision = self.model.encode_vision(vision_features)

                # Vision reconstruction loss (learn good representations)
                vision_reconstruction_loss = nn.MSELoss()(encoded_vision, vision_features)

                # Vision clustering loss (encourage diverse representations)
                vision_clustering_loss = self._compute_vision_clustering_loss(encoded_vision)

                # Combined loss for Stage 1
                loss = (stage_config['loss_weights']['vision_reconstruction'] * vision_reconstruction_loss +
                       stage_config['loss_weights']['vision_clustering'] * vision_clustering_loss)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                if self.config['training']['gradient_clip_val'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                 self.config['training']['gradient_clip_val'])

                self.optimizer.step()

                # Track metrics
                epoch_vision_loss += loss.item()
                epoch_batches += 1

                progress_bar.set_postfix({
                    'vision_loss': f"{loss.item():.4f}",
                    'recon_loss': f"{vision_reconstruction_loss.item():.4f}",
                    'cluster_loss': f"{vision_clustering_loss.item():.4f}"
                })

                # Log to wandb
                if self.wandb_logger and batch_idx % 50 == 0:
                    wandb.log({
                        'Stage_1/Vision_Loss': loss.item(),
                        'Stage_1/Vision_Reconstruction_Loss': vision_reconstruction_loss.item(),
                        'Stage_1/Vision_Clustering_Loss': vision_clustering_loss.item(),
                        'Stage_1/Epoch': epoch,
                        'Stage_1/Learning_Rate': self.optimizer.param_groups[0]['lr'],
                        'stage': 1,
                        'step': self.global_step
                    }, step=self.global_step)

                self.global_step += 1

            # Epoch metrics
            avg_vision_loss = epoch_vision_loss / epoch_batches if epoch_batches > 0 else 0
            stage_metrics['vision_loss'].append(avg_vision_loss)

            logger.info(f"Stage 1, Epoch {epoch + 1} - Average Vision Loss: {avg_vision_loss:.4f}")

        self.stage_metrics['stage_1_metrics'] = stage_metrics
        logger.info("🍼 STAGE 1 COMPLETED: Visual Understanding")
        return {'avg_vision_loss': np.mean(stage_metrics['vision_loss'])}

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
            # Ensure features are similar after fusion
            text_to_fused = F.mse_loss(text_features, fused_features)
            vision_to_fused = F.mse_loss(vision_features, fused_features)

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

    def train(self, max_samples: Optional[int] = None):
        """Main training loop with human-inspired 3-stage learning and comprehensive tracking"""
        logger.info("🚀 Starting BitMar Human-Inspired 3-Stage Training...")

        # Setup directories first
        self.setup_directories()

        # Setup logging systems before model setup
        self.setup_logging_systems()

        # Setup model and data
        self.setup_model_and_data(max_samples=max_samples)

        # Start carbon emissions tracking
        self.start_carbon_tracking()

        try:
            # Log initial stage configuration to wandb
            if self.wandb_logger:
                stage_config_metrics = {}
                for stage_num, config in self.stage_configs.items():
                    stage_config_metrics[f'config/stage_{stage_num}_epochs'] = config['epochs']
                    stage_config_metrics[f'config/stage_{stage_num}_lr'] = config['learning_rate']
                    stage_config_metrics[f'config/stage_{stage_num}_focus'] = hash(config['focus']) % 1000  # Simple encoding
                wandb.log(stage_config_metrics)

            # === STAGE 1: VISUAL UNDERSTANDING ===
            logger.info("\n" + "="*60)
            logger.info("🍼 STARTING STAGE 1: VISUAL UNDERSTANDING")
            logger.info("Learning to see like babies - visual patterns only")
            logger.info("="*60)

            stage_1_results = self.train_stage_1_visual_understanding()

            # Log stage 1 completion to wandb
            if self.wandb_logger:
                wandb.log({
                    'Stage_Completion/Stage_1_Completed': 1,
                    'Stage_Completion/Stage_1_Final_Vision_Loss': stage_1_results['avg_vision_loss'],
                    'Stage_Progress/Current_Stage': 1,
                    'step': self.global_step
                }, step=self.global_step)

            # === STAGE 2: VISUAL-LANGUAGE GROUNDING ===
            logger.info("\n" + "="*60)
            logger.info("🔗 STARTING STAGE 2: VISUAL-LANGUAGE GROUNDING")
            logger.info("Connecting words to images like children learning 'apple'")
            logger.info("="*60)

            self.current_stage = 2
            stage_2_results = self.train_stage_2_visual_language_grounding()

            # Log stage 2 completion to wandb
            if self.wandb_logger:
                wandb.log({
                    'Stage_Completion/Stage_2_Completed': 1,
                    'Stage_Completion/Stage_2_Final_Similarity': stage_2_results['avg_cross_modal_similarity'],
                    'Stage_Completion/Stage_2_Final_Alignment': stage_2_results['avg_alignment_accuracy'],
                    'Stage_Progress/Current_Stage': 2,
                    'step': self.global_step
                }, step=self.global_step)

            # === STAGE 3: ABSTRACT LANGUAGE LEARNING ===
            logger.info("\n" + "="*60)
            logger.info("📚 STARTING STAGE 3: ABSTRACT LANGUAGE LEARNING")
            logger.info("Learning language patterns like reading books")
            logger.info("="*60)

            self.current_stage = 3
            stage_3_results = self.train_stage_3_abstract_language_learning()

            # Log stage 3 completion to wandb
            if self.wandb_logger:
                wandb.log({
                    'Stage_Completion/Stage_3_Completed': 1,
                    'Stage_Completion/Stage_3_Final_Loss': stage_3_results['avg_text_loss'],
                    'Stage_Completion/Stage_3_Final_Perplexity': stage_3_results['avg_perplexity'],
                    'Stage_Completion/Stage_3_Final_Accuracy': stage_3_results['avg_language_accuracy'],
                    'Stage_Progress/Current_Stage': 3,
                    'step': self.global_step
                }, step=self.global_step)

            # === FINAL COMPREHENSIVE EVALUATION ===
            logger.info("\n" + "="*60)
            logger.info("🎯 RUNNING FINAL EVALUATION")
            logger.info("="*60)

            final_val_metrics = self.validate_epoch(0)  # Final validation

            # Log comprehensive final results
            if self.wandb_logger:
                final_summary = {
                    'Final_Results/Stage_1_Vision_Loss': stage_1_results['avg_vision_loss'],
                    'Final_Results/Stage_2_Cross_Modal_Similarity': stage_2_results['avg_cross_modal_similarity'],
                    'Final_Results/Stage_2_Alignment_Accuracy': stage_2_results['avg_alignment_accuracy'],
                    'Final_Results/Stage_3_Text_Loss': stage_3_results['avg_text_loss'],
                    'Final_Results/Stage_3_Perplexity': stage_3_results['avg_perplexity'],
                    'Final_Results/Stage_3_Language_Accuracy': stage_3_results['avg_language_accuracy'],
                    'Final_Results/Final_Validation_Loss': final_val_metrics['val_loss'],
                    'Final_Results/Total_Training_Steps': self.global_step,
                    'Training_Completed': 1
                }
                wandb.log(final_summary, step=self.global_step)

            # Create final stage comparison visualizations
            self._create_stage_comparison_plots()

            logger.info("\n" + "="*60)
            logger.info("✅ HUMAN-INSPIRED TRAINING COMPLETED!")
            logger.info("="*60)
            logger.info(f"📊 Stage 1 (Visual): Vision Loss = {stage_1_results['avg_vision_loss']:.4f}")
            logger.info(f"🔗 Stage 2 (Grounding): Similarity = {stage_2_results['avg_cross_modal_similarity']:.4f}")
            logger.info(f"📚 Stage 3 (Language): Perplexity = {stage_3_results['avg_perplexity']:.2f}")
            logger.info(f"🎯 Final Validation Loss: {final_val_metrics['val_loss']:.4f}")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise e
        finally:
            # Stop carbon tracking and log final results
            total_emissions = self.stop_carbon_tracking()

            # Final analysis and cleanup
            logger.info("Training completed! Running final analysis...")

            # Save final checkpoint with stage information
            final_checkpoint = {
                'global_step': self.global_step,
                'current_stage': self.current_stage,
                'stage_metrics': self.stage_metrics,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config
            }

            final_checkpoint_path = self.checkpoint_dir / 'final_human_inspired_checkpoint.pt'
            torch.save(final_checkpoint, final_checkpoint_path)
            logger.info(f"Final checkpoint saved: {final_checkpoint_path}")

            # Close wandb logger
            if self.wandb_logger:
                self.wandb_logger.finish()

            logger.info("Human-inspired training completed!")
            if total_emissions > 0:
                logger.info(f"🌱 Final Carbon Footprint: {total_emissions:.6f} kg CO2 ({total_emissions * 1000:.3f} g CO2)")

            return total_emissions

    def _create_stage_comparison_plots(self):
        """Create visualizations comparing metrics across all 3 stages"""
        if not self.wandb_logger:
            return

        try:
            import matplotlib.pyplot as plt

            # Stage progression plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # Stage 1: Vision Loss
            if self.stage_metrics['stage_1_metrics']['vision_loss']:
                ax1.plot(self.stage_metrics['stage_1_metrics']['vision_loss'], 'b-', linewidth=2)
                ax1.set_title('Stage 1: Vision Learning')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Vision Loss')
                ax1.grid(True, alpha=0.3)

            # Stage 2: Cross-Modal Similarity
            if self.stage_metrics['stage_2_metrics']['cross_modal_similarity']:
                ax2.plot(self.stage_metrics['stage_2_metrics']['cross_modal_similarity'], 'g-', linewidth=2)
                ax2.set_title('Stage 2: Visual-Language Grounding')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Cross-Modal Similarity')
                ax2.grid(True, alpha=0.3)

            # Stage 3: Perplexity
            if self.stage_metrics['stage_3_metrics']['text_perplexity']:
                ax3.plot(self.stage_metrics['stage_3_metrics']['text_perplexity'], 'r-', linewidth=2)
                ax3.set_title('Stage 3: Language Learning')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Perplexity')
                ax3.grid(True, alpha=0.3)

            # Combined accuracy metrics
            stage_names = ['Visual', 'Grounding', 'Language']
            accuracies = [
                1.0 - (np.mean(self.stage_metrics['stage_1_metrics']['vision_loss']) if self.stage_metrics['stage_1_metrics']['vision_loss'] else 0),
                np.mean(self.stage_metrics['stage_2_metrics']['alignment_accuracy']) if self.stage_metrics['stage_2_metrics']['alignment_accuracy'] else 0,
                np.mean(self.stage_metrics['stage_3_metrics']['language_accuracy']) if self.stage_metrics['stage_3_metrics']['language_accuracy'] else 0
            ]

            bars = ax4.bar(stage_names, accuracies, color=['blue', 'green', 'red'], alpha=0.7)
            ax4.set_title('Final Performance by Stage')
            ax4.set_ylabel('Performance Score')
            ax4.set_ylim(0, 1.0)
            ax4.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            wandb.log({"Human_Inspired_Training/Stage_Comparison": wandb.Image(fig)}, step=self.global_step)
            plt.close(fig)

            logger.info("✅ Stage comparison plots created and logged to wandb")

        except Exception as e:
            logger.warning(f"Failed to create stage comparison plots: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BitMar model with episodic memory and attention analysis")

    # Required arguments
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")

    # Optional arguments
    parser.add_argument("--wandb_project", type=str, default="bitmar-training", help="Weights & Biases project name")
    parser.add_argument("--max_epochs", type=int, help="Maximum number of epochs to train (overrides config)")
    parser.add_argument("--track_attention_every_n_steps", type=int, default=1000, help="Track attention evolution every N steps")
    parser.add_argument("--save_attention_every_n_epochs", type=int, default=5, help="Save attention analysis every N epochs")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "adamw8bit", "sgd", "lion"], help="Optimizer to use (overrides config)")
    parser.add_argument("--device", type=str, help="Device to use for training (cuda, cpu)")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    try:
        print("🚀 Starting BitMar training initialization...")

        # Initialize trainer
        print("📋 Loading configuration and initializing trainer...")
        trainer = BitMarTrainer(args.config, device=args.device)
        print("✅ Trainer initialized successfully")

        # Override config values with command line arguments
        if args.max_epochs:
            print(f"🔧 Overriding max_epochs: {trainer.config['training']['max_epochs']} -> {args.max_epochs}")
            trainer.config['training']['max_epochs'] = args.max_epochs
        if args.optimizer:
            print(f"🔧 Overriding optimizer: {trainer.config['training'].get('optimizer', 'adamw')} -> {args.optimizer}")
            trainer.config['training']['optimizer'] = args.optimizer

        # Override wandb project name if provided
        if args.wandb_project:
            print(f"🔧 Setting wandb project: {args.wandb_project}")
            trainer.config['wandb']['project'] = args.wandb_project

        # Set tracking parameters as instance variables
        trainer.track_attention_every_n_steps = args.track_attention_every_n_steps
        trainer.save_attention_every_n_epochs = args.save_attention_every_n_epochs

        # Start training (wandb setup happens inside train() method)
        print("🎯 Starting training process...")
        logger.info("Starting BitMar training...")
        logger.info(f"Configuration: {args.config}")
        logger.info(f"Device: {trainer.device}")
        logger.info(f"Max epochs: {trainer.config['training']['max_epochs']}")
        logger.info(f"Wandb project: {trainer.config['wandb']['project']}")

        print("📊 Beginning human-inspired 3-stage training...")
        trainer.train()

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        logger.info("Training interrupted by user")
    except FileNotFoundError as e:
        print(f"❌ File not found error: {e}")
        logger.error(f"File not found: {e}")
        print("💡 Please check that all required files exist:")
        print(f"   - Config file: {args.config}")
        print("   - Data directory and files")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        logger.error(f"Training failed with error: {e}")
        import traceback
        print("\n🔍 Full error traceback:")
        traceback.print_exc()
        sys.exit(1)

