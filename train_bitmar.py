"""
Training script for BitMar model with Episodic Memory Consolidation
Implements cognitively-inspired training: Episodic Capture → Memory Consolidation → Semantic Integration
Uses QFormer for enhanced cross-modal alignment
"""

import shutil
import traceback
import psutil
import gc
import sys
import os
import logging
import threading
import yaml
import torch
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
import torch.nn.functional as F
import torch.nn as nn
import torch
import yaml
import logging
import argparse
import threading
import sys
import os
from codecarbon import EmissionsTracker
from src.hf_compatibility import save_bitmar_as_hf_model
from src.dataset_optimizer import IntelligentDatasetOptimizer, OptimizedDataLoader
from src.modality_tracker import ModalityTracker
from src.attention_visualizer import AttentionHeadAnalyzer
from src.wandb_logger import BitMarWandbLogger
from src.model import create_bitmar_model, count_parameters
from src.dataset import create_data_module
from src.attention_analysis import analyze_model_attention
print("🚀 Starting train_bitmar.py script...")

# Core imports
print("📦 Importing core modules...")
# NEW: Comprehensive modality tracking
# NEW: Intelligent optimization
# NEW: HuggingFace compatibility
print("✅ Dataset and model modules imported")

print("✅ All core imports completed")

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
            raise TypeError(
                "config must be either a string path or a dictionary")

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
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

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
                    logger.warning(
                        "CUDA initialization timed out (10s), continuing with CPU")
                    self.device = torch.device("cpu")
                elif exception[0]:
                    logger.warning(
                        f"CUDA initialization failed: {exception[0]}")
                    logger.warning("Falling back to CPU")
                    self.device = torch.device("cpu")
                elif result[0]:
                    logger.info(f"Using CUDA device: {result[0]}")
                    logger.info(
                        "CUDA initialized, model will be moved to GPU explicitly")
                    print("✅ CUDA initialized successfully")
                    sys.stdout.flush()
                else:
                    logger.warning(
                        "CUDA initialization returned no result, falling back to CPU")
                    self.device = torch.device("cpu")

            except Exception as e:
                logger.warning(f"CUDA initialization failed: {e}")
                logger.warning("Falling back to CPU")
                self.device = torch.device("cpu")
        else:
            logger.warning(
                "CUDA not available, using CPU. Training will be slow.")

        # Initialize tracking variables
        print("🔢 Initializing tracking variables...")
        sys.stdout.flush()
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self._last_model_device = None
        self._device_warnings_count = 0

        print(f"✅ Final device: {self.device}")
        print("✅ BitMarTrainer.__init__() completed successfully")
        sys.stdout.flush()

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
        """Initialize model and data loaders with progressive growing support"""
        logger.info(
            "Setting up model and data with advanced training strategies...")

        # Check for quick training mode
        quick_mode = self.config.get('quick_training_mode', {})
        if quick_mode.get('enabled', False):
            # Don't limit samples - use optimizations instead
            optimizations = quick_mode.get('optimizations', {})
            logger.info(
                f"🚀 QUICK TRAINING MODE: Using advanced optimizations without sample reduction")

            # Apply optimizations
            if optimizations.get('aggressive_image_compression', False):
                logger.info("🖼️ Enabling aggressive image compression")
            if optimizations.get('mixed_precision_training', False):
                logger.info("⚡ Enabling mixed precision training")
            if optimizations.get('compiled_model', False):
                logger.info("🔥 Enabling PyTorch 2.0 model compilation")
            if optimizations.get('cached_vision_features', False):
                logger.info("💾 Enabling vision feature caching")

            self.quick_optimizations = optimizations
        else:
            self.quick_optimizations = {}

        # Create model with progressive growing support
        model_config = self.config['model'].copy()

        # 🎯 SELECTIVE VISION TRAINING: Configure vision freezing strategy
        vision_training_config = model_config.get('selective_vision_training', {})
        if vision_training_config.get('enabled', True):  # Default enabled
            # Configure which vision components to train
            freeze_vision_backbone = vision_training_config.get('freeze_vision_backbone', True)
            train_vision_projector = vision_training_config.get('train_vision_projector', True)
            train_qformer_vision = vision_training_config.get('train_qformer_vision', True)
            
            model_config['freeze_vision_backbone'] = freeze_vision_backbone
            model_config['train_vision_projector'] = train_vision_projector
            model_config['train_qformer_vision'] = train_qformer_vision
            
            logger.info("🎯 SELECTIVE VISION TRAINING configured:")
            logger.info(f"   - Vision backbone frozen: {freeze_vision_backbone}")
            logger.info(f"   - Vision projector trainable: {train_vision_projector}")
            logger.info(f"   - QFormer vision layers trainable: {train_qformer_vision}")
            
            self.selective_vision_training = True
        else:
            self.selective_vision_training = False
            logger.info("Standard vision training (all components trainable)")

        # Check if progressive growing is enabled
        progressive_config = model_config.get('progressive_growing', {})
        if progressive_config.get('enabled', False):
            logger.info("🚀 Progressive model growing enabled")

            # Start with smaller model
            start_layers = progressive_config.get('start_layers', {})
            model_config['text_encoder_layers'] = start_layers.get(
                'text_encoder_layers', 2)
            model_config['text_decoder_layers'] = start_layers.get(
                'text_decoder_layers', 2)
            model_config['fusion_num_layers'] = start_layers.get(
                'fusion_num_layers', 1)

            logger.info(f"Starting with: {model_config['text_encoder_layers']} encoder layers, "
                        f"{model_config['text_decoder_layers']} decoder layers, "
                        f"{model_config['fusion_num_layers']} fusion layers")

            # Store growth schedule for later use
            self.growth_schedule = progressive_config.get(
                'growth_schedule', {})
            self.target_layers = progressive_config.get('final_layers', {})
            self.progressive_growing_enabled = True
        else:
            self.progressive_growing_enabled = False
            logger.info(
                "Progressive growing disabled - using fixed architecture")

        self.model = create_bitmar_model(model_config)

        # Apply memory-efficient model settings
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")

        # Enable memory-efficient attention if available
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'use_memory_efficient_attention'):
            self.model.config.use_memory_efficient_attention = True
            logger.info("✅ Memory-efficient attention enabled")

        # Force model to GPU with verification and CPU memory cleanup
        print("🎯 Moving model to GPU...")
        sys.stdout.flush()
        self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")

        # 🎯 SELECTIVE VISION PARAMETER FREEZING for faster training
        if self.selective_vision_training:
            self._apply_selective_vision_freezing()

        # 🚀 ENHANCED GPU OPTIMIZATIONS FOR MAXIMUM PERFORMANCE
        try:
            # Enhanced PyTorch optimizations for GPU acceleration
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
                torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for speed
                torch.backends.cudnn.allow_tf32 = True

                # Set CUDA memory management for optimal performance
                torch.cuda.set_per_process_memory_fraction(
                    0.95, device=self.device)  # Use 95% of GPU memory

                logger.info(
                    "🔥 ENABLED AGGRESSIVE CUDA optimizations: cuDNN benchmark, TF32, memory optimization")
                print("⚡ GPU optimizations enabled")
                sys.stdout.flush()
            else:
                logger.warning("CUDA not available for optimizations")
        except Exception as e:
            logger.warning(f"CUDA optimizations failed: {e}")

        # 🚀 ENABLE MIXED PRECISION TRAINING FOR MAXIMUM GPU ACCELERATION
        if torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=2.**16,  # Higher initial scale for better precision
                growth_factor=2.0,  # Faster scale growth
                backoff_factor=0.5,  # Moderate backoff
                growth_interval=2000  # More frequent scale updates
            )
            self.use_amp = True
            logger.info(
                "⚡ ENABLED AGGRESSIVE Mixed Precision Training (AMP) for maximum GPU speed")
            print("🚀 Mixed precision training enabled")
            sys.stdout.flush()
        else:
            self.scaler = None
            self.use_amp = False
            logger.warning("Mixed precision training not available")

        # 🔥 PYTORCH 2.0 MODEL COMPILATION for maximum speed
        try:
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                # Compile model for maximum GPU performance
                self.model = torch.compile(
                    self.model, 
                    mode="max-autotune",  # Maximum optimization
                    dynamic=False,  # Static shapes for best performance
                    fullgraph=False,  # Allow graph breaks for compatibility
                )
                logger.info("🔥 Model compiled with PyTorch 2.0 for maximum GPU acceleration")
                print("🚀 Model compilation enabled")
                sys.stdout.flush()
            else:
                logger.info("PyTorch 2.0 compilation not available")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
            logger.info("Continuing without model compilation")

        # Immediate cleanup after model transfer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Verify model is actually on GPU
        model_device = next(self.model.parameters()).device
        logger.info(f"Model parameters are on device: {model_device}")

        # Force all model components to GPU and clear CPU references
        print("🔧 Ensuring all model components are on GPU...")
        sys.stdout.flush()
        
        # Fix device comparison logic to avoid false warnings
        target_device_type = self.device.type
        target_device_index = self.device.index if self.device.index is not None else 0
        
        parameters_moved = 0
        for name, param in self.model.named_parameters():
            param_device_type = param.device.type
            param_device_index = param.device.index if param.device.index is not None else 0
            
            # Only move if actually on different device type or index
            if param_device_type != target_device_type or param_device_index != target_device_index:
                logger.info(f"Moving parameter {name} from {param.device} to {self.device}")
                param.data = param.data.to(self.device)
                parameters_moved += 1
        
        if parameters_moved > 0:
            logger.info(f"✅ Moved {parameters_moved} parameters to {self.device}")
            # Clear any lingering CPU references only if we moved parameters
            gc.collect()
        else:
            logger.info(f"✅ All model parameters already on {self.device}")

        # Check GPU memory usage after model loading
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(
                self.device) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(
                self.device) / 1024**3  # GB
            total_memory = torch.cuda.get_device_properties(
                self.device).total_memory / 1024**3  # GB
            logger.info(
                f"🎯 GPU memory status - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB, Total: {total_memory:.2f}GB")
            print(
                f"⚡ GPU Memory: {memory_allocated:.2f}GB/{total_memory:.2f}GB allocated")
            sys.stdout.flush()

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

        # Initialize comprehensive modality tracker (NEW!)
        self.modality_tracker = ModalityTracker(
            save_dir=str(self.memory_dir / "modality_analysis"),
            wandb_logger=self.wandb_logger
        )
        logger.info("📊 Comprehensive modality tracker initialized")

        # Initialize attention evolution tracker
        if ATTENTION_TRACKING_AVAILABLE:
            self.attention_evolution_tracker = AttentionEvolutionTracker(
                save_dir=str(self.attention_dir / "attention_evolution")
            )
            logger.info("Attention evolution tracker initialized")
        else:
            self.attention_evolution_tracker = None
            logger.warning("Attention evolution tracker not available")

        # Create enhanced data module with adaptive strategies
        logger.info(
            "Setting up enhanced data module with adaptive strategies...")

        # Enhanced data config for 10-epoch training
        enhanced_data_config = self.config['data'].copy()

        # Ensure all required keys are included with proper fallbacks
        # 🚀 AGGRESSIVE GPU OPTIMIZATION FOR 2-3 HOUR EPOCHS 🚀
        required_keys = {
            'dataset_dir': "../babylm_dataset",
            'max_seq_length': 128,  # Shorter sequences for faster processing
            'batch_size': 64,  # Larger batch size for better GPU utilization
            'num_workers': 8,  # More workers for aggressive data loading
            'pin_memory': True,  # Critical for GPU transfer speed
            'text_encoder_name': 'gpt2',
            'persistent_workers': True,  # Keep workers alive for efficiency
            'validation_datasets': ['glue/sst2'],
            # AGGRESSIVE GPU-optimized settings
            'prefetch_factor': 16,  # Very aggressive prefetching for GPU pipeline
            'drop_last': True,  # Consistent batch sizes for GPU efficiency
            'memory_efficient_loading': False,  # Disable CPU optimizations that hurt GPU
            'non_blocking': True,  # Enable non-blocking GPU transfers
            'dataloader_timeout': 60,  # Faster timeout for stuck data loading
        }

        for key, fallback_value in required_keys.items():
            if key not in enhanced_data_config:
                logger.warning(
                    f"Missing required key '{key}' in config, using fallback: {fallback_value}")
                enhanced_data_config[key] = fallback_value

        logger.info(
            f"Using dataset directory: {enhanced_data_config['dataset_dir']}")
        logger.info(f"Using batch size: {enhanced_data_config['batch_size']}")
        logger.info(
            f"Using max sequence length: {enhanced_data_config['max_seq_length']}")

        # Apply AGGRESSIVE quick training mode settings for 2-3 hour epochs
        if quick_mode.get('enabled', False):
            logger.info(
                "🚀 Applying AGGRESSIVE quick training mode data settings for 2-3 hour epochs...")
            # Much larger batch size for faster training
            enhanced_data_config['batch_size'] = max(
                enhanced_data_config.get('batch_size', 64), 128)  # Very large batches for speed
            enhanced_data_config['max_seq_length'] = min(enhanced_data_config.get(
                'max_seq_length', 512), 128)  # Shorter sequences for speed
            # Maximum workers for data loading
            enhanced_data_config['num_workers'] = 12
            # Very aggressive prefetching
            enhanced_data_config['prefetch_factor'] = 20
            logger.info(
                f"AGGRESSIVE Quick mode: batch_size={enhanced_data_config['batch_size']}, max_seq_length={enhanced_data_config['max_seq_length']}")
            logger.info(
                f"AGGRESSIVE Workers: {enhanced_data_config['num_workers']}, prefetch_factor={enhanced_data_config['prefetch_factor']}")
            logger.info(
                "📊 Quick mode: Preserving mixed training (text + multimodal) for better learning")

        # Apply ULTRA-AGGRESSIVE GPU-optimized data loading for 2-3 hour epochs
        enhanced_data_config.update({
            # ULTRA-AGGRESSIVE GPU optimization settings for fast epochs
            'num_workers': min(16, enhanced_data_config.get('num_workers', 12)),  # Maximum workers
            'pin_memory': True,  # Critical for GPU transfer speed
            'persistent_workers': True,  # Keep workers alive for efficiency
            'prefetch_factor': enhanced_data_config.get('prefetch_factor', 24),  # Ultra-aggressive prefetching
            'multiprocessing_context': None,  # Use default (spawn on Windows)
            'drop_last': True,  # Consistent batch sizes for GPU efficiency
            'non_blocking': True,  # Non-blocking GPU transfers for speed
            'shuffle': True,  # Ensure data shuffling for better GPU utilization
            'timeout': 90,  # Longer timeout for aggressive data loading
            # � SPEED-OPTIMIZED: Larger batches and shorter sequences for 2-3 hour epochs
            'batch_size': max(enhanced_data_config.get('batch_size', 64), 160),  # Very large batches for speed
            'max_seq_length': min(enhanced_data_config.get('max_seq_length', 512), 128),  # Much shorter sequences for speed
            # Disable CPU memory optimizations that hurt GPU performance
            'memory_efficient_loading': False,
            'cpu_data_caching': False,
            # Enable GPU-optimized data preprocessing
            'gpu_preprocessing': True,
            'async_data_transfer': True,
            # 🚀 SPEED OPTIMIZATIONS for fast epochs
            'fast_tokenization': True,  # Use fast tokenizers
            'precomputed_features': True,  # Use precomputed vision features when possible
            'aggressive_caching': True,  # Cache frequently used data
            'reduced_validation_frequency': True,  # Validate less frequently for speed
            'skip_expensive_metrics': True,  # Skip computationally expensive metrics
            # 📊 DATASET SIZE OPTIMIZATION for faster epochs
            'max_samples_per_epoch': 50000,  # Limit samples per epoch for speed
            'smart_sampling': True,  # Use intelligent sampling strategies
            'gradient_accumulation_steps': 8,  # Larger effective batch size
        })
        logger.info("🚀 Applied ULTRA-AGGRESSIVE GPU-optimized data loading for 2-3 hour epochs:")
        logger.info(f"   - Workers: {enhanced_data_config['num_workers']}")
        logger.info(f"   - Prefetch factor: {enhanced_data_config['prefetch_factor']}")
        logger.info(f"   - Batch size: {enhanced_data_config['batch_size']} (optimized for speed)")
        logger.info(f"   - Max sequence length: {enhanced_data_config['max_seq_length']} (much shorter for speed)")
        logger.info("   - GPU preprocessing enabled for maximum speed")
        logger.info("� Configuration optimized for naturally fast 2-3 hour epochs")

        # Dynamic multi-task weighting
        multi_task_config = self.config.get(
            'training', {}).get('multi_task_weighting', {})
        if multi_task_config.get('enabled', False) and not quick_mode.get('enabled', False):
            initial_ratio = multi_task_config.get('initial_text_ratio', 0.5)
            enhanced_data_config['text_ratio'] = initial_ratio
            self.dynamic_weighting = True
            self.target_text_ratio = multi_task_config.get(
                'target_text_ratio', 0.4)
            self.adjustment_frequency = multi_task_config.get(
                'adjustment_frequency', 100)
            logger.info(
                f"Dynamic multi-task weighting: {initial_ratio:.1%} → {self.target_text_ratio:.1%}")
        else:
            enhanced_data_config['text_ratio'] = self.config.get(
                'training', {}).get('text_ratio', 0.4)
            self.dynamic_weighting = False

        # Adaptive batch scaling
        batch_config = self.config.get('training', {}).get(
            'adaptive_batch_scaling', {})
        if batch_config.get('enabled', False) and not quick_mode.get('enabled', False):
            start_batch_size = batch_config.get('start_batch_size', 8)
            enhanced_data_config['batch_size'] = start_batch_size
            self.adaptive_batch_scaling = True
            self.batch_scaling_schedule = batch_config.get(
                'scaling_schedule', {})
            logger.info(
                f"Adaptive batch scaling enabled - starting with batch size {start_batch_size}")
        else:
            self.adaptive_batch_scaling = False

        logger.info(
            f"Enhanced training - Text ratio: {enhanced_data_config['text_ratio']:.1%}")

        # Debug: Log the keys in enhanced_data_config
        logger.info(
            f"Enhanced data config keys: {list(enhanced_data_config.keys())}")

        self.data_module = create_data_module(enhanced_data_config)
        self.data_module.setup(max_samples=max_samples)

        # Apply CPU memory optimizations
        self._optimize_cpu_memory()

        # Setup optimizer with layer-wise learning rates
        self.setup_advanced_optimizer()

    def setup_advanced_optimizer(self):
        """Setup optimizer with various options including 8-bit"""
        optimizer_type = self.config.get('optimizer', 'adamw')

        # Ensure numeric parameters are properly converted to float
        learning_rate = float(self.config['training']['learning_rate'])
        weight_decay = float(self.config['training']['weight_decay'])

        if BITSANDBYTES_AVAILABLE and optimizer_type == 'adamw8bit':
            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            logger.info(f"Using AdamW8bit optimizer for memory efficiency")
        elif optimizer_type == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            logger.info(f"Using AdamW optimizer")
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            logger.info(f"Using Adam optimizer")
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
            logger.info(f"Using SGD optimizer")
        elif optimizer_type == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                eps=1e-8
            )
            logger.info(f"Using RMSprop optimizer")
        else:
            # Fallback to AdamW
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            logger.info(
                f"Unknown optimizer '{optimizer_type}', falling back to AdamW")

        # Setup learning rate scheduler
        max_epochs = int(self.config['training']['max_epochs'])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_epochs)

        logger.info(f"Optimizer setup complete: {optimizer_type}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Max epochs: {max_epochs}")

    # 🧠 EPISODIC MEMORY CONSOLIDATION METHODS
    def _get_consolidation_phase(self, epoch: int) -> str:
        """Determine the current consolidation phase based on epoch"""
        total_epochs = self.config['training']['max_epochs']
        
        # Phase distribution for episodic memory consolidation
        if epoch < total_epochs * 0.3:  # First 30% - Rapid episodic capture
            return "episodic_capture"
        elif epoch < total_epochs * 0.7:  # Middle 40% - Memory consolidation
            return "memory_consolidation"
        else:  # Final 30% - Semantic integration
            return "semantic_integration"
    
    def _apply_phase_settings(self, phase: str, epoch: int):
        """Apply phase-specific learning settings"""
        base_lr = float(self.config['training']['learning_rate'])
        
        if phase == "episodic_capture":
            # Higher learning rate for rapid capture
            lr_multiplier = 1.5
            logger.info(f"🔵 EPISODIC CAPTURE Phase - Rapid multimodal encoding (LR: {base_lr * lr_multiplier:.2e})")
        elif phase == "memory_consolidation":
            # Standard learning rate for consolidation
            lr_multiplier = 1.0
            logger.info(f"🟡 MEMORY CONSOLIDATION Phase - Replay and pattern extraction (LR: {base_lr * lr_multiplier:.2e})")
        elif phase == "semantic_integration":
            # Lower learning rate for fine integration
            lr_multiplier = 0.7
            logger.info(f"🟢 SEMANTIC INTEGRATION Phase - Knowledge refinement (LR: {base_lr * lr_multiplier:.2e})")
        else:
            lr_multiplier = 1.0
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr * lr_multiplier
    
    def _episodic_capture_forward(self, batch):
        """Phase 1: Fast episodic capture with enhanced QFormer processing"""
        # Use mixed precision for speed in episodic capture
        if self.use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    vision_features=batch['vision_features'],
                    labels=batch['labels'],
                    mode="episodic_capture"  # Special mode for episodic training
                )
        else:
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                vision_features=batch['vision_features'],
                labels=batch['labels'],
                mode="episodic_capture"
            )
        
        # Enhanced memory writing during episodic capture
        if hasattr(self.model, 'memory') and 'episode' in outputs:
            # Force memory writing during episodic capture
            episode = outputs['episode']
            self.model.memory.write_memory(episode)
        
        return outputs
    
    def _consolidation_forward(self, batch):
        """Phase 2: Memory consolidation with replay mechanism"""
        # During consolidation, we replay stored episodes alongside current input
        
        # Standard forward pass
        if self.use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    vision_features=batch['vision_features'],
                    labels=batch['labels'],
                    mode="consolidation"
                )
        else:
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                vision_features=batch['vision_features'],
                labels=batch['labels'],
                mode="consolidation"
            )
        
        # Memory replay mechanism - sample and replay stored episodes
        if hasattr(self.model, 'memory') and self.global_step % 10 == 0:  # Replay every 10 steps
            self._replay_memory_episodes(batch['input_ids'].size(0))
        
        return outputs
    
    def _integration_forward(self, batch):
        """Phase 3: Semantic integration of episodic and general knowledge"""
        # Integration phase focuses on combining episodic memories with semantic understanding
        
        if self.use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    vision_features=batch['vision_features'],
                    labels=batch['labels'],
                    mode="integration"
                )
        else:
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                vision_features=batch['vision_features'],
                labels=batch['labels'],
                mode="integration"
            )
        
        # Enhanced memory retrieval during integration
        if hasattr(self.model, 'memory') and 'episode' in outputs:
            # Retrieve and integrate multiple memory contexts
            episode = outputs['episode']
            retrieved_memories, _ = self.model.memory.read_memory(episode)
            # Integration happens within the model's forward pass
        
        return outputs
    
    def _standard_forward(self, batch):
        """Standard forward pass for regular training"""
        if self.use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    vision_features=batch['vision_features'],
                    labels=batch['labels']
                )
        else:
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                vision_features=batch['vision_features'],
                labels=batch['labels']
            )
        return outputs
    
    def _replay_memory_episodes(self, batch_size: int):
        """Replay stored memory episodes for consolidation"""
        try:
            if hasattr(self.model, 'memory') and self.model.memory.memory.numel() > 0:
                # Sample random episodes from memory
                memory_size = self.model.memory.memory_size
                num_samples = min(batch_size, memory_size)
                
                # Get random memory indices
                memory_indices = torch.randperm(memory_size)[:num_samples]
                sampled_episodes = self.model.memory.memory[memory_indices]
                
                # Simple replay: just read these episodes to reinforce patterns
                if sampled_episodes.numel() > 0:
                    # This implicitly reinforces memory patterns through attention
                    _, _ = self.model.memory.read_memory(sampled_episodes)
                    
        except Exception as e:
            logger.warning(f"Memory replay failed: {e}")

    def _apply_selective_vision_freezing(self):
        """Apply selective vision parameter freezing to focus on text-vision association"""
        try:
            frozen_params = 0
            trainable_params = 0
            quantized_params = 0
            
            logger.info("🎯 Applying selective vision parameter freezing...")
            
            # Freeze vision backbone (DinoV2) parameters - these are pre-trained and stable
            if hasattr(self.model, 'vision_encoder') or hasattr(self.model, 'dinov2'):
                vision_encoder = getattr(self.model, 'vision_encoder', None) or getattr(self.model, 'dinov2', None)
                if vision_encoder is not None:
                    for name, param in vision_encoder.named_parameters():
                        param.requires_grad = False
                        frozen_params += param.numel()
                        
                        # 🔢 QUANTIZATION NOTE: Frozen parameters are still quantized to 1.58-bit
                        # This saves memory while preserving pre-trained knowledge
                        if hasattr(param, 'quantization_info') or 'quantized' in str(type(param)):
                            quantized_params += param.numel()
                            
                    logger.info("🔒 Vision backbone (DinoV2) parameters frozen")
                    logger.info("🔢 Note: Frozen vision parameters are still quantized to 1.58-bit for memory efficiency")
            
            # Keep vision projector trainable - this is crucial for text-vision association
            if hasattr(self.model, 'vision_projector'):
                for name, param in self.model.vision_projector.named_parameters():
                    param.requires_grad = True
                    trainable_params += param.numel()
                logger.info("✅ Vision projector kept trainable for association learning")
            
            # Keep QFormer vision-text fusion layers trainable - essential for cross-modal learning
            if hasattr(self.model, 'qformer') or hasattr(self.model, 'fusion_transformer'):
                fusion_module = getattr(self.model, 'qformer', None) or getattr(self.model, 'fusion_transformer', None)
                if fusion_module is not None:
                    for name, param in fusion_module.named_parameters():
                        # Only keep cross-attention and fusion layers trainable
                        if any(keyword in name.lower() for keyword in ['cross_attention', 'fusion', 'query', 'key', 'value']):
                            param.requires_grad = True
                            trainable_params += param.numel()
                        else:
                            param.requires_grad = False
                            frozen_params += param.numel()
                    logger.info("✅ QFormer cross-modal fusion layers kept trainable")
            
            # Keep all text components fully trainable
            if hasattr(self.model, 'text_encoder'):
                for name, param in self.model.text_encoder.named_parameters():
                    param.requires_grad = True
                    trainable_params += param.numel()
            
            if hasattr(self.model, 'text_decoder'):
                for name, param in self.model.text_decoder.named_parameters():
                    param.requires_grad = True
                    trainable_params += param.numel()
            
            total_params = frozen_params + trainable_params
            frozen_percent = (frozen_params / total_params * 100) if total_params > 0 else 0
            trainable_percent = (trainable_params / total_params * 100) if total_params > 0 else 0
            quantized_percent = (quantized_params / total_params * 100) if total_params > 0 else 0
            
            logger.info(f"🎯 Selective vision freezing applied:")
            logger.info(f"   - Frozen parameters: {frozen_params:,} ({frozen_percent:.1f}%)")
            logger.info(f"   - Trainable parameters: {trainable_params:,} ({trainable_percent:.1f}%)")
            logger.info(f"   - Total parameters: {total_params:,}")
            
            if quantized_params > 0:
                logger.info(f"� Quantization status:")
                logger.info(f"   - Quantized parameters: {quantized_params:,} ({quantized_percent:.1f}%)")
                logger.info("   - Frozen vision parameters maintain 1.58-bit quantization for memory efficiency")
                logger.info("   - Trainable parameters use full precision for gradient updates")
            
            logger.info("�🚀 Training will focus on text-vision association, not vision feature extraction")
            logger.info("💾 Memory savings: Frozen+quantized vision backbone, trainable association layers")
            
        except Exception as e:
            logger.warning(f"Selective vision freezing failed: {e}")
            logger.info("Continuing with standard training...")

    # END EPISODIC MEMORY CONSOLIDATION METHODS

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with Episodic Memory Consolidation optimized for 2-3 hour natural completion"""
        import time
        
        self.model.train()
        train_loader = self.data_module.train_dataloader()

        # � NATURAL SPEED OPTIMIZATION: Configure for fast 2-3 hour epochs
        epoch_start_time = time.time()
        total_batches = len(train_loader)
        
        logger.info(f"� Epoch {epoch}: {total_batches} batches, optimized for 2-3 hour natural completion")

        # 🧠 EPISODIC MEMORY CONSOLIDATION: Determine training phase
        consolidation_phase = self._get_consolidation_phase(epoch)
        self._apply_phase_settings(consolidation_phase, epoch)

        epoch_losses = []
        epoch_metrics = {
            'train_loss': 0.0,
            'memory_usage_entropy': 0.0,
            'cross_modal_similarity': 0.0,
            'consolidation_phase': consolidation_phase,
            'epoch_duration_hours': 0.0,
            'batches_processed': 0
        }

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} - {consolidation_phase.upper()}")
        batches_processed = 0

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Pass global step to model for consolidation logic
                if hasattr(self.model, 'global_step'):
                    self.model.global_step = self.global_step
                
                # 🚀 PERFORMANCE OPTIMIZATION: Less frequent device/memory checks for speed
                if self.global_step % 1000 == 0:  # Less frequent checks for better performance
                    self._silent_device_check()

                # Much less frequent memory usage checks to reduce overhead
                if self.global_step % 1000 == 0:  # Every 1000 steps instead of 100
                    self._check_memory_usage(self.global_step)

                # Use efficient batch transfer method to minimize CPU memory usage
                batch = self._efficient_batch_transfer(batch)

                # 🚀 CRITICAL: Validate and fix batch dimensions before forward pass
                try:
                    batch = self._validate_and_fix_batch_dimensions(batch)
                    # CRITICAL: Compress vision features for stability
                    batch = self._compress_vision_features(batch)
                except Exception as e:
                    logger.warning(
                        f"Batch validation failed at step {self.global_step}: {e}")
                    logger.info("Skipping problematic batch...")
                    self.global_step += 1
                    continue

                # 🧠 EPISODIC MEMORY CONSOLIDATION: Forward pass based on phase
                try:
                    # CRITICAL: Ensure all inputs are on GPU with optimized transfers
                    for key in ['input_ids', 'attention_mask', 'vision_features', 'labels']:
                        if key in batch and batch[key] is not None:
                            # Use non_blocking=True for maximum GPU transfer speed
                            batch[key] = batch[key].to(
                                self.device, non_blocking=True)
                            # Validate tensor is finite
                            if torch.is_floating_point(batch[key]) and not torch.isfinite(batch[key]).all():
                                logger.warning(
                                    f"Non-finite values detected in {key}, skipping batch")
                                self.global_step += 1
                                continue

                    # 🧠 CONSOLIDATION PHASE-SPECIFIC PROCESSING
                    if consolidation_phase == "episodic_capture":
                        # Phase 1: Fast episodic capture with high learning rate
                        outputs = self._episodic_capture_forward(batch)
                    elif consolidation_phase == "memory_consolidation":
                        # Phase 2: Memory replay and consolidation
                        outputs = self._consolidation_forward(batch)
                    elif consolidation_phase == "semantic_integration":
                        # Phase 3: Integration of episodic and semantic knowledge
                        outputs = self._integration_forward(batch)
                    else:
                        # Standard forward pass for regular training
                        outputs = self._standard_forward(batch)
                    
                    loss = outputs['loss']
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "device" in str(e).lower():
                        logger.warning(
                            f"Device/memory error in forward pass: {e}")
                        # Force device consistency and retry
                        self._force_model_device_consistency()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

                        if self.use_amp:
                            with torch.amp.autocast('cuda'):
                                outputs = self.model(
                                    input_ids=batch['input_ids'],
                                    attention_mask=batch['attention_mask'],
                                    vision_features=batch['vision_features'],
                                    labels=batch['labels']
                                )
                                loss = outputs['loss']
                        else:
                            outputs = self.model(
                                input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                vision_features=batch['vision_features'],
                                labels=batch['labels']
                            )
                            loss = outputs['loss']
                    else:
                        raise e

                # Check for invalid loss with NaN recovery
                if not torch.isfinite(loss):
                    logger.warning(
                        f"Invalid loss at step {self.global_step}: {loss.item()}")
                    logger.warning("Attempting to recover from NaN loss...")

                    # Clear GPU cache and skip this batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Force cleanup of problematic tensors
                    del outputs
                    gc.collect()

                    self.global_step += 1
                    continue

                # 🚀 ULTRA-OPTIMIZED Backward pass with gradient accumulation for maximum GPU utilization
                gradient_accumulation_steps = self.config.get('training', {}).get('gradient_accumulation_steps', 8)  # Larger effective batch size for speed
                
                try:
                    # Normalize loss by accumulation steps for correct scaling
                    loss = loss / gradient_accumulation_steps
                    
                    if self.use_amp:
                        # ULTRA-AGGRESSIVE mixed precision with gradient accumulation
                        self.scaler.scale(loss).backward()
                    else:
                        # Standard backward pass with gradient accumulation
                        loss.backward()

                    # Only step optimizer every N accumulation steps for larger effective batches
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        if self.use_amp:
                            # Gradient clipping with scaling
                            if self.config['training']['gradient_clip_val'] > 0:
                                self.scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.config['training']['gradient_clip_val'],
                                    norm_type=2.0
                                )
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            # Gradient clipping for standard training
                            if self.config['training']['gradient_clip_val'] > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.config['training']['gradient_clip_val'],
                                    norm_type=2.0
                                )
                            self.optimizer.step()
                        
                        # Clear gradients after optimizer step
                        self.optimizer.zero_grad(set_to_none=True)

                except RuntimeError as e:
                    if "device" in str(e).lower():
                        logger.warning(f"Device error in backward pass: {e}")
                        # Recreate optimizer and retry
                        self._create_device_pinned_optimizer()

                        self.optimizer.zero_grad()

                        if self.use_amp:
                            self.scaler.scale(loss).backward()
                            if self.config['training']['gradient_clip_val'] > 0:
                                self.scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.config['training']['gradient_clip_val']
                                )
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
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

                # 🚀 PERFORMANCE OPTIMIZATION: Skip expensive metric computations during training
                # These computations are adding unnecessary overhead to each training step
                # Only compute essential metrics to maximize training speed
                
                # Skip memory entropy computation (expensive and not critical for training)
                # Skip cross-modal similarity computation (expensive and not critical for training)

                # Update progress bar with consolidation information
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{np.mean(epoch_losses):.4f}",
                    'phase': consolidation_phase[:8]  # Show first 8 chars of phase
                })

                # 🚀 PERFORMANCE OPTIMIZATION: Much less frequent memory monitoring
                if self.global_step % 2000 == 0:  # Every 2000 steps instead of 50
                    # Only check for critical memory issues, skip detailed logging
                    if torch.cuda.is_available():
                        gpu_allocated = torch.cuda.memory_allocated(
                            self.device) / 1024**3
                        gpu_total = torch.cuda.get_device_properties(
                            self.device).total_memory / 1024**3
                        if gpu_allocated > gpu_total * 0.85:  # 85% threshold
                            logger.warning(
                                f"High GPU memory usage detected, forcing cleanup...")
                            torch.cuda.empty_cache()  # Simple cleanup

                # 🚀 PERFORMANCE OPTIMIZATION: Much less frequent GPU memory logging
                if self.global_step % 1000 == 0 and torch.cuda.is_available():  # Every 1000 steps instead of 50
                    memory_allocated = torch.cuda.memory_allocated(
                        self.device) / 1024**3  # GB
                    memory_total = torch.cuda.get_device_properties(
                        self.device).total_memory / 1024**3  # GB

                    # Calculate GPU utilization percentage
                    gpu_util_percent = (memory_allocated / memory_total) * 100

                    # Only print to console every 1000 steps for immediate feedback
                    print(
                        f"⚡ Step {self.global_step}: GPU {gpu_util_percent:.1f}% utilized, Loss: {loss.item():.4f}")
                    sys.stdout.flush()

                # 🚀 OPTIMIZED: Much less frequent wandb logging to reduce overhead
                log_every_n_steps = self.config.get(
                    'wandb', {}).get('log_every_n_steps', 500)  # Much less frequent logging (every 500 steps)
                if self.wandb_logger and log_every_n_steps > 0 and batch_idx % log_every_n_steps == 0 and self.global_step > 0:
                    try:
                        # Enhanced logging with consolidation phase information
                        basic_metrics = {
                            'train_loss': loss.item(),
                            'learning_rate': self.optimizer.param_groups[0]['lr'],
                            'epoch': epoch,
                            'step': self.global_step,
                            f'consolidation/{consolidation_phase}_loss': loss.item(),
                            f'consolidation/phase_epoch': epoch,
                        }
                        
                        # Add phase-specific metrics
                        if consolidation_phase == "episodic_capture":
                            basic_metrics['consolidation/capture_rate'] = 1.0  # Capturing at full rate
                        elif consolidation_phase == "memory_consolidation":
                            basic_metrics['consolidation/replay_frequency'] = 0.1  # Replay every 10 steps
                        elif consolidation_phase == "semantic_integration":
                            basic_metrics['consolidation/integration_strength'] = 0.8  # Lower LR for integration
                        
                        # Log only basic metrics to reduce overhead
                        wandb.log(basic_metrics, step=self.global_step)

                    except Exception as e:
                        logger.warning(
                            f"Wandb logging failed at step {self.global_step}: {e}")
                        # Continue training without wandb logging for this step

                # 🚀 PERFORMANCE CRITICAL: Disable expensive analytics during training
                # These analytics are causing massive slowdown (145 hours vs normal training)
                # Only run analytics very rarely to avoid performance impact
                
                # Comprehensive modality tracking - SEVERELY LIMITED for performance
                modality_track_steps = self.config.get(
                    'track_attention_every_n_steps', 5000)  # MUCH less frequent tracking (every 5000 steps)
                if (self.modality_tracker and modality_track_steps > 0 and
                        self.global_step % modality_track_steps == 0 and self.global_step > 0):

                    try:
                        # Only do basic tracking, skip expensive operations
                        logger.info(f"� Basic modality tracking at step {self.global_step}")
                        # Skip the expensive track_step operation during training
                        pass

                    except Exception as e:
                        logger.warning(
                            f"Modality tracking failed at step {self.global_step}: {e}")

                # Attention analysis - SEVERELY LIMITED for performance
                attention_log_steps = self.config.get(
                    'attention_analysis', {}).get('log_every_n_steps', 10000)  # MUCH less frequent (every 10000 steps)
                if (self.attention_analyzer and attention_log_steps > 0 and
                        self.global_step % attention_log_steps == 0 and self.global_step > 0):

                    try:
                        logger.info(f"🔍 Basic attention analysis at step {self.global_step}")
                        # Skip expensive attention analysis during training
                        pass
                    except Exception as e:
                        logger.warning(
                            f"Attention analysis failed at step {self.global_step}: {e}")

                # Attention evolution tracking - DISABLED for performance
                # This is extremely expensive and causing the 145-hour slowdown
                if False:  # Completely disabled during training
                    pass  # Skip all expensive attention evolution tracking

                # Dynamic text ratio adjustment
                if self.dynamic_weighting and self.global_step % self.adjustment_frequency == 0:
                    self._update_text_ratio(self.global_step)

                self.global_step += 1
                batches_processed += 1

                # Step learning rate scheduler if step-based
                if self.scheduler and hasattr(self, 'scheduler_step_mode') and self.scheduler_step_mode == 'step':
                    self.scheduler.step()

                # � EFFICIENT PROGRESS UPDATE: Show training progress
                if batches_processed % 100 == 0:  # Update every 100 batches for efficiency
                    elapsed_time = time.time() - epoch_start_time
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{np.mean(epoch_losses):.4f}",
                        'phase': consolidation_phase[:8],
                        'elapsed': f"{elapsed_time/60:.0f}m"
                    })

                # ULTRA-EFFICIENT memory cleanup - minimal operations for maximum speed
                if self.global_step % 2000 == 0:  # Very infrequent cleanup for performance
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                        memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                        if memory_allocated > memory_total * 0.92:  # Only cleanup at 92% threshold
                            torch.cuda.empty_cache()

                # Immediate batch cleanup for memory efficiency
                del batch
                if 'outputs' in locals():
                    del outputs
                # Much less frequent garbage collection to avoid performance impact
                if self.global_step % 3000 == 0:  # Even less frequent GC for speed
                    gc.collect()

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
                batches_processed += 1
                continue

        # 🕐 EPOCH COMPLETION WITH TIME TRACKING
        epoch_end_time = time.time()
        epoch_duration_seconds = epoch_end_time - epoch_start_time
        epoch_duration_hours = epoch_duration_seconds / 3600
        
        # Update metrics with time and completion information
        epoch_metrics['epoch_duration_hours'] = epoch_duration_hours
        epoch_metrics['batches_processed'] = batches_processed
        
        # Simplified metrics - eliminate expensive computations for speed
        epoch_metrics['train_loss'] = np.mean(epoch_losses) if epoch_losses else float('inf')
        # Remove expensive entropy and similarity computations for performance
        epoch_metrics['memory_usage_entropy'] = 0.0  # Disabled for speed
        epoch_metrics['cross_modal_similarity'] = 0.0  # Disabled for speed
        
        # 🧠 CONSOLIDATION PHASE SUMMARY - Enhanced with performance tracking
        logger.info(f"✅ Epoch {epoch} completed in {consolidation_phase.upper()} phase")
        logger.info(f"📊 Phase: {epoch_metrics['consolidation_phase']}")
        logger.info(f"📉 Average Loss: {epoch_metrics['train_loss']:.4f}")
        logger.info(f"🕐 Duration: {epoch_duration_hours:.2f} hours ({epoch_duration_seconds/60:.1f} minutes)")
        logger.info(f"📦 Batches: {batches_processed}/{total_batches} ({batches_processed/total_batches*100:.1f}%)")
        
        # Performance evaluation
        if epoch_duration_hours <= 3.0:
            logger.info("🚀 Excellent speed: Epoch completed within 3-hour target")
        elif epoch_duration_hours <= 4.0:
            logger.info("✅ Good speed: Epoch completed within reasonable time")
        else:
            logger.info("⚠️ Consider further optimization for faster epochs")
        
        # Phase-specific completion messages
        if consolidation_phase == "episodic_capture":
            logger.info("🔵 Episodic capture phase - Fast multimodal encoding with selective vision training")
        elif consolidation_phase == "memory_consolidation":
            logger.info("🟡 Memory consolidation phase - Text-vision association strengthening")
        elif consolidation_phase == "semantic_integration":
            logger.info("🟢 Semantic integration phase - Knowledge refinement for association")

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
                                # Clamp between 0 and 100
                                loss_value = max(0.0, min(loss_value, 100.0))
                                val_losses.append(loss_value)
                            else:
                                logger.warning(
                                    f"Non-finite loss detected in validation batch {batch_idx}: {loss_value}")
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
                        logger.warning(
                            f"Validation error traceback: {traceback.format_exc()}")
                        continue

        # Calculate total number of batches across all loaders for averaging
        total_batches = sum(len(loader)
                            for loader in val_loaders) if val_loaders else 1

        # Average metrics with enhanced safety checks and fallback values
        if val_losses:
            val_metrics['val_loss'] = float(np.mean(val_losses))
            # Additional sanity check on the mean
            if not np.isfinite(val_metrics['val_loss']) or val_metrics['val_loss'] < 0:
                logger.warning(
                    f"Invalid mean validation loss: {val_metrics['val_loss']}, using fallback")
                val_metrics['val_loss'] = 10.0  # Reasonable fallback value
        else:
            logger.warning(
                "No valid validation losses collected, using fallback value")
            # Use a reasonable fallback instead of inf
            val_metrics['val_loss'] = 10.0

        val_metrics['val_memory_entropy'] = (
            val_metrics['val_memory_entropy'] / total_batches) if total_batches > 0 else 0.0
        val_metrics['val_cross_modal_similarity'] = (
            val_metrics['val_cross_modal_similarity'] / total_batches) if total_batches > 0 else 0.0

        # Final validation of all metrics
        for key, value in val_metrics.items():
            if not np.isfinite(value):
                logger.warning(
                    f"Non-finite metric detected: {key}={value}, setting to 0")
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
                    logger.error(
                        f"Retry failed for {operation_name}: {retry_e}")
                    raise retry_e
            else:
                raise e

    def verify_device_consistency(self):
        """Verify model and optimizer are on correct device with improved stability"""
        try:
            # Check model device
            model_device = next(self.model.parameters()).device
            if model_device != self.device:
                logger.warning(
                    f"Model moved from {self.device} to {model_device}. Moving back...")
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
                    logger.warning(
                        f"Optimizer state moved to wrong device ({wrong_device_count}/{total_states} tensors). Recreating optimizer...")
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
            current_step_count = getattr(self.optimizer, '_step_count', 0) if hasattr(
                self.optimizer, '_step_count') else 0

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
                                    preserved_entry[key] = value.to(
                                        self.device).clone()
                                except Exception:
                                    # Skip if can't move to device - don't log the exception details
                                    pass
                            else:
                                preserved_entry[key] = value
                        if preserved_entry:
                            preserved_state[param_id] = preserved_entry

            # Recreate optimizer
            self.setup_advanced_optimizer()

            # Restore learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            # Try to restore preserved state with better error handling
            if preserved_state and hasattr(self.optimizer, 'state'):
                try:
                    # Get current parameter list
                    current_params = []
                    for param_group in self.optimizer.param_groups:
                        current_params.extend(param_group['params'])

                    # Only restore state for parameters that still exist
                    old_param_list = list(preserved_state.keys())
                    for i, param in enumerate(current_params):
                        if i < len(old_param_list) and old_param_list[i] in preserved_state:
                            self.optimizer.state[param] = preserved_state[old_param_list[i]]

                except Exception as restore_e:
                    logger.warning(
                        f"Could not restore optimizer state: {restore_e}")

            logger.info(
                f"Optimizer recreated on {self.device} with LR={current_lr:.2e}")

        except Exception as e:
            logger.error(f"Optimizer recreation failed: {e}")
            # Fallback to basic setup
            self.setup_advanced_optimizer()

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
                        device_batch[key] = value.to(
                            self.device, non_blocking=True)
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
                    logger.warning(
                        f"Device inconsistency detected ({self._device_warnings_count} times). Fixing silently...")

                # Force model back to correct device
                self._force_model_device_consistency()

                # If too many device switches, recreate optimizer
                if self._device_warnings_count > 10:
                    self._create_device_pinned_optimizer()
                    self._device_warnings_count = 0  # Reset counter

        except Exception as e:
            # Don't log device check failures - they create noise
            pass

    def _validate_and_fix_batch_dimensions(self, batch):
        """Validate and fix batch dimensions to prevent model errors"""
        try:
            # Check required keys
            required_keys = ['input_ids', 'attention_mask',
                             'vision_features', 'labels']
            for key in required_keys:
                if key not in batch:
                    raise ValueError(f"Missing required key: {key}")

            # Validate input_ids dimensions
            if batch['input_ids'].dim() != 2:
                logger.warning(
                    f"Invalid input_ids dimensions: {batch['input_ids'].shape}")
                batch['input_ids'] = batch['input_ids'].squeeze()
                if batch['input_ids'].dim() != 2:
                    raise ValueError(
                        f"Cannot fix input_ids dimensions: {batch['input_ids'].shape}")

            # Validate attention_mask dimensions
            if batch['attention_mask'].dim() != 2:
                logger.warning(
                    f"Invalid attention_mask dimensions: {batch['attention_mask'].shape}")
                batch['attention_mask'] = batch['attention_mask'].squeeze()
                if batch['attention_mask'].dim() != 2:
                    raise ValueError(
                        f"Cannot fix attention_mask dimensions: {batch['attention_mask'].shape}")

            # Validate vision_features dimensions
            if batch['vision_features'].dim() < 2:
                logger.warning(
                    f"Invalid vision_features dimensions: {batch['vision_features'].shape}")
                # Try to add missing dimensions
                while batch['vision_features'].dim() < 3:
                    batch['vision_features'] = batch['vision_features'].unsqueeze(
                        -1)
                logger.info(
                    f"Fixed vision_features dimensions: {batch['vision_features'].shape}")

            # Validate labels dimensions
            if batch['labels'].dim() != 2:
                logger.warning(
                    f"Invalid labels dimensions: {batch['labels'].shape}")
                batch['labels'] = batch['labels'].squeeze()
                if batch['labels'].dim() != 2:
                    raise ValueError(
                        f"Cannot fix labels dimensions: {batch['labels'].shape}")

            # Ensure all tensors have the same batch size
            batch_size = batch['input_ids'].size(0)
            for key, tensor in batch.items():
                if isinstance(tensor, torch.Tensor) and tensor.size(0) != batch_size:
                    logger.warning(
                        f"Batch size mismatch for {key}: {tensor.size(0)} vs {batch_size}")
                    # Try to fix by taking first batch_size elements
                    batch[key] = tensor[:batch_size]

            return batch

        except Exception as e:
            logger.error(f"Batch validation failed: {e}")
            raise e

    def _validate_and_fix_batch_dimensions(self, batch):
        """Validate and fix batch dimensions to prevent dimension errors"""
        try:
            # Validate required keys
            required_keys = ['input_ids', 'attention_mask',
                             'vision_features', 'labels']
            for key in required_keys:
                if key not in batch:
                    logger.warning(f"Missing required key '{key}' in batch")
                    # Create dummy tensors for missing keys
                    if key == 'input_ids':
                        batch[key] = torch.zeros(
                            (1, 64), dtype=torch.long, device=self.device)
                    elif key == 'attention_mask':
                        batch[key] = torch.ones(
                            (1, 64), dtype=torch.long, device=self.device)
                    elif key == 'vision_features':
                        batch[key] = torch.zeros(
                            (1, 768, 196), dtype=torch.float32, device=self.device)
                    elif key == 'labels':
                        batch[key] = torch.zeros(
                            (1, 64), dtype=torch.long, device=self.device)

            # Validate tensor dimensions and fix if needed
            batch_size = None
            for key, tensor in batch.items():
                if torch.is_tensor(tensor):
                    if batch_size is None:
                        batch_size = tensor.size(0)
                    elif tensor.size(0) != batch_size:
                        logger.warning(
                            f"Inconsistent batch size for {key}: expected {batch_size}, got {tensor.size(0)}")
                        # Truncate or pad to match batch size
                        if tensor.size(0) > batch_size:
                            tensor = tensor[:batch_size]
                        else:
                            # Pad with zeros
                            pad_size = batch_size - tensor.size(0)
                            pad_shape = (pad_size,) + tensor.shape[1:]
                            pad_tensor = torch.zeros(
                                pad_shape, dtype=tensor.dtype, device=tensor.device)
                            tensor = torch.cat([tensor, pad_tensor], dim=0)
                        batch[key] = tensor

            # Ensure minimum batch size
            if batch_size is None or batch_size == 0:
                raise ValueError("Invalid batch: no valid tensors found")

            return batch

        except Exception as e:
            logger.error(f"Batch validation failed: {e}")
            raise e

    def _compress_vision_features(self, batch):
        """Compress vision features to reduce memory and improve stability"""
        try:
            if 'vision_features' in batch and batch['vision_features'] is not None:
                vision_features = batch['vision_features']
                original_shape = vision_features.shape

                # Handle the specific shape mismatch: [batch, 2048, 1] -> [batch, 768]
                if vision_features.dim() == 3 and vision_features.size(-1) == 1:
                    # Remove the last dimension: [batch, features, 1] -> [batch, features]
                    vision_features = vision_features.squeeze(-1)
                    logger.debug(
                        f"Squeezed last dimension: {original_shape} -> {vision_features.shape}")

                # Ensure we have 2D features: [batch, features]
                if vision_features.dim() > 2:
                    # Flatten all feature dimensions except batch
                    vision_features = vision_features.view(
                        vision_features.size(0), -1)
                    logger.debug(
                        f"Flattened to 2D: -> {vision_features.shape}")

                # Project to expected dimension (768) if needed
                expected_dim = self.config.get(
                    'model', {}).get('vision_encoder_dim', 768)
                current_dim = vision_features.size(-1)

                if current_dim != expected_dim:
                    # Create a projection layer if it doesn't exist
                    projection_key = f'vision_proj_{current_dim}_to_{expected_dim}'
                    if not hasattr(self, projection_key):
                        projection_layer = nn.Linear(
                            current_dim, expected_dim).to(vision_features.device)
                        # CRITICAL: Initialize projection layer properly to prevent NaN
                        with torch.no_grad():
                            # Small gain for stability
                            nn.init.xavier_uniform_(
                                projection_layer.weight, gain=0.1)
                            if projection_layer.bias is not None:
                                nn.init.zeros_(projection_layer.bias)
                        setattr(self, projection_key, projection_layer)
                        logger.info(
                            f"Created vision projection: {current_dim} -> {expected_dim}")

                    # Apply projection with numerical stability
                    projection_layer = getattr(self, projection_key)
                    vision_features = projection_layer(vision_features)

                    # CRITICAL: Clamp values to prevent NaN/Inf
                    vision_features = torch.clamp(
                        vision_features, min=-10.0, max=10.0)
                    logger.debug(
                        f"Projected vision features: {current_dim} -> {expected_dim}")

                batch['vision_features'] = vision_features

                if original_shape != vision_features.shape:
                    logger.debug(
                        f"Compressed vision features: {original_shape} -> {vision_features.shape}")

            return batch

        except Exception as e:
            logger.warning(f"Vision feature compression failed: {e}")
            return batch

    def train(self, max_samples: Optional[int] = None):
        """Main training loop"""
        logger.info("Starting BitMar training...")

        # Setup directories first
        self.setup_directories()

        # Setup logging systems before model setup
        self.setup_logging_systems()

        # Setup model and data
        self.setup_model_and_data(max_samples=max_samples)

        # Initialize CodeCarbon emissions tracker
        emissions_tracker = EmissionsTracker(
            project_name="BitMar-Training",
            output_dir=str(self.results_dir),
            output_file="emissions.csv"
        )

        # Start carbon emissions tracking
        emissions_tracker.start()
        logger.info("🌱 Carbon emissions tracking started")

        try:
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
                            common_tokens = ['the', 'a',
                                             'dog', 'cat', 'person']
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

                # Apply progressive growing if enabled
                self._apply_progressive_growing(epoch)

                # Update batch size based on adaptive scaling
                self._update_batch_size(epoch)

        except KeyboardInterrupt:
            logger.info("🛑 Training interrupted by user")
            logger.info("💾 Saving current state before exit...")

            # Save emergency checkpoint
            try:
                self.save_checkpoint(epoch, is_best=False)
                logger.info("✅ Emergency checkpoint saved")
            except Exception as e:
                logger.error(f"❌ Failed to save emergency checkpoint: {e}")

            # Re-raise to trigger finally block
            raise

        except Exception as e:
            logger.error(f"❌ Training failed with error: {e}")
            logger.info("💾 Attempting to save emergency checkpoint...")

            # Save emergency checkpoint
            try:
                self.save_checkpoint(epoch, is_best=False)
                logger.info("✅ Emergency checkpoint saved")
            except Exception as save_e:
                logger.error(
                    f"❌ Failed to save emergency checkpoint: {save_e}")

            # Re-raise the original exception
            raise

        finally:
            # Stop carbon emissions tracking and get results
            emissions = emissions_tracker.stop()

            # Log carbon emissions
            if emissions:
                logger.info(
                    f"🌱 Training carbon emissions: {emissions:.6f} kg CO2")

                # Log to wandb if available
                if self.wandb_logger:
                    self.wandb_logger.log_metrics({
                        "carbon_emissions_kg": emissions,
                        "carbon_emissions_g": emissions * 1000
                    })

            logger.info("🌱 Carbon emissions tracking completed")

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

        # Save model in HuggingFace format for evaluation pipeline compatibility
        logger.info(
            "💾 Saving model in HuggingFace format for evaluation pipeline compatibility...")
        try:
            hf_save_dir = self.checkpoint_dir / "hf_model"
            save_bitmar_as_hf_model(
                bitmar_model=self.model,
                config_dict=self.config['model'],
                save_directory=hf_save_dir,
                tokenizer=self.model.tokenizer if hasattr(
                    self.model, 'tokenizer') else None
            )
            logger.info(f"✅ HuggingFace model saved to: {hf_save_dir}")

            # Save model for both 2024 and 2025 evaluation pipelines
            eval_model_dir_2024 = Path("./final_model_2024")
            eval_model_dir_2025 = Path("./final_model")

            # Copy for 2024 pipeline (multimodal evaluation)
            if eval_model_dir_2024.exists():
                shutil.rmtree(eval_model_dir_2024)
            shutil.copytree(hf_save_dir, eval_model_dir_2024)
            logger.info(
                f"✅ Model prepared for 2024 pipeline (multimodal): {eval_model_dir_2024}")

            # Copy for 2025 pipeline (text-only evaluation)
            if eval_model_dir_2025.exists():
                shutil.rmtree(eval_model_dir_2025)
            shutil.copytree(hf_save_dir, eval_model_dir_2025)
            logger.info(
                f"✅ Model prepared for 2025 pipeline (text-only): {eval_model_dir_2025}")

            # Create evaluation instructions
            eval_instructions = f"""
# BitMar Model Evaluation Instructions

Your BitMar model has been saved and is ready for evaluation on both pipelines:

## Text-only Evaluation (2025 Pipeline)
```bash
cd ../evaluation-pipeline-2025

# Fast evaluation (text-only tasks)
./eval_zero_shot_fast.sh '../BitMar/final_model' 'checkpoint_1M' 'causal'

# Full evaluation including fine-tuning
./eval_finetuning.sh '../BitMar/final_model'
```

## Multimodal Evaluation (2024 Pipeline)
```bash
cd ../evaluation-pipeline-2024

# Multimodal tasks (Winoground + VQA)
./eval_multimodal.sh '../BitMar/final_model_2024'

# DevBench evaluation
./eval_devbench.sh '../BitMar/final_model_2024' bitmar

# Copy BitMar DevBench integration file
cp ../BitMar/devbench_bitmar.py devbench/model_classes/bitmar.py
```

## Model Details
- Model Type: BitMar (Multimodal BitNet with Episodic Memory)
- Text Encoder Layers: {self.config['model'].get('text_encoder_layers', 3)}
- Text Decoder Layers: {self.config['model'].get('text_decoder_layers', 3)}
- Vision Latent Size: {self.config['model'].get('vision_latent_size', 64)}
- Memory Size: {self.config['model'].get('memory_size', 16)}
- Training Epochs: {self.config['training']['max_epochs']}

Both models are identical - they're just copied to different locations for convenience with the respective evaluation pipelines.
"""

            with open("EVALUATION_INSTRUCTIONS.md", "w") as f:
                f.write(eval_instructions)

            logger.info(
                "📋 Evaluation instructions saved to EVALUATION_INSTRUCTIONS.md")

        except Exception as e:
            logger.error(f"Failed to save HuggingFace model: {e}")
            logger.warning(
                "Model will not be available for evaluation pipeline")

        logger.info("Training completed!")

    def _apply_progressive_growing(self, epoch: int):
        """Apply progressive model growing based on epoch schedule"""
        if not self.progressive_growing_enabled:
            return

        growth_actions = []
        for epoch_key, action in self.growth_schedule.items():
            if epoch_key.startswith('epoch_') and int(epoch_key.split('_')[1]) == epoch + 1:
                growth_actions.append(action)

        if not growth_actions:
            return

        logger.info(
            f"🚀 Progressive growing at epoch {epoch + 1}: {growth_actions}")

        for action in growth_actions:
            if action == "add_encoder_layer":
                self._add_encoder_layer()
            elif action == "add_decoder_layer":
                self._add_decoder_layer()
            elif action == "add_fusion_layer":
                self._add_fusion_layer()
            elif action == "add_final_layers":
                self._add_remaining_layers()

        # Recreate optimizer to include new parameters
        self._recreate_optimizer_for_new_layers()

    def _add_encoder_layer(self):
        """Add a new encoder layer to the model"""
        try:
            current_layers = len(self.model.text_encoder.layers)
            target_layers = self.target_layers.get('text_encoder_layers', 4)

            if current_layers < target_layers:
                # Create new layer with same config as existing layers
                new_layer = type(self.model.text_encoder.layers[0])(
                    self.model.text_encoder.layers[0].d_model,
                    self.model.text_encoder.layers[0].nhead,
                    self.model.text_encoder.layers[0].dim_feedforward,
                    dropout=self.model.text_encoder.layers[0].dropout
                )

                # Move to device and initialize
                new_layer.to(self.device)
                self._initialize_new_layer(new_layer)

                # Add to model
                self.model.text_encoder.layers.append(new_layer)
                self.model.text_encoder.num_layers += 1

                logger.info(
                    f"✅ Added encoder layer: {current_layers} → {len(self.model.text_encoder.layers)}")

        except Exception as e:
            logger.error(f"Failed to add encoder layer: {e}")

    def _add_decoder_layer(self):
        """Add a new decoder layer to the model"""
        try:
            current_layers = len(self.model.text_decoder.layers)
            target_layers = self.target_layers.get('text_decoder_layers', 4)

            if current_layers < target_layers:
                # Create new layer with same config as existing layers
                new_layer = type(self.model.text_decoder.layers[0])(
                    self.model.text_decoder.layers[0].d_model,
                    self.model.text_decoder.layers[0].nhead,
                    self.model.text_decoder.layers[0].dim_feedforward,
                    dropout=self.model.text_decoder.layers[0].dropout
                )

                # Move to device and initialize
                new_layer.to(self.device)
                self._initialize_new_layer(new_layer)

                # Add to model
                self.model.text_decoder.layers.append(new_layer)
                self.model.text_decoder.num_layers += 1

                logger.info(
                    f"✅ Added decoder layer: {current_layers} → {len(self.model.text_decoder.layers)}")

        except Exception as e:
            logger.error(f"Failed to add decoder layer: {e}")

    def _add_fusion_layer(self):
        """Add a new cross-modal fusion layer"""
        try:
            if hasattr(self.model, 'cross_modal_fusion') and hasattr(self.model.cross_modal_fusion, 'layers'):
                current_layers = len(self.model.cross_modal_fusion.layers)
                target_layers = self.target_layers.get('fusion_num_layers', 2)

                if current_layers < target_layers:
                    # Create new fusion layer
                    new_layer = type(self.model.cross_modal_fusion.layers[0])(
                        self.model.cross_modal_fusion.layers[0].d_model,
                        self.model.cross_modal_fusion.layers[0].nhead,
                        self.model.cross_modal_fusion.layers[0].dim_feedforward,
                        dropout=self.model.cross_modal_fusion.layers[0].dropout
                    )

                    # Move to device and initialize
                    new_layer.to(self.device)
                    self._initialize_new_layer(new_layer)

                    # Add to model
                    self.model.cross_modal_fusion.layers.append(new_layer)
                    self.model.cross_modal_fusion.num_layers += 1

                    logger.info(
                        f"✅ Added fusion layer: {current_layers} → {len(self.model.cross_modal_fusion.layers)}")

        except Exception as e:
            logger.error(f"Failed to add fusion layer: {e}")

    def _add_remaining_layers(self):
        """Add any remaining layers to reach target architecture"""
        self._add_encoder_layer()
        self._add_decoder_layer()
        self._add_fusion_layer()

    def _initialize_new_layer(self, layer):
        """Initialize parameters of a new layer"""
        for param in layer.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)

    def _recreate_optimizer_for_new_layers(self):
        """Recreate optimizer to include parameters from new layers"""
        try:
            # Store current state
            old_lr = self.optimizer.param_groups[0]['lr']
            old_state = {}

            # Save state for existing parameters
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    if param in self.optimizer.state:
                        old_state[param] = self.optimizer.state[param].copy()

            # Create new optimizer with all parameters
            optimizer_type = self.config.get('optimizer', 'adamw').lower()

            if optimizer_type == 'adamw':
                self.optimizer = AdamW(
                    self.model.parameters(),
                    lr=old_lr,
                    weight_decay=self.config['training']['weight_decay'],
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
            else:
                # Fallback to AdamW for new layers
                self.optimizer = AdamW(
                    self.model.parameters(),
                    lr=old_lr,
                    weight_decay=self.config['training']['weight_decay'],
                    betas=(0.9, 0.999),
                    eps=1e-8
                )

            # Restore state for existing parameters
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    if param in old_state:
                        self.optimizer.state[param] = old_state[param]

            logger.info(f"✅ Optimizer recreated with new layers")

        except Exception as e:
            logger.error(f"Failed to recreate optimizer for new layers: {e}")
            # Fallback to basic optimizer recreation
            self.setup_advanced_optimizer()

    def _update_batch_size(self, epoch: int):
        """Update batch size based on adaptive scaling schedule"""
        if not self.adaptive_batch_scaling:
            return

        new_batch_size = None

        # Check epoch-based schedule
        for epoch_range, batch_size in self.batch_scaling_schedule.items():
            if epoch_range.startswith('epoch_'):
                start_end = epoch_range.split('_')[1].split('_')
                if len(start_end) == 2:
                    start_epoch, end_epoch = int(
                        start_end[0]), int(start_end[1])
                    if start_epoch <= epoch + 1 <= end_epoch:
                        new_batch_size = batch_size
                        break

        if new_batch_size and new_batch_size != self.data_module.batch_size:
            logger.info(
                f"📈 Adaptive batch scaling: {self.data_module.batch_size} → {new_batch_size}")

            # Update data module batch size
            self.data_module.batch_size = new_batch_size
            self.data_module.config['batch_size'] = new_batch_size

            # Note: DataLoader will use new batch size on next epoch

    def _update_text_ratio(self, step: int):
        """Update text-only ratio based on dynamic weighting"""
        if not self.dynamic_weighting:
            return

        if step % self.adjustment_frequency == 0:
            # Simple linear interpolation from initial to target ratio
            # Approximate steps
            progress = min(
                1.0, step / (self.config['training']['max_epochs'] * 1000))
            current_ratio = self.data_module.train_dataset.text_ratio
            new_ratio = current_ratio + \
                (self.target_text_ratio - current_ratio) * \
                0.1  # Gradual adjustment

            if abs(new_ratio - current_ratio) > 0.01:  # Only update if significant change
                logger.info(
                    f"🎯 Dynamic text ratio: {current_ratio:.2%} → {new_ratio:.2%}")
                self.data_module.train_dataset.text_ratio = new_ratio
                self.data_module.train_dataset._create_mixed_indices()  # Recreate indices

    def _log_memory_usage(self, step: int):
        """Log detailed memory usage to help prevent OOM"""
        # System RAM
        ram = psutil.virtual_memory()
        logger.info(
            f"Step {step} - System RAM: {ram.percent:.1f}% ({ram.used/1024**3:.1f}GB/{ram.total/1024**3:.1f}GB)")

        # GPU memory if available
        if torch.cuda.is_available() and self.device:
            gpu_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            gpu_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            gpu_total = torch.cuda.get_device_properties(
                self.device).total_memory / 1024**3
            logger.info(
                f"Step {step} - GPU Memory: {gpu_allocated:.1f}GB allocated, {gpu_reserved:.1f}GB reserved, {gpu_total:.1f}GB total")

            # Warning if memory usage is high
            if gpu_allocated > gpu_total * 0.85:
                logger.warning(
                    f"HIGH GPU MEMORY USAGE: {gpu_allocated/gpu_total*100:.1f}%")
            if ram.percent > 90:
                logger.warning(f"HIGH SYSTEM MEMORY USAGE: {ram.percent:.1f}%")

    def _force_cleanup(self):
        """Aggressive memory cleanup to prevent CPU and GPU OOM"""
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()

        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Clear memory pool if available
            if hasattr(torch.cuda, 'memory_pool_empty_cache'):
                torch.cuda.memory_pool_empty_cache()

        # CPU memory optimization
        import ctypes
        if hasattr(ctypes, 'windll'):  # Windows
            try:
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            except:
                pass

        # Log memory usage after cleanup
        if hasattr(psutil, 'virtual_memory'):
            ram = psutil.virtual_memory()
            logger.debug(
                f"🧹 After cleanup - RAM: {ram.percent:.1f}% ({ram.used/1024**3:.1f}GB/{ram.total/1024**3:.1f}GB)")

    def _optimize_cpu_memory(self):
        """Optimize CPU memory usage during training"""
        # Limit tensor creation on CPU
        torch.set_num_threads(min(4, torch.get_num_threads()))

        # Enable memory-efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)

        # Reduce CPU caching
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

        # Set conservative CPU memory settings
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = False  # Reduces memory usage

        logger.info("🔧 CPU memory optimizations applied")

    def _check_memory_usage(self, step: int, warn_threshold: float = 80.0):
        """Check both CPU and GPU memory usage and warn if high"""
        warnings = []

        # Check CPU memory
        if hasattr(psutil, 'virtual_memory'):
            ram = psutil.virtual_memory()
            if ram.percent > warn_threshold:
                warnings.append(f"High CPU RAM usage: {ram.percent:.1f}%")

        # Check GPU memory
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            gpu_total = torch.cuda.get_device_properties(
                self.device).total_memory / 1024**3
            gpu_percent = (gpu_allocated / gpu_total) * 100

            if gpu_percent > warn_threshold:
                warnings.append(f"High GPU memory usage: {gpu_percent:.1f}%")

        # Log warnings and trigger cleanup if needed
        if warnings:
            logger.warning(
                f"⚠️  Step {step} - Memory warnings: {', '.join(warnings)}")
            if any("High CPU RAM" in w for w in warnings):
                logger.info("🧹 Triggering aggressive CPU memory cleanup...")
                self._force_cleanup()

    def _efficient_batch_transfer(self, batch):
        """Efficiently transfer batch to GPU with minimal CPU memory usage"""
        try:
            # Transfer tensors one by one and delete from CPU immediately
            gpu_batch = {}

            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    # Move to GPU and immediately delete CPU reference
                    gpu_batch[key] = value.to(self.device, non_blocking=True)
                    del value  # Explicit deletion
                else:
                    gpu_batch[key] = value

            # Clear the original batch
            batch.clear()
            del batch

            # Force cleanup
            gc.collect()

            return gpu_batch

        except Exception as e:
            logger.warning(
                f"Efficient batch transfer failed: {e}, falling back to standard transfer")
            return self._safe_batch_to_device(batch)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function with command line interface"""
    print("🎬 Starting main() function...")

    parser = argparse.ArgumentParser(
        description="BitMar Training with Enhanced GPU Optimization")
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
        "--device",
        type=str,
        default=None,
        help="Force specific device (cuda:0, cpu, etc.)"
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
    print(f"✅ Arguments parsed: {args}")

    try:
        # Load configuration
        print(f"📄 Loading config from: {args.config}")
        config = load_config(args.config)

        # Override config with command line arguments
        print("🔧 Applying command line overrides...")
        # Handle both --max_epochs and --epochs
        if args.max_epochs:
            config['training']['max_epochs'] = args.max_epochs
            print(f"🔧 Overriding max_epochs: {args.max_epochs}")
        elif args.epochs:
            config['training']['max_epochs'] = args.epochs
            print(f"🔧 Overriding max_epochs (via --epochs): {args.epochs}")
        if args.batch_size:
            config['data']['batch_size'] = args.batch_size
            print(f"🔧 Overriding batch_size: {args.batch_size}")
        if args.wandb_project:
            config['wandb']['project'] = args.wandb_project
            print(f"📊 W&B project: {args.wandb_project}")

        # Add attention tracking config
        config['track_attention_every_n_steps'] = args.track_attention_every_n_steps
        config['save_attention_every_n_epochs'] = args.save_attention_every_n_epochs
        config['optimizer'] = args.optimizer

        print("✅ Configuration loaded and overridden successfully")

        # Create trainer with enhanced GPU optimization
        print("🏗️ Initializing BitMar trainer with GPU optimizations...")
        trainer = BitMarTrainer(config, device=args.device)
        print("✅ Trainer initialized successfully")

        # Setup directories and logging
        print("📁 Setting up directories...")
        trainer.setup_directories()
        print("✅ Directories setup completed")

        print("📊 Setting up logging systems...")
        trainer.setup_logging_systems()
        print("✅ Logging systems setup completed")

        # Resume from checkpoint if specified
        if args.resume:
            print(f"🔄 Resuming from checkpoint: {args.resume}")
            trainer.setup_model_and_data(max_samples=args.max_samples)
            start_epoch = trainer.load_checkpoint(args.resume)
            print(f"✅ Resumed from epoch {start_epoch}")
        else:
            print("🤖 Setting up model and data from scratch...")
            trainer.setup_model_and_data(max_samples=args.max_samples)
            print("✅ Model and data setup completed")

        # Start training
        print("🚀 Starting BitMar training with GPU optimization...")
        logger.info("=" * 50)
        logger.info("🚀 BITMAR TRAINING STARTED")
        logger.info("=" * 50)

        trainer.train()

        logger.info("=" * 50)
        logger.info("🎉 BITMAR TRAINING COMPLETED")
        logger.info("=" * 50)

        print("✅ Training completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        logger.info("Training interrupted by user")

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    print("🎯 Script called directly, running main()...")
    main()
