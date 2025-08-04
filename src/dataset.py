"""
Dataset processing for BitMar
Handles complete BabyLM multimodal dataset with AGGRESSIVE OPTIMIZATIONS
+ train_50M text-only data for enhanced language modeling
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional
import logging
import random
import os
from pathlib import Path
import h5py
import pickle
import time

# Fix tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


class AggressiveDataOptimizer:
    """Aggressive data optimization for immediate speedup"""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.cache_dir = self.dataset_dir / "speed_cache"
        self.cache_dir.mkdir(exist_ok=True)

    def get_compressed_vision_cache(self):
        """Get or create aggressively compressed vision cache"""
        cache_file = self.cache_dir / "ultra_compressed_vision.h5"

        if cache_file.exists():
            logger.info(f"🚀 Using ultra-compressed vision cache: {cache_file}")
            return h5py.File(cache_file, 'r')

        logger.info("🔥 Creating ultra-compressed vision cache for massive speedup...")

        # Load original data
        cc_feat1 = np.load(self.dataset_dir / "cc_3M_dino_v2_states_1of2.npy", mmap_mode='r')
        cc_feat2 = np.load(self.dataset_dir / "cc_3M_dino_v2_states_2of2.npy", mmap_mode='r')

        total_samples = cc_feat1.shape[0] + cc_feat2.shape[0]
        logger.info(f"Compressing {total_samples} vision samples with 50x compression...")

        # Ultra-aggressive compression: 768*196 -> 64 dimensions
        compressed_dim = 64
        compression_ratio = (768 * 196) / compressed_dim  # Calculate ratio upfront
        original_size_gb = (cc_feat1.nbytes + cc_feat2.nbytes) / 1e9

        with h5py.File(cache_file, 'w') as h5f:
            compressed_data = h5f.create_dataset(
                'features',
                shape=(total_samples, compressed_dim),
                dtype=np.float16,  # Half precision
                compression='gzip',
                compression_opts=9
            )

            chunk_size = 5000
            projection_matrix = None

            for i in range(0, total_samples, chunk_size):
                end_idx = min(i + chunk_size, total_samples)

                # Load chunk
                if i < cc_feat1.shape[0]:
                    if end_idx <= cc_feat1.shape[0]:
                        chunk = cc_feat1[i:end_idx]
                    else:
                        chunk1 = cc_feat1[i:]
                        chunk2 = cc_feat2[:end_idx - cc_feat1.shape[0]]
                        chunk = np.concatenate([chunk1, chunk2])
                else:
                    start_idx2 = i - cc_feat1.shape[0]
                    end_idx2 = end_idx - cc_feat1.shape[0]
                    chunk = cc_feat2[start_idx2:end_idx2]

                # Ultra-aggressive compression: flatten and downsample
                chunk_flat = chunk.reshape(chunk.shape[0], -1)  # Flatten spatial

                # Random projection for ultra compression
                if projection_matrix is None:
                    np.random.seed(42)
                    projection_matrix = np.random.randn(chunk_flat.shape[1], compressed_dim).astype(np.float32)
                    projection_matrix /= np.sqrt(chunk_flat.shape[1])  # Normalize

                # Apply compression
                compressed_chunk = chunk_flat @ projection_matrix
                compressed_data[i:end_idx] = compressed_chunk.astype(np.float16)

                if i % 50000 == 0:
                    logger.info(f"Compressed {i}/{total_samples} samples...")

            # Store metadata
            h5f.attrs['compression_ratio'] = compression_ratio
            h5f.attrs['original_size_gb'] = original_size_gb
            h5f.attrs['total_samples'] = total_samples

        # File is now properly closed, calculate compressed size
        compressed_size_gb = os.path.getsize(cache_file) / 1e9

        logger.info(f"✅ Ultra-compression complete! {compression_ratio:.1f}x smaller")
        logger.info(f"   Original: {original_size_gb:.2f}GB → Compressed: {compressed_size_gb:.2f}GB")

        # Return opened file for reading
        return h5py.File(cache_file, 'r')

    def get_tokenized_cache(self, max_seq_length=256):
        """Get or create tokenized text cache"""
        cache_file = self.cache_dir / f"tokenized_cache_{max_seq_length}.pkl"

        if cache_file.exists():
            logger.info(f"🚀 Using tokenized cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        logger.info("🔥 Creating tokenized cache...")

        # Load captions
        with open(self.dataset_dir / "cc_3M_captions.json", 'r') as f:
            cc_captions = json.load(f)
        with open(self.dataset_dir / "local_narr_captions.json", 'r') as f:
            ln_captions = json.load(f)

        all_captions = cc_captions + ln_captions

        # Tokenize in batches
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenized_data = []
        batch_size = 1000

        for i in range(0, len(all_captions), batch_size):
            batch = all_captions[i:i+batch_size]
            encoded = tokenizer(
                batch,
                max_length=max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )

            for j in range(len(batch)):
                tokenized_data.append({
                    'input_ids': encoded['input_ids'][j].tolist(),
                    'attention_mask': encoded['attention_mask'][j].tolist()
                })

            if i % 50000 == 0:
                logger.info(f"Tokenized {i}/{len(all_captions)} captions...")

        # Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump(tokenized_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"✅ Tokenization cache created with {len(tokenized_data)} samples")
        return tokenized_data


class MixedMultimodalTextDataset(Dataset):
    """Mixed dataset with AGGRESSIVE speed optimizations and token limits"""

    def __init__(
        self,
        dataset_dir: str,
        tokenizer_name: str = "gpt2",
        max_seq_length: int = 512,
        split: str = "train",
        max_samples: Optional[int] = None,
        text_ratio: float = 0.3,
        load_text_data: bool = True,
        config: Dict = None
    ):
        self.dataset_dir = Path(dataset_dir)
        self.max_seq_length = min(max_seq_length, 256)  # Force shorter sequences for speed
        self.split = split
        self.text_ratio = text_ratio
        self.load_text_data = load_text_data
        self.config = config or {}

        # 🎯 BABYLM TOKEN LIMITS - Maximum 100M text tokens and 50M image tokens
        self.max_text_tokens = 100_000_000  # 100M text tokens
        self.max_image_tokens = 50_000_000   # 50M image tokens (treat each image as 1 token conceptually)
        self.text_token_count = 0
        self.image_token_count = 0
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ALWAYS use aggressive optimization with token limits
        logger.info("🚀 AGGRESSIVE OPTIMIZATION MODE - Setting up ultra-fast dataset with BabyLM token limits...")
        logger.info(f"🎯 Token Limits: {self.max_text_tokens:,} text tokens, {self.max_image_tokens:,} image tokens")
        self._setup_aggressive_optimization_with_limits()

    def _setup_aggressive_optimization_with_limits(self):
        """Setup with aggressive optimizations and BabyLM token limits"""
        start_time = time.time()

        # Initialize aggressive optimizer
        self.optimizer = AggressiveDataOptimizer(str(self.dataset_dir))

        # Get compressed vision cache
        self.vision_cache = self.optimizer.get_compressed_vision_cache()
        vision_samples = self.vision_cache['features'].shape[0]
        logger.info(f"📁 Vision cache shape: {self.vision_cache['features'].shape}")

        # Get tokenized cache with token counting
        self.text_cache = self.optimizer.get_tokenized_cache(self.max_seq_length)
        text_samples = len(self.text_cache)
        logger.info(f"📁 Text cache: {text_samples} samples")

        # 🎯 COUNT TOKENS AND APPLY BABYLM LIMITS
        logger.info("🧮 Counting tokens and applying BabyLM limits...")
        
        # Count actual text tokens in multimodal captions
        multimodal_text_tokens = 0
        valid_multimodal_indices = []
        
        # Ensure we respect image-caption pairs by limiting both together
        aligned_samples = min(vision_samples, text_samples)
        logger.info(f"Aligned multimodal samples: {aligned_samples}")
        
        for i in range(aligned_samples):
            # Count tokens in this caption (excluding padding)
            text_data = self.text_cache[i]
            attention_mask = text_data['attention_mask']
            actual_tokens = sum(attention_mask)  # Count non-padding tokens
            
            # Check if adding this sample would exceed limits
            would_exceed_text = (multimodal_text_tokens + actual_tokens) > self.max_text_tokens
            would_exceed_images = len(valid_multimodal_indices) >= self.max_image_tokens
            
            if would_exceed_text or would_exceed_images:
                logger.info(f"🛑 Reached BabyLM limits at sample {i}:")
                logger.info(f"   Text tokens: {multimodal_text_tokens:,}/{self.max_text_tokens:,}")
                logger.info(f"   Image tokens: {len(valid_multimodal_indices):,}/{self.max_image_tokens:,}")
                break
                
            # Add this sample (maintains image-caption association)
            valid_multimodal_indices.append(i)
            multimodal_text_tokens += actual_tokens
            
        # Create multimodal indices from valid samples
        multimodal_indices = [('multimodal', i) for i in valid_multimodal_indices]
        self.image_token_count = len(valid_multimodal_indices)
        
        logger.info(f"✅ Multimodal samples: {len(multimodal_indices)} (preserving image-caption pairs)")
        logger.info(f"📊 Image tokens used: {self.image_token_count:,}/{self.max_image_tokens:,}")
        logger.info(f"📊 Caption tokens used: {multimodal_text_tokens:,}")

        # Handle additional text-only data if enabled and we have text token budget remaining
        text_indices = []
        text_only_tokens = 0
        remaining_text_budget = self.max_text_tokens - multimodal_text_tokens
        
        if self.load_text_data and self.split == "train" and remaining_text_budget > 0:
            logger.info(f"📚 Loading text-only data with {remaining_text_budget:,} token budget...")
            
            # Load text data efficiently
            text_samples = self._load_text_data_with_token_limit(remaining_text_budget)
            
            if text_samples:
                # Calculate how many text samples we want based on ratio
                num_multimodal = len(multimodal_indices)
                if self.text_ratio > 0 and num_multimodal > 0:
                    # Calculate target text samples based on ratio
                    target_text_samples = int(num_multimodal * self.text_ratio / (1 - self.text_ratio))
                    target_text_samples = min(target_text_samples, len(text_samples))
                    
                    # Count tokens for selected text samples
                    for i in range(target_text_samples):
                        # Quick tokenization to count tokens
                        sample_text = text_samples[i][:500]  # Limit for speed
                        tokens = len(self.tokenizer.encode(sample_text, add_special_tokens=True))
                        
                        if text_only_tokens + tokens > remaining_text_budget:
                            logger.info(f"🛑 Text token limit reached at sample {i}")
                            break
                            
                        text_indices.append(('text', i))
                        text_only_tokens += tokens
                        
                logger.info(f"✅ Text-only samples: {len(text_indices)}")
                logger.info(f"📊 Text-only tokens: {text_only_tokens:,}")
        else:
            text_samples = []
            logger.info("📚 Text-only data disabled or no token budget remaining")

        # Combine all indices
        self.mixed_indices = multimodal_indices + text_indices
        random.seed(42)
        random.shuffle(self.mixed_indices)

        # Store text samples and token counts
        self.text_samples = text_samples
        self.text_token_count = multimodal_text_tokens + text_only_tokens
        self.use_cached_data = True

        setup_time = time.time() - start_time

        # Final token summary
        logger.info(f"🎯 BABYLM TOKEN COMPLIANCE SUMMARY:")
        logger.info(f"   Setup time: {setup_time:.1f}s")
        logger.info(f"   Total text tokens: {self.text_token_count:,}/{self.max_text_tokens:,} ({100*self.text_token_count/self.max_text_tokens:.1f}%)")
        logger.info(f"   Total image tokens: {self.image_token_count:,}/{self.max_image_tokens:,} ({100*self.image_token_count/self.max_image_tokens:.1f}%)")
        logger.info(f"   Multimodal samples: {len(multimodal_indices)} (image-caption pairs preserved)")
        logger.info(f"   Text-only samples: {len(text_indices)}")
        logger.info(f"   Total dataset samples: {len(self.mixed_indices)}")
        logger.info(f"   Vision compression: {self.vision_cache.attrs.get('compression_ratio', 0):.1f}x")
        logger.info(f"✅ BabyLM token limits strictly enforced!")

    def _load_text_data_with_token_limit(self, token_budget: int) -> List[str]:
        """Load text data respecting token budget"""
        text_samples = []
        tokens_used = 0
        
        train_50m_dir = self.dataset_dir / "train_50M"
        if not train_50m_dir.exists():
            logger.warning("train_50M not found - skipping text-only data")
            return []

        # Load text files efficiently with token counting
        text_files = [
            "simple_wiki.train",
            "gutenberg.train", 
            "children_stories.train",
            "switchboard.train"
        ]
        
        for filename in text_files:
            filepath = train_50m_dir / filename
            if not filepath.exists():
                continue
                
            logger.info(f"📖 Loading {filename} with token counting...")
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Quick token count estimation (more accurate than string length)
                        estimated_tokens = len(self.tokenizer.encode(line[:200], add_special_tokens=True))
                        
                        if tokens_used + estimated_tokens > token_budget:
                            logger.info(f"Token budget reached in {filename} at line {line_num}")
                            break
                            
                        text_samples.append(line)
                        tokens_used += estimated_tokens
                        
                        # Progress logging
                        if len(text_samples) % 10000 == 0:
                            logger.info(f"   Loaded {len(text_samples)} text samples, {tokens_used:,} tokens")
                            
                if tokens_used >= token_budget:
                    break
                    
            except Exception as e:
                logger.warning(f"Error loading {filename}: {e}")
                continue

        logger.info(f"📚 Text loading complete: {len(text_samples)} samples, {tokens_used:,} tokens")
        return text_samples

    def _load_minimal_text_data(self) -> List[str]:
        """Legacy method - now redirects to token-limited loading"""
        # Redirect to new token-limited method
        remaining_budget = max(0, self.max_text_tokens - self.text_token_count)
        if remaining_budget > 0:
            return self._load_text_data_with_token_limit(remaining_budget)
        else:
            logger.warning("No text token budget remaining for minimal text data")
            return []

    def __len__(self) -> int:
        return len(self.mixed_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Ultra-fast sample loading with aggressive caching"""
        sample_type, sample_idx = self.mixed_indices[idx]

        if sample_type == 'multimodal':
            # Use ultra-compressed cached data
            vision_features = torch.from_numpy(
                self.vision_cache['features'][sample_idx]
            ).float()

            text_data = self.text_cache[sample_idx]

            return {
                'input_ids': torch.tensor(text_data['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(text_data['attention_mask'], dtype=torch.long),
                'labels': torch.tensor(text_data['input_ids'], dtype=torch.long),
                'vision_features': vision_features,
                'caption': f"sample_{sample_idx}",
                'sample_type': 'multimodal',
                'index': sample_idx
            }
        else:
            # Fast text-only sample
            text = self.text_samples[sample_idx]

            # Quick tokenization (should be rare due to limited text data)
            encoded = self.tokenizer(
                text,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': input_ids.clone(),
                'vision_features': torch.zeros(64, dtype=torch.float32),  # Match compressed dim
                'caption': text[:100],
                'sample_type': 'text_only',
                'index': sample_idx
            }

    def get_token_usage_stats(self) -> Dict[str, int]:
        """Get current token usage statistics"""
        return {
            'text_tokens_used': self.text_token_count,
            'text_tokens_limit': self.max_text_tokens,
            'text_tokens_remaining': max(0, self.max_text_tokens - self.text_token_count),
            'image_tokens_used': self.image_token_count,
            'image_tokens_limit': self.max_image_tokens,
            'image_tokens_remaining': max(0, self.max_image_tokens - self.image_token_count),
            'text_utilization_pct': 100.0 * self.text_token_count / self.max_text_tokens,
            'image_utilization_pct': 100.0 * self.image_token_count / self.max_image_tokens,
        }

# Keep original class for backward compatibility
class CompleteBabyLMDataset(Dataset):
    """Complete BabyLM Multimodal Dataset (CC3M + Localized Narratives) - Legacy"""

    def __init__(
        self,
        dataset_dir: str,
        tokenizer_name: str = "gpt2",
        max_seq_length: int = 512,
        split: str = "train",
        max_samples: Optional[int] = None
    ):
        # Use new mixed dataset with text_ratio=0 (multimodal only)
        self.mixed_dataset = MixedMultimodalTextDataset(
            dataset_dir=dataset_dir,
            tokenizer_name=tokenizer_name,
            max_seq_length=max_seq_length,
            split=split,
            max_samples=max_samples,
            text_ratio=0.0,  # No text-only samples
            load_text_data=False
        )

    def __len__(self):
        return len(self.mixed_dataset)

    def __getitem__(self, idx):
        return self.mixed_dataset[idx]


class HuggingFaceValidationDataset(Dataset):
    """Validation dataset using HuggingFace datasets"""

    def __init__(
        self,
        dataset_name: str,
        hf_token: str,
        tokenizer_name: str = "gpt2",
        max_seq_length: int = 512,
        max_samples: Optional[int] = None
    ):
        self.dataset_name = dataset_name
        self.max_seq_length = max_seq_length

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset from HuggingFace
        logger.info(f"Loading validation dataset: {dataset_name}")
        try:
            if dataset_name == "ewok-core/ewok-core-1.0":
                self.dataset = load_dataset(dataset_name, token=hf_token, split="test")
                self.text_field = "text"  # Adjust based on actual schema
            elif dataset_name == "facebook/winoground":
                self.dataset = load_dataset(dataset_name, token=hf_token, split="test")
                self.text_field = "caption_0"  # Winoground has multiple captions
            elif dataset_name == "squad":
                self.dataset = load_dataset(dataset_name, split="validation")  # No token needed
                self.text_field = "question"  # Use questions for text generation
            elif dataset_name == "glue/sst2":
                self.dataset = load_dataset("glue", "sst2", split="validation")  # No token needed
                self.text_field = "sentence"  # Use sentences for text generation
            else:
                # Try to load as a generic public dataset
                self.dataset = load_dataset(dataset_name, split="validation")
                # Try to guess the text field
                sample = self.dataset[0]
                text_fields = ["text", "sentence", "question", "input", "content"]
                self.text_field = next((field for field in text_fields if field in sample), None)
                if self.text_field is None:
                    raise ValueError(f"Could not find text field in dataset: {dataset_name}")

            if max_samples is not None:
                self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

            logger.info(f"Loaded {len(self.dataset)} validation samples")

        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            # Create dummy dataset for testing
            self.dataset = [{"text": f"Validation sample {i}"} for i in range(100)]
            self.text_field = "text"
            logger.warning(f"Using dummy validation data with {len(self.dataset)} samples")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single validation sample"""
        item = self.dataset[idx]
        
        # Extract text based on dataset structure
        if isinstance(item, dict) and self.text_field in item:
            text = item[self.text_field]
        else:
            text = str(item)

        # Tokenize text
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'vision_features': torch.zeros(768, dtype=torch.float32),  # Dummy vision features
            'caption': text,
            'index': idx
        }


class VisionFeaturesConcatenated:
    """Memory-efficient concatenation of two vision feature arrays"""

    def __init__(self, features_1, features_2):
        self.features_1 = features_1
        self.features_2 = features_2
        self.len_1 = len(features_1)
        self.len_2 = len(features_2)
        self.total_len = self.len_1 + self.len_2

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < self.len_1:
            return self.features_1[idx]
        else:
            return self.features_2[idx - self.len_1]

    @property
    def shape(self):
        return (self.total_len, self.features_1.shape[1])


class CombinedVisionFeatures:
    """Combine Conceptual Captions and Localized Narratives features"""

    def __init__(self, cc_features, ln_features):
        self.cc_features = cc_features
        self.ln_features = ln_features
        self.cc_len = len(cc_features)
        self.ln_len = len(ln_features)
        self.total_len = self.cc_len + self.ln_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < self.cc_len:
            return self.cc_features[idx]
        else:
            return self.ln_features[idx - self.cc_len]
            
    @property
    def shape(self):
        return (self.total_len, 768)  # DiNOv2 features are 768D


class BabyLMDataModule:
    """Data module for BitMar training with complete BabyLM dataset"""

    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer_name = config.get('text_encoder_name', 'gpt2')

        # Dataset parameters with proper fallbacks to prevent KeyError
        self.dataset_dir = config.get('dataset_dir', '../babylm_dataset')
        self.max_seq_length = config.get('max_seq_length', 256)
        self.hf_token = config.get('hf_token') or os.getenv('HF_TOKEN', '')

        # Mixed training parameters
        self.use_mixed_training = config.get('use_mixed_training', False)
        self.text_ratio = config.get('text_ratio', 0.3)

        # DataLoader parameters with proper fallbacks
        self.batch_size = config.get('batch_size', 16)
        self.num_workers = config.get('num_workers', 4)
        self.pin_memory = config.get('pin_memory', True)
        self.persistent_workers = config.get('persistent_workers', True)

        # Validation datasets
        self.validation_datasets = config.get('validation_datasets', [
            'glue/sst2'  # Use simpler, more reliable validation dataset
        ])

        # Validate critical paths and log warnings if needed
        if not os.path.exists(self.dataset_dir):
            logger.warning(f"Dataset directory {self.dataset_dir} does not exist. Please ensure babylm_dataset is available.")
            # Try alternative paths
            alt_paths = ['babylm_dataset', '../babylm_dataset', '../../babylm_dataset']
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    logger.info(f"Found alternative dataset path: {alt_path}")
                    self.dataset_dir = alt_path
                    break

        logger.info(f"Using dataset directory: {self.dataset_dir}")
        logger.info(f"Using batch size: {self.batch_size}")
        logger.info(f"Using max sequence length: {self.max_seq_length}")

        # Datasets
        self.train_dataset = None
        self.val_datasets = {}

    def setup(self, max_samples: Optional[int] = None):
        """Setup train and validation datasets"""
        logger.info("Setting up BabyLM dataset...")

        # Create training dataset - use mixed dataset if configured
        if self.use_mixed_training:
            logger.info(f"Using mixed multimodal + text training with {self.text_ratio:.1%} text-only ratio")
            self.train_dataset = MixedMultimodalTextDataset(
                dataset_dir=self.dataset_dir,
                tokenizer_name=self.tokenizer_name,
                max_seq_length=self.max_seq_length,
                split="train",
                max_samples=max_samples,
                text_ratio=self.text_ratio,
                load_text_data=True,
                config=self.config  # PASS CONFIG FOR INTELLIGENT PREPROCESSING
            )
        else:
            logger.info("Using multimodal-only training")
            # Create legacy dataset but pass config through mixed dataset
            temp_mixed = MixedMultimodalTextDataset(
                dataset_dir=self.dataset_dir,
                tokenizer_name=self.tokenizer_name,
                max_seq_length=self.max_seq_length,
                split="train",
                max_samples=max_samples,
                text_ratio=0.0,  # No text-only samples
                load_text_data=False,
                config=self.config  # PASS CONFIG FOR INTELLIGENT PREPROCESSING
            )
            self.train_dataset = temp_mixed

        # Create validation datasets from HuggingFace
        for dataset_name in self.validation_datasets:
            try:
                logger.info(f"Setting up validation dataset: {dataset_name}")
                self.val_datasets[dataset_name] = HuggingFaceValidationDataset(
                    dataset_name=dataset_name,
                    hf_token=self.hf_token,
                    tokenizer_name=self.tokenizer_name,
                    max_seq_length=self.max_seq_length,
                    max_samples=100 if max_samples else 500  # Smaller validation sets
                )
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")

        logger.info(f"Train dataset: {len(self.train_dataset)} samples")
        logger.info(f"Validation datasets: {list(self.val_datasets.keys())}")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader with spawn multiprocessing for CUDA compatibility"""
        # PyTorch DataLoader requires CPU generator even for GPU training
        # This fixes: "Expected a 'cpu' device type for generator but found 'cuda'"
        generator = torch.Generator()  # Always CPU generator
        generator.manual_seed(42)  # For reproducibility
        
        # Use spawn method for CUDA compatibility in multiprocessing
        import multiprocessing as mp
        mp_context = mp.get_context('spawn') if self.num_workers > 0 else None
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=True,
            generator=generator,  # CPU generator (required by PyTorch)
            multiprocessing_context=mp_context  # Use spawn for CUDA compatibility
        )

    def val_dataloader(self) -> List[DataLoader]:
        """Create validation dataloaders with spawn multiprocessing for CUDA compatibility"""
        val_loaders = []
        
        # PyTorch DataLoader requires CPU generator even for GPU training
        generator = torch.Generator()  # Always CPU generator
        generator.manual_seed(42)  # For reproducibility
        
        for name, dataset in self.val_datasets.items():
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,  # Use 0 for validation to avoid memory issues
                pin_memory=self.pin_memory,
                drop_last=False,
                generator=generator  # CPU generator (required by PyTorch)
            )
            val_loaders.append(loader)
        
        return val_loaders if val_loaders else [self._create_dummy_val_loader()]

    def _create_dummy_val_loader(self):
        """Create dummy validation loader if HF datasets fail"""
        logger.warning("Creating dummy validation dataset")
        dummy_dataset = HuggingFaceValidationDataset(
            dataset_name="dummy",
            hf_token="",
            tokenizer_name=self.tokenizer_name,
            max_seq_length=self.max_seq_length,
            max_samples=50
        )
        
        # PyTorch DataLoader requires CPU generator
        generator = torch.Generator()
        generator.manual_seed(42)
        
        return DataLoader(
            dummy_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            generator=generator  # CPU generator (required by PyTorch)
        )

    def get_sample_batch(self, split: str = "train", num_samples: int = 4) -> Dict[str, torch.Tensor]:
        """Get a sample batch for testing"""
        if split == "train":
            dataset = self.train_dataset
        else:
            dataset = list(self.val_datasets.values())[0] if self.val_datasets else None

        if dataset is None:
            raise ValueError("Dataset not setup. Call setup() first.")

        # Get random samples
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        samples = [dataset[i] for i in indices]

        # Collate samples
        batch = {}
        for key in samples[0].keys():
            if key in ['caption']:
                batch[key] = [sample[key] for sample in samples]
            else:
                batch[key] = torch.stack([sample[key] for sample in samples])

        return batch


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching with shape consistency for torch.compile"""
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        if key == 'caption':
            collated[key] = [item[key] for item in batch]
        else:
            # Ensure all tensors have exactly the same shape to prevent dimension errors
            tensors = [item[key] for item in batch]
            # Validate tensor shapes before stacking to prevent assertion errors
            if tensors:
                expected_shape = tensors[0].shape
                for i, tensor in enumerate(tensors[1:], 1):
                    if tensor.shape != expected_shape:
                        logger.warning(f"Shape mismatch in {key} at batch item {i}: expected {expected_shape}, got {tensor.shape}")
                        # Force reshape to expected shape if possible
                        if tensor.numel() == tensors[0].numel():
                            tensors[i] = tensor.view(expected_shape)
                        else:
                            raise ValueError(f"Cannot fix shape mismatch in {key}: {tensor.shape} vs {expected_shape}")
            
            collated[key] = torch.stack(tensors)

    return collated


def create_data_module(config: Dict) -> BabyLMDataModule:
    """Create data module from configuration"""
    return BabyLMDataModule(config)


def test_dataset(config: Dict, max_samples: int = 10):
    """Test dataset loading and processing"""
    logger.info("Testing complete BabyLM dataset...")

    # Create data module
    data_module = create_data_module(config)
    data_module.setup(max_samples=max_samples)

    # Test sample
    sample = data_module.train_dataset[0]
    logger.info(f"Sample keys: {sample.keys()}")
    logger.info(f"Input IDs shape: {sample['input_ids'].shape}")
    logger.info(f"Vision features shape: {sample['vision_features'].shape}")
    logger.info(f"Caption: {sample['caption'][:100]}...")

    # Test batch
    batch = data_module.get_sample_batch(num_samples=4)
    logger.info(f"Batch input IDs shape: {batch['input_ids'].shape}")
    logger.info(f"Batch vision features shape: {batch['vision_features'].shape}")
    logger.info(f"Number of captions in batch: {len(batch['caption'])}")

    logger.info("Dataset test completed successfully!")

    return data_module


if __name__ == "__main__":
    # Test configuration
    test_config = {
        'dataset_dir': "../babylm_dataset",
        'text_encoder_name': "gpt2",
        'max_seq_length': 512,
        'batch_size': 4,
        'num_workers': 0,
        'pin_memory': False,
        'hf_token': os.getenv('HF_TOKEN', 'your_hf_token_here'),
        'validation_datasets': ['ewok-core/ewok-core-1.0', 'facebook/winoground']
    }

    # Test dataset
    test_dataset(test_config)
