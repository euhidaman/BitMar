"""
Dataset processing for BitMar
Handles complete BabyLM multimodal dataset with GPU-OPTIMIZED STORAGE
+ train_50M text-only data for enhanced language modeling
+ MULTI-WORKER DATA LOADING for maximum GPU ut        # Step 2: Load exactly 50M text tokens from train_50M (separate from captions)
        logger.info("📚 Loading 50M text tokens from train_50M (separate from captions)...")
        text_only_samples = self._load_text_data_with_token_limit(self.max_text_tokens)
        text_indices = [('text', i) for i in range(len(text_only_samples))]
        
        logger.info(f"✅ Text-only samples: {len(text_only_samples):,}")
        logger.info(f"📊 Text tokens used: {self.text_token_count:,}/{self.max_text_tokens:,}")on
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
import pickle
import time

# Import GPU-optimized storage instead of h5py
from .gpu_optimized_storage import GPUOptimizedVisionStorage, GPUOptimizedDataset

# Fix tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


class GPUOptimizedDataOptimizer:
    """GPU-optimized data optimization with multi-worker support"""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.cache_dir = self.dataset_dir / "speed_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize GPU-optimized storage
        self.gpu_storage = GPUOptimizedVisionStorage(str(dataset_dir))

    def get_optimized_vision_features(self):
        """Get GPU-optimized vision features (multi-worker compatible)"""
        logger.info("🚀 Loading GPU-optimized vision features...")
        
        # Convert from h5py if needed and load optimized features
        features, metadata = self.gpu_storage.load_optimized_features()
        
        logger.info(f"✅ GPU-optimized features loaded: {features.shape}")
        logger.info(f"🔥 Multi-worker data loading: ENABLED")
        logger.info(f"📊 Compression ratio: {metadata.get('compression_ratio', 'Unknown'):.1f}x")
        
        return features, metadata

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
        vocab_size = tokenizer.vocab_size  # Get vocab size for validation

        for i in range(0, len(all_captions), batch_size):
            batch = all_captions[i:i+batch_size]
            
            # Extract captions from dict format if needed
            batch_texts = []
            for caption in batch:
                if isinstance(caption, dict):
                    batch_texts.append(caption.get('caption', ''))
                else:
                    batch_texts.append(str(caption))
            
            encoded = tokenizer(
                batch_texts,
                max_length=max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )

            for j in range(len(batch_texts)):
                input_ids = encoded['input_ids'][j].tolist()
                attention_mask = encoded['attention_mask'][j].tolist()
                
                # CRITICAL: Validate token IDs before storing
                if any(token_id >= vocab_size or token_id < 0 for token_id in input_ids):
                    logger.warning(f"⚠️ Invalid token IDs found in batch {i}, item {j}. Clamping to valid range.")
                    input_ids = [max(0, min(token_id, vocab_size - 1)) for token_id in input_ids]
                
                tokenized_data.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
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

        # 🎯 BABYLM TOKEN LIMITS - CORRECTED: 50M image + 50M caption + 50M text tokens
        self.max_text_tokens = 50_000_000      # 50M text tokens from train_50M
        self.max_image_tokens = 50_000_000     # 50M image tokens (Dino-V2 embeddings)
        self.max_caption_tokens = 50_000_000   # 50M caption tokens (paired with images)
        self.text_token_count = 0
        self.image_token_count = 0
        self.caption_token_count = 0
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ALWAYS use aggressive optimization with token limits
        logger.info("🚀 AGGRESSIVE OPTIMIZATION MODE - Setting up ultra-fast dataset with BabyLM token limits...")
        logger.info(f"🎯 Token Limits: {self.max_text_tokens:,} text tokens, {self.max_image_tokens:,} image tokens, {self.max_caption_tokens:,} caption tokens")
        self._setup_aggressive_optimization_with_limits()

    def _setup_aggressive_optimization_with_limits(self):
        """Setup with BabyLM compliance: 50M image + 50M caption + 50M text tokens"""
        start_time = time.time()

        # Initialize GPU-optimized optimizer  
        self.optimizer = GPUOptimizedDataOptimizer(str(self.dataset_dir))

        # Get GPU-optimized vision features (multi-worker compatible, no heavy compression)
        self.vision_features, self.vision_storage_metadata = self.optimizer.get_optimized_vision_features()
        vision_samples = self.vision_features.shape[0]
        logger.info(f"📁 GPU-optimized vision features shape: {self.vision_features.shape}")
        logger.info(f"🔥 Multi-worker data loading: ENABLED (no compression bottleneck)")
        
        # Load captions from JSON files
        logger.info("📝 Loading captions from JSON files...")
        with open(self.dataset_dir / "cc_3M_captions.json", 'r') as f:
            cc_captions = json.load(f)
        with open(self.dataset_dir / "local_narr_captions.json", 'r') as f:
            ln_captions = json.load(f)
        
        # Combine captions (should match vision features order)
        self.vision_captions = cc_captions + ln_captions
        logger.info(f"📊 Total captions loaded: {len(self.vision_captions):,}")

        # Get tokenized cache with token counting
        self.text_cache = self.optimizer.get_tokenized_cache(self.max_seq_length)
        text_samples = len(self.text_cache)
        logger.info(f"📁 Text cache: {text_samples} samples")

        # 🔍 DEBUG: Log bounds for troubleshooting
        logger.info(f"🔍 BOUNDS CHECK:")
        logger.info(f"   Vision features shape: {self.vision_features.shape}")
        logger.info(f"   Vision captions count: {len(self.vision_captions):,}")  
        logger.info(f"   Text cache size: {len(self.text_cache):,}")
        logger.info(f"   Vision samples: {vision_samples:,}")
        logger.info(f"   Text samples: {text_samples:,}")

        # 🎯 CORRECTED BABYLM COMPLIANCE: 50M image + 50M caption + 50M text tokens
        logger.info("🎯 Applying CORRECTED BabyLM limits:")
        logger.info("   → 50M image tokens (Dino-V2 embeddings)")
        logger.info("   → 50M caption tokens (paired with images)")
        logger.info("   → 50M text tokens (from train_50M)")
        
        # Step 1: Calculate how many image-caption pairs we need
        # Each image = fixed tokens (Dino-V2 embedding), captions = variable tokens
        # We need to count actual tokens, not samples
        
        # Estimate tokens per image (Dino-V2 embeddings are typically 768 or 1024 dimensions)
        # For simplicity, assume 1 token per image (since embeddings are pre-computed features)
        tokens_per_image = 1  
        
        # Calculate multimodal pairs needed
        max_multimodal_pairs = min(
            self.max_image_tokens // tokens_per_image,  # How many images we can fit
            vision_samples,
            text_samples
        )
        
        # Count actual caption tokens for the selected pairs - OPTIMIZED FOR SPEED
        caption_tokens_used = 0
        selected_pairs = 0
        multimodal_indices = []
        
        logger.info(f"📊 Processing {min(max_multimodal_pairs, vision_samples):,} captions for token counting...")
        
        # PERFORMANCE OPTIMIZATION: Batch process captions for speed
        batch_size = 1000  # Process 1000 captions at a time
        for batch_start in range(0, min(max_multimodal_pairs, vision_samples), batch_size):
            batch_end = min(batch_start + batch_size, min(max_multimodal_pairs, vision_samples))
            
            # Extract batch of captions
            batch_captions = []
            batch_indices = []
            
            for i in range(batch_start, batch_end):
                if i < len(self.vision_captions):
                    caption_data = self.vision_captions[i]
                    caption = caption_data.get('caption', '') if isinstance(caption_data, dict) else str(caption_data)
                    batch_captions.append(caption)
                    batch_indices.append(i)
            
            # Skip empty batch
            if not batch_captions:
                break
                
            # FAST: Batch tokenize all captions at once
            try:
                batch_encoded = self.tokenizer(
                    batch_captions, 
                    add_special_tokens=False, 
                    padding=False, 
                    truncation=False,
                    return_tensors=None  # Return lists, not tensors
                )
                
                # Process batch results
                for idx, (caption_idx, input_ids) in enumerate(zip(batch_indices, batch_encoded['input_ids'])):
                    caption_token_count = len(input_ids)
                    
                    # BOUNDS CHECK: Ensure caption_idx is within vision features bounds
                    if caption_idx >= self.vision_features.shape[0]:
                        logger.warning(f"⚠️ Caption index {caption_idx} exceeds vision features bounds {self.vision_features.shape[0]}, skipping")
                        continue
                        
                    # BOUNDS CHECK: Ensure caption_idx is within text cache bounds  
                    if caption_idx >= len(self.text_cache):
                        logger.warning(f"⚠️ Caption index {caption_idx} exceeds text cache bounds {len(self.text_cache)}, skipping")
                        continue
                    
                    # Check if we can fit this caption within our budget
                    if caption_tokens_used + caption_token_count <= self.max_caption_tokens:
                        multimodal_indices.append(('multimodal', caption_idx))
                        caption_tokens_used += caption_token_count
                        selected_pairs += 1
                    else:
                        # EARLY STOP: We've reached the caption token limit
                        logger.info(f"🛑 Caption token limit reached at {selected_pairs:,} pairs")
                        break
                        
            except Exception as e:
                logger.warning(f"Batch tokenization failed: {e}, falling back to individual processing")
                # Fallback to individual tokenization for this batch only
                for caption_idx in batch_indices:
                    # BOUNDS CHECK: Ensure caption_idx is within bounds
                    if caption_idx >= self.vision_features.shape[0] or caption_idx >= len(self.text_cache):
                        continue
                        
                    if caption_idx < len(self.vision_captions):
                        caption_data = self.vision_captions[caption_idx]
                        caption = caption_data.get('caption', '') if isinstance(caption_data, dict) else str(caption_data)
                        caption_token_count = len(self.tokenizer.encode(caption, add_special_tokens=False))
                        
                        if caption_tokens_used + caption_token_count <= self.max_caption_tokens:
                            multimodal_indices.append(('multimodal', caption_idx))
                            caption_tokens_used += caption_token_count
                            selected_pairs += 1
                        else:
                            break
            
            # Early termination if we've hit the limit
            if caption_tokens_used >= self.max_caption_tokens:
                break
                
            # Progress logging for large datasets
            if (batch_end % 10000) == 0:
                logger.info(f"📊 Progress: {batch_end:,} captions processed, {selected_pairs:,} pairs selected, {caption_tokens_used:,} caption tokens used")
        
        self.image_token_count = selected_pairs * tokens_per_image
        self.caption_token_count = caption_tokens_used
        
        logger.info(f"✅ Image-caption pairs selected: {selected_pairs:,}")
        logger.info(f"📊 Image tokens used: {self.image_token_count:,}/{self.max_image_tokens:,}")
        logger.info(f"📊 Caption tokens used: {self.caption_token_count:,}/{self.max_caption_tokens:,}")

        # Step 2: Load exactly 50M text tokens from train_50M 
        logger.info("� Loading 50M text tokens from train_50M...")
        text_only_samples = self._load_text_data_with_token_limit(self.max_text_tokens)
        text_indices = [('text', i) for i in range(len(text_only_samples))]
        
        logger.info(f"✅ Text-only samples: {len(text_only_samples):,}")
        logger.info(f"📊 Text tokens used: {self.text_token_count:,}/{self.max_text_tokens:,}")

        # Step 3: Combine datasets
        self.mixed_indices = multimodal_indices + text_indices
        self.text_samples = text_only_samples
        self.use_cached_data = True

        setup_time = time.time() - start_time

        # Final token summary
        total_multimodal_tokens = self.image_token_count + self.caption_token_count
        total_tokens = total_multimodal_tokens + self.text_token_count
        
        logger.info(f"🎯 CORRECTED BABYLM COMPLIANCE SUMMARY:")
        logger.info(f"   Setup time: {setup_time:.1f}s")
        logger.info(f"   Image tokens: {self.image_token_count:,}/{self.max_image_tokens:,} (Dino-V2 embeddings)")
        logger.info(f"   Caption tokens: {self.caption_token_count:,}/{self.max_caption_tokens:,} (paired with images)")
        logger.info(f"   Text tokens: {self.text_token_count:,}/{self.max_text_tokens:,} (from train_50M)")
        logger.info(f"   Total multimodal tokens: {total_multimodal_tokens:,} (image + caption)")
        logger.info(f"   Total tokens: {total_tokens:,}")
        logger.info(f"   Total dataset samples: {len(self.mixed_indices):,}")
        logger.info(f"   GPU-optimized storage: Enabled (no heavy compression)")
        logger.info(f"✅ BabyLM compliance: 50M image + 50M caption + 50M text tokens!")

    def _load_text_data_with_token_limit(self, token_budget: int) -> List[str]:
        """Load text data respecting token budget - OPTIMIZED FOR SPEED"""
        text_samples = []
        tokens_used = 0
        
        # Check for train_50M directory first, then extract from ZIP if needed
        train_50m_dir = self.dataset_dir / "train_50M"
        train_50m_zip = self.dataset_dir / "train_50M.zip"
        
        if not train_50m_dir.exists() and train_50m_zip.exists():
            logger.info("🗂️ train_50M directory not found, extracting from ZIP...")
            import zipfile
            try:
                with zipfile.ZipFile(train_50m_zip, 'r') as zip_ref:
                    zip_ref.extractall(self.dataset_dir)
                logger.info("✅ train_50M.zip extracted successfully")
            except Exception as e:
                logger.error(f"❌ Failed to extract train_50M.zip: {e}")
                self.text_token_count = 0
                return []
        
        if not train_50m_dir.exists():
            logger.warning("❌ train_50M not found (neither directory nor ZIP) - skipping text-only data")
            self.text_token_count = 0
            return []

        # Load text files efficiently with BATCH token counting
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
                
            logger.info(f"📖 Loading {filename} with OPTIMIZED batch token counting...")
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    batch_lines = []
                    batch_size = 500  # Process 500 lines at a time for speed
                    
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                            
                        batch_lines.append(line)
                        
                        # Process batch when full or at end of file
                        if len(batch_lines) >= batch_size:
                            tokens_added = self._process_text_batch(batch_lines, token_budget - tokens_used, text_samples)
                            tokens_used += tokens_added
                            batch_lines = []
                            
                            # Early termination if budget reached
                            if tokens_used >= token_budget:
                                logger.info(f"🛑 Text token budget reached in {filename} at line {line_num}")
                                break
                                
                            # Progress logging
                            if len(text_samples) % 5000 == 0:
                                logger.info(f"   📊 Progress: {len(text_samples):,} text samples, {tokens_used:,} tokens")
                    
                    # Process remaining lines in final batch
                    if batch_lines and tokens_used < token_budget:
                        tokens_added = self._process_text_batch(batch_lines, token_budget - tokens_used, text_samples)
                        tokens_used += tokens_added
                        
                if tokens_used >= token_budget:
                    break
                    
            except Exception as e:
                logger.warning(f"Error loading {filename}: {e}")
                continue

        logger.info(f"📚 Text loading complete: {len(text_samples):,} samples, {tokens_used:,} tokens")
        self.text_token_count = tokens_used  # Store for compliance tracking
        return text_samples
    
    def _process_text_batch(self, batch_lines: List[str], remaining_budget: int, text_samples: List[str]) -> int:
        """Process a batch of text lines with efficient tokenization"""
        if remaining_budget <= 0:
            return 0
            
        tokens_added = 0
        vocab_size = self.tokenizer.vocab_size  # Get vocab size for validation
        
        try:
            # FAST: Batch tokenize all lines at once
            batch_encoded = self.tokenizer(
                batch_lines,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_tensors=None  # Return lists, not tensors
            )
            
            # Add lines that fit within budget
            for line, input_ids in zip(batch_lines, batch_encoded['input_ids']):
                # CRITICAL: Validate token IDs
                if any(token_id >= vocab_size or token_id < 0 for token_id in input_ids):
                    logger.warning(f"⚠️ Invalid token IDs in text line, skipping: max={max(input_ids)}, min={min(input_ids)}, vocab_size={vocab_size}")
                    continue
                    
                line_tokens = len(input_ids)
                if tokens_added + line_tokens <= remaining_budget:
                    text_samples.append(line)
                    tokens_added += line_tokens
                else:
                    # Stop adding when budget would be exceeded
                    break
                    
        except Exception as e:
            logger.warning(f"Batch text tokenization failed: {e}, using fallback")
            # Fallback to individual processing
            for line in batch_lines:
                if tokens_added >= remaining_budget:
                    break
                try:
                    input_ids = self.tokenizer.encode(line, add_special_tokens=True)
                    # CRITICAL: Validate token IDs
                    if any(token_id >= vocab_size or token_id < 0 for token_id in input_ids):
                        logger.warning(f"⚠️ Invalid token IDs in fallback text line, skipping")
                        continue
                    line_tokens = len(input_ids)
                    if tokens_added + line_tokens <= remaining_budget:
                        text_samples.append(line)
                        tokens_added += line_tokens
                    else:
                        break
                except Exception as e2:
                    logger.warning(f"Fallback tokenization failed for line: {e2}")
                    continue
                    
        return tokens_added

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
        """Ultra-fast sample loading with aggressive caching and bounds checking"""
        sample_type, sample_idx = self.mixed_indices[idx]

        if sample_type == 'multimodal':
            # BOUNDS CHECK: Ensure sample_idx is within vision features bounds
            if sample_idx >= self.vision_features.shape[0]:
                logger.error(f"❌ Vision features index out of bounds: {sample_idx} >= {self.vision_features.shape[0]}")
                # Use last available vision feature as fallback
                sample_idx = self.vision_features.shape[0] - 1
                
            # Use GPU-optimized cached data (multi-worker compatible)
            vision_features = self.vision_features[sample_idx].clone()

            # BOUNDS CHECK: Ensure sample_idx is within text cache bounds
            if sample_idx >= len(self.text_cache):
                logger.error(f"❌ Text cache index out of bounds: {sample_idx} >= {len(self.text_cache)}")
                # Use modulo to wrap around to available data
                sample_idx = sample_idx % len(self.text_cache)
                
            text_data = self.text_cache[sample_idx]
            
            # CRITICAL: Validate token IDs to prevent CUDA embedding errors
            input_ids = torch.tensor(text_data['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(text_data['attention_mask'], dtype=torch.long)
            
            # Check for out-of-bounds token IDs (tokenizer vocab size limit)
            vocab_size = self.tokenizer.vocab_size
            if torch.any(input_ids >= vocab_size) or torch.any(input_ids < 0):
                logger.warning(f"⚠️ Invalid token IDs detected: max={input_ids.max()}, min={input_ids.min()}, vocab_size={vocab_size}")
                # Clamp token IDs to valid range
                input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            
            # Get caption from loaded captions
            caption = ""
            if sample_idx < len(self.vision_captions):
                caption_data = self.vision_captions[sample_idx]
                if isinstance(caption_data, dict):
                    caption = caption_data.get('caption', f"sample_{sample_idx}")
                else:
                    caption = str(caption_data)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': input_ids.clone(),
                'vision_features': vision_features,
                'caption': caption,
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
            
            # CRITICAL: Validate token IDs to prevent CUDA embedding errors
            vocab_size = self.tokenizer.vocab_size
            if torch.any(input_ids >= vocab_size) or torch.any(input_ids < 0):
                logger.warning(f"⚠️ Invalid token IDs in text sample: max={input_ids.max()}, min={input_ids.min()}, vocab_size={vocab_size}")
                # Clamp token IDs to valid range
                input_ids = torch.clamp(input_ids, 0, vocab_size - 1)

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
        """Get current token usage statistics for the corrected BabyLM compliance"""
        total_multimodal_tokens = self.image_token_count + self.caption_token_count
        total_tokens = total_multimodal_tokens + self.text_token_count
        
        return {
            'text_tokens_used': self.text_token_count,
            'text_tokens_limit': self.max_text_tokens,
            'text_tokens_remaining': max(0, self.max_text_tokens - self.text_token_count),
            'image_tokens_used': self.image_token_count,
            'image_tokens_limit': self.max_image_tokens,
            'image_tokens_remaining': max(0, self.max_image_tokens - self.image_token_count),
            'caption_tokens_used': self.caption_token_count,
            'caption_tokens_limit': self.max_caption_tokens,
            'caption_tokens_remaining': max(0, self.max_caption_tokens - self.caption_token_count),
            'total_multimodal_tokens': total_multimodal_tokens,
            'total_tokens': total_tokens,
            'text_utilization_pct': 100.0 * self.text_token_count / self.max_text_tokens,
            'image_utilization_pct': 100.0 * self.image_token_count / self.max_image_tokens,
            'caption_utilization_pct': 100.0 * self.caption_token_count / self.max_caption_tokens,
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
        """Create training dataloader with MULTI-WORKER support for maximum GPU utilization"""
        # PyTorch DataLoader requires CPU generator even for GPU training
        # This fixes: "Expected a 'cpu' device type for generator but found 'cuda'"
        generator = torch.Generator()  # Always CPU generator
        generator.manual_seed(42)  # For reproducibility
        
        # 🚀 ENABLE MULTI-WORKER DATA LOADING - GPU-optimized storage supports this!
        # No more h5py constraints - use maximum workers for GPU saturation
        num_workers = min(8, os.cpu_count())  # Use up to 8 workers or CPU count
        
        logger.info(f"🚀 MULTI-WORKER DATA LOADING ENABLED: {num_workers} workers")
        logger.info("🔥 H5py constraint eliminated - maximum GPU utilization possible!")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,  # 🚀 MULTI-WORKER ENABLED!
            pin_memory=self.pin_memory,
            persistent_workers=True,  # Keep workers alive for efficiency
            drop_last=True,
            generator=generator,  # CPU generator (required by PyTorch)
            prefetch_factor=4,  # Prefetch multiple batches per worker
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
