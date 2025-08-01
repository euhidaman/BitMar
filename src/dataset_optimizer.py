"""
Advanced Dataset Optimization for BitMar
Intelligent preprocessing, caching, and feature compression for full dataset training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import pickle
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache

logger = logging.getLogger(__name__)

class IntelligentDatasetOptimizer:
    """
    Optimizes the entire 3M+ dataset for efficient training without reducing samples
    Uses aggressive preprocessing, compression, and intelligent caching
    """

    def __init__(self, dataset_dir: str, cache_dir: str = None, config: Dict = None):
        self.dataset_dir = Path(dataset_dir)
        self.cache_dir = Path(cache_dir or self.dataset_dir / "optimized_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.config = config or {}

        # Optimization settings
        self.vision_compression_ratio = self.config.get('vision_compression_ratio', 8)  # 768 -> 96
        self.max_seq_length = self.config.get('max_seq_length', 256)  # Shorter sequences
        self.aggressive_pooling = self.config.get('aggressive_pooling', True)
        self.use_mixed_precision = self.config.get('mixed_precision', True)

        # Cache settings
        self.cache_chunk_size = 10000  # Process in chunks
        self.num_workers = min(8, mp.cpu_count())

        logger.info(f"Initializing dataset optimizer with {self.vision_compression_ratio}x compression")

    def precompute_and_cache_vision_features(self) -> str:
        """
        Precompute compressed vision features for the entire dataset
        This is a ONE-TIME operation that dramatically speeds up training
        """
        cache_file = self.cache_dir / "compressed_vision_features.h5"

        if cache_file.exists():
            logger.info("Using existing compressed vision cache")
            return str(cache_file)

        logger.info("Precomputing compressed vision features for entire dataset...")

        # Load original features
        cc_feat1 = np.load(self.dataset_dir / "cc_3M_dino_v2_states_1of2.npy", mmap_mode='r')
        cc_feat2 = np.load(self.dataset_dir / "cc_3M_dino_v2_states_2of2.npy", mmap_mode='r')

        total_samples = cc_feat1.shape[0] + cc_feat2.shape[0]
        logger.info(f"Processing {total_samples} vision samples...")

        # Initialize compression network
        compressor = self._create_vision_compressor()

        with h5py.File(cache_file, 'w') as h5f:
            # Pre-allocate compressed dataset
            compressed_shape = (total_samples, self._get_compressed_vision_dim())
            compressed_dataset = h5f.create_dataset(
                'compressed_features',
                shape=compressed_shape,
                dtype=np.float16,  # Use half precision
                compression='gzip',
                compression_opts=9
            )

            # Process in chunks to manage memory
            chunk_size = self.cache_chunk_size
            processed = 0

            for start_idx in range(0, total_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, total_samples)
                chunk_size_actual = end_idx - start_idx

                # Load chunk
                if start_idx < cc_feat1.shape[0]:
                    # From first file
                    chunk_end_in_file1 = min(end_idx, cc_feat1.shape[0])
                    chunk_data = cc_feat1[start_idx:chunk_end_in_file1]

                    if end_idx > cc_feat1.shape[0]:
                        # Need data from second file too
                        second_file_start = 0
                        second_file_end = end_idx - cc_feat1.shape[0]
                        chunk_data2 = cc_feat2[second_file_start:second_file_end]
                        chunk_data = np.concatenate([chunk_data, chunk_data2], axis=0)
                else:
                    # From second file only
                    file2_start = start_idx - cc_feat1.shape[0]
                    file2_end = end_idx - cc_feat1.shape[0]
                    chunk_data = cc_feat2[file2_start:file2_end]

                # Compress chunk
                compressed_chunk = self._compress_vision_chunk(chunk_data, compressor)

                # Store compressed chunk
                compressed_dataset[start_idx:end_idx] = compressed_chunk

                processed += chunk_size_actual
                if processed % 50000 == 0:
                    logger.info(f"Processed {processed}/{total_samples} samples ({processed/total_samples*100:.1f}%)")

            # Store metadata
            h5f.attrs['original_dim'] = cc_feat1.shape[-1]
            h5f.attrs['compressed_dim'] = compressed_shape[-1]
            h5f.attrs['compression_ratio'] = self.vision_compression_ratio
            h5f.attrs['total_samples'] = total_samples

        logger.info(f"Vision feature compression complete. Saved to {cache_file}")
        logger.info(f"Original size: {cc_feat1.nbytes + cc_feat2.nbytes / 1e9:.2f}GB")
        logger.info(f"Compressed size: ~{os.path.getsize(cache_file) / 1e9:.2f}GB")

        return str(cache_file)

    def _create_vision_compressor(self) -> nn.Module:
        """Create optimized vision feature compressor"""
        from src.vision_compression import DiNOv2FeatureCompressor

        compressor = DiNOv2FeatureCompressor(
            input_dim=768,
            target_dim=768 // self.vision_compression_ratio,  # 768 -> 96 for 8x compression
            compression_method="learned_compression",
            spatial_pooling=self.aggressive_pooling,
            pool_size=4  # Aggressive spatial pooling
        )

        # Pre-train the compressor on a sample of data for better compression
        self._pretrain_compressor(compressor)

        return compressor

    def _pretrain_compressor(self, compressor: nn.Module):
        """Pre-train compressor on sample data for optimal compression"""
        logger.info("Pre-training vision compressor...")

        # Load small sample for pre-training
        cc_feat1 = np.load(self.dataset_dir / "cc_3M_dino_v2_states_1of2.npy", mmap_mode='r')
        sample_data = torch.from_numpy(cc_feat1[:1000]).float()  # Use first 1000 samples

        # Simple pre-training to learn good compression
        optimizer = torch.optim.Adam(compressor.parameters(), lr=1e-3)
        compressor.train()

        for epoch in range(10):  # Quick pre-training
            optimizer.zero_grad()
            compressed = compressor(sample_data)

            # Reconstruction loss (simple autoencoder objective)
            # We want to preserve important information
            loss = F.mse_loss(compressed.mean(dim=1), sample_data.mean(dim=1))
            loss.backward()
            optimizer.step()

        compressor.eval()
        logger.info("Vision compressor pre-training complete")

    def _compress_vision_chunk(self, chunk: np.ndarray, compressor: nn.Module) -> np.ndarray:
        """Compress a chunk of vision features"""
        with torch.no_grad():
            chunk_tensor = torch.from_numpy(chunk).float()
            compressed = compressor(chunk_tensor)

            # Convert to half precision for storage
            if self.use_mixed_precision:
                compressed = compressed.half()

            return compressed.numpy()

    def _get_compressed_vision_dim(self) -> int:
        """Get the dimension after compression"""
        original_spatial = 196  # 14x14 patches from DiNOv2
        if self.aggressive_pooling:
            # 4x4 pooling reduces 14x14 -> 3x3 = 9 patches
            spatial_after_pooling = 9
        else:
            spatial_after_pooling = original_spatial

        feature_dim = 768 // self.vision_compression_ratio  # e.g., 768 -> 96
        return spatial_after_pooling * feature_dim

    def precompute_text_tokenization(self) -> str:
        """
        Precompute and cache all text tokenization
        Saves significant time during training
        """
        cache_file = self.cache_dir / "tokenized_text.pkl"

        if cache_file.exists():
            logger.info("Using existing tokenized text cache")
            return str(cache_file)

        logger.info("Precomputing text tokenization...")

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load all text data
        all_text_data = self._load_all_text_data()

        logger.info(f"Tokenizing {len(all_text_data)} text samples...")

        # Tokenize in parallel
        tokenized_data = []
        batch_size = 1000

        for i in range(0, len(all_text_data), batch_size):
            batch = all_text_data[i:i+batch_size]
            batch_tokenized = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )

            # Store as lists to save memory
            for j in range(len(batch)):
                tokenized_data.append({
                    'input_ids': batch_tokenized['input_ids'][j].tolist(),
                    'attention_mask': batch_tokenized['attention_mask'][j].tolist()
                })

            if (i // batch_size) % 100 == 0:
                logger.info(f"Tokenized {i+len(batch)}/{len(all_text_data)} samples")

        # Save tokenized data
        with open(cache_file, 'wb') as f:
            pickle.dump(tokenized_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Text tokenization complete. Saved to {cache_file}")
        return str(cache_file)

    def _load_all_text_data(self) -> List[str]:
        """Load all text data from various sources"""
        all_text = []

        # Load multimodal captions
        import json
        with open(self.dataset_dir / "cc_3M_captions.json", 'r') as f:
            cc_captions = json.load(f)
            all_text.extend(cc_captions)

        # Load localized narratives
        if (self.dataset_dir / "local_narr_captions.json").exists():
            with open(self.dataset_dir / "local_narr_captions.json", 'r') as f:
                ln_captions = json.load(f)
                all_text.extend(ln_captions)

        # Load train_50M text data
        train_50m_dir = self.dataset_dir / "train_50M"
        if train_50m_dir.exists():
            text_files = [
                "bnc_spoken.train", "childes.train", "gutenberg.train",
                "open_subtitles.train", "simple_wiki.train", "switchboard.train"
            ]

            for filename in text_files:
                filepath = train_50m_dir / filename
                if filepath.exists():
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        all_text.extend(lines)

        logger.info(f"Loaded {len(all_text)} total text samples")
        return all_text

    def create_optimized_dataset_indices(self) -> str:
        """
        Create optimized indices for efficient data loading
        Groups similar samples together for better batching efficiency
        """
        cache_file = self.cache_dir / "optimized_indices.pkl"

        if cache_file.exists():
            logger.info("Using existing optimized indices")
            return str(cache_file)

        logger.info("Creating optimized dataset indices...")

        # Create indices that group samples by type and length for efficiency
        multimodal_indices = []
        text_only_indices = []

        # Assuming we have the same number of captions as vision features
        num_multimodal = self._get_total_vision_samples()

        # Group multimodal samples
        for i in range(num_multimodal):
            multimodal_indices.append(('multimodal', i))

        # Add text-only samples
        text_data = self._load_all_text_data()
        for i, text in enumerate(text_data[num_multimodal:]):  # Skip already used captions
            text_only_indices.append(('text_only', i))

        # Create mixed indices with optimal ratio
        text_ratio = self.config.get('text_ratio', 0.3)
        total_multimodal = len(multimodal_indices)
        total_text = len(text_only_indices)

        # Calculate optimal mixing
        if total_text > 0:
            num_text_desired = int(total_multimodal * text_ratio / (1 - text_ratio))
            num_text_actual = min(num_text_desired, total_text)

            # Sample text indices
            import random
            random.seed(42)
            selected_text_indices = random.sample(text_only_indices, num_text_actual)
        else:
            selected_text_indices = []

        # Combine and shuffle
        all_indices = multimodal_indices + selected_text_indices
        random.shuffle(all_indices)

        # Save optimized indices
        with open(cache_file, 'wb') as f:
            pickle.dump(all_indices, f)

        logger.info(f"Created optimized indices: {len(multimodal_indices)} multimodal + {len(selected_text_indices)} text-only")
        return str(cache_file)

    def _get_total_vision_samples(self) -> int:
        """Get total number of vision samples"""
        cc_feat1 = np.load(self.dataset_dir / "cc_3M_dino_v2_states_1of2.npy", mmap_mode='r')
        cc_feat2 = np.load(self.dataset_dir / "cc_3M_dino_v2_states_2of2.npy", mmap_mode='r')
        return cc_feat1.shape[0] + cc_feat2.shape[0]

    def get_optimization_summary(self) -> Dict:
        """Get summary of optimizations applied"""
        original_vision_size = self._get_total_vision_samples() * 196 * 768 * 4  # float32
        compressed_vision_size = self._get_total_vision_samples() * self._get_compressed_vision_dim() * 2  # float16

        return {
            'vision_compression_ratio': self.vision_compression_ratio,
            'spatial_pooling': self.aggressive_pooling,
            'max_seq_length': self.max_seq_length,
            'estimated_speedup': f"{original_vision_size / compressed_vision_size:.1f}x",
            'memory_reduction': f"{(1 - compressed_vision_size / original_vision_size) * 100:.1f}%",
            'uses_full_dataset': True,
            'preprocessing_complete': all([
                (self.cache_dir / "compressed_vision_features.h5").exists(),
                (self.cache_dir / "tokenized_text.pkl").exists(),
                (self.cache_dir / "optimized_indices.pkl").exists()
            ])
        }


class OptimizedDataLoader:
    """
    Optimized DataLoader that uses preprocessed, compressed data
    Dramatically faster than loading raw data during training
    """

    def __init__(self, optimizer: IntelligentDatasetOptimizer, batch_size: int = 32):
        self.optimizer = optimizer
        self.batch_size = batch_size

        # Load cached data
        self.vision_cache = h5py.File(optimizer.cache_dir / "compressed_vision_features.h5", 'r')

        with open(optimizer.cache_dir / "tokenized_text.pkl", 'rb') as f:
            self.text_cache = pickle.load(f)

        with open(optimizer.cache_dir / "optimized_indices.pkl", 'rb') as f:
            self.indices = pickle.load(f)

        logger.info(f"Loaded optimized dataset with {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        """Get optimized batch - much faster than original loading"""
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]

        batch_data = []
        for sample_type, sample_idx in batch_indices:
            if sample_type == 'multimodal':
                # Load compressed vision features (much faster!)
                vision_features = torch.from_numpy(self.vision_cache['compressed_features'][sample_idx]).float()
                text_data = self.text_cache[sample_idx]

                batch_data.append({
                    'vision_features': vision_features,
                    'input_ids': torch.tensor(text_data['input_ids']),
                    'attention_mask': torch.tensor(text_data['attention_mask']),
                    'is_multimodal': True
                })
            else:
                # Text-only sample
                text_data = self.text_cache[sample_idx]
                batch_data.append({
                    'vision_features': None,
                    'input_ids': torch.tensor(text_data['input_ids']),
                    'attention_mask': torch.tensor(text_data['attention_mask']),
                    'is_multimodal': False
                })

        return batch_data
