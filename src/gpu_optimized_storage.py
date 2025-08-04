"""
GPU-Optimized Storage System to Replace H5py Bottleneck
Eliminates single-threaded data loading constraint for maximum GPU utilization
"""

import torch
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
import os

logger = logging.getLogger(__name__)


class GPUOptimizedVisionStorage:
    """
    Replaces H5py with pickle-compatible storage for multi-worker data loading
    Enables maximum GPU utilization by removing h5py multiprocessing constraints
    """
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.cache_dir = self.dataset_dir / "gpu_optimized_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
    def convert_from_h5py_to_gpu_optimized(self, force_recreate: bool = False):
        """Convert existing h5py cache to GPU-optimized format"""
        
        # Check if already converted
        optimized_file = self.cache_dir / "vision_features_optimized.pt"
        metadata_file = self.cache_dir / "vision_metadata.pkl"
        
        if optimized_file.exists() and metadata_file.exists() and not force_recreate:
            logger.info("🚀 GPU-optimized vision cache already exists")
            return str(optimized_file), str(metadata_file)
            
        logger.info("🔥 Converting H5py cache to GPU-optimized format...")
        start_time = time.time()
        
        # Load from existing h5py cache (temporarily)
        import h5py
        h5_cache_file = self.dataset_dir / "speed_cache" / "ultra_compressed_vision.h5"
        
        if not h5_cache_file.exists():
            logger.error(f"H5py cache not found: {h5_cache_file}")
            raise FileNotFoundError("H5py cache must be created first")
            
        # Load and convert
        with h5py.File(h5_cache_file, 'r') as h5f:
            # Load all data into memory (this is the key difference)
            features_data = h5f['features'][:]  # Load everything into RAM
            total_samples = features_data.shape[0]
            compressed_dim = features_data.shape[1]
            
            # Get metadata
            compression_ratio = h5f.attrs.get('compression_ratio', 2350.0)
            original_size_gb = h5f.attrs.get('original_size_gb', 49.0)
            
            logger.info(f"📊 Converting {total_samples} samples, {compressed_dim}D features")
            
        # Convert to PyTorch tensor (GPU-transferable)
        features_tensor = torch.from_numpy(features_data).float()
        
        # Save as PyTorch tensor (pickle-compatible)
        logger.info("💾 Saving GPU-optimized tensor...")
        torch.save(features_tensor, optimized_file)
        
        # Save metadata separately (also pickle-compatible)
        metadata = {
            'total_samples': total_samples,
            'compressed_dim': compressed_dim,
            'compression_ratio': compression_ratio,
            'original_size_gb': original_size_gb,
            'optimized_size_gb': os.path.getsize(optimized_file) / 1e9,
            'created_time': time.time()
        }
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
            
        conversion_time = time.time() - start_time
        logger.info(f"✅ GPU-optimized conversion complete in {conversion_time:.1f}s")
        logger.info(f"📈 Now supports multi-worker data loading (was single-threaded)")
        logger.info(f"💾 Size: {metadata['optimized_size_gb']:.2f}GB")
        
        return str(optimized_file), str(metadata_file)
    
    def load_optimized_features(self) -> Tuple[torch.Tensor, Dict]:
        """Load GPU-optimized features for multi-worker access"""
        optimized_file = self.cache_dir / "vision_features_optimized.pt"
        metadata_file = self.cache_dir / "vision_metadata.pkl"
        
        if not optimized_file.exists():
            logger.info("GPU-optimized cache not found, converting...")
            self.convert_from_h5py_to_gpu_optimized()
            
        # Load tensor (this is fast and pickle-compatible)
        features = torch.load(optimized_file, map_location='cpu')
        
        # Load metadata
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
            
        logger.info(f"🚀 Loaded GPU-optimized features: {features.shape}")
        logger.info(f"📈 Multi-worker data loading now ENABLED!")
        
        return features, metadata


class GPUOptimizedDataset:
    """
    Drop-in replacement for H5py-based dataset
    Enables multi-worker data loading for maximum GPU utilization
    """
    
    def __init__(self, dataset_dir: str, **kwargs):
        self.dataset_dir = Path(dataset_dir)
        
        # Initialize GPU-optimized storage
        self.storage = GPUOptimizedVisionStorage(dataset_dir)
        
        # Load features into memory (enables multi-worker access)
        self.vision_features, self.metadata = self.storage.load_optimized_features()
        
        logger.info("🚀 GPU-optimized dataset initialized")
        logger.info(f"📊 Features shape: {self.vision_features.shape}")
        logger.info(f"🔥 Multi-worker loading: ENABLED")
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Fast feature access (pickle-compatible for multi-worker)"""
        return self.vision_features[idx]
        
    def __len__(self) -> int:
        return self.vision_features.shape[0]
        
    def get_feature_dim(self) -> int:
        return self.vision_features.shape[1]


def test_gpu_optimized_storage():
    """Test the GPU-optimized storage system"""
    logger.info("🧪 Testing GPU-optimized storage...")
    
    # Test conversion
    storage = GPUOptimizedVisionStorage("../babylm_dataset")
    optimized_file, metadata_file = storage.convert_from_h5py_to_gpu_optimized(force_recreate=True)
    
    # Test loading
    dataset = GPUOptimizedDataset("../babylm_dataset")
    
    # Test multi-worker compatibility
    import pickle
    try:
        # This should work now (unlike h5py)
        pickled_data = pickle.dumps(dataset.vision_features)
        logger.info("✅ Pickle compatibility test: PASSED")
    except Exception as e:
        logger.error(f"❌ Pickle compatibility test: FAILED - {e}")
        
    logger.info("🧪 GPU-optimized storage test complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_gpu_optimized_storage()
