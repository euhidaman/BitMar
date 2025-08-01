"""
Dataset processing for BitMar
Handles complete BabyLM multimodal dataset with intelligent optimization
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

# Import the intelligent optimizer
from .dataset_optimizer import IntelligentDatasetOptimizer, OptimizedDataLoader

logger = logging.getLogger(__name__)


class MixedMultimodalTextDataset(Dataset):
    """Mixed dataset combining multimodal data and text-only training from train_50M with intelligent optimization"""

    def __init__(
        self,
        dataset_dir: str,
        tokenizer_name: str = "gpt2",
        max_seq_length: int = 512,
        split: str = "train",
        max_samples: Optional[int] = None,
        text_ratio: float = 0.3,  # 30% text-only, 70% multimodal
        load_text_data: bool = True,
        config: Dict = None  # NEW: Add config for optimizations
    ):
        self.dataset_dir = Path(dataset_dir)
        self.max_seq_length = max_seq_length
        self.split = split
        self.text_ratio = text_ratio
        self.load_text_data = load_text_data
        self.config = config or {}

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check if intelligent preprocessing is enabled
        quick_mode = self.config.get('quick_training_mode', {})
        intelligent_preprocessing = quick_mode.get('intelligent_preprocessing', {})

        if intelligent_preprocessing.get('enabled', False):
            logger.info("🚀 INTELLIGENT PREPROCESSING ENABLED - Setting up optimized dataset...")
            self._setup_optimized_dataset()
        else:
            logger.info("Using standard dataset loading...")
            self._setup_standard_dataset()

    def _setup_optimized_dataset(self):
        """Setup dataset with intelligent preprocessing and caching"""
        logger.info("🔧 Setting up intelligent dataset optimization...")

        # Initialize optimizer
        optimizer_config = {
            'vision_compression_ratio': self.config.get('quick_training_mode', {}).get('intelligent_preprocessing', {}).get('vision_compression_ratio', 8),
            'max_seq_length': self.max_seq_length,
            'aggressive_pooling': self.config.get('quick_training_mode', {}).get('intelligent_preprocessing', {}).get('aggressive_spatial_pooling', True),
            'mixed_precision': self.config.get('quick_training_mode', {}).get('intelligent_preprocessing', {}).get('mixed_precision_cache', True),
            'text_ratio': self.text_ratio
        }

        self.optimizer = IntelligentDatasetOptimizer(
            dataset_dir=str(self.dataset_dir),
            config=optimizer_config
        )

        # Run preprocessing (only happens once)
        logger.info("🏗️ Running intelligent preprocessing (this may take 2-3 hours on first run)...")

        # Precompute vision features
        vision_cache_file = self.optimizer.precompute_and_cache_vision_features()
        logger.info(f"✅ Vision features cached: {vision_cache_file}")

        # Precompute text tokenization
        text_cache_file = self.optimizer.precompute_text_tokenization()
        logger.info(f"✅ Text tokenization cached: {text_cache_file}")

        # Create optimized indices
        indices_cache_file = self.optimizer.create_optimized_dataset_indices()
        logger.info(f"✅ Optimized indices created: {indices_cache_file}")

        # Load cached data
        self._load_cached_data()

        # Log optimization summary
        summary = self.optimizer.get_optimization_summary()
        logger.info(f"🎯 Optimization Summary:")
        for key, value in summary.items():
            logger.info(f"   {key}: {value}")

    def _load_cached_data(self):
        """Load preprocessed cached data for lightning-fast training"""
        cache_dir = self.optimizer.cache_dir

        # Load compressed vision features
        self.vision_cache = h5py.File(cache_dir / "compressed_vision_features.h5", 'r')
        logger.info(f"📁 Loaded compressed vision cache: {self.vision_cache['compressed_features'].shape}")

        # Load tokenized text
        with open(cache_dir / "tokenized_text.pkl", 'rb') as f:
            self.text_cache = pickle.load(f)
        logger.info(f"📁 Loaded tokenized text cache: {len(self.text_cache)} samples")

        # Load optimized indices
        with open(cache_dir / "optimized_indices.pkl", 'rb') as f:
            self.mixed_indices = pickle.load(f)
        logger.info(f"📁 Loaded optimized indices: {len(self.mixed_indices)} mixed samples")

        # Set flags for optimized mode
        self.use_cached_data = True
        self.multimodal_indices = [i for sample_type, i in self.mixed_indices if sample_type == 'multimodal']
        self.text_samples = [i for sample_type, i in self.mixed_indices if sample_type == 'text_only']

    def _setup_standard_dataset(self):
        """Setup dataset with standard loading (slower but no preprocessing)"""
        self.use_cached_data = False

        # Load multimodal data
        self._load_multimodal_data()

        # Load text-only data from train_50M
        if self.load_text_data and self.split == "train":
            self._load_text_data()
        else:
            self.text_samples = []

        # Create combined indices
        self._create_mixed_indices()

    def _load_multimodal_data(self):
        """Load multimodal data (existing implementation)"""
        logger.info("Loading multimodal data...")

        # Load Conceptual Captions 3M
        cc_captions_file = self.dataset_dir / "cc_3M_captions.json"
        cc_feat1_file = self.dataset_dir / "cc_3M_dino_v2_states_1of2.npy"
        cc_feat2_file = self.dataset_dir / "cc_3M_dino_v2_states_2of2.npy"

        with open(cc_captions_file, 'r', encoding='utf-8') as f:
            cc_captions = json.load(f)

        cc_feat1 = np.load(cc_feat1_file, mmap_mode='r')
        cc_feat2 = np.load(cc_feat2_file, mmap_mode='r')
        cc_features = VisionFeaturesConcatenated(cc_feat1, cc_feat2)

        # Load Localized Narratives
        ln_captions_file = self.dataset_dir / "local_narr_captions.json"
        ln_feat_file = self.dataset_dir / "local_narr_dino_v2_states.npy"

        with open(ln_captions_file, 'r', encoding='utf-8') as f:
            ln_captions = json.load(f)

        ln_features = np.load(ln_feat_file, mmap_mode='r')

        # Combine multimodal data
        self.multimodal_captions = cc_captions + ln_captions
        self.multimodal_features = CombinedVisionFeatures(cc_features, ln_features)
        self.multimodal_indices = list(range(len(self.multimodal_captions)))

        logger.info(f"Loaded {len(self.multimodal_captions)} multimodal samples")

    def _load_text_data(self):
        """Load text-only data from train_50M"""
        logger.info("Loading train_50M text data...")

        self.text_samples = []
        train_50m_dir = self.dataset_dir / "train_50M"

        if not train_50m_dir.exists():
            logger.warning("train_50M directory not found - extracting if needed...")
            from download_babylm_data import extract_train_50M_if_needed
            extract_result = extract_train_50M_if_needed(self.dataset_dir)
            if not extract_result:
                logger.warning("Could not extract train_50M - skipping text-only training")
                return

        # Expected train_50M files
        text_files = [
            "bnc_spoken.train",
            "childes.train",
            "gutenberg.train",
            "open_subtitles.train",
            "simple_wiki.train",
            "switchboard.train"
        ]

        total_lines = 0
        for filename in text_files:
            filepath = train_50m_dir / filename
            if filepath.exists():
                logger.info(f"Loading {filename}...")
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    self.text_samples.extend(lines)
                    total_lines += len(lines)
                    logger.info(f"  Added {len(lines)} lines from {filename}")
            else:
                logger.warning(f"Text file not found: {filepath}")

        logger.info(f"Loaded {total_lines} text-only samples from train_50M")

    def _create_mixed_indices(self):
        """Create mixed indices for multimodal and text-only samples"""
        num_multimodal = len(self.multimodal_indices)
        num_text = len(self.text_samples)

        if num_text == 0:
            # No text data, use only multimodal
            self.mixed_indices = [('multimodal', idx) for idx in self.multimodal_indices]
            return

        # Calculate how many of each type we want
        total_desired = num_multimodal + num_text
        num_text_desired = int(total_desired * self.text_ratio)
        num_multimodal_desired = total_desired - num_text_desired

        # Create indices with proper mixing
        text_indices = [('text', idx) for idx in range(min(num_text, num_text_desired))]
        multimodal_indices = [('multimodal', idx) for idx in range(min(num_multimodal, num_multimodal_desired))]

        # Combine and shuffle
        self.mixed_indices = text_indices + multimodal_indices
        random.seed(42)
        random.shuffle(self.mixed_indices)

    def __len__(self) -> int:
        return len(self.mixed_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample (either multimodal or text-only)"""
        sample_type, sample_idx = self.mixed_indices[idx]

        if sample_type == 'multimodal':
            return self._get_multimodal_sample(sample_idx)
        else:
            return self._get_text_sample(sample_idx)

    def _get_multimodal_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a multimodal sample"""
        caption = self.multimodal_captions[idx]
        vision_feature = self.multimodal_features[idx]

        # Tokenize caption
        encoded = self.tokenizer(
            caption,
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
            'vision_features': torch.tensor(vision_feature.copy(), dtype=torch.float32),
            'caption': caption,
            'sample_type': 'multimodal',
            'index': idx
        }

    def _get_text_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a text-only sample"""
        text = self.text_samples[idx]

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

        # Create dummy vision features (zeros) for consistency
        dummy_vision = torch.zeros(768, dtype=torch.float32)  # DiNOv2 feature size

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'vision_features': dummy_vision,
            'caption': text,
            'sample_type': 'text_only',
            'index': idx
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

        # Dataset parameters
        self.dataset_dir = config['dataset_dir']
        self.max_seq_length = config['max_seq_length']
        self.hf_token = config.get('hf_token') or os.getenv('HF_TOKEN', '')

        # Mixed training parameters
        self.use_mixed_training = config.get('use_mixed_training', False)
        self.text_ratio = config.get('text_ratio', 0.3)

        # DataLoader parameters
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.pin_memory = config['pin_memory']
        self.persistent_workers = config.get('persistent_workers', True)

        # Validation datasets
        self.validation_datasets = config.get('validation_datasets', [
            'ewok-core/ewok-core-1.0',
            'facebook/winoground'
        ])

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
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=True
        )

    def val_dataloader(self) -> List[DataLoader]:
        """Create validation dataloaders"""
        val_loaders = []
        for name, dataset in self.val_datasets.items():
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,  # Use 0 for validation to avoid memory issues
                pin_memory=self.pin_memory,
                drop_last=False
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
        return DataLoader(dummy_dataset, batch_size=self.batch_size, shuffle=False)

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
    """Custom collate function for batching"""
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        if key == 'caption':
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = torch.stack([item[key] for item in batch])

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
