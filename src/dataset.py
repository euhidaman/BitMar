"""
Dataset processing for BitMar
Handles complete BabyLM multimodal dataset (Conceptual Captions + Localized Narratives)
+ Human-inspired learning with train_50M text-only data
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
import zipfile

logger = logging.getLogger(__name__)


def extract_train_50M_if_needed(dataset_dir: Path):
    """Extract train_50M.zip if not already extracted"""
    zip_path = dataset_dir / "train_50M.zip"
    extract_path = dataset_dir / "train_50M"

    if not zip_path.exists():
        logger.warning(f"train_50M.zip not found at {zip_path}. Text-only training will be skipped.")
        return None

    if extract_path.exists() and any(extract_path.iterdir()):
        logger.info(f"✅ train_50M already extracted at {extract_path}")
        return extract_path

    logger.info(f"📦 Extracting train_50M.zip to {extract_path}...")
    extract_path.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path.parent)

    logger.info(f"✅ Successfully extracted train_50M.zip!")
    return extract_path


class CompleteBabyLMDataset(Dataset):
    """Complete BabyLM Multimodal Dataset (CC3M + Localized Narratives)"""

    def __init__(
        self,
        dataset_dir: str,
        tokenizer_name: str = "gpt2",
        max_seq_length: int = 512,
        split: str = "train",
        max_samples: Optional[int] = None
    ):
        self.dataset_dir = Path(dataset_dir)
        self.max_seq_length = max_seq_length
        self.split = split

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load all data sources
        self._load_all_data()
        
        # Create indices for this split (train uses full dataset, no validation split)
        if split == "train":
            self.indices = list(range(len(self.all_captions)))
        else:
            # For validation, we'll use HuggingFace datasets
            self.indices = []

        # Limit samples if specified
        if max_samples is not None and len(self.indices) > max_samples:
            random.seed(42)
            self.indices = random.sample(self.indices, max_samples)

        logger.info(f"Loaded {len(self.indices)} samples for {split} split")

    def _load_all_data(self):
        """Load all multimodal data sources with enhanced error handling"""
        logger.info("Loading complete BabyLM multimodal dataset...")

        # Load Conceptual Captions 3M
        cc_captions_file = self.dataset_dir / "cc_3M_captions.json"
        cc_feat1_file = self.dataset_dir / "cc_3M_dino_v2_states_1of2.npy"
        cc_feat2_file = self.dataset_dir / "cc_3M_dino_v2_states_2of2.npy"

        # Validate files exist
        missing_files = []
        for file_path in [cc_captions_file, cc_feat1_file, cc_feat2_file]:
            if not file_path.exists():
                missing_files.append(str(file_path))

        if missing_files:
            error_msg = f"Missing required files: {missing_files}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Load captions with error handling
        try:
            with open(cc_captions_file, 'r', encoding='utf-8') as f:
                cc_captions = json.load(f)

            if not isinstance(cc_captions, list):
                raise ValueError(f"Expected list of captions, got {type(cc_captions)}")

            logger.info(f"Loaded {len(cc_captions)} CC captions")

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Error loading CC captions: {e}")
            raise ValueError(f"Corrupted captions file: {cc_captions_file}")

        # Load features with enhanced error handling
        try:
            cc_feat1 = np.load(cc_feat1_file, mmap_mode='r')
            cc_feat2 = np.load(cc_feat2_file, mmap_mode='r')

            # Validate feature shapes
            expected_dim = 768  # DiNOv2 dimension
            if cc_feat1.shape[1] != expected_dim:
                logger.warning(f"Unexpected CC feat1 dimension: {cc_feat1.shape[1]}, expected {expected_dim}")
            if cc_feat2.shape[1] != expected_dim:
                logger.warning(f"Unexpected CC feat2 dimension: {cc_feat2.shape[1]}, expected {expected_dim}")

            cc_features = VisionFeaturesConcatenated(cc_feat1, cc_feat2)
            logger.info(f"Loaded CC features: {len(cc_features)} samples, dim={cc_feat1.shape[1]}")

        except Exception as e:
            logger.error(f"Error loading CC features: {e}")
            raise RuntimeError(f"Failed to load CC vision features: {e}")

        # Load Localized Narratives
        ln_captions_file = self.dataset_dir / "local_narr_captions.json"
        ln_feat_file = self.dataset_dir / "local_narr_dino_v2_states.npy"

        # Validate LN files exist
        missing_ln_files = []
        for file_path in [ln_captions_file, ln_feat_file]:
            if not file_path.exists():
                missing_ln_files.append(str(file_path))

        if missing_ln_files:
            error_msg = f"Missing LN files: {missing_ln_files}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Load LN captions with error handling
        try:
            with open(ln_captions_file, 'r', encoding='utf-8') as f:
                ln_captions = json.load(f)

            if not isinstance(ln_captions, list):
                raise ValueError(f"Expected list of LN captions, got {type(ln_captions)}")

            logger.info(f"Loaded {len(ln_captions)} LN captions")

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Error loading LN captions: {e}")
            raise ValueError(f"Corrupted LN captions file: {ln_captions_file}")

        # Load LN features with error handling
        try:
            ln_features = np.load(ln_feat_file, mmap_mode='r')

            # Validate LN feature shape
            if ln_features.shape[1] != expected_dim:
                logger.warning(f"Unexpected LN feature dimension: {ln_features.shape[1]}, expected {expected_dim}")

            logger.info(f"Loaded LN features: {ln_features.shape[0]} samples, dim={ln_features.shape[1]}")

        except Exception as e:
            logger.error(f"Error loading LN features: {e}")
            raise RuntimeError(f"Failed to load LN vision features: {e}")

        # Combine all data
        self.all_captions = cc_captions + ln_captions
        self.all_features = CombinedVisionFeatures(cc_features, ln_features)

        logger.info(f"Total multimodal samples: {len(self.all_captions)}")

        # Enhanced alignment verification with detailed checks
        expected_features = len(self.all_captions)
        actual_features = len(self.all_features)

        if expected_features != actual_features:
            error_msg = f"CRITICAL: Data alignment error - {expected_features} captions vs {actual_features} features"
            logger.error(error_msg)
            logger.error(f"CC captions: {len(cc_captions)}, CC features: {len(cc_features)}")
            logger.error(f"LN captions: {len(ln_captions)}, LN features: {ln_features.shape[0]}")
            raise ValueError(error_msg)

        # Additional validation - sample a few indices to verify alignment
        try:
            test_indices = [0, len(self.all_captions) // 2, len(self.all_captions) - 1]
            for idx in test_indices:
                if idx < len(self.all_captions):
                    caption = self.all_captions[idx]
                    feature = self.all_features[idx]

                    if not isinstance(caption, str) or len(caption.strip()) == 0:
                        logger.warning(f"Invalid caption at index {idx}: {type(caption)}")

                    if feature.shape[0] != expected_dim:
                        logger.warning(f"Invalid feature shape at index {idx}: {feature.shape}")

            logger.info("✅ Data alignment and integrity verification passed")

        except Exception as e:
            logger.error(f"Error during data validation: {e}")
            raise RuntimeError(f"Data validation failed: {e}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample"""
        actual_idx = self.indices[idx]

        # Get caption and vision features
        caption = self.all_captions[actual_idx]
        vision_feature = self.all_features[actual_idx]

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

        # Create labels for text generation (shifted input_ids)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding tokens in loss

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'vision_features': torch.tensor(vision_feature.copy(), dtype=torch.float32),
            'caption': caption,
            'index': actual_idx
        }


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
        logger.info("Setting up complete BabyLM dataset...")

        # Create training dataset (uses complete BabyLM data)
        self.train_dataset = CompleteBabyLMDataset(
            dataset_dir=self.dataset_dir,
            tokenizer_name=self.tokenizer_name,
            max_seq_length=self.max_seq_length,
            split="train",
            max_samples=max_samples
        )

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


class TextOnlyDataset(Dataset):
    """Dataset for Stage 3: Pure text learning from train_50M"""

    def __init__(
        self,
        dataset_dir: str,
        tokenizer_name: str = "gpt2",
        max_seq_length: int = 256,
        max_samples: Optional[int] = None
    ):
        self.dataset_dir = Path(dataset_dir)
        self.max_seq_length = max_seq_length

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Extract train_50M.zip if needed
        train_50m_dir = extract_train_50M_if_needed(self.dataset_dir)
        if train_50m_dir is None:
            logger.warning("train_50M data not available. Creating empty dataset.")
            self.texts = []
            return

        # Load all text files
        text_files = [
            train_50m_dir / 'childes.train',
            train_50m_dir / 'gutenberg.train',
            train_50m_dir / 'open_subtitles.train',
            train_50m_dir / 'simple_wiki.train',
            train_50m_dir / 'bnc_spoken.train',
            train_50m_dir / 'switchboard.train'
        ]

        self.texts = []
        for text_file in text_files:
            if text_file.exists():
                logger.info(f"Loading {text_file.name}...")
                with open(text_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    self.texts.extend([line.strip() for line in lines if line.strip()])
            else:
                logger.warning(f"Text file not found: {text_file}")

        # Limit samples if specified
        if max_samples is not None and len(self.texts) > max_samples:
            random.seed(42)
            self.texts = random.sample(self.texts, max_samples)

        logger.info(f"Loaded {len(self.texts)} text samples for Stage 3 training")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

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
            'vision_features': torch.zeros(768, dtype=torch.float32),  # No vision for text-only
            'caption': text,
            'index': idx
        }


class VisualOnlyDataset(Dataset):
    """Dataset for Stage 1: Visual-only learning"""

    def __init__(
        self,
        dataset_dir: str,
        max_samples: Optional[int] = None
    ):
        self.dataset_dir = Path(dataset_dir)

        # Load vision features from both datasets
        cc_feat1_file = self.dataset_dir / "cc_3M_dino_v2_states_1of2.npy"
        cc_feat2_file = self.dataset_dir / "cc_3M_dino_v2_states_2of2.npy"
        ln_feat_file = self.dataset_dir / "local_narr_dino_v2_states.npy"

        # Load features
        cc_feat1 = np.load(cc_feat1_file, mmap_mode='r') if cc_feat1_file.exists() else None
        cc_feat2 = np.load(cc_feat2_file, mmap_mode='r') if cc_feat2_file.exists() else None
        ln_features = np.load(ln_feat_file, mmap_mode='r') if ln_feat_file.exists() else None

        # Combine features
        all_features = []
        if cc_feat1 is not None:
            all_features.append(cc_feat1)
        if cc_feat2 is not None:
            all_features.append(cc_feat2)
        if ln_features is not None:
            all_features.append(ln_features)

        if not all_features:
            raise FileNotFoundError("No vision feature files found!")

        # Create combined features object
        if len(all_features) == 1:
            self.vision_features = all_features[0]
        else:
            # Concatenate multiple feature arrays
            cc_combined = VisionFeaturesConcatenated(cc_feat1, cc_feat2) if cc_feat1 is not None and cc_feat2 is not None else (cc_feat1 or cc_feat2)
            self.vision_features = CombinedVisionFeatures(cc_combined, ln_features)

        # Create indices
        self.indices = list(range(len(self.vision_features)))
        if max_samples is not None and len(self.indices) > max_samples:
            random.seed(42)
            self.indices = random.sample(self.indices, max_samples)

        logger.info(f"Loaded {len(self.indices)} vision samples for Stage 1 training")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        vision_feature = self.vision_features[actual_idx]

        return {
            'vision_features': torch.tensor(vision_feature.copy(), dtype=torch.float32),
            'index': actual_idx
        }


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
