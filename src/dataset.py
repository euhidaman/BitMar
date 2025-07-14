"""
Dataset processing for BitMar
Handles BabyLM multimodal dataset (text captions + DiNOv2 features)
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


class BabyLMMultimodalDataset(Dataset):
    """BabyLM Multimodal Dataset for BitMar training"""

    def __init__(
        self,
        captions_file: str,
        vision_features_1: str,
        vision_features_2: str,
        tokenizer_name: str = "bert-base-uncased",
        max_seq_length: int = 512,
        split: str = "train",
        train_ratio: float = 0.95,
        max_samples: Optional[int] = None
    ):
        self.captions_file = captions_file
        self.vision_features_1 = vision_features_1
        self.vision_features_2 = vision_features_2
        self.max_seq_length = max_seq_length
        self.split = split
        self.train_ratio = train_ratio

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load data
        self.captions = self._load_captions()
        self.vision_features = self._load_vision_features()

        # Create train/validation split
        self._create_split()

        # Limit samples if specified
        if max_samples is not None:
            self.indices = self.indices[:max_samples]

        logger.info(f"Loaded {len(self.indices)} samples for {split} split")

    def _load_captions(self) -> List[str]:
        """Load captions from JSON file"""
        logger.info(f"Loading captions from {self.captions_file}")

        try:
            with open(self.captions_file, 'r', encoding='utf-8') as f:
                captions = json.load(f)

            # Handle different JSON formats
            if isinstance(captions, list):
                return captions
            elif isinstance(captions, dict):
                # If it's a dict, extract values or specific key
                if 'captions' in captions:
                    return captions['captions']
                else:
                    return list(captions.values())
            else:
                raise ValueError(
                    f"Unexpected captions format: {type(captions)}")

        except Exception as e:
            logger.error(f"Error loading captions: {e}")
            raise

    def _load_vision_features(self) -> np.ndarray:
        """Load and concatenate vision features from two files"""
        logger.info(
            f"Loading vision features from {self.vision_features_1} and {self.vision_features_2}")

        try:
            # Load both parts
            features_1 = np.load(self.vision_features_1)
            features_2 = np.load(self.vision_features_2)

            # Concatenate along first dimension
            features = np.concatenate([features_1, features_2], axis=0)

            logger.info(f"Loaded vision features with shape: {features.shape}")
            return features

        except Exception as e:
            logger.error(f"Error loading vision features: {e}")
            raise

    def _create_split(self):
        """Create train/validation split"""
        total_samples = min(len(self.captions), len(self.vision_features))

        # Create indices
        indices = list(range(total_samples))
        random.seed(42)  # For reproducible splits
        random.shuffle(indices)

        # Split indices
        split_point = int(total_samples * self.train_ratio)

        if self.split == "train":
            self.indices = indices[:split_point]
        else:  # validation
            self.indices = indices[split_point:]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample"""
        actual_idx = self.indices[idx]

        # Get caption and vision features
        caption = self.captions[actual_idx]
        vision_feature = self.vision_features[actual_idx]

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
            'vision_features': torch.tensor(vision_feature, dtype=torch.float32),
            'caption': caption,
            'index': actual_idx
        }


class BabyLMDataModule:
    """Data module for BitMar training"""

    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer_name = config.get(
            'text_encoder_name', 'bert-base-uncased')

        # Dataset parameters
        self.captions_file = config['captions_file']
        self.vision_features_1 = config['vision_features_1']
        self.vision_features_2 = config['vision_features_2']
        self.max_seq_length = config['max_seq_length']
        self.train_split = config['train_split']
        self.val_split = config['val_split']

        # DataLoader parameters
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.pin_memory = config['pin_memory']
        self.persistent_workers = config.get('persistent_workers', True)

        # Datasets
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, max_samples: Optional[int] = None):
        """Setup train and validation datasets"""
        logger.info("Setting up datasets...")

        # Create train dataset
        self.train_dataset = BabyLMMultimodalDataset(
            captions_file=self.captions_file,
            vision_features_1=self.vision_features_1,
            vision_features_2=self.vision_features_2,
            tokenizer_name=self.tokenizer_name,
            max_seq_length=self.max_seq_length,
            split="train",
            train_ratio=self.train_split,
            max_samples=max_samples
        )

        # Create validation dataset
        self.val_dataset = BabyLMMultimodalDataset(
            captions_file=self.captions_file,
            vision_features_1=self.vision_features_1,
            vision_features_2=self.vision_features_2,
            tokenizer_name=self.tokenizer_name,
            max_seq_length=self.max_seq_length,
            split="val",
            train_ratio=self.train_split,
            max_samples=max_samples // 10 if max_samples else None
        )

        logger.info(f"Train dataset: {len(self.train_dataset)} samples")
        logger.info(f"Validation dataset: {len(self.val_dataset)} samples")

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

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=False
        )

    def get_sample_batch(self, split: str = "train", num_samples: int = 4) -> Dict[str, torch.Tensor]:
        """Get a sample batch for testing"""
        dataset = self.train_dataset if split == "train" else self.val_dataset

        if dataset is None:
            raise ValueError("Dataset not setup. Call setup() first.")

        # Get random samples
        indices = random.sample(range(len(dataset)),
                                min(num_samples, len(dataset)))
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
    logger.info("Testing dataset...")

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
    logger.info(
        f"Batch vision features shape: {batch['vision_features'].shape}")
    logger.info(f"Number of captions in batch: {len(batch['caption'])}")

    logger.info("Dataset test completed successfully!")

    return data_module


if __name__ == "__main__":
    # Test configuration
    test_config = {
        'captions_file': "../babylm_dataset/cc_3M_captions.json",
        'vision_features_1': "../babylm_dataset/cc_3M_dino_v2_states_1of2.npy",
        'vision_features_2': "../babylm_dataset/cc_3M_dino_v2_states_2of2.npy",
        'text_encoder_name': "bert-base-uncased",
        'max_seq_length': 512,
        'train_split': 0.95,
        'val_split': 0.05,
        'batch_size': 4,
        'num_workers': 0,
        'pin_memory': False,
    }

    # Test dataset
    test_dataset(test_config)
