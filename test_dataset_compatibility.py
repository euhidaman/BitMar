"""
Dataset Compatibility Test for BitMar
Tests BabyLM dataset loading and preprocessing
"""

from src.dataset import BabyLMMultimodalDataset, create_data_module
import sys
import logging
import json
import numpy as np
import torch
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dataset_files():
    """Test if dataset files exist and are readable"""
    logger.info("Testing dataset file availability...")

    dataset_files = {
        'captions': "../babylm_dataset/cc_3M_captions.json",
        'vision_1': "../babylm_dataset/cc_3M_dino_v2_states_1of2.npy",
        'vision_2': "../babylm_dataset/cc_3M_dino_v2_states_2of2.npy"
    }

    results = {}

    for name, file_path in dataset_files.items():
        path = Path(file_path)
        exists = path.exists()
        results[name] = exists

        if exists:
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ {name}: Found ({size_mb:.1f} MB)")
        else:
            logger.warning(f"❌ {name}: Not found at {file_path}")

    return results


def test_captions_file():
    """Test captions file structure and content"""
    logger.info("Testing captions file...")

    captions_file = "../babylm_dataset/cc_3M_captions.json"

    if not Path(captions_file).exists():
        logger.error(f"❌ Captions file not found: {captions_file}")
        return False

    try:
        # Load and examine captions
        with open(captions_file, 'r', encoding='utf-8') as f:
            captions = json.load(f)

        # Check structure
        if isinstance(captions, list):
            num_captions = len(captions)
            sample_captions = captions[:5]
        elif isinstance(captions, dict):
            if 'captions' in captions:
                captions_list = captions['captions']
                num_captions = len(captions_list)
                sample_captions = captions_list[:5]
            else:
                captions_list = list(captions.values())
                num_captions = len(captions_list)
                sample_captions = captions_list[:5]
        else:
            logger.error(f"❌ Unexpected captions format: {type(captions)}")
            return False

        # Analyze captions
        if sample_captions:
            avg_length = np.mean([len(caption.split())
                                 for caption in sample_captions])
            max_length = max([len(caption.split())
                             for caption in sample_captions])

            logger.info(f"✅ Captions loaded successfully!")
            logger.info(f"   Total captions: {num_captions:,}")
            logger.info(f"   Average length (words): {avg_length:.1f}")
            logger.info(f"   Max length in sample: {max_length}")
            logger.info(f"   Sample captions:")
            for i, caption in enumerate(sample_captions):
                logger.info(f"     {i+1}: {caption[:100]}...")

        return True

    except Exception as e:
        logger.error(f"❌ Error loading captions: {e}")
        return False


def test_vision_features():
    """Test vision features files"""
    logger.info("Testing vision features...")

    vision_files = [
        "../babylm_dataset/cc_3M_dino_v2_states_1of2.npy",
        "../babylm_dataset/cc_3M_dino_v2_states_2of2.npy"
    ]

    features_list = []

    for i, file_path in enumerate(vision_files):
        if not Path(file_path).exists():
            logger.error(f"❌ Vision features file not found: {file_path}")
            return False

        try:
            # Load features
            features = np.load(file_path)
            features_list.append(features)

            logger.info(f"✅ Vision features {i+1} loaded:")
            logger.info(f"   Shape: {features.shape}")
            logger.info(f"   Dtype: {features.dtype}")
            logger.info(
                f"   Range: [{features.min():.3f}, {features.max():.3f}]")
            logger.info(f"   Mean: {features.mean():.3f}")
            logger.info(f"   Std: {features.std():.3f}")

        except Exception as e:
            logger.error(f"❌ Error loading vision features {i+1}: {e}")
            return False

    # Test concatenation
    try:
        combined_features = np.concatenate(features_list, axis=0)
        logger.info(f"✅ Combined vision features:")
        logger.info(f"   Combined shape: {combined_features.shape}")
        logger.info(
            f"   Memory usage: {combined_features.nbytes / (1024**2):.1f} MB")

        return True

    except Exception as e:
        logger.error(f"❌ Error combining vision features: {e}")
        return False


def test_dataset_creation():
    """Test BabyLM dataset creation"""
    logger.info("Testing dataset creation...")

    try:
        # Create dataset with small sample
        dataset = BabyLMMultimodalDataset(
            captions_file="../babylm_dataset/cc_3M_captions.json",
            vision_features_1="../babylm_dataset/cc_3M_dino_v2_states_1of2.npy",
            vision_features_2="../babylm_dataset/cc_3M_dino_v2_states_2of2.npy",
            tokenizer_name="bert-base-uncased",
            max_seq_length=256,
            split="train",
            train_ratio=0.95,
            max_samples=100  # Small sample for testing
        )

        logger.info(f"✅ Dataset created successfully!")
        logger.info(f"   Dataset length: {len(dataset)}")

        # Test data access
        sample = dataset[0]

        logger.info(f"   Sample keys: {list(sample.keys())}")
        logger.info(f"   Input IDs shape: {sample['input_ids'].shape}")
        logger.info(
            f"   Attention mask shape: {sample['attention_mask'].shape}")
        logger.info(f"   Labels shape: {sample['labels'].shape}")
        logger.info(
            f"   Vision features shape: {sample['vision_features'].shape}")
        logger.info(f"   Caption: {sample['caption'][:100]}...")

        # Test multiple samples
        samples = [dataset[i] for i in range(min(5, len(dataset)))]
        logger.info(f"✅ Successfully accessed {len(samples)} samples")

        return True

    except Exception as e:
        logger.error(f"❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_module():
    """Test data module functionality"""
    logger.info("Testing data module...")

    try:
        config = {
            'captions_file': "../babylm_dataset/cc_3M_captions.json",
            'vision_features_1': "../babylm_dataset/cc_3M_dino_v2_states_1of2.npy",
            'vision_features_2': "../babylm_dataset/cc_3M_dino_v2_states_2of2.npy",
            'text_encoder_name': "bert-base-uncased",
            'max_seq_length': 256,
            'train_split': 0.95,
            'val_split': 0.05,
            'batch_size': 4,
            'num_workers': 0,
            'pin_memory': False,
        }

        # Create data module
        data_module = create_data_module(config)
        data_module.setup(max_samples=200)  # Small sample

        logger.info(f"✅ Data module created successfully!")
        logger.info(f"   Train dataset size: {len(data_module.train_dataset)}")
        logger.info(f"   Val dataset size: {len(data_module.val_dataset)}")

        # Test data loaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        logger.info(f"   Train batches: {len(train_loader)}")
        logger.info(f"   Val batches: {len(val_loader)}")

        # Test batch loading
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        logger.info(f"✅ Batch loading successful!")
        logger.info(
            f"   Train batch input shape: {train_batch['input_ids'].shape}")
        logger.info(
            f"   Train batch vision shape: {train_batch['vision_features'].shape}")
        logger.info(
            f"   Val batch input shape: {val_batch['input_ids'].shape}")
        logger.info(
            f"   Val batch vision shape: {val_batch['vision_features'].shape}")

        return True

    except Exception as e:
        logger.error(f"❌ Data module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_consistency():
    """Test data consistency between captions and vision features"""
    logger.info("Testing data consistency...")

    try:
        # Load captions count
        with open("../babylm_dataset/cc_3M_captions.json", 'r') as f:
            captions = json.load(f)

        if isinstance(captions, list):
            num_captions = len(captions)
        elif isinstance(captions, dict):
            if 'captions' in captions:
                num_captions = len(captions['captions'])
            else:
                num_captions = len(list(captions.values()))

        # Load vision features count
        features_1 = np.load("../babylm_dataset/cc_3M_dino_v2_states_1of2.npy")
        features_2 = np.load("../babylm_dataset/cc_3M_dino_v2_states_2of2.npy")
        num_vision_features = len(features_1) + len(features_2)

        logger.info(f"   Captions count: {num_captions:,}")
        logger.info(f"   Vision features count: {num_vision_features:,}")

        # Check consistency
        min_count = min(num_captions, num_vision_features)
        max_count = max(num_captions, num_vision_features)

        if min_count == max_count:
            logger.info(f"✅ Perfect alignment: {min_count:,} samples")
        else:
            logger.warning(
                f"⚠️  Count mismatch: will use {min_count:,} samples")
            logger.info(f"   Difference: {max_count - min_count:,} samples")

        return True

    except Exception as e:
        logger.error(f"❌ Data consistency check failed: {e}")
        return False


def test_tokenization():
    """Test tokenization quality"""
    logger.info("Testing tokenization...")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Sample captions
        test_captions = [
            "A cat sitting on a windowsill",
            "Beautiful sunset over the mountains with orange and pink colors in the sky",
            "Children playing in a park with green grass and blue sky",
            "A red sports car driving on a highway",
            "Fresh vegetables and fruits arranged on a wooden table"
        ]

        for i, caption in enumerate(test_captions):
            encoded = tokenizer(
                caption,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)

            # Decode to verify
            decoded = tokenizer.decode(input_ids, skip_special_tokens=True)

            logger.info(
                f"   Caption {i+1}: {len(caption.split())} words -> {attention_mask.sum().item()} tokens")
            logger.info(f"      Original: {caption}")
            logger.info(f"      Decoded:  {decoded}")

        logger.info(f"✅ Tokenization test successful!")
        return True

    except Exception as e:
        logger.error(f"❌ Tokenization test failed: {e}")
        return False


def run_dataset_compatibility_test():
    """Run complete dataset compatibility test"""
    logger.info("🚀 Starting BitMar Dataset Compatibility Test")
    logger.info("=" * 60)

    total_start_time = time.time()
    tests_passed = 0
    total_tests = 6

    try:
        # Test 1: File Availability
        logger.info("\n1️⃣ Testing Dataset File Availability")
        file_results = test_dataset_files()
        if all(file_results.values()):
            tests_passed += 1
            logger.info("✅ All dataset files found")
        else:
            logger.error("❌ Some dataset files missing")
            logger.info(
                "Please ensure BabyLM dataset is downloaded to ../babylm_dataset/")
            return False

        # Test 2: Captions File
        logger.info("\n2️⃣ Testing Captions File")
        if test_captions_file():
            tests_passed += 1

        # Test 3: Vision Features
        logger.info("\n3️⃣ Testing Vision Features")
        if test_vision_features():
            tests_passed += 1

        # Test 4: Data Consistency
        logger.info("\n4️⃣ Testing Data Consistency")
        if test_data_consistency():
            tests_passed += 1

        # Test 5: Dataset Creation
        logger.info("\n5️⃣ Testing Dataset Creation")
        if test_dataset_creation():
            tests_passed += 1

        # Test 6: Data Module
        logger.info("\n6️⃣ Testing Data Module")
        if test_data_module():
            tests_passed += 1

        # Bonus Test: Tokenization
        logger.info("\n🔤 Testing Tokenization (Bonus)")
        test_tokenization()

    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

    total_time = time.time() - total_start_time

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("🎯 Dataset Compatibility Test Results")
    logger.info("=" * 60)
    logger.info(f"Tests passed: {tests_passed}/{total_tests}")
    logger.info(f"Total test time: {total_time:.2f}s")

    if tests_passed == total_tests:
        logger.info("🎉 All tests passed! Dataset is compatible with BitMar.")
        logger.info("\n📋 Dataset Summary:")
        logger.info("- ✅ Captions: CC3M dataset with image descriptions")
        logger.info("- ✅ Vision: DiNOv2 features (768D embeddings)")
        logger.info("- ✅ Format: Compatible with BitMar multimodal training")
        logger.info("\n🚀 Ready for training!")
        logger.info("Next: Run 'python train_bitmar.py' for full training")
    else:
        logger.warning(f"⚠️  {total_tests - tests_passed} tests failed.")
        logger.info("Please check dataset files and try again.")

    return tests_passed == total_tests


if __name__ == "__main__":
    success = run_dataset_compatibility_test()

    if success:
        exit(0)
    else:
        exit(1)
