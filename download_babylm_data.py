"""
Download BabyLM Multimodal Dataset for BitMar
Downloads the complete BabyLM multimodal dataset following the official structure:
- Text-only data from train_50M.zip 
- Precomputed DiNOv2 visual embeddings + captions from Conceptual Captions 3M
"""

import os
import sys
import requests
import json
import numpy as np
import zipfile
from pathlib import Path
import logging
from tqdm import tqdm
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, filepath: str, chunk_size: int = 8192) -> bool:
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.info(f"✅ Downloaded: {filepath}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download {url}: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str) -> bool:
    """Extract ZIP file"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"✅ Extracted {zip_path} to {extract_to}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to extract {zip_path}: {e}")
        return False


def check_file_exists(filepath: str) -> bool:
    """Check if file exists and is not empty"""
    path = Path(filepath)
    return path.exists() and path.stat().st_size > 0


def verify_existing_data():
    """Verify if BabyLM data already exists in the expected location"""
    logger.info("🔍 Checking for existing BabyLM dataset...")
    
    # Check parent directory for BabyLM data
    dataset_dir = Path("../babylm_dataset")
    
    # Required multimodal files from OSF dataset
    required_files = [
        "cc_3M_captions.json",  # Conceptual Captions 3M captions
        "cc_3M_dino_v2_states_1of2.npy",  # DiNOv2 embeddings part 1
        "cc_3M_dino_v2_states_2of2.npy",  # DiNOv2 embeddings part 2
        "local_narr_captions.json",  # Localized Narratives captions
        "local_narr_dino_v2_states.npy",  # Localized Narratives DiNOv2 embeddings
    ]
    
    # Optional files
    optional_files = [
        "train_50M.zip",  # Text-only training data
        "README.pdf",  # Dataset documentation
    ]

    existing_files = []
    missing_files = []
    
    for filename in required_files:
        filepath = dataset_dir / filename
        if check_file_exists(filepath):
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"✅ {filename}: Found ({size_mb:.1f} MB)")
            existing_files.append(filename)
        else:
            logger.warning(f"❌ {filename}: Missing")
            missing_files.append(filename)
    
    # Check optional files
    for filename in optional_files:
        filepath = dataset_dir / filename
        if check_file_exists(filepath):
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"📁 {filename}: Found ({size_mb:.1f} MB)")

    if len(existing_files) == len(required_files):
        logger.info("🎉 All required multimodal dataset files found!")
        
        # Verify data integrity
        try:
            verify_core_files(dataset_dir)
            return True
                
        except Exception as e:
            logger.error(f"❌ Data integrity check failed: {e}")
            
    else:
        logger.warning(f"❌ Missing {len(missing_files)} required files: {missing_files}")
            
    return False


def download_babylm_multimodal_dataset():
    """Download BabyLM multimodal dataset from OSF"""
    logger.info("🚀 Starting BabyLM Multimodal Dataset Download")
    logger.info("=" * 60)
    
    # Target directory - external to BitMar project
    dataset_dir = Path("../babylm_dataset")
    dataset_dir.mkdir(exist_ok=True)

    # Official BabyLM multimodal dataset URL
    dataset_url = "https://files.osf.io/v1/resources/ad7qg/providers/osfstorage/6603014bb3a1e301127dfa59/?zip="
    zip_filename = dataset_dir / "babylm_multimodal.zip"

    logger.info(f"\n� Dataset directory: {dataset_dir.absolute()}")
    logger.info(f"📥 Downloading from: {dataset_url}")
    
    # Expected files after extraction
    expected_files = [
        "cc_3M_captions.json",  # 136.1 MB
        "cc_3M_dino_v2_states_1of2.npy",  # 3.5 GB  
        "cc_3M_dino_v2_states_2of2.npy",  # 3.5 GB
        "local_narr_captions.json",  # 139.2 MB
        "local_narr_dino_v2_states.npy",  # 2.4 GB
        "train_50M.zip",  # 90.2 MB
        "README.pdf"  # 63.0 kB
    ]

    logger.info("\n📋 Expected dataset files (~9.8 GB total):")
    for filename in expected_files:
        logger.info(f"   📄 {filename}")

    # Download the ZIP file
    logger.info(f"\n📥 Downloading BabyLM dataset...")
    if download_file(dataset_url, str(zip_filename)):
        logger.info("✅ Download completed successfully!")

        # Extract ZIP file
        logger.info("📦 Extracting dataset...")
        if extract_zip(str(zip_filename), str(dataset_dir)):
            logger.info("✅ Extraction completed successfully!")

            # Clean up ZIP file
            zip_filename.unlink()
            logger.info("🧹 Cleaned up ZIP file")

            # Verify extraction
            logger.info("🔍 Verifying extracted files...")
            verified_count = 0

            for filename in expected_files:
                filepath = dataset_dir / filename
                if check_file_exists(filepath):
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    logger.info(f"✅ {filename}: Verified ({size_mb:.1f} MB)")
                    verified_count += 1
                else:
                    logger.warning(f"⚠️  {filename}: Missing after extraction")

            if verified_count >= 5:  # At least the core multimodal files
                logger.info("🎉 Dataset download and setup completed!")
                logger.info(f"✅ {verified_count}/{len(expected_files)} files verified")
                
                # Additional verification for core files
                try:
                    verify_core_files(dataset_dir)
                    return True
                except Exception as e:
                    logger.error(f"❌ Core file verification failed: {e}")
                    return False
            else:
                logger.error(f"❌ Only {verified_count}/{len(expected_files)} files verified")
                return False

        else:
            logger.error("❌ Failed to extract dataset")
            return False
    else:
        logger.error("❌ Failed to download dataset")
        return False


def verify_core_files(dataset_dir: Path):
    """Verify core multimodal files"""
    logger.info("🔍 Verifying core dataset files...")
    
    # Test loading Conceptual Captions
    cc_captions_path = dataset_dir / "cc_3M_captions.json"
    with open(cc_captions_path, 'r') as f:
        cc_captions = json.load(f)
    logger.info(f"📝 Conceptual Captions: {len(cc_captions)} entries")
    
    # Test loading Localized Narratives
    ln_captions_path = dataset_dir / "local_narr_captions.json"
    with open(ln_captions_path, 'r') as f:
        ln_captions = json.load(f)
    logger.info(f"📝 Localized Narratives: {len(ln_captions)} entries")
    
    # Test vision features shapes
    cc_feat1 = np.load(dataset_dir / "cc_3M_dino_v2_states_1of2.npy", mmap_mode='r')
    cc_feat2 = np.load(dataset_dir / "cc_3M_dino_v2_states_2of2.npy", mmap_mode='r')
    ln_feat = np.load(dataset_dir / "local_narr_dino_v2_states.npy", mmap_mode='r')
    
    logger.info(f"🖼️  CC features part 1: {cc_feat1.shape}")
    logger.info(f"🖼️  CC features part 2: {cc_feat2.shape}")
    logger.info(f"🖼️  LN features: {ln_feat.shape}")
    
    # Verify alignment
    total_cc_features = cc_feat1.shape[0] + cc_feat2.shape[0]
    if total_cc_features == len(cc_captions):
        logger.info("✅ Conceptual Captions alignment verified!")
    else:
        raise ValueError(f"CC alignment error: {total_cc_features} features vs {len(cc_captions)} captions")
        
    if ln_feat.shape[0] == len(ln_captions):
        logger.info("✅ Localized Narratives alignment verified!")
    else:
        raise ValueError(f"LN alignment error: {ln_feat.shape[0]} features vs {len(ln_captions)} captions")
    
    total_samples = total_cc_features + ln_feat.shape[0]
    logger.info(f"🎯 Total multimodal samples: {total_samples:,}")
    
    return True

    # Download ZIP file
    zip_filename = dataset_dir / "babylm_dataset.zip"

    logger.info("📥 Downloading BabyLM multimodal dataset...")
    logger.info(f"   URL: {dataset_url}")
    logger.info(f"   Target: {zip_filename}")

    if download_file(dataset_url, zip_filename):
        logger.info("✅ Download completed successfully!")

        # Extract ZIP file
        logger.info("📦 Extracting dataset...")
        if extract_zip(str(zip_filename), str(dataset_dir)):
            logger.info("✅ Extraction completed successfully!")

            # Clean up ZIP file
            zip_filename.unlink()
            logger.info("🧹 Cleaned up ZIP file")

            # Verify extraction
            logger.info("🔍 Verifying extracted files...")
            verified_count = 0

            for filename in required_files:
                filepath = dataset_dir / filename
                if check_file_exists(filepath):
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    logger.info(f"✅ {filename}: Verified ({size_mb:.1f} MB)")
                    verified_count += 1
                else:
                    logger.error(f"❌ {filename}: Missing after extraction")

            if verified_count == len(required_files):
                logger.info("🎉 Dataset download and setup completed!")
                return True
            else:
                logger.error(
                    f"❌ Only {verified_count}/{len(required_files)} files verified")
                return False

        else:
            logger.error("❌ Failed to extract dataset")
            return False
    else:
        logger.error("❌ Failed to download dataset")
        logger.info("📋 Manual download instructions:")
        logger.info(f"   1. Open: {dataset_url}")
        logger.info(f"   2. Save to: {zip_filename}")
        logger.info(f"   3. Extract to: {dataset_dir}")
    
def create_test_dataset():
    """Create a small test dataset for development"""
    logger.info("🧪 Creating test dataset for development...")
    
    dataset_dir = Path("../babylm_dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    # Create test captions (100 samples)
    test_captions = [
        f"A beautiful landscape with mountains and trees in scene {i}"
        for i in range(100)
    ]
    
    captions_file = dataset_dir / "cc_3M_captions.json"
    with open(captions_file, 'w') as f:
        json.dump(test_captions, f)
    
    # Create test vision features (DiNOv2 base is 768 dimensions)
    test_features_1 = np.random.randn(50, 768).astype(np.float32)
    test_features_2 = np.random.randn(50, 768).astype(np.float32)
    
    feat1_file = dataset_dir / "cc_3M_dino_v2_states_1of2.npy"
    feat2_file = dataset_dir / "cc_3M_dino_v2_states_2of2.npy"
    
    np.save(feat1_file, test_features_1)
    np.save(feat2_file, test_features_2)
    
    logger.info(f"✅ Created test captions: {len(test_captions)} samples")
    logger.info(f"✅ Created test features: {test_features_1.shape} + {test_features_2.shape}")
    logger.info("🧪 Test dataset ready for development!")
    
    return True


def main():
    """Main function to download or verify BabyLM dataset"""
    logger.info("🚀 BabyLM Dataset Setup for BitMar")
    logger.info("=" * 50)
    
    # First check if data already exists
    if verify_existing_data():
        logger.info("✅ Dataset ready for training!")
        return True
    
    # Try to download from official sources
    logger.info("\n📥 Attempting dataset download...")
    success = download_babylm_multimodal_dataset()
    
    if not success:
        logger.info("\n🧪 Creating test dataset for development...")
        create_test_dataset()
        logger.info("\n⚠️  Using test data - replace with real BabyLM dataset for final training")
        
    return True


if __name__ == "__main__":
    main()
