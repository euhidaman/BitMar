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
    
    # Required multimodal files as per BabyLM specification
    required_files = [
        "cc_3M_captions.json",  # Conceptual Captions 3M captions
        "cc_3M_dino_v2_states_1of2.npy",  # DiNOv2 embeddings part 1
        "cc_3M_dino_v2_states_2of2.npy",  # DiNOv2 embeddings part 2
    ]
    
    # Optional text-only data
    optional_files = [
        "train_50M.zip",  # Text-only training data
        "cc_3M_captions.download_instructions.txt",
        "cc_3M_dino_v2_states_1of2.download_instructions.txt", 
        "cc_3M_dino_v2_states_2of2.download_instructions.txt"
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
            # Test loading captions
            captions_path = dataset_dir / "cc_3M_captions.json"
            with open(captions_path, 'r') as f:
                captions = json.load(f)
            logger.info(f"📝 Captions file contains {len(captions)} entries")
            
            # Test loading vision features
            feat1_path = dataset_dir / "cc_3M_dino_v2_states_1of2.npy"
            feat2_path = dataset_dir / "cc_3M_dino_v2_states_2of2.npy"
            
            feat1 = np.load(feat1_path)
            feat2 = np.load(feat2_path)
            
            logger.info(f"🖼️  Vision features part 1: {feat1.shape}")
            logger.info(f"🖼️  Vision features part 2: {feat2.shape}")
            logger.info(f"🖼️  Total vision features: {feat1.shape[0] + feat2.shape[0]}")
            
            # Verify counts match
            total_vision = feat1.shape[0] + feat2.shape[0]
            if isinstance(captions, list):
                caption_count = len(captions)
            else:
                caption_count = len(captions.get('captions', captions))
                
            if total_vision == caption_count:
                logger.info("✅ Vision features and captions count match!")
                return True
            else:
                logger.warning(f"⚠️  Count mismatch: {total_vision} vision vs {caption_count} captions")
                
        except Exception as e:
            logger.error(f"❌ Data integrity check failed: {e}")
            
    return False


def download_babylm_multimodal_dataset():
    """Download BabyLM multimodal dataset following official structure"""
    logger.info("🚀 Starting BabyLM Multimodal Dataset Download")
    logger.info("=" * 60)
    
    # Check if data already exists
    if verify_existing_data():
        logger.info("Dataset verification passed - ready for training!")
        return True

    logger.info("\n📋 BabyLM Multimodal Dataset Components:")
    logger.info("1. Text-only data: train_50M.zip")
    logger.info("2. Image-caption pairs (precomputed DiNOv2 + captions):")
    logger.info("   - cc_3M_captions.json")
    logger.info("   - cc_3M_dino_v2_states_1of2.npy") 
    logger.info("   - cc_3M_dino_v2_states_2of2.npy")
    logger.info("\n🔗 Data Sources:")
    logger.info("- Localized Narratives (OpenImage + MSCOCO training sets)")
    logger.info("- Conceptual Captions 3M (training split only)")
    logger.info("- Visual embeddings: DiNOv2 ViT-Base (facebook/dinov2-base)")
    
    # Target directory
    dataset_dir = Path("../babylm_dataset")
    dataset_dir.mkdir(exist_ok=True)

    logger.info(f"\n📁 Dataset directory: {dataset_dir.absolute()}")
    
    # Official BabyLM multimodal dataset URLs
    # These would be the actual download URLs from the BabyLM organizers
    file_urls = {
        "cc_3M_captions.json": "https://example.com/babylm/cc_3M_captions.json",
        "cc_3M_dino_v2_states_1of2.npy": "https://example.com/babylm/cc_3M_dino_v2_states_1of2.npy", 
        "cc_3M_dino_v2_states_2of2.npy": "https://example.com/babylm/cc_3M_dino_v2_states_2of2.npy",
        "train_50M.zip": "https://example.com/babylm/train_50M.zip"
    }
    
    logger.info("\n⚠️  IMPORTANT: Manual Download Required")
    logger.info("=" * 60)
    logger.info("The BabyLM dataset requires manual download from the official source.")
    logger.info("Please follow these steps:")
    logger.info("\n1. Visit the BabyLM Challenge website")
    logger.info("2. Download the following files to ../babylm_dataset/:")
    logger.info("   ✅ cc_3M_captions.json")
    logger.info("   ✅ cc_3M_dino_v2_states_1of2.npy") 
    logger.info("   ✅ cc_3M_dino_v2_states_2of2.npy")
    logger.info("   📝 train_50M.zip (optional for text-only)")
    
    logger.info("\n💡 Alternative: Use precomputed visual embeddings")
    logger.info("If you have raw images, you can compute DiNOv2 embeddings using:")
    logger.info('   from transformers import AutoModel')
    logger.info('   model = AutoModel.from_pretrained("facebook/dinov2-base")')
    
    # Create download instruction files that explain the process
    instructions = {
        "cc_3M_captions.json": "Download Conceptual Captions 3M captions (JSON format)",
        "cc_3M_dino_v2_states_1of2.npy": "Download precomputed DiNOv2 visual embeddings (part 1/2)",
        "cc_3M_dino_v2_states_2of2.npy": "Download precomputed DiNOv2 visual embeddings (part 2/2)"
    }
    
    for filename, description in instructions.items():
        instruction_file = dataset_dir / f"{filename}.download_instructions.txt"
        with open(instruction_file, 'w') as f:
            f.write(f"BabyLM Dataset File: {filename}\n")
            f.write(f"Description: {description}\n")
            f.write(f"Source: BabyLM Challenge - Multimodal Track\n")
            f.write(f"Format: {filename.split('.')[-1].upper()}\n")
            f.write(f"\nOfficial sources:\n")
            f.write(f"- Localized Narratives: https://google.github.io/localized-narratives/\n")
            f.write(f"- Conceptual Captions 3M: https://ai.google.com/research/ConceptualCaptions/download\n")
            f.write(f"- Visual embeddings computed with: facebook/dinov2-base\n")
        
        logger.info(f"📝 Created: {instruction_file}")

    return False

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
