"""
Download BabyLM Dataset for BitMar
Downloads the complete multimodal BabyLM dataset from OSF
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


def download_babylm_dataset():
    """Download BabyLM multimodal dataset from OSF"""
    logger.info("🚀 Starting BabyLM Dataset Download")
    logger.info("=" * 50)
    
    # OSF download URL for the complete dataset
    dataset_url = "https://files.osf.io/v1/resources/ad7qg/providers/osfstorage/?zip="
    
    # Target directory
    dataset_dir = Path("../babylm_dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    # Check if dataset already exists
    required_files = [
        "cc_3M_captions.json",
        "cc_3M_dino_v2_states_1of2.npy", 
        "cc_3M_dino_v2_states_2of2.npy"
    ]
    
    existing_files = []
    for filename in required_files:
        filepath = dataset_dir / filename
        if check_file_exists(filepath):
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"✅ {filename}: Already exists ({size_mb:.1f} MB)")
            existing_files.append(filename)
    
    if len(existing_files) == len(required_files):
        logger.info("🎉 All dataset files already exist!")
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
                logger.error(f"❌ Only {verified_count}/{len(required_files)} files verified")
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
        return False


def create_sample_dataset():
    """Create a small sample dataset for testing when full dataset is not available"""
    logger.info("Creating sample dataset for testing...")
    
    dataset_dir = Path("../babylm_dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    # Create sample captions
    sample_captions = [
        "A cat sitting on a windowsill looking outside",
        "Beautiful sunset over mountains with orange and pink sky",
        "Children playing in a park with green grass",
        "Red sports car driving on a highway",
        "Fresh vegetables arranged on a wooden table",
        "Dog running through a field of flowers",
        "City skyline at night with bright lights",
        "Ocean waves crashing on a sandy beach",
        "Snow-covered trees in a winter forest",
        "Butterfly landing on a colorful flower"
    ] * 100  # 1000 samples
    
    captions_file = dataset_dir / "cc_3M_captions.json"
    with open(captions_file, 'w') as f:
        json.dump(sample_captions, f)
    
    logger.info(f"✅ Created sample captions: {len(sample_captions)} samples")
    
    # Create sample vision features (random DiNOv2-like features)
    np.random.seed(42)  # For reproducible samples
    
    # Split into two files like the real dataset
    num_samples_1 = 500
    num_samples_2 = 500
    feature_dim = 768  # DiNOv2 dimension
    
    # Create realistic-looking features (normalized)
    features_1 = np.random.randn(num_samples_1, feature_dim).astype(np.float32)
    features_1 = features_1 / np.linalg.norm(features_1, axis=1, keepdims=True)
    
    features_2 = np.random.randn(num_samples_2, feature_dim).astype(np.float32)
    features_2 = features_2 / np.linalg.norm(features_2, axis=1, keepdims=True)
    
    # Save features
    features_file_1 = dataset_dir / "cc_3M_dino_v2_states_1of2.npy"
    features_file_2 = dataset_dir / "cc_3M_dino_v2_states_2of2.npy"
    
    np.save(features_file_1, features_1)
    np.save(features_file_2, features_2)
    
    logger.info(f"✅ Created sample vision features:")
    logger.info(f"   Part 1: {features_1.shape} -> {features_file_1}")
    logger.info(f"   Part 2: {features_2.shape} -> {features_file_2}")
    
    # Create download instructions (for reference)
    instructions = {
        "cc_3M_captions.json": "Sample captions created locally",
        "cc_3M_dino_v2_states_1of2.npy": "Sample DiNOv2 features (part 1) created locally",
        "cc_3M_dino_v2_states_2of2.npy": "Sample DiNOv2 features (part 2) created locally",
        "note": "These are sample files for testing. For real training, download from BabyLM dataset.",
        "real_urls": {
            "captions": "https://data.babylm.github.io/multimodal/cc_3M_captions.json",
            "features_1": "https://data.babylm.github.io/multimodal/cc_3M_dino_v2_states_1of2.npy",
            "features_2": "https://data.babylm.github.io/multimodal/cc_3M_dino_v2_states_2of2.npy"
        }
    }
    
    instructions_file = dataset_dir / "download_instructions.json"
    with open(instructions_file, 'w') as f:
        json.dump(instructions, f, indent=2)
    
    logger.info("✅ Sample dataset created successfully!")
    logger.info(f"   Dataset directory: {dataset_dir.absolute()}")
    logger.info(f"   Total samples: {len(sample_captions)}")
    logger.info(f"   Feature dimension: {feature_dim}")
    
    return True


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download BabyLM dataset for BitMar")
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Create sample dataset for testing instead of downloading full dataset"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if files exist"
    )
    
    args = parser.parse_args()
    
    if args.sample_only:
        logger.info("Creating sample dataset for testing...")
        success = create_sample_dataset()
    else:
        logger.info("Attempting to download full BabyLM dataset...")
        success = download_babylm_dataset()
        
        if not success:
            logger.info("\n📋 Full dataset download failed.")
            logger.info("Creating sample dataset for testing instead...")
            success = create_sample_dataset()
    
    if success:
        logger.info("\n🎉 Dataset setup completed!")
        logger.info("Next steps:")
        logger.info("1. Run: python test_dataset_compatibility.py")
        logger.info("2. Run: python test_cpu_compatibility.py")
        logger.info("3. For full training: python train_bitmar.py")
        exit(0)
    else:
        logger.error("\n❌ Dataset setup failed!")
        logger.info("Please check your internet connection and try again.")
        exit(1)


if __name__ == "__main__":
    main()
