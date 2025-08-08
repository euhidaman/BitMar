"""
Extract and Setup Evaluation Data
Handles extraction of compressed files in the evaluation data directories
"""

import os
import sys
import zipfile
import logging
from pathlib import Path
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_zip_files_in_directory(directory: Path, cleanup: bool = True):
    """Extract all ZIP files in a directory"""
    extracted_count = 0

    for zip_file in directory.rglob("*.zip"):
        logger.info(f"📦 Found ZIP file: {zip_file}")

        # Determine extraction directory (same directory as ZIP file)
        extract_dir = zip_file.parent / zip_file.stem

        try:
            # Create extraction directory
            extract_dir.mkdir(exist_ok=True)

            # Extract ZIP file
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            logger.info(f"✅ Extracted {zip_file.name} to {extract_dir.name}")
            extracted_count += 1

            # List extracted contents
            extracted_files = list(extract_dir.iterdir())
            logger.info(f"   📁 Extracted {len(extracted_files)} items:")
            for item in extracted_files[:5]:  # Show first 5 items
                if item.is_file():
                    size_kb = item.stat().st_size / 1024
                    logger.info(f"     📄 {item.name} ({size_kb:.1f} KB)")
                else:
                    logger.info(f"     📁 {item.name}/")

            if len(extracted_files) > 5:
                logger.info(f"     ... and {len(extracted_files) - 5} more items")

            # Clean up ZIP file if requested
            if cleanup:
                zip_file.unlink()
                logger.info(f"🗑️ Removed {zip_file.name}")

        except Exception as e:
            logger.error(f"❌ Failed to extract {zip_file}: {e}")

    return extracted_count


def setup_evaluation_data_extraction():
    """Setup and extract evaluation data files"""
    logger.info("🔧 Setting up evaluation data extraction...")

    # Check for evaluation data directory
    eval_data_paths = [
        Path("../evaluation_data"),
        Path("./evaluation_data"),
        Path("../evaluation-pipeline-2024/evaluation_data"),
        Path("../evaluation-pipeline-2025/evaluation_data")
    ]

    found_eval_data = False

    for eval_path in eval_data_paths:
        if eval_path.exists():
            logger.info(f"✅ Found evaluation data at: {eval_path.absolute()}")
            found_eval_data = True

            # Extract ZIP files in fast_eval
            fast_eval_dir = eval_path / "fast_eval"
            if fast_eval_dir.exists():
                logger.info("📦 Extracting ZIP files in fast_eval/...")
                fast_extracted = extract_zip_files_in_directory(fast_eval_dir)
                logger.info(f"✅ Extracted {fast_extracted} ZIP files in fast_eval/")

            # Extract ZIP files in full_eval
            full_eval_dir = eval_path / "full_eval"
            if full_eval_dir.exists():
                logger.info("📦 Extracting ZIP files in full_eval/...")
                full_extracted = extract_zip_files_in_directory(full_eval_dir)
                logger.info(f"✅ Extracted {full_extracted} ZIP files in full_eval/")

            # Summary
            total_extracted = (fast_extracted if 'fast_extracted' in locals() else 0) + \
                            (full_extracted if 'full_extracted' in locals() else 0)

            if total_extracted > 0:
                logger.info(f"🎉 Successfully extracted {total_extracted} ZIP files!")
            else:
                logger.info("📁 No ZIP files found to extract")

    if not found_eval_data:
        logger.error("❌ No evaluation data directory found!")
        logger.info("💡 Run download_evaluation_data.py first to download the evaluation data")
        return False

    return True


def verify_ewok_extraction():
    """Specifically verify EWoK data extraction"""
    logger.info("🔍 Verifying EWoK data extraction...")

    # Check for ewok_fast.zip and its extraction
    eval_data_paths = [
        Path("../evaluation_data"),
        Path("./evaluation_data"),
    ]

    for eval_path in eval_data_paths:
        ewok_zip_path = eval_path / "fast_eval" / "ewok_fast.zip"
        ewok_dir_path = eval_path / "fast_eval" / "ewok_fast"

        if ewok_zip_path.exists():
            logger.info(f"📦 Found ewok_fast.zip at: {ewok_zip_path}")

            if not ewok_dir_path.exists():
                logger.info("📦 Extracting ewok_fast.zip...")
                try:
                    with zipfile.ZipFile(ewok_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(ewok_dir_path)

                    logger.info(f"✅ Extracted ewok_fast.zip to {ewok_dir_path}")

                    # List contents
                    contents = list(ewok_dir_path.iterdir())
                    logger.info(f"📁 EWoK fast eval contents ({len(contents)} items):")
                    for item in contents[:10]:  # Show first 10 items
                        if item.is_file():
                            size_kb = item.stat().st_size / 1024
                            logger.info(f"  📄 {item.name} ({size_kb:.1f} KB)")
                        else:
                            logger.info(f"  📁 {item.name}/")

                    if len(contents) > 10:
                        logger.info(f"  ... and {len(contents) - 10} more items")

                    # Remove ZIP file
                    ewok_zip_path.unlink()
                    logger.info("🗑️ Removed ewok_fast.zip")

                    return True

                except Exception as e:
                    logger.error(f"❌ Failed to extract ewok_fast.zip: {e}")
                    return False
            else:
                logger.info(f"✅ EWoK fast eval already extracted at: {ewok_dir_path}")
                return True
        else:
            logger.info(f"ℹ️ ewok_fast.zip not found at: {ewok_zip_path}")

    return False


def main():
    """Main function to extract evaluation data"""
    logger.info("🚀 Evaluation Data Extraction Setup")
    logger.info("=" * 50)

    # First, check and extract EWoK specifically
    ewok_success = verify_ewok_extraction()

    # Then, extract any other ZIP files
    extraction_success = setup_evaluation_data_extraction()

    if ewok_success or extraction_success:
        logger.info("✅ Evaluation data extraction completed!")

        # Copy/update evaluation data to pipeline directories
        logger.info("🔗 Updating pipeline directories...")

        # Re-run symlink setup from download script
        try:
            from download_evaluation_data import setup_symlinks
            setup_symlinks()
            logger.info("✅ Pipeline directories updated")
        except ImportError:
            logger.warning("⚠️ Could not import setup_symlinks function")
            logger.info("💡 You may need to manually copy evaluation_data to pipeline directories")

        return True
    else:
        logger.error("❌ No evaluation data found to extract")
        logger.info("💡 Run download_evaluation_data.py first")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
