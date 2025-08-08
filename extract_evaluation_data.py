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


def extract_zip_files_in_directory(directory: Path, cleanup: bool = True, max_iterations: int = 3):
    """Extract all ZIP files in a directory recursively with multiple iterations"""
    total_extracted = 0

    for iteration in range(max_iterations):
        logger.info(f"🔄 Iteration {iteration + 1}/{max_iterations}: Scanning {directory}")

        # Find all ZIP files in current iteration
        zip_files = list(directory.rglob("*.zip"))

        if not zip_files:
            logger.info(f"  📂 No ZIP files found in iteration {iteration + 1}")
            break

        logger.info(f"  📦 Found {len(zip_files)} ZIP files to extract")
        iteration_extracted = 0

        for zip_file in zip_files:
            logger.info(f"    📦 Processing: {zip_file.relative_to(directory)}")

            # Determine extraction directory (same directory as ZIP file)
            extract_dir = zip_file.parent / zip_file.stem

            try:
                # Create extraction directory
                extract_dir.mkdir(exist_ok=True)

                # Extract ZIP file
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

                logger.info(f"      ✅ Extracted {zip_file.name} to {extract_dir.name}/")
                iteration_extracted += 1
                total_extracted += 1

                # List extracted contents
                extracted_items = list(extract_dir.iterdir())
                logger.info(f"      📁 Extracted {len(extracted_items)} items:")

                # Show first 3 items and count ZIP files
                zip_count = 0
                for i, item in enumerate(extracted_items[:3]):
                    if item.is_file():
                        size_kb = item.stat().st_size / 1024
                        if item.suffix.lower() == '.zip':
                            logger.info(f"        📦 {item.name} ({size_kb:.1f} KB) - ZIP file detected!")
                            zip_count += 1
                        else:
                            logger.info(f"        📄 {item.name} ({size_kb:.1f} KB)")
                    else:
                        logger.info(f"        📁 {item.name}/")

                # Count remaining ZIP files
                for item in extracted_items[3:]:
                    if item.suffix.lower() == '.zip':
                        zip_count += 1

                if len(extracted_items) > 3:
                    logger.info(f"        ... and {len(extracted_items) - 3} more items")

                if zip_count > 0:
                    logger.info(f"      🔍 Found {zip_count} additional ZIP files to extract in next iteration")

                # Clean up ZIP file if requested
                if cleanup:
                    zip_file.unlink()
                    logger.info(f"      🗑️ Removed {zip_file.name}")

            except Exception as e:
                logger.error(f"      ❌ Failed to extract {zip_file.name}: {e}")

        logger.info(f"  ✅ Iteration {iteration + 1} completed: {iteration_extracted} files extracted")

        # If no files were extracted in this iteration, we're done
        if iteration_extracted == 0:
            break

    logger.info(f"🎉 Total extraction completed: {total_extracted} ZIP files extracted across {iteration + 1} iterations")
    return total_extracted


def scan_and_report_directory_structure(directory: Path, max_depth: int = 3):
    """Scan and report the directory structure to identify all ZIP files"""
    logger.info(f"🔍 Scanning directory structure: {directory}")

    zip_files = []
    total_files = 0

    def scan_recursive(path: Path, current_depth: int = 0):
        nonlocal total_files

        if current_depth > max_depth:
            return

        try:
            for item in path.iterdir():
                total_files += 1

                if item.is_file():
                    if item.suffix.lower() == '.zip':
                        zip_files.append(item)
                        size_mb = item.stat().st_size / (1024 * 1024)
                        indent = "  " * current_depth
                        logger.info(f"{indent}📦 {item.relative_to(directory)} ({size_mb:.1f} MB)")
                elif item.is_dir() and current_depth < max_depth:
                    # Recursively scan subdirectories
                    scan_recursive(item, current_depth + 1)

        except PermissionError:
            logger.warning(f"⚠️ Permission denied accessing {path}")
        except Exception as e:
            logger.warning(f"⚠️ Error scanning {path}: {e}")

    scan_recursive(directory)

    logger.info(f"📊 Scan complete:")
    logger.info(f"  • Total items scanned: {total_files}")
    logger.info(f"  • ZIP files found: {len(zip_files)}")

    return zip_files


def setup_evaluation_data_extraction():
    """Setup and extract evaluation data files with comprehensive scanning"""
    logger.info("🔧 Setting up evaluation data extraction...")

    # Check for evaluation data directory
    eval_data_paths = [
        Path("../evaluation_data"),
        Path("./evaluation_data"),
        Path("../evaluation-pipeline-2024/evaluation_data"),
        Path("../evaluation-pipeline-2025/evaluation_data")
    ]

    found_eval_data = False
    total_extracted = 0

    for eval_path in eval_data_paths:
        if eval_path.exists():
            logger.info(f"✅ Found evaluation data at: {eval_path.absolute()}")
            found_eval_data = True

            # First, scan and report all ZIP files
            logger.info("🔍 Initial scan for ZIP files...")
            initial_zip_files = scan_and_report_directory_structure(eval_path)

            if initial_zip_files:
                logger.info(f"📦 Found {len(initial_zip_files)} ZIP files to extract")

                # Extract ZIP files in fast_eval with iterations
                fast_eval_dir = eval_path / "fast_eval"
                if fast_eval_dir.exists():
                    logger.info("📦 Extracting ZIP files in fast_eval/ (with iterations)...")
                    fast_extracted = extract_zip_files_in_directory(fast_eval_dir, max_iterations=5)
                    total_extracted += fast_extracted
                    logger.info(f"✅ Extracted {fast_extracted} ZIP files in fast_eval/")

                # Extract ZIP files in full_eval with iterations
                full_eval_dir = eval_path / "full_eval"
                if full_eval_dir.exists():
                    logger.info("📦 Extracting ZIP files in full_eval/ (with iterations)...")
                    full_extracted = extract_zip_files_in_directory(full_eval_dir, max_iterations=5)
                    total_extracted += full_extracted
                    logger.info(f"✅ Extracted {full_extracted} ZIP files in full_eval/")

                # Final scan to verify all ZIP files are extracted
                logger.info("🔍 Final scan to verify extraction...")
                remaining_zip_files = scan_and_report_directory_structure(eval_path)

                if remaining_zip_files:
                    logger.warning(f"⚠️ {len(remaining_zip_files)} ZIP files still remain:")
                    for zip_file in remaining_zip_files:
                        logger.warning(f"  📦 {zip_file.relative_to(eval_path)}")
                else:
                    logger.info("✅ All ZIP files successfully extracted!")
            else:
                logger.info("📁 No ZIP files found to extract")

    if not found_eval_data:
        logger.error("❌ No evaluation data directory found!")
        logger.info("💡 Run download_evaluation_data.py first to download the evaluation data")
        return False

    if total_extracted > 0:
        logger.info(f"🎉 Successfully extracted {total_extracted} ZIP files total!")
    else:
        logger.info("📁 No ZIP files found to extract")

    return True


def verify_specific_extractions():
    """Specifically verify known extractions like EWoK"""
    logger.info("🔍 Verifying specific known extractions...")

    # Check for specific known ZIP files and their extractions
    known_extractions = [
        {
            'name': 'EWoK Fast',
            'zip_pattern': '**/ewok_fast.zip',
            'expected_dir': 'ewok_fast'
        },
        {
            'name': 'DevBench',
            'zip_pattern': '**/devbench*.zip',
            'expected_dir': 'devbench'
        }
    ]

    eval_data_paths = [
        Path("../evaluation_data"),
        Path("./evaluation_data"),
    ]

    for eval_path in eval_data_paths:
        if eval_path.exists():
            logger.info(f"🔍 Checking specific extractions in: {eval_path}")

            for extraction in known_extractions:
                # Look for ZIP files matching pattern
                zip_files = list(eval_path.glob(extraction['zip_pattern']))

                if zip_files:
                    logger.info(f"📦 Found {extraction['name']} ZIP files:")
                    for zip_file in zip_files:
                        size_mb = zip_file.stat().st_size / (1024 * 1024)
                        logger.info(f"  📦 {zip_file.relative_to(eval_path)} ({size_mb:.1f} MB)")

                        # Check if already extracted
                        expected_extract_dir = zip_file.parent / extraction['expected_dir']
                        if expected_extract_dir.exists():
                            items = list(expected_extract_dir.iterdir())
                            logger.info(f"    ✅ Already extracted to {expected_extract_dir.name}/ ({len(items)} items)")
                        else:
                            logger.info(f"    ⚠️ Not yet extracted - will be handled by main extraction")

    return True


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
