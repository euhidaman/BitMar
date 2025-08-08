"""
Download BabyLM Challenge Evaluation Data
Downloads evaluation data from the official OSF repository for both 2024 and 2025 evaluation pipelines
URL: https://files.osf.io/v1/resources/ryjfm/providers/osfstorage/6819f54f5dc6fc2bff0a7bba/?zip=
"""

import os
import sys
import requests
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
    """Extract ZIP file and handle nested directory structure"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # First extract to a temporary location to inspect structure
            temp_extract = Path(extract_to) / "temp_extract"
            temp_extract.mkdir(exist_ok=True)

            zip_ref.extractall(temp_extract)

            # Check the structure and move files appropriately
            extracted_items = list(temp_extract.iterdir())

            # If there's a single directory containing everything, use that
            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                source_dir = extracted_items[0]
                logger.info(f"Found nested directory: {source_dir.name}")

                # Move contents from nested directory to target location
                target_dir = Path(extract_to)
                for item in source_dir.iterdir():
                    target_path = target_dir / item.name
                    if target_path.exists():
                        if target_path.is_dir():
                            shutil.rmtree(target_path)
                        else:
                            target_path.unlink()
                    shutil.move(str(item), str(target_path))
            else:
                # Move all items directly
                target_dir = Path(extract_to)
                for item in extracted_items:
                    target_path = target_dir / item.name
                    if target_path.exists():
                        if target_path.is_dir():
                            shutil.rmtree(target_path)
                        else:
                            target_path.unlink()
                    shutil.move(str(item), str(target_path))

            # Clean up temp directory
            shutil.rmtree(temp_extract)

        logger.info(f"✅ Extracted {zip_path} to {extract_to}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to extract {zip_path}: {e}")
        return False


def check_evaluation_data_exists():
    """Check if evaluation data already exists"""
    logger.info("🔍 Checking for existing evaluation data...")

    # Check both pipeline locations
    possible_locations = [
        Path("../evaluation_data"),
        Path("./evaluation_data"),
        Path("../evaluation-pipeline-2024/evaluation_data"),
        Path("../evaluation-pipeline-2025/evaluation_data")
    ]

    for location in possible_locations:
        if location.exists():
            # Check for key directories
            fast_eval = location / "fast_eval"
            full_eval = location / "full_eval"

            if fast_eval.exists() and full_eval.exists():
                logger.info(f"✅ Found evaluation data at: {location.absolute()}")

                # Count files in each directory
                fast_files = len(list(fast_eval.rglob("*.*")))
                full_files = len(list(full_eval.rglob("*.*")))

                logger.info(f"  • fast_eval: {fast_files} files")
                logger.info(f"  • full_eval: {full_files} files")

                if fast_files > 50 and full_files > 100:  # Reasonable threshold
                    logger.info("🎉 Evaluation data appears complete!")
                    return True, location
                else:
                    logger.warning(f"⚠️ Evaluation data seems incomplete at {location}")

    logger.info("❌ No complete evaluation data found")
    return False, None


def fix_existing_evaluation_data():
    """Fix the evaluation data structure if it was extracted incorrectly"""
    logger.info("🔧 Checking and fixing evaluation data structure...")

    # Check if fast_eval and full_eval are in the wrong location (parent directory)
    parent_fast_eval = Path("../../fast_eval")
    parent_full_eval = Path("../../full_eval")

    # Target location
    target_eval_data = Path("../evaluation_data")
    target_fast_eval = target_eval_data / "fast_eval"
    target_full_eval = target_eval_data / "full_eval"

    moved_something = False

    # Move fast_eval if it's in the wrong place
    if parent_fast_eval.exists() and not target_fast_eval.exists():
        logger.info(f"📁 Moving fast_eval from {parent_fast_eval} to {target_fast_eval}")
        target_eval_data.mkdir(exist_ok=True)
        shutil.move(str(parent_fast_eval), str(target_fast_eval))
        moved_something = True

    # Move full_eval if it's in the wrong place
    if parent_full_eval.exists() and not target_full_eval.exists():
        logger.info(f"📁 Moving full_eval from {parent_full_eval} to {target_full_eval}")
        target_eval_data.mkdir(exist_ok=True)
        shutil.move(str(parent_full_eval), str(target_full_eval))
        moved_something = True

    if moved_something:
        logger.info("✅ Fixed evaluation data structure!")
        return True
    else:
        logger.info("📁 Evaluation data structure is already correct")
        return False


def download_evaluation_data():
    """Download BabyLM evaluation data from OSF"""
    logger.info("🚀 Starting BabyLM Evaluation Data Download")
    logger.info("=" * 60)

    # First try to fix existing data structure
    fix_existing_evaluation_data()

    # Target directory for evaluation data
    eval_data_dir = Path("../evaluation_data")
    eval_data_dir.mkdir(exist_ok=True)

    # Official evaluation data URL
    eval_data_url = "https://files.osf.io/v1/resources/ryjfm/providers/osfstorage/6819f54f5dc6fc2bff0a7bba/?zip="
    zip_filename = eval_data_dir / "babylm_evaluation_data.zip"

    logger.info(f"📁 Target directory: {eval_data_dir.absolute()}")
    logger.info(f"📥 Downloading from: {eval_data_url}")

    # Expected structure after extraction
    expected_structure = [
        "evaluation_data/fast_eval/",
        "evaluation_data/full_eval/",
        "evaluation_data/fast_eval/blimp_fast/",
        "evaluation_data/full_eval/blimp_filtered/",
        "evaluation_data/full_eval/glue_filtered/",
        "evaluation_data/full_eval/winoground_filtered/",
        "evaluation_data/full_eval/vqa_filtered/"
    ]

    logger.info("\n📋 Expected evaluation data structure:")
    for path in expected_structure:
        logger.info(f"   📁 {path}")

    # Download the ZIP file
    logger.info(f"\n📥 Downloading evaluation data...")
    if download_file(eval_data_url, str(zip_filename)):
        logger.info("✅ Download completed successfully!")

        # Extract ZIP file with improved handling
        logger.info("📦 Extracting evaluation data...")
        if extract_zip(str(zip_filename), str(eval_data_dir)):
            logger.info("✅ Extraction completed successfully!")

            # Clean up ZIP file
            zip_filename.unlink()
            logger.info("🧹 Cleaned up ZIP file")

            # Verify extraction
            return verify_evaluation_data(eval_data_dir)
        else:
            logger.error("❌ Failed to extract evaluation data")
            return False
    else:
        logger.error("❌ Failed to download evaluation data")
        return False


def verify_evaluation_data(eval_data_dir: Path):
    """Verify the downloaded evaluation data"""
    logger.info("🔍 Verifying evaluation data...")

    # Check main directories
    fast_eval_dir = eval_data_dir / "fast_eval"
    full_eval_dir = eval_data_dir / "full_eval"

    if not fast_eval_dir.exists() or not full_eval_dir.exists():
        logger.error("❌ Missing fast_eval or full_eval directories")
        return False

    # Check key subdirectories for 2025 pipeline
    required_2025_dirs = [
        "blimp_filtered",
        "cdi_childes",
        "comps",
        "entity_tracking",
        "glue_filtered",
        "reading",
        "supplement_filtered",
        "winoground_filtered",
        "vqa_filtered",
        "wug_adj_nominalization",
        "wug_past_tense"
    ]

    # Check key subdirectories for 2024 pipeline
    required_2024_dirs = [
        "winoground_filtered",
        "vqa_filtered"
    ]

    missing_dirs = []
    found_dirs = []

    # Check full_eval directories
    for dir_name in required_2025_dirs:
        dir_path = full_eval_dir / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.rglob("*.*")))
            logger.info(f"  ✅ {dir_name}: {file_count} files")
            found_dirs.append(dir_name)
        else:
            logger.warning(f"  ❌ {dir_name}: Missing")
            missing_dirs.append(dir_name)

    # Check fast_eval directories (should be subsets)
    fast_dirs = [d.name for d in fast_eval_dir.iterdir() if d.is_dir()]
    logger.info(f"\n📁 Fast eval directories found: {len(fast_dirs)}")
    for dir_name in fast_dirs:
        file_count = len(list((fast_eval_dir / dir_name).rglob("*.*")))
        logger.info(f"  📂 {dir_name}: {file_count} files")

    # Summary
    logger.info(f"\n📊 Verification Summary:")
    logger.info(f"  • Found directories: {len(found_dirs)}/{len(required_2025_dirs)}")
    logger.info(f"  • Missing directories: {len(missing_dirs)}")

    if missing_dirs:
        logger.warning(f"  • Missing: {missing_dirs}")

    # Check if we have the essentials for both pipelines
    has_2024_essentials = all(d in found_dirs for d in required_2024_dirs)
    has_2025_essentials = len(found_dirs) >= len(required_2025_dirs) * 0.8  # 80% threshold

    if has_2024_essentials and has_2025_essentials:
        logger.info("🎉 Evaluation data verification successful!")
        logger.info("✅ Both 2024 and 2025 pipeline data available")
        return True
    elif has_2024_essentials:
        logger.info("⚠️ 2024 pipeline data available, 2025 data incomplete")
        return True
    else:
        logger.error("❌ Evaluation data verification failed")
        return False


def setup_symlinks():
    """Create symlinks to evaluation data in both pipeline directories"""
    logger.info("🔗 Setting up symlinks to evaluation data...")

    eval_data_dir = Path("../evaluation_data")

    if not eval_data_dir.exists():
        logger.error("❌ Evaluation data directory not found")
        return False

    # Correct symlinks for both pipelines (they're at the same level as BitMar)
    pipeline_dirs = [
        Path("../evaluation-pipeline-2024"),  # D:\BabyLM\evaluation-pipeline-2024
        Path("../evaluation-pipeline-2025")   # D:\BabyLM\evaluation-pipeline-2025
    ]

    success_count = 0

    for pipeline_dir in pipeline_dirs:
        pipeline_name = pipeline_dir.name
        logger.info(f"🔍 Checking pipeline: {pipeline_dir.absolute()}")

        if pipeline_dir.exists():
            logger.info(f"✅ Found {pipeline_name} at: {pipeline_dir.absolute()}")
            symlink_path = pipeline_dir / "evaluation_data"

            # Remove existing symlink/directory if it exists
            if symlink_path.exists() or symlink_path.is_symlink():
                if symlink_path.is_symlink():
                    symlink_path.unlink()
                    logger.info(f"🗑️ Removed existing symlink: {symlink_path}")
                else:
                    logger.warning(f"⚠️ Directory exists at {symlink_path}, skipping symlink creation")
                    # Copy instead of symlink if directory exists
                    logger.info(f"📁 Copying evaluation_data to {pipeline_name} instead...")
                    try:
                        shutil.copytree(eval_data_dir, symlink_path, dirs_exist_ok=True)
                        logger.info(f"✅ Copied evaluation_data to {pipeline_name}")
                        success_count += 1
                        continue
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to copy to {pipeline_name}: {e}")
                        continue

            try:
                # Create symlink (Windows requires admin rights for directory symlinks)
                if os.name == 'nt':  # Windows
                    import subprocess
                    result = subprocess.run([
                        'mklink', '/D',
                        str(symlink_path.absolute()),
                        str(eval_data_dir.absolute())
                    ], shell=True, capture_output=True, text=True)

                    if result.returncode == 0:
                        logger.info(f"✅ Created symlink: {pipeline_name}/evaluation_data")
                        success_count += 1
                    else:
                        logger.warning(f"⚠️ Failed to create symlink for {pipeline_name}: {result.stderr}")
                        # Copy instead of symlink on Windows if no admin rights
                        logger.info(f"📁 Copying evaluation_data to {pipeline_name} instead...")
                        try:
                            shutil.copytree(eval_data_dir, symlink_path, dirs_exist_ok=True)
                            logger.info(f"✅ Copied evaluation_data to {pipeline_name}")
                            success_count += 1
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to copy to {pipeline_name}: {e}")
                else:  # Unix-like
                    symlink_path.symlink_to(eval_data_dir.absolute())
                    logger.info(f"✅ Created symlink: {pipeline_name}/evaluation_data")
                    success_count += 1

            except Exception as e:
                logger.warning(f"⚠️ Failed to create symlink for {pipeline_name}: {e}")
                # Try copying as fallback
                try:
                    logger.info(f"📁 Trying to copy evaluation_data to {pipeline_name}...")
                    shutil.copytree(eval_data_dir, symlink_path, dirs_exist_ok=True)
                    logger.info(f"✅ Copied evaluation_data to {pipeline_name}")
                    success_count += 1
                except Exception as copy_error:
                    logger.warning(f"⚠️ Failed to copy to {pipeline_name}: {copy_error}")
        else:
            logger.warning(f"⚠️ Pipeline directory not found: {pipeline_dir.absolute()}")
            logger.info(f"💡 Expected location: {pipeline_dir.absolute()}")

    if success_count > 0:
        logger.info(f"✅ Successfully set up evaluation data for {success_count} pipeline(s)")
        return True
    else:
        logger.error("❌ Failed to set up evaluation data for any pipeline")
        logger.info("💡 Manual setup instructions:")
        logger.info(f"1. Copy {eval_data_dir.absolute()} to:")
        logger.info("   • D:\\BabyLM\\evaluation-pipeline-2024\\evaluation_data")
        logger.info("   • D:\\BabyLM\\evaluation-pipeline-2025\\evaluation_data")
        return False


def main():
    """Main function to download and setup evaluation data"""
    logger.info("🚀 BabyLM Evaluation Data Setup")
    logger.info("=" * 50)

    # First check if data already exists
    data_exists, existing_location = check_evaluation_data_exists()

    if data_exists:
        logger.info("✅ Evaluation data already exists!")
        # Still try to set up symlinks
        setup_symlinks()
        return True

    # Download evaluation data
    logger.info("📥 Downloading evaluation data...")

    try:
        success = download_evaluation_data()
        if success:
            logger.info("✅ Download and extraction completed successfully!")

            # Set up symlinks to both evaluation pipelines
            setup_success = setup_symlinks()

            if setup_success:
                logger.info("🎉 Evaluation data setup completed successfully!")
                logger.info("\n📋 Next steps:")
                logger.info("1. Install evaluation pipeline dependencies")
                logger.info("2. Run evaluations using the provided scripts")
                return True
            else:
                logger.warning("⚠️ Download successful but symlink setup failed")
                logger.info("💡 You can manually copy evaluation_data to pipeline directories")
                return True
        else:
            logger.error("❌ Download failed!")
            return False
    except Exception as e:
        logger.error(f"❌ Setup failed with error: {e}")
        logger.info("\n📋 Manual setup instructions:")
        logger.info("1. Download the evaluation data from:")
        logger.info("   https://files.osf.io/v1/resources/ryjfm/providers/osfstorage/6819f54f5dc6fc2bff0a7bba/?zip=")
        logger.info("2. Extract to ../evaluation_data/")
        logger.info("3. Copy or symlink to both evaluation pipeline directories")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
