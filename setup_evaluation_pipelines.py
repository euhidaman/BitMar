"""
Setup and Install Dependencies for BabyLM Evaluation Pipelines
Handles installation of both 2024 and 2025 evaluation pipeline dependencies
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(cmd, cwd=None, shell=True):
    """Run command and return success status"""
    try:
        logger.info(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=shell, cwd=cwd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("✅ Command successful")
            return True
        else:
            logger.error(f"❌ Command failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ Command execution failed: {e}")
        return False


def install_2024_pipeline():
    """Install 2024 evaluation pipeline dependencies"""
    logger.info("🔧 Setting up 2024 Evaluation Pipeline...")

    pipeline_2024_dir = Path("../evaluation-pipeline-2024")

    if not pipeline_2024_dir.exists():
        logger.error(f"❌ 2024 pipeline directory not found: {pipeline_2024_dir}")
        return False

    # Install the package in editable mode
    success = run_command(f"pip install -e .", cwd=pipeline_2024_dir)

    if not success:
        logger.error("❌ Failed to install 2024 pipeline base package")
        return False

    # Install additional dependencies for multimodal evaluation
    additional_deps = [
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "transformers>=4.20.0",
        "accelerate>=0.20.0",
        "minicons",
        "pillow",
        "requests"
    ]

    for dep in additional_deps:
        success = run_command(f"pip install {dep}")
        if not success:
            logger.warning(f"⚠️ Failed to install {dep}")

    logger.info("✅ 2024 pipeline setup completed")
    return True


def install_2025_pipeline():
    """Install 2025 evaluation pipeline dependencies"""
    logger.info("🔧 Setting up 2025 Evaluation Pipeline...")

    pipeline_2025_dir = Path("../evaluation-pipeline-2025")

    if not pipeline_2025_dir.exists():
        logger.error(f"❌ 2025 pipeline directory not found: {pipeline_2025_dir}")
        return False

    # Install requirements from requirements.txt
    requirements_file = pipeline_2025_dir / "requirements.txt"

    if requirements_file.exists():
        success = run_command(f"pip install -r {requirements_file}", cwd=pipeline_2025_dir)
        if not success:
            logger.error("❌ Failed to install 2025 pipeline requirements")
            return False
    else:
        logger.warning("⚠️ requirements.txt not found for 2025 pipeline")

    # Install additional dependencies
    additional_deps = [
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "wandb",
        "scikit-learn",
        "nltk",
        "statsmodels"
    ]

    for dep in additional_deps:
        success = run_command(f"pip install {dep}")
        if not success:
            logger.warning(f"⚠️ Failed to install {dep}")

    logger.info("✅ 2025 pipeline setup completed")
    return True


def setup_huggingface_login():
    """Setup HuggingFace login for datasets that require authentication"""
    logger.info("🔐 Setting up HuggingFace authentication...")

    try:
        result = subprocess.run(["huggingface-cli", "whoami"], capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("✅ Already logged in to HuggingFace")
            return True
        else:
            logger.info("📝 Please log in to HuggingFace for dataset access")
            logger.info("Run: huggingface-cli login")
            logger.info("This is needed for Winoground and other datasets")
            return False
    except FileNotFoundError:
        logger.warning("⚠️ huggingface-cli not found. Installing...")
        success = run_command("pip install huggingface_hub[cli]")
        if success:
            logger.info("📝 Please run: huggingface-cli login")
        return success


def download_nltk_data():
    """Download required NLTK data"""
    logger.info("📚 Downloading NLTK data...")

    try:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        logger.info("✅ NLTK data downloaded")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Failed to download NLTK data: {e}")
        return False


def verify_installations():
    """Verify that both pipelines are properly installed"""
    logger.info("🔍 Verifying installations...")

    # Test 2024 pipeline
    try:
        pipeline_2024_dir = Path("../evaluation-pipeline-2024")
        result = subprocess.run([
            sys.executable, "-c",
            "import lm_eval; print('2024 pipeline OK')"
        ], cwd=pipeline_2024_dir, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("✅ 2024 pipeline import successful")
        else:
            logger.warning("⚠️ 2024 pipeline import failed")
    except Exception as e:
        logger.warning(f"⚠️ 2024 pipeline verification failed: {e}")

    # Test 2025 pipeline
    try:
        pipeline_2025_dir = Path("../evaluation-pipeline-2025")
        result = subprocess.run([
            sys.executable, "-c",
            "import evaluation_pipeline; print('2025 pipeline OK')"
        ], cwd=pipeline_2025_dir, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("✅ 2025 pipeline import successful")
        else:
            logger.warning("⚠️ 2025 pipeline import failed")
    except Exception as e:
        logger.warning(f"⚠️ 2025 pipeline verification failed: {e}")


def main():
    """Main setup function"""
    logger.info("🚀 BabyLM Evaluation Pipeline Setup")
    logger.info("=" * 50)

    # Install both pipelines
    success_2024 = install_2024_pipeline()
    success_2025 = install_2025_pipeline()

    # Setup additional components
    hf_login = setup_huggingface_login()
    nltk_success = download_nltk_data()

    # Verify installations
    verify_installations()

    # Summary
    logger.info("\n📊 Setup Summary:")
    logger.info(f"  • 2024 Pipeline: {'✅' if success_2024 else '❌'}")
    logger.info(f"  • 2025 Pipeline: {'✅' if success_2025 else '❌'}")
    logger.info(f"  • HuggingFace Auth: {'✅' if hf_login else '⚠️'}")
    logger.info(f"  • NLTK Data: {'✅' if nltk_success else '⚠️'}")

    if success_2024 and success_2025:
        logger.info("\n🎉 Evaluation pipeline setup completed!")
        logger.info("\n📋 Next steps:")
        logger.info("1. Download evaluation data: python download_evaluation_data.py")
        logger.info("2. Run evaluations using your training script")

        if not hf_login:
            logger.info("3. Login to HuggingFace: huggingface-cli login")

        return True
    else:
        logger.error("\n❌ Setup incomplete. Please check errors above.")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
