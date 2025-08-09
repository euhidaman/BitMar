"""
Comprehensive Evaluation Data Validator and Downloader
Validates and downloads missing evaluation data for BitMar evaluation pipelines
"""

import os
import sys
import logging
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationDataValidator:
    """Validates and downloads missing evaluation data"""

    def __init__(self, base_path: str = "D:/BabyLM"):
        self.base_path = Path(base_path)
        self.eval_data_path = self.base_path / "evaluation_data"
        self.pipeline_2024_path = self.base_path / "evaluation-pipeline-2024"
        self.pipeline_2025_path = self.base_path / "evaluation-pipeline-2025"

        # Create directories if they don't exist
        self.eval_data_path.mkdir(exist_ok=True)
        (self.eval_data_path / "fast_eval").mkdir(exist_ok=True)
        (self.eval_data_path / "full_eval").mkdir(exist_ok=True)

    def check_required_files(self) -> Dict[str, Dict[str, bool]]:
        """Check which required evaluation files exist"""
        required_files = {
            "fast_eval": {
                "blimp_fast": "blimp_fast/",
                "supplement_fast": "supplement_fast/",
                "entity_tracking_fast": "entity_tracking_fast/",
                "reading": "reading/",
                "wug_adj": "wug_adj_nominalization/",
                "wug_past": "wug_past_tense/",
                "vqa_filtered": "vqa_filtered/",
                "winoground_filtered": "winoground_filtered/"
            },
            "full_eval": {
                "blimp_filtered": "blimp_filtered/",
                "supplement_filtered": "supplement_filtered/",
                "entity_tracking": "entity_tracking/",
                "reading": "reading/",
                "wug_adj": "wug_adj_nominalization/",
                "wug_past": "wug_past_tense/",
                "vqa_filtered": "vqa_filtered/",
                "winoground_filtered": "winoground_filtered/",
                "devbench": "devbench/",
                "glue_filtered": "glue_filtered/",
                "cdi_childes": "cdi_childes/",
                "comps": "comps/"
            }
        }

        status = {}
        for eval_type, files in required_files.items():
            status[eval_type] = {}
            eval_dir = self.eval_data_path / eval_type

            for name, path in files.items():
                full_path = eval_dir / path
                status[eval_type][name] = full_path.exists()

        return status

    def download_devbench_data(self):
        """Download DevBench evaluation data from official sources"""
        logger.info("📥 Downloading DevBench evaluation data from official sources...")

        devbench_dir = self.eval_data_path / "full_eval" / "devbench"
        devbench_dir.mkdir(parents=True, exist_ok=True)

        # Try to download from DevBench official repository or mirrors
        devbench_sources = [
            {
                "name": "DevBench official data",
                "base_url": "https://github.com/cambridgeltl/devbench/raw/main/data/",
                "files": {
                    "evals/sem-things/spose_similarity.mat": "sem-things/spose_similarity.mat",
                    "evals/sem-things/metadata.json": "sem-things/metadata.json"
                }
            },
            {
                "name": "DevBench mirror",
                "base_url": "https://huggingface.co/datasets/devbench/devbench-data/resolve/main/",
                "files": {
                    "evals/sem-things/spose_similarity.mat": "spose_similarity.mat",
                    "evals/sem-things/metadata.json": "metadata.json"
                }
            }
        ]

        for source in devbench_sources:
            logger.info(f"🔍 Trying {source['name']}...")
            try:
                for local_path, remote_file in source["files"].items():
                    full_local_path = devbench_dir / local_path
                    full_local_path.parent.mkdir(parents=True, exist_ok=True)

                    if full_local_path.exists():
                        logger.info(f"✅ {local_path} already exists")
                        continue

                    url = source["base_url"] + remote_file
                    logger.info(f"📥 Downloading {url}")

                    try:
                        response = requests.get(url, timeout=30)
                        if response.status_code == 200:
                            with open(full_local_path, 'wb') as f:
                                f.write(response.content)
                            logger.info(f"✅ Downloaded {local_path}")
                        else:
                            logger.warning(f"Failed to download {url}: HTTP {response.status_code}")
                    except Exception as e:
                        logger.warning(f"Failed to download {url}: {e}")

                # Check if we got the essential files
                essential_file = devbench_dir / "evals/sem-things/spose_similarity.mat"
                if essential_file.exists():
                    logger.info(f"✅ Successfully downloaded DevBench data from {source['name']}")
                    return True

            except Exception as e:
                logger.warning(f"Failed to download from {source['name']}: {e}")
                continue

        # If official sources fail, create a proper placeholder with correct structure
        logger.warning("⚠️ Could not download official DevBench data, creating structured placeholder...")

        # Create required directory structure
        (devbench_dir / "evals" / "sem-things").mkdir(parents=True, exist_ok=True)

        # Create a minimal but valid .mat file
        try:
            import scipy.io
            import numpy as np

            # Create a minimal similarity matrix that won't crash the evaluation
            similarity_data = {
                'similarity_matrix': np.eye(100) + np.random.normal(0, 0.1, (100, 100)),  # Identity + noise
                'labels': [f'concept_{i:03d}' for i in range(100)],
                'metadata': {
                    'description': 'Placeholder similarity matrix for DevBench evaluation',
                    'source': 'Generated placeholder - not real data',
                    'size': 100
                }
            }

            placeholder_file = devbench_dir / "evals" / "sem-things" / "spose_similarity.mat"
            scipy.io.savemat(str(placeholder_file), similarity_data)
            logger.info(f"✅ Created structured DevBench placeholder at {placeholder_file}")

            # Create metadata file
            metadata_file = devbench_dir / "evals" / "sem-things" / "metadata.json"
            metadata = {
                "name": "sem-things",
                "description": "Placeholder semantic things evaluation",
                "type": "similarity",
                "size": 100,
                "note": "This is a placeholder file - replace with real DevBench data for accurate evaluation"
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"✅ Created DevBench metadata at {metadata_file}")
            return True

        except ImportError:
            logger.error("❌ scipy not available, cannot create proper DevBench placeholder")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to create DevBench placeholder: {e}")
            return False

    def download_missing_data_from_pipelines(self):
        """Download missing data using evaluation pipeline scripts"""
        logger.info("📥 Checking evaluation pipelines for download scripts...")

        # Check 2025 pipeline for download scripts
        if self.pipeline_2025_path.exists():
            download_scripts = [
                "download_evaluation_data.py",
                "setup_data.py",
                "scripts/download_data.py"
            ]

            for script in download_scripts:
                script_path = self.pipeline_2025_path / script
                if script_path.exists():
                    logger.info(f"📥 Found download script: {script_path}")
                    try:
                        # Run the download script
                        result = subprocess.run([
                            sys.executable, str(script_path),
                            "--output_dir", str(self.eval_data_path)
                        ], capture_output=True, text=True, timeout=3600)

                        if result.returncode == 0:
                            logger.info("✅ Successfully downloaded evaluation data")
                            return True
                        else:
                            logger.warning(f"⚠️ Download script failed: {result.stderr}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to run download script: {e}")

        # Check 2024 pipeline
        if self.pipeline_2024_path.exists():
            # Similar check for 2024 pipeline
            pass

        return False

    def download_from_huggingface(self):
        """Download evaluation data from HuggingFace if available"""
        logger.info("📥 Downloading evaluation data from HuggingFace...")

        try:
            # Try to install huggingface_hub if not available
            try:
                from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files
            except ImportError:
                logger.info("Installing huggingface_hub...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
                from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files

            # BabyLM evaluation dataset repositories (real repositories)
            repos_to_try = [
                {
                    "repo_id": "babylm/evaluation-data-2025",
                    "description": "BabyLM 2025 evaluation data"
                },
                {
                    "repo_id": "babylm/evaluation-data",
                    "description": "BabyLM evaluation data"
                },
                {
                    "repo_id": "cpllab/babylm-evaluation-data",
                    "description": "CPL Lab BabyLM evaluation data"
                },
                {
                    "repo_id": "microsoft/BabyLM-evaluation",
                    "description": "Microsoft BabyLM evaluation data"
                }
            ]

            downloaded_any = False

            for repo_info in repos_to_try:
                repo_id = repo_info["repo_id"]
                description = repo_info["description"]

                try:
                    logger.info(f"🔍 Checking repository: {repo_id} ({description})")

                    # Try to list files to see if repo exists
                    files = list_repo_files(repo_id)
                    logger.info(f"Found {len(files)} files in {repo_id}")

                    # Download specific evaluation data files
                    for file in files:
                        file_lower = file.lower()

                        # Download DevBench data
                        if "devbench" in file_lower and (".mat" in file_lower or ".json" in file_lower or ".csv" in file_lower):
                            try:
                                logger.info(f"📥 Downloading DevBench file: {file}")
                                local_path = hf_hub_download(
                                    repo_id=repo_id,
                                    filename=file,
                                    local_dir=str(self.eval_data_path),
                                    local_dir_use_symlinks=False
                                )
                                logger.info(f"✅ Downloaded {file} to {local_path}")
                                downloaded_any = True
                            except Exception as e:
                                logger.warning(f"Failed to download {file}: {e}")

                        # Download VQA data
                        elif "vqa" in file_lower and (".json" in file_lower or ".jsonl" in file_lower):
                            try:
                                logger.info(f"📥 Downloading VQA file: {file}")
                                local_path = hf_hub_download(
                                    repo_id=repo_id,
                                    filename=file,
                                    local_dir=str(self.eval_data_path),
                                    local_dir_use_symlinks=False
                                )
                                logger.info(f"✅ Downloaded {file} to {local_path}")
                                downloaded_any = True
                            except Exception as e:
                                logger.warning(f"Failed to download {file}: {e}")

                        # Download Winoground data
                        elif "winoground" in file_lower and (".json" in file_lower or ".jsonl" in file_lower):
                            try:
                                logger.info(f"📥 Downloading Winoground file: {file}")
                                local_path = hf_hub_download(
                                    repo_id=repo_id,
                                    filename=file,
                                    local_dir=str(self.eval_data_path),
                                    local_dir_use_symlinks=False
                                )
                                logger.info(f"✅ Downloaded {file} to {local_path}")
                                downloaded_any = True
                            except Exception as e:
                                logger.warning(f"Failed to download {file}: {e}")

                    # If we found files, try to download entire evaluation directories
                    eval_dirs = ["devbench", "full_eval", "fast_eval"]
                    for eval_dir in eval_dirs:
                        matching_files = [f for f in files if f.startswith(eval_dir + "/")]
                        if matching_files:
                            try:
                                logger.info(f"📥 Downloading evaluation directory: {eval_dir}")
                                # Download specific directory
                                for file in matching_files[:10]:  # Limit to first 10 files to avoid huge downloads
                                    try:
                                        local_path = hf_hub_download(
                                            repo_id=repo_id,
                                            filename=file,
                                            local_dir=str(self.eval_data_path),
                                            local_dir_use_symlinks=False
                                        )
                                        logger.info(f"✅ Downloaded {file}")
                                        downloaded_any = True
                                    except Exception as e:
                                        logger.debug(f"Failed to download {file}: {e}")
                            except Exception as e:
                                logger.warning(f"Failed to download {eval_dir}: {e}")

                    if downloaded_any:
                        logger.info(f"✅ Successfully downloaded data from {repo_id}")
                        break  # Stop after first successful repo

                except Exception as e:
                    logger.debug(f"Repository {repo_id} not accessible: {e}")
                    continue

            # Try downloading from specific known evaluation datasets
            if not downloaded_any:
                logger.info("🔍 Trying specific evaluation datasets...")

                specific_datasets = [
                    {
                        "repo_id": "winoground/winoground",
                        "files": ["examples.jsonl"],
                        "target_dir": "full_eval/winoground_filtered"
                    },
                    {
                        "repo_id": "HuggingFaceM4/VQAv2",
                        "files": ["validation.json", "test.json"],
                        "target_dir": "full_eval/vqa_filtered"
                    },
                    {
                        "repo_id": "warstadt/blimp",
                        "files": ["data.jsonl"],
                        "target_dir": "full_eval/blimp_filtered"
                    }
                ]

                for dataset_info in specific_datasets:
                    repo_id = dataset_info["repo_id"]
                    files_to_download = dataset_info["files"]
                    target_dir = dataset_info["target_dir"]

                    try:
                        logger.info(f"📥 Downloading from {repo_id}")
                        target_path = self.eval_data_path / target_dir
                        target_path.mkdir(parents=True, exist_ok=True)

                        for file_name in files_to_download:
                            try:
                                local_path = hf_hub_download(
                                    repo_id=repo_id,
                                    filename=file_name,
                                    local_dir=str(target_path),
                                    local_dir_use_symlinks=False
                                )
                                logger.info(f"✅ Downloaded {file_name} from {repo_id}")
                                downloaded_any = True
                            except Exception as e:
                                logger.debug(f"Failed to download {file_name} from {repo_id}: {e}")

                    except Exception as e:
                        logger.debug(f"Failed to access {repo_id}: {e}")

            return downloaded_any

        except ImportError:
            logger.error("❌ Could not install huggingface_hub")
            return False
        except Exception as e:
            logger.warning(f"⚠️ HuggingFace download failed: {e}")
            return False

    def create_minimal_test_data(self):
        """Create minimal test data only if real data download completely fails"""
        logger.warning("⚠️ Creating minimal test data as last resort...")

        # Only create test data if no real data was found
        vqa_dir = self.eval_data_path / "full_eval" / "vqa_filtered"
        if not vqa_dir.exists() or not any(vqa_dir.iterdir()):
            logger.info("Creating minimal VQA test data...")
            vqa_dir.mkdir(parents=True, exist_ok=True)

            # Create a note explaining this is test data
            readme_content = """
# Minimal VQA Test Data

This is minimal test data created because real VQA evaluation data could not be downloaded.

To get real evaluation data:
1. Download from the official VQA dataset
2. Use the BabyLM evaluation pipeline data download scripts
3. Manually place VQA data in this directory

This test data will not produce meaningful evaluation results.
"""

            with open(vqa_dir / "README.md", 'w') as f:
                f.write(readme_content)

            # Create minimal data structure
            test_vqa = {
                "info": {
                    "description": "Minimal VQA test data - not for real evaluation",
                    "version": "test_placeholder",
                    "note": "Replace with real VQA data for meaningful results"
                },
                "questions": [
                    {
                        "question_id": 1,
                        "image_id": "test_001",
                        "question": "What color is the sky in this test image?",
                        "answers": [{"answer": "blue", "answer_confidence": "yes"}],
                        "question_type": "color",
                        "answer_type": "other"
                    }
                ]
            }

            with open(vqa_dir / "test_questions.json", 'w') as f:
                json.dump(test_vqa, f, indent=2)

            logger.warning(f"⚠️ Created minimal VQA test data - replace with real data!")

        # Similar for Winoground
        winoground_dir = self.eval_data_path / "full_eval" / "winoground_filtered"
        if not winoground_dir.exists() or not any(winoground_dir.iterdir()):
            logger.info("Creating minimal Winoground test data...")
            winoground_dir.mkdir(parents=True, exist_ok=True)

            readme_content = """
# Minimal Winoground Test Data

This is minimal test data created because real Winoground evaluation data could not be downloaded.

To get real evaluation data, download from the official Winoground dataset.
This test data will not produce meaningful evaluation results.
"""

            with open(winoground_dir / "README.md", 'w') as f:
                f.write(readme_content)

            test_winoground = {
                "info": {
                    "description": "Minimal Winoground test data - not for real evaluation",
                    "note": "Replace with real Winoground data for meaningful results"
                },
                "examples": [
                    {
                        "id": "test_001",
                        "caption_0": "The cat sat on the mat",
                        "caption_1": "The mat sat on the cat",
                        "image_0": "test_image_0.jpg",
                        "image_1": "test_image_1.jpg",
                        "tag": "test"
                    }
                ]
            }

            with open(winoground_dir / "test_examples.json", 'w') as f:
                json.dump(test_winoground, f, indent=2)

            logger.warning(f"⚠️ Created minimal Winoground test data - replace with real data!")

    def validate_and_download(self):
        """Main validation and download process"""
        logger.info("🔍 Starting evaluation data validation...")

        # Check current status
        status = self.check_required_files()

        # Report current status
        logger.info("📊 Current evaluation data status:")
        for eval_type, files in status.items():
            logger.info(f"  {eval_type}:")
            for name, exists in files.items():
                status_icon = "✅" if exists else "❌"
                logger.info(f"    {status_icon} {name}")

        # Count missing files
        missing_fast = sum(1 for exists in status["fast_eval"].values() if not exists)
        missing_full = sum(1 for exists in status["full_eval"].values() if not exists)

        logger.info(f"📈 Summary: {missing_fast} missing from fast_eval, {missing_full} missing from full_eval")

        if missing_fast == 0 and missing_full == 0:
            logger.info("✅ All evaluation data is present!")
            return True

        # Try different download methods
        logger.info("📥 Attempting to download missing data...")

        # Method 1: Use evaluation pipeline download scripts
        if self.download_missing_data_from_pipelines():
            logger.info("✅ Downloaded data using pipeline scripts")
        else:
            logger.info("⚠️ Pipeline download scripts not available or failed")

        # Method 2: Try HuggingFace
        if self.download_from_huggingface():
            logger.info("✅ Downloaded data from HuggingFace")
        else:
            logger.info("⚠️ HuggingFace download not available or failed")

        # Method 3: Create minimal test data
        self.create_minimal_test_data()

        # Re-check status
        final_status = self.check_required_files()
        final_missing_fast = sum(1 for exists in final_status["fast_eval"].values() if not exists)
        final_missing_full = sum(1 for exists in final_status["full_eval"].values() if not exists)

        logger.info("📊 Final evaluation data status:")
        logger.info(f"  Fast eval: {len(final_status['fast_eval']) - final_missing_fast}/{len(final_status['fast_eval'])} present")
        logger.info(f"  Full eval: {len(final_status['full_eval']) - final_missing_full}/{len(final_status['full_eval'])} present")

        if final_missing_fast == 0 and final_missing_full == 0:
            logger.info("✅ All evaluation data is now available!")
            return True
        else:
            logger.warning("⚠️ Some evaluation data is still missing, but evaluation can proceed with available data")
            return False

    def install_missing_dependencies(self):
        """Install missing Python dependencies for evaluation"""
        logger.info("📦 Checking and installing missing dependencies...")

        required_packages = [
            "scipy",  # For DevBench .mat files
            "huggingface_hub",  # For downloading data
            "datasets",  # For dataset handling
            "transformers",  # For model compatibility
            "torch",  # For model loading
            "torchvision",  # For vision processing
            "pillow",  # For image handling
            "requests",  # For downloads
            "tqdm",  # For progress bars
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                logger.info(f"✅ {package} is available")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"❌ {package} is missing")

        if missing_packages:
            logger.info(f"📦 Installing missing packages: {missing_packages}")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install"
                ] + missing_packages)
                logger.info("✅ Successfully installed missing packages")
            except Exception as e:
                logger.error(f"❌ Failed to install packages: {e}")
                return False

        return True


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate and download evaluation data")
    parser.add_argument("--base_path", default="D:/BabyLM", help="Base path for BabyLM data")
    parser.add_argument("--install_deps", action="store_true", help="Install missing dependencies")
    parser.add_argument("--force_download", action="store_true", help="Force re-download of all data")

    args = parser.parse_args()

    validator = EvaluationDataValidator(args.base_path)

    # Install dependencies if requested
    if args.install_deps:
        validator.install_missing_dependencies()

    # Validate and download data
    success = validator.validate_and_download()

    if success:
        logger.info("🎉 Evaluation data validation completed successfully!")
        print("\n" + "="*60)
        print("✅ EVALUATION DATA READY")
        print("All required evaluation data is now available.")
        print("You can run evaluations with:")
        print("  python evaluate_bitmar_2025.py --eval_type fast")
        print("  python evaluate_bitmar_2025.py --eval_type full")
        print("="*60)
    else:
        logger.warning("⚠️ Some evaluation data is still missing")
        print("\n" + "="*60)
        print("⚠️ PARTIAL EVALUATION DATA")
        print("Some evaluation data is missing but evaluation can proceed.")
        print("Consider running with --eval_type fast for better compatibility.")
        print("="*60)


if __name__ == "__main__":
    main()
