"""
BitMar Evaluation Script for 2025 Pipeline
Evaluates BitMar models on both text and multimodal tasks using the 2025 evaluation pipeline
"""

import os
import sys
import argparse
import logging
import json
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_checkpoint(checkpoint_path: str, device: str = 'cuda:0'):
    """Load BitMar model from checkpoint"""
    try:
        # Add src to path for model loading
        sys.path.append(str(Path(__file__).parent / "src"))
        from src.model import create_bitmar_model

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract config and model state
        config = checkpoint.get('config', {})
        model_state = checkpoint['model_state_dict']

        # Create model with config
        model = create_bitmar_model(config['model'])
        model.load_state_dict(model_state)

        # Move to device
        device = torch.device(device)
        model = model.to(device)
        model.eval()

        logger.info(f"✅ Model loaded successfully on {device}")
        return model, config

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


def setup_evaluation_environment(pipeline_path: Path, eval_data_path: Path):
    """Setup evaluation environment for 2025 pipeline"""
    try:
        # Change to pipeline directory
        original_cwd = os.getcwd()
        os.chdir(pipeline_path)

        # Check if evaluation_data exists, if not create symlink
        eval_data_link = pipeline_path / "evaluation_data"

        if not eval_data_link.exists():
            logger.info(f"Creating evaluation data link: {eval_data_link} -> {eval_data_path}")
            try:
                eval_data_link.symlink_to(eval_data_path.resolve(), target_is_directory=True)
            except OSError:
                # Fallback to copying on Windows
                shutil.copytree(eval_data_path, eval_data_link)
                logger.info(f"Copied evaluation data to: {eval_data_link}")

        return original_cwd

    except Exception as e:
        logger.error(f"Failed to setup evaluation environment: {e}")
        raise


def run_text_evaluations(model_path: str, eval_type: str = "fast", output_dir: str = "results_2025"):
    """Run text-only evaluations (BLIMP, etc.)"""
    logger.info(f"🔤 Running text evaluations ({eval_type})...")

    results = {}

    try:
        # Determine eval script based on type
        if eval_type == "fast":
            eval_script = "eval_zero_shot_fast.sh"
        else:
            eval_script = "eval_zero_shot.sh"

        # Check if script exists
        if not Path(eval_script).exists():
            logger.warning(f"⚠️ Evaluation script not found: {eval_script}")
            return results

        # Run text evaluation
        cmd = [
            "bash", eval_script,
            "--model_path", model_path,
            "--output_dir", f"{output_dir}/text_results"
        ]

        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode == 0:
            logger.info("✅ Text evaluation completed successfully")

            # Try to parse results
            results_file = Path(f"{output_dir}/text_results/results.json")
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results['text'] = json.load(f)

        else:
            logger.warning(f"⚠️ Text evaluation failed: {result.stderr}")
            results['text'] = {"error": result.stderr}

    except Exception as e:
        logger.error(f"❌ Text evaluation error: {e}")
        results['text'] = {"error": str(e)}

    return results


def run_multimodal_evaluations(model_path: str, eval_type: str = "fast", output_dir: str = "results_2025"):
    """Run multimodal evaluations (VQA, Winoground, etc.)"""
    logger.info(f"🖼️ Running multimodal evaluations ({eval_type})...")

    results = {}

    try:
        # Check for multimodal evaluation script
        multimodal_scripts = [
            "eval_multimodal.sh",
            "evaluation_pipeline/multimodal/run_evaluation.py"
        ]

        eval_script = None
        for script in multimodal_scripts:
            if Path(script).exists():
                eval_script = script
                break

        if not eval_script:
            logger.warning("⚠️ No multimodal evaluation script found")
            return results

        # Run multimodal evaluation
        if eval_script.endswith('.py'):
            cmd = [
                sys.executable, eval_script,
                "--model_path", model_path,
                "--eval_type", eval_type,
                "--output_dir", f"{output_dir}/multimodal_results"
            ]
        else:
            cmd = [
                "bash", eval_script,
                "--model_path", model_path,
                "--eval_type", eval_type,
                "--output_dir", f"{output_dir}/multimodal_results"
            ]

        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode == 0:
            logger.info("✅ Multimodal evaluation completed successfully")

            # Try to parse results
            results_file = Path(f"{output_dir}/multimodal_results/results.json")
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results['multimodal'] = json.load(f)
        else:
            logger.warning(f"⚠️ Multimodal evaluation failed: {result.stderr}")
            results['multimodal'] = {"error": result.stderr}

    except Exception as e:
        logger.error(f"❌ Multimodal evaluation error: {e}")
        results['multimodal'] = {"error": str(e)}

    return results


def run_devbench_evaluation(model_path: str, output_dir: str = "results_2025"):
    """Run DevBench evaluation"""
    logger.info("🧪 Running DevBench evaluation...")

    results = {}

    try:
        # Check for DevBench script
        devbench_script = "eval_devbench.sh"

        if not Path(devbench_script).exists():
            logger.warning("⚠️ DevBench evaluation script not found")
            return results

        # Run DevBench evaluation
        cmd = [
            "bash", devbench_script,
            "--model_path", model_path,
            "--output_dir", f"{output_dir}/devbench_results"
        ]

        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode == 0:
            logger.info("✅ DevBench evaluation completed successfully")

            # Try to parse results
            results_file = Path(f"{output_dir}/devbench_results/results.json")
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results['devbench'] = json.load(f)
        else:
            logger.warning(f"⚠️ DevBench evaluation failed: {result.stderr}")
            results['devbench'] = {"error": result.stderr}

    except Exception as e:
        logger.error(f"❌ DevBench evaluation error: {e}")
        results['devbench'] = {"error": str(e)}

    return results


def evaluate_bitmar_2025(
    model_path: str,
    eval_type: str = "fast",
    evaluation_pipeline_path: str = "../evaluation-pipeline-2025",
    output_dir: str = "results_2025"
):
    """
    Evaluate BitMar model using 2025 pipeline (text + multimodal tasks)

    Args:
        model_path: Path to model checkpoint
        eval_type: "fast" or "full" evaluation
        evaluation_pipeline_path: Path to 2025 evaluation pipeline
        output_dir: Output directory for results
    """
    logger.info("🚀 Starting BitMar 2025 Pipeline Evaluation")
    logger.info(f"📁 Model: {model_path}")
    logger.info(f"🎯 Evaluation type: {eval_type}")
    logger.info(f"📂 Pipeline: {evaluation_pipeline_path}")
    logger.info(f"💾 Output: {output_dir}")

    # Convert paths
    pipeline_path = Path(evaluation_pipeline_path).resolve()
    model_path = Path(model_path).resolve()
    output_path = Path(output_dir)

    # Validate paths
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline path not found: {pipeline_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Find evaluation data
    eval_data_paths = [
        pipeline_path / "evaluation_data",
        pipeline_path.parent / "evaluation_data"
    ]

    eval_data_path = None
    for path in eval_data_paths:
        if path.exists():
            eval_data_path = path
            break

    if not eval_data_path:
        raise FileNotFoundError("Evaluation data not found. Run download_evaluation_data.py first.")

    logger.info(f"📊 Using evaluation data: {eval_data_path}")

    # Setup evaluation environment
    original_cwd = setup_evaluation_environment(pipeline_path, eval_data_path)

    try:
        all_results = {}

        # Run text evaluations
        text_results = run_text_evaluations(str(model_path), eval_type, str(output_path))
        all_results.update(text_results)

        # Run multimodal evaluations
        multimodal_results = run_multimodal_evaluations(str(model_path), eval_type, str(output_path))
        all_results.update(multimodal_results)

        # Run DevBench if available
        if eval_type == "full":
            devbench_results = run_devbench_evaluation(str(model_path), str(output_path))
            all_results.update(devbench_results)

        # Save combined results
        results_file = output_path / "combined_results_2025.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"📊 Results saved to: {results_file}")

        # Log summary
        logger.info("📈 Evaluation Summary:")
        for task_type, results in all_results.items():
            if isinstance(results, dict) and "error" not in results:
                logger.info(f"  ✅ {task_type}: Success")
            else:
                logger.info(f"  ❌ {task_type}: Failed")

        return all_results

    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate BitMar using 2025 pipeline")

    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--eval_type", type=str, choices=["fast", "full"], default="fast",
                       help="Type of evaluation to run")
    parser.add_argument("--evaluation_pipeline_path", type=str, default="../evaluation-pipeline-2025",
                       help="Path to 2025 evaluation pipeline")
    parser.add_argument("--output_dir", type=str, default="results_2025",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use for evaluation")

    args = parser.parse_args()

    try:
        results = evaluate_bitmar_2025(
            model_path=args.model_path,
            eval_type=args.eval_type,
            evaluation_pipeline_path=args.evaluation_pipeline_path,
            output_dir=args.output_dir
        )

        logger.info("🎉 2025 Pipeline evaluation completed successfully!")

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
