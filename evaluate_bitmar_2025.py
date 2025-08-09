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
        # Create output directory
        text_output_dir = Path(output_dir) / "text_results"
        text_output_dir.mkdir(parents=True, exist_ok=True)

        # Convert BitMar checkpoint to HuggingFace format first
        hf_model_dir = f"hf_model_temp_text_{eval_type}"
        try:
            sys.path.append(str(Path(__file__).parent))
            from bitmar_hf_adapter import save_hf_compatible_model
            hf_model_path = save_hf_compatible_model(model_path, hf_model_dir)
            logger.info(f"✅ Created HuggingFace compatible model for text evaluation at: {hf_model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to create HF adapter for text evaluation: {e}")
            results['text'] = {"error": f"HF adapter failed: {e}"}
            return results

        # Instead of running the batch script, run individual tasks with better error handling
        eval_dir = "evaluation_data/fast_eval" if eval_type == "fast" else "evaluation_data/full_eval"

        # Define evaluation tasks to run
        if eval_type == "fast":
            tasks = [
                ("blimp", f"{eval_dir}/blimp_fast"),
                ("supplement", f"{eval_dir}/supplement_fast"),
                ("wug_adj", f"{eval_dir}/wug_adj_nominalization"),
                ("wug_past", f"{eval_dir}/wug_past_tense"),
                ("entity_tracking", f"{eval_dir}/entity_tracking_fast")
            ]
        else:
            tasks = [
                ("blimp", f"{eval_dir}/blimp_filtered"),
                ("supplement", f"{eval_dir}/supplement_filtered"),
                ("wug_adj", f"{eval_dir}/wug_adj_nominalization"),
                ("wug_past", f"{eval_dir}/wug_past_tense"),
                ("entity_tracking", f"{eval_dir}/entity_tracking")
            ]

        task_results = {}
        successful_tasks = 0

        for task_name, data_path in tasks:
            try:
                # Check if data path exists
                if not Path(data_path).exists():
                    logger.warning(f"⚠️ Data path not found: {data_path}")
                    task_results[task_name] = {"error": f"Data path not found: {data_path}"}
                    continue

                logger.info(f"Running {task_name} evaluation...")

                # Run individual task
                cmd = [
                    "python", "-m", "evaluation_pipeline.sentence_zero_shot.run",
                    "--model_path_or_name", hf_model_path,
                    "--backend", "causal",
                    "--task", task_name if task_name != "supplement" else "blimp",  # supplement uses blimp task
                    "--data_path", data_path,
                    "--save_predictions",
                    "--revision_name", "main"
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 minutes per task
                    cwd=Path.cwd()
                )

                if result.returncode == 0:
                    logger.info(f"✅ {task_name} evaluation completed successfully")
                    task_results[task_name] = {
                        "status": "completed",
                        "stdout": result.stdout[-500:] if result.stdout else ""
                    }
                    successful_tasks += 1
                else:
                    logger.warning(f"⚠️ {task_name} evaluation failed with return code {result.returncode}")
                    logger.warning(f"STDERR: {result.stderr[:500]}")
                    task_results[task_name] = {
                        "error": f"Return code {result.returncode}",
                        "stderr": result.stderr[:500] if result.stderr else "",
                        "stdout": result.stdout[:500] if result.stdout else ""
                    }

            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️ {task_name} evaluation timed out")
                task_results[task_name] = {"error": "Timeout after 30 minutes"}
            except Exception as e:
                logger.error(f"❌ {task_name} evaluation error: {e}")
                task_results[task_name] = {"error": str(e)}

        # Try reading evaluation if available
        try:
            reading_data_path = f"{eval_dir}/reading/reading_data.csv"
            if Path(reading_data_path).exists():
                logger.info("Running reading evaluation...")

                cmd = [
                    "python", "-m", "evaluation_pipeline.reading.run",
                    "--model_path_or_name", hf_model_path,
                    "--backend", "causal",
                    "--data_path", reading_data_path,
                    "--revision_name", "main"
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    cwd=Path.cwd()
                )

                if result.returncode == 0:
                    logger.info("✅ Reading evaluation completed successfully")
                    task_results["reading"] = {
                        "status": "completed",
                        "stdout": result.stdout[-500:] if result.stdout else ""
                    }
                    successful_tasks += 1
                else:
                    logger.warning(f"⚠️ Reading evaluation failed: {result.stderr[:500]}")
                    task_results["reading"] = {
                        "error": f"Return code {result.returncode}",
                        "stderr": result.stderr[:500] if result.stderr else ""
                    }
            else:
                logger.warning(f"⚠️ Reading data not found: {reading_data_path}")
                task_results["reading"] = {"error": f"Data path not found: {reading_data_path}"}

        except Exception as e:
            logger.error(f"❌ Reading evaluation error: {e}")
            task_results["reading"] = {"error": str(e)}

        # Collect prediction files
        predictions_dir = Path("predictions")
        prediction_files = []
        if predictions_dir.exists():
            prediction_files = list(predictions_dir.glob("*.json"))

        # Compile results
        if successful_tasks > 0:
            results['text'] = {
                "status": "partially_completed" if successful_tasks < len(tasks) else "completed",
                "successful_tasks": successful_tasks,
                "total_tasks": len(tasks) + 1,  # +1 for reading
                "task_results": task_results,
                "prediction_files": [str(f) for f in prediction_files]
            }
            logger.info(f"✅ Text evaluation completed: {successful_tasks}/{len(tasks)+1} tasks successful")
        else:
            results['text'] = {
                "error": "All tasks failed",
                "task_results": task_results
            }
            logger.error("❌ All text evaluation tasks failed")

        # Cleanup temp HF model
        if Path(hf_model_dir).exists():
            shutil.rmtree(hf_model_dir)

    except Exception as e:
        logger.error(f"❌ Text evaluation error: {e}")
        results['text'] = {"error": str(e)}
        # Cleanup temp HF model
        if 'hf_model_dir' in locals() and Path(hf_model_dir).exists():
            shutil.rmtree(hf_model_dir)

    return results


def run_multimodal_evaluations(model_path: str, eval_type: str = "fast", output_dir: str = "results_2025"):
    """Run multimodal evaluations (VQA, Winoground, etc.)"""
    logger.info(f"🖼️ Running multimodal evaluations ({eval_type})...")

    results = {}

    try:
        # Look for available multimodal evaluation tasks in the pipeline
        multimodal_tasks = []

        # Check for VQA evaluation
        vqa_data_dir = f"evaluation_data/{'fast_eval' if eval_type == 'fast' else 'full_eval'}/vqa_filtered"
        if Path(vqa_data_dir).exists():
            multimodal_tasks.append(("vqa", vqa_data_dir))

        # Check for Winoground evaluation
        winoground_data_dir = f"evaluation_data/{'fast_eval' if eval_type == 'fast' else 'full_eval'}/winoground_filtered"
        if Path(winoground_data_dir).exists():
            multimodal_tasks.append(("winoground", winoground_data_dir))

        if not multimodal_tasks:
            logger.warning("⚠️ No multimodal evaluation data found")
            results['multimodal'] = {"error": "No multimodal data found"}
            return results

        # Convert BitMar model to HuggingFace format for evaluation
        hf_model_dir = f"hf_model_temp_{eval_type}"
        try:
            from bitmar_hf_adapter import save_hf_compatible_model
            hf_model_path = save_hf_compatible_model(model_path, hf_model_dir)
            logger.info(f"✅ Created HuggingFace compatible model at: {hf_model_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create HF adapter: {e}")
            results['multimodal'] = {"error": f"HF adapter failed: {e}"}
            return results

        # Run each multimodal task
        task_results = {}
        for task_name, data_dir in multimodal_tasks:
            try:
                logger.info(f"Running {task_name} evaluation...")

                # Use the evaluation pipeline's multimodal runner
                cmd = [
                    "python", "-m", "evaluation_pipeline.sentence_zero_shot.run",
                    "--model_path_or_name", hf_model_path,
                    "--backend", "causal",
                    "--task", task_name,
                    "--data_path", data_dir,
                    "--save_predictions"
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

                if result.returncode == 0:
                    logger.info(f"✅ {task_name} evaluation completed")
                    task_results[task_name] = {"status": "completed", "stdout": result.stdout[:500]}
                else:
                    logger.warning(f"⚠️ {task_name} evaluation failed: {result.stderr[:500]}")
                    task_results[task_name] = {"error": result.stderr[:500]}

            except Exception as e:
                logger.error(f"❌ {task_name} evaluation error: {e}")
                task_results[task_name] = {"error": str(e)}

        results['multimodal'] = task_results

        # Cleanup temp HF model
        import shutil
        if Path(hf_model_dir).exists():
            shutil.rmtree(hf_model_dir)

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
