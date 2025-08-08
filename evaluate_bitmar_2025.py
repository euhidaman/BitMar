"""
BitMar Evaluation Script for 2025 BabyLM Challenge
Evaluates on text-only and multimodal tasks using evaluation-pipeline-2025
"""

import os
import sys
import argparse
import logging
import subprocess
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import torch
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation_2025.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BitMar2025Evaluator:
    """Evaluator for BitMar using 2025 evaluation pipeline"""

    def __init__(self,
                 model_path: str,
                 evaluation_pipeline_path: str = "../evaluation-pipeline-2025",
                 evaluation_data_path: str = None,
                 output_dir: str = "evaluation_results_2025"):
        """
        Initialize evaluator

        Args:
            model_path: Path to BitMar model checkpoint
            evaluation_pipeline_path: Path to evaluation-pipeline-2025 repository
            evaluation_data_path: Path to evaluation_data directory (will auto-detect if None)
            output_dir: Directory to save evaluation results
        """
        self.model_path = Path(model_path)
        self.eval_pipeline_path = Path(evaluation_pipeline_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect evaluation data path if not provided
        if evaluation_data_path is None:
            # Look for evaluation_data in common locations
            possible_paths = [
                self.eval_pipeline_path / "evaluation_data",
                Path("../evaluation_data"),
                Path("./evaluation_data"),
                Path("../babylm_dataset/evaluation_data")
            ]
            for path in possible_paths:
                if path.exists():
                    self.evaluation_data_path = path
                    break
            else:
                raise FileNotFoundError(
                    "Could not find evaluation_data directory. Please download from OSF and specify path."
                )
        else:
            self.evaluation_data_path = Path(evaluation_data_path)

        logger.info(f"Initialized BitMar 2025 Evaluator:")
        logger.info(f"  • Model: {self.model_path}")
        logger.info(f"  • Pipeline: {self.eval_pipeline_path}")
        logger.info(f"  • Data: {self.evaluation_data_path}")
        logger.info(f"  • Output: {self.output_dir}")

        # Verify paths exist
        self._verify_paths()

        # Setup model for evaluation
        self._setup_model()

    def _verify_paths(self):
        """Verify all required paths exist"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        if not self.eval_pipeline_path.exists():
            raise FileNotFoundError(f"Evaluation pipeline not found: {self.eval_pipeline_path}")

        if not self.evaluation_data_path.exists():
            raise FileNotFoundError(f"Evaluation data not found: {self.evaluation_data_path}")

        # Check for fast_eval and full_eval directories
        self.fast_eval_path = self.evaluation_data_path / "fast_eval"
        self.full_eval_path = self.evaluation_data_path / "full_eval"

        if not self.fast_eval_path.exists():
            logger.warning(f"Fast eval directory not found: {self.fast_eval_path}")

        if not self.full_eval_path.exists():
            logger.warning(f"Full eval directory not found: {self.full_eval_path}")

    def _setup_model(self):
        """Setup model for evaluation (convert to HuggingFace format if needed)"""
        logger.info("Setting up model for evaluation...")

        # Check if model is already in HuggingFace format
        hf_model_path = self.model_path.parent / "hf_model"

        if not hf_model_path.exists():
            logger.info("Converting BitMar checkpoint to HuggingFace format...")
            self._convert_to_hf_format(hf_model_path)

        self.hf_model_path = hf_model_path
        logger.info(f"HuggingFace model ready at: {self.hf_model_path}")

    def _convert_to_hf_format(self, output_path: Path):
        """Convert BitMar checkpoint to HuggingFace format"""
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Load BitMar checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            config = checkpoint.get('config', {})

            # Create a minimal HuggingFace config
            hf_config = {
                "architectures": ["BitMarForCausalLM"],
                "model_type": "bitmar",
                "vocab_size": config.get('model', {}).get('vocab_size', 50257),
                "hidden_size": config.get('model', {}).get('text_encoder_dim', 128),
                "num_hidden_layers": config.get('model', {}).get('text_encoder_layers', 4),
                "num_attention_heads": config.get('model', {}).get('text_encoder_heads', 4),
                "max_position_embeddings": config.get('model', {}).get('max_seq_len', 256),
                "torch_dtype": "float32",
                "transformers_version": "4.36.0"
            }

            # Save config
            with open(output_path / "config.json", "w") as f:
                json.dump(hf_config, f, indent=2)

            # Save model state dict
            torch.save(checkpoint['model_state_dict'], output_path / "pytorch_model.bin")

            # Create a simple tokenizer config (using GPT-2 tokenizer)
            tokenizer_config = {
                "tokenizer_class": "GPT2Tokenizer",
                "name_or_path": "gpt2"
            }

            with open(output_path / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)

            logger.info(f"Model converted to HuggingFace format: {output_path}")

        except Exception as e:
            logger.error(f"Failed to convert model to HuggingFace format: {e}")
            raise

    def run_fast_evaluation(self) -> Dict:
        """Run fast evaluation (for epoch checkpoints)"""
        logger.info("🚀 Starting fast evaluation...")

        results = {}

        # Text-only tasks (fast)
        text_tasks = [
            "blimp_fast",
            "entity_tracking_fast",
            "supplement_fast",
            "wug_adj_nominalization",
            "wug_past_tense"
        ]

        for task in text_tasks:
            task_path = self.fast_eval_path / task
            if task_path.exists():
                logger.info(f"Evaluating {task}...")
                try:
                    result = self._run_text_evaluation(task, fast=True)
                    results[task] = result
                except Exception as e:
                    logger.error(f"Failed to evaluate {task}: {e}")
                    results[task] = {"error": str(e)}

        # Multimodal tasks (fast) - if available in fast_eval
        multimodal_tasks = ["winoground_filtered", "vqa_filtered"]

        for task in multimodal_tasks:
            # Check if task exists in fast_eval (might be symlink to full_eval)
            task_files = list(self.fast_eval_path.glob(f"*{task}*"))
            if task_files:
                logger.info(f"Evaluating {task} (multimodal)...")
                try:
                    result = self._run_multimodal_evaluation(task, fast=True)
                    results[task] = result
                except Exception as e:
                    logger.error(f"Failed to evaluate {task}: {e}")
                    results[task] = {"error": str(e)}

        # Save fast evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"fast_eval_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Fast evaluation completed. Results saved to: {results_file}")
        return results

    def run_full_evaluation(self) -> Dict:
        """Run full evaluation (for final model)"""
        logger.info("🚀 Starting full evaluation...")

        results = {}

        # Text-only tasks (full)
        text_tasks = [
            "blimp_filtered",
            "cdi_childes",
            "comps",
            "entity_tracking",
            "glue_filtered",
            "reading",
            "supplement_filtered",
            "wug_adj_nominalization",
            "wug_past_tense"
        ]

        for task in text_tasks:
            task_path = self.full_eval_path / task
            if task_path.exists():
                logger.info(f"Evaluating {task}...")
                try:
                    result = self._run_text_evaluation(task, fast=False)
                    results[task] = result
                except Exception as e:
                    logger.error(f"Failed to evaluate {task}: {e}")
                    results[task] = {"error": str(e)}

        # Multimodal tasks (full)
        multimodal_tasks = ["winoground_filtered", "vqa_filtered"]

        for task in multimodal_tasks:
            task_path = self.full_eval_path / task
            if task_path.exists():
                logger.info(f"Evaluating {task} (multimodal)...")
                try:
                    result = self._run_multimodal_evaluation(task, fast=False)
                    results[task] = result
                except Exception as e:
                    logger.error(f"Failed to evaluate {task}: {e}")
                    results[task] = {"error": str(e)}

        # Save full evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"full_eval_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Full evaluation completed. Results saved to: {results_file}")
        return results

    def _run_text_evaluation(self, task: str, fast: bool = False) -> Dict:
        """Run text-only evaluation using 2025 pipeline"""
        eval_type = "fast" if fast else "full"
        output_file = self.output_dir / f"{task}_{eval_type}_results.json"

        try:
            # Different evaluation methods based on task type
            if task.startswith("blimp"):
                return self._run_sentence_zero_shot(task, output_file)
            elif task == "reading":
                return self._run_reading_evaluation(task, output_file)
            elif task.startswith("wug"):
                return self._run_word_level_evaluation(task, output_file)
            elif task == "cdi_childes":
                return self._run_aoa_evaluation(task, output_file)
            elif task.startswith("glue"):
                return self._run_finetune_evaluation(task, output_file)
            else:
                # Default to sentence zero-shot
                return self._run_sentence_zero_shot(task, output_file)

        except Exception as e:
            logger.error(f"Text evaluation failed for {task}: {e}")
            return {"error": str(e)}

    def _run_multimodal_evaluation(self, task: str, fast: bool = False) -> Dict:
        """Run multimodal evaluation"""
        eval_type = "fast" if fast else "full"
        output_file = self.output_dir / f"{task}_{eval_type}_results.json"

        try:
            # Use multimodal evaluation pipeline
            cmd = [
                sys.executable, "-m", "evaluation_pipeline.multimodal.run",
                "--model_path", str(self.hf_model_path),
                "--task", task,
                "--output_file", str(output_file),
                "--data_path", str(self.evaluation_data_path)
            ]

            if fast:
                cmd.extend(["--subset", "fast"])

            result = subprocess.run(
                cmd,
                cwd=str(self.eval_pipeline_path),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )

            if result.returncode == 0:
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        return json.load(f)
                else:
                    return {"success": True, "output": result.stdout}
            else:
                logger.error(f"Multimodal evaluation failed: {result.stderr}")
                return {"error": result.stderr}

        except Exception as e:
            logger.error(f"Multimodal evaluation failed for {task}: {e}")
            return {"error": str(e)}

    def _run_sentence_zero_shot(self, task: str, output_file: Path) -> Dict:
        """Run sentence-level zero-shot evaluation"""
        try:
            cmd = [
                sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
                "--model_path", str(self.hf_model_path),
                "--task", task,
                "--output_file", str(output_file),
                "--data_path", str(self.evaluation_data_path)
            ]

            result = subprocess.run(
                cmd,
                cwd=str(self.eval_pipeline_path),
                capture_output=True,
                text=True,
                timeout=1800
            )

            if result.returncode == 0:
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        return json.load(f)
                else:
                    return {"success": True, "output": result.stdout}
            else:
                return {"error": result.stderr}

        except Exception as e:
            return {"error": str(e)}

    def _run_reading_evaluation(self, task: str, output_file: Path) -> Dict:
        """Run reading evaluation"""
        try:
            cmd = [
                sys.executable, "-m", "evaluation_pipeline.reading.run",
                "--model_path", str(self.hf_model_path),
                "--output_file", str(output_file),
                "--data_path", str(self.evaluation_data_path / "full_eval" / "reading")
            ]

            result = subprocess.run(
                cmd,
                cwd=str(self.eval_pipeline_path),
                capture_output=True,
                text=True,
                timeout=1800
            )

            if result.returncode == 0:
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        return json.load(f)
                else:
                    return {"success": True, "output": result.stdout}
            else:
                return {"error": result.stderr}

        except Exception as e:
            return {"error": str(e)}

    def _run_word_level_evaluation(self, task: str, output_file: Path) -> Dict:
        """Run word-level evaluation (for WUG tasks)"""
        try:
            cmd = [
                sys.executable, "-m", "evaluation_pipeline.AoA_word.run",
                "--model_path", str(self.hf_model_path),
                "--task", task,
                "--output_file", str(output_file),
                "--data_path", str(self.evaluation_data_path)
            ]

            result = subprocess.run(
                cmd,
                cwd=str(self.eval_pipeline_path),
                capture_output=True,
                text=True,
                timeout=1800
            )

            if result.returncode == 0:
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        return json.load(f)
                else:
                    return {"success": True, "output": result.stdout}
            else:
                return {"error": result.stderr}

        except Exception as e:
            return {"error": str(e)}

    def _run_aoa_evaluation(self, task: str, output_file: Path) -> Dict:
        """Run Age of Acquisition evaluation"""
        try:
            cmd = [
                sys.executable, "-m", "evaluation_pipeline.AoA_word.run",
                "--model_path", str(self.hf_model_path),
                "--task", "cdi_childes",
                "--output_file", str(output_file),
                "--data_path", str(self.evaluation_data_path / "full_eval" / "cdi_childes")
            ]

            result = subprocess.run(
                cmd,
                cwd=str(self.eval_pipeline_path),
                capture_output=True,
                text=True,
                timeout=1800
            )

            if result.returncode == 0:
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        return json.load(f)
                else:
                    return {"success": True, "output": result.stdout}
            else:
                return {"error": result.stderr}

        except Exception as e:
            return {"error": str(e)}

    def _run_finetune_evaluation(self, task: str, output_file: Path) -> Dict:
        """Run fine-tuning evaluation (for GLUE tasks)"""
        try:
            cmd = [
                sys.executable, "-m", "evaluation_pipeline.finetune.run",
                "--model_path", str(self.hf_model_path),
                "--task", task,
                "--output_file", str(output_file),
                "--data_path", str(self.evaluation_data_path / "full_eval" / "glue_filtered")
            ]

            result = subprocess.run(
                cmd,
                cwd=str(self.eval_pipeline_path),
                capture_output=True,
                text=True,
                timeout=3600  # Longer timeout for fine-tuning
            )

            if result.returncode == 0:
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        return json.load(f)
                else:
                    return {"success": True, "output": result.stdout}
            else:
                return {"error": result.stderr}

        except Exception as e:
            return {"error": str(e)}


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate BitMar using 2025 BabyLM pipeline")

    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to BitMar model checkpoint")
    parser.add_argument("--eval_type", type=str, choices=["fast", "full", "both"],
                       default="both", help="Type of evaluation to run")
    parser.add_argument("--evaluation_pipeline_path", type=str,
                       default="../evaluation-pipeline-2025",
                       help="Path to evaluation-pipeline-2025")
    parser.add_argument("--evaluation_data_path", type=str, default=None,
                       help="Path to evaluation_data directory")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_2025",
                       help="Output directory for results")

    args = parser.parse_args()

    try:
        # Initialize evaluator
        evaluator = BitMar2025Evaluator(
            model_path=args.model_path,
            evaluation_pipeline_path=args.evaluation_pipeline_path,
            evaluation_data_path=args.evaluation_data_path,
            output_dir=args.output_dir
        )

        # Run evaluation
        if args.eval_type in ["fast", "both"]:
            logger.info("Running fast evaluation...")
            fast_results = evaluator.run_fast_evaluation()
            logger.info(f"Fast evaluation completed with {len(fast_results)} tasks")

        if args.eval_type in ["full", "both"]:
            logger.info("Running full evaluation...")
            full_results = evaluator.run_full_evaluation()
            logger.info(f"Full evaluation completed with {len(full_results)} tasks")

        logger.info("✅ Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
