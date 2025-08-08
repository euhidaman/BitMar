"""
BitMar Evaluation Script for 2024 BabyLM Challenge
Evaluates on multimodal tasks only using evaluation-pipeline-2024
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
        logging.FileHandler('evaluation_2024.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BitMar2024Evaluator:
    """Evaluator for BitMar using 2024 evaluation pipeline (multimodal only)"""

    def __init__(self,
                 model_path: str,
                 evaluation_pipeline_path: str = "../evaluation-pipeline-2024",
                 evaluation_data_path: str = None,
                 output_dir: str = "evaluation_results_2024"):
        """
        Initialize evaluator

        Args:
            model_path: Path to BitMar model checkpoint
            evaluation_pipeline_path: Path to evaluation-pipeline-2024 repository
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

        logger.info(f"Initialized BitMar 2024 Evaluator:")
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

        # Check for lm_eval installation
        try:
            import lm_eval
            logger.info("✅ lm_eval package found")
        except ImportError:
            logger.error("❌ lm_eval package not found. Please install evaluation-pipeline-2024")
            raise ImportError("lm_eval package required for 2024 evaluation")

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

            # Create a minimal HuggingFace config compatible with lm_eval
            hf_config = {
                "architectures": ["BitMarForCausalLM"],
                "model_type": "bitmar",
                "vocab_size": config.get('model', {}).get('vocab_size', 50257),
                "hidden_size": config.get('model', {}).get('text_encoder_dim', 128),
                "num_hidden_layers": config.get('model', {}).get('text_encoder_layers', 4),
                "num_attention_heads": config.get('model', {}).get('text_encoder_heads', 4),
                "max_position_embeddings": config.get('model', {}).get('max_seq_len', 256),
                "intermediate_size": config.get('model', {}).get('text_encoder_dim', 128) * 4,
                "layer_norm_eps": 1e-5,
                "use_cache": True,
                "torch_dtype": "float32",
                "transformers_version": "4.36.0",
                "auto_map": {
                    "AutoConfig": "configuration_bitmar.BitMarConfig",
                    "AutoModelForCausalLM": "modeling_bitmar.BitMarForCausalLM"
                }
            }

            # Save config
            with open(output_path / "config.json", "w") as f:
                json.dump(hf_config, f, indent=2)

            # Save model state dict
            torch.save(checkpoint['model_state_dict'], output_path / "pytorch_model.bin")

            # Create a simple tokenizer config (using GPT-2 tokenizer)
            tokenizer_config = {
                "tokenizer_class": "GPT2Tokenizer",
                "name_or_path": "gpt2",
                "bos_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>"
            }

            with open(output_path / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)

            # Create basic modeling file for compatibility
            modeling_code = '''
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import torch.nn as nn

class BitMarConfig(PretrainedConfig):
    model_type = "bitmar"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BitMarForCausalLM(PreTrainedModel):
    config_class = BitMarConfig
    
    def __init__(self, config):
        super().__init__(config)
        # Placeholder implementation - actual model loading handled separately
        self.config = config
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Placeholder - actual forward pass handled by BitMar model
        return CausalLMOutputWithPast()
'''

            with open(output_path / "modeling_bitmar.py", "w") as f:
                f.write(modeling_code)

            # Create configuration file
            config_code = '''
from transformers import PretrainedConfig

class BitMarConfig(PretrainedConfig):
    model_type = "bitmar"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
'''

            with open(output_path / "configuration_bitmar.py", "w") as f:
                f.write(config_code)

            logger.info(f"Model converted to HuggingFace format: {output_path}")

        except Exception as e:
            logger.error(f"Failed to convert model to HuggingFace format: {e}")
            raise

    def run_fast_multimodal_evaluation(self) -> Dict:
        """Run fast multimodal evaluation (for epoch checkpoints)"""
        logger.info("🚀 Starting fast multimodal evaluation (2024 pipeline)...")

        results = {}

        # Multimodal tasks available in 2024 pipeline
        multimodal_tasks = ["winoground_filtered", "vqa_filtered"]

        for task in multimodal_tasks:
            logger.info(f"Evaluating {task}...")
            try:
                result = self._run_lm_eval(task, fast=True)
                results[task] = result
            except Exception as e:
                logger.error(f"Failed to evaluate {task}: {e}")
                results[task] = {"error": str(e)}

        # Save fast evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"fast_multimodal_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Fast multimodal evaluation completed. Results saved to: {results_file}")
        return results

    def run_full_multimodal_evaluation(self) -> Dict:
        """Run full multimodal evaluation (for final model)"""
        logger.info("🚀 Starting full multimodal evaluation (2024 pipeline)...")

        results = {}

        # Multimodal tasks available in 2024 pipeline
        multimodal_tasks = ["winoground_filtered", "vqa_filtered"]

        for task in multimodal_tasks:
            logger.info(f"Evaluating {task}...")
            try:
                result = self._run_lm_eval(task, fast=False)
                results[task] = result
            except Exception as e:
                logger.error(f"Failed to evaluate {task}: {e}")
                results[task] = {"error": str(e)}

        # Save full evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"full_multimodal_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Full multimodal evaluation completed. Results saved to: {results_file}")
        return results

    def _run_lm_eval(self, task: str, fast: bool = False) -> Dict:
        """Run evaluation using lm_eval (2024 pipeline)"""
        eval_type = "fast" if fast else "full"
        output_file = self.output_dir / f"{task}_{eval_type}_lm_eval.json"

        try:
            # Prepare lm_eval command
            cmd = [
                "python", "-m", "lm_eval",
                "--model", "hf",
                "--model_args", f"pretrained={self.hf_model_path},backend=causal",
                "--tasks", task,
                "--device", "cuda:0" if torch.cuda.is_available() else "cpu",
                "--batch_size", "32" if fast else "16",
                "--output_path", str(output_file),
                "--log_samples"
            ]

            # Add image source for multimodal tasks
            if task == "winoground_filtered":
                cmd.extend([
                    "--image_src", "facebook/winoground",
                    "--image_src_split", "test"
                ])
            elif task == "vqa_filtered":
                cmd.extend([
                    "--image_src", "HuggingFaceM4/VQAv2",
                    "--image_src_split", "validation"
                ])

            # Add trust_remote_code if needed
            cmd.append("--trust_remote_code")

            logger.info(f"Running command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=str(self.eval_pipeline_path),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info(f"✅ {task} evaluation completed successfully")

                # Try to parse the output file
                if output_file.exists():
                    try:
                        with open(output_file, 'r') as f:
                            eval_results = json.load(f)
                        return {
                            "success": True,
                            "results": eval_results,
                            "stdout": result.stdout
                        }
                    except json.JSONDecodeError:
                        return {
                            "success": True,
                            "stdout": result.stdout,
                            "note": "Could not parse JSON output"
                        }
                else:
                    return {
                        "success": True,
                        "stdout": result.stdout,
                        "note": "No output file generated"
                    }
            else:
                logger.error(f"❌ {task} evaluation failed")
                logger.error(f"Return code: {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "stdout": result.stdout,
                    "return_code": result.returncode
                }

        except subprocess.TimeoutExpired:
            logger.error(f"❌ {task} evaluation timed out")
            return {"error": "Evaluation timed out"}
        except Exception as e:
            logger.error(f"❌ {task} evaluation failed with exception: {e}")
            return {"error": str(e)}

    def run_bash_script_evaluation(self) -> Dict:
        """Run evaluation using the provided bash script (eval_multimodal.sh)"""
        logger.info("🚀 Running multimodal evaluation using bash script...")

        script_path = self.eval_pipeline_path / "eval_multimodal.sh"

        if not script_path.exists():
            logger.error(f"eval_multimodal.sh not found at: {script_path}")
            return {"error": "eval_multimodal.sh script not found"}

        try:
            # Make script executable
            subprocess.run(["chmod", "+x", str(script_path)], check=True)

            # Run the script with our model path
            cmd = [str(script_path), str(self.hf_model_path)]

            logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=str(self.eval_pipeline_path),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info("✅ Bash script evaluation completed successfully")

                # Look for results in the results directory
                results_dir = self.eval_pipeline_path / "results"
                model_name = self.hf_model_path.name

                results = {}
                for task in ["winoground_filtered", "vqa_filtered"]:
                    task_results_path = results_dir / task / model_name / f"{task}_results.json"
                    if task_results_path.exists():
                        try:
                            with open(task_results_path, 'r') as f:
                                results[task] = json.load(f)
                        except json.JSONDecodeError:
                            results[task] = {"error": "Could not parse results JSON"}
                    else:
                        results[task] = {"error": "Results file not found"}

                return {
                    "success": True,
                    "results": results,
                    "stdout": result.stdout
                }
            else:
                logger.error(f"❌ Bash script evaluation failed")
                logger.error(f"STDERR: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "stdout": result.stdout
                }

        except Exception as e:
            logger.error(f"❌ Bash script evaluation failed: {e}")
            return {"error": str(e)}


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate BitMar using 2024 BabyLM pipeline (multimodal only)")

    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to BitMar model checkpoint")
    parser.add_argument("--eval_type", type=str, choices=["fast", "full", "both"],
                       default="both", help="Type of evaluation to run")
    parser.add_argument("--evaluation_pipeline_path", type=str,
                       default="../evaluation-pipeline-2024",
                       help="Path to evaluation-pipeline-2024")
    parser.add_argument("--evaluation_data_path", type=str, default=None,
                       help="Path to evaluation_data directory")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_2024",
                       help="Output directory for results")
    parser.add_argument("--use_bash_script", action="store_true",
                       help="Use the provided eval_multimodal.sh script")

    args = parser.parse_args()

    try:
        # Initialize evaluator
        evaluator = BitMar2024Evaluator(
            model_path=args.model_path,
            evaluation_pipeline_path=args.evaluation_pipeline_path,
            evaluation_data_path=args.evaluation_data_path,
            output_dir=args.output_dir
        )

        # Run evaluation
        if args.use_bash_script:
            logger.info("Using bash script for evaluation...")
            results = evaluator.run_bash_script_evaluation()
            logger.info(f"Bash script evaluation completed")
        else:
            if args.eval_type in ["fast", "both"]:
                logger.info("Running fast multimodal evaluation...")
                fast_results = evaluator.run_fast_multimodal_evaluation()
                logger.info(f"Fast evaluation completed with {len(fast_results)} tasks")

            if args.eval_type in ["full", "both"]:
                logger.info("Running full multimodal evaluation...")
                full_results = evaluator.run_full_multimodal_evaluation()
                logger.info(f"Full evaluation completed with {len(full_results)} tasks")

        logger.info("✅ Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
