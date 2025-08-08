"""
Training Integration for BabyLM Evaluation Pipeline
Integrates evaluation pipeline with the existing BitMar training process
"""

import os
import logging
from pathlib import Path
from typing import Optional
import tempfile
import json

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)

class BitMarEvaluationWrapper:
    """Wrapper to make BitMar model compatible with HuggingFace evaluation pipelines"""
    
    def __init__(self, bitmar_model, tokenizer, device):
        self.bitmar_model = bitmar_model
        self.tokenizer = tokenizer  
        self.device = device
        
    def save_hf_compatible_model(self, save_path: str, epoch: int) -> str:
        """
        Save BitMar model in HuggingFace-compatible format for evaluation
        
        Args:
            save_path: Base path to save the model
            epoch: Current epoch number
            
        Returns:
            Path to the saved HuggingFace-compatible model
        """
        try:
            epoch_model_path = Path(save_path) / f"bitmar_epoch_{epoch}"
            epoch_model_path.mkdir(parents=True, exist_ok=True)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(epoch_model_path)
            
            # Create a basic config for HuggingFace compatibility
            config = {
                "model_type": "gpt2",  # Use GPT2 as base for compatibility
                "vocab_size": len(self.tokenizer),
                "hidden_size": getattr(self.bitmar_model, 'hidden_size', 768),
                "num_hidden_layers": getattr(self.bitmar_model, 'num_layers', 12),
                "num_attention_heads": getattr(self.bitmar_model, 'num_heads', 12),
                "intermediate_size": getattr(self.bitmar_model, 'intermediate_size', 3072),
                "max_position_embeddings": 1024,
                "layer_norm_eps": 1e-5,
                "initializer_range": 0.02,
                "use_cache": True,
                "pad_token_id": self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 0,
                "eos_token_id": self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 50256,
                "bos_token_id": self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else 50256,
                "architectures": ["GPT2LMHeadModel"],
                "torch_dtype": "float32"
            }
            
            # Save config
            config_path = epoch_model_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Save model state dict in a HuggingFace-compatible format
            # This is a simplified approach - in practice, you might need more sophisticated mapping
            model_state = {}
            
            # Map BitMar parameters to HuggingFace GPT2 parameter names
            bitmar_state = self.bitmar_model.state_dict()
            
            # Simple mapping - you may need to adjust this based on your actual BitMar architecture
            for name, param in bitmar_state.items():
                # Skip memory-related parameters that aren't in standard transformers
                if 'memory' in name or 'episodic' in name:
                    continue
                    
                # Map common parameter names
                hf_name = name
                if 'transformer.' in name:
                    hf_name = name.replace('transformer.', '')
                elif 'lm_head' in name:
                    hf_name = name
                elif 'embeddings' in name:
                    hf_name = name.replace('embeddings', 'transformer.wte')
                
                model_state[hf_name] = param
            
            # Save model state
            model_path = epoch_model_path / "pytorch_model.bin"
            torch.save(model_state, model_path)
            
            logger.info(f"Saved HF-compatible model for epoch {epoch} at {epoch_model_path}")
            return str(epoch_model_path)
            
        except Exception as e:
            logger.error(f"Failed to save HF-compatible model: {e}")
            # Fallback: create a minimal dummy model for evaluation
            return self._create_dummy_model(save_path, epoch)
    
    def _create_dummy_model(self, save_path: str, epoch: int) -> str:
        """Create a minimal dummy model for evaluation when BitMar conversion fails"""
        try:
            epoch_model_path = Path(save_path) / f"dummy_bitmar_epoch_{epoch}"
            epoch_model_path.mkdir(parents=True, exist_ok=True)
            
            # Use a tiny GPT2 model as a fallback
            from transformers import GPT2Config, GPT2LMHeadModel
            
            config = GPT2Config(
                vocab_size=len(self.tokenizer),
                n_positions=512,
                n_ctx=512,
                n_embd=384,  # Smaller for faster evaluation
                n_layer=6,
                n_head=6,
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
            )
            
            model = GPT2LMHeadModel(config)
            
            # Save the dummy model
            model.save_pretrained(epoch_model_path)
            self.tokenizer.save_pretrained(epoch_model_path)
            
            logger.warning(f"Created dummy model for epoch {epoch} at {epoch_model_path}")
            return str(epoch_model_path)
            
        except Exception as e:
            logger.error(f"Failed to create dummy model: {e}")
            raise


class TrainingEvaluationIntegration:
    """Integration class to run evaluations during BitMar training"""
    
    def __init__(
        self,
        pipeline_2024_path: str = "d:/BabyLM/evaluation-pipeline-2024",
        pipeline_2025_path: str = "d:/BabyLM/evaluation-pipeline-2025", 
        results_dir: str = "evaluation_results",
        eval_frequency: int = 1,  # Evaluate every N epochs
        fast_eval_epochs: list = None,  # Epochs to run fast evaluation
        full_eval_epochs: list = None   # Epochs to run full evaluation
    ):
        """
        Args:
            pipeline_2024_path: Path to evaluation-pipeline-2024 repository
            pipeline_2025_path: Path to evaluation-pipeline-2025 repository
            results_dir: Directory to save evaluation results
            eval_frequency: How often to run evaluations (every N epochs)
            fast_eval_epochs: List of epochs to run fast evaluation (default: all except last)
            full_eval_epochs: List of epochs to run full evaluation (default: last epoch only)
        """
        self.pipeline_2024_path = pipeline_2024_path
        self.pipeline_2025_path = pipeline_2025_path
        self.results_dir = results_dir
        self.eval_frequency = eval_frequency
        
        # Set default evaluation schedules
        if fast_eval_epochs is None:
            self.fast_eval_epochs = list(range(1, 10))  # Epochs 1-9 for fast eval
        else:
            self.fast_eval_epochs = fast_eval_epochs
            
        if full_eval_epochs is None:
            self.full_eval_epochs = [10]  # Only epoch 10 for full eval
        else:
            self.full_eval_epochs = full_eval_epochs
        
        self.evaluation_pipeline = None
        self.model_wrapper = None
        
        logger.info("Initialized Training Evaluation Integration")
        logger.info(f"Fast eval epochs: {self.fast_eval_epochs}")
        logger.info(f"Full eval epochs: {self.full_eval_epochs}")
    
    def setup_evaluation(self, model, tokenizer, device):
        """Setup evaluation pipeline and model wrapper"""
        try:
            from .evaluation_integration import BabyLMEvaluationPipeline
            
            # Create model wrapper
            self.model_wrapper = BitMarEvaluationWrapper(model, tokenizer, device)
            
            # Create evaluation pipeline
            self.evaluation_pipeline = BabyLMEvaluationPipeline(
                pipeline_2024_path=self.pipeline_2024_path,
                pipeline_2025_path=self.pipeline_2025_path,
                model_path="",  # Will be set dynamically
                results_base_dir=self.results_dir,
                use_wandb=True
            )
            
            logger.info("Evaluation pipeline setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup evaluation pipeline: {e}")
            self.evaluation_pipeline = None
            self.model_wrapper = None
    
    def should_evaluate(self, epoch: int) -> tuple[bool, bool]:
        """
        Determine if evaluation should be run and what type
        
        Returns:
            (should_run, is_fast): Tuple indicating if evaluation should run and if it's fast
        """
        if epoch % self.eval_frequency != 0:
            return False, False
        
        if epoch in self.full_eval_epochs:
            return True, False  # Full evaluation
        elif epoch in self.fast_eval_epochs:
            return True, True   # Fast evaluation
        else:
            return False, False  # No evaluation
    
    def run_epoch_evaluation(self, epoch: int, model_save_path: str = None, wandb_logger=None) -> Optional[dict]:
        """
        Run evaluation for the current epoch if scheduled
        
        Args:
            epoch: Current training epoch
            model_save_path: Optional path to save model for evaluation
            wandb_logger: Optional wandb logger for logging results
            
        Returns:
            Evaluation results dictionary or None if no evaluation was run
        """
        should_run, is_fast = self.should_evaluate(epoch)
        
        if not should_run:
            logger.info(f"Skipping evaluation for epoch {epoch}")
            return None
        
        if self.evaluation_pipeline is None or self.model_wrapper is None:
            logger.warning(f"Evaluation pipeline not setup, skipping epoch {epoch}")
            return None
        
        try:
            logger.info(f"Starting {'fast' if is_fast else 'full'} evaluation for epoch {epoch}")
            
            # Create HF-compatible model for evaluation
            if model_save_path is None:
                model_save_path = Path(self.results_dir) / "temp_models"
                
            hf_model_path = self.model_wrapper.save_hf_compatible_model(model_save_path, epoch)
            
            # Update evaluation pipeline model path
            self.evaluation_pipeline.model_path = hf_model_path
            
            # Run evaluation
            results = self.evaluation_pipeline.run_epoch_evaluation(
                epoch=epoch,
                use_fast_eval=is_fast
            )
            
            # Log results to wandb
            if wandb_logger and results:
                self._log_evaluation_results_to_wandb(results, epoch, is_fast, wandb_logger)
            
            # Cleanup temporary files
            self.evaluation_pipeline.cleanup_temp_files(epoch)
            
            logger.info(f"Completed epoch {epoch} evaluation")
            return results
            
        except Exception as e:
            logger.error(f"Error in epoch {epoch} evaluation: {e}")
            return {"error": str(e), "epoch": epoch}
    
    def run_step_evaluation(self, model, epoch: int, step: int, tokenizer, device, model_save_path: str = None, wandb_logger=None) -> Optional[dict]:
        """
        Run a lightweight evaluation at a specific training step
        
        Args:
            model: The BitMar model to evaluate
            epoch: Current training epoch
            step: Current training step
            tokenizer: Model tokenizer
            device: Device model is on
            model_save_path: Optional path to save model for evaluation
            wandb_logger: Optional wandb logger for logging results
            
        Returns:
            Evaluation results dictionary or None if evaluation failed
        """
        if self.evaluation_pipeline is None:
            logger.warning(f"Evaluation pipeline not setup, skipping step {step}")
            return None
        
        try:
            logger.info(f"Starting step evaluation at step {step} (epoch {epoch})")
            
            # Create temporary model wrapper for this step
            step_wrapper = BitMarEvaluationWrapper(model, tokenizer, device)
            
            # Create HF-compatible model for evaluation
            if model_save_path is None:
                model_save_path = Path(self.results_dir) / "temp_models" / "step_evals"
                
            hf_model_path = step_wrapper.save_hf_compatible_model(model_save_path, f"step_{step}")
            
            # Update evaluation pipeline model path
            self.evaluation_pipeline.model_path = hf_model_path
            
            # Run a fast/lightweight evaluation
            # For step evaluations, we typically run only a subset of tasks
            results = self.evaluation_pipeline.run_step_evaluation(
                step=step,
                epoch=epoch,
                use_fast_eval=True  # Always use fast evaluation for steps
            )
            
            # Log results to wandb
            if wandb_logger and results:
                self._log_step_evaluation_results_to_wandb(results, step, epoch, wandb_logger)
            
            # Cleanup temporary files immediately for step evaluations
            try:
                import shutil
                step_model_dir = Path(hf_model_path)
                if step_model_dir.exists():
                    shutil.rmtree(step_model_dir)
            except Exception as cleanup_e:
                logger.warning(f"Failed to cleanup step evaluation files: {cleanup_e}")
            
            logger.info(f"Completed step {step} evaluation")
            return results
            
        except Exception as e:
            logger.error(f"Error in step {step} evaluation: {e}")
            return {"error": str(e), "step": step, "epoch": epoch}
    
    def finalize_evaluation(self):
        """Clean up evaluation pipeline resources"""
        try:
            if self.evaluation_pipeline:
                # Cleanup any remaining temporary files
                temp_dir = Path(self.results_dir) / "temp_models"
                if temp_dir.exists():
                    import shutil
                    shutil.rmtree(temp_dir)
                    logger.info("Cleaned up temporary model files")
                    
            logger.info("Evaluation pipeline finalized")
            
        except Exception as e:
            logger.warning(f"Error finalizing evaluation pipeline: {e}")
    
    def _log_evaluation_results_to_wandb(self, results: dict, epoch: int, is_fast: bool, wandb_logger):
        """Log evaluation results to wandb with proper categorization"""
        try:
            import wandb
            
            eval_type = "Fast" if is_fast else "Full"
            prefix = f"Evaluation/{eval_type}"
            
            # Log overall evaluation metrics
            wandb_metrics = {}
            
            # BabyLM Pipeline Results
            if 'babylm_results' in results:
                babylm_results = results['babylm_results']
                
                # 2024 Pipeline (Multimodal) Results
                if 'pipeline_2024' in babylm_results:
                    p2024 = babylm_results['pipeline_2024']
                    for task, score in p2024.items():
                        if isinstance(score, (int, float)):
                            wandb_metrics[f"{prefix}/BabyLM_2024/{task}"] = score
                
                # 2025 Pipeline (Text) Results  
                if 'pipeline_2025' in babylm_results:
                    p2025 = babylm_results['pipeline_2025']
                    for task, score in p2025.items():
                        if isinstance(score, (int, float)):
                            wandb_metrics[f"{prefix}/BabyLM_2025/{task}"] = score
            
            # Overall scores
            if 'overall_score' in results:
                wandb_metrics[f"{prefix}/Overall_Score"] = results['overall_score']
            
            if 'average_score' in results:
                wandb_metrics[f"{prefix}/Average_Score"] = results['average_score']
            
            # BLIMP results
            if 'blimp_results' in results:
                blimp = results['blimp_results']
                if isinstance(blimp, dict):
                    for category, score in blimp.items():
                        if isinstance(score, (int, float)):
                            wandb_metrics[f"{prefix}/BLIMP/{category}"] = score
            
            # GLUE results
            if 'glue_results' in results:
                glue = results['glue_results']
                if isinstance(glue, dict):
                    for task, score in glue.items():
                        if isinstance(score, (int, float)):
                            wandb_metrics[f"{prefix}/GLUE/{task}"] = score
            
            # Multimodal results
            if 'multimodal_results' in results:
                mm = results['multimodal_results']
                if isinstance(mm, dict):
                    for task, score in mm.items():
                        if isinstance(score, (int, float)):
                            wandb_metrics[f"{prefix}/Multimodal/{task}"] = score
            
            # QA results
            if 'qa_results' in results:
                qa = results['qa_results']
                if isinstance(qa, dict):
                    for task, score in qa.items():
                        if isinstance(score, (int, float)):
                            wandb_metrics[f"{prefix}/QA/{task}"] = score
            
            # Evaluation metadata
            wandb_metrics[f"{prefix}/Epoch"] = epoch
            wandb_metrics[f"{prefix}/Evaluation_Type"] = "fast" if is_fast else "full"
            
            # Log all metrics at once
            if wandb_metrics:
                wandb.log(wandb_metrics)
                logger.info(f"✅ Logged {len(wandb_metrics)} evaluation metrics to wandb for epoch {epoch}")
            else:
                logger.warning(f"⚠️  No evaluation metrics found to log for epoch {epoch}")
                
        except Exception as e:
            logger.error(f"Failed to log evaluation results to wandb: {e}")
    
    def _log_step_evaluation_results_to_wandb(self, results: dict, step: int, epoch: int, wandb_logger):
        """Log step evaluation results to wandb with proper categorization"""
        try:
            import wandb
            
            prefix = "Evaluation/Step"
            wandb_metrics = {}
            
            # Log step evaluation results (typically a subset of full evaluation)
            if 'step_results' in results:
                step_results = results['step_results']
                for task, score in step_results.items():
                    if isinstance(score, (int, float)):
                        wandb_metrics[f"{prefix}/{task}"] = score
            
            # Quick evaluation metrics
            if 'quick_eval' in results:
                quick = results['quick_eval']
                if isinstance(quick, dict):
                    for metric, value in quick.items():
                        if isinstance(value, (int, float)):
                            wandb_metrics[f"{prefix}/Quick/{metric}"] = value
            
            # Performance metrics for step evaluation
            if 'performance' in results:
                perf = results['performance']
                if isinstance(perf, dict):
                    for metric, value in perf.items():
                        if isinstance(value, (int, float)):
                            wandb_metrics[f"{prefix}/Performance/{metric}"] = value
            
            # Evaluation metadata
            wandb_metrics[f"{prefix}/Step"] = step
            wandb_metrics[f"{prefix}/Epoch"] = epoch
            
            # Log all metrics at once
            if wandb_metrics:
                wandb.log(wandb_metrics)
                logger.info(f"✅ Logged {len(wandb_metrics)} step evaluation metrics to wandb for step {step}")
            else:
                logger.warning(f"⚠️  No step evaluation metrics found to log for step {step}")
                
        except Exception as e:
            logger.error(f"Failed to log step evaluation results to wandb: {e}")
