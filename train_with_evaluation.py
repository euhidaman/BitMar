"""
Integrated Training and Evaluation Pipeline for BitMar
Automatically runs fast evaluation after each epoch and full evaluation at the end
"""

import os
import sys
import argparse
import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, Optional
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import the training script components
from train_100M_tokens import TokenAwareTrainer, logger

class IntegratedTrainingEvaluator:
    """Integrated trainer that automatically runs evaluations"""

    def __init__(self,
                 config_path: str,
                 device: Optional[str] = None,
                 save_every_n_steps: Optional[int] = None,
                 rebuild_cache: bool = False,
                 enable_fast_eval: bool = True,
                 enable_full_eval: bool = True,
                 eval_2025_path: str = "../evaluation-pipeline-2025",
                 eval_2024_path: str = "../evaluation-pipeline-2024",
                 evaluation_data_path: str = None):
        """
        Initialize integrated trainer with evaluation

        Args:
            config_path: Path to training configuration
            device: Device to use for training
            save_every_n_steps: Optional step-based checkpointing
            rebuild_cache: Whether to rebuild dataset cache
            enable_fast_eval: Whether to run fast evaluation after each epoch
            enable_full_eval: Whether to run full evaluation at the end
            eval_2025_path: Path to 2025 evaluation pipeline
            eval_2024_path: Path to 2024 evaluation pipeline
            evaluation_data_path: Path to evaluation data
        """
        self.config_path = config_path
        self.device = device
        self.save_every_n_steps = save_every_n_steps
        self.rebuild_cache = rebuild_cache
        self.enable_fast_eval = enable_fast_eval
        self.enable_full_eval = enable_full_eval
        self.eval_2025_path = Path(eval_2025_path)
        self.eval_2024_path = Path(eval_2024_path)
        self.evaluation_data_path = evaluation_data_path

        # Initialize trainer
        self.trainer = TokenAwareTrainer(config_path, device)
        self.trainer.rebuild_cache = rebuild_cache
        self.trainer.save_every_n_steps = save_every_n_steps

        # Setup evaluation directories
        self.eval_results_dir = Path("evaluation_results")
        self.eval_results_dir.mkdir(exist_ok=True)

        logger.info("🔄 Integrated Training & Evaluation Pipeline initialized")
        logger.info(f"  • Fast evaluation after epochs: {enable_fast_eval}")
        logger.info(f"  • Full evaluation at end: {enable_full_eval}")
        logger.info(f"  • 2025 pipeline: {self.eval_2025_path}")
        logger.info(f"  • 2024 pipeline: {self.eval_2024_path}")

    def run_epoch_evaluation(self, epoch: int, checkpoint_path: Path) -> Dict:
        """Run fast evaluation after an epoch"""
        if not self.enable_fast_eval:
            logger.info("Fast evaluation disabled, skipping...")
            return {}

        logger.info(f"🧪 Running fast evaluation after epoch {epoch}")

        results = {}

        try:
            # Run 2025 pipeline (text + multimodal fast evaluation)
            if self.eval_2025_path.exists():
                logger.info("Running 2025 pipeline fast evaluation...")
                result_2025 = self._run_evaluation_script(
                    script="evaluate_bitmar_2025.py",
                    model_path=checkpoint_path,
                    eval_type="fast",
                    output_suffix=f"epoch_{epoch}_2025"
                )
                results["2025_pipeline"] = result_2025
            else:
                logger.warning("2025 evaluation pipeline not found")

            # Run 2024 pipeline (multimodal only fast evaluation)
            if self.eval_2024_path.exists():
                logger.info("Running 2024 pipeline fast evaluation...")
                result_2024 = self._run_evaluation_script(
                    script="evaluate_bitmar_2024.py",
                    model_path=checkpoint_path,
                    eval_type="fast",
                    output_suffix=f"epoch_{epoch}_2024"
                )
                results["2024_pipeline"] = result_2024
            else:
                logger.warning("2024 evaluation pipeline not found")

            # Save combined results
            epoch_results_file = self.eval_results_dir / f"epoch_{epoch}_evaluation_results.json"
            with open(epoch_results_file, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"✅ Fast evaluation completed for epoch {epoch}")
            logger.info(f"Results saved to: {epoch_results_file}")

        except Exception as e:
            logger.error(f"❌ Fast evaluation failed for epoch {epoch}: {e}")
            results["error"] = str(e)

        return results

    def run_final_evaluation(self, final_checkpoint_path: Path) -> Dict:
        """Run full evaluation at the end of training"""
        if not self.enable_full_eval:
            logger.info("Full evaluation disabled, skipping...")
            return {}

        logger.info("🧪 Running full evaluation on final model")

        results = {}

        try:
            # Run 2025 pipeline (text + multimodal full evaluation)
            if self.eval_2025_path.exists():
                logger.info("Running 2025 pipeline full evaluation...")
                result_2025 = self._run_evaluation_script(
                    script="evaluate_bitmar_2025.py",
                    model_path=final_checkpoint_path,
                    eval_type="full",
                    output_suffix="final_2025"
                )
                results["2025_pipeline"] = result_2025
            else:
                logger.warning("2025 evaluation pipeline not found")

            # Run 2024 pipeline (multimodal only full evaluation)
            if self.eval_2024_path.exists():
                logger.info("Running 2024 pipeline full evaluation...")
                result_2024 = self._run_evaluation_script(
                    script="evaluate_bitmar_2024.py",
                    model_path=final_checkpoint_path,
                    eval_type="full",
                    output_suffix="final_2024"
                )
                results["2024_pipeline"] = result_2024
            else:
                logger.warning("2024 evaluation pipeline not found")

            # Save combined results
            final_results_file = self.eval_results_dir / "final_evaluation_results.json"
            with open(final_results_file, "w") as f:
                json.dump(results, f, indent=2)

            logger.info("✅ Full evaluation completed")
            logger.info(f"Results saved to: {final_results_file}")

        except Exception as e:
            logger.error(f"❌ Full evaluation failed: {e}")
            results["error"] = str(e)

        return results

    def _run_evaluation_script(self,
                             script: str,
                             model_path: Path,
                             eval_type: str,
                             output_suffix: str) -> Dict:
        """Run an evaluation script"""
        script_path = Path(__file__).parent / script

        if not script_path.exists():
            raise FileNotFoundError(f"Evaluation script not found: {script_path}")

        # Prepare command
        cmd = [
            sys.executable, str(script_path),
            "--model_path", str(model_path),
            "--eval_type", eval_type,
            "--output_dir", str(self.eval_results_dir / output_suffix)
        ]

        # Add evaluation data path if specified
        if self.evaluation_data_path:
            cmd.extend(["--evaluation_data_path", self.evaluation_data_path])

        # Add pipeline paths
        if "2025" in script:
            cmd.extend(["--evaluation_pipeline_path", str(self.eval_2025_path)])
        elif "2024" in script:
            cmd.extend(["--evaluation_pipeline_path", str(self.eval_2024_path)])

        try:
            logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hours timeout
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "stdout": result.stdout,
                    "command": " ".join(cmd)
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                    "stdout": result.stdout,
                    "command": " ".join(cmd)
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Evaluation timed out",
                "command": " ".join(cmd)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": " ".join(cmd)
            }

    def train_with_evaluation(self):
        """Run training with integrated evaluation"""
        logger.info("🚀 Starting integrated training with evaluation...")

        # Setup model and data
        self.trainer.setup_model_and_data()

        # Start carbon tracking
        if self.trainer.carbon_tracker:
            self.trainer.carbon_tracker.start()

        try:
            for epoch in range(self.trainer.config['training']['max_epochs']):
                logger.info(f"Starting epoch {epoch + 1}/{self.trainer.config['training']['max_epochs']}")

                self.trainer.current_epoch = epoch
                epoch_metrics = self.trainer.train_epoch(epoch)

                # Save checkpoint after each epoch
                self.trainer.save_token_checkpoint()

                # Get the checkpoint path for evaluation
                checkpoint_path = self.trainer.checkpoint_dir / f'checkpoint_epoch_{epoch}_tokens_{self.trainer.tokens_processed}.pt'

                # Run fast evaluation after each epoch
                eval_results = self.run_epoch_evaluation(epoch, checkpoint_path)

                # Log epoch summary to wandb with evaluation results
                if self.trainer.use_wandb:
                    try:
                        log_dict = {
                            'epoch/train_loss': epoch_metrics['train_loss'],
                            'epoch/cross_modal_similarity': epoch_metrics['cross_modal_similarity'],
                            'epoch/tokens_processed': self.trainer.tokens_processed,
                            'epoch/tokens_in_epoch': epoch_metrics['tokens_in_epoch'],
                            'epoch/number': epoch
                        }

                        # Add evaluation success flags
                        if eval_results:
                            log_dict['epoch/eval_2025_success'] = eval_results.get('2025_pipeline', {}).get('success', False)
                            log_dict['epoch/eval_2024_success'] = eval_results.get('2024_pipeline', {}).get('success', False)

                        import wandb
                        wandb.log(log_dict, step=self.trainer.global_step)
                    except Exception as e:
                        logger.warning(f"Failed to log epoch summary to wandb: {e}")
                        self.trainer.use_wandb = False

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            # Stop carbon tracking
            if self.trainer.carbon_tracker:
                emissions = self.trainer.carbon_tracker.stop()
                logger.info(f"🌱 Carbon emissions: {emissions:.6f} kg CO2")

            # Final checkpoint
            self.trainer.save_token_checkpoint()

            # Get final checkpoint path
            final_checkpoint = self.trainer.checkpoint_dir / 'latest_checkpoint.pt'

            # Run full evaluation on final model
            final_eval_results = self.run_final_evaluation(final_checkpoint)

            # Final cleanup (same as original trainer)
            if self.trainer.flops_tracker:
                try:
                    final_stats = self.trainer.flops_tracker.get_summary_stats()
                    logger.info("🔢 Final FLOPS Summary:")
                    logger.info(f"  • Total FLOPS: {final_stats.get('flops_formatted', 'N/A')}")
                    logger.info(f"  • Total training time: {final_stats.get('total_time', 0):.1f}s")

                    self.trainer.flops_tracker.save_statistics("final_flops_statistics.json")
                    self.trainer.flops_tracker.cleanup()
                    logger.info("✅ FLOPS tracking completed and cleaned up")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to complete FLOPS tracking: {e}")

            if self.trainer.memory_viz is not None:
                try:
                    self.trainer.memory_viz.generate_final_report()
                    logger.info("✅ Generated final memory visualization report")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to generate final memory report: {e}")

            # Final summary
            logger.info("🎯 Final Training Summary:")
            logger.info(f"  • Target tokens: {self.trainer.target_tokens:,}")
            logger.info(f"  • Processed tokens: {self.trainer.tokens_processed:,}")
            logger.info(f"  • Completion: {(self.trainer.tokens_processed/self.trainer.target_tokens)*100:.2f}%")
            logger.info(f"  • Best cross-modal similarity: {self.trainer.best_similarity:.4f}")

            if self.trainer.use_wandb:
                try:
                    import wandb
                    # Log final evaluation results to wandb
                    if final_eval_results:
                        wandb.log({
                            'final/eval_2025_success': final_eval_results.get('2025_pipeline', {}).get('success', False),
                            'final/eval_2024_success': final_eval_results.get('2024_pipeline', {}).get('success', False)
                        })
                    wandb.finish()
                except Exception as e:
                    logger.warning(f"Failed to finish wandb run: {e}")


def main():
    """Main function for integrated training and evaluation"""
    parser = argparse.ArgumentParser(description="Train BitMar with integrated evaluation")

    # Training arguments
    parser.add_argument("--config", type=str, default="configs/bitmar_100M_tokens.yaml",
                       help="Path to configuration file")
    parser.add_argument("--device", type=str, help="Device to use (cuda:0, cpu)")
    parser.add_argument("--rebuild_cache", action="store_true",
                       help="Rebuild token-constrained dataset cache")
    parser.add_argument("--save_every_n_steps", type=int, default=None,
                       help="Save checkpoint every N training steps (optional)")

    # Evaluation arguments
    parser.add_argument("--disable_fast_eval", action="store_true",
                       help="Disable fast evaluation after each epoch")
    parser.add_argument("--disable_full_eval", action="store_true",
                       help="Disable full evaluation at the end")
    parser.add_argument("--eval_2025_path", type=str, default="../evaluation-pipeline-2025",
                       help="Path to evaluation-pipeline-2025")
    parser.add_argument("--eval_2024_path", type=str, default="../evaluation-pipeline-2024",
                       help="Path to evaluation-pipeline-2024")
    parser.add_argument("--evaluation_data_path", type=str, default=None,
                       help="Path to evaluation_data directory")

    args = parser.parse_args()

    try:
        # Initialize integrated trainer
        integrated_trainer = IntegratedTrainingEvaluator(
            config_path=args.config,
            device=args.device,
            save_every_n_steps=args.save_every_n_steps,
            rebuild_cache=args.rebuild_cache,
            enable_fast_eval=not args.disable_fast_eval,
            enable_full_eval=not args.disable_full_eval,
            eval_2025_path=args.eval_2025_path,
            eval_2024_path=args.eval_2024_path,
            evaluation_data_path=args.evaluation_data_path
        )

        # Start integrated training with evaluation
        integrated_trainer.train_with_evaluation()

        logger.info("✅ Integrated training and evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Integrated training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
