"""
Evaluation script for BitMar model
Tests vision-language understanding and episodic memory capabilities
"""

import sys
import torch
import yaml
import logging
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.model import create_bitmar_model
from src.dataset import create_data_module
from src.attention_analysis import AttentionAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BitMarEvaluator:
    """Evaluation suite for BitMar model"""
    
    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model()
        self.tokenizer = self.model.tokenizer
        
        # Setup data
        self.data_module = create_data_module(self.config['data'])
        self.data_module.setup(max_samples=1000)  # Limit for evaluation
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model(self):
        """Load trained model from checkpoint"""
        logger.info(f"Loading model from {self.model_path}")
        
        # Create model
        model = create_bitmar_model(self.config['model'])
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
    
    def evaluate_cross_modal_retrieval(self, num_samples: int = 100) -> Dict[str, float]:
        """Evaluate cross-modal retrieval capabilities"""
        logger.info("Evaluating cross-modal retrieval...")
        
        val_loader = self.data_module.val_dataloader()
        
        similarities = []
        correct_retrievals = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader)):
                if total_samples >= num_samples:
                    break
                
                # Move to device
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    vision_features=batch['vision_features']
                )
                
                # Get text and vision latents
                text_latent = outputs['text_latent']  # [batch_size, seq_len, dim]
                vision_latent = outputs['vision_latent']  # [batch_size, dim]
                
                # Pool text features
                text_pooled = text_latent.mean(dim=1)  # [batch_size, dim]
                
                # Compute similarities within batch
                batch_size = text_pooled.size(0)
                for i in range(batch_size):
                    if total_samples >= num_samples:
                        break
                    
                    # Current text and vision
                    current_text = text_pooled[i:i+1]  # [1, dim]
                    current_vision = vision_latent[i:i+1]  # [1, dim]
                    
                    # Compute similarity with all vision features in batch
                    similarities_to_all = torch.cosine_similarity(
                        current_text, vision_latent, dim=1
                    )  # [batch_size]
                    
                    # Check if highest similarity is with correct pair
                    best_match = torch.argmax(similarities_to_all).item()
                    if best_match == i:
                        correct_retrievals += 1
                    
                    # Store similarity score
                    correct_similarity = similarities_to_all[i].item()
                    similarities.append(correct_similarity)
                    
                    total_samples += 1
        
        # Compute metrics
        retrieval_accuracy = correct_retrievals / total_samples if total_samples > 0 else 0
        avg_similarity = np.mean(similarities) if similarities else 0
        
        results = {
            'retrieval_accuracy': retrieval_accuracy,
            'average_similarity': avg_similarity,
            'total_samples': total_samples,
            'correct_retrievals': correct_retrievals
        }
        
        logger.info(f"Cross-modal retrieval results:")
        logger.info(f"  Accuracy: {retrieval_accuracy:.4f}")
        logger.info(f"  Average similarity: {avg_similarity:.4f}")
        logger.info(f"  Samples evaluated: {total_samples}")
        
        return results
    
    def evaluate_memory_usage(self, num_samples: int = 100) -> Dict[str, float]:
        """Evaluate episodic memory usage patterns"""
        logger.info("Evaluating memory usage...")
        
        val_loader = self.data_module.val_dataloader()
        
        memory_patterns = []
        memory_entropies = []
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader):
                if total_samples >= num_samples:
                    break
                
                # Move to device
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    vision_features=batch['vision_features']
                )
                
                # Get memory attention patterns
                if outputs['memory_attention'] is not None:
                    memory_attention = outputs['memory_attention']  # [batch_size, memory_size]
                    
                    for i in range(memory_attention.size(0)):
                        if total_samples >= num_samples:
                            break
                        
                        pattern = memory_attention[i].cpu().numpy()
                        memory_patterns.append(pattern)
                        
                        # Compute entropy
                        entropy = -np.sum(pattern * np.log(pattern + 1e-8))
                        memory_entropies.append(entropy)
                        
                        total_samples += 1
        
        if memory_patterns:
            # Analyze patterns
            all_patterns = np.array(memory_patterns)
            
            results = {
                'average_entropy': np.mean(memory_entropies),
                'entropy_std': np.std(memory_entropies),
                'memory_usage_variance': np.var(all_patterns, axis=0).mean(),
                'most_used_slot': np.argmax(all_patterns.mean(axis=0)),
                'least_used_slot': np.argmin(all_patterns.mean(axis=0)),
                'usage_uniformity': 1.0 - np.std(all_patterns.mean(axis=0)) / np.mean(all_patterns),
                'total_samples': total_samples
            }
            
            logger.info(f"Memory usage results:")
            logger.info(f"  Average entropy: {results['average_entropy']:.4f}")
            logger.info(f"  Usage uniformity: {results['usage_uniformity']:.4f}")
            logger.info(f"  Most used slot: {results['most_used_slot']}")
            logger.info(f"  Least used slot: {results['least_used_slot']}")
        else:
            results = {'error': 'No memory patterns found'}
            logger.warning("No memory patterns found in evaluation")
        
        return results
    
    def evaluate_generation_quality(self, num_samples: int = 50) -> Dict[str, float]:
        """Evaluate text generation quality"""
        logger.info("Evaluating generation quality...")
        
        val_loader = self.data_module.val_dataloader()
        
        generated_texts = []
        reference_texts = []
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader):
                if total_samples >= num_samples:
                    break
                
                # Move to device
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)
                
                batch_size = batch['input_ids'].size(0)
                
                for i in range(batch_size):
                    if total_samples >= num_samples:
                        break
                    
                    # Use first half of text as input, second half as reference
                    input_ids = batch['input_ids'][i:i+1]
                    attention_mask = batch['attention_mask'][i:i+1]
                    vision_features = batch['vision_features'][i:i+1]
                    
                    # Find actual text length
                    text_length = attention_mask[0].sum().item()
                    half_length = text_length // 2
                    
                    # Split input
                    input_ids_partial = input_ids.clone()
                    input_ids_partial[0, half_length:] = self.tokenizer.pad_token_id
                    attention_mask_partial = attention_mask.clone()
                    attention_mask_partial[0, half_length:] = 0
                    
                    # Generate completion
                    generated = self.model.generate(
                        input_ids=input_ids_partial,
                        attention_mask=attention_mask_partial,
                        vision_features=vision_features,
                        max_length=text_length,
                        temperature=0.7,
                        top_p=0.9
                    )
                    
                    # Decode texts
                    generated_text = generated['generated_text'][0]
                    reference_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    
                    generated_texts.append(generated_text)
                    reference_texts.append(reference_text)
                    
                    total_samples += 1
        
        # Compute simple metrics (in practice, you'd use BLEU, ROUGE, etc.)
        avg_generated_length = np.mean([len(text.split()) for text in generated_texts])
        avg_reference_length = np.mean([len(text.split()) for text in reference_texts])
        
        # Simple vocabulary overlap metric
        vocab_overlaps = []
        for gen, ref in zip(generated_texts, reference_texts):
            gen_words = set(gen.lower().split())
            ref_words = set(ref.lower().split())
            overlap = len(gen_words.intersection(ref_words)) / len(ref_words.union(gen_words))
            vocab_overlaps.append(overlap)
        
        results = {
            'average_generated_length': avg_generated_length,
            'average_reference_length': avg_reference_length,
            'average_vocab_overlap': np.mean(vocab_overlaps),
            'total_samples': total_samples,
            'sample_generations': list(zip(generated_texts[:5], reference_texts[:5]))
        }
        
        logger.info(f"Generation quality results:")
        logger.info(f"  Avg generated length: {avg_generated_length:.2f} words")
        logger.info(f"  Avg reference length: {avg_reference_length:.2f} words")
        logger.info(f"  Avg vocabulary overlap: {np.mean(vocab_overlaps):.4f}")
        
        return results
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run all evaluation tasks"""
        logger.info("🚀 Starting comprehensive BitMar evaluation")
        logger.info("=" * 60)
        
        results = {}
        
        try:
            # Cross-modal retrieval
            logger.info("\n1️⃣ Cross-Modal Retrieval Evaluation")
            results['cross_modal_retrieval'] = self.evaluate_cross_modal_retrieval()
            
            # Memory usage
            logger.info("\n2️⃣ Memory Usage Evaluation")
            results['memory_usage'] = self.evaluate_memory_usage()
            
            # Generation quality
            logger.info("\n3️⃣ Generation Quality Evaluation")
            results['generation_quality'] = self.evaluate_generation_quality()
            
            # Attention analysis
            logger.info("\n4️⃣ Attention Analysis")
            analyzer = AttentionAnalyzer(
                model=self.model,
                tokenizer=self.tokenizer,
                save_dir="evaluation_attention"
            )
            
            # Analyze a few batches
            val_loader = self.data_module.val_dataloader()
            for i, batch in enumerate(val_loader):
                if i >= 10:  # Analyze 10 batches
                    break
                
                # Move to device
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)
                
                analyzer.extract_attention_patterns(batch)
            
            # Generate analysis report
            attention_report = analyzer.generate_report()
            results['attention_analysis'] = attention_report
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
        
        # Save results
        results_file = "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n💾 Results saved to: {results_file}")
        
        return results


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate BitMar model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bitmar_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model_path).exists():
        logger.error(f"Model checkpoint not found: {args.model_path}")
        logger.info("Available checkpoints:")
        checkpoint_dir = Path("checkpoints")
        if checkpoint_dir.exists():
            for ckpt in checkpoint_dir.glob("*.pt"):
                logger.info(f"  {ckpt}")
        else:
            logger.info("  No checkpoints directory found")
        return
    
    # Create evaluator and run evaluation
    evaluator = BitMarEvaluator(args.model_path, args.config)
    results = evaluator.run_comprehensive_evaluation()
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Evaluation Summary")
    logger.info("=" * 60)
    
    if 'cross_modal_retrieval' in results:
        retrieval = results['cross_modal_retrieval']
        logger.info(f"Cross-Modal Retrieval Accuracy: {retrieval.get('retrieval_accuracy', 0):.4f}")
    
    if 'memory_usage' in results:
        memory = results['memory_usage']
        logger.info(f"Memory Usage Entropy: {memory.get('average_entropy', 0):.4f}")
        logger.info(f"Memory Usage Uniformity: {memory.get('usage_uniformity', 0):.4f}")
    
    if 'generation_quality' in results:
        generation = results['generation_quality']
        logger.info(f"Generation Vocab Overlap: {generation.get('average_vocab_overlap', 0):.4f}")
    
    logger.info("\n🎉 Evaluation completed!")


if __name__ == "__main__":
    main()
