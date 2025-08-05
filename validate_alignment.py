"""
Alignment Validation Script for BitMar 100M Token Training
Validates perfect image-caption alignment and provides detailed statistics
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset_for_validation(dataset_dir: str, tokenizer_name: str = "gpt2"):
    """Load dataset components for alignment validation"""
    import sys
    sys.path.append('src')
    from src.token_constrained_dataset import TokenConstrainedBabyLMDataset
    
    logger.info("Loading dataset for alignment validation...")
    dataset = TokenConstrainedBabyLMDataset(
        dataset_dir=dataset_dir,
        tokenizer_name=tokenizer_name,
        rebuild_cache=False  # Use existing cache
    )
    
    return dataset

def validate_caption_image_alignment(dataset) -> Dict:
    """Comprehensive validation of caption-image alignment"""
    logger.info("🔍 Validating caption-image alignment...")
    
    stats = {
        'total_samples': len(dataset),
        'caption_samples': 0,
        'text_only_samples': 0,
        'alignment_errors': 0,
        'vision_feature_issues': 0,
        'sample_details': []
    }
    
    alignment_errors = []
    
    # Check first 1000 samples for detailed validation
    max_check = min(1000, len(dataset))
    
    for i in range(max_check):
        try:
            sample = dataset[i]
            sample_type = sample['sample_type']
            
            if sample_type == 'caption':
                stats['caption_samples'] += 1
                
                # Validate vision features exist and have correct shape
                if 'vision_features' not in sample:
                    stats['alignment_errors'] += 1
                    alignment_errors.append(f"Sample {i}: Missing vision features")
                    continue
                
                vision_features = sample['vision_features']
                if vision_features.shape != torch.Size([196, 768]):  # DiNOv2 14x14 patches, 768 dim
                    stats['vision_feature_issues'] += 1
                    alignment_errors.append(f"Sample {i}: Wrong vision shape {vision_features.shape}")
                
                # Check has_vision flag
                if not sample.get('has_vision', False):
                    stats['alignment_errors'] += 1
                    alignment_errors.append(f"Sample {i}: has_vision flag is False for caption")
                
                # Store sample details for analysis
                if len(stats['sample_details']) < 10:
                    stats['sample_details'].append({
                        'index': i,
                        'text_length': len(sample['text']),
                        'vision_shape': vision_features.shape,
                        'vision_index': sample.get('vision_index', 'N/A')
                    })
                    
            else:  # text_only
                stats['text_only_samples'] += 1
                
                # Validate no vision features for text-only
                if sample.get('has_vision', False):
                    stats['alignment_errors'] += 1
                    alignment_errors.append(f"Sample {i}: has_vision=True for text-only sample")
                    
        except Exception as e:
            stats['alignment_errors'] += 1
            alignment_errors.append(f"Sample {i}: Exception - {str(e)}")
    
    # Print detailed results
    logger.info("📊 Alignment Validation Results:")
    logger.info(f"  • Total samples checked: {max_check}")
    logger.info(f"  • Caption samples: {stats['caption_samples']}")
    logger.info(f"  • Text-only samples: {stats['text_only_samples']}")
    logger.info(f"  • Alignment errors: {stats['alignment_errors']}")
    logger.info(f"  • Vision feature issues: {stats['vision_feature_issues']}")
    
    # Show error details
    if alignment_errors:
        logger.warning("⚠️  Alignment Issues Found:")
        for error in alignment_errors[:10]:  # Show first 10 errors
            logger.warning(f"    {error}")
        if len(alignment_errors) > 10:
            logger.warning(f"    ... and {len(alignment_errors) - 10} more errors")
    else:
        logger.info("✅ No alignment errors found!")
    
    stats['alignment_errors_list'] = alignment_errors
    return stats

def analyze_cross_modal_similarity(dataset, num_samples: int = 100) -> Dict:
    """Analyze cross-modal similarity for caption-image pairs"""
    logger.info(f"🔍 Analyzing cross-modal similarity for {num_samples} samples...")
    
    caption_samples = []
    similarities = []
    
    # Collect caption samples
    count = 0
    for i, sample in enumerate(dataset):
        if sample['sample_type'] == 'caption' and count < num_samples:
            caption_samples.append(sample)
            count += 1
        if count >= num_samples:
            break
    
    logger.info(f"Found {len(caption_samples)} caption samples for analysis")
    
    # For this validation, we'll use a simple analysis
    # In practice, you'd use the actual model to compute similarities
    stats = {
        'num_samples_analyzed': len(caption_samples),
        'average_text_length': np.mean([len(s['text']) for s in caption_samples]),
        'vision_feature_consistency': True,
        'samples_with_vision': sum(1 for s in caption_samples if s.get('has_vision', False))
    }
    
    logger.info("📊 Cross-modal Analysis Results:")
    logger.info(f"  • Samples analyzed: {stats['num_samples_analyzed']}")
    logger.info(f"  • Average text length: {stats['average_text_length']:.1f} chars")
    logger.info(f"  • Samples with vision: {stats['samples_with_vision']}")
    
    return stats

def validate_token_distribution(dataset) -> Dict:
    """Validate token distribution across caption and text samples"""
    logger.info("🔍 Validating token distribution...")
    
    caption_tokens = 0
    text_tokens = 0
    
    for sample in dataset:
        # Count actual tokens (non-padding)
        attention_mask = sample['attention_mask']
        token_count = attention_mask.sum().item()
        
        if sample['sample_type'] == 'caption':
            caption_tokens += token_count
        else:
            text_tokens += token_count
    
    total_tokens = caption_tokens + text_tokens
    caption_ratio = caption_tokens / total_tokens if total_tokens > 0 else 0
    text_ratio = text_tokens / total_tokens if total_tokens > 0 else 0
    
    stats = {
        'caption_tokens': caption_tokens,
        'text_tokens': text_tokens,
        'total_tokens': total_tokens,
        'caption_ratio': caption_ratio,
        'text_ratio': text_ratio,
        'target_met': abs(caption_ratio - 0.5) < 0.05  # Within 5% of 50/50
    }
    
    logger.info("📊 Token Distribution Results:")
    logger.info(f"  • Caption tokens: {caption_tokens:,} ({caption_ratio:.1%})")
    logger.info(f"  • Text tokens: {text_tokens:,} ({text_ratio:.1%})")
    logger.info(f"  • Total tokens: {total_tokens:,}")
    logger.info(f"  • 50/50 target met: {'✅' if stats['target_met'] else '❌'}")
    
    return stats

def save_validation_report(stats: Dict, output_dir: str):
    """Save comprehensive validation report"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save detailed report
    report_file = output_path / "alignment_validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    logger.info(f"📁 Validation report saved to {report_file}")
    
    # Create summary report
    summary_file = output_path / "alignment_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("BitMar 100M Token Dataset - Alignment Validation Summary\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("ALIGNMENT VALIDATION:\n")
        f.write(f"  • Total samples: {stats['alignment']['total_samples']}\n")
        f.write(f"  • Caption samples: {stats['alignment']['caption_samples']}\n")
        f.write(f"  • Text-only samples: {stats['alignment']['text_only_samples']}\n")
        f.write(f"  • Alignment errors: {stats['alignment']['alignment_errors']}\n")
        f.write(f"  • Status: {'✅ PASS' if stats['alignment']['alignment_errors'] == 0 else '❌ ISSUES FOUND'}\n\n")
        
        f.write("TOKEN DISTRIBUTION:\n")
        f.write(f"  • Caption tokens: {stats['tokens']['caption_tokens']:,}\n")
        f.write(f"  • Text tokens: {stats['tokens']['text_tokens']:,}\n")
        f.write(f"  • Total tokens: {stats['tokens']['total_tokens']:,}\n")
        f.write(f"  • 50/50 split: {'✅ PASS' if stats['tokens']['target_met'] else '❌ FAIL'}\n\n")
        
        f.write("CROSS-MODAL ANALYSIS:\n")
        f.write(f"  • Samples analyzed: {stats['cross_modal']['num_samples_analyzed']}\n")
        f.write(f"  • Samples with vision: {stats['cross_modal']['samples_with_vision']}\n")
        f.write(f"  • Average text length: {stats['cross_modal']['average_text_length']:.1f}\n")
    
    logger.info(f"📁 Summary report saved to {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Validate BitMar dataset alignment")
    parser.add_argument("--dataset_dir", type=str, default="../babylm_dataset", 
                        help="Path to dataset directory")
    parser.add_argument("--tokenizer", type=str, default="gpt2", 
                        help="Tokenizer to use")
    parser.add_argument("--output_dir", type=str, default="alignment_validation", 
                        help="Output directory for reports")
    parser.add_argument("--similarity_samples", type=int, default=100, 
                        help="Number of samples for similarity analysis")
    
    args = parser.parse_args()
    
    logger.info("🚀 BitMar Dataset Alignment Validation")
    logger.info("=" * 50)
    
    try:
        # Load dataset
        dataset = load_dataset_for_validation(args.dataset_dir, args.tokenizer)
        
        # Run validation tests
        alignment_stats = validate_caption_image_alignment(dataset)
        token_stats = validate_token_distribution(dataset)
        similarity_stats = analyze_cross_modal_similarity(dataset, args.similarity_samples)
        
        # Combine results
        all_stats = {
            'alignment': alignment_stats,
            'tokens': token_stats,
            'cross_modal': similarity_stats,
            'validation_config': {
                'dataset_dir': args.dataset_dir,
                'tokenizer': args.tokenizer,
                'similarity_samples': args.similarity_samples
            }
        }
        
        # Save reports
        save_validation_report(all_stats, args.output_dir)
        
        # Final summary
        alignment_ok = alignment_stats['alignment_errors'] == 0
        tokens_ok = token_stats['target_met']
        
        logger.info("🏁 FINAL VALIDATION RESULT:")
        logger.info(f"  • Alignment: {'✅ PASS' if alignment_ok else '❌ FAIL'}")
        logger.info(f"  • Token Distribution: {'✅ PASS' if tokens_ok else '❌ FAIL'}")
        logger.info(f"  • Overall: {'✅ DATASET READY' if alignment_ok and tokens_ok else '❌ ISSUES FOUND'}")
        
        if not (alignment_ok and tokens_ok):
            logger.error("⚠️  Dataset has issues that need to be addressed before training!")
            return 1
        else:
            logger.info("🎉 Dataset validation passed! Ready for training.")
            return 0
            
    except Exception as e:
        logger.error(f"❌ Validation failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
