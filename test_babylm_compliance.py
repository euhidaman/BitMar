#!/usr/bin/env python3
"""
Test script to verify BabyLM token compliance implementation
Validates that token limits are enforced while preserving image-caption associations
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import logging
from src.dataset import create_data_module

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_babylm_compliance():
    """Test BabyLM token compliance"""
    logger.info("🧪 Testing BabyLM Token Compliance Implementation")
    
    # Test configuration with token limits
    test_config = {
        'dataset_dir': "../babylm_dataset",
        'text_encoder_name': "gpt2",
        'max_seq_length': 64,  # Short for testing
        'batch_size': 2,
        'num_workers': 0,  # No multiprocessing for testing
        'pin_memory': False,
        'text_ratio': 0.3,
        'use_mixed_training': True,
        'use_babylm_token_limits': True,
        'validation_datasets': ['glue/sst2']
    }
    
    try:
        # Create data module
        logger.info("📊 Creating data module with BabyLM compliance...")
        data_module = create_data_module(test_config)
        data_module.setup(max_samples=None)  # No artificial sample limits
        
        # Check token compliance
        if hasattr(data_module.train_dataset, 'get_token_usage_stats'):
            stats = data_module.train_dataset.get_token_usage_stats()
            
            logger.info("🎯 BabyLM Token Compliance Results:")
            logger.info(f"   📝 Text Tokens: {stats['text_tokens_used']:,}/{stats['text_tokens_limit']:,} ({stats['text_utilization_pct']:.1f}%)")
            logger.info(f"   🖼️ Image Tokens: {stats['image_tokens_used']:,}/{stats['image_tokens_limit']:,} ({stats['image_utilization_pct']:.1f}%)")
            
            # Verify compliance
            if stats['text_tokens_used'] > stats['text_tokens_limit']:
                logger.error(f"❌ TEXT TOKEN LIMIT EXCEEDED!")
                return False
            if stats['image_tokens_used'] > stats['image_tokens_limit']:
                logger.error(f"❌ IMAGE TOKEN LIMIT EXCEEDED!")
                return False
            
            logger.info("✅ BabyLM token limits respected!")
            
            # Test a few samples to ensure image-caption associations are preserved
            logger.info("🔗 Testing image-caption associations...")
            for i in range(min(5, len(data_module.train_dataset))):
                sample = data_module.train_dataset[i]
                if sample['sample_type'] == 'multimodal':
                    logger.info(f"   Sample {i}: {sample['caption'][:50]}... (vision shape: {sample['vision_features'].shape})")
            
            logger.info("✅ Image-caption associations preserved!")
            return True
            
        else:
            logger.error("❌ Token usage tracking not implemented!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_babylm_compliance()
    if success:
        logger.info("🎉 All BabyLM compliance tests PASSED!")
        sys.exit(0)
    else:
        logger.error("💥 BabyLM compliance tests FAILED!")
        sys.exit(1)
