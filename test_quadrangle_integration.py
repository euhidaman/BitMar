"""
Test script for Quadrangle Attention integration with BitMar
Verifies that the new QFormer Quadrangle Attention mechanism works correctly
"""

import torch
import yaml
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import BitMarModel, LearnableQueryFusion
from quadrangle_attention import QuadrangleAttention, EpisodicQuadrangleProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_quadrangle_attention():
    """Test basic Quadrangle Attention functionality"""
    print("🔬 Testing Quadrangle Attention...")
    
    # Test parameters
    batch_size = 2
    text_seq_len = 32
    image_seq_len = 16
    dim = 256
    num_heads = 4
    
    # Create test data
    text_features = torch.randn(batch_size, text_seq_len, dim)
    image_features = torch.randn(batch_size, image_seq_len, dim)
    text_mask = torch.ones(batch_size, text_seq_len)
    image_mask = torch.ones(batch_size, image_seq_len)
    
    # Test Quadrangle Attention
    quad_attn = QuadrangleAttention(dim=dim, num_heads=num_heads)
    
    enhanced_text, enhanced_image, attention_maps = quad_attn(
        text_features, image_features, text_mask, image_mask
    )
    
    # Verify outputs
    assert enhanced_text.shape == text_features.shape
    assert enhanced_image.shape == image_features.shape
    assert 'image_to_text' in attention_maps
    assert 'text_to_image' in attention_maps
    assert 'image_to_image' in attention_maps
    assert 'text_to_text' in attention_maps
    assert 'pattern_weights' in attention_maps
    
    print("✅ Quadrangle Attention basic functionality test passed!")
    return True


def test_episodic_quadrangle_processor():
    """Test Episodic Quadrangle Processor with memory integration"""
    print("🧠 Testing Episodic Quadrangle Processor...")
    
    # Test parameters
    batch_size = 2
    text_seq_len = 32
    image_seq_len = 16
    dim = 256
    num_heads = 4
    memory_size = 64
    
    # Create test data
    text_features = torch.randn(batch_size, text_seq_len, dim)
    image_features = torch.randn(batch_size, image_seq_len, dim)
    
    # Test Episodic Quadrangle Processor
    processor = EpisodicQuadrangleProcessor(
        dim=dim,
        num_heads=num_heads,
        num_layers=2,
        memory_size=memory_size,
        episode_dim=dim // 2
    )
    
    # Test different modes
    modes = ["episodic_capture", "consolidation", "integration", "train"]
    
    for mode in modes:
        print(f"  Testing mode: {mode}")
        result = processor(text_features, image_features, mode=mode)
        
        # Verify outputs
        assert 'text_features' in result
        assert 'image_features' in result
        assert 'episode' in result
        assert 'attention_maps' in result
        assert result['text_features'].shape == text_features.shape
        assert result['image_features'].shape == image_features.shape
        
        print(f"    ✅ Mode {mode} passed!")
    
    print("✅ Episodic Quadrangle Processor test passed!")
    return True


def test_learnable_query_fusion():
    """Test updated LearnableQueryFusion with Quadrangle Attention"""
    print("🔗 Testing Enhanced LearnableQueryFusion...")
    
    # Test parameters
    batch_size = 2
    seq_len = 32
    text_dim = 256
    vision_dim = 256
    hidden_dim = 256
    
    # Create test data
    text_features = torch.randn(batch_size, seq_len, text_dim)
    vision_features = torch.randn(batch_size, vision_dim)
    
    # Test with Quadrangle Attention enabled
    fusion_quad = LearnableQueryFusion(
        text_dim=text_dim,
        vision_dim=vision_dim,
        hidden_dim=hidden_dim,
        num_queries=16,
        num_heads=4,
        num_layers=2,
        use_quadrangle=True,
        episodic_memory_size=64
    )
    
    # Test different modes
    modes = ["train", "episodic_capture", "consolidation", "integration"]
    
    for mode in modes:
        print(f"  Testing mode: {mode}")
        output, attention_maps = fusion_quad(text_features, vision_features, mode=mode)
        
        # Verify outputs
        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert isinstance(attention_maps, dict)
        
        if mode != "train":
            # Should have episodic attention maps
            assert 'episodic_episode' in attention_maps or len(attention_maps) > 0
        
        print(f"    ✅ Mode {mode} passed!")
    
    # Test with Quadrangle Attention disabled (fallback)
    fusion_classic = LearnableQueryFusion(
        text_dim=text_dim,
        vision_dim=vision_dim,
        hidden_dim=hidden_dim,
        num_queries=16,
        num_heads=4,
        num_layers=2,
        use_quadrangle=False
    )
    
    output_classic, attention_classic = fusion_classic(text_features, vision_features)
    assert output_classic.shape == (batch_size, seq_len, hidden_dim)
    print("  ✅ Fallback to classic fusion passed!")
    
    print("✅ Enhanced LearnableQueryFusion test passed!")
    return True


def test_bitmar_integration():
    """Test full BitMar model with Quadrangle Attention"""
    print("🤖 Testing BitMar Model Integration...")
    
    # Load config
    config_path = Path("configs/bitmar_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['model']
    
    # Create model
    model = BitMarModel(config)
    
    # Test parameters
    batch_size = 2
    seq_len = 32
    vision_dim = 768  # DiNOv2 features
    
    # Create test data
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    vision_features = torch.randn(batch_size, vision_dim)
    labels = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    # Test different modes
    modes = ["train", "episodic_capture", "consolidation", "integration", "inference"]
    
    for mode in modes:
        print(f"  Testing mode: {mode}")
        
        with torch.no_grad():  # Prevent gradient accumulation during testing
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_features=vision_features,
                labels=labels if mode != "inference" else None,
                mode=mode
            )
        
        # Verify outputs
        assert 'logits' in outputs
        assert outputs['logits'].shape == (batch_size, seq_len, config['vocab_size'])
        
        if mode != "inference":
            assert 'loss' in outputs
            assert outputs['loss'].item() >= 0  # Loss should be non-negative
        
        print(f"    ✅ Mode {mode} passed!")
    
    print("✅ BitMar Model Integration test passed!")
    return True


def run_all_tests():
    """Run all Quadrangle Attention tests"""
    print("🚀 Starting Quadrangle Attention Integration Tests...")
    print("=" * 60)
    
    tests = [
        test_quadrangle_attention,
        test_episodic_quadrangle_processor,
        test_learnable_query_fusion,
        test_bitmar_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Quadrangle Attention tests passed!")
        print("✨ QFormer Quadrangle Attention successfully integrated with BitMar!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
