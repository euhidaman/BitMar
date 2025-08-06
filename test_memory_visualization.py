"""
Test script for Memory Visualization Integration
Validates that the memory visualization works correctly
"""

import sys
import torch
import numpy as np
from pathlib import Path
import yaml
import logging
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.memory_visualizer import EpisodicMemoryVisualizer
from src.memory_visualization_integration import MemoryVisualizationIntegration

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock wandb for testing
class MockWandB:
    def log(self, data, step=None):
        logger.info(f"Mock wandb.log: {list(data.keys())} at step {step}")
    
    def Image(self, image_data):
        return f"MockImage({type(image_data).__name__})"

# Patch wandb globally
sys.modules['wandb'] = MockWandB()


class MockModel:
    """Mock model for testing"""
    def __init__(self, memory_size=32, episode_dim=128):
        self.memory_slots = torch.randn(memory_size, episode_dim)
        
    def parameters(self):
        return [self.memory_slots]


def test_memory_visualizer():
    """Test the memory visualizer standalone"""
    logger.info("🧪 Testing Memory Visualizer...")
    
    # Create visualizer
    visualizer = EpisodicMemoryVisualizer(
        memory_size=32,
        episode_dim=128,
        snapshot_frequency=2,  # Very frequent for testing
        visualization_frequency=5,
        save_dir="test_memory_plots"
    )
    
    # Simulate memory evolution over several steps
    for epoch in range(3):
        for step in range(10):
            global_step = epoch * 10 + step
            
            # Create evolving memory (gradually becomes more structured)
            base_memory = torch.randn(32, 128)
            evolution_factor = global_step * 0.1
            
            # Make memory gradually more structured
            for i in range(32):
                if i < 16:  # First half becomes text-specialized
                    base_memory[i] = torch.randn(128) * (1 - evolution_factor) + \
                                   torch.ones(128) * evolution_factor
                else:  # Second half becomes vision-specialized
                    base_memory[i] = torch.randn(128) * (1 - evolution_factor) + \
                                   torch.ones(128) * -evolution_factor
            
            # Create mock episode types
            episode_types = ['text_only'] * 5 + ['multimodal'] * 3
            
            # Create mock access counts (some slots used more than others)
            access_counts = torch.zeros(32)
            preferred_slots = [0, 1, 16, 17]  # Prefer some slots
            for slot in preferred_slots:
                access_counts[slot] = torch.randint(5, 15, (1,)).item()
            
            # Log snapshot
            visualizer.log_memory_snapshot(
                memory_slots=base_memory,
                epoch=epoch,
                step=global_step,
                episode_types=episode_types,
                slot_access_counts=access_counts
            )
            
            logger.info(f"Logged snapshot: epoch {epoch}, step {global_step}")
    
    # Generate final report
    visualizer.generate_final_report()
    logger.info("✅ Memory visualizer test completed")


def test_integration():
    """Test the integration with mock config"""
    logger.info("🧪 Testing Memory Visualization Integration...")
    
    # Create mock config
    config = {
        'model': {
            'memory_size': 32,
            'episode_dim': 128
        },
        'output': {
            'memory_dir': 'test_memory_integration'
        },
        'wandb': {
            'log_memory_evolution': True,
            'memory_snapshot_frequency': 2,
            'memory_visualization_frequency': 5
        }
    }
    
    # Create mock model
    mock_model = MockModel()
    
    # Create integration
    integration = MemoryVisualizationIntegration(config, mock_model)
    
    # Test logging
    for epoch in range(2):
        for step in range(5):
            global_step = epoch * 5 + step
            
            # Mock batch
            batch = {
                'input_ids': torch.randint(0, 1000, (4, 10)),
                'has_vision': torch.tensor([True, True, False, False])
            }
            
            # Mock model outputs
            outputs = {
                'loss': torch.tensor(2.5),
                'text_features': torch.randn(4, 128),
                'vision_latent': torch.randn(4, 128)
            }
            
            # Log training step
            integration.log_training_step(
                batch=batch,
                epoch=epoch,
                step=global_step,
                model_outputs=outputs
            )
            
            logger.info(f"Logged training step: epoch {epoch}, step {global_step}")
    
    # Generate final report
    integration.generate_final_report()
    logger.info("✅ Integration test completed")


def main():
    """Run all tests"""
    logger.info("🚀 Starting Memory Visualization Tests...")
    
    try:
        # Test standalone visualizer
        test_memory_visualizer()
        
        # Test integration
        test_integration()
        
        logger.info("✅ All tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
