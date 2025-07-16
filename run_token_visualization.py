"""
Run Token-to-Pixel Attention Visualization for BitMar
Uses saved attention data from training to create token-pixel attention maps
"""

import numpy as np
import json
from pathlib import Path
import argparse
from visualize_token_attention import TokenPixelAttentionVisualizer
import matplotlib.pyplot as plt

def load_attention_data(attention_dir: str, step: int, layer: str = "layer_0") -> dict:
    """Load saved attention data for a specific step and layer"""
    
    attention_path = Path(attention_dir) / "token_pixel_attention"
    
    # Load attention weights
    attention_file = attention_path / f"attention_{layer}_step_{step}.npz"
    if not attention_file.exists():
        print(f"Attention file not found: {attention_file}")
        return None
        
    attention_data = np.load(attention_file)
    
    # Load input context
    context_dir = Path(attention_dir) / "input_context"
    context_file = context_dir / f"context_step_{step}.json"
    
    if not context_file.exists():
        print(f"Context file not found: {context_file}")
        return None
        
    with open(context_file, 'r') as f:
        context_data = json.load(f)
    
    return {
        'attention_weights': attention_data['attention_weights'],
        'tokens': context_data['input_ids'],
        'caption': context_data['caption'],
        'token_texts': context_data['tokens'],
        'step': step,
        'layer': layer,
        'shape': attention_data['shape']
    }

def create_single_step_visualization(attention_dir: str, step: int, layer: str = "layer_0"):
    """Create token-to-pixel visualization for a single training step"""
    
    # Load attention data
    data = load_attention_data(attention_dir, step, layer)
    if data is None:
        return
    
    # Initialize visualizer
    save_dir = Path(attention_dir) / "token_pixel_visualizations"
    visualizer = TokenPixelAttentionVisualizer(str(save_dir))
    
    # Prepare attention data in the format expected by visualizer
    attention_tensor_data = {
        'attention_weights': data['attention_weights'][np.newaxis, :, :],  # Add batch dimension
        'tokens': data['tokens'],
        'step': data['step']
    }
    
    # Create visualization
    print(f"Creating visualization for step {step}, layer {layer}")
    print(f"Caption: {data['caption']}")
    print(f"Attention shape: {data['shape']}")
    
    visualizer.visualize_caption_attention(
        caption=data['caption'],
        attention_data=attention_tensor_data,
        epoch=0,  # We don't track epochs in this data
        step=step,
        save_name=f"{layer}_single_step"
    )

def create_evolution_visualization(attention_dir: str, steps: list, layer: str = "layer_0", token_idx: int = 0):
    """Create attention evolution visualization across multiple steps"""
    
    attention_history = []
    caption = None
    
    for step in steps:
        data = load_attention_data(attention_dir, step, layer)
        if data is None:
            continue
            
        # Store caption from first successful load
        if caption is None:
            caption = data['caption']
            
        # Add to history
        attention_history.append({
            'attention_weights': data['attention_weights'][np.newaxis, :, :],  # Add batch dimension
            'tokens': data['tokens'],
            'step': step
        })
    
    if not attention_history:
        print("No attention data found for evolution visualization")
        return
        
    # Initialize visualizer
    save_dir = Path(attention_dir) / "token_pixel_visualizations"
    visualizer = TokenPixelAttentionVisualizer(str(save_dir))
    
    print(f"Creating evolution visualization for {len(attention_history)} steps")
    print(f"Caption: {caption}")
    print(f"Token index: {token_idx}")
    
    # Create evolution visualization
    visualizer.visualize_attention_evolution(
        attention_history=attention_history,
        caption=caption,
        token_idx=token_idx,
        save_name=f"{layer}_evolution"
    )
    
    # Create attention statistics and trends
    stats = visualizer.create_attention_statistics(
        attention_history=attention_history,
        caption=caption,
        save_name=f"{layer}_stats"
    )
    
    visualizer.plot_attention_trends(stats, save_name=f"{layer}_trends")

def find_available_steps(attention_dir: str) -> list:
    """Find all available training steps with saved attention data"""
    
    attention_path = Path(attention_dir) / "token_pixel_attention"
    if not attention_path.exists():
        return []
        
    # Find all attention files
    attention_files = list(attention_path.glob("attention_*_step_*.npz"))
    
    # Extract step numbers
    steps = []
    for file in attention_files:
        try:
            step_str = file.stem.split('_step_')[-1]
            step = int(step_str)
            steps.append(step)
        except (ValueError, IndexError):
            continue
            
    return sorted(set(steps))

def analyze_attention_patterns(attention_dir: str):
    """Analyze and summarize attention patterns across all saved data"""
    
    steps = find_available_steps(attention_dir)
    if not steps:
        print(f"No attention data found in {attention_dir}")
        return
        
    print(f"Found attention data for {len(steps)} training steps:")
    print(f"Steps: {steps}")
    
    # Load a sample to get caption and structure info
    sample_data = load_attention_data(attention_dir, steps[0])
    if sample_data:
        print(f"\\nSample caption: {sample_data['caption']}")
        print(f"Sequence length: {sample_data['shape'][1]} tokens")
        print(f"Vision dimension: {sample_data['shape'][2]} features")
        print(f"Token texts: {sample_data['token_texts'][:10]}...")  # Show first 10 tokens
    
    return steps, sample_data

def main():
    parser = argparse.ArgumentParser(description="Visualize BitMar token-to-pixel attention")
    parser.add_argument("--attention_dir", type=str, default="./attention_analysis",
                       help="Directory containing saved attention data")
    parser.add_argument("--mode", type=str, choices=['single', 'evolution', 'analyze'], 
                       default='analyze', help="Visualization mode")
    parser.add_argument("--step", type=int, help="Specific step for single visualization")
    parser.add_argument("--steps", type=int, nargs='+', help="Multiple steps for evolution")
    parser.add_argument("--layer", type=str, default="layer_0", help="Attention layer to visualize")
    parser.add_argument("--token_idx", type=int, default=0, help="Token index for evolution visualization")
    
    args = parser.parse_args()
    
    print(f"BitMar Token-to-Pixel Attention Visualizer")
    print(f"Attention directory: {args.attention_dir}")
    print(f"Mode: {args.mode}")
    
    if args.mode == 'analyze':
        # Analyze what data is available
        steps, sample_data = analyze_attention_patterns(args.attention_dir)
        
        if steps and len(steps) > 0:
            print(f"\\nTo create visualizations, use:")
            print(f"# Single step visualization:")
            print(f"python run_token_visualization.py --mode single --step {steps[0]} --layer {args.layer}")
            print(f"# Evolution across training:")
            print(f"python run_token_visualization.py --mode evolution --steps {' '.join(map(str, steps[:5]))} --layer {args.layer} --token_idx 1")
            
    elif args.mode == 'single':
        if args.step is None:
            print("Error: --step required for single mode")
            return
        create_single_step_visualization(args.attention_dir, args.step, args.layer)
        
    elif args.mode == 'evolution':
        if args.steps is None:
            # Use all available steps
            steps = find_available_steps(args.attention_dir)
            if not steps:
                print("No attention data found")
                return
            args.steps = steps[:10]  # Limit to first 10 steps for reasonable visualization
            
        create_evolution_visualization(args.attention_dir, args.steps, args.layer, args.token_idx)
        
    print("\\nVisualization complete!")

if __name__ == "__main__":
    main()
