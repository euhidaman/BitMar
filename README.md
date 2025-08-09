# BitMar: Vision-Language Episodic Memory Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

BitMar is a **Vision-Language Episodic Memory Transformer** designed for the BabyLM Challenge. It combines BitNet-quantized text processing, DiNOv2 vision encoding, and episodic memory mechanisms to achieve efficient multimodal understanding with exactly 100M tokens.

## 🌟 Key Features

- **Token-Constrained Training**: Exactly 100M tokens with perfect alignment
- **BitNet Quantization**: 1.58-bit quantized text encoder/decoder for efficient inference
- **Episodic Memory**: Cross-modal memory system for visual-text associations
- **Comprehensive Logging**: Detailed WandB visualizations and metrics tracking
- **Automatic Evaluation**: Built-in evaluation pipelines for both 2024 and 2025 tracks
- **Hugging Face Integration**: Automatic model uploads after each epoch
- **Carbon Tracking**: Environmental impact monitoring

## 🛠️ Installation

```bash
git clone <your-repo-url>
cd BitMar
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Training Commands

### Basic Training (100M Tokens)

```bash
# Standard training with all features enabled
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml

# Training with specific GPU device
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --device cuda:0

# Training with cache rebuild (if dataset changes)
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --rebuild_cache
```

### Training with Custom Checkpoint Frequency

```bash
# Save checkpoint every 1000 steps (in addition to epoch-based saves)
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --save_every_n_steps 1000

# Save checkpoint every 500 steps for frequent monitoring
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --save_every_n_steps 500
```

### Training with Evaluation Control

```bash
# Enable fast evaluation after each epoch (default: enabled)
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --enable_fast_eval

# Disable fast evaluation to speed up training
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --disable_fast_eval

# Enable full evaluation at the end (default: enabled)
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --enable_full_eval

# Disable full evaluation to save time
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --disable_full_eval

# Custom evaluation setup
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --disable_fast_eval --enable_full_eval
```

### Environment Variable Control

```bash
# Set evaluation flags via environment variables (useful for bash scripts)
export BITMAR_ENABLE_FAST_EVAL=true
export BITMAR_ENABLE_FULL_EVAL=false
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml
```

### WandB Complete Training Command with All Options

```bash
python train_100M_tokens.py \
    --config configs/bitmar_100M_tokens.yaml \
    --device cuda:0 \
    --save_every_n_steps 1000 \
    --enable_fast_eval \
    --enable_full_eval
```

## 📊 Comprehensive Visualization Guide

BitMar generates extensive visualizations and metrics during training to monitor model performance, memory utilization, attention patterns, and computational efficiency. This guide explains every graph and chart produced by the system.

### 🎯 Training Metrics Dashboard

#### Loss Curves
- **Training/Loss**: Primary training loss over time
  - **X-axis**: Training step
  - **Y-axis**: Cross-entropy loss value
  - **Interpretation**: Should decrease steadily; spikes indicate instability
  - **Expected Range**: 2.0-8.0 initially, converging to 1.5-3.0

- **Training/Learning_Rate**: Learning rate schedule visualization
  - **X-axis**: Training step
  - **Y-axis**: Learning rate value
  - **Interpretation**: Shows cosine annealing with restarts; restarts appear as sudden jumps back to higher values
  - **Pattern**: Smooth decay with periodic restarts every T_0 steps

#### Cross-Modal Alignment Metrics
- **Features/CrossModal_Similarity**: Cosine similarity between text and vision features
  - **X-axis**: Training step
  - **Y-axis**: Similarity score (0.0 to 1.0)
  - **Interpretation**: Higher values indicate better alignment; target >0.75
  - **Critical**: This is the key metric for multimodal understanding quality

### 🧠 Memory System Visualizations

#### Memory Utilization Tracking
- **Memory/Active_Slots_Percentage**: Percentage of memory slots actively used
  - **X-axis**: Training step
  - **Y-axis**: Percentage (0-100%)
  - **Interpretation**: Shows how efficiently the model uses available memory slots
  - **Healthy Range**: 60-90% (too low = underutilization, too high = potential conflicts)

- **Memory/Usage_Mean/Max/Min/Std**: Statistical distribution of memory slot usage
  - **Usage_Mean**: Average usage across all slots
  - **Usage_Max**: Most heavily used slot
  - **Usage_Min**: Least used slot
  - **Usage_Std**: Variance in usage (higher = more specialization)

#### Memory Evolution Heatmaps
- **Memory Slot Evolution**: 2D heatmap showing memory slot values over time
  - **X-axis**: Training step
  - **Y-axis**: Memory slot index (0 to memory_size-1)
  - **Color**: Memory activation strength (darker = higher activation)
  - **Interpretation**: Shows which slots specialize for different content types
  - **Patterns to Look For**: Vertical stripes indicate slot specialization

#### Memory Access Patterns
- **Memory/Top_1_Slot_Access** through **Memory/Top_5_Slot_Access**: Access frequency for most-used slots
  - **X-axis**: Training step
  - **Y-axis**: Access frequency
  - **Interpretation**: Shows which memory slots the model relies on most
  - **Healthy Pattern**: Gradual specialization with some slots becoming dominant

#### Memory Diversity and Specialization
- **Memory/Analysis_Avg_Similarity**: Average similarity between active memory slots
  - **Range**: 0.0-1.0 (lower = more diverse)
  - **Interpretation**: Measures how different memory slots are from each other
  - **Target**: 0.3-0.7 (too low = chaotic, too high = redundant)

- **Memory/Analysis_Age_Distribution**: Age of memory slots
  - **Avg_Age**: Average age of all slots
  - **Max_Age**: Oldest memory slot
  - **Interpretation**: Shows memory turnover rate; very old slots may indicate stagnation

### 🔍 Attention Analysis Visualizations

#### Multi-Head Attention Patterns
- **Attention/CrossModal_[Layer]_Mean/Max/Entropy**: Cross-modal attention statistics per layer
  - **Mean**: Average attention weight
  - **Max**: Peak attention weight
  - **Entropy**: Attention distribution sharpness (higher = more distributed)
  - **Interpretation**: Shows how focused the model's attention is

#### Attention Head Heatmaps
- **Attention Evolution Heatmaps**: 2D visualization of attention heads over time
  - **X-axis**: Training step
  - **Y-axis**: Attention head index
  - **Color**: Attention strength
  - **Interpretation**: Shows which heads become specialized for different tasks

#### Token-to-Pixel Attention Maps
- **Cross-Modal Attention Visualization**: Shows how text tokens attend to image regions
  - **Rows**: Text tokens
  - **Columns**: Image patch features
  - **Color Intensity**: Attention weight
  - **Interpretation**: Reveals which image regions are relevant for each word

### ⚡ Computational Efficiency Metrics

#### FLOPS Tracking
- **FLOPS/Total_Per_Step**: Floating point operations per training step
  - **X-axis**: Training step
  - **Y-axis**: FLOPS count (formatted: K, M, G, T)
  - **Interpretation**: Shows computational cost consistency
  - **Expected**: Should be relatively stable after initial warmup

- **FLOPS/Component_Breakdown**: FLOPS distribution across model components
  - **Components**: Attention, FFN, LayerNorm, Embeddings, Vision, Cross-Modal
  - **Visualization**: Stacked bar chart or pie chart
  - **Interpretation**: Shows which components are most computationally expensive

#### Throughput Analysis
- **FLOPS/Throughput**: Model processing speed
  - **X-axis**: Training step
  - **Y-axis**: Samples/second or FLOPS/second
  - **Interpretation**: Higher is better; drops may indicate bottlenecks
  - **Monitoring**: Look for consistent performance vs hardware limits

### 🎨 Feature Space Visualizations

#### Feature Distribution Analysis
- **Features/Text_Mean/Std/Norm**: Text feature statistics
- **Features/Vision_Mean/Std/Norm**: Vision feature statistics
- **Features/Episode_Mean/Std/Norm**: Episode feature statistics
  - **Mean**: Average activation (should be near 0 for normalized features)
  - **Std**: Feature variance (healthy range: 0.5-2.0)
  - **Norm**: Feature magnitude (indicates feature strength)

#### Feature Space Evolution
- **Feature Trajectory Plots**: 2D/3D PCA visualization of feature evolution
  - **Points**: Individual training steps
  - **Color**: Training progress (early=blue, late=red)
  - **Trajectory**: Shows how features evolve during training
  - **Interpretation**: Smooth trajectories indicate stable learning

### 📈 Gradient Flow Analysis

#### Gradient Magnitude Tracking
- **Gradients/Total_Norm**: Overall gradient magnitude
  - **X-axis**: Training step
  - **Y-axis**: Gradient norm
  - **Interpretation**: Should decrease over time; spikes indicate instability
  - **Warning Signs**: Values >1.0 or sudden spikes

#### Component-Wise Gradient Analysis
- **Gradients/[Component]_Norm**: Gradient norms for each model component
  - **Components**: Encoder, Decoder, Fusion, Memory, Vision, Projection
  - **Interpretation**: Shows which parts of the model are learning most actively
  - **Balanced Learning**: All components should have non-zero gradients

### 🔧 Quantization Monitoring

#### BitNet Quantization Effects
- **Quantization/Weight_Distribution**: Distribution of quantized weights
  - **Visualization**: Histogram showing weight value distribution
  - **Expected**: Should show discrete values (-1, 0, +1 for ternary)
  - **Quality Check**: Sharp peaks at expected quantization levels

- **Quantization/Activation_Range**: Range of activation values before quantization
  - **Interpretation**: Shows if activations are properly scaled for quantization
  - **Optimal Range**: Values should utilize full quantization range

### 🎯 Token Processing Analytics

#### Token Distribution Tracking
- **Tokens/Processed**: Cumulative token count
  - **X-axis**: Training step
  - **Y-axis**: Total tokens processed
  - **Target Line**: 100M tokens (dataset size)
  - **Interpretation**: Should increase linearly; shows progress toward token limit

- **Tokens/Batch_Size**: Tokens per batch (after padding removal)
  - **Interpretation**: Shows actual computational load per batch
  - **Efficiency**: Higher values indicate better padding efficiency

#### Token-Type Analysis
- **Token/Caption_vs_Text_Ratio**: Distribution of caption vs text-only tokens
  - **Target**: 50/50 split for balanced multimodal training
  - **Interpretation**: Deviations indicate data loading imbalance

### 🎪 Alignment Quality Metrics

#### Image-Caption Alignment
- **Alignment/Cosine_Similarity**: Similarity between image and caption embeddings
  - **Range**: -1.0 to 1.0 (higher = better alignment)
  - **Target**: >0.8 for well-aligned pairs
  - **Distribution**: Should show bimodal distribution (aligned vs misaligned)

- **Alignment/Retrieval_Accuracy**: Accuracy of cross-modal retrieval
  - **Text→Image**: How often correct image is retrieved for caption
  - **Image→Text**: How often correct caption is retrieved for image
  - **Interpretation**: Higher percentages indicate better cross-modal understanding

### 📊 Comprehensive Dashboard Layout

The complete BitMar dashboard organizes these visualizations into logical sections:

1. **Training Overview** (top row)
   - Loss curves, learning rate, basic metrics

2. **Memory System** (second row)
   - Utilization, evolution heatmaps, access patterns

3. **Attention Analysis** (third row)
   - Cross-modal attention, head specialization

4. **Computational Efficiency** (fourth row)
   - FLOPS tracking, throughput analysis

5. **Feature Analysis** (fifth row)
   - Feature distributions, gradient flow

6. **Quality Metrics** (bottom row)
   - Alignment scores, retrieval accuracy

### 🔍 Interpreting Common Patterns

#### Healthy Training Patterns
- **Smooth Loss Decrease**: Steady decline with occasional plateaus
- **Stable Memory Usage**: 60-90% slot utilization
- **Increasing Similarity**: Cross-modal similarity trending upward
- **Balanced Gradients**: All components showing learning activity

#### Warning Signs
- **Loss Spikes**: Sudden increases may indicate instability
- **Memory Saturation**: >95% utilization suggests inadequate capacity
- **Gradient Explosion**: Norms >1.0 indicate unstable training
- **Attention Collapse**: All heads showing similar patterns

#### Optimization Indicators
- **Memory Specialization**: Different slots showing distinct patterns
- **Attention Focus**: Increasing entropy in cross-modal attention
- **Feature Separation**: Clear clustering in feature space visualizations
- **Consistent Throughput**: Stable FLOPS/second measurements
