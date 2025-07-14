# Enhanced BitMar Training with Wandb Logging and Attention Visualization

## New Features Added

### 1. Enhanced Wandb Logging (`src/wandb_logger.py`)

Comprehensive logging system with properly labeled axes and categorized metrics:

**Categories:**
- `Training/` - Training loss, learning rate, epochs, steps
- `Validation/` - Validation loss, perplexity, additional metrics  
- `Memory/` - Episodic memory usage, entropy, slot utilization
- `Attention/` - Cross-modal attention, memory attention patterns
- `Features/` - Text features, vision features, episode representations
- `Quantization/` - BitNet weight distributions, scaling factors, sparsity
- `Gradients/` - Gradient norms by component (encoder, decoder, fusion, etc.)
- `Model/` - Parameter counts, model size, compression ratios

**Visualizations:**
- Memory usage/age heatmaps
- Attention distribution plots  
- Quantization weight distribution plots
- All with proper X/Y axis labels

### 2. Attention Head Visualization (`src/attention_visualizer.py`)

Inspired by lo-fit repository approach:

**Features:**
- Track importance of individual attention heads during training
- Create attention head heatmaps (similar to lo-fit paper)
- Save top-K attention heads like lo-fit methodology
- Timeline visualization of attention patterns
- Memory attention analysis
- Cross-modal attention pattern tracking

**Visualizations Created:**
- Attention head importance heatmaps (raw and sorted)
- Timeline plots showing attention evolution
- Individual head attention pattern visualization
- Memory access pattern analysis

### 3. Configuration Updates (`configs/bitmar_config.yaml`)

Added new sections:
```yaml
# Enhanced attention analysis configuration
attention_analysis:
  track_top_k: 20  # Track top 20 most important heads
  log_every_n_steps: 100  # Analyze attention every 100 steps
  viz_every_n_epochs: 2  # Create visualizations every 2 epochs
  save_head_patterns: true
  analyze_memory_attention: true
  analyze_cross_modal: true

# Enhanced wandb configuration
wandb:
  log_attention: true
  log_memory: true
  log_gradients: true
  log_quantization: true
  log_features: true
  create_plots: true
  plot_attention_heatmaps: true
  plot_memory_usage: true
  plot_quantization_dist: true
```

## Usage

### 1. Training with Enhanced Logging

```bash
# Make sure you have wandb API key set
export WANDB_API_KEY="your_api_key_here"

# Run training with enhanced logging
python train_bitmar.py --config configs/bitmar_config.yaml
```

### 2. Analyzing Attention Heads Post-Training

```bash
# Analyze all attention types
python analyze_attention_heads.py --analysis_dir ./attention_analysis --attention_type all

# Analyze specific attention type
python analyze_attention_heads.py --analysis_dir ./attention_analysis --attention_type encoder
```

### 3. What Gets Logged to Wandb

**Step-level metrics (every 50 steps):**
- Training loss, learning rate
- Memory usage statistics
- Attention entropy and concentration
- Cross-modal similarity scores
- Gradient norms by component
- Feature statistics

**Epoch-level metrics:**
- Validation loss and perplexity
- Epoch summaries
- Model compression ratios

**Visualizations (every 2 epochs):**
- Attention head importance heatmaps
- Memory usage/age heatmaps  
- Quantization distribution plots
- Attention timeline evolution

**Files Saved Locally:**
- `attention_analysis/top_heads_encoder_step_X.npy` - Top attention heads
- `attention_analysis/attention_heads_encoder_step_X.png` - Attention heatmaps
- `attention_analysis/attention_timeline_step_X.png` - Timeline plots
- `attention_analysis/reports/` - Comprehensive analysis reports

## Key Improvements Over Original

### 1. Proper Axis Labels
- All plots have clearly labeled X and Y axes
- Metrics are categorized (Training/, Memory/, Attention/, etc.)
- Consistent naming conventions

### 2. Lo-fit Style Attention Analysis
- Track individual attention head importance over training
- Save top-K heads for analysis (like lo-fit paper)
- Attention head consistency analysis
- Evolution tracking over training steps

### 3. Comprehensive Visualizations
- Memory heatmaps showing slot usage patterns
- Quantization distribution plots for BitNet analysis
- Cross-modal attention pattern visualization
- Timeline plots showing training evolution

### 4. Automated Reporting
- Generate markdown reports with attention analysis
- JSON files with detailed statistics
- Attention head stability analysis
- Consistency scoring across training

## Files Modified/Added

**New Files:**
- `src/wandb_logger.py` - Enhanced wandb logging system
- `src/attention_visualizer.py` - Attention head analysis system
- `analyze_attention_heads.py` - Post-training analysis utility

**Modified Files:**
- `train_bitmar.py` - Integrated new logging systems
- `configs/bitmar_config.yaml` - Added new configuration sections

## Sample Outputs

### Wandb Dashboard Categories:
```
Training/
├── Loss
├── Learning_Rate
├── Epoch
└── Step

Memory/
├── Usage_Mean
├── Active_Slots_Percentage
├── Analysis_Avg_Similarity
└── Top_5_Slot_Access

Attention/
├── CrossModal_layer_0_Entropy
├── Memory_Mean
└── Memory_Entropy

Features/
├── Text_Mean, Text_Std, Text_Norm
├── Vision_Mean, Vision_Std, Vision_Norm
└── CrossModal_Similarity

Quantization/
├── WeightScale_*
├── Sparsity_*
└── Compression_Ratio

Gradients/
├── Total_Norm
├── Encoder_Norm
├── Decoder_Norm
└── Fusion_Norm
```

### Attention Analysis Reports:
- Head consistency scores (which heads are consistently important)
- Layer-wise attention distribution
- Cross-modal attention patterns
- Memory access patterns over training

This enhanced system provides deep insights into BitMar's attention mechanisms and training dynamics, similar to the analysis capabilities in the lo-fit repository but tailored for the multimodal episodic memory architecture.
