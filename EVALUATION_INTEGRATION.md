# BitMar - BabyLM 2025 Evaluation Pipeline Integration

This document describes how BitMar is integrated with the BabyLM 2025 evaluation pipeline for compatibility and evaluation.

## Overview

BitMar has been modified to be fully compatible with the BabyLM 2025 evaluation pipeline. The model is automatically converted to HuggingFace format after training for seamless evaluation.

## Quick Start

### 1. Validate Setup (Recommended)
```bash
python validate_setup.py
```
This checks all dependencies, GPU setup, data paths, and configuration.

### 2. Train the Model
```bash
python train_bitmar.py \
    --config configs/bitmar_10epoch_memory_optimized.yaml \
    --optimizer adamw \
    --epochs 10 \
    --wandb_project "bitmar-10epoch-memory-safe"
```

### 3. Prepare for Evaluation (Automatic)
The model is automatically converted to HuggingFace format during training and saved to:
- `checkpoints/hf_model/` - HuggingFace format model
- `final_model/` - Copy for evaluation

### 4. Manual Preparation (if needed)
```bash
python prepare_for_evaluation.py --generate_commands
```

## Evaluation Pipeline Integration

### Model Format Compatibility

BitMar is saved in HuggingFace format with:
- **Model Type**: `bitmar` (custom)
- **Compatible with**: `AutoModelForCausalLM`
- **Architecture**: Causal language model for evaluation
- **Tokenizer**: GPT-2 based

### Supported Evaluation Tasks

#### Zero-shot Evaluation
- **BLiMP**: Linguistic acceptability
- **EWoK**: Entity tracking  
- **Entity Tracking**: State tracking
- **WUG**: Morphological generalization
- **Reading**: Human alignment

#### Fine-tuning Evaluation
- **GLUE**: General language understanding
- **SuperGLUE**: Advanced language understanding

### Evaluation Commands

Navigate to the evaluation pipeline directory:
```bash
cd ../evaluation-pipeline-2025
```

#### Fast Zero-shot Evaluation
```bash
./eval_zero_shot_fast.sh "../BitMar/final_model" "checkpoint_1M" "causal"
```

#### Full Zero-shot Evaluation
```bash
./eval_zero_shot.sh "../BitMar/final_model" "causal"
```

#### Fine-tuning Evaluation
```bash
./eval_finetune.sh "../BitMar/final_model"
```

#### Manual Python Commands

**BLiMP Evaluation:**
```bash
python -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name "../BitMar/final_model" \
    --backend causal \
    --task blimp \
    --data_path "evaluation_data/fast_eval/blimp_filtered" \
    --save_predictions
```

**EWoK Evaluation:**
```bash
python -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name "../BitMar/final_model" \
    --backend causal \
    --task ewok \
    --data_path "evaluation_data/fast_eval/ewok_filtered" \
    --save_predictions
```

**GLUE Fine-tuning (example - SST-2):**
```bash
python -m evaluation_pipeline.finetune.run \
    --model_path_or_name "../BitMar/final_model" \
    --task_name "sst2" \
    --learning_rate 3e-5 \
    --batch_size 32 \
    --max_epochs 10
```

## Memory Optimization for RTX A6000

The configuration is optimized for RTX A6000 (48GB):

### Model Size Optimizations
- **Reduced dimensions**: 192D encoders/decoders (vs 256D)
- **Fewer layers**: 3 layers each (vs 4)
- **Smaller memory**: 16 slots (vs 32)
- **Compact fusion**: 1 layer (vs 2)

### Training Optimizations
- **Small batch size**: 8 (vs 32)
- **Short sequences**: 256 tokens (vs 512)
- **Minimal workers**: 2 workers
- **No pin memory**: Disabled for GPU memory
- **Conservative caching**: All caching disabled

### Memory Monitoring
- Automatic GPU memory monitoring
- Aggressive cleanup every 100 steps
- OOM prevention with 80% threshold warnings
- Device consistency verification

## Architecture Compatibility

### HuggingFace Integration

BitMar implements the required HuggingFace interfaces:

```python
# Configuration
class BitMarConfig(PretrainedConfig):
    model_type = "bitmar"

# Model
class BitMarForCausalLM(PreTrainedModel):
    config_class = BitMarConfig
    supports_gradient_checkpointing = True
```

### Evaluation Interface

The model provides standard causal LM interface:
- `forward()` - Standard forward pass with loss computation
- `generate()` - Text generation
- `get_input_embeddings()` / `set_input_embeddings()` - Embedding access

### Text-only Mode

For text-only evaluation tasks, BitMar automatically:
- Creates dummy vision features when not provided
- Falls back to text-only processing
- Maintains full compatibility with evaluation pipeline

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size in config (`batch_size: 4`)
   - Reduce sequence length (`max_seq_length: 128`)
   - Disable additional workers (`num_workers: 0`)

2. **Device Errors**
   - Check CUDA availability: `torch.cuda.is_available()`
   - Verify GPU memory: `nvidia-smi`
   - Restart if device state is corrupted

3. **Missing Dependencies**
   - Run `python validate_setup.py` to check
   - Install missing packages with pip

4. **Evaluation Pipeline Issues**
   - Ensure evaluation_data directory exists
   - Download required datasets (BLiMP, EWoK, GLUE)
   - Check evaluation pipeline README for setup

### Memory Debugging

Use the validation script to check memory usage:
```bash
python validate_setup.py
```

Monitor training memory:
```bash
# During training, check GPU memory
nvidia-smi
```

### Evaluation Debugging

Test model compatibility:
```bash
python test_evaluation_compatibility.py --create_test_model
```

## File Structure

```
BitMar/
├── src/
│   ├── model.py                 # Core BitMar model
│   ├── hf_compatibility.py      # HuggingFace compatibility layer
│   ├── dataset.py              # Data loading
│   ├── wandb_logger.py          # Logging
│   └── ...
├── configs/
│   └── bitmar_10epoch_memory_optimized.yaml
├── train_bitmar.py             # Main training script
├── validate_setup.py           # Pre-training validation
├── prepare_for_evaluation.py   # Post-training preparation
├── test_evaluation_compatibility.py  # Compatibility testing
└── final_model/                # Auto-generated HF model
```

## Performance Notes

### Training Performance
- **RTX A6000**: ~2-4 minutes per epoch (depends on data size)
- **Memory usage**: ~15-25GB GPU memory
- **CPU usage**: 2 workers for data loading

### Evaluation Performance
- **Zero-shot**: Fast evaluation in minutes
- **Fine-tuning**: Depends on task size (minutes to hours)

## Contact

For issues specific to BitMar evaluation integration, check:
1. Validation script output
2. Training logs for HuggingFace conversion
3. Evaluation pipeline compatibility test results
