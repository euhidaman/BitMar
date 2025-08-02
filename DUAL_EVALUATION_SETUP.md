# BitMar Dual Evaluation Pipeline Support

BitMar now supports both the **2024** and **2025** BabyLM evaluation pipelines:

- **2024 Pipeline**: Multimodal evaluation (Winoground, VQA, DevBench)
- **2025 Pipeline**: Text-only evaluation (BLiMP, EWoK, GLUE, etc.)

## Quick Setup

### 1. Install Dependencies
```bash
# Install BitMar requirements
pip install -r requirements.txt

# For 2024 pipeline multimodal evaluation
cd ../evaluation-pipeline-2024
pip install -e .
pip install minicons
pip install --upgrade accelerate

# For 2025 pipeline text evaluation  
cd ../evaluation-pipeline-2025
pip install -r requirements.txt
```

### 2. Setup Evaluation Integration
```bash
# Windows
setup_2024_evaluation.bat

# Linux/Mac
chmod +x setup_2024_evaluation.sh
./setup_2024_evaluation.sh
```

## Training Your Model

Train with the memory-optimized config for RTX A6000:

```bash
python train_bitmar.py \
  --config configs/bitmar_10epoch_memory_optimized.yaml \
  --optimizer adamw \
  --epochs 10 \
  --wandb_project 'bitmar-10epoch-memory-safe'
```

After training, your model will be automatically saved in both formats:
- `final_model/` - For 2025 pipeline (text-only)
- `final_model_2024/` - For 2024 pipeline (multimodal)

## Evaluation

### Text-only Evaluation (2025 Pipeline)
```bash
cd ../evaluation-pipeline-2025

# Fast evaluation (recommended first)
./eval_zero_shot_fast.sh '../BitMar/final_model' 'checkpoint_1M' 'causal'

# Full evaluation with fine-tuning
./eval_finetuning.sh '../BitMar/final_model'
```

### Multimodal Evaluation (2024 Pipeline)
```bash
cd ../evaluation-pipeline-2024

# Quick evaluation using our script
cd ../BitMar
./eval_bitmar.sh ./final_model_2024

# Or run individual components:
cd ../evaluation-pipeline-2024
./eval_multimodal.sh '../BitMar/final_model_2024'  # Winoground + VQA
./eval_devbench.sh '../BitMar/final_model_2024' bitmar  # DevBench
```

## Technical Details

### HuggingFace Compatibility
BitMar implements full HuggingFace `AutoModelForCausalLM` compatibility:
- Text-only tasks use dummy vision features automatically
- Multimodal tasks use real vision features when available
- Compatible with both evaluation pipeline interfaces

### DevBench Integration
For the 2024 pipeline, BitMar provides:
- `BitMarEvalModel` class implementing the DevBench `EvalModel` interface
- `BitMarProcessor` for handling text + image inputs
- Automatic similarity scoring using causal language modeling loss

### Memory Optimization
The configuration includes RTX A6000-specific optimizations:
- Conservative batch sizes (8-16)
- Memory monitoring and cleanup
- Device consistency checks
- OOM prevention strategies

## File Structure
```
BitMar/
├── src/hf_compatibility.py      # Enhanced HF compatibility + 2024 support
├── devbench_bitmar.py           # DevBench model class
├── setup_2024_evaluation.sh    # Integration script (Linux/Mac)
├── setup_2024_evaluation.bat   # Integration script (Windows)
├── eval_bitmar.sh              # Convenient evaluation script
└── configs/
    └── bitmar_10epoch_memory_optimized.yaml  # RTX A6000 config
```

## Evaluation Flow

1. **Training**: Single BitMar model training
2. **Saving**: Automatic dual-format saving (both pipelines)
3. **Text Evaluation**: 2025 pipeline → BLiMP, EWoK, GLUE
4. **Multimodal Evaluation**: 2024 pipeline → Winoground, VQA, DevBench
5. **Results**: Combined scores from both evaluation tracks

## Troubleshooting

### Common Issues
- **Import errors**: Make sure BitMar src is in Python path
- **CUDA OOM**: Use the memory-optimized config provided
- **DevBench errors**: Ensure `bitmar.py` is copied to correct location
- **Missing dependencies**: Install both pipeline requirements

### Verification
After setup, verify integration:
```bash
# Check DevBench integration
ls ../evaluation-pipeline-2024/devbench/model_classes/bitmar.py

# Check model format
python -c "from transformers import AutoModel; print('HF compatibility: OK')"
```

## Performance Expectations

Based on baseline comparisons:
- **Text tasks**: Competitive with GPT-BERT and BabyLlama baselines
- **Multimodal tasks**: Should achieve 50-60% on Winoground/VQA (baseline range)
- **Memory usage**: ~30-40GB on RTX A6000 with optimized config

Your BitMar model is now ready for comprehensive evaluation on both the text-only (2025) and multimodal (2024) BabyLM evaluation pipelines! 🚀
