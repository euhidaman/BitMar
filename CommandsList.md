# BitMar Training & Evaluation Commands

Complete guide for training and evaluating BitMar model on both BabyLM 2024 and 2025 evaluation pipelines.

## 🚀 Complete BitMar Training & Evaluation Commands

### **1. Initial Setup & Dataset Download**

```bash
# Clone evaluation pipelines (if not done)
cd d:\BabyLM
git clone https://github.com/babylm/evaluation-pipeline-2024.git
git clone https://github.com/babylm/evaluation-pipeline-2025.git

# Download BabyLM dataset (automatically creates ../babylm_dataset)
cd BitMar
python download_babylm_data.py

# Alternatively, manual download:
# cd ../babylm_dataset  # Dataset goes outside BitMar directory  
# Download from: https://osf.io/ad7qg/
# Extract train_50M.zip and babylm_multimodal.zip to ../babylm_dataset directory

# Install evaluation pipeline dependencies
cd ../evaluation-pipeline-2024
pip install -e .
pip install minicons
pip install --upgrade accelerate

cd ../evaluation-pipeline-2025
pip install -r requirements.txt

# Install BitMar dependencies
cd ../BitMar
pip install -r requirements.txt
```

### **2. Setup Multimodal Evaluation Data (2024 Pipeline)**

```bash
# Login to HuggingFace (required for Winoground and EWoK)
huggingface-cli login

# Download evaluation data for 2024 pipeline
cd ../evaluation-pipeline-2024

# Download DevBench data
bash devbench/download_data.sh

# Download EWoK data
python ewok/dl_and_filter.py

# Note: Winoground and VQA will be downloaded automatically during evaluation
```

### **3. Setup BitMar for Dual Evaluation**

```bash
cd ../BitMar

# Windows
setup_2024_evaluation.bat

# Linux/Mac
chmod +x setup_2024_evaluation.sh
./setup_2024_evaluation.sh
```

### **4. Train BitMar Model**

```bash
cd d:\BabyLM\BitMar

# Train with memory-optimized config for RTX A6000
python train_bitmar.py \
  --config configs/bitmar_10epoch_memory_optimized.yaml \
  --optimizer adamw \
  --epochs 10 \
  --wandb_project 'bitmar-10epoch-memory-safe'

# Alternative: Train with custom parameters
python train_bitmar.py \
  --config configs/bitmar_10epoch_memory_optimized.yaml \
  --optimizer adamw \
  --epochs 10 \
  --batch_size 8 \
  --wandb_project 'bitmar-custom-run'
```

**Training will automatically:**
- Save model to `final_model/` (for 2025 pipeline)
- Save model to `final_model_2024/` (for 2024 pipeline)  
- Create `EVALUATION_INSTRUCTIONS.md` with next steps

### **5. Text-only Evaluation (2025 Pipeline)**

```bash
cd ../evaluation-pipeline-2025

# Quick evaluation (recommended first)
./eval_zero_shot_fast.sh '../BitMar/final_model' 'checkpoint_1M' 'causal'

# Full zero-shot evaluation  
./eval_zero_shot.sh '../BitMar/final_model' 'causal'

# Fine-tuning evaluation (GLUE tasks)
./eval_finetuning.sh '../BitMar/final_model'

# Collect results for submission
python collect_results.py bitmar-model-name
```

### **6. Multimodal Evaluation (2024 Pipeline)**

```bash
cd ../evaluation-pipeline-2024

# Quick multimodal evaluation using our script
cd ../BitMar
./eval_bitmar.sh ./final_model_2024

# OR run individual components:
cd ../evaluation-pipeline-2024

# Winoground + VQA evaluation
./eval_multimodal.sh '../BitMar/final_model_2024'

# DevBench evaluation  
./eval_devbench.sh '../BitMar/final_model_2024' bitmar

# Collect results for submission (with vision tasks)
python collect_results.py bitmar-model-name --include_vision_tasks
```

### **7. Verify Results**

```bash
# Check evaluation results
ls -la results/

# View specific task results
cat results/blimp/bitmar-model-name/blimp_results.json
cat results/winoground_filtered/bitmar-model-name/winoground_filtered_results.json
```

## 📊 **Expected Directory Structure After Training**

```
d:\BabyLM\
├── BitMar/
│   ├── final_model/              # For 2025 pipeline (text-only)
│   ├── final_model_2024/         # For 2024 pipeline (multimodal)  
│   ├── checkpoints/              # Training checkpoints
│   └── EVALUATION_INSTRUCTIONS.md
├── babylm_dataset/               # Dataset directory (outside BitMar)
│   ├── train_50M/               # Text training data
│   ├── cc_3M_captions.json      # Multimodal captions
│   ├── cc_3M_dino_v2_states_1of2.npy  # Visual features
│   ├── cc_3M_dino_v2_states_2of2.npy  # Visual features
│   ├── local_narr_captions.json # Local narrative captions
│   └── local_narr_dino_v2_states.npy  # Local narrative features
├── evaluation-pipeline-2024/
│   └── results/                  # Multimodal evaluation results
├── evaluation-pipeline-2025/
│   └── results/                  # Text-only evaluation results
```

## 🎯 **Complete Workflow Summary**

```bash
# 1. Setup (one-time)
cd d:\BabyLM\BitMar
./setup_2024_evaluation.sh

# 2. Train model  
python train_bitmar.py --config configs/bitmar_10epoch_memory_optimized.yaml --optimizer adamw --epochs 10 --wandb_project 'bitmar-10epoch-memory-safe'

# 3. Evaluate text-only (2025)
cd ../evaluation-pipeline-2025
./eval_zero_shot_fast.sh '../BitMar/final_model' 'checkpoint_1M' 'causal'
./eval_finetuning.sh '../BitMar/final_model'

# 4. Evaluate multimodal (2024)  
cd ../BitMar
./eval_bitmar.sh ./final_model_2024

# 5. Collect results
cd ../evaluation-pipeline-2025 && python collect_results.py bitmar-model
cd ../evaluation-pipeline-2024 && python collect_results.py bitmar-model --include_vision_tasks
```

## 📋 **Evaluation Tasks Overview**

### **Text-only Tasks (2025 Pipeline)**
- **BLiMP**: Linguistic minimal pairs for syntax
- **BLiMP Supplement**: Additional linguistic phenomena
- **EWoK**: Grounded language understanding
- **GLUE**: General language understanding (fine-tuning)
- **Reading**: Eye-tracking and self-paced reading
- **Entity Tracking**: Coreference and entity understanding
- **WUGs**: Word usage generalization

### **Multimodal Tasks (2024 Pipeline)**
- **Winoground**: Visual reasoning with unpaired text scoring
- **VQA**: Visual question answering with 7 distractors
- **DevBench**: Developmental visual reasoning benchmark

## ⚠️ **Important Notes**

1. **Memory Management**: 
   - RTX A6000 config uses conservative batch sizes (8-16) to prevent GPU OOM
   - **CPU Memory Optimized**: Reduced data workers, disabled pin_memory, aggressive cleanup
   - **Automatic Memory Monitoring**: Warns at 80% CPU/GPU usage and triggers cleanup
   - **Efficient Batch Transfer**: Minimizes CPU-GPU memory transfer overhead
2. **HuggingFace Login**: Required for Winoground and EWoK dataset access
3. **Training Time**: Expect ~8-12 hours for 10 epochs on RTX A6000
4. **Disk Space**: Ensure ~50GB free space for datasets and model checkpoints
5. **Internet**: Required for dataset downloads and HuggingFace model uploads

## 🚨 **Troubleshooting Commands**

```bash
# Check GPU memory
nvidia-smi

# Verify model format
python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('./final_model'); print('✅ Model loads correctly')"

# Check evaluation pipeline integration
ls ../evaluation-pipeline-2024/devbench/model_classes/bitmar.py

# Verify dependencies
pip list | grep -E "(torch|transformers|accelerate)"

# Test BitMar compatibility
python -c "from src.hf_compatibility import BitMarForCausalLM; print('✅ BitMar HF compatibility working')"

# Check dataset paths
ls ../babylm_dataset/train_50M/
ls ../babylm_dataset/

# Memory troubleshooting commands
python -c "import psutil; ram=psutil.virtual_memory(); print(f'CPU RAM: {ram.percent:.1f}% ({ram.used/1024**3:.1f}GB/{ram.total/1024**3:.1f}GB)')"
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB allocated') if torch.cuda.is_available() else None"

# Test memory-efficient training
python test_training_fixes.py
```

## 🔧 **Advanced Configuration Options**

### **Custom Training Parameters**

```bash
# Adjust batch size for different GPU memory
python train_bitmar.py --config configs/bitmar_10epoch_memory_optimized.yaml --batch_size 4  # For 16GB GPU
python train_bitmar.py --config configs/bitmar_10epoch_memory_optimized.yaml --batch_size 16 # For 80GB GPU

# Different optimizers
python train_bitmar.py --config configs/bitmar_10epoch_memory_optimized.yaml --optimizer adamw8bit  # Memory efficient
python train_bitmar.py --config configs/bitmar_10epoch_memory_optimized.yaml --optimizer adam      # Standard Adam

# Custom epochs and project names
python train_bitmar.py --config configs/bitmar_10epoch_memory_optimized.yaml --epochs 20 --wandb_project 'bitmar-extended-training'
```

### **Selective Evaluation**

```bash
# Run only specific text tasks
cd ../evaluation-pipeline-2025
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name '../BitMar/final_model' --backend causal --task blimp --data_path "evaluation_data/fast_eval/blimp_fast"

# Run only specific multimodal tasks
cd ../evaluation-pipeline-2024
./eval_multimodal.sh '../BitMar/final_model_2024'  # Only Winoground + VQA
./eval_devbench.sh '../BitMar/final_model_2024' bitmar  # Only DevBench
```

## 📈 **Performance Expectations**

### **Training Metrics**
- **Training Loss**: Should decrease from ~4.0 to ~2.5-3.0
- **Validation Loss**: Should follow training loss trend
- **Memory Usage**: ~30-40GB on RTX A6000
- **Training Speed**: ~5-10 minutes per epoch (depends on dataset size)

### **Evaluation Scores (Expected Ranges)**
- **BLiMP**: 65-75% (competitive with baselines)
- **EWoK**: 50-55% (above random)
- **GLUE**: 60-70% (depends on task)
- **Winoground**: 50-60% (baseline range)
- **VQA**: 45-55% (above random with 7 distractors)
- **DevBench**: 50-65% (developmental reasoning)

## 🎉 **Success Indicators**

✅ **Training Complete**: Model saved to both `final_model/` and `final_model_2024/`  
✅ **Text Evaluation**: Results in `evaluation-pipeline-2025/results/`  
✅ **Multimodal Evaluation**: Results in `evaluation-pipeline-2024/results/`  
✅ **No OOM Errors**: Training completed without memory issues  
✅ **Model Compatibility**: Both evaluation pipelines load model successfully  

This complete workflow will train your BitMar model once and evaluate it on both evaluation pipelines, giving you comprehensive results for the BabyLM Challenge! 🎉
