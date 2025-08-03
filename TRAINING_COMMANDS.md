# BitMar Training Commands - BabyLM Compliant

## ✅ Pre-Training Validation
```bash
# Run comprehensive validation (RECOMMENDED FIRST)
python validate_comprehensive_setup.py

# Quick GPU test
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## 🚀 Main Training Commands

### 1. Standard Training (Recommended)
```bash
# Full BabyLM compliant training with GPU optimization
python train_bitmar.py configs/bitmar_config.yaml
```

### 2. Background Training (For Long Sessions)
```bash
# Start training in background process
Start-Process powershell -ArgumentList "-Command python train_bitmar.py configs/bitmar_config.yaml" -WindowStyle Minimized
```

### 3. Quick GPU Optimized Training
```bash
# Alternative optimized training script
python train_gpu_optimized.py
```

### 4. Specific Device Training
```bash
# Force specific GPU device
python train_bitmar.py configs/bitmar_config.yaml --device cuda:0
```

## 📊 Key Features Enabled

✅ **BabyLM Token Compliance**
- 100M text tokens maximum (strictly enforced)
- 50M image tokens maximum (strictly enforced)  
- Image-caption pairs preserved during limiting
- Real-time token counting and verification

✅ **GPU Optimization**
- Batch size: 2 with gradient accumulation: 16 (effective batch = 32)
- Mixed precision training (AMP)
- CUDA optimizations (cuDNN, TF32, FlashAttention)
- Memory fraction: 85% GPU utilization
- Aggressive vision feature compression

✅ **Advanced Training Strategy**
- 4-Phase Training: Episodic Capture → Memory Consolidation → Quadrangle Optimization → Semantic Integration
- Component-specific learning rates
- QFormer Quadrangle Attention patterns
- Episodic memory with 16 slots

✅ **Comprehensive Monitoring**
- Cross-modal similarity tracking
- Component-wise loss metrics
- Quadrangle attention pattern analysis
- GPU memory monitoring
- BabyLM compliance verification

## 🎯 Expected Performance
- **Training time**: ~3-4 hours per epoch (vs 1400+ hours with old settings)
- **Total training**: ~30-40 hours for 10 epochs
- **Memory usage**: <40GB on RTX A6000 (85% utilization)
- **Token compliance**: Strict adherence to 100M text + 50M image limits

## 🔧 Troubleshooting

### GPU Out of Memory
```bash
# Reduce batch size if needed
export CUDA_VISIBLE_DEVICES=0
python train_bitmar.py configs/bitmar_config.yaml
```

### Slow Data Loading
```bash
# Check dataset validation first
python validate_comprehensive_setup.py
```

### Token Limit Issues
The training will automatically:
1. Count tokens in real-time
2. Enforce 100M text + 50M image limits
3. Preserve image-caption associations
4. Log compliance metrics to wandb

## 📈 Monitoring

### Weights & Biases
- Project: `bitmar-babylm-edge`
- Key metrics: Cross-modal similarity, component losses, GPU utilization
- BabyLM compliance tracking

### Local Logs
- Training progress: `training.log`
- Checkpoints: `checkpoints/`
- Results: `results/`

## 🎓 Training Phases

1. **Epochs 0-2**: Episodic Capture (Foundation building)
2. **Epochs 3-5**: Memory Consolidation (Cross-modal fusion)  
3. **Epochs 6-7**: Quadrangle Optimization (Attention mastery)
4. **Epochs 8-9**: Semantic Integration (Knowledge refinement)

Each phase has optimized learning rates and component focus for maximum efficiency.
