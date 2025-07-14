#!/bin/bash

# BitMar RunPod Setup Script for RTX 4090
# Optimized for maximum performance on RunPod GPU instances

set -e  # Exit on any error

echo "🚀 BitMar RunPod Setup for RTX 4090"
echo "===================================="

# System information
echo "� System Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "Storage: $(df -h / | awk 'NR==2 {print $4}')"

# Update system
echo "🔧 Updating system packages..."
apt-get update -y
apt-get install -y git wget curl unzip htop nvtop

# Install Python dependencies
echo "� Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (optimized for RTX 4090)
echo "� Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo "📚 Installing BitMar requirements..."
pip install -r requirements.txt

# Set up environment variables for optimal performance
echo "⚡ Configuring environment for RTX 4090..."
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8  # Optimize for CPU cores
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings

# Create environment file
cat > .env << EOF
# BitMar Environment Configuration for RunPod RTX 4090
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=8
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TOKENIZERS_PARALLELISM=false

# WandB Configuration
WANDB_PROJECT=bitmar-babylm
WANDB_ENTITY=babylm-ntust-org
WANDB_API_KEY=5fba3726e4e32540d9fcba403f880dfaad983051

# HuggingFace Configuration  
HF_TOKEN=your_hf_token_here
HF_HOME=/workspace/.cache/huggingface
TRANSFORMERS_CACHE=/workspace/.cache/transformers

# Performance optimizations
CUDA_LAUNCH_BLOCKING=0
PYTHONUNBUFFERED=1
EOF

# Source environment
source .env

# Download and verify dataset
echo "📥 Downloading BabyLM multimodal dataset..."
python download_babylm_data.py

if [ $? -eq 0 ]; then
    echo "✅ Dataset download completed successfully!"
else
    echo "❌ Dataset download failed. Please check your internet connection."
    exit 1
fi

# Verify dataset integrity and setup
echo "🔍 Verifying dataset integrity..."
python -c "
import sys
sys.path.append('src')
from dataset import test_dataset

config = {
    'dataset_dir': '../babylm_dataset',
    'text_encoder_name': 'gpt2', 
    'max_seq_length': 512,
    'batch_size': 4,
    'num_workers': 0,
    'pin_memory': False,
    'hf_token': 'your_hf_token_here',
    'validation_datasets': ['ewok-core/ewok-core-1.0', 'facebook/winoground']
}

try:
    test_dataset(config, max_samples=100)
    print('✅ Dataset verification successful!')
except Exception as e:
    print(f'❌ Dataset verification failed: {e}')
    sys.exit(1)
"

# Run CPU compatibility tests
echo "🧪 Running CPU compatibility tests..."
python test_cpu_compatibility.py

# Create training directory structure
echo "📁 Creating directory structure..."
mkdir -p checkpoints logs attention_analysis memory_analysis results

# Final system check
echo "🔍 Final System Check:"
echo "====================="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
    
# Test tensor operations
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = x @ y
print(f'GPU tensor ops: ✅ Working')
"

echo ""
echo "🎯 RTX 4090 Optimization Settings:"
echo "=================================="
echo "• Batch size: 32 (optimized for 24GB VRAM)"
echo "• Mixed precision: 16-bit for faster training"
echo "• Data workers: 8 parallel workers"
echo "• Memory optimization: Gradient checkpointing enabled"
echo ""

echo "🎉 Setup Complete! Next Steps:"
echo "=============================="
echo "1. Run benchmarks: python run_all_benchmarks.py"
echo "2. Start training: python train_bitmar.py"
echo "3. Monitor training: wandb login (optional)"
echo ""
echo "📊 Monitoring:"
echo "• WandB: https://wandb.ai/babylm-ntust-org/bitmar-babylm"
echo "• GPU: nvidia-smi -l 1"
echo "• System: htop"
echo ""

echo "✅ BitMar RunPod setup completed successfully!"
echo "🚀 Ready for training on RTX 4090!"
