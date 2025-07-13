#!/bin/bash

# BitMar RunPod Setup Script
# Sets up the environment for GPU training on RunPod

echo "🚀 Setting up BitMar on RunPod"
echo "=============================="

# Update system
echo "📦 Updating system packages..."
apt-get update && apt-get upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
apt-get install -y git wget curl build-essential

# Setup Python environment
echo "🐍 Setting up Python environment..."
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv

# Clone repository (if not already present)
if [ ! -d "BitMar" ]; then
    echo "📥 Cloning BitMar repository..."
    git clone <YOUR_GITHUB_REPO_URL> BitMar
fi

cd BitMar

# Create virtual environment
echo "🌟 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set up environment variables
echo "🔑 Setting up environment variables..."
export WANDB_API_KEY="5fba3726e4e32540d9fcba403f880dfaad983051"
export CUDA_VISIBLE_DEVICES=0

# Download dataset (if needed)
echo "📊 Setting up dataset..."
if [ ! -d "../babylm_dataset" ]; then
    echo "Downloading BabyLM dataset..."
    python download_babylm_data.py
else
    echo "Dataset already exists"
fi

# Verify GPU setup
echo "🔍 Verifying GPU setup..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Test dataset compatibility
echo "🧪 Testing dataset compatibility..."
python test_dataset_compatibility.py

# Run a quick training test (1 epoch)
echo "🏃 Running training test..."
python train_bitmar.py --config configs/bitmar_config.yaml --max_epochs 1 --batch_size 8

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🚀 To start full training, run:"
echo "   source venv/bin/activate"
echo "   export WANDB_API_KEY=\"5fba3726e4e32540d9fcba403f880dfaad983051\""
echo "   python train_bitmar.py --config configs/bitmar_config.yaml"
echo ""
echo "📊 Monitor training at:"
echo "   https://wandb.ai/babylm-ntust-org/bitmar-babylm"
echo ""
echo "🔍 For attention analysis:"
echo "   python evaluate_bitmar.py --model-path checkpoints/best_checkpoint.pt"
