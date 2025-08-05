#!/bin/bash

# BitMar 100M Token Dataset Training Script (10 epochs max)
# Complete training pipeline with token analysis and monitoring

echo "🚀 BitMar 100M Token Dataset Training Pipeline"
echo "==============================================="

# Configuration
CONFIG_FILE="configs/bitmar_100M_tokens.yaml"
DATASET_DIR="../babylm_dataset"
DEVICE="cuda:0"

# Step 1: Verify dataset exists
echo "📊 Step 1: Verifying dataset..."
python download_babylm_data.py
if [ $? -ne 0 ]; then
    echo "❌ Dataset verification failed!"
    exit 1
fi

# Step 2: Analyze token distribution
echo "🔍 Step 2: Analyzing token distribution..."
python analyze_token_distribution.py \
    --dataset_dir "$DATASET_DIR" \
    --tokenizer "gpt2" \
    --target_caption_tokens 50000000 \
    --target_text_tokens 50000000

if [ $? -ne 0 ]; then
    echo "⚠️  Token analysis failed, but continuing with training..."
fi

# Step 3: Start training (up to 10 epochs on 100M token dataset)
echo "🎯 Step 3: Starting training on 100M token dataset (max 10 epochs)..."
python train_100M_tokens.py \
    --config "$CONFIG_FILE" \
    --device "$DEVICE" \
    --rebuild_cache

# Check training result
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📊 Results saved in:"
    echo "  • Checkpoints: checkpoints_100M_tokens/"
    echo "  • Logs: logs_100M_tokens/"
    echo "  • Token analysis: token_analysis_results/"
else
    echo "❌ Training failed!"
    exit 1
fi

echo "🎉 BitMar 100M Token Training Complete!"
