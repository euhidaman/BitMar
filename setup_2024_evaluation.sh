#!/bin/bash

# BitMar 2024 Evaluation Pipeline Integration Script
# Run this script to set up BitMar for evaluation with the 2024 pipeline

echo "🔧 Setting up BitMar for 2024 BabyLM evaluation pipeline..."

# Check if evaluation-pipeline-2024 exists
if [ ! -d "../evaluation-pipeline-2024" ]; then
    echo "❌ Error: evaluation-pipeline-2024 directory not found at ../evaluation-pipeline-2024"
    echo "Please make sure the 2024 evaluation pipeline is cloned and available."
    exit 1
fi

# Copy BitMar DevBench integration file
echo "📁 Copying BitMar DevBench integration..."
cp devbench_bitmar.py ../evaluation-pipeline-2024/devbench/model_classes/bitmar.py
echo "✅ BitMar DevBench model class installed"

# Update DevBench eval.py to support BitMar
echo "🔧 Updating DevBench eval.py to support BitMar..."

# Check if BitMar is already added to eval.py
if grep -q "bitmar" ../evaluation-pipeline-2024/devbench/eval.py; then
    echo "ℹ️  BitMar already integrated in DevBench eval.py"
else
    # Add BitMar import and model loading
    cat >> ../evaluation-pipeline-2024/devbench/eval.py << 'EOF'

# BitMar model support
elif args.model_type == "bitmar":
    from devbench.model_classes.bitmar import load_bitmar_model
    model = load_bitmar_model(args.model, device=device)

EOF
    echo "✅ BitMar integration added to DevBench eval.py"
fi

# Create a BitMar-specific evaluation script
echo "📝 Creating BitMar evaluation script..."
cat > eval_bitmar.sh << 'EOF'
#!/bin/bash

MODEL_PATH=$1

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: ./eval_bitmar.sh <path_to_bitmar_model>"
    echo "Example: ./eval_bitmar.sh ../BitMar/final_model_2024"
    exit 1
fi

echo "🚀 Evaluating BitMar model on 2024 BabyLM multimodal tasks..."
echo "Model path: $MODEL_PATH"

# Change to evaluation pipeline directory
cd ../evaluation-pipeline-2024

echo "📊 Running Winoground and VQA evaluation..."
./eval_multimodal.sh $MODEL_PATH

echo "🖼️ Running DevBench evaluation..."
./eval_devbench.sh $MODEL_PATH bitmar

echo "✅ BitMar evaluation complete!"
echo "Results are saved in the evaluation-pipeline-2024/results/ directory"

EOF

chmod +x eval_bitmar.sh
echo "✅ BitMar evaluation script created: eval_bitmar.sh"

echo ""
echo "🎉 BitMar integration with 2024 evaluation pipeline complete!"
echo ""
echo "📋 Usage Instructions:"
echo "1. Train your BitMar model:"
echo "   python train_bitmar.py --config configs/bitmar_10epoch_memory_optimized.yaml --optimizer adamw --epochs 10 --wandb_project 'bitmar-10epoch-memory-safe'"
echo ""
echo "2. Evaluate on 2024 pipeline (multimodal tasks):"
echo "   ./eval_bitmar.sh ./final_model_2024"
echo ""
echo "3. Or run individual evaluations:"
echo "   cd ../evaluation-pipeline-2024"
echo "   ./eval_multimodal.sh ../BitMar/final_model_2024"
echo "   ./eval_devbench.sh ../BitMar/final_model_2024 bitmar"
echo ""
echo "📁 Model locations after training:"
echo "   - final_model/        : For 2025 pipeline (text-only)"
echo "   - final_model_2024/   : For 2024 pipeline (multimodal)"
echo ""
echo "🔍 Both models are identical, just placed for convenience with respective pipelines"
