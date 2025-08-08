#!/bin/bash

# BitMar Evaluation Helper Script
# This script helps you run evaluations on saved checkpoints

set -e

# Default values
MODEL_PATH=""
EVAL_TYPE="both"
PIPELINE_2025="../evaluation-pipeline-2025"
PIPELINE_2024="../evaluation-pipeline-2024"
EVAL_DATA_PATH=""
OUTPUT_DIR="evaluation_results"

# Function to show usage
show_usage() {
    echo "Usage: $0 --model_path <path> [options]"
    echo ""
    echo "Required:"
    echo "  --model_path <path>        Path to BitMar checkpoint (.pt file)"
    echo ""
    echo "Optional:"
    echo "  --eval_type <type>         Type of evaluation: fast, full, or both (default: both)"
    echo "  --pipeline_2025 <path>     Path to evaluation-pipeline-2025 (default: ../evaluation-pipeline-2025)"
    echo "  --pipeline_2024 <path>     Path to evaluation-pipeline-2024 (default: ../evaluation-pipeline-2024)"
    echo "  --eval_data_path <path>    Path to evaluation_data directory (auto-detect if not specified)"
    echo "  --output_dir <path>        Output directory for results (default: evaluation_results)"
    echo "  --help, -h                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Run both fast and full evaluation on a checkpoint"
    echo "  $0 --model_path checkpoints_100M_dataset/checkpoint_epoch_1_tokens_50000000.pt"
    echo ""
    echo "  # Run only fast evaluation"
    echo "  $0 --model_path checkpoints_100M_dataset/latest_checkpoint.pt --eval_type fast"
    echo ""
    echo "  # Specify custom evaluation data path"
    echo "  $0 --model_path my_model.pt --eval_data_path /path/to/evaluation_data"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --eval_type)
            EVAL_TYPE="$2"
            shift 2
            ;;
        --pipeline_2025)
            PIPELINE_2025="$2"
            shift 2
            ;;
        --pipeline_2024)
            PIPELINE_2024="$2"
            shift 2
            ;;
        --eval_data_path)
            EVAL_DATA_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check required arguments
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model_path is required"
    show_usage
    exit 1
fi

# Check if model file exists
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "🚀 Starting BitMar Evaluation"
echo "  • Model: $MODEL_PATH"
echo "  • Evaluation type: $EVAL_TYPE"
echo "  • Output directory: $OUTPUT_DIR"
echo "  • 2025 pipeline: $PIPELINE_2025"
echo "  • 2024 pipeline: $PIPELINE_2024"

# Build evaluation data path argument
EVAL_DATA_ARG=""
if [[ -n "$EVAL_DATA_PATH" ]]; then
    EVAL_DATA_ARG="--evaluation_data_path $EVAL_DATA_PATH"
fi

# Run 2025 evaluation (text + multimodal)
if [[ -d "$PIPELINE_2025" ]]; then
    echo ""
    echo "📊 Running 2025 evaluation pipeline (text + multimodal)..."
    python evaluate_bitmar_2025.py \
        --model_path "$MODEL_PATH" \
        --eval_type "$EVAL_TYPE" \
        --evaluation_pipeline_path "$PIPELINE_2025" \
        --output_dir "${OUTPUT_DIR}/2025_results" \
        $EVAL_DATA_ARG

    if [[ $? -eq 0 ]]; then
        echo "✅ 2025 evaluation completed successfully"
    else
        echo "❌ 2025 evaluation failed"
    fi
else
    echo "⚠️  2025 evaluation pipeline not found at: $PIPELINE_2025"
fi

# Run 2024 evaluation (multimodal only)
if [[ -d "$PIPELINE_2024" ]]; then
    echo ""
    echo "📊 Running 2024 evaluation pipeline (multimodal only)..."
    python evaluate_bitmar_2024.py \
        --model_path "$MODEL_PATH" \
        --eval_type "$EVAL_TYPE" \
        --evaluation_pipeline_path "$PIPELINE_2024" \
        --output_dir "${OUTPUT_DIR}/2024_results" \
        $EVAL_DATA_ARG

    if [[ $? -eq 0 ]]; then
        echo "✅ 2024 evaluation completed successfully"
    else
        echo "❌ 2024 evaluation failed"
    fi
else
    echo "⚠️  2024 evaluation pipeline not found at: $PIPELINE_2024"
fi

echo ""
echo "🎉 Evaluation completed! Results are in: $OUTPUT_DIR"
echo ""
echo "📂 Directory structure:"
find "$OUTPUT_DIR" -name "*.json" | head -10 | while read file; do
    echo "  • $file"
done

if [[ $(find "$OUTPUT_DIR" -name "*.json" | wc -l) -gt 10 ]]; then
    echo "  • ... and more"
fi
